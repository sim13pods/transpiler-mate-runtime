# Copyright 2026 Terradue
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Click CLI that exposes installed transpiler-mate plugins as subcommands.

Each plugin is discovered through Python entry points and translated into a
Click command. The plugin's Pydantic ``options_model`` defines the command
options.

Each plugin command owns runtime input/session setup and constructs its
``TranspilerContext`` immediately before executing the plugin.
"""

from __future__ import annotations

import json
import time
import types
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

import click
from click.core import ParameterSource
from cwl_loader import _is_url, load_cwl_from_location
from cwl_utils.parser import Process
from loguru import logger
from pydantic import AnyUrl, TypeAdapter, ValidationError
from requests import Session
from requests.adapters import HTTPAdapter
from session_adapters.bearer_auth_http_adapter import BearerAuthHTTPAdapter
from session_adapters.file_adapter import FileAdapter
from session_adapters.oci_adapter import OCIAdapter
from transpiler_mate.api import (
    PluginError,
    PluginExecutionError,
    PluginFailureError,
    TranspilerContext,
)

from .plugin_loader import (
    PluginLoaderError,
    discover_plugins,
    load_plugin,
)
from .software_application_extractor import software_application_from_process

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from click.core import Parameter
    from requests.adapters import BaseAdapter
    from transpiler_mate.api import SoftwareApplication, TranspilerPlugin

_EXECUTION_SEPARATOR = (
    "------------------------------------------------------------------------"
)


class PydanticParamType(click.ParamType[Any]):
    """Click parameter type backed by a Pydantic ``TypeAdapter``."""

    def __init__(self, annotation: Any) -> None:
        self.annotation = annotation
        self.adapter = TypeAdapter(annotation)
        self.name = _type_name(annotation)

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Any:
        if value is None:
            return None

        if not isinstance(value, str):
            try:
                return self.adapter.validate_python(value)
            except (TypeError, ValueError, ValidationError) as exc:
                self.fail(str(exc), param, ctx)

        try:
            return self.adapter.validate_json(value)
        except (TypeError, ValueError, ValidationError):
            try:
                return self.adapter.validate_python(value)
            except (TypeError, ValueError, ValidationError) as exc:
                self.fail(str(exc), param, ctx)


class LiteralParamType(click.Choice[Any]):
    """Choice type that preserves the original Literal value type."""

    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values
        labels = tuple(_choice_label(value) for value in values)

        if len(labels) != len(set(labels)):
            raise TypeError(
                "Literal values must have unique string representations "
                "to be exposed as a Click option"
            )

        super().__init__(labels, case_sensitive=True)

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Any:
        label = super().convert(value, param, ctx)
        return self.values[self.choices.index(label)]


class EnumParamType(click.Choice[Any]):
    """Choice type that exposes enum member names and returns enum values."""

    def __init__(self, enum_type: type[Enum]) -> None:
        self.enum_type = enum_type
        self.members = tuple(enum_type)
        super().__init__(
            tuple(member.name for member in self.members),
            case_sensitive=False,
        )

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Enum:
        if isinstance(value, self.enum_type):
            return value

        member_name = super().convert(value, param, ctx)
        return self.enum_type[member_name]


class PluginGroup(click.Group):
    """Click group backed by installed transpiler-mate entry points."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._entry_points: dict[str, EntryPoint] | None = None
        self._plugin_commands: dict[str, click.Command] = {}

    def _discover(self) -> dict[str, EntryPoint]:
        if self._entry_points is None:
            try:
                self._entry_points = discover_plugins()
            except PluginLoaderError as exc:
                raise click.ClickException(str(exc)) from exc

        return self._entry_points

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = set(super().list_commands(ctx))
        commands.update(self._discover())
        return sorted(commands)

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | None:
        static_command = super().get_command(ctx, cmd_name)
        if static_command is not None:
            return static_command

        cached = self._plugin_commands.get(cmd_name)
        if cached is not None:
            return cached

        entry_point = self._discover().get(cmd_name)
        if entry_point is None:
            return None

        try:
            plugin = load_plugin(entry_point)
        except PluginLoaderError as exc:
            raise click.ClickException(str(exc)) from exc

        if plugin.name != entry_point.name:
            raise click.ClickException(
                f"Plugin entry point {entry_point.name!r} loads a plugin named "
                f"{plugin.name!r}; the entry-point name and plugin.name must match"
            )

        command = plugin_to_click_command(plugin)
        self._plugin_commands[cmd_name] = command
        return command


def time_to_string(time: float) -> str:
    return datetime.fromtimestamp(time).isoformat(timespec="milliseconds")


def _runtime_click_parameters() -> list[Parameter]:
    return [
        click.Argument(
            ["_runtime_source"],
            metavar="SOURCE",
            type=click.STRING,
            required=True,
        ),
        click.Option(
            ["--oci-hostname", "_runtime_oci_hostname"],
            envvar="OCI_HOSTNAME",
            show_envvar=True,
        ),
        click.Option(
            ["--oci-username", "_runtime_oci_username"],
            envvar="OCI_USERNAME",
            show_envvar=True,
        ),
        click.Option(
            ["--oci-password", "_runtime_oci_password"],
            envvar="OCI_PASSWORD",
            show_envvar=True,
        ),
        click.Option(
            ["--oauth2-bearer", "_runtime_oauth2_bearer"],
            envvar="OAUTH2_BEARER",
            show_envvar=True,
        ),
    ]


def plugin_to_click_command(
    plugin: TranspilerPlugin[Any],
) -> click.Command:
    """Translate one loaded plugin into a Click command."""

    options_model = plugin.options_model
    params: list[Parameter] = [
        *_runtime_click_parameters(),
        *(
            field_to_click_option(field_name, field)
            for field_name, field in options_model.model_fields.items()
        ),
    ]

    @click.pass_context
    def invoke_plugin(ctx: click.Context, /, **values: Any) -> None:
        source: str = values.pop("_runtime_source")
        oci_hostname: str | None = values.pop("_runtime_oci_hostname")
        oci_username: str | None = values.pop("_runtime_oci_username")
        oci_password: str | None = values.pop("_runtime_oci_password")
        oauth2_bearer: str | None = values.pop("_runtime_oauth2_bearer")

        model_values: dict[str, Any] = {}

        for field_name, field in options_model.model_fields.items():
            parameter_source = ctx.get_parameter_source(field_name)

            if not field.is_required() and parameter_source is ParameterSource.DEFAULT:
                continue

            model_values[field_name] = values[field_name]

        try:
            options = options_model.model_validate(
                model_values,
                by_alias=True,
                by_name=True,
            )
        except ValidationError as exc:
            raise click.UsageError(
                _format_validation_error(exc),
                ctx=ctx,
            ) from exc
        logger.info(f"""
━┏┛┏━┃┏━┃┏━ ┏━┛┏━┃┛┃  ┏━┛┏━┃  ┏┏ ┏━┃━┏┛┏━┛
 ┃ ┏┏┛┏━┃┃ ┃━━┃┏━┛┃┃  ┏━┛┏┏┛  ┃┃┃┏━┃ ┃ ┏━┛
 ┛ ┛ ┛┛ ┛┛ ┛━━┛┛  ┛━━┛━━┛┛ ┛  ┛┛┛┛ ┛ ┛ ━━┛

 v{version("transpiler-mate-runtime")} by Terradue srl
 info[at]terradue[dot]com
""")

        session = Session()

        def mount_session(scheme: str, adapter: BaseAdapter) -> None:
            logger.debug(f"Mounting '{scheme}' scheme to '{type(adapter).__name__}'...")
            session.mount(scheme, adapter)
            logger.debug(
                f"Scheme '{scheme}' successfully mount to '{type(adapter).__name__}'"
            )

        http_adapter = (
            BearerAuthHTTPAdapter(oauth2_bearer) if oauth2_bearer else HTTPAdapter()
        )
        mount_session("http://", http_adapter)
        mount_session("https://", http_adapter)
        mount_session("file://", FileAdapter())
        mount_session(
            "oci://",
            OCIAdapter(
                hostname=oci_hostname,
                username=oci_username,
                password=oci_password,
            ),
        )

        content_path: Path | AnyUrl = (
            AnyUrl(source)
            if _is_url(path_or_url=source, session=session)
            else Path(source).absolute()
        )

        cwl_document: Process | list[Process] = load_cwl_from_location(
            path=source,
            session=session,
        )
        metadata: SoftwareApplication = software_application_from_process(cwl_document)
        context = TranspilerContext(
            source=content_path,
            metadata=metadata,
            document=(
                cwl_document
                if isinstance(cwl_document, Process)
                else tuple(cwl_document)
            ),
        )

        start_time = time.time()
        logger.info(
            "Started at: {}",
            time_to_string(start_time),
        )

        try:
            plugin.execute(context, options)
        except PluginFailureError as exc:
            logger.error(_EXECUTION_SEPARATOR)
            logger.error("FAILURE")
            logger.error(
                "Plugin '{}' failed to produce expected results: {}",
                plugin.name,
                exc,
            )
            logger.error(_EXECUTION_SEPARATOR)
            ctx.exit(1)
        except PluginExecutionError as exc:
            logger.error(_EXECUTION_SEPARATOR)
            logger.error("ERROR")
            logger.exception(
                "Plugin '{}' execution failed unexpectedly: {}",
                plugin.name,
                exc,
            )
            logger.error(_EXECUTION_SEPARATOR)
            ctx.exit(1)
        except PluginError as exc:
            raise click.ClickException(str(exc)) from exc
        else:
            logger.success(_EXECUTION_SEPARATOR)
            logger.success("SUCCESS")
            logger.success(_EXECUTION_SEPARATOR)
        finally:
            end_time = time.time()
            logger.info("Total time: {:.4f} seconds", end_time - start_time)
            logger.info(
                "Finished at: {}",
                time_to_string(end_time),
            )

    return click.Command(
        name=plugin.name,
        help=plugin.description,
        params=params,
        callback=invoke_plugin,
    )


def field_to_click_option(
    field_name: str,
    field: Any,
) -> click.Option:
    """Translate one Pydantic model field into a Click option."""

    annotation = _strip_optional(_unwrap_annotated(field.annotation))
    option_name = field_name.replace("_", "-")
    required = field.is_required()
    default, show_default = _click_default(field, annotation)

    if annotation is bool and not required:
        return click.Option(
            [f"--{option_name}/--no-{option_name}"],
            required=False,
            default=default,
            show_default=show_default,
            help=field.description,
        )

    click_type, multiple = _click_type_and_cardinality(annotation)

    return click.Option(
        [f"--{option_name}"],
        type=click_type,
        required=required,
        multiple=multiple,
        default=default,
        show_default=show_default,
        help=field.description,
    )


def _click_type_and_cardinality(
    annotation: Any,
) -> tuple[click.ParamType[Any] | type[Any], bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return _click_type(args[0]), True

        if args:
            return click.Tuple([_click_type(item) for item in args]), False

    if origin in {list, set, frozenset, Sequence}:
        item_type = args[0] if args else str
        return _click_type(item_type), True

    return _click_type(annotation), False


def _click_type(annotation: Any) -> click.ParamType[Any] | type[Any]:
    annotation = _strip_optional(_unwrap_annotated(annotation))
    origin = get_origin(annotation)

    if annotation is str:
        return click.STRING
    if annotation is int:
        return click.INT
    if annotation is float:
        return click.FLOAT
    if annotation is bool:
        return click.BOOL
    if annotation is Path:
        return click.Path(path_type=Path)

    if origin is Literal:
        return LiteralParamType(get_args(annotation))

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return EnumParamType(annotation)

    return PydanticParamType(annotation)


def _click_default(
    field: Any,
    annotation: Any,
) -> tuple[Any, bool]:
    if field.is_required():
        if _is_multiple(annotation):
            return (), False
        return None, False

    if field.default_factory is not None:
        if _is_multiple(annotation):
            return (), False
        return None, False

    default = field.default

    if _is_multiple(annotation):
        if default is None:
            return (), False
        if isinstance(default, (list, tuple, set, frozenset)):
            return tuple(default), True

    return default, default is not None


def _is_multiple(annotation: Any) -> bool:
    annotation = _strip_optional(_unwrap_annotated(annotation))
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {list, set, frozenset, Sequence}:
        return True

    return origin is tuple and len(args) == 2 and args[1] is Ellipsis


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)

    if origin not in {Union, types.UnionType}:
        return annotation

    args = get_args(annotation)
    non_none = tuple(arg for arg in args if arg is not type(None))

    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0]

    return annotation


def _unwrap_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _choice_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return str(value)


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation).removeprefix("typing.")


def _format_validation_error(exc: ValidationError) -> str:
    messages = []

    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        messages.append(f"{location}: {message}" if location else message)

    return "Invalid plugin options:\n  " + "\n  ".join(messages)


@click.group(cls=PluginGroup, name="transpiler-mate")
@click.version_option(package_name="transpiler-mate-runtime")
def main() -> None:
    """Discover and invoke installed transpiler-mate plugins."""
