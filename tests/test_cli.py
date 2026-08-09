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

from __future__ import annotations

from enum import Enum
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner
from cwl_utils.parser import Process
from cwl_utils.parser.cwl_v1_2 import Workflow
from loguru import logger
from pydantic import BaseModel, Field
from transpiler_mate.api import (
    PluginExecutionError,
    PluginFailureError,
    SoftwareApplication,
    TranspilerContext,
    transpiler_plugin,
)

from transpiler_mate.runtime import cli

if TYPE_CHECKING:
    from transpiler_mate.api import TranspilerPlugin


class Mode(Enum):
    FAST = "fast"
    SAFE = "safe"


class ExampleOptions(BaseModel):
    output: Path = Field(description="Output path")
    retries: int = Field(default=2, ge=0, description="Retry count")
    verbose: bool = Field(default=False, description="Verbose output")
    tags: list[str] = Field(default_factory=list, description="Output tags")
    mode: Mode = Field(default=Mode.SAFE, description="Execution mode")


class FakeEntryPoint:
    def __init__(self, name: str = "demo") -> None:
        self.name = name
        self.value = "example:plugin"


class FakeSession:
    def __init__(self) -> None:
        self.mounts: list[tuple[str, object]] = []

    def mount(self, scheme: str, adapter: object) -> None:
        self.mounts.append((scheme, adapter))


@pytest.fixture(autouse=True)
def _runtime_context_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Workflow(inputs=[], outputs=[], steps=[])
    metadata = SoftwareApplication.model_construct()

    def adapter() -> object:
        return object()

    def oci_adapter(
        *,
        hostname: str | None,
        username: str | None,
        password: str | None,
    ) -> object:
        del hostname, username, password
        return object()

    def is_url(*, path_or_url: str, session: object) -> bool:
        del path_or_url, session
        return False

    def load_document(*, path: str, session: object) -> Process:
        del path, session
        return process

    def extract_metadata(
        document: Process | list[Process],
    ) -> SoftwareApplication:
        del document
        return metadata

    monkeypatch.setattr(cli, "Session", FakeSession)
    monkeypatch.setattr(cli, "HTTPAdapter", adapter)
    monkeypatch.setattr(cli, "FileAdapter", adapter)
    monkeypatch.setattr(cli, "OCIAdapter", oci_adapter)
    monkeypatch.setattr(cli, "_is_url", is_url)
    monkeypatch.setattr(cli, "load_cwl_from_location", load_document)
    monkeypatch.setattr(
        cli,
        "software_application_from_process",
        extract_metadata,
    )


def _entry_point(name: str = "demo") -> EntryPoint:
    return cast("EntryPoint", FakeEntryPoint(name))


def _recording_plugin(
    calls: list[tuple[TranspilerContext, ExampleOptions]],
) -> TranspilerPlugin[ExampleOptions]:
    @transpiler_plugin(
        name="demo",
        description="Demo plugin",
        options_model=ExampleOptions,
    )
    def execute(
        context: TranspilerContext,
        options: ExampleOptions,
    ) -> None:
        calls.append((context, options))

    return execute


def test_plugin_command_maps_pydantic_fields_to_click_options() -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    command = cli.plugin_to_click_command(_recording_plugin(calls))
    runner = CliRunner()

    help_result = runner.invoke(command, ["--help"])

    assert help_result.exit_code == 0
    assert "Usage: demo [OPTIONS] SOURCE" in help_result.output
    assert "--oci-hostname TEXT" in help_result.output
    assert "--oci-username TEXT" in help_result.output
    assert "--oci-password TEXT" in help_result.output
    assert "--oauth2-bearer TEXT" in help_result.output
    assert "--output PATH" in help_result.output
    assert "--retries INTEGER" in help_result.output
    assert "--verbose / --no-verbose" in help_result.output
    assert "--tags TEXT" in help_result.output
    assert "[fast|safe]" in help_result.output


def test_plugin_command_builds_options_and_context() -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    command = cli.plugin_to_click_command(_recording_plugin(calls))
    runner = CliRunner()

    result = runner.invoke(
        command,
        [
            "--output",
            "result.json",
            "--retries",
            "3",
            "--verbose",
            "--tags",
            "alpha",
            "--tags",
            "beta",
            "--mode",
            "fast",
            "workflow.cwl",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    received_context, options = calls[0]
    assert received_context.source == Path("workflow.cwl").absolute()
    assert options.output == Path("result.json")
    assert options.retries == 3
    assert options.verbose is True
    assert options.tags == ["alpha", "beta"]
    assert options.mode is Mode.FAST


def test_plugin_command_logs_successful_execution() -> None:
    messages: list[str] = []

    def collect_log(message: object) -> None:
        messages.append(str(message))

    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    command = cli.plugin_to_click_command(_recording_plugin(calls))
    sink_id = logger.add(collect_log, format="{message}")

    try:
        result = CliRunner().invoke(
            command,
            ["--output", "result.json", "workflow.cwl"],
        )
    finally:
        logger.remove(sink_id)

    logs = "".join(messages)

    assert result.exit_code == 0, result.output
    assert "Started at:" in logs
    assert "SUCCESS" in logs
    assert "Total time:" in logs
    assert "Finished at:" in logs


def test_plugin_command_reports_pydantic_validation_errors() -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    command = cli.plugin_to_click_command(_recording_plugin(calls))
    runner = CliRunner()

    result = runner.invoke(
        command,
        ["--output", "result.json", "--retries", "-1", "workflow.cwl"],
    )

    assert result.exit_code == 2
    assert "Invalid plugin options" in result.output
    assert "retries" in result.output
    assert calls == []


def test_plugin_failure_error_logs_failure_and_exits_one() -> None:
    messages: list[str] = []

    def collect_log(message: object) -> None:
        messages.append(str(message))

    @transpiler_plugin(
        name="failing",
        description="Failing plugin",
        options_model=ExampleOptions,
    )
    def failing_plugin(
        context: TranspilerContext,
        options: ExampleOptions,
    ) -> None:
        del context, options
        raise PluginFailureError("expected failure")

    command = cli.plugin_to_click_command(failing_plugin)
    sink_id = logger.add(collect_log, format="{message}")

    try:
        result = CliRunner().invoke(
            command,
            ["--output", "result.json", "workflow.cwl"],
        )
    finally:
        logger.remove(sink_id)

    logs = "".join(messages)

    assert result.exit_code == 1
    assert "FAILURE" in logs
    assert (
        "Plugin 'failing' failed to produce expected results: expected failure" in logs
    )
    assert "Traceback (most recent call last)" not in logs
    assert "Total time:" in logs
    assert "Finished at:" in logs


def test_plugin_execution_error_logs_traceback_and_exits_one() -> None:
    messages: list[str] = []

    def collect_log(message: object) -> None:
        messages.append(str(message))

    @transpiler_plugin(
        name="failing",
        description="Failing plugin",
        options_model=ExampleOptions,
    )
    def failing_plugin(
        context: TranspilerContext,
        options: ExampleOptions,
    ) -> None:
        del context, options
        raise PluginExecutionError("unexpected technical failure")

    command = cli.plugin_to_click_command(failing_plugin)
    sink_id = logger.add(collect_log, format="{message}")

    try:
        result = CliRunner().invoke(
            command,
            ["--output", "result.json", "workflow.cwl"],
        )
    finally:
        logger.remove(sink_id)

    logs = "".join(messages)

    assert result.exit_code == 1
    assert "ERROR" in logs
    assert "Plugin 'failing' execution failed unexpectedly" in logs
    assert "Traceback (most recent call last)" in logs
    assert "PluginExecutionError: unexpected technical failure" in logs
    assert "Total time:" in logs
    assert "Finished at:" in logs


def _install_runtime_plugin(
    monkeypatch: pytest.MonkeyPatch,
    plugin: TranspilerPlugin[ExampleOptions],
) -> None:
    entry_point = _entry_point()
    group = cli.main
    group._entry_points = None
    group._plugin_commands.clear()

    def discover_plugins() -> dict[str, EntryPoint]:
        return {"demo": entry_point}

    def load_plugin(entry: EntryPoint) -> TranspilerPlugin[ExampleOptions]:
        del entry
        return plugin

    monkeypatch.setattr(cli, "discover_plugins", discover_plugins)
    monkeypatch.setattr(cli, "load_plugin", load_plugin)


def test_root_help_only_exposes_global_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    _install_runtime_plugin(monkeypatch, _recording_plugin(calls))

    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: transpiler-mate [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "--version" in result.output
    assert "--oci-hostname" not in result.output
    assert "--oauth2-bearer" not in result.output
    assert calls == []


def test_plugin_help_does_not_initialize_transpiler_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    _install_runtime_plugin(monkeypatch, _recording_plugin(calls))

    def unexpected_session() -> None:
        raise AssertionError("plugin callback must not execute for plugin help")

    monkeypatch.setattr(cli, "Session", unexpected_session)

    result = CliRunner().invoke(
        cli.main,
        ["demo", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert "Usage: transpiler-mate demo [OPTIONS] SOURCE" in result.output
    assert "Demo plugin" in result.output
    assert "--oci-hostname TEXT" in result.output
    assert calls == []


def test_plugin_builds_context_and_passes_it_to_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[TranspilerContext, ExampleOptions]] = []
    _install_runtime_plugin(monkeypatch, _recording_plugin(calls))
    process = Workflow(inputs=[], outputs=[], steps=[])
    metadata = SoftwareApplication.model_construct()

    def adapter() -> object:
        return object()

    def oci_adapter(
        *,
        hostname: str | None,
        username: str | None,
        password: str | None,
    ) -> object:
        del hostname, username, password
        return object()

    def is_url(*, path_or_url: str, session: object) -> bool:
        del path_or_url, session
        return False

    def load_document(*, path: str, session: object) -> Process:
        del path, session
        return process

    def extract_metadata(
        document: Process | list[Process],
    ) -> SoftwareApplication:
        del document
        return metadata

    monkeypatch.setattr(cli, "Session", FakeSession)
    monkeypatch.setattr(cli, "HTTPAdapter", adapter)
    monkeypatch.setattr(cli, "FileAdapter", adapter)
    monkeypatch.setattr(cli, "OCIAdapter", oci_adapter)
    monkeypatch.setattr(cli, "_is_url", is_url)
    monkeypatch.setattr(cli, "load_cwl_from_location", load_document)
    monkeypatch.setattr(
        cli,
        "software_application_from_process",
        extract_metadata,
    )

    result = CliRunner().invoke(
        cli.main,
        ["demo", "--output", "result.json", "workflow.cwl"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    context, options = calls[0]
    assert context.source == Path("workflow.cwl").absolute()
    assert context.document is process
    assert context.metadata is metadata
    assert options.output == Path("result.json")
