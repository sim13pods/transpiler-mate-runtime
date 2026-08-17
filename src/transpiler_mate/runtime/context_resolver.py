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

from pathlib import Path
from typing import TYPE_CHECKING

from cwl_loader import _is_url, load_cwl_from_location
from cwl_loader.utils import get_ids, search_process
from cwl_utils.parser import Process
from loguru import logger
from pydantic import AnyUrl
from requests import Session
from requests.adapters import BaseAdapter, HTTPAdapter
from session_adapters.bearer_auth_http_adapter import BearerAuthHTTPAdapter
from session_adapters.file_adapter import FileAdapter
from session_adapters.oci_adapter import OCIAdapter
from transpiler_mate.api import (
    PluginExecutionError,
    PluginFailureError,
    TranspilerContextResolver,
)
from transpiler_mate.api.plugin import TranspilerContext

from .software_application_extractor import software_application_from_process

if TYPE_CHECKING:
    from transpiler_mate.api import SoftwareApplication


class DefaultTranspilerContextResolver(TranspilerContextResolver):
    def __init__(
        self,
        *,
        oci_hostname: str | None = None,
        oci_username: str | None = None,
        oci_password: str | None = None,
        oauth2_bearer: str | None = None,
    ) -> None:
        self._session = Session()

        http_adapter = (
            BearerAuthHTTPAdapter(oauth2_bearer) if oauth2_bearer else HTTPAdapter()
        )
        self._mount_session("http://", http_adapter)
        self._mount_session("https://", http_adapter)
        self._mount_session("file://", FileAdapter())
        self._mount_session(
            "oci://",
            OCIAdapter(
                hostname=oci_hostname,
                username=oci_username,
                password=oci_password,
            ),
        )

    def _mount_session(self, scheme: str, adapter: BaseAdapter) -> None:
        logger.debug(f"Mounting '{scheme}' scheme to '{type(adapter).__name__}'...")
        self._session.mount(scheme, adapter)
        logger.debug(
            f"Scheme '{scheme}' successfully mount to '{type(adapter).__name__}'"
        )

    def resolve(self, location: str) -> TranspilerContext:
        location, separator, process_id = location.partition("#")

        if separator and not process_id:
            raise PluginExecutionError(f"Empty process id in location '{location}'")

        source: Path | AnyUrl = (
            AnyUrl(location)
            if _is_url(path_or_url=location, session=self._session)
            else Path(location).absolute()
        )

        try:
            cwl_document: list[Process] | Process = load_cwl_from_location(
                path=location, session=self._session
            )

            metadata: SoftwareApplication = software_application_from_process(cwl_document)
        except Exception as exc:
            raise PluginFailureError(
                f"Impossible to load a CWL document from {location}"
            ) from exc

        resolved_process: Process | None = None

        if process_id:
            resolved_process = search_process(
                process_id=process_id, process=cwl_document
            )
            if resolved_process is None:
                raise PluginFailureError(
                    f"Process {process_id} does not exist in {location}, "
                    f"only {get_ids(cwl_document)} available."
                )
        elif isinstance(cwl_document, list):
            logger.warning(
                f"Process list found from {location}, but no process id "
                f"was provided via '{location}#<process-id>'; "
                f"{get_ids(cwl_document)} available."
            )
        else:
            resolved_process = cwl_document

        if resolved_process:
            logger.debug(
                f"Selected '{resolved_process.class_}' Process '{resolved_process.id}' from {location}"
            )

        return TranspilerContext(
            source=source,
            metadata=metadata,
            document=(
                cwl_document
                if isinstance(cwl_document, Process)
                else tuple(cwl_document)
            ),
            resolved_process=resolved_process,
            resolver=self,
        )
