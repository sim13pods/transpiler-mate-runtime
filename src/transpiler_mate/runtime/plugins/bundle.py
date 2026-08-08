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

"""Built-in plugin that serializes the resolved CWL document."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cwl_loader import dump_cwl
from pydantic import BaseModel, ConfigDict, Field
from transpiler_mate.api import PluginExecutionError, transpiler_plugin

if TYPE_CHECKING:
    from transpiler_mate.api import TranspilerContext


class BundleOption(BaseModel):
    """Options accepted by the built-in bundle plugin."""

    model_config = ConfigDict(extra="forbid")

    output: Path = Field(description="Path where the bundled CWL document is written")


@transpiler_plugin(
    name="bundle",
    description="Bundle the resolved CWL document to a local file.",
    options_model=BundleOption,
)
def bundle(context: TranspilerContext, options: BundleOption) -> None:
    """Serialize the resolved CWL document to ``options.output``."""

    document = (
        list(context.document)
        if isinstance(context.document, tuple)
        else context.document
    )

    try:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open("w", encoding="utf-8") as output_stream:
            dump_cwl(document, output_stream)
    except Exception as exc:
        raise PluginExecutionError(
            f"Unable to bundle the resolved CWL document to {options.output}"
        ) from exc


__all__ = ["BundleOption", "bundle"]
