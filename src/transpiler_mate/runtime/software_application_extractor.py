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

"""Convert preserved CWL document metadata into ``SoftwareApplication``.

The CWL document is expected to have been loaded through ``cwl-loader``, which
preserves document-level metadata on the resulting ``cwl_utils.parser.Process``
instance(s). This module deliberately operates on that preserved metadata
rather than serializing the parsed CWL document again.

The conversion follows the JSON-LD strategy historically used by
``transpiler-mate``:

1. obtain preserved document metadata from ``cwl-loader``;
2. use ``$namespaces`` as the JSON-LD expansion context;
3. compact against an empty output context, yielding canonical IRIs;
4. validate the result as ``transpiler_mate.api.SoftwareApplication``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from cwl_loader import _preserved_document_metadata
from pyld import jsonld
from transpiler_mate.api import SoftwareApplication

if TYPE_CHECKING:
    from typing import Any

    from cwl_utils.parser import Process

_NAMESPACES_KEY = "$namespaces"


class SoftwareApplicationMetadataError(ValueError):
    """Base error raised before ``SoftwareApplication`` validation can start."""


class MissingSoftwareApplicationMetadataError(SoftwareApplicationMetadataError):
    """Raised when cwl-loader did not preserve document-level metadata."""


def software_application_from_process(
    process: Process | list[Process],
) -> SoftwareApplication:
    """Build a ``SoftwareApplication`` from preserved CWL document metadata.

    Args:
        process:
            A process, or graph of processes, previously loaded by
            ``cwl-loader``.

    Returns:
        The validated ``SoftwareApplication`` represented by the document
        metadata.

    Raises:
        MissingSoftwareApplicationMetadataError:
            If no document metadata can be recovered from the supplied
            process(es).

        pyld.jsonld.JsonLdError:
            If the preserved metadata cannot be processed as JSON-LD.

        pydantic.ValidationError:
            If the compacted JSON-LD does not satisfy the
            ``SoftwareApplication`` model.

    Notes:
        ``cwl_loader._preserved_document_metadata`` intentionally returns the
        first preserved document-level metadata mapping available on the
        supplied process or process list. This function preserves that
        behavior.

        The metadata is deep-copied before being passed to PyLD so conversion
        cannot mutate metadata retained by the parsed CWL process.
    """

    preserved_metadata = _preserved_document_metadata(process)

    if not preserved_metadata:
        raise MissingSoftwareApplicationMetadataError(
            "No preserved CWL document metadata is available. "
            "Load the document with cwl-loader before converting its metadata."
        )

    metadata: dict[str, Any] = deepcopy(dict(preserved_metadata))
    namespaces = metadata.get(_NAMESPACES_KEY)

    compacted = jsonld.compact(
        input_=metadata,
        ctx={},
        options={"expandContext": namespaces},
    )

    return SoftwareApplication.model_validate(
        compacted,
        by_alias=True,
    )


__all__ = [
    "MissingSoftwareApplicationMetadataError",
    "SoftwareApplicationMetadataError",
    "software_application_from_process",
]
