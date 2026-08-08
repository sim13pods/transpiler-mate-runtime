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

from typing import TYPE_CHECKING, cast

import pytest
from cwl_utils.parser import Process
from transpiler_mate.api import SoftwareApplication

from transpiler_mate.runtime import software_application_extractor as extractor

if TYPE_CHECKING:
    from typing import Any


def test_missing_preserved_metadata_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_preserved_metadata(process: object) -> None:
        del process

    monkeypatch.setattr(
        extractor,
        "_preserved_document_metadata",
        no_preserved_metadata,
    )

    with pytest.raises(extractor.MissingSoftwareApplicationMetadataError):
        extractor.software_application_from_process(cast("Process", object()))


def test_preserved_metadata_is_compacted_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata: dict[str, Any] = {
        "$namespaces": {"s": "https://schema.org/"},
        "s:name": "Example",
        "nested": {"items": []},
    }
    expected = SoftwareApplication.model_construct()
    observed: dict[str, Any] = {}

    def preserved_metadata(process: object) -> dict[str, Any]:
        del process
        return metadata

    def compact(
        input_: dict[str, Any],
        ctx: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        observed["input"] = input_
        observed["ctx"] = ctx
        observed["options"] = options
        input_["nested"]["items"].append("mutated")
        return {"https://schema.org/name": "Example"}

    def validate(
        value: dict[str, Any],
        *,
        by_alias: bool,
    ) -> SoftwareApplication:
        observed["validated"] = value
        observed["by_alias"] = by_alias
        return expected

    monkeypatch.setattr(
        extractor,
        "_preserved_document_metadata",
        preserved_metadata,
    )
    monkeypatch.setattr(
        "transpiler_mate.runtime.software_application_extractor.jsonld.compact",
        compact,
    )
    monkeypatch.setattr(
        SoftwareApplication,
        "model_validate",
        validate,
    )

    result = extractor.software_application_from_process(cast("Process", object()))

    assert result is expected
    assert observed["ctx"] == {}
    assert observed["options"] == {"expandContext": {"s": "https://schema.org/"}}
    assert observed["validated"] == {"https://schema.org/name": "Example"}
    assert observed["by_alias"] is True
    assert metadata["nested"]["items"] == []
    assert observed["input"] is not metadata
