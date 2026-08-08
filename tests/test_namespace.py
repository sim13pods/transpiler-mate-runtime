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

"""Regression tests for API/runtime namespace-package coexistence."""

from __future__ import annotations

import importlib


def test_api_and_runtime_share_native_namespace() -> None:
    namespace = importlib.import_module("transpiler_mate")
    api = importlib.import_module("transpiler_mate.api")
    runtime = importlib.import_module("transpiler_mate.runtime")
    spec = namespace.__spec__

    assert spec is not None
    assert spec.origin is None
    assert spec.submodule_search_locations is not None
    assert api.__package__ == "transpiler_mate.api"
    assert runtime.__package__ == "transpiler_mate.runtime"
