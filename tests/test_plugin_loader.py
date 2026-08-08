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

from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, cast

import pytest
from transpiler_mate.api import EmptyOptions, transpiler_plugin

from transpiler_mate.runtime import plugin_loader

if TYPE_CHECKING:
    from transpiler_mate.api import TranspilerContext, TranspilerPlugin


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        outcome: object | Exception,
    ) -> None:
        self.name = name
        self.value = value
        self._outcome = outcome

    def load(self) -> object:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _entry_point(
    name: str,
    value: str,
    outcome: object | Exception,
) -> EntryPoint:
    return cast("EntryPoint", FakeEntryPoint(name, value, outcome))


def _plugin(name: str = "demo") -> TranspilerPlugin[EmptyOptions]:
    @transpiler_plugin(
        name=name,
        description="Demo plugin",
        options_model=EmptyOptions,
    )
    def execute(
        context: TranspilerContext,
        options: EmptyOptions,
    ) -> None:
        del context, options

    return execute


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    entries: tuple[EntryPoint, ...],
) -> None:
    def fake_entry_points(**params: str) -> tuple[EntryPoint, ...]:
        assert params == {"group": plugin_loader.PLUGIN_ENTRY_POINT_GROUP}
        return entries

    monkeypatch.setattr(plugin_loader, "entry_points", fake_entry_points)


def test_discover_plugins_returns_sorted_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zulu = _entry_point("zulu", "pkg:zulu", _plugin("zulu"))
    alpha = _entry_point("alpha", "pkg:alpha", _plugin("alpha"))
    _install_entry_points(monkeypatch, (zulu, alpha))

    discovered = plugin_loader.discover_plugins()

    assert list(discovered) == ["alpha", "zulu"]
    assert discovered == {"alpha": alpha, "zulu": zulu}


def test_discover_plugins_uses_requested_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: dict[str, str] = {}

    def fake_entry_points(**params: str) -> tuple[EntryPoint, ...]:
        requested.update(params)
        return ()

    monkeypatch.setattr(plugin_loader, "entry_points", fake_entry_points)

    assert plugin_loader.discover_plugins("example.plugins") == {}
    assert requested == {"group": "example.plugins"}


def test_discover_plugins_rejects_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry_point("demo", "first:plugin", _plugin())
    second = _entry_point("demo", "second:plugin", _plugin())
    _install_entry_points(monkeypatch, (first, second))

    with pytest.raises(plugin_loader.DuplicatePluginError) as caught:
        plugin_loader.discover_plugins()

    assert caught.value.name == "demo"
    assert caught.value.entries == (first, second)
    assert "first:plugin" in str(caught.value)
    assert "second:plugin" in str(caught.value)


def test_find_plugin_returns_matching_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _entry_point("demo", "pkg:plugin", _plugin())
    _install_entry_points(monkeypatch, (demo,))

    assert plugin_loader.find_plugin("demo") is demo


def test_find_plugin_reports_missing_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch, ())

    with pytest.raises(plugin_loader.PluginNotFoundError) as caught:
        plugin_loader.find_plugin("missing")

    assert caught.value.name == "missing"
    assert caught.value.group == plugin_loader.PLUGIN_ENTRY_POINT_GROUP


def test_load_plugin_returns_valid_plugin() -> None:
    plugin = _plugin()
    entry_point = _entry_point("demo", "pkg:plugin", plugin)

    assert plugin_loader.load_plugin(entry_point) is plugin


def test_load_plugin_wraps_entry_point_error() -> None:
    cause = RuntimeError("cannot import")
    entry_point = _entry_point("demo", "pkg:plugin", cause)

    with pytest.raises(plugin_loader.PluginLoadError) as caught:
        plugin_loader.load_plugin(entry_point)

    assert caught.value.entry_point is entry_point
    assert caught.value.__cause__ is cause


def test_load_plugin_rejects_invalid_object() -> None:
    invalid = object()
    entry_point = _entry_point("demo", "pkg:plugin", invalid)

    with pytest.raises(plugin_loader.InvalidPluginError) as caught:
        plugin_loader.load_plugin(entry_point)

    assert caught.value.entry_point is entry_point
    assert caught.value.plugin is invalid


def test_load_plugin_by_name_discovers_then_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin()
    entry_point = _entry_point("demo", "pkg:plugin", plugin)
    _install_entry_points(monkeypatch, (entry_point,))

    assert plugin_loader.load_plugin_by_name("demo") is plugin


def test_load_plugins_loads_all_discovered_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = _plugin("alpha")
    beta = _plugin("beta")
    _install_entry_points(
        monkeypatch,
        (
            _entry_point("beta", "pkg:beta", beta),
            _entry_point("alpha", "pkg:alpha", alpha),
        ),
    )

    loaded = plugin_loader.load_plugins()

    assert loaded == {"alpha": alpha, "beta": beta}
