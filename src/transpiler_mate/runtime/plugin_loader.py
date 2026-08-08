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

"""Discover and load installed transpiler-mate plugins.

Plugin distributions advertise one plugin object through a standard Python
entry point, for example::

    [project.entry-points."transpiler_mate.plugins"]
    argo = "transpiler_mate_argo:plugin"

Discovery is intentionally separate from loading:

* :func:`discover_plugins` reads installed package metadata only.
* :func:`load_plugin` imports and validates one selected plugin.
* :func:`load_plugins` is a convenience for loading every discovered plugin.

The entry-point name is the runtime discovery identifier. Plugin loading is
lazy: importing the plugin implementation does not happen until ``load()`` is
invoked on its entry point.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from transpiler_mate.api import TranspilerPlugin

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint
    from typing import Any, Final

PLUGIN_ENTRY_POINT_GROUP: Final = "transpiler_mate.plugins"


class PluginLoaderError(RuntimeError):
    """Base class for plugin discovery and loading errors."""


class DuplicatePluginError(PluginLoaderError):
    """Raised when multiple installed entry points use the same plugin name."""

    def __init__(self, name: str, entries: tuple[EntryPoint, ...]) -> None:
        self.name = name
        self.entries = entries

        definitions = ", ".join(entry.value for entry in entries)
        super().__init__(f"Multiple plugins are registered as {name!r}: {definitions}")


class PluginNotFoundError(PluginLoaderError):
    """Raised when a requested plugin is not installed."""

    def __init__(self, name: str, group: str) -> None:
        self.name = name
        self.group = group
        super().__init__(
            f"No plugin named {name!r} is registered in entry-point group {group!r}"
        )


class PluginLoadError(PluginLoaderError):
    """Raised when an installed plugin entry point cannot be loaded."""

    def __init__(self, entry_point: EntryPoint) -> None:
        self.entry_point = entry_point
        super().__init__(
            f"Unable to load plugin {entry_point.name!r} from {entry_point.value!r}"
        )


class InvalidPluginError(PluginLoaderError):
    """Raised when a loaded entry point does not satisfy TranspilerPlugin."""

    def __init__(self, entry_point: EntryPoint, plugin: object) -> None:
        self.entry_point = entry_point
        self.plugin = plugin
        super().__init__(
            f"Entry point {entry_point.name!r} loaded {type(plugin)!r}, "
            "which does not implement TranspilerPlugin"
        )


def discover_plugins(
    group: str = PLUGIN_ENTRY_POINT_GROUP,
) -> dict[str, EntryPoint]:
    """Discover installed plugins without importing their implementations.

    Returns a mapping from entry-point name to entry-point metadata.

    Raises:
        DuplicatePluginError:
            If two or more installed distributions register the same name in
            the selected entry-point group.
    """

    discovered: dict[str, EntryPoint] = {}
    duplicates: dict[str, list[EntryPoint]] = {}

    for entry_point in entry_points(group=group):
        previous = discovered.get(entry_point.name)

        if previous is None:
            discovered[entry_point.name] = entry_point
            continue

        duplicate_entries = duplicates.setdefault(
            entry_point.name,
            [previous],
        )
        duplicate_entries.append(entry_point)

    if duplicates:
        # Fail deterministically on the lexicographically first duplicate.
        name = min(duplicates)
        raise DuplicatePluginError(name, tuple(duplicates[name]))

    return dict(sorted(discovered.items()))


def find_plugin(
    name: str,
    group: str = PLUGIN_ENTRY_POINT_GROUP,
) -> EntryPoint:
    """Find one installed plugin entry point by name without loading it."""

    plugins = discover_plugins(group)

    try:
        return plugins[name]
    except KeyError as exc:
        raise PluginNotFoundError(name, group) from exc


def load_plugin(entry_point: EntryPoint) -> TranspilerPlugin[Any]:
    """Load and validate one plugin from an entry point.

    The runtime protocol check verifies that the loaded object exposes the
    public ``TranspilerPlugin`` shape. It intentionally does not attempt to
    inspect or enforce callable type annotations at runtime.

    Raises:
        PluginLoadError:
            If importing or resolving the entry point fails.
        InvalidPluginError:
            If the resolved object does not implement ``TranspilerPlugin``.
    """

    try:
        plugin = entry_point.load()
    except Exception as exc:
        raise PluginLoadError(entry_point) from exc

    if not isinstance(plugin, TranspilerPlugin):
        raise InvalidPluginError(entry_point, plugin)

    return plugin


def load_plugin_by_name(
    name: str,
    group: str = PLUGIN_ENTRY_POINT_GROUP,
) -> TranspilerPlugin[Any]:
    """Discover and load one installed plugin by its entry-point name."""

    return load_plugin(find_plugin(name, group))


def load_plugins(
    group: str = PLUGIN_ENTRY_POINT_GROUP,
) -> dict[str, TranspilerPlugin[Any]]:
    """Discover and load every installed plugin in the selected group."""

    return {
        name: load_plugin(entry_point)
        for name, entry_point in discover_plugins(group).items()
    }


__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "DuplicatePluginError",
    "InvalidPluginError",
    "PluginLoadError",
    "PluginLoaderError",
    "PluginNotFoundError",
    "discover_plugins",
    "find_plugin",
    "load_plugin",
    "load_plugin_by_name",
    "load_plugins",
]
