# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registration, discovery and bad-plugin isolation (4.2.1 discovery rules)."""

from __future__ import annotations

import pytest

from hyperloom.common import entry_points as ep
from hyperloom.plugs import Implementation, PlugError, PlugRegistry, Socket
from hyperloom.plugs.builtin import builtin_registry


def _impl(socket="measurement", name="alt", **kwargs):
    return Implementation(socket, name, lambda ctx: object(), **kwargs)


class _Entry:
    def __init__(self, name, loaded=None, error=None, dist=None):
        self.name, self._loaded, self._error, self.dist = name, loaded, error, dist

    def load(self):
        if self._error is not None:
            raise self._error
        return self._loaded


class _Dist:
    name = "vendor-pkg"
    version = "2.0"


def _patch(monkeypatch, groups):
    class _EntryPoints:
        @staticmethod
        def select(*, group):
            return groups.get(group, [])

    monkeypatch.setattr(ep.metadata, "entry_points", _EntryPoints)


def _registry():
    registry = PlugRegistry()
    registry.define(Socket("measurement", object, entry_point_group="hyperloom.plugs.measurement"))
    registry.define(Socket("events", object))
    return registry


def test_registered_implementation_is_found_by_socket_and_name() -> None:
    registry = _registry()
    impl = _impl()
    registry.register(impl)
    assert registry.implementation("measurement", "alt") is impl


def test_defining_a_socket_twice_is_refused() -> None:
    with pytest.raises(PlugError, match="socket 'events' is already defined"):
        _registry().define(Socket("events", object))


def test_registering_for_an_undefined_socket_is_refused() -> None:
    with pytest.raises(PlugError, match="unknown socket 'nope'"):
        _registry().register(_impl("nope"))


def test_registering_a_name_twice_is_refused() -> None:
    registry = _registry()
    registry.register(_impl())
    with pytest.raises(PlugError, match="implementation 'alt' of socket 'measurement' is already registered"):
        registry.register(_impl())


def test_unknown_implementation_names_the_available_ones() -> None:
    registry = _registry()
    registry.register(_impl(name="local"))
    with pytest.raises(PlugError, match="no implementation 'ghost'; available: local"):
        registry.implementation("measurement", "ghost")


def test_discovery_registers_an_implementation_and_a_zero_argument_factory(monkeypatch) -> None:
    group = "hyperloom.plugs.measurement"
    _patch(
        monkeypatch,
        {group: [_Entry("direct", _impl(name="direct"), dist=_Dist()), _Entry("made", lambda: _impl(name="made"))]},
    )
    registry = _registry()
    registry.discover()
    assert registry.implementation("measurement", "direct").origin == "vendor-pkg==2.0"
    assert registry.implementation("measurement", "made").name == "made"
    assert registry.plugin_errors() == {}


def test_a_plugin_cannot_claim_to_be_builtin(monkeypatch) -> None:
    _patch(
        monkeypatch,
        {"hyperloom.plugs.measurement": [_Entry("liar", _impl(name="liar", origin="builtin"), dist=_Dist())]},
    )
    registry = _registry()
    registry.discover()
    assert registry.implementation("measurement", "liar").origin == "vendor-pkg==2.0"


def test_entry_point_name_must_equal_implementation_name(monkeypatch) -> None:
    _patch(monkeypatch, {"hyperloom.plugs.measurement": [_Entry("one", _impl(name="other"))]})
    registry = _registry()
    registry.discover()
    assert "does not match implementation name 'other'" in registry.plugin_errors()["measurement/one"]
    with pytest.raises(PlugError, match="no implementation 'one'"):
        registry.implementation("measurement", "one")


def test_entry_point_must_resolve_to_an_implementation(monkeypatch) -> None:
    _patch(monkeypatch, {"hyperloom.plugs.measurement": [_Entry("odd", 42)]})
    registry = _registry()
    registry.discover()
    assert registry.plugin_errors()["measurement/odd"].startswith(
        "TypeError: entry point must resolve to Implementation"
    )


def test_entry_point_of_another_socket_is_refused(monkeypatch) -> None:
    _patch(monkeypatch, {"hyperloom.plugs.measurement": [_Entry("x", _impl("events", "x"))]})
    registry = _registry()
    registry.discover()
    assert "resolved to an implementation of 'events'" in registry.plugin_errors()["measurement/x"]


def test_a_plugin_cannot_replace_a_builtin_implementation(monkeypatch) -> None:
    _patch(monkeypatch, {"hyperloom.plugs.measurement": [_Entry("local", _impl(name="local"))]})
    registry = _registry()
    builtin = _impl(name="local")
    registry.register(builtin)
    registry.discover()
    assert registry.implementation("measurement", "local") is builtin
    assert "already registered" in registry.plugin_errors()["measurement/local"]


def test_two_plugins_declaring_one_name_keep_the_first_and_record_the_second(monkeypatch) -> None:
    class _OtherDist:
        name = "other-pkg"
        version = "1.0"

    _patch(
        monkeypatch,
        {
            "hyperloom.plugs.measurement": [
                _Entry("dup", _impl(name="dup"), dist=_Dist()),
                _Entry("dup", _impl(name="dup"), dist=_OtherDist()),
            ]
        },
    )
    registry = _registry()
    registry.discover()
    assert registry.implementation("measurement", "dup").origin == "vendor-pkg==2.0"
    assert "already registered" in registry.plugin_errors()["measurement/dup"]


def test_a_broken_plugin_does_not_stop_the_others(monkeypatch) -> None:
    group = "hyperloom.plugs.measurement"
    _patch(monkeypatch, {group: [_Entry("bad", error=RuntimeError("boom")), _Entry("good", _impl(name="good"))]})
    registry = _registry()
    registry.discover()
    assert registry.plugin_errors() == {"measurement/bad": "RuntimeError: boom"}
    assert registry.implementation("measurement", "good").name == "good"


def test_only_sockets_with_an_entry_point_group_are_scanned(monkeypatch) -> None:
    scanned = []

    class _EntryPoints:
        @staticmethod
        def select(*, group):
            scanned.append(group)
            return []

    monkeypatch.setattr(ep.metadata, "entry_points", _EntryPoints)
    _registry().discover()
    assert scanned == ["hyperloom.plugs.measurement"]


def test_unknown_name_error_carries_the_plugin_errors(monkeypatch) -> None:
    _patch(monkeypatch, {"hyperloom.plugs.measurement": [_Entry("bad", error=RuntimeError("boom"))]})
    with pytest.raises(PlugError, match=r"plugin errors: .*measurement/bad.*RuntimeError: boom"):
        _registry().implementation("measurement", "bad")


def test_the_default_registry_scans_no_plugins_until_an_unknown_name_is_asked_for(monkeypatch) -> None:
    scanned = []

    class _EntryPoints:
        @staticmethod
        def select(*, group):
            scanned.append(group)
            return []

    monkeypatch.setattr(ep.metadata, "entry_points", _EntryPoints)
    registry = builtin_registry()
    assert registry.implementation("events", "local").origin == "builtin"
    assert scanned == []
    with pytest.raises(PlugError):
        registry.implementation("measurement", "hosted")
    assert scanned  # a name the built-ins do not have is what sends it looking
