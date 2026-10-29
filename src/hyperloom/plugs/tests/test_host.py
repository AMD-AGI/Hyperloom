# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start order, reverse-order close, the current host and the trace (4.2.1 start/close/current host, 4.2.3)."""

from __future__ import annotations

import pytest

from hyperloom.plugs import (
    Implementation,
    PlugError,
    PlugHost,
    PlugRegistry,
    ResolvedManifest,
    Socket,
    activated,
    active,
)
from hyperloom.plugs.events_local import LocalProgressEvents
from hyperloom.plugs.manifest import SocketSetting


class _Plug:
    def __init__(self, name, log, *, fail_start=False, fail_close=False):
        self.name, self.log, self.fail_start, self.fail_close = name, log, fail_start, fail_close

    def start(self):
        if self.fail_start:
            raise RuntimeError(f"{self.name} refused to start")
        self.log.append(f"start {self.name}")

    def close(self):
        self.log.append(f"close {self.name}")
        if self.fail_close:
            raise OSError(f"{self.name} refused to close")


def _build(log, spec, *, config=None):
    """spec: {socket: (requires, options)}; every socket gets one implementation called 'impl'."""
    registry = PlugRegistry()
    for socket in spec:
        registry.define(Socket(socket, object))
    seen = {}
    for socket, (requires, options) in spec.items():

        def factory(ctx, _socket=socket, _options=options, _requires=requires):
            seen[_socket] = ctx
            for dep in _requires:
                ctx.get(dep)
            return _Plug(_socket, log, **_options)

        registry.register(Implementation(socket, "impl", factory, requires=tuple(requires), origin="builtin"))
    manifest = ResolvedManifest("t", {s: SocketSetting("impl", (config or {}).get(s, {}), "builtin") for s in spec})
    return PlugHost(manifest, registry, session_dir=None, env={}), seen


def test_dependencies_start_before_their_dependents() -> None:
    log = []
    host, _ = _build(log, {"c": (("b",), {}), "b": (("a",), {}), "a": ((), {})})
    host.start()
    assert log == ["start a", "start b", "start c"]


def test_close_runs_in_the_reverse_of_the_start_order() -> None:
    log = []
    host, _ = _build(log, {"c": (("b",), {}), "b": (("a",), {}), "a": ((), {})})
    host.start()
    log.clear()
    host.close()
    assert log == ["close c", "close b", "close a"]
    host.close()
    assert log == ["close c", "close b", "close a"]


def test_a_failing_close_is_recorded_and_the_rest_still_close() -> None:
    log = []
    host, _ = _build(log, {"b": (("a",), {"fail_close": True}), "a": ((), {})})
    host.start()
    log.clear()
    host.close()
    assert log == ["close b", "close a"]
    assert host.close_errors() == {"b": "OSError: b refused to close"}


def test_a_missing_dependency_refuses_the_start() -> None:
    log = []
    host, _ = _build(log, {"b": (("a",), {})})
    with pytest.raises(
        PlugError, match=r"socket 'b' implementation 'impl' requires \['a'\], which the manifest does not select"
    ):
        host.start()
    assert log == []


def test_a_dependency_cycle_refuses_the_start() -> None:
    log = []
    host, _ = _build(log, {"a": (("b",), {}), "b": (("a",), {}), "c": ((), {})})
    with pytest.raises(PlugError, match=r"plug dependencies form a cycle among \['a', 'b'\]"):
        host.start()
    assert log == []


def test_an_unknown_implementation_refuses_the_start_with_the_plugin_errors() -> None:
    registry = PlugRegistry()
    registry.define(Socket("s", object))
    registry._errors["s/broken"] = "RuntimeError: boom"
    host = PlugHost(ResolvedManifest("t", {"s": SocketSetting("ghost", {}, "cli")}), registry, session_dir=None)
    with pytest.raises(PlugError, match=r"no implementation 'ghost'.*plugin errors:.*s/broken.*RuntimeError: boom"):
        host.start()


def test_a_start_failure_closes_what_already_started_and_names_the_socket() -> None:
    log = []
    host, _ = _build(log, {"b": (("a",), {"fail_start": True}), "a": ((), {})})
    with pytest.raises(
        PlugError, match="socket 'b' implementation 'impl' failed to start: RuntimeError: b refused to start"
    ):
        host.start()
    assert log == ["start a", "close a"]
    with pytest.raises(PlugError, match="socket 'a' is not started"):
        host.get("a")


def test_get_returns_the_started_implementation_and_refuses_an_unstarted_socket() -> None:
    log = []
    host, _ = _build(log, {"a": ((), {})})
    with pytest.raises(PlugError, match="socket 'a' is not started"):
        host.get("a")
    host.start()
    assert host.get("a").name == "a"


def test_starting_twice_is_refused() -> None:
    host, _ = _build([], {"a": ((), {})})
    host.start()
    with pytest.raises(PlugError, match="already started"):
        host.start()


def test_a_factory_can_only_get_a_socket_it_declared_in_requires() -> None:
    log = []
    registry = PlugRegistry()
    for name in ("a", "b"):
        registry.define(Socket(name, object))
    registry.register(Implementation("a", "impl", lambda ctx: _Plug("a", log)))
    registry.register(Implementation("b", "impl", lambda ctx: ctx.get("a")))
    manifest = ResolvedManifest("t", {n: SocketSetting("impl", {}, "cli") for n in ("a", "b")})
    with pytest.raises(PlugError, match="socket 'b' did not declare 'a' in requires"):
        PlugHost(manifest, registry, session_dir=None).start()


def test_the_factory_receives_expanded_config_and_the_session_dir(tmp_path) -> None:
    registry = PlugRegistry()
    registry.define(Socket("a", object))
    got = {}
    registry.register(Implementation("a", "impl", lambda ctx: got.setdefault("ctx", ctx)))
    manifest = ResolvedManifest("t", {"a": SocketSetting("impl", {"dir": "${ROOT}/ev"}, "cli")})
    PlugHost(manifest, registry, session_dir=tmp_path, host_info="info", env={"ROOT": "/r"}).start()
    ctx = got["ctx"]
    assert (ctx.socket, dict(ctx.config), ctx.session_dir, ctx.host_info) == ("a", {"dir": "/r/ev"}, tmp_path, "info")


def test_describe_traces_the_manifest_without_expanding_variables() -> None:
    registry = PlugRegistry()
    registry.define(Socket("a", object))
    registry.register(Implementation("a", "impl", lambda ctx: object(), origin="vendor==1"))
    manifest = ResolvedManifest("prof", {"a": SocketSetting("impl", {"token": "${SECRET}"}, "locked")})
    host = PlugHost(manifest, registry, session_dir=None, env={"SECRET": "s3cret"})
    traced = host.describe()
    assert traced["sockets"] == {
        "a": {"impl": "impl", "origin": "vendor==1", "layer": "locked", "config": {"token": "${SECRET}"}}
    }
    assert traced["profile"] == "prof" and traced["fingerprint"] == host.fingerprint()
    assert "s3cret" not in repr(traced)


def test_the_fingerprint_follows_the_expanded_configuration_and_the_origin() -> None:
    def fp(env, origin):
        registry = PlugRegistry()
        registry.define(Socket("a", object))
        registry.register(Implementation("a", "impl", lambda ctx: object(), origin=origin))
        manifest = ResolvedManifest("p", {"a": SocketSetting("impl", {"k": "${V}"}, "cli")})
        return PlugHost(manifest, registry, session_dir=None, env=env).fingerprint()

    assert fp({"V": "1"}, "builtin") == fp({"V": "1"}, "builtin")
    assert fp({"V": "1"}, "builtin") != fp({"V": "2"}, "builtin")
    assert fp({"V": "1"}, "builtin") != fp({"V": "1"}, "vendor==1")


def test_active_is_the_builtin_local_host_when_nothing_is_activated() -> None:
    host = active()
    assert isinstance(host.get("events"), LocalProgressEvents)
    assert active() is host


def test_activated_replaces_the_current_host_and_restores_the_previous_one() -> None:
    before = active()
    other, _ = _build([], {"a": ((), {})})
    other.start()
    with activated(other) as inside:
        assert inside is other and active() is other
        nested, _ = _build([], {"a": ((), {})})
        with activated(nested):
            assert active() is nested
        assert active() is other
    assert active() is before
    other.close()


def test_activated_restores_the_previous_host_when_the_body_raises() -> None:
    before = active()
    other, _ = _build([], {"a": ((), {})})

    def body() -> None:
        assert active() is other
        raise RuntimeError("body failed")

    with pytest.raises(RuntimeError):
        with activated(other):
            body()
    restored = active()
    assert restored is before
    assert restored is not other
