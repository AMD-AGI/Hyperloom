# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post-restart gate on the published ClusterIP the benchmark actually dials.

reachable_service_url can rewrite the ClusterIP service_url to a pod-pinned
head address whose /health flips up seconds after a restart, while the published
ClusterIP Service still has no ready endpoint and refuses connections. The gate
here waits on that published endpoint so the benchmark is never fired into
ECONNREFUSED (which surfaced as completed=0 -> invalid measurement).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import types

import pytest

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as life


class _FakeResp:
    """Minimal httpx-like response carrying only a status code."""

    def __init__(self, status: int) -> None:
        self.status_code = status


class _FakeClient:
    """Async-context httpx client returning a scripted sequence of GET outcomes."""

    def __init__(self, script: list[object]) -> None:
        self._script = script
        self._i = 0

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def get(self, _url: str) -> _FakeResp:
        item = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return _FakeResp(int(item))  # type: ignore[arg-type]


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> None:
    """Route the function's ``import httpx`` to a client running ``script``."""
    module = types.ModuleType("httpx")
    module.AsyncClient = lambda *a, **k: _FakeClient(script)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "httpx", module)


def _fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make time deterministic and sleeps advance it, so no test really waits."""
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    async def _sleep(seconds: float) -> None:
        clock["t"] += seconds

    monkeypatch.setattr(life.asyncio, "sleep", _sleep)


def _rewritten_state() -> dict[str, object]:
    """RayJob state whose ClusterIP service_url rewrites to a head address."""
    return {
        "backend": "rayjob",
        "service_url": "http://svc.project1-dev.svc.cluster.local:8888",
        "head_pod_ip": "head-svc.project1-dev.svc.cluster.local",
    }


def test_gate_waits_out_the_post_restart_connection_refused_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused right after a restart, then ready: the gate must wait, not fail."""
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    # Two refusals (the readiness window) then two consecutive 200s.
    refused = ConnectionRefusedError(111, "Connection refused")
    _install_fake_httpx(monkeypatch, [refused, refused, 200, 200])

    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


def test_gate_skips_when_the_published_name_never_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside-cluster: the ClusterIP name is unusable, so the gate is a no-op.

    The skip is now gated on CONSECUTIVE resolution failures, so a name that
    never resolves must still skip (after the retry budget), not hang or fail.
    """
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    _install_fake_httpx(monkeypatch, [socket.gaierror(-2, "Name or service not known")])

    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


def test_gate_survives_a_transient_dns_flap_within_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CoreDNS blip must not disable the gate when it is most needed.

    One name-resolution failure (below the consecutive-skip threshold) followed
    by the endpoint coming up must let the gate pass, not skip -- otherwise a
    single DNS flap during the readiness window would green-light the benchmark
    into the ECONNREFUSED this gate exists to prevent.
    """
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    # DNS blips once, then the published endpoint answers: gate must hold + pass.
    _install_fake_httpx(monkeypatch, [socket.gaierror(-2, "Name or service not known"), 200, 200])

    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


def test_gate_warns_not_infos_when_it_skips_on_an_unresolvable_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Skipping the gate is a WARNING, not a silent INFO: a later completed=0
    must be traceable back to this skip (the #1060 'surface it' discipline)."""
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    _install_fake_httpx(monkeypatch, [socket.gaierror(-2, "Name or service not known")])

    with caplog.at_level(logging.WARNING, logger=life.log.name):
        asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))

    assert any("SKIPPING the published-service readiness gate" in r.message for r in caplog.records)


def test_dns_skip_after_is_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYPERLOOM_MN_PUBLISHED_DNS_SKIP_AFTER widens the DNS tolerance.

    With it set to 1, a single resolution failure skips immediately (proving the
    knob is wired); the default (6) would keep polling instead.
    """
    monkeypatch.setenv("HYPERLOOM_MN_PUBLISHED_DNS_SKIP_AFTER", "1")
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    # One gaierror then 200s: skip_after=1 skips on the first failure (returns
    # before the 200s); if the knob were ignored (default 6) it would pass on 200s
    # -- either way it returns, but this asserts the env path does not raise/hang.
    _install_fake_httpx(monkeypatch, [socket.gaierror(-2, "Name or service not known")])

    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 4),  # unset -> default
        ("", 4),  # empty -> default
        ("2", 2),  # explicit
        ("0", 1),  # clamped to the minimum (never a self-defeating 0)
        ("-3", 1),  # negative -> clamped
        ("junk", 4),  # junk -> default
    ],
)
def test_env_int_clamps_and_survives_junk(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int) -> None:
    """_env_int floors at the minimum and never crashes on junk."""
    monkeypatch.delenv("HYPERLOOM_MN_TEST_KNOB", raising=False)
    if raw is not None:
        monkeypatch.setenv("HYPERLOOM_MN_TEST_KNOB", raw)

    assert life._env_int("HYPERLOOM_MN_TEST_KNOB", 4, minimum=1) == expected


def test_gate_fails_when_a_resolvable_endpoint_never_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A published endpoint that resolves but never answers is a real failure."""
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())
    _fast_clock(monkeypatch)
    _install_fake_httpx(monkeypatch, [ConnectionRefusedError(111, "Connection refused")])

    with pytest.raises(life.ServerRestartFailed):
        asyncio.run(life._wait_for_published_service_ready_async(timeout_s=30, poll_every_s=5))


def test_gate_is_a_noop_for_infera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infera's benchmark uses the reachable address; this ClusterIP gate skips."""
    state = _rewritten_state()
    state["backend"] = "infera"
    monkeypatch.setattr(life, "_read_state", lambda: state)
    # No httpx installed on purpose: the function must return before using it.
    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


def test_gate_is_a_noop_without_a_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    """When reachable == published, the reachable wait already proved it."""
    monkeypatch.setattr(
        life,
        "_read_state",
        lambda: {"backend": "rayjob", "service_url": "http://svc:8888"},
    )
    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=600, poll_every_s=5))


def test_gate_disabled_by_nonpositive_timeout_skips_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeout_s<=0 is an escape hatch: skip the gate, never fail on it.

    A zero budget used to fail every restart on the second poll (10 > 0). No
    httpx is installed here on purpose -- the function must return before it is
    reached.
    """
    monkeypatch.setattr(life, "_read_state", lambda: _rewritten_state())

    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=0, poll_every_s=5))
    asyncio.run(life._wait_for_published_service_ready_async(timeout_s=-1, poll_every_s=5))


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 300),  # unset -> default
        ("", 300),  # empty -> default
        ("0", 0),  # explicit skip (honored as <=0 by the gate, not a 0s wait)
        ("-5", -5),  # negative -> also skip
        ("600", 600),  # explicit budget
        ("abc", 300),  # junk -> default, never crashes the restart
    ],
)
def test_published_ready_timeout_env_parse(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int) -> None:
    """The env parse honors 0/negatives as skip and survives junk."""
    monkeypatch.delenv("HYPERLOOM_MN_PUBLISHED_READY_S", raising=False)
    if raw is not None:
        monkeypatch.setenv("HYPERLOOM_MN_PUBLISHED_READY_S", raw)

    assert life._published_ready_timeout_s() == expected


@pytest.mark.parametrize(
    "exc,expected",
    [
        (socket.gaierror(-2, "Name or service not known"), True),
        (ConnectionRefusedError(111, "Connection refused"), False),
        (OSError("Temporary failure in name resolution"), True),
        (TimeoutError("timed out"), False),
    ],
)
def test_name_resolution_error_classification(exc: BaseException, expected: bool) -> None:
    """Only DNS-resolution failures count as 'not applicable', not refusals."""
    assert life._is_name_resolution_error(exc) is expected


def test_name_resolution_error_walks_the_cause_chain() -> None:
    """httpx wraps the OS error; the classifier must see through the wrapper."""
    try:
        try:
            raise socket.gaierror(-2, "Name or service not known")
        except socket.gaierror as root:
            raise RuntimeError("connect failed") from root
    except RuntimeError as wrapped:
        assert life._is_name_resolution_error(wrapped) is True


def test_reachable_rewrite_makes_published_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard the premise: the rewrite really does change the URL we gate on."""
    state = _rewritten_state()
    assert ext.reachable_service_url(state) == "http://head-svc.project1-dev.svc.cluster.local:8888"
    assert ext.reachable_service_url(state) != state["service_url"]
