# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gates a restarted multi-node server must pass before a benchmark round runs against it."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

from hyperloom.common.env import EnvValueError
from hyperloom.inference_optimizer.multi_node._internal import external_state
from hyperloom.orchestrator.actions.executors import _multi_node_server_readiness as readiness

_SERVICE_URL = "http://svc.project1-dev.svc.cluster.local:8888"


class _Reply:
    """A scripted HTTP reply: the status it carries, and the JSON body it decodes to."""

    def __init__(self, status: int, payload: object | None = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("response carries no JSON body")
        return self._payload


class _TransportError(Exception):
    """Stand-in for ``httpx.HTTPError``: the one failure class the readiness gates poll through."""


class _ScriptedClient:
    """An httpx-like async client answering each route from its own queue, keyed by a URL fragment."""

    def __init__(self, routes: dict[str, list[object]]) -> None:
        self._routes = sorted(routes.items(), key=lambda item: -len(item[0]))
        self.requested: list[str] = []

    async def __aenter__(self) -> "_ScriptedClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    def _answer(self, url: str) -> _Reply:
        self.requested.append(url)
        for fragment, queue in self._routes:
            if fragment not in url:
                continue
            reply = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(reply, BaseException):
                raise reply
            return reply
        raise ConnectionRefusedError(111, "nothing is listening on " + url)

    async def get(self, url: str) -> _Reply:
        return self._answer(url)

    async def post(self, url: str, json: object = None) -> _Reply:
        return self._answer(url)


def _serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, list[object]]) -> _ScriptedClient:
    """Route the gates' ``import httpx`` to a client answering ``routes``."""
    client = _ScriptedClient(routes)
    module = types.ModuleType("httpx")
    module.AsyncClient = lambda *_a, **_k: client  # type: ignore[attr-defined]
    module.HTTPError = _TransportError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)
    return client


def _clock_advances_only_on_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the poll loops reach their deadlines without any test really waiting."""
    reading = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: reading["now"])

    async def _advance(seconds: float) -> None:
        reading["now"] += seconds

    monkeypatch.setattr(readiness.asyncio, "sleep", _advance)


def _with_state(monkeypatch: pytest.MonkeyPatch, state: dict) -> None:
    """Make both readiness gates read ``state`` instead of the multi_node state file."""
    monkeypatch.setattr(readiness, "_read_state", lambda: state)


def test_worker_targets_are_one_per_pod_ip_in_role_order() -> None:
    """A pod listed under two roles is one worker; a malformed entry is not a worker at all."""
    state = {
        "prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0"}, "not-a-dict", {"podIP": "  "}],
        "decode_pods": [{"podIP": "10.0.0.1"}, {"podIP": "10.0.0.2"}],
        "worker_pods": [{"podIP": "10.0.0.3"}],
    }

    targets = readiness._worker_targets(state)

    assert [(t.role, t.ip, t.pod_id) for t in targets] == [
        ("prefill", "10.0.0.1", "p0"),
        ("decode", "10.0.0.2", "10.0.0.2"),
        ("worker", "10.0.0.3", "10.0.0.3"),
    ]


def test_the_worker_gate_returns_once_every_leg_answers_its_own_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both PD legs must be up before the frontend probe starts, or its grace expires against a half-ready pair."""
    _with_state(monkeypatch, {"prefill_pods": [{"podIP": "10.0.0.1"}], "decode_pods": [{"podIP": "10.0.0.2"}]})
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"10.0.0.1": [_Reply(503), _Reply(200)], "10.0.0.2": [_Reply(200)]})

    asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=600, poll_every_s=10))


def test_the_worker_gate_names_the_legs_still_initializing_when_it_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator needs to know which leg to look at, not just that the wait ended."""
    _with_state(monkeypatch, {"prefill_pods": [{"podIP": "10.0.0.1"}], "decode_pods": [{"podIP": "10.0.0.2"}]})
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"10.0.0.1": [_Reply(200)], "10.0.0.2": [_Reply(503)]})

    with pytest.raises(readiness.ServerRestartFailed, match="10.0.0.2"):
        asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=30, poll_every_s=10))


def test_the_worker_gate_is_a_noop_when_state_lists_no_pods(monkeypatch: pytest.MonkeyPatch) -> None:
    """An external or single-pod deployment has no per-worker /health to gate on."""
    _with_state(monkeypatch, {})
    _serve(monkeypatch, {})

    asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=600, poll_every_s=10))


def test_a_worker_that_crashed_on_startup_fails_the_gate_within_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An argparse rejection is terminal: waiting out the (aiter-widened) budget buys nothing."""
    _with_state(monkeypatch, {"prefill_pods": [{"podIP": "10.0.0.1"}]})
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"10.0.0.1": [_Reply(503)]})
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    (tmp_path / "mn_infera_server_10.0.0.1_r0.log").write_text(
        "sglang: error: unrecognized arguments: --nope\n", encoding="utf-8"
    )

    with pytest.raises(readiness.ServerRestartFailed, match="crashed on startup"):
        asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=3600, poll_every_s=60))


def test_a_detokenizer_wedged_worker_fails_the_gate_before_the_full_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Weights are loaded but /health never flips, so the remaining hours of the budget are wasted."""
    _with_state(monkeypatch, {"prefill_pods": [{"podIP": "10.0.0.1"}]})
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"10.0.0.1": [_Reply(503)]})
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    (tmp_path / "mn_infera_server_10.0.0.1_r0.log").write_text(
        "Health check failed: detokenizer stalled\n" * 20, encoding="utf-8"
    )

    with pytest.raises(readiness.ServerRestartFailed, match="detokenizer-wedged"):
        asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=3600, poll_every_s=500))


def test_the_terminal_state_fast_fail_can_be_switched_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With the fast-fail disabled the gate waits out its budget instead of reading logs."""
    monkeypatch.setenv("HYPERLOOM_MN_WORKER_WEDGE_FASTFAIL", "0")
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    (tmp_path / "mn_infera_server_10.0.0.1_r0.log").write_text("x: error: boom\n", encoding="utf-8")

    readiness._reject_terminal_worker_state(["10.0.0.1"], readiness._worker_gate_rules(), elapsed=600, timeout_s=3600)


def test_the_worker_gate_rejects_a_malformed_knob_before_it_dials_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast-fail windows are parsed before the client opens, so a typo cannot surface mid-poll or go unread."""
    monkeypatch.setenv("HYPERLOOM_MN_WORKER_WEDGE_GRACE_S", "7m")
    monkeypatch.setenv("HYPERLOOM_MN_WORKER_WEDGE_FASTFAIL", "0")
    _with_state(monkeypatch, {"prefill_pods": [{"podIP": "10.0.0.1"}]})
    _clock_advances_only_on_sleep(monkeypatch)
    client = _serve(monkeypatch, {"10.0.0.1": [_Reply(503)]})

    with pytest.raises(EnvValueError):
        asyncio.run(readiness._wait_for_workers_ready_async(timeout_s=3600, poll_every_s=60))

    assert client.requested == []


def test_serving_probe_rules_only_wait_for_model_registration_on_infera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infera lists a model before its workers accept traffic; sglang's /health already means ready."""
    monkeypatch.delenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", raising=False)

    assert readiness._serving_probe_rules("infera").wait_model_ready is True
    assert readiness._serving_probe_rules("rayjob").wait_model_ready is False
    assert readiness._serving_probe_rules("infera").required_streak == 2

    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", "5")
    assert readiness._serving_probe_rules("infera").required_streak == 5


def test_no_configured_probe_streak_still_requires_one_served_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness is a streak of served probes; at zero a server that never answered would count as ready."""
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", "0")
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", "0")
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS", "0")

    rules = readiness._serving_probe_rules("infera")

    assert (rules.required_streak, rules.max_tokens, rules.min_tokens) == (1, 1, 1)


def test_the_models_empty_fast_fail_only_fires_after_health_came_up() -> None:
    """Before /health there is nothing to be late for, and once models registered the window is closed."""
    assert readiness._models_empty_too_long(elapsed=900, health_ok_at=None, models_ready_at=None, grace_s=600) is False
    assert readiness._models_empty_too_long(elapsed=900, health_ok_at=10, models_ready_at=20, grace_s=600) is False
    assert readiness._models_empty_too_long(elapsed=900, health_ok_at=10, models_ready_at=None, grace_s=0) is False
    assert readiness._models_empty_too_long(elapsed=900, health_ok_at=10, models_ready_at=None, grace_s=600) is True


def test_serving_urls_hang_off_one_service_url() -> None:
    urls = readiness._serving_urls(_SERVICE_URL + "/")

    assert urls.health == _SERVICE_URL + "/health"
    assert urls.models == _SERVICE_URL + "/v1/models"
    assert urls.completions == _SERVICE_URL + "/v1/completions"


def _reachable(monkeypatch: pytest.MonkeyPatch, backend: str, url: str = _SERVICE_URL) -> None:
    """Point the frontend gate at ``url`` for a cluster running ``backend``."""
    _with_state(monkeypatch, {"backend": backend, "service_url": url})
    monkeypatch.setattr(external_state, "reachable_service_url", lambda _state: url)


def test_a_non_infera_frontend_is_ready_as_soon_as_health_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only Infera registers a model before it can serve, so elsewhere /health is the whole contract."""
    _reachable(monkeypatch, "rayjob")
    _clock_advances_only_on_sleep(monkeypatch)
    client = _serve(monkeypatch, {"/health": [_Reply(503), _Reply(200)]})

    asyncio.run(readiness._wait_for_server_health_async(timeout_s=600, poll_every_s=10))

    assert not any("/v1/models" in url for url in client.requested)


def test_infera_is_ready_only_after_consecutive_completions_produce_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 from /v1/models is not traffic; the decode leg has to produce tokens, twice running."""
    monkeypatch.delenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", raising=False)
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    client = _serve(
        monkeypatch,
        {
            "/health": [_Reply(200)],
            "/v1/models": [_Reply(200, {"data": [{"id": "m"}]})],
            "/v1/completions": [_Reply(200, {"usage": {"completion_tokens": 8}})],
        },
    )

    asyncio.run(readiness._wait_for_server_health_async(timeout_s=600, poll_every_s=10))

    assert sum("/v1/completions" in url for url in client.requested) == 2


def test_a_completion_that_returns_no_tokens_does_not_count_as_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 200 with an empty body is the broken PD KV handoff this gate exists to catch."""
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(
        monkeypatch,
        {
            "/health": [_Reply(200)],
            "/v1/models": [_Reply(200, {"data": [{"id": "m"}]})],
            "/v1/completions": [_Reply(200, {"usage": {"completion_tokens": 0}})],
        },
    )

    with pytest.raises(readiness.ServerRestartFailed, match="/v1/completions still"):
        asyncio.run(readiness._wait_for_server_health_async(timeout_s=36000, poll_every_s=60))


def test_a_probe_that_never_succeeds_is_not_ready_with_no_configured_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the streak configured to 0 a 503 from /v1/completions must still fail the gate, never pass it."""
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", "0")
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(
        monkeypatch,
        {
            "/health": [_Reply(200)],
            "/v1/models": [_Reply(200, {"data": [{"id": "m"}]})],
            "/v1/completions": [_Reply(503)],
        },
    )

    with pytest.raises(readiness.ServerRestartFailed, match="/v1/completions still"):
        asyncio.run(readiness._wait_for_server_health_async(timeout_s=36000, poll_every_s=60))


def test_an_empty_completion_is_not_ready_with_no_configured_token_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token floor of 0 must not make a 200 that produced nothing count as served traffic."""
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS", "0")
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(
        monkeypatch,
        {
            "/health": [_Reply(200)],
            "/v1/models": [_Reply(200, {"data": [{"id": "m"}]})],
            "/v1/completions": [_Reply(200, {"usage": {"completion_tokens": 0}})],
        },
    )

    with pytest.raises(readiness.ServerRestartFailed, match="/v1/completions still"):
        asyncio.run(readiness._wait_for_server_health_async(timeout_s=36000, poll_every_s=60))


def test_models_that_never_register_fail_before_the_whole_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """/health up with an empty /v1/models means the GPU workers died on launch, not that they are slow."""
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"/health": [_Reply(200)], "/v1/models": [_Reply(200, {"data": []})]})

    with pytest.raises(readiness.ServerRestartFailed, match="/v1/models still empty"):
        asyncio.run(readiness._wait_for_server_health_async(timeout_s=36000, poll_every_s=60))


def test_a_frontend_that_never_answers_fails_on_its_own_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    _reachable(monkeypatch, "rayjob")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(monkeypatch, {"/health": [_TransportError("Connection refused")]})

    with pytest.raises(readiness.ServerRestartFailed, match="did not return 200 within"):
        asyncio.run(readiness._wait_for_server_health_async(timeout_s=30, poll_every_s=10))


def test_a_models_probe_that_errors_leaves_the_gate_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport error on /v1/models is a retry, not a verdict."""
    _reachable(monkeypatch, "infera")
    _clock_advances_only_on_sleep(monkeypatch)
    _serve(
        monkeypatch,
        {
            "/health": [_Reply(200)],
            "/v1/models": [_TransportError("reset"), _Reply(500), _Reply(200, {"data": [{"id": "m"}]})],
            "/v1/completions": [_Reply(200, {"usage": {"completion_tokens": 8}})],
        },
    )

    asyncio.run(readiness._wait_for_server_health_async(timeout_s=600, poll_every_s=10))


def test_the_frontend_gate_skips_a_cluster_with_no_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to dial means nothing to prove; the round fails later with a clearer error."""
    _reachable(monkeypatch, "rayjob", url="")

    asyncio.run(readiness._wait_for_server_health_async(timeout_s=600, poll_every_s=10))


def test_a_restarted_server_is_proven_worker_first_then_frontend_then_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontend probe's grace must not expire against a half-ready pair, and the benchmark dials the published URL."""
    proven: list[str] = []

    async def _record(name: str, **_kwargs: object) -> None:
        proven.append(name)

    monkeypatch.setattr(readiness, "_wait_for_workers_ready_async", lambda **kw: _record("workers", **kw))
    monkeypatch.setattr(readiness, "_wait_for_server_health_async", lambda **kw: _record("reachable", **kw))
    monkeypatch.setattr(readiness, "_wait_for_published_service_ready_async", lambda **kw: _record("published", **kw))

    asyncio.run(readiness.wait_for_restarted_server(900))

    assert proven == ["workers", "reachable", "published"]
