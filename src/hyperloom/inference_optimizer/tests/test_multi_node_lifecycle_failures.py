# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node lifecycle probes poll through transport failures and let anything else surface."""

from __future__ import annotations

import asyncio
import subprocess

import httpx
import pytest

from hyperloom.inference_optimizer import multi_node
from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as life


class _Resp:
    """Minimal httpx-like response."""

    def __init__(self, status: int, payload: object = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _Client:
    """Async httpx client answering each endpoint from a scripted sequence; the last outcome repeats."""

    def __init__(self, routes: dict[str, list[object]], calls: list[str]) -> None:
        self._routes = routes
        self._seen = dict.fromkeys(routes, 0)
        self.calls = calls

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def _answer(self, url: str) -> _Resp:
        route = url.rsplit("/", 1)[-1]
        self.calls.append(route)
        script = self._routes[route]
        outcome = script[min(self._seen[route], len(script) - 1)]
        self._seen[route] += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]

    async def get(self, url: str, **_kwargs: object) -> _Resp:
        return await self._answer(url)

    async def post(self, url: str, **_kwargs: object) -> _Resp:
        return await self._answer(url)


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted httpx client and return the list of endpoints it was asked for."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(life.asyncio, "sleep", _no_sleep)

    def _install(routes: dict[str, list[object]], state: dict[str, object]) -> list[str]:
        calls: list[str] = []
        monkeypatch.setattr(life, "_read_state", lambda: state)
        monkeypatch.setattr(life, "is_multi_node", lambda: True)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(routes, calls))
        return calls

    return _install


_OK = _Resp(200)
_BASE = {
    "health": [_OK],
    "models": [_Resp(200, {"data": [{"id": "m"}]})],
    "completions": [_Resp(200, {"usage": {"completion_tokens": 8}})],
}
_INFERA = {"backend": "infera", "service_url": "http://front:8000"}
_WORKERS = {"prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0"}]}


def _profile() -> None:
    asyncio.run(life.trigger_infera_engine_profile("stop"))


def _workers() -> None:
    asyncio.run(life._wait_for_workers_ready_async(timeout_s=600, poll_every_s=1))


def _health() -> None:
    asyncio.run(life._wait_for_server_health_async(timeout_s=600, poll_every_s=1))


# site -> (state, routes, call, endpoints requested when its first request fails once)
_SITES = {
    "profile": (_INFERA, {"stop": [_OK]}, _profile, ["stop"]),
    "workers": (_WORKERS, {"health": [_OK]}, _workers, ["health", "health"]),
    "health": ({"service_url": "http://front:8000"}, {"health": [_OK]}, _health, ["health", "health"]),
    "models": (_INFERA, _BASE, _health, ["health", "models", "health", "models", "completions"]),
    "completions": (_INFERA, _BASE, _health, ["health", "models", "completions", "health", "models", "completions"]),
}


def _run_site(serve, monkeypatch, site: str, failure: BaseException) -> list[str]:
    state, base, call, expected = _SITES[site]
    monkeypatch.setenv("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", "1")
    stage = "stop" if site == "profile" else "health" if site in ("workers", "health") else site
    calls = serve({**base, stage: [failure, *base[stage]]}, state)
    call()
    return calls


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
@pytest.mark.parametrize("site", _SITES)
def test_probes_poll_through_a_transport_failure(serve, monkeypatch, site, failure) -> None:
    assert _run_site(serve, monkeypatch, site, failure) == _SITES[site][3]


@pytest.mark.parametrize("site", _SITES)
def test_probes_let_a_non_transport_failure_propagate(serve, monkeypatch, site) -> None:
    with pytest.raises(RuntimeError, match="client bug"):
        _run_site(serve, monkeypatch, site, RuntimeError("client bug"))


_SSH_STATE = {
    "ssh_key_path": "/keys/id",
    "ssh_known_hosts": "/keys/known_hosts",
    "prefill_pods": [
        {"podIP": "10.0.0.1", "podId": "p0", "sshPort": "2233"},
        {"podIP": "10.0.0.2", "podId": "p1"},
    ],
}


@pytest.fixture
def collect(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Run the log sweep with ssh_run replaced and return the tags of the logs it wrote."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(life, "_shared_worker_log_tail", lambda _ip: None)

    def _run(state: dict, fake_ssh) -> list[str]:
        monkeypatch.setattr(multi_node, "ssh_run", fake_ssh)
        life._collect_worker_server_logs(state, "probe failed")
        return sorted(path.name.split("_", 1)[1] for path in (tmp_path / "server_logs").iterdir())

    return _run


def _ssh_answering(failure_for: str, error: BaseException):
    def _fake(target, _command, *, timeout):
        if target.host == failure_for:
            raise error
        return subprocess.CompletedProcess([], 0, stdout="tail\n", stderr="")

    return _fake


def _pods_with_first(**overrides) -> dict:
    first = {"podIP": "10.0.0.1", "podId": "p0", **overrides}
    return {**_SSH_STATE, "prefill_pods": [first, *_SSH_STATE["prefill_pods"][1:]]}


@pytest.mark.parametrize(
    ("state", "error"),
    [
        (_SSH_STATE, OSError("ssh missing")),
        (_SSH_STATE, subprocess.TimeoutExpired(cmd="ssh", timeout=60)),
        (_pods_with_first(sshPort="not-a-port"), OSError("unused")),
    ],
)
def test_log_sweep_skips_a_pod_whose_ssh_cannot_run(collect, state, error) -> None:
    assert collect(state, _ssh_answering("10.0.0.1", error)) == ["prefill_p1.log"]


def test_log_sweep_lets_an_unexpected_ssh_failure_propagate(collect) -> None:
    with pytest.raises(RuntimeError, match="ssh bug"):
        collect(_SSH_STATE, _ssh_answering("10.0.0.1", RuntimeError("ssh bug")))
