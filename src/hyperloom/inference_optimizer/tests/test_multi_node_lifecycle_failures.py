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
from hyperloom.orchestrator.actions.executors._multi_node_restart_settings import RestartSettings


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted httpx transport and return the list of endpoints it was asked for."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(life.asyncio, "sleep", _no_sleep)
    real_client = httpx.AsyncClient

    def _install(routes: dict[str, list[object]], state: dict[str, object]) -> list[str]:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            route = request.url.path.rsplit("/", 1)[-1]
            script = routes[route]
            outcome = script[min(calls.count(route), len(script) - 1)]
            calls.append(route)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(life, "_read_state", lambda: state)
        monkeypatch.setattr(life, "is_multi_node", lambda: True)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
        return calls

    return _install


_OK = httpx.Response(200)
_BASE = {
    "health": [_OK],
    "models": [httpx.Response(200, json={"data": [{"id": "m"}]})],
    "completions": [httpx.Response(200, json={"usage": {"completion_tokens": 8}})],
}
_INFERA = {"backend": "infera", "service_url": "http://front:8000"}
_WORKERS = {"prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0"}]}


def _profile() -> None:
    asyncio.run(life.trigger_infera_engine_profile("stop"))


def _workers() -> None:
    asyncio.run(life._wait_for_workers_ready_async(RestartSettings.from_env(), timeout_s=600))


def _health() -> None:
    asyncio.run(life._wait_for_server_health_async(RestartSettings.from_env(), timeout_s=600))


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


@pytest.mark.parametrize("name", ["HYPERLOOM_MN_POLL_TIMEOUT_S", "HYPERLOOM_MN_HEALTH_POLL_S"])
def test_an_invalid_restart_setting_fails_before_the_cluster_is_touched(monkeypatch, tmp_path, name) -> None:
    from hyperloom.inference_optimizer.multi_node import cli
    from hyperloom.inference_optimizer.multi_node._internal import external_state
    from hyperloom.orchestrator.actions.executors import _server_patcher

    calls: list[str] = []
    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", dict)
    monkeypatch.setattr(external_state, "external_service_url", lambda: "")
    monkeypatch.setenv(name, "invalid")
    monkeypatch.setenv("HYPERLOOM_MN_RESTART_RECLAIM_RETRY", "1")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    monkeypatch.setattr(_server_patcher, "resolve_sglang_patch_set", lambda: "roofline")
    monkeypatch.setattr(cli, "cmd_apply_tracelens_patch", lambda _ns: calls.append("patch") or 0)
    monkeypatch.setattr(cli, "cmd_restart_server", lambda _ns: calls.append("restart") or 0)
    monkeypatch.setattr(cli, "kill_inference_for_kernel_agent_best_effort", lambda: calls.append("reclaim"))
    monkeypatch.setattr(life, "_collect_worker_server_logs", lambda _state, _why: calls.append("logs"))

    with pytest.raises(life.ServerRestartFailed, match=f"invalid multi-node restart setting: {name}='invalid'"):
        asyncio.run(life.restart_server_for_round(framework="vllm", model_path="/model", tp=1, ep=1))

    assert calls == []
