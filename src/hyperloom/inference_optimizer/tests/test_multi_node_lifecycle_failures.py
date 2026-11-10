# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node lifecycle probes poll through transport failures and let anything else surface."""

from __future__ import annotations

import asyncio
import json
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
        asyncio.run(life.restart_server_for_round(life.RoundServer(framework="vllm", model_path="/model", tp=1, ep=1)))

    assert calls == []


def test_a_crashed_worker_of_any_lws_rank_fails_fast(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher names each pod's log after its LWS rank, so a rank-1 crash must be read like a rank-0 one."""
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    (tmp_path / "mn_infera_server_10.0.0.2_r1.log").write_text(
        "loading weights\nlaunch_server.py: error: unrecognized arguments: --bogus-flag\n", encoding="utf-8"
    )
    (tmp_path / "mn_infera_server_10.0.0.3_r2.log").write_text(
        "Health check failed: detokenizer did not respond\n" * 15, encoding="utf-8"
    )

    crash = life._worker_fastfail_reason(["10.0.0.2"], elapsed=60, crash_grace=45, wedge_grace=420, timeout_s=900)
    wedge = life._worker_fastfail_reason(["10.0.0.3"], elapsed=500, crash_grace=45, wedge_grace=420, timeout_s=900)

    assert crash is not None and "unrecognized arguments: --bogus-flag" in crash
    assert wedge is not None and "10.0.0.3" in wedge


_PD_WORKERS = {
    "backend": "sglangrouter",
    "prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0", "enginePort": 8010}],
    "decode_pods": [{"podIP": "10.0.0.2", "podId": "d0", "enginePort": 8020}],
}


def test_pd_worker_health_dials_the_engine_port_of_each_role(serve, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.host}:{request.url.port}")
        return _OK

    serve({"health": [_OK]}, _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(life, "_pd_engines_not_running", lambda _state: [])
    _workers()
    assert seen == ["10.0.0.1:8010", "10.0.0.2:8020"]


def test_pd_worker_wait_fails_when_the_launched_engine_was_replaced(serve, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(life, "_pd_engines_not_running", lambda _state: ["decode 10.0.0.2"])
    serve({"health": [_OK]}, _PD_WORKERS)
    with pytest.raises(life.ServerRestartFailed, match="decode 10.0.0.2"):
        _workers()


def test_pd_engine_profile_dials_the_engine_port_of_each_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))

    asyncio.run(life.trigger_pd_engine_profile("stop"))

    assert seen == ["http://10.0.0.1:8010/stop_profile", "http://10.0.0.2:8020/stop_profile"]


def test_pd_engine_profile_stops_every_role_without_waiting_on_another(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow trace export on prefill must not keep decode recording."""
    real_client = httpx.AsyncClient
    decode_seen = asyncio.Event()
    order: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8010:
            try:
                await asyncio.wait_for(decode_seen.wait(), timeout=2)
                order.append("prefill-after-decode")
            except asyncio.TimeoutError:
                order.append("prefill-alone")
        else:
            decode_seen.set()
            order.append("decode")
        return httpx.Response(200)

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))

    asyncio.run(life.trigger_pd_engine_profile("stop"))

    assert order == ["decode", "prefill-after-decode"]


def test_multi_node_round_skips_the_local_server_argv_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote engine parses the argv, so the sandbox's framework never judges it."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.actions.executors import _multi_node_env, magpie_agentx
    from hyperloom.orchestrator.actions.executors.benchmark_run.launch import _BenchmarkLaunchMixin

    def _must_not_run(_path: object) -> bool:
        raise AssertionError("local argv preflight ran for a multi-node round")

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(magpie_agentx, "native_config", _must_not_run)
    round_ = SimpleNamespace(config_path="/nonexistent.yaml")
    assert _BenchmarkLaunchMixin._preflight_server_argv(SimpleNamespace(), round_, 0) is None


def test_shared_server_log_dir_defaults_to_user_data_server_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_MN_SERVER_LOG_DIR", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", "/shared/hl")
    assert life._shared_server_log_dir() == "/shared/hl/server_logs"
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "/elsewhere/logs")
    assert life._shared_server_log_dir() == "/elsewhere/logs"


def test_pd_engines_lost_reason_names_engines_that_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(life, "_pd_engines_not_running", lambda _state: ["decode 10.0.0.2"])
    reason = life.pd_engines_lost_reason()
    assert reason is not None and "decode 10.0.0.2" in reason
    monkeypatch.setattr(life, "_pd_engines_not_running", lambda _state: [])
    assert life.pd_engines_lost_reason() is None


def test_pd_engine_profile_uses_engine_urls_without_pods(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen: list[tuple[str, dict]] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content or b"{}")))
        return httpx.Response(200)

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: {"backend": "sglangrouter", "prefill_pods": [], "decode_pods": []})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_URL", "http://p-svc:8010/")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_URL", "http://d-svc:8020")

    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))

    assert seen == [
        ("http://p-svc:8010/start_profile", {"output_dir": str(tmp_path / "prefill"), "with_stack": False}),
        ("http://d-svc:8020/start_profile", {"output_dir": str(tmp_path / "decode"), "with_stack": False}),
    ]


@pytest.mark.parametrize(("env_value", "expected"), [(None, False), ("1", True), ("true", True), ("0", False)])
def test_pd_engine_profile_with_stack_follows_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path, env_value: str | None, expected: bool
) -> None:
    seen: list[tuple[str, dict]] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200)

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    if env_value is None:
        monkeypatch.delenv("HYPERLOOM_MN_PROFILE_WITH_STACK", raising=False)
    else:
        monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WITH_STACK", env_value)

    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))
    asyncio.run(life.trigger_pd_engine_profile("stop"))

    starts = [body for path, body in seen if path == "/start_profile"]
    stops = [body for path, body in seen if path == "/stop_profile"]
    assert len(starts) == 2 and all(body["with_stack"] is expected for body in starts)
    assert stops == [{}, {}]


def test_profiling_round_restart_enables_graph_capture_profiling(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import yaml

    from hyperloom.orchestrator.actions.executors import profile

    monkeypatch.setenv("FRAMEWORK", "sglang")
    shipped = yaml.safe_load(profile._default_profile_config().read_text(encoding="utf-8"))["benchmark"]["envs"]
    calls: list[tuple[life.RoundServer, dict]] = []

    async def _restart(server, **kwargs):
        calls.append((server, kwargs))

    monkeypatch.setattr(life, "restart_server_for_round", _restart)

    assert (
        asyncio.run(profile._restart_round_server({"extra_server_args": "--context-length 11264"}, str(tmp_path)))
        is None
    )
    assert (
        asyncio.run(profile._restart_round_server({"extra_server_args": "--disable-cuda-graph"}, str(tmp_path))) is None
    )

    graph, eager = calls
    assert graph[0].extra_server_args.split() == ["--context-length", "11264", "--enable-profile-cuda-graph"]
    assert graph[1]["torch_profiler_dir"] == str(tmp_path)
    assert graph[1]["extra_env"] == {
        key: str(shipped[key])
        for key in (
            "SGLANG_GRAPH_BATCH_CAPTURE",
            "DEBUG_CLR_GRAPH_PACKET_CAPTURE",
            "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING",
            "DEBUG_HIP_GRAPH_CLASSIC_PATH",
        )
    }
    assert eager[0].extra_server_args == "--disable-cuda-graph"


@pytest.mark.parametrize(
    ("state", "skipped"),
    [
        ({"profile_trace_shared": False, "profile_trace_share_error": "decode pod 10.0.0.2: ssh failed"}, True),
        ({"profile_trace_shared": True}, False),
        ({}, False),
    ],
    ids=["not-shared", "shared", "unprobed"],
)
def test_profile_is_skipped_when_the_trace_dir_is_not_shared(
    monkeypatch: pytest.MonkeyPatch, tmp_path, state: dict, skipped: bool
) -> None:
    from hyperloom.orchestrator.actions.executors import _multi_node_env, profile

    monkeypatch.setattr(_multi_node_env, "_read_state", lambda: dict(state))

    refusal = profile._unshared_trace_dir_refusal(str(tmp_path))

    if skipped:
        assert refusal["status"] == "failed"
        assert refusal["error_class"] == profile.PROFILE_TRACE_NOT_SHARED
        assert "decode pod 10.0.0.2: ssh failed" in refusal["error"]
    else:
        assert refusal is None


def test_pd_role_traces_are_kept_when_the_trace_filesystem_clock_lags(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import os
    import time

    from hyperloom.inference_optimizer.multi_node._internal import pd_cluster
    from hyperloom.orchestrator.actions.executors import profile

    monkeypatch.setattr(pd_cluster, "record_profile_paths", lambda **_kwargs: None)
    root = tmp_path / "torch_trace"
    client_now = time.time() + 1260
    round_start = profile._trace_fs_round_start(str(root), client_now)
    for role in life.PD_PROFILE_ROLES:
        (root / role).mkdir(parents=True)
        (root / role / "2-TP-0.trace.json.gz").write_bytes(b"new")
    stale = root / "decode" / "1-TP-0.trace.json.gz"
    stale.write_bytes(b"stale-and-heavier")
    os.utime(stale, (round_start - 600, round_start - 600))

    result: dict = {}
    profile._record_mn_round_traces(result, str(root), round_start)

    assert result["role_traces"] == {
        "prefill": str(root / "prefill" / "2-TP-0.trace.json.gz"),
        "decode": str(root / "decode" / "2-TP-0.trace.json.gz"),
    }


def test_pd_role_traces_take_the_first_gpu_of_each_node(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import os

    root = tmp_path / "torch_trace"
    (root / "decode").mkdir(parents=True)
    (root / "prefill").mkdir(parents=True)
    for rank in range(16):
        (root / "decode" / f"5-TP-{rank}.trace.json.gz").write_bytes(b"x" * (100 if rank == 3 else 10))
    for rank in range(8):
        older = root / "decode" / f"4-TP-{rank}.trace.json.gz"
        older.write_bytes(b"x")
        os.utime(older, (older.stat().st_mtime - 60, older.stat().st_mtime - 60))
        (root / "prefill" / f"7-TP-{rank}-DECODE.trace.json.gz").write_bytes(b"x" * 50)
        (root / "prefill" / f"7-TP-{rank}-EXTEND.trace.json.gz").write_bytes(b"x")
    pods = [{"podIP": "10.0.0.1"}, {"podIP": "10.0.0.2"}]
    monkeypatch.setattr(life, "_read_state", lambda: {"decode_pods": pods, "prefill_pods": pods[:1]})

    since = min(p.stat().st_mtime for p in (root / "decode").iterdir()) - 1
    by_node = life.select_pd_node_traces(str(root), since)

    assert by_node == {
        "prefill": [str(root / "prefill" / "7-TP-0-EXTEND.trace.json.gz")],
        "decode": [str(root / "decode" / "5-TP-0.trace.json.gz"), str(root / "decode" / "5-TP-8.trace.json.gz")],
    }
    assert life.select_pd_role_traces(str(root), since) == {role: paths[0] for role, paths in by_node.items()}
