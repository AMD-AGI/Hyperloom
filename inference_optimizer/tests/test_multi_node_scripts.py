# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``multi_node/scripts`` launch and kill helpers.

A tiny ``sys.modules`` ray stub lets CI import the scripts without Ray.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne


def _install_min_ray_stub() -> None:
    """Minimal ``ray`` package graph so script modules can import."""
    if "ray" in sys.modules:
        existing = sys.modules["ray"]
        if getattr(existing, "util", None) is not None:
            return
        if "ray.util" not in sys.modules:
            ray_util_fix = types.ModuleType("ray.util")
            ray_util_fix.get_node_ip_address = lambda: "127.0.0.1"
            sys.modules["ray.util"] = ray_util_fix
        existing.util = sys.modules["ray.util"]
        return

    def _remote_decorator(**_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    ray_mod = types.ModuleType("ray")
    ray_mod.init = lambda **kwargs: None
    ray_mod.nodes = lambda: []
    ray_mod.remote = _remote_decorator
    ray_mod.get = lambda ref, timeout=None: ref

    ray_util = types.ModuleType("ray.util")
    ray_util.get_node_ip_address = lambda: "127.0.0.1"

    sched = types.ModuleType("ray.util.scheduling_strategies")

    class NodeAffinitySchedulingStrategy:
        def __init__(self, node_id: str, soft: bool = False) -> None:
            self.node_id = node_id
            self.soft = soft

    sched.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy

    ray_mod.util = ray_util
    sys.modules["ray"] = ray_mod
    sys.modules["ray.util"] = ray_util
    sys.modules["ray.util.scheduling_strategies"] = sched


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script_module(unique_name: str, script_name: str):
    _install_min_ray_stub()
    path = _repo_root() / "multi_node" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kill_remote_missing_pid_dir():
    km = _load_script_module("km_test_missing", "kill_multinode.py")
    out = km._kill_remote("/no/such/dir/exists", grace_sec=1)
    assert out == {"killed": [], "stale": [], "missing": []}


def test_kill_remote_non_digit_pid_file_removed(tmp_path):
    km = _load_script_module("km_test_nondigit", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text("not-a-number", encoding="utf-8")
    out = km._kill_remote(str(d), grace_sec=0)
    assert "rank_0.pid" in out["stale"]
    assert not (d / "rank_0.pid").exists()


def test_kill_remote_sentinel_zero_pid_removed(tmp_path):
    km = _load_script_module("km_test_zero", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_1.pid").write_text("0", encoding="utf-8")
    out = km._kill_remote(str(d), grace_sec=0)
    assert any("rank_1.pid" in s for s in out["stale"])
    assert not (d / "rank_1.pid").exists()


def test_kill_remote_dead_pid_stale(tmp_path, monkeypatch):
    km = _load_script_module("km_test_dead", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_2.pid").write_text("99999", encoding="utf-8")

    def _kill(pid, sig):
        if sig == 0:
            raise ProcessLookupError()

    monkeypatch.setattr("os.kill", _kill)
    out = km._kill_remote(str(d), grace_sec=0)
    assert any("rank_2.pid" in s for s in out["stale"])
    assert not (d / "rank_2.pid").exists()


def test_kill_remote_sigterms_then_process_exits(tmp_path, monkeypatch):
    km = _load_script_module("km_test_term", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text("4242", encoding="utf-8")

    after_term = {"done": False}

    def _kill(pid, sig):
        if sig == 0:
            if after_term["done"]:
                raise ProcessLookupError()
            return None
        assert pid == 4242

    def _getpgid(pid):
        assert pid == 4242
        return 424200

    def _killpg(pgid, sig):
        assert pgid == 424200
        assert sig in (signal.SIGTERM, signal.SIGKILL)
        if sig == signal.SIGTERM:
            after_term["done"] = True

    monkeypatch.setattr("os.kill", _kill)
    monkeypatch.setattr("os.getpgid", _getpgid)
    monkeypatch.setattr("os.killpg", _killpg)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = km._kill_remote(str(d), grace_sec=0)
    assert any(x.startswith("rank_0.pid:") for x in out["killed"])
    assert not (d / "rank_0.pid").exists()


def test_pd_decode_dist_init_port_derives_from_prefill():
    """PD-disaggregated decode rendezvous port = prefill port + 1, so an operator override shifts both in lock-step (regression guard for the hard-coded `_PD_DECODE_DIST_INIT_PORT`)."""
    lm = _load_script_module("lm_test_pd_decode_port", "launch_multinode.py")
    # Default: 29500 → 29501
    assert lm._pd_decode_dist_init_port(lm._DEFAULT_DIST_INIT_PORT) == 29501
    # Operator override (the exact override example the source-comment cites)
    assert lm._pd_decode_dist_init_port(29501) == 29502
    # Arbitrary value: still + 1
    assert lm._pd_decode_dist_init_port(40000) == 40001
    # Hard-coded constant must NOT come back (regression guard).
    assert not hasattr(lm, "_PD_DECODE_DIST_INIT_PORT"), (
        "_PD_DECODE_DIST_INIT_PORT was reintroduced — decode port must "
        "derive from args.dist_init_port + 1, not a separate constant, "
        "or override of $RAYJOB_DIST_INIT_PORT silently collides."
    )


def test_build_sglang_cmd_head_has_host_port():
    lm = _load_script_module("lm_test_sglang_head", "launch_multinode.py")
    cmd = lm._build_sglang_cmd(
        model="/m",
        tp=2,
        nnodes=2,
        node_rank=0,
        dist_init_addr="10.0.0.1:5000",
        extra_args=["--x"],
    )
    assert "--host" in cmd and "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "8888"
    assert "--x" in cmd


def test_build_sglang_cmd_worker_omits_http_bind():
    lm = _load_script_module("lm_test_sglang_worker", "launch_multinode.py")
    cmd = lm._build_sglang_cmd(
        model="/m",
        tp=2,
        nnodes=2,
        node_rank=1,
        dist_init_addr="10.0.0.1:5000",
        extra_args=[],
    )
    assert "--host" not in cmd
    assert "--port" not in cmd


def test_build_vllm_cmd_includes_ray_backend():
    lm = _load_script_module("lm_test_vllm", "launch_multinode.py")
    cmd = lm._build_vllm_cmd(model="/m", tp=16, extra_args=["--enforce-eager"])
    assert cmd[0] == "vllm"
    assert "--distributed-executor-backend" in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "16"
    assert "--enforce-eager" in cmd


def test_subprocess_env_prepends_venv_bin(monkeypatch):
    lm = _load_script_module("lm_test_env", "launch_multinode.py")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = lm._subprocess_env()
    assert env["PATH"].startswith("/opt/venv/bin:")


def test_subprocess_env_idempotent_if_already_first(monkeypatch):
    lm = _load_script_module("lm_test_env2", "launch_multinode.py")
    monkeypatch.setenv("PATH", "/opt/venv/bin:/usr/bin")
    env = lm._subprocess_env()
    assert env["PATH"].startswith("/opt/venv/bin")
    assert env["PATH"].count("/opt/venv/bin") == 1


def test_detach_framework_launch_starts_sleep(tmp_path):
    lm = _load_script_module("lm_test_detach_sleep", "launch_multinode.py")
    log_f = tmp_path / "r0.log"
    pid_f = tmp_path / "r0.pid"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    pid = lm._detach_framework_launch(
        ["sleep", "60"],
        log_f,
        pid_f,
        env,
        node_rank=0,
    )
    assert int(pid_f.read_text(encoding="utf-8").strip()) == pid
    os.kill(pid, 0)
    os.kill(pid, signal.SIGTERM)


def test_detach_framework_launch_raises_on_immediate_exit(tmp_path):
    lm = _load_script_module("lm_test_detach_false", "launch_multinode.py")
    log_f = tmp_path / "r0.log"
    pid_f = tmp_path / "r0.pid"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        lm._detach_framework_launch(
            ["sh", "-c", "exit 42"],
            log_f,
            pid_f,
            env,
            node_rank=0,
        )
    except RuntimeError as exc:
        assert "immediately" in str(exc).lower() or "not alive" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_pick_head_first_orders_local_first(monkeypatch):
    lm = _load_script_module("lm_test_pick", "launch_multinode.py")
    monkeypatch.setattr(
        lm.ray.util, "get_node_ip_address", lambda: "10.0.0.1",
    )
    nodes = [
        {"NodeManagerAddress": "10.0.0.2", "NodeID": "b"},
        {"NodeManagerAddress": "10.0.0.1", "NodeID": "a"},
    ]
    ordered = lm._pick_head_first(nodes)
    assert ordered[0]["NodeManagerAddress"] == "10.0.0.1"
    assert ordered[1]["NodeManagerAddress"] == "10.0.0.2"


def test_pick_head_first_fallback_sorts_when_local_missing(monkeypatch):
    lm = _load_script_module("lm_test_pick2", "launch_multinode.py")
    monkeypatch.setattr(
        lm.ray.util, "get_node_ip_address", lambda: "192.168.99.99",
    )
    nodes = [
        {"NodeManagerAddress": "10.0.0.2", "NodeID": "b"},
        {"NodeManagerAddress": "10.0.0.1", "NodeID": "a"},
    ]
    ordered = lm._pick_head_first(nodes)
    addrs = [n["NodeManagerAddress"] for n in ordered]
    assert addrs == sorted(addrs)


def test_wait_for_nodes_returns_when_enough_gpu_nodes(monkeypatch):
    lm = _load_script_module("lm_test_wait_ok", "launch_multinode.py")
    sample = [
        {"Alive": True, "Resources": {"GPU": 1.0}},
        {"Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr(lm.ray, "nodes", lambda: sample)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    got = lm._wait_for_nodes(2, timeout_s=10)
    assert len(got) == 2


def test_wait_for_nodes_times_out(monkeypatch):
    lm = _load_script_module("lm_test_wait_to", "launch_multinode.py")
    one = [{"Alive": True, "Resources": {"GPU": 1.0}}]
    monkeypatch.setattr(lm.ray, "nodes", lambda: one)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    times = iter([0.0, 0.0, 200.0])

    def _mono():
        return next(times)

    monkeypatch.setattr(lm.time, "monotonic", _mono)
    try:
        lm._wait_for_nodes(3, timeout_s=120)
    except RuntimeError as exc:
        assert "only 1/3" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_launch_main_rejects_nnodes_below_two(monkeypatch):
    lm = _load_script_module("lm_test_main_nnodes", "launch_multinode.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_multinode.py",
            "--framework",
            "sglang",
            "--model",
            "/m",
            "--tp",
            "1",
            "--nnodes",
            "1",
            "--pid-dir",
            "/tmp",
            "--log-dir",
            "/tmp",
        ],
    )
    assert lm.main() == 2


def test_spawn_remote_vllm_worker_writes_sentinel_pid(tmp_path):
    lm = _load_script_module("lm_test_spawn_vllm", "launch_multinode.py")
    pid_dir = tmp_path / "p"
    log_dir = tmp_path / "l"
    rc = lm._spawn_remote(
        framework="vllm",
        model="/m",
        tp=8,
        nnodes=2,
        node_rank=1,
        head_ip="127.0.0.1",
        dist_init_port=5000,
        pid_dir=str(pid_dir),
        log_dir=str(log_dir),
        extra_args=[],
    )
    assert rc == 0
    assert (pid_dir / "rank_1.pid").read_text(encoding="utf-8") == "0"


def test_wait_health_true_on_200(monkeypatch):
    lm = _load_script_module("lm_test_health", "launch_multinode.py")
    calls = {"n": 0}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(*_a, **_k):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        _urlopen,
    )
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    assert lm._wait_health(timeout_s=30) is True
    assert calls["n"] == 1


def test_wait_health_false_on_timeout(monkeypatch):
    lm = _load_script_module("lm_test_health2", "launch_multinode.py")

    def _urlopen(*_a, **_k):
        raise OSError("down")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    times = iter([0.0, 100.0])

    def _mono():
        return next(times)

    monkeypatch.setattr(lm.time, "monotonic", _mono)
    assert lm._wait_health(timeout_s=5) is False


def test_find_head_pod_ip_prefers_kuberay_head_suffix():
    """SaFE may list submitter as resourceId 0; real Ray head podId contains '-head-'."""
    from inference_optimizer.multi_node import cli as mn_cli

    wl = {
        "pods": [
            {
                "podId": "primus-claw-40290b670f98c7-twr9t-khlfj",
                "resourceId": 0,
                "podIP": "172.16.152.122",
            },
            {
                "podId": "primus-claw-40290b670f98c7-twr9t-x6fkf-head-ddtvg",
                "resourceId": 1,
                "podIP": "10.245.131.77",
            },
        ],
    }
    assert mn_cli._find_head_pod_ip(wl) == "10.245.131.77"


def test_find_head_pod_ip_fallback_resource_id_zero():
    from inference_optimizer.multi_node import cli as mn_cli

    wl = {
        "pods": [
            {"podId": "legacy-pod", "resourceId": 0, "podIP": "10.0.0.1"},
        ],
    }
    assert mn_cli._find_head_pod_ip(wl) == "10.0.0.1"


def test_build_rayjob_entrypoints_empty_submitter_tail():
    import base64

    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        workspace="core42-hyperloom",
        display_name="t",
        image="harbor.example/sglang:1",
        nodes=2,
        gpus_per_node=8,
        cpus_per_node=96,
        mem_gi_per_node=1024,
        ephemeral_gi_per_node=400,
    )
    assert b["entryPoints"] == ["", ""]
    dec = base64.b64decode(b["env"]["RAY_JOB_ENTRYPOINT"]).decode().strip()
    assert dec == "tail -f /dev/null"


# (formerly test_multi_node_env_ray.py)

# Common kwargs for builder tests; keeps each test focused on the one field under test.
_BUILDER_MIN_KWARGS = dict(
    workspace="ws-a",
    display_name="t",
    image="img:1",
    nodes=2,
    gpus_per_node=8,
    cpus_per_node=96,
    mem_gi_per_node=1024,
    ephemeral_gi_per_node=400,
)


def test_extra_env_rayjob_long_lived_passthrough():
    # RAYJOB_LONG_LIVED is no longer stripped; user-supplied values reach body.env unchanged.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAYJOB_LONG_LIVED": "true", "NCCL_DEBUG": "INFO"},
        **_BUILDER_MIN_KWARGS,
    )
    assert b["env"].get("RAYJOB_LONG_LIVED") == "true"
    assert b["env"].get("NCCL_DEBUG") == "INFO"


def test_extra_env_ray_job_entrypoint_still_stripped_and_forced():
    # RAY_JOB_ENTRYPOINT is reserved: the builder overwrites it with base64("tail -f /dev/null") so the cluster lives the whole session.
    import base64

    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAY_JOB_ENTRYPOINT": base64.b64encode(b"echo hi").decode()},
        **_BUILDER_MIN_KWARGS,
    )
    decoded = base64.b64decode(b["env"]["RAY_JOB_ENTRYPOINT"]).decode().strip()
    assert decoded == "tail -f /dev/null"


def test_session_id_injects_primus_claw_label():
    # session_id emits the primus-claw/session-id label so Brain can correlate the RayJob with its parent session.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="sess-123",
        **_BUILDER_MIN_KWARGS,
    )
    assert b["labels"].get("primus-claw/session-id") == "sess-123"
    # The builder writes no primus-safe.* labels.
    assert not any(
        k.startswith("primus-safe.") for k in b["labels"]
    )


def test_session_id_omitted_skips_label():
    # No session_id → the label is absent rather than written with an empty value.
    from inference_optimizer.multi_node._internal import workload_spec

    b_none = workload_spec.build_rayjob_workload_body(
        session_id=None,
        **_BUILDER_MIN_KWARGS,
    )
    b_empty = workload_spec.build_rayjob_workload_body(
        session_id="",
        **_BUILDER_MIN_KWARGS,
    )
    b_default = workload_spec.build_rayjob_workload_body(**_BUILDER_MIN_KWARGS)
    for b in (b_none, b_empty, b_default):
        assert "primus-claw/session-id" not in b["labels"]


def test_extra_label_primus_claw_prefix_still_stripped():
    # The reserved-namespace guard for extra_labels stays intact: users can't inject their own session-id.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="real-sess",
        extra_labels={
            "primus-claw/session-id": "spoofed",  # must be dropped
            "primus-safe.amd.com/ray-role": "worker",  # must be dropped
            "custom.example/team": "infra",  # must survive
        },
        **_BUILDER_MIN_KWARGS,
    )
    # Builder-injected value wins; user spoof is dropped before merge.
    assert b["labels"].get("primus-claw/session-id") == "real-sess"
    # primus-safe.* never reaches body.labels (sanitize strips it).
    assert b["labels"].get("primus-safe.amd.com/ray-role") != "worker"
    # Non-reserved labels survive unchanged.
    assert b["labels"].get("custom.example/team") == "infra"


# ---------------------------------------------------------------------------
# build_dynamo_workload_body tests (Dynamo idle-pod backend).

_DYNAMO_MIN_KWARGS = dict(
    workspace="ws-a",
    display_name="t",
    image="img:1-ssh",
    nodes=2,
    gpus_per_node=8,
    cpus_per_node=96,
    mem_gi_per_node=1024,
    ephemeral_gi_per_node=400,
    ssh_authorized_key="ssh-ed25519 AAAAC3xx mn",
)


def test_dynamo_body_multinode_idle_shape():
    # Multi-node Dynamo body: DynamoDeployment kind, [frontend, worker]
    # resources, worker.replica == node count, multinodeRoles=["worker"],
    # idle worker entryPoint, frontend on :8000.
    import base64

    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_dynamo_workload_body(**_DYNAMO_MIN_KWARGS)

    assert b["groupVersionKind"] == {"kind": "DynamoDeployment", "version": "v1"}
    assert b["dynamoOptions"]["serviceRoles"] == ["frontend", "worker"]
    assert b["dynamoOptions"]["multinodeRoles"] == ["worker"]
    assert b["dynamoOptions"]["backendFramework"] == "sglang"
    assert b["dynamoOptions"]["kvTransferBackend"] == "nixl"

    fe, worker = b["resources"]
    assert fe["replica"] == 1 and "gpu" not in fe
    # worker.replica IS the node count (new API), not a Deployment replica.
    assert worker["replica"] == 2
    assert worker["gpu"] == "8"
    assert worker["sharedMemory"] == "200Gi"
    assert worker["rdmaResource"] == "1"  # cross-node RDMA on multi-node

    # entryPoints are base64; worker is the idle launcher, frontend is :8000.
    fe_ep = base64.b64decode(b["entryPoints"][0]).decode()
    wk_ep = base64.b64decode(b["entryPoints"][1]).decode()
    assert "dynamo.frontend" in fe_ep and "--http-port 8000" in fe_ep
    assert wk_ep == "/usr/local/bin/mn-idle.sh"

    # SSH control-plane env injected; service fronts :8000 (not sglang :8888).
    assert b["env"]["MN_SSH_AUTHORIZED_KEY"] == "ssh-ed25519 AAAAC3xx mn"
    assert b["env"]["MN_SSH_PORT"] == "2222"
    assert b["service"]["port"] == 8000 and b["service"]["targetPort"] == 8000


def test_dynamo_body_single_node_omits_multinode_and_rdma():
    # nodes=1: no multinodeRoles, no rdmaResource (single-pod aggregated).
    from inference_optimizer.multi_node._internal import workload_spec

    kw = dict(_DYNAMO_MIN_KWARGS)
    kw["nodes"] = 1
    b = workload_spec.build_dynamo_workload_body(**kw)

    assert "multinodeRoles" not in b["dynamoOptions"]
    _, worker = b["resources"]
    assert worker["replica"] == 1
    assert "rdmaResource" not in worker


def test_dynamo_body_requires_ssh_key():
    # The idle-pod control plane is unreachable without an authorized key,
    # so the builder fails fast rather than producing a dead deployment.
    import pytest as _pytest

    from inference_optimizer.multi_node._internal import workload_spec

    kw = dict(_DYNAMO_MIN_KWARGS)
    kw["ssh_authorized_key"] = "   "
    with _pytest.raises(ValueError, match="ssh_authorized_key"):
        workload_spec.build_dynamo_workload_body(**kw)


def test_dynamo_body_rejects_bad_enums():
    import pytest as _pytest

    from inference_optimizer.multi_node._internal import workload_spec

    with _pytest.raises(ValueError, match="backend_framework"):
        workload_spec.build_dynamo_workload_body(
            **{**_DYNAMO_MIN_KWARGS, "backend_framework": "tensorrt"},
        )
    with _pytest.raises(ValueError, match="kv_transfer_backend"):
        workload_spec.build_dynamo_workload_body(
            **{**_DYNAMO_MIN_KWARGS, "kv_transfer_backend": "rdma"},
        )


def test_dynamo_body_pd_independent_instances_no_multinode():
    # PD with TP that fits one pod (tp <= gpus_per_node): prefill/decode are
    # independent single-node instances -> NO multinodeRoles, NO rdma.
    # Matches the canonical PD body (replica=2, TP=8).
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_dynamo_workload_body(
        **{**_DYNAMO_MIN_KWARGS, "nodes": 1, "pd_mode": "disaggregated",
           "pd_prefill_nodes": 2, "pd_decode_nodes": 2,
           "pd_prefill_tp": 8, "pd_decode_tp": 8},
    )
    assert b["dynamoOptions"]["serviceRoles"] == ["frontend", "prefill", "decode"]
    assert "multinodeRoles" not in b["dynamoOptions"]
    assert len(b["resources"]) == 3 and len(b["images"]) == 3
    assert b["resources"][1]["replica"] == 2 and b["resources"][2]["replica"] == 2
    assert "rdmaResource" not in b["resources"][1]


def test_dynamo_body_pd_multinode_when_tp_exceeds_node():
    # PD with TP that exceeds one pod's GPUs: prefill/decode span nodes (LWS)
    # -> multinodeRoles + rdmaResource.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_dynamo_workload_body(
        **{**_DYNAMO_MIN_KWARGS, "nodes": 1, "pd_mode": "disaggregated",
           "pd_prefill_nodes": 2, "pd_decode_nodes": 2,
           "pd_prefill_tp": 16, "pd_decode_tp": 16},
    )
    assert b["dynamoOptions"]["multinodeRoles"] == ["prefill", "decode"]
    assert b["resources"][1]["rdmaResource"] == "1"
    assert b["resources"][2]["rdmaResource"] == "1"


def test_dynamo_discover_role_pods_groups_prefill_decode():
    from inference_optimizer.multi_node._internal import dynamo_support

    wl = {"pods": [
        {"podId": "x-frontend-a", "resourceId": 0, "podIP": "10.0.0.9"},
        {"podId": "x-prefillworker-1", "resourceId": 1, "podIP": "10.0.1.1"},
        {"podId": "x-prefillworker-0", "resourceId": 1, "podIP": "10.0.1.0"},
        {"podId": "x-decodeworker-0", "resourceId": 2, "podIP": "10.0.2.0"},
    ]}
    r = dynamo_support.discover_role_pods(wl)
    assert [p["podIP"] for p in r["prefill"]] == ["10.0.1.0", "10.0.1.1"]
    assert [p["podIP"] for p in r["decode"]] == ["10.0.2.0"]
    assert r["frontend"] and not r["worker"]


def test_dynamo_disagg_flags_and_launch_args():
    from inference_optimizer.multi_node._internal import dynamo_support

    df = dynamo_support.disagg_flags("prefill", "nixl")
    assert "--disaggregation-mode prefill" in df
    assert "--disaggregation-transfer-backend nixl" in df
    assert "30001" in df
    assert dynamo_support.disagg_flags("bogus", "nixl") == ""

    la = dynamo_support.build_node_launch_args(
        framework="sglang", model="/m", tp=8, nnodes=1,
        disagg_mode="decode", kv_transfer_backend="nixl",
    )
    assert "--extra-args" in la and "decode" in la


def test_dynamo_body_vllm_backend_and_session_label():
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_dynamo_workload_body(
        **{
            **_DYNAMO_MIN_KWARGS,
            "backend_framework": "vllm",
            "kv_transfer_backend": "mooncake",
            "session_id": "sess-xyz",
        },
    )
    assert b["dynamoOptions"]["backendFramework"] == "vllm"
    assert b["dynamoOptions"]["kvTransferBackend"] == "mooncake"
    assert b["labels"].get("primus-claw/session-id") == "sess-xyz"


# ---------------------------------------------------------------------------
# dynamo_support pure-helper tests (Dynamo backend SSH fan-out).


def test_dynamo_discover_worker_pods_excludes_frontend_sorts_by_ordinal():
    from inference_optimizer.multi_node._internal import dynamo_support

    wl = {"pods": [
        {"podId": "dyn-frontend-abc", "resourceId": 0, "podIP": "10.0.0.9"},
        {"podId": "dyn-worker-1", "resourceId": 1, "podIP": "10.0.0.2"},
        {"podId": "dyn-worker-0", "resourceId": 1, "podIP": "10.0.0.1"},
        {"podId": "dyn-worker-pending", "resourceId": 1, "podIP": ""},
    ]}
    w = dynamo_support.discover_worker_pods(wl)
    assert [p["podIP"] for p in w] == ["10.0.0.1", "10.0.0.2"]
    assert [p["lwsIndex"] for p in w] == [0, 1]


def test_dynamo_frontend_service_url_prefers_live_then_dns():
    from inference_optimizer.multi_node._internal import dynamo_support

    assert (
        dynamo_support.frontend_service_url("wid", "ws")
        == "http://wid.ws.svc.cluster.local:8000"
    )
    assert (
        dynamo_support.frontend_service_url(
            "wid", "ws", {"clusterIp": "10.1.2.3", "port": 8000}
        ) == "http://10.1.2.3:8000"
    )


def test_dynamo_frontend_service_url_internal_domain_and_nested_port():
    from inference_optimizer.multi_node._internal import dynamo_support

    # internalDomain wins.
    assert dynamo_support.frontend_service_url(
        "w", "ns", {"internalDomain": "w.ns.svc.cluster.local:8000",
                    "clusterIp": "1.2.3.4", "port": {"port": 8000}},
    ) == "http://w.ns.svc.cluster.local:8000"
    # Nested port dict (SaFE shape) -> integer port, not the dict repr.
    assert dynamo_support.frontend_service_url(
        "w", "ns", {"clusterIp": "192.168.154.0",
                    "port": {"protocol": "TCP", "port": 8000, "targetPort": 8000}},
    ) == "http://192.168.154.0:8000"


def test_dynamo_build_node_launch_args_sglang_and_kill_only():
    from inference_optimizer.multi_node._internal import dynamo_support

    a = dynamo_support.build_node_launch_args(
        framework="sglang", model="/m/x", tp=16, nnodes=2, ep=16,
        extra_args="--mem-fraction-static 0.7",
    )
    assert "--framework sglang" in a
    assert "--tp 16" in a and "--nnodes 2" in a and "--ep 16" in a
    # Rank is self-determined pod-side from $LWS_WORKER_INDEX, never encoded.
    assert "--node-rank" not in a
    assert "--extra-args" in a and "0.7" in a

    k = dynamo_support.build_node_launch_args(
        framework="vllm", model="", tp=0, nnodes=2, kill_only=True,
    )
    assert "--kill-only" in k and "--framework vllm" in k
    assert "--model" not in k


# ---------------------------------------------------------------------------
# Dynamo kernel ops routing/isolation (apply-patch / kernel-bench / install-geak).


def test_resolve_geak_src_resolution_order(monkeypatch):
    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert mn_cli._resolve_geak_src("/x/geak") == "/x/geak"  # explicit wins
    monkeypatch.setenv("USER_DATA_PATH", "/data")
    assert mn_cli._resolve_geak_src(None) == "/data/runtime/geak"
    monkeypatch.setenv("HYPERLOOM_ROOT", "/r")
    assert mn_cli._resolve_geak_src(None) == "/r/geak"
    monkeypatch.setenv("HYPERLOOM_GEAK_SRC", "/explicit/geak")
    assert mn_cli._resolve_geak_src(None) == "/explicit/geak"


def test_apply_patch_routes_to_dynamo_only_when_backend_dynamo(tmp_path, monkeypatch):
    # backend=dynamo -> _dynamo_apply_patch; backend=rayjob -> legacy head_pod_ip
    # path (EXIT_CONFIG_ERROR without head_pod_ip). Proves isolation.
    from inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    monkeypatch.setattr(mn_cli, "STATE_FILE", sp)  # module-level const
    monkeypatch.setattr(mn_cli, "_dynamo_apply_patch", lambda a: 4242)

    ns = argparse.Namespace(
        patch_file="/p", target_path="/t", backup_dir="/b", kernel_id="k",
        timeout_sec=60, print_logs=False, poll_interval=6, poll_timeout=110,
    )
    sp.write_text('{"backend":"dynamo","nodes":2}', encoding="utf-8")
    assert mn_cli.cmd_apply_patch(ns) == 4242  # routed to dynamo

    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    assert mn_cli.cmd_apply_patch(ns) == mn_cli.EXIT_CONFIG_ERROR  # legacy path


def test_kernel_bench_routes_to_dynamo_only_when_backend_dynamo(tmp_path, monkeypatch):
    from inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    monkeypatch.setattr(mn_cli, "STATE_FILE", sp)
    monkeypatch.setattr(mn_cli, "_dynamo_kernel_bench", lambda a: 7777)

    ns = argparse.Namespace(
        workspace="/w", bench_command="true", files_b64_json="{}",
        result_glob="*.json", timeout_sec=60, print_logs=False,
        poll_interval=6, poll_timeout=110,
    )
    sp.write_text('{"backend":"dynamo","nodes":2}', encoding="utf-8")
    assert mn_cli.cmd_kernel_bench(ns) == 7777
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    assert mn_cli.cmd_kernel_bench(ns) == mn_cli.EXIT_CONFIG_ERROR


def test_install_geak_best_effort_noop_for_rayjob(tmp_path, monkeypatch):
    # The provisioner hook must be a no-op (0) for non-dynamo backends.
    from inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    monkeypatch.setattr(mn_cli, "STATE_FILE", sp)
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    assert mn_cli.install_geak_on_pods_best_effort() == 0


def test_install_oob_and_kernel_tools_noop_for_rayjob(tmp_path, monkeypatch):
    # OOB + combined kernel-tools install hooks must no-op (0) for non-dynamo.
    from inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    monkeypatch.setattr(mn_cli, "STATE_FILE", sp)
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    assert mn_cli.install_oob_on_pods_best_effort() == 0
    assert mn_cli.install_kernel_tools_on_pods_best_effort() == 0


# ---------------------------------------------------------------------------
# dynamo_ssh_env_from_state isolation tests (kernel-agent GEAK SSH placement).


def _write_mn_state(tmp_path, monkeypatch, payload):
    import json as _json
    sp = tmp_path / "mn_state.json"
    sp.write_text(_json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    return sp


def test_dynamo_ssh_env_empty_for_rayjob(tmp_path, monkeypatch):
    # RayJob backend (or single-node) must yield {} so the Ray RAY_ADDRESS
    # placement path is left completely untouched.
    from inference_optimizer.orchestrator.action_executors import _multi_node_env

    _write_mn_state(tmp_path, monkeypatch, {
        "backend": "rayjob", "nodes": 2, "head_pod_ip": "10.0.0.1",
        "ray_address": "10.0.0.1:6379",
    })
    assert _multi_node_env.dynamo_ssh_env_from_state() == {}


def test_dynamo_ssh_env_aggregated_picks_worker_pod(tmp_path, monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env

    _write_mn_state(tmp_path, monkeypatch, {
        "backend": "dynamo", "nodes": 2, "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/mn_ssh/k", "ssh_port": 2222,
    })
    env = _multi_node_env.dynamo_ssh_env_from_state()
    assert env["KERNEL_AGENT_GPU_PLACEMENT"] == "ssh"
    assert env["MN_SSH_HOST"] == "10.0.1.0"
    assert env["MN_SSH_PORT"] == "2222"
    assert env["MN_SSH_KEY"] == "/tmp/mn_ssh/k"


def test_dynamo_ssh_env_pd_picks_prefill_then_decode(tmp_path, monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env

    _write_mn_state(tmp_path, monkeypatch, {
        "backend": "dynamo", "nodes": 2, "pd_mode": "disaggregated",
        "prefill_pod_ips": ["10.0.2.0"], "decode_pod_ips": ["10.0.3.0"],
        "ssh_key_path": "/tmp/mn_ssh/k", "ssh_port": 2222,
    })
    env = _multi_node_env.dynamo_ssh_env_from_state()
    assert env["MN_SSH_HOST"] == "10.0.2.0"  # prefill first


def test_dynamo_ssh_env_empty_without_pods_or_key(tmp_path, monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env

    _write_mn_state(tmp_path, monkeypatch, {
        "backend": "dynamo", "nodes": 2, "worker_pod_ips": [],
        "ssh_key_path": "/tmp/k",
    })
    assert _multi_node_env.dynamo_ssh_env_from_state() == {}


# ---------------------------------------------------------------------------
# _write_rayjob_meta sidecar JSON tests.


def _write_meta_kwargs(**overrides):
    """Default kwargs for _write_rayjob_meta calls under test."""
    defaults = dict(
        wid="wid-abc",
        workspace="ws-a",
        session_id="sess-1",
        owner_id="owner-1",
        display_name="demo",
        nodes=2,
        gpus_per_node=8,
    )
    defaults.update(overrides)
    return defaults


def test_write_rayjob_meta_writes_expected_payload(tmp_path, monkeypatch):
    # Meta lands at <profile_traces>/<wid>/<session_id> with all fields populated and JSON-decodable.
    import json as _json

    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_cli._write_rayjob_meta(**_write_meta_kwargs())

    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    assert meta_path.is_file()
    payload = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["rayjob_id"] == "wid-abc"
    assert payload["session_id"] == "sess-1"
    assert payload["owner_id"] == "owner-1"
    assert payload["workspace"] == "ws-a"
    assert payload["display_name"] == "demo"
    assert payload["nodes"] == 2
    assert payload["gpus_per_node"] == 8
    # created_at is an ISO-8601 UTC string (datetime.isoformat output).
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"].endswith("+00:00")


def test_write_rayjob_meta_skipped_when_session_id_missing(tmp_path, monkeypatch):
    # Empty/None session_id → the helper short-circuits and creates nothing under profile-traces/.
    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_cli._write_rayjob_meta(**_write_meta_kwargs(session_id=None))
    mn_cli._write_rayjob_meta(**_write_meta_kwargs(session_id=""))

    profile_traces = tmp_path / "profile-traces"
    # The directory may not even exist; if it does, it must be empty.
    if profile_traces.exists():
        assert list(profile_traces.iterdir()) == []


def test_write_rayjob_meta_allows_null_owner_id(tmp_path, monkeypatch):
    # owner_id is optional; the meta must still serialize cleanly with owner_id=None.
    import json as _json

    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_cli._write_rayjob_meta(**_write_meta_kwargs(owner_id=None))

    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    payload = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["owner_id"] is None


def test_write_rayjob_meta_best_effort_on_oserror(tmp_path, monkeypatch):
    # A filesystem failure must NOT propagate (meta is audit data); force Path.mkdir to raise OSError.
    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    def _boom(self, *a, **kw):
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr("pathlib.Path.mkdir", _boom)

    # If the helper re-raised, this call would fail the test.
    mn_cli._write_rayjob_meta(**_write_meta_kwargs())

    # And no file should have been created.
    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    assert not meta_path.exists()


def test_ray_gcs_address_from_state_prefers_ray_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"ray_address": "10.1.2.3:6379", "head_pod_ip": "10.9.9.9"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.3:6379"


def test_ray_gcs_address_from_state_fallback_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.1.2.4"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.4:6379"


def test_export_ray_address_to_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    mne.export_ray_address_to_os()
    assert os.environ.get("RAY_ADDRESS") == "10.0.0.5:6379"
