"""Unit tests for ``multi_node/scripts`` launch and kill helpers.

The scripts depend on ``ray`` at import time (in-cluster runtime). Tests
install a tiny ``sys.modules`` stub so CI can import the modules without
installing Ray, then exercise pure helpers and a few side-effect paths
with mocks / ``tmp_path``.
"""

from __future__ import annotations

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
    """PD-disaggregated decode rendezvous port = prefill port + 1.

    Covers operator override scenario: if `$RAYJOB_DIST_INIT_PORT` shifts
    prefill (e.g. to 29501), decode must shift in lock-step (29502) so
    the two endpoints never collide when both groups land on the same
    host. Regression guard for the previous bug where decode was the
    hard-coded constant `_PD_DECODE_DIST_INIT_PORT = 29501` and silently
    clashed with an override of prefill to 29501.
    """
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


# ===========================================================================
# (formerly test_multi_node_env_ray.py)
# ===========================================================================

# Common kwargs for builder tests below. Keeps each test focused on the
# one field under test instead of repeating the 8 required arguments.
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
    # RAYJOB_LONG_LIVED was a legacy SaFE toggle stripped by an earlier
    # version of the builder. The strip is gone; user-supplied values
    # must now reach body.env unchanged so callers can opt into the
    # legacy code path explicitly if SaFE still honours it.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAYJOB_LONG_LIVED": "true", "NCCL_DEBUG": "INFO"},
        **_BUILDER_MIN_KWARGS,
    )
    assert b["env"].get("RAYJOB_LONG_LIVED") == "true"
    assert b["env"].get("NCCL_DEBUG") == "INFO"


def test_extra_env_ray_job_entrypoint_still_stripped_and_forced():
    # RAY_JOB_ENTRYPOINT remains reserved: the builder strips any
    # user-supplied value and overwrites with base64("tail -f /dev/null")
    # so KubeRay's spec.entrypoint never exits and the cluster lives
    # for the whole session.
    import base64

    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAY_JOB_ENTRYPOINT": base64.b64encode(b"echo hi").decode()},
        **_BUILDER_MIN_KWARGS,
    )
    decoded = base64.b64decode(b["env"]["RAY_JOB_ENTRYPOINT"]).decode().strip()
    assert decoded == "tail -f /dev/null"


def test_session_id_injects_primus_claw_label():
    # When the CLI passes session_id (resolved from $CLAW_SESSION_ID),
    # the builder must emit the primus-claw/session-id label so Brain
    # can correlate the RayJob with its parent sandbox session.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="sess-123",
        **_BUILDER_MIN_KWARGS,
    )
    assert b["labels"].get("primus-claw/session-id") == "sess-123"
    # No primus-safe.* labels are written by the builder (SaFE strips
    # that namespace from caller input -- empty would be a no-op).
    assert not any(
        k.startswith("primus-safe.") for k in b["labels"]
    )


def test_session_id_omitted_skips_label():
    # No session_id (sandbox env var unset) means the label is absent
    # rather than being written with an empty value. Mirrors the
    # ownerId / description optional-field semantics.
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
    # The reserved-namespace guard for caller-supplied extra_labels must
    # remain intact even though the builder now writes a primus-claw/*
    # label itself. Users cannot inject their own session-id by bypassing
    # the builder parameter.
    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="real-sess",
        extra_labels={
            "primus-claw/session-id": "spoofed",  # must be dropped
            "example-internal-host.invalid/ray-role": "worker",  # must be dropped
            "custom.example/team": "infra",  # must survive
        },
        **_BUILDER_MIN_KWARGS,
    )
    # Builder-injected value wins; user spoof is dropped before merge.
    assert b["labels"].get("primus-claw/session-id") == "real-sess"
    # primus-safe.* never reaches body.labels (sanitize strips it).
    assert b["labels"].get("example-internal-host.invalid/ray-role") != "worker"
    # Non-reserved labels survive unchanged.
    assert b["labels"].get("custom.example/team") == "infra"


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
    # Happy path: meta lands at <profile_traces>/<wid>/<session_id> with
    # all the documented fields populated and JSON-decodable.
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
    # Empty / None session_id leaves us without a filename, so the helper
    # MUST short-circuit and create nothing under profile-traces/. Matches
    # the label-skip semantics in cmd_create_rayjob.
    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_cli._write_rayjob_meta(**_write_meta_kwargs(session_id=None))
    mn_cli._write_rayjob_meta(**_write_meta_kwargs(session_id=""))

    profile_traces = tmp_path / "profile-traces"
    # The directory may not even exist; if it does, it must be empty.
    if profile_traces.exists():
        assert list(profile_traces.iterdir()) == []


def test_write_rayjob_meta_allows_null_owner_id(tmp_path, monkeypatch):
    # owner_id is optional in the workload body (no $WORKLOAD_ID exported
    # in dev). The meta must still serialize cleanly with owner_id=None
    # so callers don't need to special-case it.
    import json as _json

    from inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_cli._write_rayjob_meta(**_write_meta_kwargs(owner_id=None))

    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    payload = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["owner_id"] is None


def test_write_rayjob_meta_best_effort_on_oserror(tmp_path, monkeypatch):
    # Filesystem failure (read-only mount, quota, permission) MUST NOT
    # propagate: meta is audit data and failing RayJob creation over it
    # would be a worse outcome. We force Path.mkdir to raise OSError and
    # assert the call returns normally.
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
