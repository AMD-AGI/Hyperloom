# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Consolidated sole-cover unit tests for common/utility modules.

Tests here cover: common.env, common.io, common.gain_math, common.llm_config,
inference_optimizer credentials, breakdown reporters, orchestrator kb_writeback,
orchestrator retry/backoff, orchestrator actions, orchestrator dispatcher,
orchestrator state/objective, multi-node state paths, framework agent helpers,
model_compat, gpu_types, and CLI multi-node utilities.

These tests were consolidated from several coverage-padding files:
  test_coverage_boost_unit.py, test_coverage_boost2_unit.py,
  test_coverage_gap_units.py, test_coverage_margin3_unit.py,
  test_coverage_margin_unit.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeSafe:
    def __init__(self, *, workload: dict | None = None, service: dict | None = None) -> None:
        self.workload = workload or {"phase": "Running"}
        self.service = service or {"clusterIp": "10.9.0.10", "port": 8000}
        self.created: list[dict] = []
        self.get_calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def create_workload(self, body: dict) -> str:
        self.created.append(body)
        return "wid-test"

    def get_workload(self, wid: str) -> dict:
        self.get_calls.append(wid)
        return dict(self.workload)

    def get_workload_service(self, wid: str) -> dict:
        return dict(self.service)


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> str:
        return self._raw


# ---------------------------------------------------------------------------
# common.env
# ---------------------------------------------------------------------------

def test_common_env_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.common import env

    monkeypatch.delenv("HL_BOOL", raising=False)
    assert env.env_bool("HL_BOOL", default=True) is True
    monkeypatch.setenv("HL_BOOL", " YES ")
    assert env.env_bool("HL_BOOL") is True
    monkeypatch.setenv("HL_BOOL", "0")
    assert env.env_bool("HL_BOOL", default=True) is False

    monkeypatch.setenv("HL_INT", " 7 ")
    assert env.env_int("HL_INT") == 7
    monkeypatch.setenv("HL_INT", "bad")
    assert env.env_int("HL_INT", default=3) == 3

    monkeypatch.setenv("HL_FLOAT", " 2.5 ")
    assert env.env_float("HL_FLOAT") == pytest.approx(2.5)
    monkeypatch.setenv("HL_FLOAT", "")
    assert env.env_float("HL_FLOAT", default=1.25) == pytest.approx(1.25)


def test_env_float_invalid_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.common.env import env_float

    monkeypatch.setenv("HL_BAD_FLOAT", "not-a-float")
    assert env_float("HL_BAD_FLOAT", 3.5) == 3.5


# ---------------------------------------------------------------------------
# common.io
# ---------------------------------------------------------------------------

def test_common_atomic_writes_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.common import io

    text_path = tmp_path / "nested" / "value.txt"
    io.atomic_write_text(text_path, "hello", make_parents=True)
    assert text_path.read_text(encoding="utf-8") == "hello"

    json_path = tmp_path / "data.json"
    io.atomic_write_json(json_path, {"b": 2, "a": 1}, indent=None, trailing_newline=True)
    assert json_path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'

    def _boom(_tmp, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", _boom)
    with pytest.raises(OSError, match="replace failed"):
        io.atomic_write_text(tmp_path / "will_fail.txt", "x")
    assert not list(tmp_path.glob(".will_fail.txt.*.tmp"))


def test_common_io_bytes_and_safe_mtime_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.common import io

    path = tmp_path / "nested" / "payload.bin"
    io.atomic_write_bytes(path, b"abc", make_parents=True, fsync=True, mode=0o777)
    assert path.read_bytes() == b"abc"
    assert path.stat().st_mode & 0o777 == 0o700

    assert io.safe_mtime(tmp_path / "missing") == 0.0

    def _boom_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", _boom_replace)
    with pytest.raises(OSError, match="replace failed"):
        io.atomic_write_bytes(tmp_path / "will_fail.bin", b"x")
    assert not list(tmp_path.glob(".will_fail.bin.*.tmp"))


# ---------------------------------------------------------------------------
# common.gain_math
# ---------------------------------------------------------------------------

def test_gain_math_branches() -> None:
    from hyperloom.common import gain_math

    assert gain_math.gain_pct(0, 100.0) is None
    assert gain_math.gain_pct(120.0, 0.0) is None
    assert gain_math.gain_pct(110.0, 100.0) == pytest.approx(10.0)

    assert gain_math.gain_pct_or_zero(120.0, 0.0) == 0.0
    assert gain_math.gain_pct_or_zero(90.0, 100.0) == pytest.approx(-10.0)

    assert gain_math.incremental_gain_pct(120.0, 0.0) is None
    assert gain_math.incremental_gain_pct(110.0, 100.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# common.llm_config
# ---------------------------------------------------------------------------

def test_llm_config_parse_and_derive_edges() -> None:
    from hyperloom.common.llm_config import (
        claude_sdk_env_options,
        derive_openai_base_url,
        parse_custom_headers,
        resolve_openai_client_config,
    )

    assert parse_custom_headers(None) == {}
    assert parse_custom_headers("   ") == {}
    assert parse_custom_headers('{"X-Team": " hyperloom ", "": "drop"}') == {"X-Team": "hyperloom"}
    assert parse_custom_headers("{not json}\nX-Fallback: yes") == {"X-Fallback": "yes"}

    assert derive_openai_base_url(None) is None
    assert derive_openai_base_url("   ") is None
    assert derive_openai_base_url("https://gw.example/Unified") == "https://gw.example/Unified/v1"
    assert derive_openai_base_url("https://gw.example/custom") == "https://gw.example/custom"

    cfg = resolve_openai_client_config(
        api_key_env="CUSTOM_KEY",
        base_url_env="CUSTOM_BASE",
        env={
            "CUSTOM_KEY": " key ",
            "CUSTOM_BASE": " https://base.example/v1 ",
            "OPENAI_CUSTOM_HEADERS": '{"X-Trace": " 1 "}',
        },
    )
    assert cfg.as_kwargs() == {
        "api_key": "key",
        "base_url": "https://base.example/v1",
        "default_headers": {"X-Trace": "1"},
    }
    assert claude_sdk_env_options(env={}) == {}


# ---------------------------------------------------------------------------
# inference_optimizer.cli.credentials
# ---------------------------------------------------------------------------

def test_credentials_validate_and_reset_claude_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import credentials

    for key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        credentials._validate_credentials()
    assert exc.value.code == 2

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    credentials._validate_credentials()

    monkeypatch.setenv("HOME", str(tmp_path))
    credentials._reset_claude_config_to_upstream("sk-ant", "https://anthropic.example")
    cfg_path = tmp_path / ".claude" / "config.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["primaryApiKey"] == "sk-ant"
    assert payload["customApiUrl"] == "https://anthropic.example"
    assert oct(cfg_path.stat().st_mode & 0o777) == "0o600"
    credentials._reset_claude_config_to_upstream("ignored", "https://anthropic.example")


# ---------------------------------------------------------------------------
# inference_optimizer.cli.recover
# ---------------------------------------------------------------------------

def test_recover_session_status_and_run_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import recover
    import hyperloom.inference_optimizer.breakdown as breakdown_mod
    import hyperloom.orchestrator.trace.langfuse_emitter as emitter

    session = tmp_path / "session"
    session.mkdir()
    (session / "state.json").write_text('{"close_sequence_done": true}', encoding="utf-8")
    (session / breakdown_mod.BREAKDOWN_FILENAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        emitter,
        "read_receipt",
        lambda _session_dir: {"counts": {"breakdown_recorded": 1}, "counts_final": True},
    )
    status = recover._session_recovery_status(session)
    assert status["looks_complete"] is True
    assert status["counts_final"] is True

    assert recover._run_recover_session(argparse.Namespace(session_dir=tmp_path / "missing", force=False)) == 2
    assert recover._run_recover_session(argparse.Namespace(session_dir=session, force=False)) == 0

    calls: list[str] = []
    monkeypatch.setattr(recover, "_session_recovery_status", lambda _s: {"looks_complete": False, "close_done": False, "breakdown_exists": False, "breakdown_recorded": False, "counts_final": False})
    monkeypatch.setattr(
        breakdown_mod,
        "write_breakdown_json",
        lambda s: calls.append("write") or s / breakdown_mod.BREAKDOWN_FILENAME,
    )
    monkeypatch.setattr(breakdown_mod, "patch_breakdown_langfuse", lambda s: calls.append("patch"))
    monkeypatch.setattr(breakdown_mod, "package_session_artifacts", lambda s: calls.append("package") or s / "bundle.zip")
    monkeypatch.setattr(emitter, "flush_session", lambda s: calls.append("flush"))
    monkeypatch.setattr(emitter, "record_session_breakdown", lambda s: calls.append("record"))
    rc = recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=False))
    assert rc == 0
    assert calls == ["write", "flush", "patch", "record", "package"]

    monkeypatch.setattr(breakdown_mod, "write_breakdown_json", lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=False)) == 1


def test_recover_session_nonfatal_backfill_and_package_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import recover
    import hyperloom.inference_optimizer.breakdown as breakdown_mod
    import hyperloom.orchestrator.trace.langfuse_emitter as emitter

    session = tmp_path / "session"
    session.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        recover,
        "_session_recovery_status",
        lambda _s: {
            "looks_complete": False,
            "close_done": False,
            "breakdown_exists": False,
            "breakdown_recorded": False,
            "counts_final": False,
        },
    )
    monkeypatch.setattr(breakdown_mod, "write_breakdown_json", lambda s: calls.append("write") or s / "session_breakdown.json")
    monkeypatch.setattr(emitter, "flush_session", lambda _s: (_ for _ in ()).throw(RuntimeError("langfuse down")))
    monkeypatch.setattr(breakdown_mod, "package_session_artifacts", lambda _s: (_ for _ in ()).throw(RuntimeError("zip failed")))

    fake_backfill = SimpleNamespace(
        build_plan=lambda s: calls.append("plan") or {"session": str(s)},
        ingest=lambda plan: calls.append("ingest") or 0,
    )
    monkeypatch.setitem(__import__("sys").modules, "hyperloom.inference_optimizer.tools.backfill_langfuse", fake_backfill)

    rc = recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=True))
    assert rc == 0
    assert calls == ["write", "plan", "ingest"]


# ---------------------------------------------------------------------------
# inference_optimizer.cli.multi_node / multi_node commands
# ---------------------------------------------------------------------------

def test_cli_multi_node_gc_backend_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as mn

    root = tmp_path / "profile-traces"
    old = root / "old"
    keep = root / "keep"
    new = root / "new"
    for p in (old, keep, new):
        p.mkdir(parents=True)
    now = 10_000_000.0
    os.utime(old, (now - 10 * 86400, now - 10 * 86400))
    os.utime(keep, (now - 10 * 86400, now - 10 * 86400))
    os.utime(new, (now, now))
    monkeypatch.setattr(mn.time, "time", lambda: now)
    mn._gc_old_profile_traces(str(root), retention_days=7, keep="keep")
    assert not old.exists()
    assert keep.exists()
    assert new.exists()

    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "infera")
    assert mn._resolve_mn_backend(argparse.Namespace(mn_backend=None)) == "infera"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "bad")
    with pytest.raises(SystemExit) as exc:
        mn._resolve_mn_backend(argparse.Namespace(mn_backend=None))
    assert exc.value.code == 2

    session = tmp_path / "sess"
    ws = session / "kernel-agent-workspace" / "attempt"
    ws.mkdir(parents=True)
    patch_path = ws / "change.patch"
    patch_path.write_text("diff --git a/x b/x\n", encoding="utf-8")
    (ws / "manifest.json").write_text(
        json.dumps(
            {
                "status": "applied",
                "target_file": "/remote/x.py",
                "patch_path": str(patch_path),
                "kernel_id": "kernel-a",
                "multinode": {"backup_dir_on_pod": "/backups"},
            }
        ),
        encoding="utf-8",
    )
    (ws / "bad.json").write_text("{", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(mn, "_session_dir_resolve", lambda: session)
    monkeypatch.setattr(mn.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Completed(returncode=0))
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=2))
    assert calls and calls[0][2:4] == ["hyperloom.inference_optimizer.multi_node", "apply-patch"]


def test_provision_multi_node_rayjob_and_infera_stacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as opt_mn
    import hyperloom.inference_optimizer.multi_node.cli as mn_cli
    import hyperloom.inference_optimizer.multi_node.state_paths as state_paths
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn_env

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"last_create_request": {"image": "prior:tag"}}), encoding="utf-8")
    monkeypatch.setattr(state_paths, "resolve_state_file", lambda: state_file)
    monkeypatch.setattr(opt_mn, "mn_profile_trace_root", lambda: tmp_path / "traces")
    monkeypatch.setattr(opt_mn, "_replay_kernel_patches_for_multi_node", lambda args: None)

    created: list[argparse.Namespace] = []
    booted: list[argparse.Namespace] = []
    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: created.append(ns) or 0)
    monkeypatch.setattr(mn_cli, "cmd_bootstrap", lambda ns: booted.append(ns) or 0)
    loads = iter([
        {"rayjob_id": "wid-rj"},
        {"rayjob_id": "wid-rj", "head_pod_ip": "10.0.0.2"},
    ])
    monkeypatch.setattr(mn_cli, "_load_state", lambda: next(loads))
    monkeypatch.setattr(mn_env, "export_ray_address_to_os", lambda: os.environ.__setitem__("RAY_ADDRESS", "10.0.0.2:6379"))
    opt_mn._provision_multi_node_rayjob_stack(
        argparse.Namespace(
            nodes=2,
            mn_backend="rayjob",
            mn_image="",
            gpus_per_node=None,
            rayjob_extra_env=["A=B"],
        )
    )
    assert created[0].image == "prior:tag"
    assert booted
    assert os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"].endswith("wid-rj/torch_trace")

    infera_created: list[argparse.Namespace] = []
    monkeypatch.setattr(mn_cli, "cmd_create_infera", lambda ns: infera_created.append(ns) or 0)
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"service_url": "http://svc:8000"})
    monkeypatch.setattr(mn_cli, "install_geak_on_pods_best_effort", lambda: 0)
    opt_mn._provision_multi_node_infera_stack(
        argparse.Namespace(
            nodes=2,
            mn_image="dyn:tag",
            model="/models/m",
            gpus_per_node=4,
            rayjob_extra_env=["X=Y"],
            framework="sglang",
            pd_transfer_backend="mooncake",
            pd_mode="disaggregated",
            pd_prefill_nodes=2,
            pd_decode_nodes=1,
            pd_prefill_tp=8,
            pd_decode_tp=4,
            no_kernel=False,
        )
    )
    assert infera_created[0].image == "dyn:tag"
    assert infera_created[0].kv_transfer_backend == "mooncake"
    assert os.environ["BENCHMARK_BASE_URL"] == "http://svc:8000"


def _patch_infera_state(monkeypatch: pytest.MonkeyPatch, state: dict) -> list[dict]:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    saved: list[dict] = []
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    return saved


def test_infera_create_and_state_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    saved: list[dict] = []
    monkeypatch.setenv("SAFE_WORKSPACE", "ws")
    monkeypatch.setenv("CLAW_SESSION_ID", "sess-1")
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: saved[-1] if saved else {})
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(inf._mn_cli, "_infera_ssh_dir", lambda: tmp_path / "ssh")
    monkeypatch.setattr(inf.ssh_client, "generate_session_keypair", lambda _d: (tmp_path / "id_ed25519", "ssh-ed25519 pub"))
    monkeypatch.setattr(inf.workload_spec, "build_infera_workload_body", lambda **kw: {"body": kw})
    workload = {
        "phase": "Running",
        "pods": [
            {"podId": "app-worker-0", "resourceId": 1, "podIP": "10.0.1.0"},
            {"podId": "app-worker-1", "resourceId": 1, "podIP": "10.0.1.1"},
        ],
    }
    fake_safe = _FakeSafe(workload=workload)
    monkeypatch.setattr(inf.safe_client, "from_env", lambda: fake_safe)
    monkeypatch.setattr(inf._mn_cli, "_short_poll", lambda **kw: workload)
    monkeypatch.setattr(inf._mn_cli, "_refresh_infera_known_hosts", lambda *a, **kw: tmp_path / "known_hosts")
    monkeypatch.setattr(inf._mn_cli, "_infera_known_hosts_path", lambda state: tmp_path / "known_hosts")
    monkeypatch.setattr(inf.ssh_client, "probe_ssh", lambda *a, **kw: True)

    args = argparse.Namespace(
        extra_env=["ENV=1"],
        extra_label=["team=perf"],
        owner_id=None,
        workspace=None,
        display_name="inf",
        image="inf:tag",
        model="/models/m",
        nodes=2,
        gpus_per_node=8,
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        shared_mem_per_node=200,
        backend_framework="sglang",
        kv_transfer_backend="nixl",
        ssh_port=2222,
        pd_mode="aggregated",
        pd_prefill_nodes=0,
        pd_decode_nodes=0,
        pd_prefill_tp=0,
        pd_decode_tp=0,
        description=None,
        no_wait=False,
        recreate=False,
        poll_interval=1,
        poll_timeout=2,
    )
    assert inf.cmd_create_infera(args) == 0
    assert saved[-1]["backend"] == "infera"
    assert saved[-1]["worker_pod_ips"] == ["10.0.1.0", "10.0.1.1"]
    assert saved[-1]["ssh_known_hosts"] == str(tmp_path / "known_hosts")

    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: {"backend": "rayjob"})
    with pytest.raises(RuntimeError, match="state backend"):
        inf._infera_require_state()
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: {"backend": "infera", "worker_pod_ips": ["10.0.1.0"]})
    with pytest.raises(RuntimeError, match="ssh_key_path"):
        inf._infera_require_state()


def test_infera_forward_env_and_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setenv("MORI_FOO", "1")
    monkeypatch.setenv("SGLANG_MORI_BAR", "2")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", "/shared/traces")
    monkeypatch.setenv(
        "HYPERLOOM_MN_EXTRA_FWD_ENV",
        json.dumps({"SGLANG_USE_AITER": "1", "MORI_FOO": "override", "SGLANG_MORI_BAR": "explicit"}),
    )
    monkeypatch.setenv("HYPERLOOM_MN_UNSET_FWD_ENV", json.dumps(["SGLANG_MORI_BAR"]))
    for k in ("SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.setenv(k, f"secret-{k}")
    fwd = inf._collect_forward_env()
    assert fwd["MORI_FOO"] == "override"
    assert fwd["SGLANG_TORCH_PROFILER_DIR"] == "/shared/traces"
    assert fwd["SGLANG_USE_AITER"] == "1"
    assert fwd["SGLANG_MORI_BAR"] == "explicit"
    for k in ("SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        assert k not in fwd

    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", "{bad")
    assert inf._collect_forward_env()["MORI_FOO"] == "1"

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: f"script:{name}")

    def _run(state, ip, script, python, launch_args, **kw):
        calls.append((ip, launch_args))
        if ip == "10.0.0.2":
            raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=kw["timeout"])
        return _Completed(returncode=1 if ip == "10.0.0.3" else 0, stdout='noise {"status":"ok"}\n', stderr="bad")

    monkeypatch.setattr(inf._mn_cli, "_infera_ssh_run_script", _run)
    targets = [
        {"podIP": "10.0.0.1", "sshPort": 2222},
        {"podIP": "10.0.0.2", "sshPort": 2222},
        {"podIP": "10.0.0.3", "sshPort": 2222},
    ]
    rc, results = inf._infera_fanout_launch(
        {"ssh_port": 2222},
        "--model /m",
        targets,
        label="restart",
        poll_timeout=5,
        print_logs=True,
    )
    assert rc == 1
    assert [r["podIP"] for r in results] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert results[1]["rc"] == 124
    assert calls[0] == ("10.0.0.1", "--model /m")


def _restart_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        framework="",
        model="/models/m",
        tp=8,
        ep=1,
        extra_args="--mem-fraction-static 0.7",
        pd_mode="",
        pd_transfer_backend="nixl",
        pd_prefill_nodes=0,
        pd_decode_nodes=0,
        pd_prefill_tp=0,
        pd_decode_tp=0,
        pd_prefill_ep=0,
        pd_decode_ep=0,
        pd_prefill_extra_args="",
        pd_decode_extra_args="",
        poll_timeout=10,
        print_logs=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_infera_restart_and_kill_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    saved: list[dict] = []
    state = {
        "backend": "infera",
        "framework": "sglang",
        "nodes": 2,
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    build_calls: list[dict] = []
    monkeypatch.setattr(
        inf.infera_support,
        "build_node_launch_args",
        lambda **kw: build_calls.append(kw) or "launch-args",
    )
    fanout_calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        inf,
        "_infera_fanout_launch",
        lambda st, args, targets, **kw: fanout_calls.append((kw["label"], [t["podIP"] for t in targets])) or (0, [{"ok": True}]),
    )
    assert inf._infera_restart_server(_restart_args()) == 0
    assert build_calls[-1]["nnodes"] == 2
    assert fanout_calls[-1] == ("restart", ["10.0.1.0", "10.0.1.1"])
    assert saved[-1]["last_restart_pd_mode"] == "aggregated"

    pd_state = {
        **state,
        "pd_mode": "disaggregated",
        "prefill_pod_ips": ["10.0.2.0"],
        "decode_pod_ips": ["10.0.3.0"],
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(pd_state))
    assert inf._infera_restart_server(
        _restart_args(
            pd_prefill_nodes=1,
            pd_decode_nodes=1,
            pd_prefill_tp=8,
            pd_decode_tp=4,
            pd_prefill_extra_args="--prefill",
            pd_decode_extra_args="--decode",
        )
    ) == 0
    assert [call[0] for call in fanout_calls[-2:]] == ["restart-prefill", "restart-decode"]
    assert saved[-1]["last_restart_pd_decode_tp"] == 4
    assert saved[-1]["last_restart_pd_prefill_nodes"] == 1
    assert saved[-1]["last_restart_pd_decode_nodes"] == 1

    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(pd_state))
    assert inf._infera_restart_server(_restart_args()) == 0
    assert saved[-1]["pd_prefill_nodes"] == 1
    assert saved[-1]["pd_decode_nodes"] == 1
    assert saved[-1]["last_restart_pd_prefill_nodes"] == 1
    assert saved[-1]["last_restart_pd_decode_nodes"] == 1

    pd_state["framework"] = "vllm"
    with pytest.raises(RuntimeError, match="sglang-only"):
        inf._infera_restart_server(_restart_args())

    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state | {"last_restart_framework": "vllm"}))
    assert inf._infera_kill_inference(argparse.Namespace(poll_timeout=10, print_logs=False)) == 0
    assert build_calls[-1]["kill_only"] is True
    assert saved[-1]["last_kill_results"] == [{"ok": True}]


def test_infera_node_ops_apply_revert_and_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_read_bundled_pod_python_script", lambda name: f"script:{name}")
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: _Completed(returncode=0, stdout='logs {"status":"ok","backup_path":"/b"}', stderr=""),
    )
    target = {"podIP": "10.0.1.0", "sshPort": 2222}
    parsed, tx = inf._infera_ssh_node_op(state, target, "apply", timeout=5)
    assert parsed and parsed["status"] == "ok"
    assert tx["rc"] == 0

    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)),
    )
    parsed, tx = inf._infera_ssh_node_op(state, target, "apply", timeout=5)
    assert parsed is None and tx["rc"] == 124

    responses = iter(
        [
            ({"status": "ok", "backup_path": "/b0"}, {"rc": 0, "stderr": ""}),
            ({"status": "failed", "error": "nope"}, {"rc": 1, "stderr": "bad"}),
        ]
    )
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: next(responses))
    patch_file = tmp_path / "p.diff"
    patch_file.write_text("diff", encoding="utf-8")
    rc = inf._infera_apply_patch(
        argparse.Namespace(
            patch_file=str(patch_file),
            target_path="/remote/x.py",
            backup_dir="/backups",
            kernel_id="k1",
            timeout_sec=10,
        )
    )
    assert rc == 1

    missing_rc = inf._infera_apply_patch(
        argparse.Namespace(
            patch_file=str(tmp_path / "missing.diff"),
            target_path="/remote/x.py",
            backup_dir="/backups",
            kernel_id="k1",
            timeout_sec=10,
        )
    )
    assert missing_rc == inf.EXIT_CONFIG_ERROR

    assert inf._infera_revert_patch(argparse.Namespace(backup_map_json="{", target_path="/x", timeout_sec=1)) == inf.EXIT_CONFIG_ERROR
    assert inf._infera_revert_patch(argparse.Namespace(backup_map_json="{}", target_path="/x", timeout_sec=1)) == inf.EXIT_CONFIG_ERROR
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: ({"status": "restored"}, {"rc": 0, "stderr": ""}))
    assert inf._infera_revert_patch(
        argparse.Namespace(backup_map_json=json.dumps({"10.0.1.0": "/b"}), target_path="/x", timeout_sec=1)
    ) == 0

    assert inf._infera_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{bad",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=False,
        )
    ) == inf.EXIT_CONFIG_ERROR
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: (None, {"rc": 1, "stderr": "no json"}))
    assert inf._infera_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{}",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=True,
        )
    ) == inf.EXIT_TRANSIENT
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: ({"status": "ok", "result": 1}, {"rc": 0, "stderr": ""}))
    assert inf._infera_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{}",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=False,
        )
    ) == 0


def test_infera_tracelens_and_geak_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    saved = _patch_infera_state(monkeypatch, state)
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 5)

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda st, ip, script, python, op_args, **kw: calls.append((ip, python, op_args))
        or _Completed(returncode=0, stdout='{"status":"applied","per_pod":[{"status":"applied"}]}', stderr=""),
    )
    assert inf._infera_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tracelens", sglang_version_pin="v1", poll_timeout=5)
    ) == 0
    assert calls[0][1] == "/opt/venv/bin/python"
    assert "--sglang-version-pin v1" in calls[0][2]

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert inf.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=False)) == inf.EXIT_CONFIG_ERROR
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: _Completed(returncode=0, stdout='{"status":"installed"}', stderr=""),
    )
    assert inf.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=True)) == 0
    assert saved == []


def test_rayjob_create_reuse_and_failure_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.rayjob as rayjob

    saved: list[dict] = []
    monkeypatch.setenv("SAFE_WORKSPACE", "ws")
    monkeypatch.setenv("CLAW_SESSION_ID", "sess-rj")
    monkeypatch.setattr(rayjob._mn_cli(), "_load_state", lambda: saved[-1] if saved else {})
    monkeypatch.setattr(rayjob._mn_cli(), "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(rayjob.workload_spec, "build_rayjob_workload_body", lambda **kw: {"body": kw})
    monkeypatch.setattr(rayjob, "_write_rayjob_meta", lambda **kw: None)
    workload = {
        "phase": "Running",
        "pods": [
            {"podId": "submitter", "resourceId": 0, "podIP": "10.0.0.9"},
            {"podId": "cluster-head-abc", "resourceId": 1, "podIP": "10.0.0.1"},
        ],
    }
    monkeypatch.setattr(rayjob._mn_cli(), "_short_poll", lambda **kw: workload)
    monkeypatch.setattr(rayjob.safe_client, "from_env", lambda: _FakeSafe(workload=workload))

    args = argparse.Namespace(
        extra_env=["A=B"],
        extra_label=["team=perf"],
        owner_id=None,
        workspace=None,
        display_name="rj",
        image="ray:tag",
        nodes=2,
        gpus_per_node=8,
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        description=None,
        no_wait=False,
        recreate=False,
        poll_interval=1,
        poll_timeout=2,
    )
    assert rayjob.cmd_create_rayjob(args) == 0
    assert saved[-1]["backend"] == "rayjob"
    assert saved[-1]["head_pod_ip"] == "10.0.0.1"
    assert saved[-1]["ray_address"] == "10.0.0.1:6379"

    diag, snapshot = rayjob._summarize_workload_failure(
        {
            "workloadId": "wid",
            "phase": "Failed",
            "message": "image pull",
            "dispatchCount": 2,
            "queuePosition": 5,
            "pods": [
                {"podId": "p0", "phase": "Failed", "resourceId": 0, "adminNodeName": "n", "podIP": "", "failedMessage": "boom"},
                "not-a-pod",
            ],
        }
    )
    assert "failed_pods" in diag
    assert snapshot["pods"][0]["failedMessage"] == "boom"

    err = rayjob.safe_client.SafeApiError(404, "missing", endpoint="GET /api/v1/workloads/wid")
    assert rayjob._is_safe_get_workload_404(err) is True
    assert rayjob._find_head_pod_ip({"pods": [{"resourceId": 0, "podIP": "10.0.0.2"}]}) == "10.0.0.2"
    assert rayjob.ray_gcs_address("") == ""


def test_infera_process_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    assert inf._infera_all_gpu_ips({"pd_mode": "disaggregated", "prefill_pod_ips": ["p"], "decode_pod_ips": ["d"]}) == ["p", "d"]
    assert inf._infera_all_gpu_ips({"worker_pod_ips": ["w"]}) == ["w"]

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0"],
        "ssh_key_path": "/tmp/k",
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    assert inf._infera_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="", sglang_version_pin="", poll_timeout=1)
    ) == inf.EXIT_CONFIG_ERROR

    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 1)
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=1)),
    )
    assert inf._infera_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", poll_timeout=1)
    ) == 1


def test_rayjob_create_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.rayjob as rayjob

    monkeypatch.delenv("SAFE_WORKSPACE", raising=False)
    monkeypatch.setattr(rayjob.safe_client, "from_env", lambda: _FakeSafe())
    monkeypatch.setattr(rayjob._mn_cli(), "_load_state", lambda: {})
    monkeypatch.setattr(rayjob._mn_cli(), "_save_state", lambda payload: None)
    missing_ws = argparse.Namespace(
        extra_env=[], extra_label=[], owner_id=None, workspace=None,
        display_name="rj", image="ray:tag", nodes=2, gpus_per_node=8,
    )
    with pytest.raises(RuntimeError, match="workspace is required"):
        rayjob.cmd_create_rayjob(missing_ws)

    saved: list[dict] = []
    monkeypatch.setenv("SAFE_WORKSPACE", "ws")
    monkeypatch.setenv("WORKLOAD_ID", "owner-1")
    monkeypatch.setattr(rayjob._mn_cli(), "_load_state", lambda: (saved[-1] if saved else {"rayjob_id": "wid-prior"}))
    monkeypatch.setattr(rayjob._mn_cli(), "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(rayjob, "_write_rayjob_meta", lambda **kw: None)

    class _ReuseSafe(_FakeSafe):
        def create_workload(self, body: dict) -> str:  # pragma: no cover - must not be called
            raise AssertionError("create_workload should not run when reusing")

    reuse_safe = _ReuseSafe(workload={"phase": "Running", "pods": []})
    monkeypatch.setattr(rayjob.safe_client, "from_env", lambda: reuse_safe)
    args = argparse.Namespace(
        extra_env=[], extra_label=[], owner_id=None, workspace=None,
        display_name="rj", image="ray:tag", nodes=2, gpus_per_node=8,
        cpus_per_node=96, mem_per_node=1024, ephemeral_per_node=400,
        description=None, no_wait=True, recreate=False, poll_interval=1, poll_timeout=2,
    )
    assert rayjob.cmd_create_rayjob(args) == 0
    assert saved[-1]["head_pod_ip"] == ""
    assert saved[-1]["ray_address"] == ""

    assert rayjob._find_head_pod_ip({"pods": []}) == ""
    assert rayjob._find_head_pod_ip({"pods": [{"podId": "x-head-1", "podIP": "1.1.1.1"}]}) == "1.1.1.1"
    assert rayjob._find_head_pod_ip({"pods": [{"podIP": "2.2.2.2"}]}) == "2.2.2.2"
    assert rayjob.ray_gcs_address("3.3.3.3") == "3.3.3.3:6379"

    diag, snap = rayjob._summarize_workload_failure(
        {"phase": "Pending", "queuePosition": 4, "pods": [{"podId": "p", "phase": "Pending", "podIP": "9.9.9.9"}]}
    )
    assert "queuePosition=4" in diag
    assert "pods=1" in diag
    assert snap["phase"] == "Pending"


def test_infera_restart_config_and_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    assert inf._infera_restart_config_matches({}, argparse.Namespace(), "sglang", "aggregated") is False

    agg_state = {
        "last_restart_framework": "sglang",
        "last_restart_model": "/m",
        "last_restart_tp": 8,
        "last_restart_ep": 8,
        "last_restart_pd_mode": "aggregated",
        "last_restart_extra_args": "--foo 1",
    }
    agg_args = argparse.Namespace(model="/m", tp=8, ep=8, extra_args="--foo 1")
    assert inf._infera_restart_config_matches(agg_state, agg_args, "sglang", "aggregated") is True
    assert inf._infera_restart_config_matches(
        agg_state, argparse.Namespace(model="/m", tp=4, ep=8, extra_args="--foo 1"), "sglang", "aggregated"
    ) is False

    pd_state = {
        "last_restart_framework": "sglang",
        "last_restart_model": "/m",
        "last_restart_tp": 8,
        "last_restart_ep": 8,
        "last_restart_pd_mode": "disaggregated",
        "last_restart_extra_args": "",
        "last_restart_pd_prefill_nodes": 1,
        "last_restart_pd_decode_nodes": 1,
        "prefill_pod_ips": ["10.0.0.1"],
        "decode_pod_ips": ["10.0.0.2"],
    }
    pd_args = argparse.Namespace(
        model="/m", tp=8, ep=8, extra_args="",
        pd_prefill_nodes=1, pd_decode_nodes=1,
        pd_prefill_tp=0, pd_decode_tp=0, pd_prefill_ep=0, pd_decode_ep=0,
        pd_prefill_extra_args="", pd_decode_extra_args="",
    )
    assert inf._infera_restart_config_matches(pd_state, pd_args, "sglang", "disaggregated") is True

    assert inf._infera_servers_alive({}, [], timeout=5) is False

    state = {"ssh_key_path": "/tmp/k"}
    targets = [{"podIP": "10.0.0.1", "sshPort": 2222}]
    monkeypatch.setattr(inf._mn_cli, "_infera_default_ssh_port", lambda st: 2222)
    monkeypatch.setattr(
        inf._mn_cli, "_infera_ssh_bash_with_env",
        lambda *a, **kw: _Completed(returncode=0, stdout="MN_ALIVE\n"),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is True

    monkeypatch.setattr(
        inf._mn_cli, "_infera_ssh_bash_with_env",
        lambda *a, **kw: _Completed(returncode=0, stdout="dead"),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is False

    monkeypatch.setattr(
        inf._mn_cli, "_infera_ssh_bash_with_env",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=1)),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is False

    assert inf._infera_servers_alive(state, [{"podIP": ""}], timeout=5) is False


def test_infera_restart_resume_fast_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0"],
        "ssh_key_path": "/tmp/k",
        "last_restart_framework": "sglang",
        "last_restart_model": "/m",
        "last_restart_tp": 8,
        "last_restart_ep": 8,
        "last_restart_pd_mode": "aggregated",
        "last_restart_extra_args": "",
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 20)
    monkeypatch.setattr(inf, "_infera_all_gpu_targets", lambda st: [{"podIP": "10.0.1.0", "sshPort": 2222}])
    monkeypatch.setattr(inf, "_infera_servers_alive", lambda st, targets, timeout: True)
    monkeypatch.setenv("MULTI_NODE_RESTART_RESUME_RUNNING", "1")

    args = argparse.Namespace(
        framework="sglang", model="/m", tp=8, ep=8, extra_args="",
        pd_mode="", pd_transfer_backend="", print_logs=False,
        pd_prefill_extra_args="", pd_decode_extra_args="",
    )
    assert inf._infera_restart_server(args) == 0
    out = capsys.readouterr().out
    assert '"resumed": true' in out


def test_infera_install_timeout_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: dict(state))
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 5)

    geak_calls = {"n": 0}

    def _geak_run(*_args, **_kwargs):
        geak_calls["n"] += 1
        if geak_calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)
        return _Completed(returncode=0, stdout='{"status":"failed","reason":"pip"}', stderr="")

    monkeypatch.setattr(inf._mn_cli, "_infera_ssh_run_script", _geak_run)
    assert inf.cmd_install_geak(argparse.Namespace(geak_src="/geak", poll_timeout=5, print_logs=False)) == 1


def test_infera_create_reuse_and_restart_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    def _args(**overrides):
        defaults = dict(
            extra_env=[],
            extra_label=[],
            owner_id=None,
            workspace="ws",
            display_name="inf",
            image="inf:tag",
            model="/models/m",
            nodes=2,
            gpus_per_node=8,
            cpus_per_node=96,
            mem_per_node=1024,
            ephemeral_per_node=400,
            shared_mem_per_node=200,
            backend_framework="sglang",
            kv_transfer_backend="nixl",
            ssh_port=2222,
            pd_mode="aggregated",
            pd_prefill_nodes=0,
            pd_decode_nodes=0,
            pd_prefill_tp=0,
            pd_decode_tp=0,
            description=None,
            no_wait=True,
            recreate=False,
            poll_interval=1,
            poll_timeout=2,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    saved: list[dict] = []
    state = {"backend": "infera", "rayjob_id": "wid-old", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k"}
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: saved[-1] if saved else dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(inf._mn_cli, "_infera_ssh_dir", lambda: tmp_path / "ssh")
    monkeypatch.setattr(inf.ssh_client, "generate_session_keypair", lambda _d: (tmp_path / "id_ed25519", "ssh-ed25519 pub"))
    monkeypatch.setattr(inf.workload_spec, "build_infera_workload_body", lambda **kw: {"body": kw})
    monkeypatch.setattr(inf.ssh_client, "probe_ssh", lambda *a, **kw: False)
    monkeypatch.setattr(inf._mn_cli, "_refresh_infera_known_hosts", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("keyscan")))

    class _ReuseSafe(_FakeSafe):
        def delete_workload(self, wid: str) -> None:
            return None

        def get_workload(self, wid: str) -> dict:
            return {"phase": "Running", "pods": []}

        def get_workload_service(self, wid: str) -> dict:
            raise inf.safe_client.SafeApiError(500, "svc down", endpoint="GET /service")

    monkeypatch.setattr(inf.safe_client, "from_env", lambda: _ReuseSafe())
    assert inf.cmd_create_infera(_args()) == 0
    assert saved[-1]["rayjob_id"] == "wid-old"
    assert saved[-1]["worker_pod_ips"] == []

    class _GoneSafe(_FakeSafe):
        def delete_workload(self, wid: str) -> None:
            return None

        def get_workload(self, wid: str) -> dict:
            raise inf.safe_client.SafeApiError(404, "gone", endpoint="GET /api/v1/workloads/wid-old")

    saved.clear()
    monkeypatch.setattr(inf.safe_client, "from_env", lambda: _GoneSafe(workload={"phase": "Running", "pods": []}))
    assert inf.cmd_create_infera(_args()) == 0
    assert saved[-1]["rayjob_id"] == "wid-test"

    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: {"backend": "infera", "worker_pod_ips": [], "ssh_key_path": "/tmp/k"})
    with pytest.raises(RuntimeError, match="no GPU pod IPs"):
        inf._infera_require_state()

    monkeypatch.setattr(inf, "_infera_require_state", lambda: {"backend": "infera", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k", "framework": "bad"})
    with pytest.raises(RuntimeError, match="unsupported framework"):
        inf._infera_restart_server(_restart_args())

    monkeypatch.setattr(inf, "_infera_require_state", lambda: {"backend": "infera", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k", "framework": "sglang"})
    monkeypatch.setattr(inf, "validate_server_args", lambda *a, **kw: (_ for _ in ()).throw(inf.ServerArgsRejected("denied")))
    assert inf._infera_restart_server(_restart_args(extra_args="--bad")) == inf.EXIT_CONFIG_ERROR

    monkeypatch.setattr(inf, "_infera_require_state", lambda: {"backend": "infera", "worker_pod_ips": [], "ssh_key_path": "/tmp/k"})
    assert inf._infera_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", poll_timeout=1)
    ) == inf.EXIT_CONFIG_ERROR

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.setenv("HYPERLOOM_ROOT", "/root/hyperloom")
    assert inf._resolve_geak_src(None) == "/root/hyperloom/geak"


def test_cli_multi_node_remaining_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as opt_mn
    import hyperloom.inference_optimizer.multi_node.cli as mn_cli
    import hyperloom.inference_optimizer.multi_node.state_paths as state_paths
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn_env

    root = tmp_path / "gc"
    root.mkdir()
    (root / "not-a-dir").write_text("x", encoding="utf-8")
    old = root / "old"
    old.mkdir()
    monkeypatch.setattr(opt_mn.time, "time", lambda: 10_000_000.0)
    os.utime(old, (10_000_000.0 - 10 * 86400, 10_000_000.0 - 10 * 86400))
    monkeypatch.setattr(opt_mn.shutil, "rmtree", lambda _p: (_ for _ in ()).throw(OSError("rm failed")))
    opt_mn._gc_old_profile_traces(str(root), retention_days=7)

    real_iterdir = Path.iterdir

    def _bad_iterdir(self):
        if self == root:
            raise OSError("scan failed")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _bad_iterdir)
    opt_mn._gc_old_profile_traces(str(root), retention_days=7)
    monkeypatch.setattr(Path, "iterdir", real_iterdir)

    state_file = tmp_path / "state.json"
    monkeypatch.setattr(state_paths, "resolve_state_file", lambda: state_file)
    state_file.write_text(json.dumps({"last_create_request": {"image": "dyn:old"}}), encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "bad")
    monkeypatch.setattr(mn_cli, "cmd_create_infera", lambda ns: 6)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_infera_stack(
            argparse.Namespace(
                nodes=2,
                mn_image="",
                model="/models/m",
                gpus_per_node=None,
                rayjob_extra_env=[],
                framework="sglang",
                pd_transfer_backend="",
                pd_mode="aggregated",
                pd_prefill_nodes=0,
                pd_decode_nodes=0,
                pd_prefill_tp=0,
                pd_decode_tp=0,
                no_kernel=True,
            )
        )
    assert exc.value.code == 6

    state_file.write_text("{bad", encoding="utf-8")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", raising=False)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(
            argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image="", gpus_per_node=None, rayjob_extra_env=[])
        )
    assert exc.value.code == 2

    loads = iter([
        {"last_bootstrap_submission_id": "boot", "rayjob_id": "wid"},
        {"last_bootstrap_submission_id": "boot", "rayjob_id": "wid"},
    ])
    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: 0)
    monkeypatch.setattr(mn_cli, "_load_state", lambda: next(loads))
    monkeypatch.setattr(mn_env, "export_ray_address_to_os", lambda: None)
    monkeypatch.setattr(opt_mn, "_replay_kernel_patches_for_multi_node", lambda args: None)

    class _BadTracePath:
        def __init__(self, value: Path) -> None:
            self.value = value

        def __truediv__(self, child: str) -> "_BadTracePath":
            return _BadTracePath(self.value / child)

        def __str__(self) -> str:
            return str(self.value)

        def mkdir(self, *args, **kwargs):
            raise OSError("readonly")

    trace_root = _BadTracePath(tmp_path / "traces")
    monkeypatch.setattr(opt_mn, "mn_profile_trace_root", lambda: trace_root)
    opt_mn._provision_multi_node_rayjob_stack(
        argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image="ray:tag", gpus_per_node=8, rayjob_extra_env=[])
    )


def test_cli_multi_node_error_and_early_return_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as opt_mn
    import hyperloom.inference_optimizer.multi_node.cli as mn_cli
    import hyperloom.inference_optimizer.multi_node.state_paths as state_paths

    opt_mn._gc_old_profile_traces(str(tmp_path / "missing"))
    monkeypatch.setenv("HYPERLOOM_MN_TRACE_GC_DISABLE", "true")
    opt_mn._gc_old_profile_traces(str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_MN_TRACE_GC_DISABLE", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_TRACE_RETENTION_DAYS", "bad")
    opt_mn._gc_old_profile_traces(str(tmp_path))

    args = argparse.Namespace(nodes=1, mn_backend="rayjob")
    assert opt_mn._provision_multi_node_rayjob_stack(args) is None

    called: list[str] = []
    with monkeypatch.context() as mp:
        mp.setattr(opt_mn, "_provision_multi_node_infera_stack", lambda _args: called.append("infera"))
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="infera"))
    assert called == ["infera"]

    state_file = tmp_path / "missing_state.json"
    monkeypatch.setattr(state_paths, "resolve_state_file", lambda: state_file)
    for key in ("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", "INFERENCE_OPTIMIZER_GPUS_PER_NODE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image="", gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 2

    state_file.write_text(json.dumps({"last_create_request": {"image": "img:old"}}), encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "bad")
    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: 7)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image="", gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 7

    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: 0)
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"rayjob_id": "wid"})
    monkeypatch.setattr(mn_cli, "cmd_bootstrap", lambda ns: 9)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", mn_image="img:new", gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 9

    state_file.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_infera_stack(
            argparse.Namespace(
                nodes=2,
                mn_image="",
                model="/models/m",
                gpus_per_node=None,
                rayjob_extra_env=[],
                framework="sglang",
                pd_transfer_backend="",
                pd_mode="aggregated",
                pd_prefill_nodes=0,
                pd_decode_nodes=0,
                pd_prefill_tp=0,
                pd_decode_tp=0,
                no_kernel=True,
            )
        )
    assert exc.value.code == 2

    state_file.write_text("{bad", encoding="utf-8")
    with pytest.raises(SystemExit):
        opt_mn._provision_multi_node_infera_stack(
            argparse.Namespace(
                nodes=2,
                mn_image="",
                model="/models/m",
                gpus_per_node=None,
                rayjob_extra_env=[],
                framework="sglang",
                pd_transfer_backend="",
                pd_mode="aggregated",
                pd_prefill_nodes=0,
                pd_decode_nodes=0,
                pd_prefill_tp=0,
                pd_decode_tp=0,
                no_kernel=True,
            )
        )


def test_multi_node_patch_replay_skip_and_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as mn

    session = tmp_path / "sess"
    monkeypatch.setattr(mn, "_session_dir_resolve", lambda: session)
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=1))
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=2))

    ws = session / "kernel-agent-workspace"
    (ws / "empty").mkdir(parents=True)
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=2))

    manifests = ws / "attempt"
    manifests.mkdir()
    (manifests / "manifest.json").write_text("{bad", encoding="utf-8")
    (manifests / "skip_status" / "manifest.json").parent.mkdir()
    (manifests / "skip_status" / "manifest.json").write_text(json.dumps({"status": "pending"}), encoding="utf-8")
    (manifests / "skip_mn" / "manifest.json").parent.mkdir()
    (manifests / "skip_mn" / "manifest.json").write_text(json.dumps({"status": "applied"}), encoding="utf-8")
    (manifests / "skip_fields" / "manifest.json").parent.mkdir()
    (manifests / "skip_fields" / "manifest.json").write_text(json.dumps({"status": "applied", "multinode": {"backup_dir_on_pod": "/b"}}), encoding="utf-8")
    (manifests / "missing_patch" / "manifest.json").parent.mkdir()
    (manifests / "missing_patch" / "manifest.json").write_text(
        json.dumps({"status": "applied", "multinode": {"backup_dir_on_pod": "/b"}, "target_file": "/x", "patch_path": str(tmp_path / "missing.diff")}),
        encoding="utf-8",
    )
    patch = tmp_path / "p.diff"
    patch.write_text("diff", encoding="utf-8")
    (manifests / "failed" / "manifest.json").parent.mkdir()
    (manifests / "failed" / "manifest.json").write_text(
        json.dumps({"status": "applied", "multinode": {"backup_dir_on_pod": "/b"}, "target_file": "/x", "patch_path": str(patch), "kernel_id": "k"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mn.subprocess, "run", lambda *a, **kw: _Completed(returncode=5, stderr="failed patch"))
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=2))


# ---------------------------------------------------------------------------
# agents.framework helpers
# ---------------------------------------------------------------------------

def test_framework_audit_common_patch_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.agents.framework import _audit_common as common

    diff = (
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/pkg/a.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+class Added:\n"
        "+    def run(self): return 1\n"
        " context line\n"
        "diff --git a/pkg/old.py b/pkg/old.py\n"
        "deleted file mode 100644\n"
        "--- a/pkg/old.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-def gone(): pass\n"
    )
    changes = common.parse_unified_diff(diff)
    assert [c.path for c in changes] == ["pkg/a.py"]
    assert changes[0].is_new is True
    assert common._symbols(changes[0].added) == ["Added", "run"]

    root = tmp_path / "src"
    target = root / "a.py"
    target.parent.mkdir()
    target.write_text("class Added:\n    pass\n", encoding="utf-8")
    assert common._resolve_local_file("pkg/a.py", [root]) == target

    patch_file = tmp_path / "patch.diff"
    patch_file.write_text(diff, encoding="utf-8")
    text, source = common._obtain_patch_text({"patches_path": str(patch_file)}, tmp_path)
    assert text == diff and source == "patches_path"

    text, source = common._obtain_patch_text({"diff_url": f"file://{patch_file}"}, tmp_path)
    assert text == diff and source == "diff_url"
    assert common._fetch_diff_url(f"file://{tmp_path / 'missing.diff'}", tmp_path) == ""

    monkeypatch.setenv("PR_KB_ENABLE", "0")
    assert common._obtain_patch_text({"diff_text": " inline diff "}, tmp_path) == (" inline diff ", "inline")


def test_framework_static_audit_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.agents.framework import audit

    root = tmp_path / "framework"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    local = pkg / "model.py"
    local.write_text(
        "def existing():\n"
        "    return 'old'\n"
        "def already_here():\n"
        "    return 'new'\n",
        encoding="utf-8",
    )

    direct_diff = (
        "diff --git a/pkg/model.py b/pkg/model.py\n"
        "--- a/pkg/model.py\n"
        "+++ b/pkg/model.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def existing():\n"
        "     return 'old'\n"
        "+def absent_new():\n"
        "+    return 'fast'\n"
    )
    result = audit.run_phase_audit(
        {
            "candidate": {"candidate_id": "direct"},
            "framework": "sglang",
            "framework_source_roots": [str(root)],
            "diff_text": direct_diff,
            "work_dir": str(tmp_path / "audit-direct"),
        }
    )
    assert result["semantic_status"] == "not_present"
    assert result["applicability"] == "direct_apply"
    assert result["metrics"]["patch_source"] == "inline"
    assert (tmp_path / "audit-direct" / "semantic_audit.json").is_file()

    already_diff = (
        "diff --git a/pkg/model.py b/pkg/model.py\n"
        "--- a/pkg/model.py\n"
        "+++ b/pkg/model.py\n"
        "@@ -1 +1,3 @@\n"
        "+def already_here():\n"
        "+    return 'new'\n"
    )
    already = audit.run_phase_audit(
        {
            "candidate": {"candidate_id": "already"},
            "framework": "sglang",
            "framework_source_roots": [str(root)],
            "diff_text": already_diff,
            "work_dir": str(tmp_path / "audit-already"),
        }
    )
    assert already["semantic_status"] == "already_equivalent"
    assert already["recommended_next_step"] == "skip"

    unknown = audit.run_phase_audit(
        {
            "candidate": {"candidate_id": "unknown"},
            "framework": "sglang",
            "framework_source_roots": [],
            "diff_text": direct_diff,
            "work_dir": str(tmp_path / "audit-unknown"),
        }
    )
    assert unknown["semantic_status"] == "unknown"
    assert "no framework_source_roots" in unknown["risks"][0]

    no_patch = audit.run_phase_audit(
        {
            "candidate": {"candidate_id": "no-patch"},
            "framework": "sglang",
            "framework_source_roots": [str(root)],
            "work_dir": str(tmp_path / "audit-no-patch"),
        }
    )
    assert no_patch["confidence"] == 0.0
    assert "no patch material" in no_patch["risks"][0]

    monkeypatch.setattr(audit, "_obtain_patch_text", lambda req, wd: ("diff --git a/x b/x\n", "stub"))
    zero_changes = audit.run_phase_audit(
        {
            "candidate": {"candidate_id": "zero"},
            "framework": "sglang",
            "framework_source_roots": [str(root)],
            "work_dir": str(tmp_path / "audit-zero"),
        }
    )
    assert zero_changes["metrics"]["patch_source"] == "stub"
    assert "zero file changes" in zero_changes["risks"][0]


def test_framework_audit_llm_refine_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.agents.framework import audit

    static = {
        "candidate_id": "c1",
        "semantic_status": "not_present",
        "applicability": "direct_apply",
        "confidence": 0.5,
        "evidence": [{"reason": "context present"}],
        "risks": [],
        "recommended_next_step": "direct_framework",
        "layer": "static",
        "metrics": {},
    }
    monkeypatch.delenv("SAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    out = audit._maybe_llm_refine({}, dict(static), "diff")
    assert out["layer"] == "static"
    assert "missing SAFE_API_KEY" in out["risks"][-1]

    class _Message:
        content = '{"semantic_status":"partially_present","applicability":"needs_rewrite","confidence":0.77,"recommended_next_step":"author_via_specialist","note":"drift"}'

    class _Choice:
        message = _Message()

    class _Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(choices=[_Choice()])

    class _OpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    refined = audit._maybe_llm_refine(
        {"api_key": "sk", "openai_base_url": "https://llm.example/v1", "model": "m"},
        dict(static),
        "diff",
    )
    assert refined["layer"] == "llm"
    assert refined["semantic_status"] == "partially_present"
    assert refined["confidence"] == pytest.approx(0.77)
    assert refined["risks"][-1] == "llm: drift"

    assert audit._parse_llm_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert audit._parse_llm_json("no json") is None


def test_framework_isolation_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.agents.framework import isolation
    from hyperloom.agents.framework.models import Baseline, Candidate, ExploreRequest

    req = ExploreRequest(
        framework="sglang",
        repo_url="https://github.com/sgl-project/sglang.git",
        work_dir=tmp_path,
        baseline=Baseline(throughput=100.0),
    )
    candidate = Candidate(ref="PR:42", repo=req.repo_url, head_sha="")
    assert isolation._repo_cache_dir(req).name == "https---github-com-sgl-project-sglang-git"
    assert isolation._worktree_ref(candidate) == "refs/pull/42/head"
    assert isolation._worktree_ref(Candidate(ref="main", repo=req.repo_url, head_sha="abc123")) == "abc123"

    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "bad")
    assert isolation._resolve_min_free_gb(None) == pytest.approx(20.0)
    assert isolation._resolve_min_free_gb(3.5) == pytest.approx(3.5)

    usage = SimpleNamespace(free=2 * 1024**3)
    monkeypatch.setattr(isolation.shutil, "disk_usage", lambda _p: usage)
    isolation.disk_preflight(tmp_path / "ok", n_candidates=1, min_free_gb=1.0, per_candidate_gb=0.5)
    with pytest.raises(isolation.DiskPreflightError, match="insufficient disk"):
        isolation.disk_preflight(tmp_path / "bad", n_candidates=3, min_free_gb=1.0, per_candidate_gb=1.0)

    git_calls: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(isolation, "_run_git", lambda args, cwd=None, timeout_sec=1800: git_calls.append((args, cwd)))
    repo_dir = isolation.prepare_repo_cache(req)
    assert git_calls[-1][0][:3] == ["git", "clone", "--mirror"]
    repo_dir.mkdir(parents=True, exist_ok=True)
    assert isolation.prepare_repo_cache(req) == repo_dir
    assert git_calls[-1][0] == ["git", "fetch", "--all", "--tags", "--prune"]

    isolation.fetch_candidate_ref(repo_dir, Candidate(ref="main", repo=req.repo_url))
    assert git_calls[-1][0] == ["git", "fetch", "--all", "--tags", "--prune"]
    isolation.fetch_candidate_ref(repo_dir, candidate)
    assert "refs/pull/42/head:refs/pull/42/head" in git_calls[-1][0]

    plan_req = ExploreRequest(
        framework="sglang",
        repo_url=req.repo_url,
        work_dir=tmp_path / "plan",
        baseline=Baseline(throughput=100.0),
        prepare_candidate_env=False,
    )
    paths = isolation.prepare_candidate_workspace(plan_req, candidate, index=3, execute=True)
    assert paths.candidate_dir.name == "03_pr-42"
    assert not paths.worktree_dir.exists()

    worktree = tmp_path / "cleanup" / "worktree"
    venv = tmp_path / "cleanup" / "venv"
    worktree.mkdir(parents=True)
    venv.mkdir(parents=True)
    isolation.cleanup_workspace(
        isolation.WorkspacePaths(tmp_path / "cleanup", worktree, venv),
        is_winner=False,
        keep_winner_only=True,
        repo_dir=repo_dir,
    )
    assert not worktree.exists()
    assert not venv.exists()


def test_gbrain_page_client_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.agents.framework import gbrain_page_client as gbrain

    assert list(gbrain._iter_sse_objects("not json\n\ndata: {bad}\n\ndata: {\"id\":\"1\",\"result\":{\"ok\":true}}\n\n")) == [
        {"id": "1", "result": {"ok": True}}
    ]
    assert gbrain._select_mcp_response('data: {"id":"0","result":{"fallback":true}}\n\n', want_id="missing") == {
        "id": "0",
        "result": {"fallback": True},
    }
    assert gbrain._as_hit_list({"pages": [{"slug": "a"}, "bad"]}) == [{"slug": "a"}]
    assert gbrain._as_hit_list("bad") == []

    class _Resp:
        headers = {"Content-Type": "application/json", "Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            payload = {"result": {"content": [{"text": json.dumps({"slug": "page-1"})}]}}
            return json.dumps(payload).encode()

    captured = {}

    def _urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.headers.get("Authorization")
        return _Resp()

    monkeypatch.setattr(gbrain.urllib.request, "urlopen", _urlopen)
    client = gbrain.GbrainPageClient("https://gbrain.example/", "tok", timeout_sec=0.1)
    assert client.call("get_page", {"slug": "page-1"}) == {"slug": "page-1"}
    assert captured["url"] == "https://gbrain.example/mcp"
    assert captured["auth"] == "Bearer tok"
    assert client.get_page("page-1") == {"slug": "page-1"}

    class _ErrorResp(_Resp):
        def read(self, *_args):
            return b'{"error":{"message":"nope"}}'

    monkeypatch.setattr(gbrain.urllib.request, "urlopen", lambda req, timeout: _ErrorResp())
    with pytest.raises(gbrain.GbrainPageError, match="JSON-RPC error"):
        client.call("search", {"query": "x"})

    monkeypatch.setattr(gbrain.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(gbrain.GbrainPageError, match="transport error"):
        client.call("search", {"query": "x"})

    monkeypatch.setenv("GBRAIN_BASE_URL", "https://gbrain.example")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    monkeypatch.setenv("GBRAIN_HTTP_TIMEOUT_SEC", "not-a-number")
    assert isinstance(gbrain.build_gbrain_page_client_from_env(), gbrain.GbrainPageClient)


# ---------------------------------------------------------------------------
# orchestrator.knowledge.kb_writeback
# ---------------------------------------------------------------------------

def test_kb_writeback_default_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    root = kb_writeback._default_kb_root()
    assert root == tmp_path / "framework_optimization"


def test_kb_writeback_default_root_uses_user_data_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
    assert kb_writeback._default_kb_root() == tmp_path / "workspace" / "kb" / "framework_optimization"


async def test_kb_writeback_rejects_unknown_outcome() -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    with pytest.raises(ValueError):
        await kb_writeback.write_framework_record(
            pr_url="u",
            pr_sha="s",
            patch_path="p",
            outcome="not_a_real_outcome",
            tps_delta_pct=1.0,
            session_id="sess",
        )


# ---------------------------------------------------------------------------
# orchestrator.roles.base — retry / backoff
# ---------------------------------------------------------------------------

def test_retry_policy_env_and_on_retry_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.roles import base
    from hyperloom.orchestrator.roles.base import RetryPolicy

    monkeypatch.setenv("HL_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("HL_RETRY_BASE_S", "bad")
    monkeypatch.setenv("HL_RETRY_MAX_S", "inf")
    monkeypatch.setenv("HL_RETRY_MULT", "-1")
    monkeypatch.setenv("HL_RETRY_JITTER_S", "2")
    policy = RetryPolicy.from_env("HL_RETRY")
    assert policy.max_attempts == 1
    assert policy.base_delay_s == 1.0
    assert policy.max_delay_s == 30.0
    assert policy.multiplier == 2.0
    assert policy.jitter_s == 2.0

    monkeypatch.setattr(base.random, "uniform", lambda _lo, _hi: 0.25)
    assert RetryPolicy(base_delay_s=2.0, max_delay_s=3.0, multiplier=10.0, jitter_s=0.5).delay_for(2) == 3.25


@pytest.mark.asyncio
async def test_retry_with_backoff_swallows_on_retry_callback_error() -> None:
    from hyperloom.orchestrator.roles.base import RetryPolicy, retry_with_backoff

    calls = {"n": 0}
    slept: list[float] = []

    async def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return "ok"

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    def _bad_on_retry(*_args):
        raise RuntimeError("telemetry failed")

    out = await retry_with_backoff(
        _flaky,
        policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter_s=0.0),
        retry_on=(ConnectionError,),
        sleep=_sleep,
        on_retry=_bad_on_retry,
    )
    assert out == "ok"
    assert slept == [0.0]


# ---------------------------------------------------------------------------
# orchestrator.roles._runtime_bridge
# ---------------------------------------------------------------------------

def test_runtime_bridge_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.orchestrator.roles import _runtime_bridge as rb
    from hyperloom.orchestrator.roles.base import BackendError

    def fake_run(cmd, **kwargs):
        raise rb.subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    call = rb.RuntimeCall(
        phase="prepare-review",
        request_path=tmp_path / "req.json",
        out_path=tmp_path / "out.json",
        cwd=tmp_path,
        env={},
    )
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="critic", timeout_sec=5.0)


def test_runtime_bridge_not_found_and_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.orchestrator.roles import _runtime_bridge as rb
    from hyperloom.orchestrator.roles.base import BackendError

    call = rb.RuntimeCall(
        phase="tick",
        request_path=tmp_path / "req.json",
        out_path=tmp_path / "out.json",
        cwd=tmp_path,
        env={},
    )

    def fake_run_missing(cmd, **kwargs):
        raise FileNotFoundError("python gone")

    monkeypatch.setattr(rb.subprocess, "run", fake_run_missing)
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="a", timeout_sec=1.0)

    def fake_run_rc(cmd, **kwargs):
        return SimpleNamespace(returncode=3, stderr="boom")

    monkeypatch.setattr(rb.subprocess, "run", fake_run_rc)
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="a", timeout_sec=1.0)


# ---------------------------------------------------------------------------
# orchestrator.state.objective
# ---------------------------------------------------------------------------

def test_tput_objective_progress_zero() -> None:
    from hyperloom.orchestrator.state.objective import TargetTputObjective

    obj = TargetTputObjective(target_tput_per_gpu=100.0)
    state = SimpleNamespace(current_best={}, baseline_tput=0.0)
    assert obj.progress(state) == 0.0
    assert obj.reached(state) is False
    assert obj.describe() == "target_tput_per_gpu=100.0"


def test_baseline_objective_progress_zero_ref(tmp_path: Path) -> None:
    from hyperloom.orchestrator.state.objective import TargetBaselineObjective

    report = tmp_path / "benchmark_report.json"
    report.write_text(json.dumps({"throughput": {"output_throughput": 50.0}}), encoding="utf-8")
    obj = TargetBaselineObjective(baseline_dir=str(tmp_path))
    assert obj.kind() == "baseline"
    obj._ref_tput = 0.0
    state = SimpleNamespace(current_best={"tput": 10.0}, baseline_tput=0.0)
    assert obj.progress(state) == 0.0


# ---------------------------------------------------------------------------
# orchestrator.knowledge.kb_writeback — quantization prompt
# ---------------------------------------------------------------------------

def test_quantization_join_and_prompt() -> None:
    from hyperloom.orchestrator.phases import quantization_schemes as qs

    assert qs._join_clauses([]) == ""
    assert qs._join_clauses(["a"]) == "a"
    assert qs._join_clauses(["a", "b", "c"]) == "a, b and c"

    cfg = qs.QuantizationConfig(global_scheme="fp8")
    prompt = qs.build_quantization_prompt(cfg, model_path="/m", gpu_type="mi300x")
    assert "Quantize /m on an MI300X target." in prompt
    assert "Quantization strategy" in prompt


# ---------------------------------------------------------------------------
# orchestrator.actions.executors._file_lock
# ---------------------------------------------------------------------------

def test_file_lock_no_fcntl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    from hyperloom.orchestrator.actions.executors import _file_lock

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with _file_lock.best_effort_file_lock(str(tmp_path / "lock")):
        ran = True
    assert ran


# ---------------------------------------------------------------------------
# orchestrator.actions.executors._framework_gap_composer
# ---------------------------------------------------------------------------

def test_framework_gap_bottleneck(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors import _framework_gap_composer as gc

    bp = tmp_path / "bd.json"
    bp.write_text(json.dumps({"top_kernels": ["fused_moe_gemm_kernel"]}), encoding="utf-8")
    assert gc._extract_bottleneck_from_breakdown(str(bp)) == "moe"

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"top_kernels": []}), encoding="utf-8")
    assert gc._extract_bottleneck_from_breakdown(str(empty)) == ""

    assert gc._extract_bottleneck_from_breakdown(None) == ""
    assert gc._extract_bottleneck_from_breakdown(str(tmp_path / "missing.json")) == ""


# ---------------------------------------------------------------------------
# inference_optimizer.protocol.intent
# ---------------------------------------------------------------------------

def test_validate_envelope_structural_errors() -> None:
    from hyperloom.inference_optimizer.protocol.intent import IntentValidationError, validate_envelope

    with pytest.raises(IntentValidationError):
        validate_envelope("not a dict")  # type: ignore[arg-type]
    with pytest.raises(IntentValidationError):
        validate_envelope({})
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": "x"})
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": ["x"]})
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "alert"}]})
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "alert", "payload": "x"}]})
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "bad_type", "payload": {}}]})


def test_validate_envelope_review_verdict_map_keys() -> None:
    from hyperloom.inference_optimizer.protocol.intent import IntentValidationError, validate_envelope

    bad = {
        "intents": [
            {
                "intent_type": "review_verdict",
                "payload": {"target_proposal_msg_id": "m1", "verdict_map": {"": {"verdict": "approve"}}},
            }
        ]
    }
    with pytest.raises(IntentValidationError):
        validate_envelope(bad)


# ---------------------------------------------------------------------------
# inference_optimizer.session.paths
# ---------------------------------------------------------------------------

def test_paths_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session import paths

    assert paths._sanitize_model_basename("   ") == "session"
    assert paths._sanitize_model_basename("/a/b/Model:X") == "Model_X"

    monkeypatch.setenv(paths.ENV_OVERRIDE_ASSET_ROOT, str(tmp_path))
    assert paths.asset_root() == tmp_path
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "does_not_exist"))
    assert paths.find_latest_per_session_dir() is None


# ---------------------------------------------------------------------------
# orchestrator.actions.registry
# ---------------------------------------------------------------------------

def test_action_registry_names_all_and_lazy_load(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.registry import ActionRegistry

    meta_dir = tmp_path / "_meta"
    meta_dir.mkdir()
    (meta_dir / "_ignored.yaml").write_text("name: ignored\n", encoding="utf-8")
    (meta_dir / "target_analysis.yaml").write_text(
        "\n".join(
            [
                "name: target_analysis",
                "family: prep",
                "cost_minutes_p50: 0.1",
                "cost_minutes_p75: 0.2",
                "expected_gain_pct: [0, 0]",
                "accuracy_risk: 0",
                "crash_risk: 0",
                "requires_lanes: []",
                "allowed_tools: [Read]",
                "side_effects: [writes_state]",
                "pipeline_phase: prep",
                "verdict_class: archival",
            ]
        ),
        encoding="utf-8",
    )

    reg = ActionRegistry(tmp_path)
    assert reg.names() == ["target_analysis"]
    assert [meta.name for meta in reg.all()] == ["target_analysis"]
    meta = reg.get("target_analysis")
    assert meta is not None
    assert meta.description == "target_analysis"
    assert reg.get("missing") is None


# ---------------------------------------------------------------------------
# orchestrator.loop.dispatcher
# ---------------------------------------------------------------------------

def test_dispatcher_inline_whitelist_filters_and_registry_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    reg = SimpleNamespace(names=lambda: ["report", "missing", "lane_action", "ok_action"])
    coord = SimpleNamespace(
        action_registry=reg,
        sub=SimpleNamespace(executor_registry={"lane_action": object(), "ok_action": object()}),
        _INLINE_ACTION_DENY=frozenset({"report"}),
    )
    disp = DispatcherCollaborator(coord)
    monkeypatch.setattr(disp, "_registry_lanes_ttl", lambda name: (["gpu"] if name == "lane_action" else [], 60))
    assert disp._inline_action_whitelist() == frozenset({"ok_action"})

    coord.action_registry = SimpleNamespace(names=lambda: (_ for _ in ()).throw(RuntimeError("bad registry")))
    assert disp._inline_action_whitelist() == frozenset()

    coord.action_registry = None
    assert disp._inline_action_whitelist() == frozenset()


def test_dispatcher_inline_whitelist_all_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    reg = SimpleNamespace(all=lambda: [SimpleNamespace(name="from_all")])
    coord = SimpleNamespace(
        action_registry=reg,
        sub=SimpleNamespace(executor_registry={"from_all": object()}),
        _INLINE_ACTION_DENY=frozenset(),
    )
    disp = DispatcherCollaborator(coord)
    monkeypatch.setattr(disp, "_registry_lanes_ttl", lambda _name: ([], 60))
    assert disp._inline_action_whitelist() == frozenset({"from_all"})


def test_dispatcher_run_action_now_sync_edge_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.loop import dispatcher as dispatcher_mod
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    coord = SimpleNamespace(
        _inline_fast_actions_enabled=True,
        _coordinator_loop=None,
        _INLINE_ACTION_DENY=frozenset(),
        action_registry=None,
        sub=SimpleNamespace(executor_registry={}),
    )
    disp = DispatcherCollaborator(coord)
    assert "action_name required" in disp._run_action_now_sync("  ", {})

    monkeypatch.setattr(disp, "_inline_action_whitelist", lambda: frozenset({"probe"}))
    assert "coordinator loop not running" in disp._run_action_now_sync("probe", {})

    coord._coordinator_loop = SimpleNamespace(is_closed=lambda: False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "not-a-float")
    monkeypatch.setattr(disp, "_run_action_now", lambda _name, _params: object())

    class _TimeoutFuture:
        def result(self, timeout):
            assert timeout == 120.0
            raise FuturesTimeoutError()

    monkeypatch.setattr(dispatcher_mod.asyncio, "run_coroutine_threadsafe", lambda _coro, _loop: _TimeoutFuture())
    assert "still running after 120s" in disp._run_action_now_sync("probe", {})

    class _ErrorFuture:
        def result(self, timeout):
            raise RuntimeError("boom")

    monkeypatch.setattr(dispatcher_mod.asyncio, "run_coroutine_threadsafe", lambda _coro, _loop: _ErrorFuture())
    assert "errored" in disp._run_action_now_sync("probe", {})

    monkeypatch.setattr(
        dispatcher_mod.asyncio,
        "run_coroutine_threadsafe",
        lambda _coro, _loop: (_ for _ in ()).throw(RuntimeError("closed")),
    )
    assert "could not schedule" in disp._run_action_now_sync("probe", {})


# ---------------------------------------------------------------------------
# inference_optimizer.multi_node.state_paths
# ---------------------------------------------------------------------------

def test_multi_node_state_paths_resolution_and_migration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    with pytest.raises(RuntimeError, match="cannot resolve"):
        state_paths.resolve_state_file()

    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(explicit))
    assert state_paths.resolve_state_file() == explicit

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    session = tmp_path / "session"
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))
    assert state_paths.resolve_state_file() == session / "runtime" / "multi_node_state.json"

    missing = tmp_path / "missing.json"
    assert state_paths.state_file_safe_to_read(missing) is False
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text("{}", encoding="utf-8")
    unsafe.chmod(0o666)
    assert state_paths.state_file_safe_to_read(unsafe) is False
    unsafe.chmod(0o600)
    assert state_paths.state_file_safe_to_read(unsafe) is True

    src = tmp_path / "source_state.json"
    src.write_text('{"nodes": []}', encoding="utf-8")
    src.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(src))
    bound = state_paths.bind_state_file_to_session(session)
    assert bound == session / "runtime" / "multi_node_state.json"
    assert bound.read_text(encoding="utf-8") == '{"nodes": []}'
    assert state_paths.resolve_state_file() == bound
    assert bound.stat().st_mode & 0o777 == 0o600
    assert bound.parent.stat().st_mode & 0o777 == 0o700


def test_multi_node_state_paths_warn_on_permission_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths

    messages: list[str] = []
    monkeypatch.setattr(state_paths, "warn", messages.append)

    class _BadPath:
        def chmod(self, _mode):
            raise OSError("chmod denied")

    state_paths._chmod_state_file(_BadPath())
    assert "could not chmod state file" in messages[-1]

    runtime_dir = tmp_path / "runtime"
    original_chmod = type(runtime_dir).chmod

    def _bad_chmod(self, mode):
        if self == runtime_dir:
            raise OSError("runtime chmod denied")
        return original_chmod(self, mode)

    monkeypatch.setattr(type(runtime_dir), "chmod", _bad_chmod)
    state_paths._ensure_runtime_dir(runtime_dir)
    assert runtime_dir.is_dir()
    assert "could not chmod runtime dir" in messages[-1]


# ---------------------------------------------------------------------------
# model_compat
# ---------------------------------------------------------------------------

def test_model_compat_whitelist_and_local_file_edges(tmp_path: Path) -> None:
    import model_compat

    model_compat.load_whitelist.cache_clear()
    whitelist = tmp_path / "whitelist.json"
    whitelist.write_text(
        json.dumps({"candidates": [{"repo_id": "org/a"}, {"repo_id": ""}, {"no_repo": "ignored"}]}),
        encoding="utf-8",
    )
    assert model_compat.load_whitelist(str(whitelist)) == frozenset({"org/a"})
    assert model_compat.load_whitelist(str(tmp_path / "missing.json")) == frozenset()

    missing_dir = tmp_path / "missing-model"
    assert model_compat.has_weights(missing_dir) is False
    assert model_compat.has_tokenizer(missing_dir) is True


def test_hf_gated_rotates_auth_failures_to_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    calls = {"n": 0}

    def _urlopen(_req, timeout):
        assert timeout == 20
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _urlopen)
    assert model_compat.hf_gated("org/model", ["tok-a", "tok-b"]) == "gated"
    assert calls["n"] == 3


def test_hf_gated_retries_rate_limit_and_generic_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    sleeps: list[int] = []
    monkeypatch.setattr(model_compat.time, "sleep", lambda delay: sleeps.append(delay))

    calls = {"n": 0}

    def _rate_limit_then_open(_req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _JsonResponse({"gated": False})

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _rate_limit_then_open)
    assert model_compat.hf_gated("org/model", ["tok"]) is None
    assert sleeps == [5]

    def _always_boom(_req, timeout):
        raise RuntimeError("network")

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _always_boom)
    assert model_compat.hf_gated("org/model", ["tok"]) is None


def test_hf_missing_tokenizer_rate_limit_and_http_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_args: None)
    calls = {"n": 0}

    def _rate_limit_then_missing(_req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _JsonResponse({"siblings": [{"rfilename": "model.safetensors"}]})

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _rate_limit_then_missing)
    assert model_compat.hf_missing_tokenizer("org/model", ["tok"]) == "missing_tokenizer"

    def _server_error(_req, timeout):
        raise urllib.error.HTTPError("u", 500, "server", {}, None)

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _server_error)
    assert model_compat.hf_missing_tokenizer("org/model", ["tok"]) is None


# ---------------------------------------------------------------------------
# inference_optimizer.gpu_types
# ---------------------------------------------------------------------------

def test_gpu_type_autodetect_rocm_and_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    class _Completed:
        stdout = "GPU[0] : Card series: AMD Instinct MI325X"

    def _rocm_ok(cmd, capture_output, text, timeout):
        assert cmd == ["rocm-smi", "--showproductname"]
        assert capture_output is True
        assert text is True
        assert timeout == 5
        return _Completed()

    monkeypatch.setattr(subprocess, "run", _rocm_ok)
    assert gpu_types._autodetect_gpu_type() == "mi325x"

    def _rocm_missing(*_args, **_kwargs):
        raise FileNotFoundError("rocm-smi")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        )
    )
    monkeypatch.setattr(subprocess, "run", _rocm_missing)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert gpu_types._autodetect_gpu_type() == "mi355x"

    fake_torch.cuda.get_device_properties = lambda _idx: (_ for _ in ()).throw(RuntimeError("no gpu"))
    assert gpu_types._autodetect_gpu_type() is None


# ---------------------------------------------------------------------------
# breakdown.recorder.section_shape / breakdown.reporters
# ---------------------------------------------------------------------------

def test_section_shape_unknown_is_none() -> None:
    from hyperloom.inference_optimizer.breakdown.recorder import section_shape

    assert section_shape("not_registered") is None


def test_source_files_renderer_skips_empty_entries() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers import source_files

    sec = source_files.render(
        {
            "source_files": {
                "empty": [],
                "none": None,
                "single": "state.json",
                "many": ["a", "b", "c", "d"],
            }
        }
    )
    assert not sec.skipped
    assert "single" in sec.markdown_block
    assert "none" not in sec.markdown_block
    assert "a, b, c" in sec.markdown_block


def test_decision_journal_standard_caps_rounds() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers import decision_journal

    rounds = [
        {
            "phase": "explore",
            "round_id": f"r{i}",
            "variants": [{"name": f"v{i}", "outcome": "tested", "gain_pct_vs_base": i}],
            "round_decision": {"outcome": "discarded"},
        }
        for i in range(35)
    ]
    sec = decision_journal.render({"decision_journal": rounds})
    assert "Showing last 20 of 35 rounds" in sec.markdown_block
    assert any(d.kind == "rejected" for d in sec.decisions)


def test_roofline_and_workload_render_minimal_inputs() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers import roofline, workload

    roof = roofline.render(
        {
            "roofline": [
                {
                    "source_path": "final.json",
                    "mode": "compare",
                    "baseline": {"top_kernel": {"name": "k1", "gpu_pct": 12.3}},
                    "delta": {"compute_pct": "+1.0"},
                }
            ]
        }
    )
    assert not roof.skipped
    assert "k1" in roof.markdown_block

    wk = workload.render({"workload": {"model_name": "m", "framework_name": "sglang"}})
    assert not wk.skipped
    assert "sglang" in wk.markdown_block


def test_llm_prompt_parse_response_edges() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.llm_prompt import parse_llm_response

    fenced = """```json
{"executive_summary": "  ok  ", "section_narratives": {"a": "  first  ", "2": "two"}}
```"""
    parsed = parse_llm_response(fenced)
    assert parsed["executive_summary"] == "ok"
    assert parsed["section_narratives"] == {"a": "first", "2": "two"}

    assert parse_llm_response("not json") == {"executive_summary": "", "section_narratives": {}}
    assert parse_llm_response("[]") == {"executive_summary": "", "section_narratives": {}}


# ---------------------------------------------------------------------------
# orchestrator.state.orchestration_memory.deterministic_memory_fallback
# ---------------------------------------------------------------------------

def test_deterministic_memory_fallback_bad_gain() -> None:
    from hyperloom.orchestrator.state.orchestration_memory import deterministic_memory_fallback

    state = SimpleNamespace(
        current_best={"tput": 123.0},
        optimization_stack=[],
        cumulative_gain_validated="not-a-number",
        phase="EXPLORE",
        macro_cycle=2,
    )
    record = deterministic_memory_fallback(state)
    assert "current_plan" in record
    assert "phase=EXPLORE" in record["current_plan"]


# ---------------------------------------------------------------------------
# orchestrator.specialists.profile
# ---------------------------------------------------------------------------

def test_coerce_bool_and_infer_scope() -> None:
    from hyperloom.orchestrator.specialists import profile as sp

    assert sp._coerce_bool("off", default=True) is False
    assert sp._coerce_bool("yes", default=False) is True
    assert sp._coerce_bool(None, default=True) is True
    assert sp._coerce_bool("???", default=True) is True

    profile = sp.resolve_specialist_profile({})
    assert profile.scope == sp.SCOPE_FREEFORM


# ---------------------------------------------------------------------------
# orchestrator.actions.executors._accuracy_gate
# ---------------------------------------------------------------------------

def test_parse_quality_gate_paths(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag

    assert ag.parse_quality_gate(tmp_path)["quality_gate"] is None

    (tmp_path / "benchmark_report.json").write_text("{not valid json", encoding="utf-8")
    res = ag.parse_quality_gate(tmp_path)
    assert res["quality_gate"] is None
    assert "parse error" in res["error"]

    (tmp_path / "benchmark_report.json").write_text(json.dumps({"throughput": {}}), encoding="utf-8")
    res2 = ag.parse_quality_gate(tmp_path)
    assert res2["quality_gate"] is None

    (tmp_path / "benchmark_report.json").write_text(
        json.dumps({"quality_gate": {"passed": True}}), encoding="utf-8"
    )
    res3 = ag.parse_quality_gate(tmp_path)
    assert res3["quality_gate"] == {"passed": True}


# ---------------------------------------------------------------------------
# orchestrator.trace.trace_env
# ---------------------------------------------------------------------------

def test_env_flag_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.trace import trace_env

    monkeypatch.setenv("HL_TEST_FLAG", "on")
    assert trace_env.env_flag("HL_TEST_FLAG") is True
    monkeypatch.setenv("HL_TEST_FLAG", "off")
    assert trace_env.env_flag("HL_TEST_FLAG") is False
    monkeypatch.setenv("HL_TEST_FLAG", "maybe")
    assert trace_env.env_flag("HL_TEST_FLAG", default=True) is True
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    assert trace_env.env_flag("HL_TEST_FLAG", default=False) is False


# ---------------------------------------------------------------------------
# orchestrator.bus.gpu_pool._parse_gpu_list
# ---------------------------------------------------------------------------

def test_parse_gpu_list() -> None:
    from hyperloom.orchestrator.bus.gpu_pool import _parse_gpu_list

    assert _parse_gpu_list("0, 1 ; 2, x, 1, -3") == [0, 1, 2]
    assert _parse_gpu_list("") == []
    assert _parse_gpu_list(None) == []  # type: ignore[arg-type]
