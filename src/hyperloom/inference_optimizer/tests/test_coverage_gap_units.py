# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


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

    monkeypatch.delenv("HL_STR", raising=False)
    assert env.env_str("HL_STR", default="fallback") == "fallback"
    monkeypatch.setenv("HL_STR", " value ")
    assert env.env_str("HL_STR") == "value"


def test_common_atomic_writes_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.common import io

    text_path = tmp_path / "nested" / "value.txt"
    io.atomic_write_text(text_path, "hello", make_parents=True)
    assert text_path.read_text(encoding="utf-8") == "hello"

    bytes_path = tmp_path / "payload.bin"
    io.atomic_write_bytes(bytes_path, b"\x00\x01")
    assert bytes_path.read_bytes() == b"\x00\x01"

    json_path = tmp_path / "data.json"
    io.atomic_write_json(json_path, {"b": 2, "a": 1}, indent=None, trailing_newline=True)
    assert json_path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'

    def _boom(_tmp, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", _boom)
    with pytest.raises(OSError, match="replace failed"):
        io.atomic_write_text(tmp_path / "will_fail.txt", "x")
    assert not list(tmp_path.glob(".will_fail.txt.*.tmp"))


def test_credentials_endpoint_resolution_and_geak_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import credentials

    assert credentials._derive_anthropic_base_url("https://gw.example/v1") == "https://gw.example"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://open.example/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert credentials._resolve_llm_endpoints() == ("https://open.example", "https://open.example/v1")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "https://open.example/v1")

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "https://anthropic.example")

    cfg = tmp_path / "geak.yaml"
    cfg.write_text("model: x\n  base_url: https://old/v1\n", encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is True
    assert r"https://new.example/\g/v1" in cfg.read_text(encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is False
    assert credentials._sync_geak_config_base_url(str(tmp_path / "missing.yaml"), "https://x") is False


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

    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "dynamo")
    assert mn._resolve_mn_backend(argparse.Namespace(mn_backend=None)) == "dynamo"
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


def test_provision_multi_node_rayjob_and_dynamo_stacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            rayjob_image="",
            rayjob_gpus_per_node=None,
            rayjob_extra_env=["A=B"],
        )
    )
    assert created[0].image == "prior:tag"
    assert booted
    assert os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"].endswith("wid-rj/torch_trace")

    dynamo_created: list[argparse.Namespace] = []
    monkeypatch.setattr(mn_cli, "cmd_create_dynamo", lambda ns: dynamo_created.append(ns) or 0)
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"service_url": "http://svc:8000"})
    monkeypatch.setattr(mn_cli, "install_geak_on_pods_best_effort", lambda: 0)
    opt_mn._provision_multi_node_dynamo_stack(
        argparse.Namespace(
            nodes=2,
            rayjob_image="dyn:tag",
            rayjob_gpus_per_node=4,
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
    assert dynamo_created[0].image == "dyn:tag"
    assert dynamo_created[0].kv_transfer_backend == "mooncake"
    assert os.environ["BENCHMARK_BASE_URL"] == "http://svc:8000"


def _patch_dynamo_state(monkeypatch: pytest.MonkeyPatch, state: dict) -> list[dict]:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    saved: list[dict] = []
    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    return saved


def test_dynamo_create_and_state_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    saved: list[dict] = []
    monkeypatch.setenv("SAFE_WORKSPACE", "ws")
    monkeypatch.setenv("CLAW_SESSION_ID", "sess-1")
    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: saved[-1] if saved else {})
    monkeypatch.setattr(dyn._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(dyn._mn_cli, "_dynamo_ssh_dir", lambda: tmp_path / "ssh")
    monkeypatch.setattr(dyn.ssh_client, "generate_session_keypair", lambda _d: (tmp_path / "id_ed25519", "ssh-ed25519 pub"))
    monkeypatch.setattr(dyn.workload_spec, "build_dynamo_workload_body", lambda **kw: {"body": kw})
    workload = {
        "phase": "Running",
        "pods": [
            {"podId": "app-worker-0", "resourceId": 1, "podIP": "10.0.1.0"},
            {"podId": "app-worker-1", "resourceId": 1, "podIP": "10.0.1.1"},
        ],
    }
    fake_safe = _FakeSafe(workload=workload)
    monkeypatch.setattr(dyn.safe_client, "from_env", lambda: fake_safe)
    monkeypatch.setattr(dyn._mn_cli, "_short_poll", lambda **kw: workload)
    monkeypatch.setattr(dyn._mn_cli, "_refresh_dynamo_known_hosts", lambda ips, port, state: tmp_path / "known_hosts")
    monkeypatch.setattr(dyn._mn_cli, "_dynamo_known_hosts_path", lambda state: tmp_path / "known_hosts")
    monkeypatch.setattr(dyn.ssh_client, "probe_ssh", lambda *a, **kw: True)

    args = argparse.Namespace(
        extra_env=["ENV=1"],
        extra_label=["team=perf"],
        owner_id=None,
        workspace=None,
        display_name="dyn",
        image="dyn:tag",
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
    assert dyn.cmd_create_dynamo(args) == 0
    assert saved[-1]["backend"] == "dynamo"
    assert saved[-1]["worker_pod_ips"] == ["10.0.1.0", "10.0.1.1"]
    assert saved[-1]["ssh_known_hosts"] == str(tmp_path / "known_hosts")

    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: {"backend": "rayjob"})
    with pytest.raises(RuntimeError, match="state backend"):
        dyn._dynamo_require_state()
    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: {"backend": "dynamo", "worker_pod_ips": ["10.0.1.0"]})
    with pytest.raises(RuntimeError, match="ssh_key_path"):
        dyn._dynamo_require_state()


def test_dynamo_forward_env_and_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    monkeypatch.setenv("MORI_FOO", "1")
    monkeypatch.setenv("SGLANG_MORI_BAR", "2")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", "/shared/traces")
    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", json.dumps({"SGLANG_USE_AITER": "1", "MORI_FOO": "override"}))
    fwd = dyn._collect_forward_env()
    assert fwd["MORI_FOO"] == "override"
    assert fwd["SGLANG_TORCH_PROFILER_DIR"] == "/shared/traces"
    assert fwd["SGLANG_USE_AITER"] == "1"

    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", "{bad")
    assert dyn._collect_forward_env()["MORI_FOO"] == "1"

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(dyn._mn_cli, "_read_pod_script", lambda name: f"script:{name}")

    def _run(state, ip, script, python, launch_args, **kw):
        calls.append((ip, launch_args))
        if ip == "10.0.0.2":
            raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=kw["timeout"])
        return _Completed(returncode=1 if ip == "10.0.0.3" else 0, stdout='noise {"status":"ok"}\n', stderr="bad")

    monkeypatch.setattr(dyn._mn_cli, "_dynamo_ssh_run_script", _run)
    rc, results = dyn._dynamo_fanout_launch(
        {"ssh_port": 2222},
        "--model /m",
        ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
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


def test_dynamo_restart_and_kill_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    saved: list[dict] = []
    state = {
        "backend": "dynamo",
        "framework": "sglang",
        "nodes": 2,
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    build_calls: list[dict] = []
    monkeypatch.setattr(
        dyn.dynamo_support,
        "build_node_launch_args",
        lambda **kw: build_calls.append(kw) or "launch-args",
    )
    fanout_calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        dyn,
        "_dynamo_fanout_launch",
        lambda st, args, ips, **kw: fanout_calls.append((kw["label"], list(ips))) or (0, [{"ok": True}]),
    )
    assert dyn._dynamo_restart_server(_restart_args()) == 0
    assert build_calls[-1]["nnodes"] == 2
    assert fanout_calls[-1] == ("restart", ["10.0.1.0", "10.0.1.1"])
    assert saved[-1]["last_restart_pd_mode"] == "aggregated"

    pd_state = {
        **state,
        "pd_mode": "disaggregated",
        "prefill_pod_ips": ["10.0.2.0"],
        "decode_pod_ips": ["10.0.3.0"],
    }
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(pd_state))
    assert dyn._dynamo_restart_server(
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

    pd_state["framework"] = "vllm"
    with pytest.raises(RuntimeError, match="sglang-only"):
        dyn._dynamo_restart_server(_restart_args())

    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state | {"last_restart_framework": "vllm"}))
    assert dyn._dynamo_kill_inference(argparse.Namespace(poll_timeout=10, print_logs=False)) == 0
    assert build_calls[-1]["kill_only"] is True
    assert saved[-1]["last_kill_results"] == [{"ok": True}]


def test_dynamo_node_ops_apply_revert_and_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    state = {
        "backend": "dynamo",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_read_bundled_pod_python_script", lambda name: f"script:{name}")
    monkeypatch.setattr(
        dyn._mn_cli,
        "_dynamo_ssh_run_script",
        lambda *a, **kw: _Completed(returncode=0, stdout='logs {"status":"ok","backup_path":"/b"}', stderr=""),
    )
    parsed, tx = dyn._dynamo_ssh_node_op(state, "10.0.1.0", "apply", timeout=5)
    assert parsed and parsed["status"] == "ok"
    assert tx["rc"] == 0

    monkeypatch.setattr(
        dyn._mn_cli,
        "_dynamo_ssh_run_script",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)),
    )
    parsed, tx = dyn._dynamo_ssh_node_op(state, "10.0.1.0", "apply", timeout=5)
    assert parsed is None and tx["rc"] == 124

    responses = iter(
        [
            ({"status": "ok", "backup_path": "/b0"}, {"rc": 0, "stderr": ""}),
            ({"status": "failed", "error": "nope"}, {"rc": 1, "stderr": "bad"}),
        ]
    )
    monkeypatch.setattr(dyn, "_dynamo_ssh_node_op", lambda *a, **kw: next(responses))
    patch_file = tmp_path / "p.diff"
    patch_file.write_text("diff", encoding="utf-8")
    rc = dyn._dynamo_apply_patch(
        argparse.Namespace(
            patch_file=str(patch_file),
            target_path="/remote/x.py",
            backup_dir="/backups",
            kernel_id="k1",
            timeout_sec=10,
        )
    )
    assert rc == 1

    missing_rc = dyn._dynamo_apply_patch(
        argparse.Namespace(
            patch_file=str(tmp_path / "missing.diff"),
            target_path="/remote/x.py",
            backup_dir="/backups",
            kernel_id="k1",
            timeout_sec=10,
        )
    )
    assert missing_rc == dyn.EXIT_CONFIG_ERROR

    assert dyn._dynamo_revert_patch(argparse.Namespace(backup_map_json="{", target_path="/x", timeout_sec=1)) == dyn.EXIT_CONFIG_ERROR
    assert dyn._dynamo_revert_patch(argparse.Namespace(backup_map_json="{}", target_path="/x", timeout_sec=1)) == dyn.EXIT_CONFIG_ERROR
    monkeypatch.setattr(dyn, "_dynamo_ssh_node_op", lambda *a, **kw: ({"status": "restored"}, {"rc": 0, "stderr": ""}))
    assert dyn._dynamo_revert_patch(
        argparse.Namespace(backup_map_json=json.dumps({"10.0.1.0": "/b"}), target_path="/x", timeout_sec=1)
    ) == 0

    assert dyn._dynamo_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{bad",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=False,
        )
    ) == dyn.EXIT_CONFIG_ERROR
    monkeypatch.setattr(dyn, "_dynamo_ssh_node_op", lambda *a, **kw: (None, {"rc": 1, "stderr": "no json"}))
    assert dyn._dynamo_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{}",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=True,
        )
    ) == dyn.EXIT_TRANSIENT
    monkeypatch.setattr(dyn, "_dynamo_ssh_node_op", lambda *a, **kw: ({"status": "ok", "result": 1}, {"rc": 0, "stderr": ""}))
    assert dyn._dynamo_kernel_bench(
        argparse.Namespace(
            workspace="/w",
            bench_command="true",
            files_b64_json="{}",
            result_glob="*.json",
            timeout_sec=10,
            print_logs=False,
        )
    ) == 0


def test_dynamo_tracelens_and_geak_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    state = {
        "backend": "dynamo",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    saved = _patch_dynamo_state(monkeypatch, state)
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(dyn._mn_cli, "_poll_timeout_from_args", lambda args: 5)

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        dyn._mn_cli,
        "_dynamo_ssh_run_script",
        lambda st, ip, script, python, op_args, **kw: calls.append((ip, python, op_args))
        or _Completed(returncode=0, stdout='{"status":"applied","per_pod":[{"status":"applied"}]}', stderr=""),
    )
    assert dyn._dynamo_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tracelens", sglang_version_pin="v1", poll_timeout=5)
    ) == 0
    assert calls[0][1] == "/opt/venv/bin/python"
    assert "--sglang-version-pin v1" in calls[0][2]

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert dyn.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=False)) == dyn.EXIT_CONFIG_ERROR
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        dyn._mn_cli,
        "_dynamo_ssh_run_script",
        lambda *a, **kw: _Completed(returncode=0, stdout='{"status":"installed"}', stderr=""),
    )
    assert dyn.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=True)) == 0
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


def test_dynamo_process_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    assert dyn._dynamo_all_gpu_ips({"pd_mode": "disaggregated", "prefill_pod_ips": ["p"], "decode_pod_ips": ["d"]}) == ["p", "d"]
    assert dyn._dynamo_all_gpu_ips({"worker_pod_ips": ["w"]}) == ["w"]

    state = {
        "backend": "dynamo",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0"],
        "ssh_key_path": "/tmp/k",
    }
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state))
    monkeypatch.setattr(dyn, "_dynamo_all_gpu_ips", lambda st: [])
    assert dyn._dynamo_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="", sglang_version_pin="", poll_timeout=1)
    ) == dyn.EXIT_CONFIG_ERROR

    monkeypatch.setattr(dyn, "_dynamo_all_gpu_ips", lambda st: ["10.0.1.0"])
    monkeypatch.setattr(dyn._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(dyn._mn_cli, "_poll_timeout_from_args", lambda args: 1)
    monkeypatch.setattr(
        dyn._mn_cli,
        "_dynamo_ssh_run_script",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=1)),
    )
    assert dyn._dynamo_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", poll_timeout=1)
    ) == 1


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
    # Deleted-file sections end at /dev/null and are filtered as placeholder
    # sections by the current parser contract; the branch is still exercised.
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
        mp.setattr(opt_mn, "_provision_multi_node_dynamo_stack", lambda _args: called.append("dynamo"))
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="dynamo"))
    assert called == ["dynamo"]

    state_file = tmp_path / "missing_state.json"
    monkeypatch.setattr(state_paths, "resolve_state_file", lambda: state_file)
    for key in ("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", "INFERENCE_OPTIMIZER_GPUS_PER_NODE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", rayjob_image="", rayjob_gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 2

    state_file.write_text(json.dumps({"last_create_request": {"image": "img:old"}}), encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "bad")
    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: 7)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", rayjob_image="", rayjob_gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 7

    monkeypatch.setattr(mn_cli, "cmd_create_rayjob", lambda ns: 0)
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"rayjob_id": "wid"})
    monkeypatch.setattr(mn_cli, "cmd_bootstrap", lambda ns: 9)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_rayjob_stack(argparse.Namespace(nodes=2, mn_backend="rayjob", rayjob_image="img:new", rayjob_gpus_per_node=None, rayjob_extra_env=[]))
    assert exc.value.code == 9

    state_file.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_dynamo_stack(
            argparse.Namespace(
                nodes=2,
                rayjob_image="",
                rayjob_gpus_per_node=None,
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
        opt_mn._provision_multi_node_dynamo_stack(
            argparse.Namespace(
                nodes=2,
                rayjob_image="",
                rayjob_gpus_per_node=None,
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


def test_dynamo_install_timeout_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    state = {
        "backend": "dynamo",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: dict(state))
    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(dyn._mn_cli, "_poll_timeout_from_args", lambda args: 5)

    geak_calls = {"n": 0}

    def _geak_run(*_args, **_kwargs):
        geak_calls["n"] += 1
        if geak_calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)
        return _Completed(returncode=0, stdout='{"status":"failed","reason":"pip"}', stderr="")

    monkeypatch.setattr(dyn._mn_cli, "_dynamo_ssh_run_script", _geak_run)
    assert dyn.cmd_install_geak(argparse.Namespace(geak_src="/geak", poll_timeout=5, print_logs=False)) == 1


def test_dynamo_create_reuse_and_restart_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.dynamo as dyn

    def _args(**overrides):
        defaults = dict(
            extra_env=[],
            extra_label=[],
            owner_id=None,
            workspace="ws",
            display_name="dyn",
            image="dyn:tag",
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
    state = {"backend": "dynamo", "rayjob_id": "wid-old", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k"}
    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: saved[-1] if saved else dict(state))
    monkeypatch.setattr(dyn._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(dyn._mn_cli, "_dynamo_ssh_dir", lambda: tmp_path / "ssh")
    monkeypatch.setattr(dyn.ssh_client, "generate_session_keypair", lambda _d: (tmp_path / "id_ed25519", "ssh-ed25519 pub"))
    monkeypatch.setattr(dyn.workload_spec, "build_dynamo_workload_body", lambda **kw: {"body": kw})
    monkeypatch.setattr(dyn.ssh_client, "probe_ssh", lambda *a, **kw: False)
    monkeypatch.setattr(dyn._mn_cli, "_refresh_dynamo_known_hosts", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("keyscan")))

    class _ReuseSafe(_FakeSafe):
        def get_workload(self, wid: str) -> dict:
            return {"phase": "Running", "pods": []}

        def get_workload_service(self, wid: str) -> dict:
            raise dyn.safe_client.SafeApiError(500, "svc down", endpoint="GET /service")

    monkeypatch.setattr(dyn.safe_client, "from_env", lambda: _ReuseSafe())
    assert dyn.cmd_create_dynamo(_args()) == 0
    assert saved[-1]["rayjob_id"] == "wid-old"
    assert saved[-1]["worker_pod_ips"] == []

    class _GoneSafe(_FakeSafe):
        def get_workload(self, wid: str) -> dict:
            raise dyn.safe_client.SafeApiError(404, "gone", endpoint="GET /api/v1/workloads/wid-old")

    saved.clear()
    monkeypatch.setattr(dyn.safe_client, "from_env", lambda: _GoneSafe(workload={"phase": "Running", "pods": []}))
    assert dyn.cmd_create_dynamo(_args()) == 0
    assert saved[-1]["rayjob_id"] == "wid-test"

    monkeypatch.setattr(dyn._mn_cli, "_load_state", lambda: {"backend": "dynamo", "worker_pod_ips": [], "ssh_key_path": "/tmp/k"})
    with pytest.raises(RuntimeError, match="no GPU pod IPs"):
        dyn._dynamo_require_state()

    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: {"backend": "dynamo", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k", "framework": "bad"})
    with pytest.raises(RuntimeError, match="unsupported framework"):
        dyn._dynamo_restart_server(_restart_args())

    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: {"backend": "dynamo", "worker_pod_ips": ["10.0.1.0"], "ssh_key_path": "/tmp/k", "framework": "sglang"})
    monkeypatch.setattr(dyn, "validate_server_args", lambda *a, **kw: (_ for _ in ()).throw(dyn.ServerArgsRejected("denied")))
    assert dyn._dynamo_restart_server(_restart_args(extra_args="--bad")) == dyn.EXIT_CONFIG_ERROR

    monkeypatch.setattr(dyn, "_dynamo_require_state", lambda: {"backend": "dynamo", "worker_pod_ips": [], "ssh_key_path": "/tmp/k"})
    monkeypatch.setattr(dyn, "_dynamo_all_gpu_ips", lambda _state: [])
    assert dyn._dynamo_apply_tracelens_patch(
        argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", poll_timeout=1)
    ) == dyn.EXIT_CONFIG_ERROR

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.setenv("HYPERLOOM_ROOT", "/root/hyperloom")
    assert dyn._resolve_geak_src(None) == "/root/hyperloom/geak"


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
    monkeypatch.setattr(mn_cli, "cmd_create_dynamo", lambda ns: 6)
    with pytest.raises(SystemExit) as exc:
        opt_mn._provision_multi_node_dynamo_stack(
            argparse.Namespace(
                nodes=2,
                rayjob_image="",
                rayjob_gpus_per_node=None,
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
            argparse.Namespace(nodes=2, mn_backend="rayjob", rayjob_image="", rayjob_gpus_per_node=None, rayjob_extra_env=[])
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
        argparse.Namespace(nodes=2, mn_backend="rayjob", rayjob_image="ray:tag", rayjob_gpus_per_node=8, rayjob_extra_env=[])
    )


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
