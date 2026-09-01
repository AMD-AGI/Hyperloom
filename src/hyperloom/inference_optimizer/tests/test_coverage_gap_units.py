# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
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


def test_credentials_endpoint_resolution_and_geak_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import credentials

    # Each side resolves on its own; an unconfigured side stays empty.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://open.example/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert credentials._resolve_llm_endpoints() == ("", "https://open.example/v1")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "https://open.example/v1")

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert credentials._resolve_llm_endpoints() == ("https://anthropic.example", "")

    cfg = tmp_path / "geak.yaml"
    cfg.write_text("model: x\n  base_url: https://old/v1\n", encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is True
    assert r"https://new.example/\g/v1" in cfg.read_text(encoding="utf-8")
    assert credentials._sync_geak_config_base_url(str(cfg), r"https://new.example/\g/v1") is False
    assert credentials._sync_geak_config_base_url(str(tmp_path / "missing.yaml"), "https://x") is False


def test_credentials_validate_and_reset_claude_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import credentials

    for key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        credentials._validate_credentials()
    assert exc.value.code == 2

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    credentials._validate_credentials()

    monkeypatch.setenv("HOME", str(tmp_path))
    credentials._reset_claude_config_to_upstream("anthropic-test-key", "https://anthropic.example")
    cfg_path = tmp_path / ".claude" / "config.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["primaryApiKey"] == "anthropic-test-key"
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
    monkeypatch.setattr(
        breakdown_mod,
        "write_breakdown_json",
        lambda s: calls.append("write") or s / breakdown_mod.BREAKDOWN_FILENAME,
    )
    monkeypatch.setattr(breakdown_mod, "patch_breakdown_langfuse", lambda s: calls.append("patch"))
    monkeypatch.setattr(
        breakdown_mod, "package_session_artifacts", lambda s: calls.append("package") or s / "bundle.zip"
    )
    monkeypatch.setattr(emitter, "flush_session", lambda s: calls.append("flush"))
    monkeypatch.setattr(emitter, "record_session_breakdown", lambda s: calls.append("record"))
    rc = recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=False))
    assert rc == 0
    assert calls == ["write", "flush", "patch", "record", "package"]

    monkeypatch.setattr(breakdown_mod, "write_breakdown_json", lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=False)) == 1


def test_cli_multi_node_gc_backend_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import multi_node as mn

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


def _patch_infera_state(monkeypatch: pytest.MonkeyPatch, state: dict) -> list[dict]:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    saved: list[dict] = []
    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda payload: saved.append(dict(payload)))
    return saved


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
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.setenv(k, f"secret-{k}")
    fwd = inf._collect_forward_env()
    assert fwd["MORI_FOO"] == "override"
    assert fwd["SGLANG_TORCH_PROFILER_DIR"] == "/shared/traces"
    assert fwd["SGLANG_USE_AITER"] == "1"
    assert fwd["SGLANG_MORI_BAR"] == "explicit"
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
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


def test_pd_restart_against_an_aggregated_state_fails_instead_of_reporting_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart that touches nothing must not report that it launched servers.

    ``--pd-mode disaggregated`` is honoured even when the state's own pd_mode is
    aggregated, but the pod lists are chosen by the state alone, so both legs
    resolved empty. Each was then skipped, ``rc_total`` stayed 0, and the
    command printed "infera servers launched" and exited 0 without opening a
    single SSH connection -- after which the round benchmarked whatever was
    already running and recorded it as this candidate's result.
    """
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "framework": "sglang",
        "nodes": 2,
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
        "ssh_key_path": "/tmp/k",
        "ssh_port": 2222,
    }
    fanout_calls: list[str] = []
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    monkeypatch.setattr(inf._mn_cli, "_save_state", lambda _payload: None)
    monkeypatch.setattr(
        inf,
        "_infera_fanout_launch",
        lambda st, args, targets, **kw: fanout_calls.append(kw["label"]) or (0, [{"ok": True}]),
    )

    with pytest.raises(inf._mn_cli.ConfigurationError, match="no prefill or decode pods"):
        inf._infera_restart_server(_restart_args(pd_mode="disaggregated"))

    assert fanout_calls == []


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"backend": "rayjob"}, "backend is not 'infera'"),
        ({"backend": "infera", "pd_mode": "aggregated"}, "no GPU pod IPs"),
        ({"backend": "infera", "worker_pod_ips": ["10.0.1.0"]}, "no ssh_key_path"),
    ],
)
def test_infera_state_errors_are_config_errors_not_transient(
    monkeypatch: pytest.MonkeyPatch,
    state: dict,
    match: str,
) -> None:
    """Rerunning cannot supply an SSH key, so these must not read as retryable.

    ``main`` classifies a bare RuntimeError by message substring and none of
    these matched, so all three fell through to EXIT_TRANSIENT -- which the
    controller is told means "rerun the same subcommand". A permanently
    misconfigured hand-off was retried forever. ConfigurationError is matched by
    type instead, which is what it exists for.
    """
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setattr(inf._mn_cli, "_load_state", lambda: dict(state))

    with pytest.raises(inf._mn_cli.ConfigurationError, match=match):
        inf._infera_require_state()


def test_missing_gpu_pods_names_the_mode_that_chose_the_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message has to explain why IPs that ARE set look absent.

    A PD hand-off invoked without ``PD_MODE=disaggregated`` synthesizes as
    aggregated, so only ``_WORKER_IPS`` is consulted and the error read "check
    PREFILL / DECODE / WORKER" while both of those were in fact set.
    """
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setattr(
        inf._mn_cli,
        "_load_state",
        lambda: {"backend": "infera", "pd_mode": "aggregated", "prefill_pod_ips": ["10.0.2.0"]},
    )

    with pytest.raises(inf._mn_cli.ConfigurationError) as excinfo:
        inf._infera_require_state()

    message = str(excinfo.value)
    assert "pd_mode='aggregated'" in message
    assert "HYPERLOOM_MN_EXT_WORKER_IPS" in message
    assert "PD_MODE=disaggregated" in message


def test_resolve_pd_node_counts_infers_from_pod_lists() -> None:
    """PD node counts fall back to discovered pod-list lengths when CLI is unset."""
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    args = argparse.Namespace(pd_prefill_nodes=0, pd_decode_nodes=0)
    state = {"prefill_pod_ips": ["10.0.1.1", "10.0.1.2"], "decode_pod_ips": ["10.0.2.1"]}
    assert inf._resolve_pd_node_counts(args, state) == (2, 1)
    assert inf._resolve_pd_node_counts(
        argparse.Namespace(pd_prefill_nodes=3, pd_decode_nodes=0),
        state,
    ) == (3, 1)


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
        lambda st, args, targets, **kw: (
            fanout_calls.append((kw["label"], [t["podIP"] for t in targets])) or (0, [{"ok": True}])
        ),
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
    assert (
        inf._infera_restart_server(
            _restart_args(
                pd_prefill_nodes=1,
                pd_decode_nodes=1,
                pd_prefill_tp=8,
                pd_decode_tp=4,
                pd_prefill_extra_args="--prefill",
                pd_decode_extra_args="--decode",
            )
        )
        == 0
    )
    assert [call[0] for call in fanout_calls[-2:]] == ["restart-prefill", "restart-decode"]
    assert saved[-1]["last_restart_pd_decode_tp"] == 4
    assert saved[-1]["last_restart_pd_prefill_nodes"] == 1
    assert saved[-1]["last_restart_pd_decode_nodes"] == 1

    # CLI omits pd_*_nodes (0): infer from pod lists and persist inferred counts.
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
            ({"status": "restored"}, {"rc": 0, "stderr": ""}),
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

    assert (
        inf._infera_revert_patch(argparse.Namespace(backup_map_json="{", target_path="/x", timeout_sec=1))
        == inf.EXIT_CONFIG_ERROR
    )
    assert (
        inf._infera_revert_patch(argparse.Namespace(backup_map_json="{}", target_path="/x", timeout_sec=1))
        == inf.EXIT_CONFIG_ERROR
    )
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: ({"status": "restored"}, {"rc": 0, "stderr": ""}))
    assert (
        inf._infera_revert_patch(
            argparse.Namespace(backup_map_json=json.dumps({"10.0.1.0": "/b"}), target_path="/x", timeout_sec=1)
        )
        == 0
    )

    assert (
        inf._infera_kernel_bench(
            argparse.Namespace(
                workspace="/w",
                bench_command="true",
                files_b64_json="{bad",
                result_glob="*.json",
                timeout_sec=10,
                print_logs=False,
            )
        )
        == inf.EXIT_CONFIG_ERROR
    )
    monkeypatch.setattr(inf, "_infera_ssh_node_op", lambda *a, **kw: (None, {"rc": 1, "stderr": "no json"}))
    assert (
        inf._infera_kernel_bench(
            argparse.Namespace(
                workspace="/w",
                bench_command="true",
                files_b64_json="{}",
                result_glob="*.json",
                timeout_sec=10,
                print_logs=True,
            )
        )
        == inf.EXIT_TRANSIENT
    )
    monkeypatch.setattr(
        inf, "_infera_ssh_node_op", lambda *a, **kw: ({"status": "ok", "result": 1}, {"rc": 0, "stderr": ""})
    )
    assert (
        inf._infera_kernel_bench(
            argparse.Namespace(
                workspace="/w",
                bench_command="true",
                files_b64_json="{}",
                result_glob="*.json",
                timeout_sec=10,
                print_logs=False,
            )
        )
        == 0
    )


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
        lambda st, ip, script, python, op_args, **kw: (
            calls.append((ip, python, op_args))
            or _Completed(returncode=0, stdout='{"status":"applied","per_pod":[{"status":"applied"}]}', stderr="")
        ),
    )
    assert (
        inf._infera_apply_tracelens_patch(
            argparse.Namespace(tracelens_root="/tracelens", sglang_version_pin="v1", poll_timeout=5)
        )
        == 0
    )
    assert calls[0][1] == "/opt/venv/bin/python"
    assert "--sglang-version-pin v1" in calls[0][2]

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert (
        inf.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=False)) == inf.EXIT_CONFIG_ERROR
    )
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: _Completed(returncode=0, stdout='{"status":"installed"}', stderr=""),
    )
    assert inf.cmd_install_geak(argparse.Namespace(geak_src="", poll_timeout=5, print_logs=True)) == 0
    assert saved == []


def test_infera_process_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    state = {
        "backend": "infera",
        "pd_mode": "aggregated",
        "worker_pod_ips": ["10.0.1.0"],
        "ssh_key_path": "/tmp/k",
    }
    monkeypatch.setattr(inf, "_infera_require_state", lambda: dict(state))
    assert (
        inf._infera_apply_tracelens_patch(argparse.Namespace(tracelens_root="", sglang_version_pin="", poll_timeout=1))
        == inf.EXIT_CONFIG_ERROR
    )

    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: f"script:{name}")
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 1)
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_run_script",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=1)),
    )
    assert (
        inf._infera_apply_tracelens_patch(
            argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", poll_timeout=1)
        )
        == 1
    )


def test_infera_restart_config_and_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    # No prior launch recorded -> never a match.
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
    # A changed served flag breaks the match.
    assert (
        inf._infera_restart_config_matches(
            agg_state, argparse.Namespace(model="/m", tp=4, ep=8, extra_args="--foo 1"), "sglang", "aggregated"
        )
        is False
    )

    # Disaggregated PD topology match (node counts inferred from pod lists).
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
        model="/m",
        tp=8,
        ep=8,
        extra_args="",
        pd_prefill_nodes=1,
        pd_decode_nodes=1,
        pd_prefill_tp=0,
        pd_decode_tp=0,
        pd_prefill_ep=0,
        pd_decode_ep=0,
        pd_prefill_extra_args="",
        pd_decode_extra_args="",
    )
    assert inf._infera_restart_config_matches(pd_state, pd_args, "sglang", "disaggregated") is True

    # _infera_servers_alive: empty targets -> False.
    assert inf._infera_servers_alive({}, [], timeout=5) is False

    state = {"ssh_key_path": "/tmp/k"}
    targets = [{"podIP": "10.0.0.1", "sshPort": 2222}]
    monkeypatch.setattr(inf._mn_cli, "_infera_default_ssh_port", lambda st: 2222)
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_bash_with_env",
        lambda *a, **kw: _Completed(returncode=0, stdout="MN_ALIVE\n"),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is True

    # Dead pod (no MN_ALIVE marker) -> False.
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_bash_with_env",
        lambda *a, **kw: _Completed(returncode=0, stdout="dead"),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is False

    # SSH timeout -> False.
    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_bash_with_env",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["ssh"], timeout=1)),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is False

    # A target missing podIP -> False.
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
        framework="sglang",
        model="/m",
        tp=8,
        ep=8,
        extra_args="",
        pd_mode="",
        pd_transfer_backend="",
        print_logs=False,
        pd_prefill_extra_args="",
        pd_decode_extra_args="",
    )
    assert inf._infera_restart_server(args) == 0
    out = capsys.readouterr().out
    assert '"resumed": true' in out


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
    from hyperloom.common import jsonio

    assert list(jsonio.iter_sse_objects('not json\n\ndata: {bad}\n\ndata: {"id":"1","result":{"ok":true}}\n\n')) == [
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
    (manifests / "skip_fields" / "manifest.json").write_text(
        json.dumps({"status": "applied", "multinode": {"backup_dir_on_pod": "/b"}}), encoding="utf-8"
    )
    (manifests / "missing_patch" / "manifest.json").parent.mkdir()
    (manifests / "missing_patch" / "manifest.json").write_text(
        json.dumps(
            {
                "status": "applied",
                "multinode": {"backup_dir_on_pod": "/b"},
                "target_file": "/x",
                "patch_path": str(tmp_path / "missing.diff"),
            }
        ),
        encoding="utf-8",
    )
    patch = tmp_path / "p.diff"
    patch.write_text("diff", encoding="utf-8")
    (manifests / "failed" / "manifest.json").parent.mkdir()
    (manifests / "failed" / "manifest.json").write_text(
        json.dumps(
            {
                "status": "applied",
                "multinode": {"backup_dir_on_pod": "/b"},
                "target_file": "/x",
                "patch_path": str(patch),
                "kernel_id": "k",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mn.subprocess, "run", lambda *a, **kw: _Completed(returncode=5, stderr="failed patch"))
    mn._replay_kernel_patches_for_multi_node(argparse.Namespace(nodes=2))


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


def test_server_lifecycle_remaining_resolution_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
    from hyperloom.orchestrator.actions.executors import benchmark_backend as bb

    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("benchmark: [", encoding="utf-8")
    info = sl.resolve_lifecycle_params(bad_yaml)
    assert info["eligible"] is False
    assert "could not read" in info["reason"]

    invalid_port = tmp_path / "invalid-port.yaml"
    invalid_port.write_text(
        yaml.safe_dump({"benchmark": {"framework": "xdit", "envs": {"PORT": "bad"}}}),
        encoding="utf-8",
    )
    info = sl.resolve_lifecycle_params(invalid_port)
    assert info["port"] == sl.REUSE_PORT_DEFAULT
    assert "scriptable framework" in info["reason"]

    profiler = tmp_path / "profiler.yaml"
    profiler.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "benchmark_script": "vllm_mi300x.sh",
                    "envs": {"PORT": 8888},
                    "profiler": {"torch_profiler": {"enabled": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    info = sl.resolve_lifecycle_params(profiler)
    assert info["eligible"] is False
    assert "torch_profiler" in info["reason"]


def test_canonical_fingerprint_remaining_normalization_branches() -> None:
    from hyperloom.orchestrator.actions.executors import _canonical_fingerprint as fp

    with_controls = fp.canonical_fingerprint(
        '--flag "unterminated',
        {"B": 2},
        remove_args="--old",
        unset_envs=123,
        args_mode="bad-mode",
    )
    without_controls = fp.canonical_fingerprint("--flag unterminated", {"B": 2})
    assert len(with_controls) == 16
    assert len(without_controls) == 16
    assert with_controls != without_controls


def test_conc_sweep_plot_helper_series_and_payload_loading(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

    payload = {
        "baseline": {
            "points": [
                {"conc": 4, "output_throughput": 800},
                {"conc": 2, "output_throughput": "600"},
                {"conc": 0, "output_throughput": 100},
                {"conc": "bad", "output_throughput": 100},
                {"conc": 1, "output_throughput": None},
            ]
        },
        "roofline_ceiling": {
            "rows": [
                {"conc": 8, "t_peak_tok_s": 1600},
                {"conc": 4, "t_peak_tok_s": "1200"},
                {"conc": 0, "t_peak_tok_s": 999},
                {"conc": "bad", "t_peak_tok_s": 999},
                {"conc": 1, "t_peak_tok_s": None},
            ]
        },
    }
    payload_file = tmp_path / "conc_sweep_summary.json"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")

    assert plot._load_payload(payload) is payload
    assert plot._load_payload(payload_file) == payload

    xs, ys = plot._arm_series(payload["baseline"]["points"], tp_eff=2.0)
    assert xs == [200.0, 300.0]
    assert ys == [400.0, 300.0]
    assert plot._arm_series([{"conc": "bad", "output_throughput": -1}], tp_eff=1.0) == ([], [])

    cx, cy = plot._ceiling_series(payload["roofline_ceiling"], tp_eff=4.0)
    assert cx == [200.0, 300.0]
    assert cy == [400.0, 300.0]
    assert plot._ceiling_series({"rows": [{"conc": 0, "t_peak_tok_s": 0}]}, tp_eff=1.0) == ([], [])


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
    monkeypatch.setattr(
        breakdown_mod, "write_breakdown_json", lambda s: calls.append("write") or s / "session_breakdown.json"
    )
    monkeypatch.setattr(emitter, "flush_session", lambda _s: (_ for _ in ()).throw(RuntimeError("langfuse down")))
    monkeypatch.setattr(
        breakdown_mod, "package_session_artifacts", lambda _s: (_ for _ in ()).throw(RuntimeError("zip failed"))
    )

    fake_backfill = SimpleNamespace(
        build_plan=lambda s: calls.append("plan") or {"session": str(s)},
        ingest=lambda plan: calls.append("ingest") or 0,
    )
    monkeypatch.setitem(
        __import__("sys").modules, "hyperloom.inference_optimizer.tools.backfill_langfuse", fake_backfill
    )

    rc = recover._run_recover_session(argparse.Namespace(session_dir=session, force=True, backfill_trace=True))
    assert rc == 0
    assert calls == ["write", "plan", "ingest"]
