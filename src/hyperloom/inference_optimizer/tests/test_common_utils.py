# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consolidated sole-cover unit tests for common/utility modules.

Tests here cover: common.env, common.io, common.gain_math, common.llm_config,
inference_optimizer credentials, breakdown reporters, orchestrator kb_writeback,
orchestrator retry/backoff, orchestrator actions, orchestrator dispatcher,
orchestrator state/objective, multi-node state paths, framework agent helpers,
gpu_types, and CLI multi-node utilities.

Nearly every case here is still duplicated in the coverage-padding files this
one was consolidated from, all of which remain on disk:
  test_coverage_boost_unit.py, test_coverage_boost2_unit.py,
  test_coverage_gap_units.py, test_coverage_margin3_unit.py,
  test_coverage_margin_unit.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def test_reset_claude_config_leaves_file_alone_for_oauth_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """primaryApiKey is API-credits billing; with only a subscription token there
    is no key to write, so the installers' no-op behaviour applies here too.

    Path.home() is patched rather than HOME: the function returns before ever
    resolving a home directory here, so an assertion that the file is absent
    would hold even if the environment override had done nothing at all.
    """
    from hyperloom.inference_optimizer.cli import credentials

    oauth_env = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))
    monkeypatch.setenv(oauth_env, "sk-ant-oat01-fake")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    # Pre-seeded so "left alone" is observable rather than indistinguishable
    # from "was never going to be written".
    cfg_path = tmp_path / ".claude" / "config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text('{"customApiUrl": "https://operator.example"}\n', encoding="utf-8")

    credentials._reset_claude_config_to_upstream("sk-ant-oat01-fake", "https://api.anthropic.com")

    assert json.loads(cfg_path.read_text(encoding="utf-8")) == {"customApiUrl": "https://operator.example"}


def test_reset_claude_config_refuses_a_token_that_also_sits_in_the_key_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one shape where this guard is the only thing standing in the way.

    An operator who exports the same subscription token into both variables
    makes anthropic_synthesizable_key() return it, so preflight hands it in as
    the primary key and the subscription-mode check below sees a synthesizable
    key and declines to fire. Without this guard the token is persisted into
    ~/.claude/config.json, which both leaks it to disk and moves the run onto
    API billing.
    """
    from hyperloom.inference_optimizer.cli import credentials

    token = "sk-ant-oat01-same"
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), token)
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), token)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    credentials._reset_claude_config_to_upstream(token, "https://api.anthropic.com")

    payload = json.loads((tmp_path / ".claude" / "config.json").read_text(encoding="utf-8"))
    assert payload["primaryApiKey"] == "", "the subscription token must never be persisted"
    assert payload["customApiUrl"] == "https://api.anthropic.com"


def test_reset_claude_config_preserves_existing_file_for_oauth_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator's logged-in Claude config survives an oauth-only preflight."""
    from hyperloom.inference_optimizer.cli import credentials

    oauth_env = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))
    monkeypatch.setenv(oauth_env, "sk-ant-oat01-fake")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    cfg_path = tmp_path / ".claude" / "config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text('{"theme": "light", "oauthAccount": {"emailAddress": "a@b.c"}}\n', encoding="utf-8")

    credentials._reset_claude_config_to_upstream("", "https://api.anthropic.com")

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload == {"theme": "light", "oauthAccount": {"emailAddress": "a@b.c"}}


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


def test_recover_looks_complete_requires_breakdown_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.cli import recover
    import hyperloom.inference_optimizer.breakdown as breakdown_mod
    import hyperloom.orchestrator.trace.langfuse_emitter as emitter

    session = tmp_path / "session"
    session.mkdir()
    (session / "state.json").write_text('{"close_sequence_done": true}', encoding="utf-8")
    monkeypatch.setattr(
        emitter,
        "read_receipt",
        lambda _session_dir: {"counts": {"breakdown_recorded": 1}, "counts_final": True},
    )

    # A recorded breakdown that is no longer on disk must not read as complete.
    status = recover._session_recovery_status(session)
    assert status["close_done"] is True
    assert status["breakdown_recorded"] is True
    assert status["breakdown_exists"] is False
    assert status["looks_complete"] is False

    rebuilt: list[Path] = []
    monkeypatch.setattr(
        breakdown_mod,
        "write_breakdown_json",
        lambda s: rebuilt.append(s) or s / breakdown_mod.BREAKDOWN_FILENAME,
    )
    monkeypatch.setattr(breakdown_mod, "patch_breakdown_langfuse", lambda _s: None)
    monkeypatch.setattr(breakdown_mod, "package_session_artifacts", lambda _s: None)
    monkeypatch.setattr(emitter, "flush_session", lambda _s: None)
    monkeypatch.setattr(emitter, "record_session_breakdown", lambda _s: None)

    assert recover._run_recover_session(argparse.Namespace(session_dir=session, force=False, backfill_trace=False)) == 0
    assert rebuilt == [session]


# ---------------------------------------------------------------------------
# inference_optimizer.cli.multi_node / multi_node commands
# ---------------------------------------------------------------------------


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


def test_rayjob_forward_runtime_env_carries_extra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RayJob launch must ship per-round env to every rank via runtime_env.

    A shell export in the entrypoint reaches no rank (each rank is a Ray actor
    inheriting the pod env), so an omitted runtime_env silently drops knobs like
    SGLANG_USE_AITER=0 that the prompt asked for.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.delenv("HYPERLOOM_MN_EXTRA_FWD_ENV", raising=False)
    assert mn_cli._forward_runtime_env() is None

    monkeypatch.setenv(
        "HYPERLOOM_MN_EXTRA_FWD_ENV",
        json.dumps({"SGLANG_USE_AITER": "0", "LD_PRELOAD": "/evil.so"}),
    )
    payload = mn_cli._forward_runtime_env()
    assert payload == {"env_vars": {"SGLANG_USE_AITER": "0"}}, "denied keys must not reach the pods"

    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", "{bad")
    assert mn_cli._forward_runtime_env() is None
    monkeypatch.setenv("HYPERLOOM_MN_EXTRA_FWD_ENV", json.dumps(["not", "a", "dict"]))
    assert mn_cli._forward_runtime_env() is None


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
    assert (
        inf._infera_restart_config_matches(
            agg_state, argparse.Namespace(model="/m", tp=4, ep=8, extra_args="--foo 1"), "sglang", "aggregated"
        )
        is False
    )

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

    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_bash_with_env",
        lambda *a, **kw: _Completed(returncode=0, stdout="dead"),
    )
    assert inf._infera_servers_alive(state, targets, timeout=5) is False

    monkeypatch.setattr(
        inf._mn_cli,
        "_infera_ssh_bash_with_env",
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


# ---------------------------------------------------------------------------
# agents.framework helpers
# ---------------------------------------------------------------------------


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

    # An absent page is an in-band isError; get_page reports it as a miss.
    class _MissingResp(_Resp):
        def read(self, *_args):
            payload = {"result": {"isError": True, "content": [{"text": "page_not_found"}]}}
            return json.dumps(payload).encode()

    monkeypatch.setattr(gbrain.urllib.request, "urlopen", lambda req, timeout: _MissingResp())
    assert client.get_page("absent") is None
    with pytest.raises(gbrain.GbrainPageError, match="page_not_found"):
        client.call("get_page", {"slug": "absent"})

    # A transport failure is still an outage, not a miss.
    monkeypatch.setattr(gbrain.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(gbrain.GbrainPageError, match="transport error"):
        client.get_page("page-1")

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
    assert kb_writeback._default_kb_root() == tmp_path / "workspace" / "framework-kb" / "framework_optimization"


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
# orchestrator.phases.quantization_schemes — quantization prompt
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
    assert paths.workspace_root() == tmp_path / "does_not_exist"


# ---------------------------------------------------------------------------
# orchestrator.loop.dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_inline_whitelist_filters_denied_unregistered_and_lane_holding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    coord = SimpleNamespace(
        action_registry={name: object() for name in ("report", "missing", "lane_action", "ok_action")},
        sub=SimpleNamespace(executor_registry={"lane_action": object(), "ok_action": object()}),
        _INLINE_ACTION_DENY=frozenset({"report"}),
    )
    disp = DispatcherCollaborator(coord)
    monkeypatch.setattr(disp, "_registry_lanes_ttl", lambda name: (["gpu"] if name == "lane_action" else [], 60))
    # report is denied, missing has no executor, lane_action holds a lane.
    assert disp._inline_action_whitelist() == frozenset({"ok_action"})

    coord.action_registry = {}
    assert disp._inline_action_whitelist() == frozenset()


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


def test_multi_node_state_paths_resolution_and_binding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "source_state.json"))
    bound = state_paths.bind_state_file_to_session(session)
    assert bound == session / "runtime" / "multi_node_state.json"
    assert not bound.exists()
    assert state_paths.resolve_state_file() == bound
    assert bound.parent.stat().st_mode & 0o777 == 0o700


def test_multi_node_state_paths_warn_on_permission_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths

    messages: list[str] = []
    monkeypatch.setattr(state_paths, "warn", messages.append)

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
        cuda=SimpleNamespace(get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"))
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

    (tmp_path / "benchmark_report.json").write_text(json.dumps({"quality_gate": {"passed": True}}), encoding="utf-8")
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
