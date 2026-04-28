"""Tests for ``orchestrator.process_management`` — IMPL-CHECKLIST §1.21‒1.29."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import process_management as pm
from inference_optimizer.orchestrator.process_management import (
    FRAMEWORK_PATTERNS,
    ProcessManagementError,
    assert_user_tp_respected,
    enforce_run_baseline_sh,
    pick_filtered_trace,
    prepend_venv_path,
    safe_kill_server,
    unset_profile_envs,
    vllm_flag_translator,
    wait_kill_settle,
)


# ---------------------------------------------------------------------------
# prepend_venv_path
# ---------------------------------------------------------------------------
def test_prepend_venv_path_inserts_when_missing():
    env = {"PATH": "/usr/bin"}
    out = prepend_venv_path(env, venv_bin="/opt/venv/bin")
    assert out["PATH"].split(os.pathsep)[0] == "/opt/venv/bin"


def test_prepend_venv_path_idempotent():
    env = {"PATH": "/opt/venv/bin:/usr/bin"}
    out = prepend_venv_path(env, venv_bin="/opt/venv/bin")
    parts = out["PATH"].split(os.pathsep)
    assert parts.count("/opt/venv/bin") == 1
    assert parts[0] == "/opt/venv/bin"


def test_prepend_venv_path_does_not_mutate_input():
    env = {"PATH": "/usr/bin"}
    _ = prepend_venv_path(env, venv_bin="/opt/venv/bin")
    assert env == {"PATH": "/usr/bin"}


def test_prepend_venv_path_handles_empty_path():
    env: dict[str, str] = {}
    out = prepend_venv_path(env, venv_bin="/opt/venv/bin")
    assert out["PATH"] == "/opt/venv/bin"


# ---------------------------------------------------------------------------
# safe_kill_server (mocks out subprocess seam — IR-5 enforcement is in iron_rules)
# ---------------------------------------------------------------------------
def test_safe_kill_server_calls_pgrep_with_specific_pattern(monkeypatch):
    captured: dict[str, str] = {}

    def fake_pgrep(pattern: str) -> list[int]:
        captured["pattern"] = pattern
        return [123, 456]

    killed: list[int] = []

    def fake_kill(pid: int, *, signal_num: int = 15) -> bool:
        killed.append(pid)
        return True

    monkeypatch.setattr(pm, "_run_pgrep", fake_pgrep)
    monkeypatch.setattr(pm, "_run_kill", fake_kill)

    n = safe_kill_server("sglang")
    assert n == 2
    assert captured["pattern"] == FRAMEWORK_PATTERNS["sglang"]
    assert killed == [123, 456]
    # Guarantee we never invoke ``pkill -f sglang`` (IR-5)
    assert "pkill" not in captured["pattern"]
    assert captured["pattern"].endswith("launch_server")


def test_safe_kill_server_returns_zero_when_no_match(monkeypatch):
    monkeypatch.setattr(pm, "_run_pgrep", lambda pattern: [])
    monkeypatch.setattr(pm, "_run_kill", lambda pid, signal_num=15: True)
    assert safe_kill_server("vllm") == 0


def test_safe_kill_server_skips_dead_pids(monkeypatch):
    monkeypatch.setattr(pm, "_run_pgrep", lambda pattern: [1, 2, 3])
    # Only pid=2 still alive
    monkeypatch.setattr(
        pm, "_run_kill", lambda pid, signal_num=15: pid == 2,
    )
    assert safe_kill_server("sglang") == 1


def test_safe_kill_server_rejects_unknown_framework():
    with pytest.raises(ProcessManagementError):
        safe_kill_server("triton")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# wait_kill_settle
# ---------------------------------------------------------------------------
def test_wait_kill_settle_returns_true_when_no_framework(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(pm.time, "sleep", lambda s: sleeps.append(s))
    assert wait_kill_settle(0) is True
    assert sleeps == [0]


def test_wait_kill_settle_checks_pgrep_on_framework(monkeypatch):
    monkeypatch.setattr(pm.time, "sleep", lambda s: None)
    monkeypatch.setattr(pm, "_run_pgrep", lambda pattern: [])
    assert wait_kill_settle(0, framework="sglang") is True


def test_wait_kill_settle_reports_survivors(monkeypatch):
    monkeypatch.setattr(pm.time, "sleep", lambda s: None)
    monkeypatch.setattr(pm, "_run_pgrep", lambda pattern: [42])
    assert wait_kill_settle(0, framework="vllm") is False


# ---------------------------------------------------------------------------
# unset_profile_envs
# ---------------------------------------------------------------------------
def test_unset_profile_envs_strips_both_keys():
    env = {
        "PATH": "/usr/bin",
        "PROFILE": "1",
        "SGLANG_TORCH_PROFILER_DIR": "/tmp/p",
        "OTHER": "keep",
    }
    out = unset_profile_envs(env)
    assert "PROFILE" not in out
    assert "SGLANG_TORCH_PROFILER_DIR" not in out
    assert out["PATH"] == "/usr/bin"
    assert out["OTHER"] == "keep"
    # Original is untouched
    assert env["PROFILE"] == "1"


def test_unset_profile_envs_handles_missing_keys():
    env = {"PATH": "/usr/bin"}
    out = unset_profile_envs(env)
    assert out == {"PATH": "/usr/bin"}


# ---------------------------------------------------------------------------
# pick_filtered_trace
# ---------------------------------------------------------------------------
def test_pick_filtered_trace_finds_recursive(tmp_path: Path):
    nested = tmp_path / "deep" / "trace_run_3"
    nested.mkdir(parents=True)
    target = nested / "filtered-TP-0.trace.json.gz"
    target.write_bytes(b"x")

    found = pick_filtered_trace(tmp_path)
    assert found == target


def test_pick_filtered_trace_returns_most_recent(tmp_path: Path):
    a = tmp_path / "a" / "filtered-TP-0.trace.json.gz"
    b = tmp_path / "b" / "filtered-TP-0.trace.json.gz"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"old")
    b.write_bytes(b"new")
    # Force a's mtime older
    os.utime(a, (1_000_000, 1_000_000))
    os.utime(b, (2_000_000, 2_000_000))

    assert pick_filtered_trace(tmp_path) == b


def test_pick_filtered_trace_returns_none_when_dir_missing(tmp_path: Path):
    assert pick_filtered_trace(tmp_path / "does_not_exist") is None


def test_pick_filtered_trace_returns_none_when_no_match(tmp_path: Path):
    (tmp_path / "other.trace.json.gz").write_bytes(b"x")
    assert pick_filtered_trace(tmp_path) is None


# ---------------------------------------------------------------------------
# assert_user_tp_respected
# ---------------------------------------------------------------------------
def test_assert_user_tp_respected_passes_when_within_budget():
    assert_user_tp_respected(prompt_tp=4, detected_gpus=8)


def test_assert_user_tp_respected_rejects_overscale():
    with pytest.raises(ProcessManagementError):
        assert_user_tp_respected(prompt_tp=16, detected_gpus=8)


def test_assert_user_tp_respected_rejects_zero():
    with pytest.raises(ProcessManagementError):
        assert_user_tp_respected(prompt_tp=0, detected_gpus=4)


# ---------------------------------------------------------------------------
# vllm_flag_translator
# ---------------------------------------------------------------------------
def test_vllm_flag_translator_maps_known_flag():
    out = vllm_flag_translator(["--port", "8000", "--disable-log-requests"])
    assert out == ["--port", "8000", "--disable-log-stats"]


def test_vllm_flag_translator_passes_unknown_through():
    out = vllm_flag_translator(["--tensor-parallel-size", "4"])
    assert out == ["--tensor-parallel-size", "4"]


# ---------------------------------------------------------------------------
# enforce_run_baseline_sh
# ---------------------------------------------------------------------------
def test_enforce_run_baseline_sh_noop_for_unrelated_action(tmp_path: Path):
    # 'profile' is NOT in the required set → no script needed
    enforce_run_baseline_sh("profile", script_path=tmp_path / "missing.sh")


def test_enforce_run_baseline_sh_passes_when_script_exists(tmp_path: Path):
    p = tmp_path / "run_baseline.sh"
    p.write_text("#!/bin/bash\n", encoding="utf-8")
    enforce_run_baseline_sh("baseline", script_path=p)


def test_enforce_run_baseline_sh_raises_when_script_missing(tmp_path: Path):
    with pytest.raises(ProcessManagementError):
        enforce_run_baseline_sh("baseline", script_path=tmp_path / "nope.sh")


def test_enforce_run_baseline_sh_for_integrate(tmp_path: Path):
    p = tmp_path / "run_baseline.sh"
    p.write_text("#!/bin/bash\n", encoding="utf-8")
    enforce_run_baseline_sh("integrate", script_path=p)
