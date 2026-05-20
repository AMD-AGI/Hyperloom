"""Smoke tests for ``fa agent`` subcommand (P2 PR-D)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_fa(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m framework_agent.runtime.cli ARGS``."""
    return subprocess.run(
        [sys.executable, "-m", "framework_agent.runtime.cli", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_fa_agent_help_rc0():
    proc = _run_fa("agent", "--help")
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert "prepare-task" in proc.stdout
    assert "commit-result" in proc.stdout


def test_fa_agent_prepare_task_help_rc0():
    proc = _run_fa("agent", "prepare-task", "--help")
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert "--task" in proc.stdout
    assert "--output-bundle" in proc.stdout


def test_fa_agent_commit_result_help_rc0():
    proc = _run_fa("agent", "commit-result", "--help")
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert "--envelope" in proc.stdout
    assert "--task-id" in proc.stdout
    assert "--session-dir" in proc.stdout


def test_prepare_task_emits_bundle_json(tmp_path: Path):
    task = {
        "task_id": "fw-test-001",
        "kind": "framework_optimize",
        "session_dir": str(tmp_path / "session"),
        "target_framework": "sglang",
        "ast_scan_enabled": True,
        "ast_frameworks": ["sglang"],
        "kb_partition": "framework_optimization",
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    bundle_path = tmp_path / "bundle.json"

    proc = _run_fa(
        "agent", "prepare-task",
        "--task", str(task_path),
        "--output-bundle", str(bundle_path),
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert bundle_path.is_file()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["bundle_version"] == "1"
    assert bundle["task"]["task_id"] == "fw-test-001"
    assert bundle["task"]["kind"] == "framework_optimize"
    assert "prepared_at_ms" in bundle


def test_prepare_task_rejects_unknown_kind(tmp_path: Path):
    task = {
        "task_id": "fw-test-002",
        "kind": "unknown_kind",
        "session_dir": str(tmp_path),
        "target_framework": "vllm",
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")

    proc = _run_fa("agent", "prepare-task", "--task", str(task_path))
    assert proc.returncode == 2
    assert "unknown_kind" in proc.stderr or "kind" in proc.stderr


def test_commit_result_validates_and_persists(tmp_path: Path):
    envelope = {
        "payload_kind": "OptimizeSuccess",
        "patch_path": "/sess/runs/framework/fw-test/proposal.diff",
        "predicted_gain_pct": 7.5,
        "rationale": "smoke test",
        "stage_a_elapsed_ms": 100,
    }
    env_path = tmp_path / "envelope.json"
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    sd = tmp_path / "session"
    sd.mkdir()

    proc = _run_fa(
        "agent", "commit-result",
        "--envelope", str(env_path),
        "--task-id", "fw-test-001",
        "--session-dir", str(sd),
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    persisted = sd / "runs" / "framework" / "fw-test-001" / "envelope.json"
    assert persisted.is_file()
    loaded = json.loads(persisted.read_text(encoding="utf-8"))
    assert loaded["payload_kind"] == "OptimizeSuccess"


def test_commit_result_rejects_invalid_envelope(tmp_path: Path):
    """Missing required field -> rc=2."""
    bad = {
        "payload_kind": "OptimizeSuccess",
        "patch_path": "/x",
        # missing predicted_gain_pct / rationale / stage_a_elapsed_ms
    }
    env_path = tmp_path / "bad.json"
    env_path.write_text(json.dumps(bad), encoding="utf-8")
    proc = _run_fa("agent", "commit-result", "--envelope", str(env_path))
    assert proc.returncode == 2
    assert "envelope invalid" in proc.stderr or "failed schema" in proc.stderr
