"""framework_integrate handler P3 real-path tests (fault matrix).

The 4-stage flow (backup -> apply -> server restart -> bench + gate)
has 4 happy + 6 failure cells covered here. Hooks are injected via
``set_integrate_hooks`` so we exercise the verdict logic without any
real server / Magpie / accuracy gate.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.framework_request_handlers import (
    ServerRestartError,
    framework_integrate_handler,
    reset_integrate_hooks,
    set_integrate_hooks,
)


@pytest.fixture(autouse=True)
def _reset_hooks():
    reset_integrate_hooks()
    yield
    reset_integrate_hooks()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _seed_source(tmp_path: Path, body: str = "x = 1\n") -> Path:
    src_dir = tmp_path / "framework_source" / "vllm" / "engine"
    src_dir.mkdir(parents=True)
    p = src_dir / "scheduler.py"
    p.write_text(body)
    return p


def _write_valid_patch(tmp_path: Path, src: Path) -> Path:
    """Write a unified diff that flips x=1 -> x=2 on ``src``."""
    rel = src.as_posix().lstrip("/")
    patch = tmp_path / "p.diff"
    patch.write_text(
        "--- a/" + rel + "\n"
        "+++ b/" + rel + "\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    return patch


def _ok_server(*a, **k) -> None:
    return None


def _ok_bench(*a, **k) -> dict[str, Any]:
    return {"ok": True, "tput": 110.0}


def _ok_accuracy(*a, **k) -> float:
    return 0.895


_GIT_OR_PATCH_AVAILABLE = (
    shutil.which("git") is not None or shutil.which("patch") is not None
)
skip_no_patch_tool = pytest.mark.skipif(
    not _GIT_OR_PATCH_AVAILABLE,
    reason="needs git or patch on PATH",
)


# ---------------------------------------------------------------------------
# P1 mock branch (no hooks injected) -- still works
# ---------------------------------------------------------------------------
def test_handler_returns_p1_mock_keep_when_no_hooks(tmp_path: Path):
    result = _run(framework_integrate_handler(
        {"patch_id": "fw-mock-1"},
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateSuccess"
    assert result["verdict"] == "KEEP"
    assert result["patch_id"] == "fw-mock-1"


# ---------------------------------------------------------------------------
# Happy path -- KEEP verdict
# ---------------------------------------------------------------------------
@skip_no_patch_tool
def test_handler_keep_verdict_when_all_gates_pass(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)
    set_integrate_hooks(
        server_restart=_ok_server,
        bench=_ok_bench,
        accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-p3-keep",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateSuccess", (
        f"got failure: {result.get('reason')!r} -- {result.get('detail')!r}"
    )
    assert result["verdict"] == "KEEP"
    assert result["tput_after"] == 110.0
    assert result["gain_pct"] == pytest.approx(10.0, abs=0.01)
    # File stays patched.
    assert "x = 2" in src.read_text()


@skip_no_patch_tool
def test_handler_revert_rolls_back_when_gain_below_threshold(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)
    set_integrate_hooks(
        server_restart=_ok_server,
        bench=lambda *a, **k: {"ok": True, "tput": 101.0},  # 1% gain, below 3% threshold
        accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-p3-revert-gain",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateSuccess"
    assert result["verdict"] == "REVERT"
    # File restored.
    assert src.read_text() == "x = 1\n"


@skip_no_patch_tool
def test_handler_revert_rolls_back_when_accuracy_drop_too_large(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)
    set_integrate_hooks(
        server_restart=_ok_server,
        bench=_ok_bench,
        accuracy_gate=lambda *a, **k: 0.85,  # 5.5% drop
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-p3-revert-acc",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["verdict"] == "REVERT"
    assert src.read_text() == "x = 1\n"


@skip_no_patch_tool
def test_handler_needs_review_when_accuracy_missing(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)
    set_integrate_hooks(
        server_restart=_ok_server,
        bench=_ok_bench,
        accuracy_gate=lambda *a, **k: None,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-p3-review",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["verdict"] == "NEEDS_REVIEW"
    # NEEDS_REVIEW does NOT rollback -- operator decides.
    assert "x = 2" in src.read_text()


# ---------------------------------------------------------------------------
# Failure cells
# ---------------------------------------------------------------------------
def test_handler_failure_when_patch_path_missing(tmp_path: Path):
    set_integrate_hooks(
        server_restart=_ok_server, bench=_ok_bench, accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {"patch_id": "fw-x"},
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    assert result["reason"] == "missing_patch_path"


def test_handler_failure_when_patch_file_not_found(tmp_path: Path):
    set_integrate_hooks(
        server_restart=_ok_server, bench=_ok_bench, accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {"patch_id": "fw-x", "patch_path": str(tmp_path / "nope.diff")},
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    assert result["reason"] == "patch_not_found"


def test_handler_failure_when_patch_empty(tmp_path: Path):
    set_integrate_hooks(
        server_restart=_ok_server, bench=_ok_bench, accuracy_gate=_ok_accuracy,
    )
    empty = tmp_path / "empty.diff"
    empty.write_text("not a real diff")
    result = _run(framework_integrate_handler(
        {"patch_id": "fw-x", "patch_path": str(empty)},
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    assert result["reason"] == "patch_empty"


@skip_no_patch_tool
def test_handler_failure_when_server_restart_raises(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)

    def crash(*a, **k):
        raise ServerRestartError("server failed to bind port")

    set_integrate_hooks(
        server_restart=crash, bench=_ok_bench, accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-srv-fail",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    assert result["reason"] == "server_restart_failed"
    # Rolled back: file restored.
    assert src.read_text() == "x = 1\n"


@skip_no_patch_tool
def test_handler_failure_when_bench_hook_raises(tmp_path: Path):
    src = _seed_source(tmp_path)
    patch = _write_valid_patch(tmp_path, src)

    def crash(*a, **k):
        raise RuntimeError("magpie timeout")

    set_integrate_hooks(
        server_restart=_ok_server, bench=crash, accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-bench-fail",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    assert result["reason"] == "bench_failed"
    # Rolled back.
    assert src.read_text() == "x = 1\n"


def test_handler_failure_when_apply_fails(tmp_path: Path):
    """A diff that doesn't match the source -> patch_apply_failed; no
    rollback needed because apply ran *after* backup but mutated
    nothing on disk."""
    src = _seed_source(tmp_path, body="x = 1\n")
    # Write a diff whose context doesn't match.
    rel = src.as_posix().lstrip("/")
    patch = tmp_path / "bad.diff"
    patch.write_text(
        "--- a/" + rel + "\n"
        "+++ b/" + rel + "\n"
        "@@ -1 +1 @@\n"
        "-y = 9\n"
        "+y = 10\n"
    )
    set_integrate_hooks(
        server_restart=_ok_server, bench=_ok_bench, accuracy_gate=_ok_accuracy,
    )
    result = _run(framework_integrate_handler(
        {
            "patch_id": "fw-apply-fail",
            "patch_path": str(patch),
            "baseline_tput": 100.0,
            "baseline_accuracy": 0.9,
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "IntegrateFailure"
    if _GIT_OR_PATCH_AVAILABLE:
        assert result["reason"] == "patch_apply_failed"
    # Original file untouched.
    assert src.read_text() == "x = 1\n"
