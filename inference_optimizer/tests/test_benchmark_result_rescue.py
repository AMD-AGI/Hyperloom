"""Rescue-path salvage tests for :func:`extract_benchmark_measurement`.

The Magpie ``dsr1_fp8_mi300x.sh`` script hardcodes
``--result-dir /workspace/`` so a benchmark that *numerically* succeeds
can still leave the per-task workspace empty (no ``inferencex_result.json``).
The optimizer's second-chance salvage:

* honours ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` (files or dirs).
* gates by ``subprocess_started_unix`` mtime so stale leaks from a
  previous run cannot be misattributed to this run.
* tags the adopted path in the ``nonfatal_warnings`` list as
  ``rescued_from_leaked_path:<path>``.
* COPIES the leaked file into the task workspace (best-effort) so the
  canonical NFS-clone of ``<session>/runs/<action>/<task_id>/`` is
  self-contained and ``raw_result_path`` points at the in-workspace
  copy rather than the leak location.

This module exercises those four behaviours plus the negative case
(rescue still finds nothing → ``valid_measurement`` stays False).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.benchmark_result import (
    extract_benchmark_measurement,
)


def _write_inferencex(path: Path, tput: float = 1761.6, completed: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "output_throughput": tput,
                "request_throughput": tput / 10,
                "completed_requests": completed,
                "duration_seconds": 120.0,
            }
        ),
        encoding="utf-8",
    )


def test_rescue_from_env_path_adopted_after_subprocess_start(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_path = leak_dir / "inferencex_result.json"
    _write_inferencex(leak_path)
    # mtime cutoff strictly before the file's mtime
    cutoff = leak_path.stat().st_mtime - 5.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    measurement = extract_benchmark_measurement(
        report={"success": False},  # wrapper reported failure
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(1761.6)
    assert measurement["completed_requests"] == 640
    assert any(
        w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"]
    )
    # Salvage now copies the leaked file into the workspace so the
    # NFS clone is self-contained. ``raw_result_path`` advertises the
    # in-workspace copy; the original leak path remains untouched
    # (we don't delete it — only copy) and is captured in the warning
    # tag for audit.
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)
    assert json.loads(copied.read_text())["output_throughput"] == 1761.6
    assert leak_path.exists()
    assert any(
        w == f"rescued_from_leaked_path:{leak_path}"
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_rejects_stale_leak_with_older_mtime(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    # subprocess "started" well *after* the leak file's mtime → stale
    cutoff = leak_path.stat().st_mtime + 10.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is False
    assert measurement.get("output_throughput") is None
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_skips_when_in_workspace_salvage_succeeded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    in_ws = workspace / "inferencex_result.json"
    _write_inferencex(in_ws, tput=2000.0)
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=9999.0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": True},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 5.0,
    )
    # in-workspace value wins; rescue path should not have been used.
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(2000.0)
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_no_candidates_keeps_invalid_measurement(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # explicit empty env to avoid the default /workspace/ path from picking up
    # something under the test sandbox.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(tmp_path / "nope"))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=0.0,
    )
    assert measurement["valid_measurement"] is False
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_directory_scanned_for_inferencex_result_glob(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    # Two leaked files; the helper sorts via Path.glob() then mtime-gates.
    a = leak_dir / "inferencex_result_eval.json"
    b = leak_dir / "inferencex_result.json"
    _write_inferencex(a, tput=100.0)
    _write_inferencex(b, tput=200.0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=min(a.stat().st_mtime, b.stat().st_mtime) - 1.0,
    )
    assert measurement["valid_measurement"] is True
    # Any of the two valid files is acceptable; assert at least one of them
    # produced a positive throughput and the salvage tag is present.
    assert measurement["output_throughput"] in (100.0, 200.0)
    assert any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_subprocess_started_unix_none_disables_mtime_gate(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        # No subprocess_started_unix → mtime gate disabled → adopt.
    )
    assert measurement["valid_measurement"] is True
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)


def test_rescue_copies_leaked_file_into_workspace(tmp_path, monkeypatch):
    """Salvage must materialise the rescued file inside the workspace so
    downstream NFS clones of ``<session>/runs/<action>/<task_id>/`` carry
    the canonical artifact instead of a path pointing at ``/workspace/``.
    """
    workspace = tmp_path / "session" / "runs" / "baseline" / "task-7" / "bench_run"
    workspace.mkdir(parents=True)
    leak_path = tmp_path / "workspace_leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=1234.5, completed=500)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 2.0,
    )
    copied = workspace / leak_path.name
    assert copied.exists(), (
        "salvage must physically copy the leaked file into the workspace "
        "so the NFS-clone of the task dir contains the InferenceX result"
    )
    body = json.loads(copied.read_text())
    assert body["output_throughput"] == 1234.5
    assert body["completed_requests"] == 500
    # raw_result_path advertises the in-workspace copy (canonical
    # destination); the original leak path is preserved in the warning
    # tag so audits still see where the file came from.
    assert measurement["raw_result_path"] == str(copied)
    assert any(
        w == f"rescued_from_leaked_path:{leak_path}"
        for w in measurement["nonfatal_warnings"]
    )
    # The leak file itself is not touched (we copy, never move/delete).
    assert leak_path.exists()


def test_rescue_copy_failure_falls_back_to_leak_path(
    tmp_path, monkeypatch,
):
    """If the copy step fails (permission denied / disk full), salvage
    must still advertise the leaked measurement so a usable baseline
    isn't discarded just because the artifact couldn't be materialised
    into the workspace.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=999.9, completed=42)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    # Force the copy helper to fail so we exercise the fallback branch.
    from inference_optimizer.orchestrator.action_executors import (
        benchmark_result as bench_result_mod,
    )
    monkeypatch.setattr(
        bench_result_mod,
        "_materialize_rescue_into_workspace",
        lambda *_a, **_k: None,
    )

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 1.0,
    )
    # Measurement is still valid (we don't discard real numbers because
    # of a copy failure) and raw_result_path falls back to the leak.
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(999.9)
    assert measurement["raw_result_path"] == str(leak_path)
    warnings = measurement["nonfatal_warnings"]
    assert any(w.startswith("rescued_from_leaked_path:") for w in warnings)
    assert any(
        w.startswith("rescued_copy_into_workspace_failed:")
        for w in warnings
    )
