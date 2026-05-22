"""Regression tests for ``_grid_runner._write_variant_abort_marker``.

Pins the silent-abort fix:

* The helper writes an ``abort_reason.json`` marker in the variant slot
  dir when grid_runner catches a failure before benchmark_report.json
  is produced, so a session reader cannot confuse "tested-but-failed"
  with "untested".
* The marker payload carries the variant name, an error_class label
  (one of yaml_build_error / mn_server_restart_failed / magpie_timeout
  / no_benchmark_workspace / magpie_nonzero_invalid_measurement /
  benchmark_report_missing / benchmark_report_invalid_metric), the
  truncated error summary, the variant's extra_sglang_args, and a UTC
  abort timestamp.
* Failure to write the marker (e.g. read-only fs) is non-fatal: helper
  logs at WARN and returns silently so the grid keeps making progress.

Why these matter: silent abort of a variant (e.g. max_num_seqs_128
burning 30 min of inference cluster time after sglang launch-poll
timed out) used to leave the bare ``config.yaml`` on disk with no
``log.warning`` in the main process log and no entry in
``SharedState.params_attempts``. Final-report counted the variant as
"untested" instead of "tested-but-failed". This module pins both
properties so future refactors of the catch sites cannot regress.
"""

from __future__ import annotations

import json

import pytest

from inference_optimizer.orchestrator.action_executors import _grid_runner


def _read_marker(slot):
    """Convenience: load abort_reason.json from a variant slot dir."""
    marker_path = slot / "abort_reason.json"
    assert marker_path.exists(), f"marker not written at {marker_path}"
    return json.loads(marker_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# helper basic behavior
# ---------------------------------------------------------------------------
def test_write_variant_abort_marker_creates_file_with_expected_fields(tmp_path):
    slot = tmp_path / "variant_00_max_num_seqs_128"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="max_num_seqs_128",
        error_class="mn_server_restart_failed",
        error_summary=(
            "server /health did not return 200 within 1800s "
            "(url=http://10.245.131.67:8888/health, last_err=...)"
        ),
        extra_args="--max-num-seqs 128",
    )
    marker = _read_marker(slot)
    assert marker["variant"] == "max_num_seqs_128"
    assert marker["error_class"] == "mn_server_restart_failed"
    assert "server /health did not return 200" in marker["error"]
    assert marker["extra_args"] == "--max-num-seqs 128"
    assert marker["aborted_at_utc"].endswith("Z")
    # ISO 8601 UTC: YYYY-MM-DDTHH:MM:SSZ - 20 chars
    assert len(marker["aborted_at_utc"]) == 20


def test_write_variant_abort_marker_truncates_huge_error_summary(tmp_path):
    """Cap the embedded error blob so a long stderr dump can't bloat the
    marker (and the breakdown JSON that aggregates it) past sensible
    limits. The 2000-char cap matches the same window other VariantResult
    error fields use when capturing stderr tails."""
    slot = tmp_path / "variant"
    huge = "x" * 5000
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="big",
        error_class="magpie_timeout",
        error_summary=huge,
    )
    marker = _read_marker(slot)
    assert len(marker["error"]) == 2000


def test_write_variant_abort_marker_creates_parent_dirs(tmp_path):
    """Slot dir might not pre-exist when restart fails before yaml build
    materializes it (e.g. early ServerRestartFailed). The helper must
    mkdir -p its way to a writable parent so the marker still lands."""
    slot = tmp_path / "deeper" / "than" / "expected" / "variant"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="x",
        error_class="yaml_build_error",
        error_summary="config.yaml unwritable",
    )
    assert (slot / "abort_reason.json").exists()


def test_write_variant_abort_marker_swallows_oserror(monkeypatch, tmp_path, caplog):
    """A full-disk / permissions failure on marker write must not raise
    - the grid runner should keep iterating remaining variants."""
    slot = tmp_path / "variant"
    slot.mkdir()

    def boom(*_args, **_kwargs):
        raise OSError("read-only fs")

    monkeypatch.setattr(_grid_runner.Path, "write_text", boom)
    with caplog.at_level("WARNING"):
        _grid_runner._write_variant_abort_marker(
            slot,
            variant_name="x",
            error_class="mn_server_restart_failed",
            error_summary="oops",
        )
    assert any(
        "failed to write abort_reason.json" in r.message
        for r in caplog.records
    )


def test_write_variant_abort_marker_json_is_stable_sorted(tmp_path):
    """sort_keys=True so two reads of the same logical marker hash the
    same - useful for breakdown deduplication / regression diffing."""
    slot = tmp_path / "variant"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="v",
        error_class="ec",
        error_summary="msg",
    )
    raw = (slot / "abort_reason.json").read_text(encoding="utf-8")
    # Keys appear in sorted order, not insertion order
    assert (
        raw.index('"aborted_at_utc"')
        < raw.index('"error"')
        < raw.index('"error_class"')
        < raw.index('"extra_args"')
        < raw.index('"variant"')
    )


# ---------------------------------------------------------------------------
# 5 catch-site error_class labels are all spelled the way the helper
# tests / final-report renderer expect. Pin them with a lightweight
# string-level check against the source file so a future rename of any
# label gets caught.
# ---------------------------------------------------------------------------
def test_grid_runner_emits_expected_error_class_labels():
    """The 5 failure catch sites in run_grid must use a stable set of
    error_class labels so the final-report renderer / SharedState
    attempts_history downstream (PR-2) can switch on them.
    """
    import inspect
    src = inspect.getsource(_grid_runner)
    expected = {
        "yaml_build_error",
        "mn_server_restart_failed",
        "magpie_timeout",
        "no_benchmark_workspace",
        "magpie_nonzero_invalid_measurement",
        "benchmark_report_missing",
        "benchmark_report_invalid_metric",
    }
    missing = [label for label in expected if f'"{label}"' not in src]
    assert not missing, f"missing error_class labels in run_grid: {missing}"
