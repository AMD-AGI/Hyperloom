"""Roofline-v2 C1: ``record_trace_analyze`` caches TraceLens analysis.md.

These tests pin the contract the downstream ``roofline`` action (C4) and
the prompt renderer (C5) depend on:

* When the kernel handler surfaces ``trace_report_path`` in its result,
  ``SharedState.last_trace_analyze`` must contain the **full** text of
  that file under ``analysis_md_text`` (Decision A3: no truncation).
* ``analysis_md_path`` must always be a string (never ``None``) so prompt
  formatters can render it verbatim without a ``None`` guard.
* ``roofline_snapshot_id`` must monotonically increase on each call so
  re-profile guidance (C5) can detect "new snapshot available" and so
  audit (C6) can pair every roofline-action invocation with the specific
  TraceLens report it analyzed.
* ``roofline_baseline_gain_at_snapshot`` must capture
  ``cumulative_gain_validated`` **at the moment the snapshot is taken**
  so the prompt can show "gain since snapshot: current − baseline".
* Read failures (missing file, permission denied, decode error) must
  degrade silently — empty ``analysis_md_text`` signals "no report
  available" without breaking the existing trace_analyze path.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.orchestrator.shared_state import SharedState


def _payload(trace: Path) -> dict:
    return {"trace_input": str(trace)}


def _result_with_report(analysis_md: Path) -> dict:
    return {
        "hot_kernels": [],
        "trace_report_path": str(analysis_md),
        "trace_health_warnings": [],
    }


def test_caches_analysis_md_full_text(tmp_path: Path) -> None:
    """Path provided → file content cached verbatim."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text(
        "# Executive Summary\nCompute 51.6%, Idle 48.3%\n"
        "## Top Operations\n| aten::mm | 25% |\n",
        encoding="utf-8",
    )

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "trace.json"),
                                _result_with_report(analysis_md))

    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(analysis_md)
    assert "Executive Summary" in cached["analysis_md_text"]
    assert "Compute 51.6%" in cached["analysis_md_text"]
    assert "Top Operations" in cached["analysis_md_text"]


def test_missing_trace_report_path_yields_empty_text() -> None:
    """No ``trace_report_path`` field → ``analysis_md_text`` is ``""``."""
    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == ""
    assert cached["analysis_md_text"] == ""


def test_unreadable_analysis_md_degrades_silently(tmp_path: Path) -> None:
    """Path points at a nonexistent file → empty text, no exception."""
    missing = tmp_path / "does_not_exist.md"

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {
            "hot_kernels": [],
            "trace_report_path": str(missing),
            "trace_health_warnings": [],
        },
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(missing)
    assert cached["analysis_md_text"] == ""


def test_caches_large_analysis_md_without_truncation(tmp_path: Path) -> None:
    """Real-world 200 KB report (Case A-D scale) round-trips intact.

    Decision A3: no truncation. The cache must hold the full text even
    for the largest analysis.md a TraceLens run can produce, because the
    roofline action (C4) reads it once and the structured analyzer
    output replaces verbatim prompt injection from C5 onward.
    """
    analysis_md = tmp_path / "big_analysis.md"
    big_content = "# Analysis\n" + ("filler line\n" * 20000)
    analysis_md.write_text(big_content, encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {
            "hot_kernels": [],
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [],
        },
    )
    cached_text = state.last_trace_analyze["analysis_md_text"]
    assert len(cached_text) > 200_000
    assert cached_text.startswith("# Analysis\n")
    assert cached_text.endswith("filler line\n")


def test_snapshot_id_monotonically_increases(tmp_path: Path) -> None:
    """Every successful call bumps ``roofline_snapshot_id`` by exactly 1.

    The counter lives inside ``last_trace_analyze`` itself (rather than
    a new top-level SharedState field) so a stale ``last_trace_analyze``
    is reset implicitly by the next ``record_trace_analyze`` call.
    """
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("first", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "t1"),
                                _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1

    analysis_md.write_text("second", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t2"),
                                _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 2

    analysis_md.write_text("third", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t3"),
                                _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 3


def test_snapshot_id_starts_at_one_after_empty_state() -> None:
    """Fresh SharedState (empty ``last_trace_analyze``) → snapshot 1."""
    state = SharedState()
    assert state.last_trace_analyze == {}
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1


def test_baseline_gain_captured_at_snapshot_time() -> None:
    """``roofline_baseline_gain_at_snapshot`` reflects cumulative_gain
    at the **moment** the snapshot is recorded.

    Later mutations of ``cumulative_gain_validated`` must not retroact
    onto the cached snapshot — the prompt's "gain since snapshot" delta
    depends on this being a true point-in-time capture.
    """
    state = SharedState()
    state.cumulative_gain_validated = 0.0
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 0.0

    state.cumulative_gain_validated = 3.2
    state.record_trace_analyze(
        {"trace_input": "x2"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 3.2

    state.cumulative_gain_validated = 7.5
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 3.2


def test_non_dict_result_is_ignored() -> None:
    """``result`` not being a dict short-circuits without raising.

    Pre-existing contract — re-pinned here so a future refactor cannot
    accidentally regress the type-guard while wiring in the new fields.
    """
    state = SharedState()
    state.record_trace_analyze({"trace_input": "x"}, None)  # type: ignore[arg-type]
    assert state.last_trace_analyze == {}


def test_existing_fields_still_populated(tmp_path: Path) -> None:
    """New fields are additive: pre-existing keys keep their values."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# hi", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "/some/trace.json"},
        {
            "hot_kernels": [
                {
                    "kernel_id": "k1",
                    "name": "aten::mm",
                    "gpu_pct": 25.0,
                    "reusable_native_kernel": False,
                },
                {
                    "kernel_id": "k2",
                    "name": "rmsnorm",
                    "gpu_pct": 10.0,
                    "reusable_native_kernel": True,
                },
            ],
            "candidates_path": "/some/kc.json",
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [
                {"code": "high_gpu_idle_pct", "severity": "warning",
                 "message": "idle=48%", "idle_pct": 48.0},
            ],
        },
    )

    cached = state.last_trace_analyze
    assert cached["trace_input"] == "/some/trace.json"
    assert cached["candidates_path"] == "/some/kc.json"
    assert len(cached["hot_kernels_top15"]) == 2
    assert cached["reusable_native_kernel_ids"] == ["k2"]
    assert len(cached["trace_health_warnings"]) == 1
    assert cached["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"
    assert cached["analysis_md_path"] == str(analysis_md)
    assert cached["analysis_md_text"] == "# hi"
    assert cached["roofline_snapshot_id"] == 1
    assert cached["roofline_baseline_gain_at_snapshot"] == 0.0
    assert cached["ts"]


def test_record_trace_analyze_preserves_kernel_roofline_fields(
    tmp_path: Path,
) -> None:
    """Kernel-roofline fields from TraceLens candidates survive caching."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# hi", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "/some/trace.json"},
        {
            "hot_kernels": [
                {
                    "kernel_id": "k1",
                    "name": "rmsnorm_kernel",
                    "gpu_pct": 8.2,
                    "bottleneck": "memory",
                    "bound_type": "memory",
                    "flops_per_byte": 0.45,
                    "efficiency_percent": 31.2,
                    "compute_utilization_pct": 9.1,
                    "bandwidth_utilization_pct": 72.4,
                    "suggestion": "reduce memory traffic",
                    "roofline_name": "rmsnorm_kernel",
                    "source_file": "/tmp/rmsnorm.py",
                    "recommended_actions": ["fuse adjacent ops"],
                    "reusable_native_kernel": True,
                },
            ],
            "candidates_path": "/some/kc.json",
            "kernel_roofline_path": "/some/reports/kernel_roofline.json",
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [],
        },
    )

    cached = state.last_trace_analyze
    assert cached["kernel_roofline_path"] == "/some/reports/kernel_roofline.json"
    row = cached["hot_kernels_top15"][0]
    assert row["bound_type"] == "memory"
    assert row["arithmetic_intensity"] == 0.45
    assert row["flops_per_byte"] == 0.45
    assert row["efficiency_percent"] == 31.2
    assert row["compute_utilization_pct"] == 9.1
    assert row["bandwidth_utilization_pct"] == 72.4
    assert row["suggestion"] == "reduce memory traffic"
    assert cached["kernel_roofline_top15"][0] == row
