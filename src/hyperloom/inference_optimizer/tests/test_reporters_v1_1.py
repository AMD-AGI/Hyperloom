# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the decision_journal, kernel_profiling, invocation and phase-timeline renderers."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors import collect_capability_summary
from hyperloom.inference_optimizer.breakdown.reporters import render_session_report
from hyperloom.inference_optimizer.breakdown.reporters._renderers.capability_summary import (
    render as render_capability_summary,
)
from hyperloom.inference_optimizer.breakdown.reporters._renderers.decision_journal import render as render_dj
from hyperloom.inference_optimizer.breakdown.reporters._renderers.invocations import render_forge, render_geak
from hyperloom.inference_optimizer.breakdown.reporters._renderers.kernel_profiling import render as render_kp
from hyperloom.inference_optimizer.breakdown.reporters._renderers.phase_timeline import render as render_phase_timeline


def _base_breakdown(**overrides):
    base = {
        "session": {"session_id": "dj-test", "session_dir": "/tmp/session"},
        "decision_journal": [],
        "kernel_profiling": [],
    }
    base.update(overrides)
    return base


def test_decision_journal_skipped_when_empty() -> None:
    sec = render_dj(_base_breakdown())
    assert sec.skipped
    assert sec.section_id == "decision_journal"


def test_decision_journal_renders_round_and_variants() -> None:
    bd = _base_breakdown(
        decision_journal=[
            {
                "ts": "2026-05-15T10:00:00+00:00",
                "phase": "params",
                "round_id": "params-001",
                "baseline_ref_tput": 1000.0,
                "variants": [
                    {
                        "name": "ncds_16",
                        "outcome": "round_winner",
                        "gain_pct_vs_base": 0.56,
                        "output_throughput": 1005.6,
                        "status": "succeeded",
                    },
                    {
                        "name": "bad_knob",
                        "outcome": "rejected",
                        "gain_pct_vs_base": -2.0,
                        "reject_reason": "not_keep",
                        "status": "succeeded",
                    },
                ],
                "round_decision": {
                    "outcome": "discarded",
                    "best_variant_name": "ncds_16",
                    "gain_vs_cb_pct": 0.56,
                    "promotion_rule": "below_threshold",
                    "promotion_rule_detail": "gain_vs_cb=0.56% < threshold",
                    "keep_threshold_pct": 1.0,
                    "variants_tested_count": 2,
                },
            }
        ]
    )
    sec = render_dj(bd)
    assert not sec.skipped
    assert "params-001" in sec.markdown_block
    assert "below_threshold" in sec.markdown_block
    assert "ncds_16" in sec.markdown_block
    assert "bad_knob" in sec.markdown_block
    assert "not_keep" in sec.markdown_block
    assert any("1 round(s)" in f for f in sec.key_facts)


def test_kernel_profiling_skipped_when_empty() -> None:
    sec = render_kp(_base_breakdown())
    assert sec.skipped


def test_kernel_profiling_renders_top_kernels(tmp_path: Path) -> None:
    rel_log = "kernel-agent/runs/sid-1/logs/tracelens_analysis/run-a.log"

    bd = _base_breakdown(
        session={"session_id": "kp-test", "session_dir": str(tmp_path)},
        kernel_profiling=[
            {
                "run_id": "run-a",
                "task_id": "sid-1",
                "ts": "2026-05-15T11:00:00+00:00",
                "framework": "sglang",
                "launch": {
                    "framework_args": "python -m sglang.launch_server --tp 8",
                    "framework_args_source": "yaml_cmd",
                },
                "artifacts": {
                    "tracelens_status_json": "kernel-agent/runs/sid-1/status/tracelens_analysis/run-a.json",
                    "tracelens_log": rel_log,
                    "trace_paths": ["runs/profile/t1/torch_trace/foo.trace.json.gz"],
                },
                "outputs": {
                    "tool": "tracelens_analysis",
                    "analysis_summary": "3 compute-bound kernels",
                    "top_kernels": [
                        {
                            "kernel_id": "k0",
                            "name": "gemm_kernel",
                            "gpu_pct": 12.5,
                            "duration_us": 90000,
                            "bottleneck": "compute",
                        }
                    ],
                },
            }
        ],
    )
    sec = render_kp(bd)
    assert not sec.skipped
    assert "run-a" in sec.markdown_block
    assert "tracelens_analysis" in sec.markdown_block
    assert "gemm_kernel" in sec.markdown_block
    assert "3 compute-bound kernels" in sec.markdown_block
    assert rel_log not in sec.markdown_block


def test_kernel_profiling_never_reads_the_logs_it_points_at(tmp_path: Path) -> None:
    """Artifact locations are data; rendering must not open what they name."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "run.log").write_text("secret-tail-line\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-the-session.log"
    outside.write_text("outside-secret-line\n", encoding="utf-8")

    for pointer in ("logs/run.log", str(outside), "../outside-the-session.log"):
        bd = _base_breakdown(
            session={"session_id": "kp-test", "session_dir": str(tmp_path)},
            kernel_profiling=[
                {
                    "run_id": "run-b",
                    "task_id": "t1",
                    "artifacts": {"tracelens_log": pointer},
                    "outputs": {"tool": "tracelens_analysis", "top_kernels": []},
                }
            ],
        )
        sec = render_kp(bd)
        assert "secret-tail-line" not in sec.markdown_block
        assert "outside-secret-line" not in sec.markdown_block


def test_invocation_renderer_normalizes_and_caps_attempt_rows() -> None:
    attempts = [{"ts": f"t{i}", "kernel_id": f"k{i}", "decision": "REVERT"} for i in range(26)]
    attempts.extend(
        [
            "legacy-kernel-id",
            {
                "ts": "done",
                "kernel_name": "named-kernel",
                "decision": "KEEP",
                "micro_speedup": 1.25,
                "workspace_path": "/tmp/ws",
                "error": "x" * 100,
            },
            {"kernel_id": "failed-kernel", "decision": "FAILED"},
            {"kernel_id": "error-kernel", "decision": "ERROR"},
        ]
    )

    sec = render_geak({"geak_invocations": attempts})

    assert not sec.skipped
    assert any("30 invocation(s), 1 KEEP, 2 FAILED" in fact for fact in sec.key_facts)
    assert sec.decisions[0].kind == "kept"
    assert "_Showing last 25 of 30 attempts._" in sec.markdown_block
    assert "legacy-kernel-id" in sec.markdown_block
    assert "named-kernel" in sec.markdown_block
    assert "/tmp/ws" in sec.markdown_block
    assert "x" * 80 in sec.markdown_block
    assert "x" * 81 not in sec.markdown_block

    forge_sec = render_forge({"forge_invocations": attempts})

    assert not forge_sec.skipped
    assert forge_sec.section_id == "forge_invocations"
    assert any("30 invocation(s), 1 KEEP, 2 FAILED" in fact for fact in forge_sec.key_facts)
    assert "named-kernel" in forge_sec.markdown_block


def test_phase_timeline_renderer_renders_capped_histogram() -> None:
    events = ["bootstrap"]
    events.extend(
        {
            "ts": f"t{i}",
            "action": f"action-{i}",
            "decision": "KEEP" if i % 2 == 0 else "REVERT",
            "task_id": f"task-{i}",
            "error_class": "RuntimeError" if i == 30 else "",
        }
        for i in range(31)
    )

    sec = render_phase_timeline({"phase_timeline": events})

    assert not sec.skipped
    assert any("Recorded 32 phase event(s); newest = `action-30` (KEEP)." in fact for fact in sec.key_facts)
    assert any("KEEP=16" in fact and "REVERT=15" in fact and "(none)=1" in fact for fact in sec.key_facts)
    assert "_Showing last 30 of 32 events._" in sec.markdown_block
    assert "action-0" not in sec.markdown_block
    assert "action-30" in sec.markdown_block
    assert "RuntimeError" in sec.markdown_block


def test_compose_includes_v1_1_sections_in_report() -> None:
    bd = _base_breakdown(
        decision_journal=[
            {
                "phase": "backends",
                "round_id": "b-1",
                "variants": [{"name": "v1", "outcome": "promoted", "gain_pct_vs_base": 5.0}],
                "round_decision": {"outcome": "promoted", "promotion_rule": "single_shot"},
            }
        ],
        kernel_profiling=[
            {
                "run_id": "p1",
                "task_id": "t1",
                "outputs": {"tool": "magpie_torch_profiler", "top_kernels": []},
                "artifacts": {},
                "launch": {},
            }
        ],
        workload={"model_name": "m", "framework": "sglang", "gpu_type": "MI300X"},
        baseline={"throughput_tok_s_per_gpu": 100.0},
        final={"cumulative_gain_pct_validated": 5.0},
        capability_summary={},
        param_search={},
        sweep={},
        kernel_lifecycle={"detected": []},
        geak_invocations=[],
        forge_invocations=[],
        phase_timeline=[],
        attribution={"method": "missing"},
        source_files={},
    )
    md = render_session_report(bd).markdown
    assert "### Decision Journal" in md
    assert "### Kernel Profiling" in md
    assert "single_shot" in md
    assert "magpie_torch_profiler" in md


# ---- capability_summary.compute_partition ----
def test_an_unused_partition_lever_is_reported_as_never_offered() -> None:
    """The row exists to be discoverable: the operator could have used it and did not."""
    cap = collect_capability_summary({"framework": "xdit", "compute_partition_modes": []}, [], [])
    row = cap["compute_partition"]
    assert row["status"] == "not_attempted"
    assert row["attempts"] == 0 and row["keeps"] == 0
    # No tested=0 next to attempts=0; the row should not pad itself with zeroes.
    assert "tested" not in row
    # The flag has to travel with the verdict, or the row only says "no".
    assert "--compute-partition-modes spx,dpx,qpx,cpx" in row["reason"]
    assert "--max-latency-ms" in row["reason"]


def test_a_framework_that_cannot_partition_gets_no_row_at_all() -> None:
    """Absent, not not_attempted.

    ``not_attempted`` reads as a missed opportunity, and it flows into
    "Capabilities not attempted" in the report. On a serving framework the
    launch would have been refused, so that would send an operator to a flag
    that exits 2.
    """
    for framework in ("sglang", "vllm", "atom", ""):
        cap = collect_capability_summary({"framework": framework, "compute_partition_modes": ["DPX"]}, [], [])
        assert "compute_partition" not in cap
    # The empty-state callers every other collector test uses must stay unaffected.
    assert "compute_partition" not in collect_capability_summary({}, [], [])


def test_a_kept_mode_is_credited_to_the_lever() -> None:
    cap = collect_capability_summary(
        {
            "framework": "custom",
            "compute_partition_modes": ["SPX", "DPX"],
            "current_best": {"action": "explore", "extra_envs": {"HYPERLOOM_PARTITION_MODE": "DPX"}},
        },
        [],
        [],
    )
    row = cap["compute_partition"]
    assert row["status"] == "kept"
    assert (row["attempts"], row["keeps"], row["tested"]) == (2, 1, 2)
    assert "DPX" in row["reason"]


def test_modes_that_all_lost_read_as_tried_not_untried() -> None:
    # A measured loss is evidence. Reporting it as not_attempted would put the
    # lever in "Capabilities not attempted" after it had actually run.
    cap = collect_capability_summary(
        {
            "framework": "custom",
            "compute_partition_modes": ["CPX"],
            "current_best": {"action": "baseline", "tput": 100.0},
        },
        [],
        [],
    )
    row = cap["compute_partition"]
    assert row["status"] == "tried"
    assert row["keeps"] == 0
    assert "none beat the unpartitioned card" in row["reason"]


def test_the_capability_table_shows_the_reason() -> None:
    """``reason`` is in the documented contract but was rendered nowhere."""
    sec = render_capability_summary(
        {"capability_summary": {"compute_partition": {"status": "not_attempted", "reason": "never offered — enable X"}}}
    )
    assert "never offered — enable X" in sec.markdown_block


def test_an_unused_lever_lands_in_capabilities_not_attempted() -> None:
    """End to end: the section group an operator actually reads."""
    bd = _base_breakdown(
        session={"session_id": "cp", "session_dir": "/tmp/s"},
        workload={"model_name": "m", "framework_name": "xdit"},
        capability_summary={
            "compute_partition": {
                "status": "not_attempted",
                "attempts": 0,
                "keeps": 0,
                "reason": "never offered — enable with `--compute-partition-modes spx,dpx,qpx,cpx`",
            }
        },
        attribution={"method": "missing"},
        source_files={},
    )
    md = render_session_report(bd).markdown
    assert "**Capabilities not attempted**: `compute_partition`" in md
    assert "Capabilities never invoked: `compute_partition`" in md
    assert "--compute-partition-modes spx,dpx,qpx,cpx" in md
