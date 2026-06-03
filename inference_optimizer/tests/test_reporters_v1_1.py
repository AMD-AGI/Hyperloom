"""Tests for v1.1 decision_journal + kernel_profiling renderers."""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.breakdown.reporters import render_session_report
from inference_optimizer.breakdown.reporters._renderers.decision_journal import render as render_dj
from inference_optimizer.breakdown.reporters._renderers.kernel_profiling import render as render_kp


def _base_breakdown(**overrides):
    base = {
        "detail_level": "standard",
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
    bd = _base_breakdown(decision_journal=[{
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
    }])
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
    log_dir = tmp_path / "kernel-agent" / "runs" / "sid-1" / "logs" / "tracelens_analysis"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "run-a.log"
    log_file.write_text("\n".join(f"line-{i}" for i in range(50)), encoding="utf-8")
    rel_log = "kernel-agent/runs/sid-1/logs/tracelens_analysis/run-a.log"

    bd = _base_breakdown(
        detail_level="verbose",
        session={"session_id": "kp-test", "session_dir": str(tmp_path)},
        kernel_profiling=[{
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
                "top_kernels": [{
                    "kernel_id": "k0",
                    "name": "gemm_kernel",
                    "gpu_pct": 12.5,
                    "duration_us": 90000,
                    "bottleneck": "compute",
                }],
            },
        }],
    )
    sec = render_kp(bd)
    assert not sec.skipped
    assert "run-a" in sec.markdown_block
    assert "tracelens_analysis" in sec.markdown_block
    assert "gemm_kernel" in sec.markdown_block
    assert "3 compute-bound kernels" in sec.markdown_block
    assert "line-49" in sec.markdown_block
    assert "line-0" not in sec.markdown_block
    assert "line-9" not in sec.markdown_block


def test_kernel_profiling_standard_hides_cli_log_tail(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "run.log").write_text("secret-tail-line\n", encoding="utf-8")
    bd = _base_breakdown(
        detail_level="standard",
        session={"session_id": "kp-test", "session_dir": str(tmp_path)},
        kernel_profiling=[{
            "run_id": "run-b",
            "task_id": "t1",
            "artifacts": {"tracelens_log": "logs/run.log"},
            "outputs": {"tool": "tracelens_analysis", "top_kernels": []},
        }],
    )
    sec = render_kp(bd)
    assert "secret-tail-line" not in sec.markdown_block


def test_compose_includes_v1_1_sections_in_report() -> None:
    bd = _base_breakdown(
        decision_journal=[{
            "phase": "backends",
            "round_id": "b-1",
            "variants": [{"name": "v1", "outcome": "promoted", "gain_pct_vs_base": 5.0}],
            "round_decision": {"outcome": "promoted", "promotion_rule": "single_shot"},
        }],
        kernel_profiling=[{
            "run_id": "p1",
            "task_id": "t1",
            "outputs": {"tool": "magpie_torch_profiler", "top_kernels": []},
            "artifacts": {},
            "launch": {},
        }],
        workload={"model_name": "m", "framework": "sglang", "gpu_type": "MI300X"},
        baseline={"throughput_tok_s_per_gpu": 100.0},
        final={"cumulative_gain_pct_validated": 5.0},
        capability_summary={},
        param_search={},
        sweep={},
        kernel_lifecycle={"detected": []},
        geak_invocations=[],
        oob_invocations=[],
        phase_timeline=[],
        attribution={"method": "missing"},
        source_files={},
    )
    md = render_session_report(bd).markdown
    assert "### Decision Journal" in md
    assert "### Kernel Profiling" in md
    assert "single_shot" in md
    assert "magpie_torch_profiler" in md
