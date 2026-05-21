"""N31 — finalize-time roofline + report Roofline Comparison section.

Three orthogonal pieces pinned here:

1. **SharedState baseline freeze**: ``last_trace_analyze_baseline``
   captures snapshot #1 once and never overwrites. Subsequent
   rooflines only update ``last_trace_analyze`` (current).

2. **N21 gate exception**: when ``cumulative_gain_validated > 0`` AND
   ``optimization_stack`` non-empty AND ``snapshot_id == 1``, a
   re-roofline is allowed even when ``discovered_flags`` are
   unchanged -- the validated gain itself is the signal that the
   hot-kernel distribution has shifted.

3. **Report Roofline Comparison section**: ``report.py``
   ``_format_roofline_comparison_section`` extracts the Executive
   Summary out of both snapshots' analysis.md and renders a
   before/after pair. When only one snapshot exists (no
   re-roofline was triggered), it renders a single Executive
   Summary block with a clear explanation.

The closing_phase auto-inject of a final roofline (part of N31) is
tested separately in test_p1_4_resume.py-style integration tests --
this file focuses on the pure-function contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.report import (
    _extract_executive_summary,
    _format_roofline_comparison_section,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# SharedState baseline freeze field
# ---------------------------------------------------------------------------


def test_baseline_field_defaults_empty():
    s = SharedState()
    assert s.last_trace_analyze_baseline == {}


def test_baseline_field_persists_across_save_load(tmp_path):
    s = SharedState()
    s.last_trace_analyze_baseline = {
        "roofline_snapshot_id": 1,
        "analysis_md_path": "/tmp/analysis-baseline.md",
        "trace_input": "/tmp/profile.tar.gz",
        "ts": "2026-05-21T07:50:00+00:00",
    }
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert s2.last_trace_analyze_baseline["roofline_snapshot_id"] == 1
    assert s2.last_trace_analyze_baseline["analysis_md_path"] == "/tmp/analysis-baseline.md"


# ---------------------------------------------------------------------------
# N21 gate exception (cumulative_gain_validated > 0 + snapshot_id == 1)
# ---------------------------------------------------------------------------


def _make_coord_with_snapshot_1():
    coord = Coordinator.__new__(Coordinator)
    coord.shared_state = SharedState()
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "analysis_md_path": "/tmp/baseline-analysis.md",
        "analysis_md_text": "# Stub\n## Executive Summary\nCompute 60%\n",
    }
    coord.shared_state.discovered_flags = {"sglang": {"backend_flags": ["--foo"]}}
    coord.shared_state.discovered_flags_at_last_snapshot = {
        "sglang": {"backend_flags": ["--foo"]},
    }
    return coord


def test_n21_blocks_when_no_validated_gain():
    """N31 exception should NOT fire when validated_gain == 0 (no real
    improvement; running an expensive 20min roofline for the report
    would be wasted)."""
    coord = _make_coord_with_snapshot_1()
    coord.shared_state.cumulative_gain_validated = 0.0
    coord.shared_state.optimization_stack = []
    # Flags unchanged + no validated gain -> N21 still blocks.
    denied = coord._proposal_denial_for_roofline()
    assert denied is not None
    assert denied.rule == "execution_order"


def test_n21_blocks_when_no_stack_entries():
    """N31 exception requires AT LEAST ONE optimization_stack entry
    so a real before/after comparison makes sense."""
    coord = _make_coord_with_snapshot_1()
    coord.shared_state.cumulative_gain_validated = 1.5
    coord.shared_state.optimization_stack = []  # empty -> N31 doesn't fire
    denied = coord._proposal_denial_for_roofline()
    assert denied is not None


def test_n31_exception_unlocks_when_validated_gain_and_stack():
    """The SOLAR-shaped happy path: cumulative_gain_validated > 0 +
    stack has KEEP entries + snapshot_id == 1 -> N31 allows the
    re-roofline so the final snapshot can be captured for the
    Comparison section."""
    coord = _make_coord_with_snapshot_1()
    coord.shared_state.cumulative_gain_validated = 1.5
    coord.shared_state.optimization_stack = [
        {"action": "params", "variant_name": "decode_steps_16", "tput": 3199.4},
    ]
    denied = coord._proposal_denial_for_roofline()
    assert denied is None


def test_n31_exception_only_fires_once():
    """When snapshot_id has already advanced to 2 (post-N31), the
    exception no longer applies -- subsequent rerolls go back to the
    regular flags-changed check."""
    coord = _make_coord_with_snapshot_1()
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 2
    coord.shared_state.cumulative_gain_validated = 1.5
    coord.shared_state.optimization_stack = [
        {"action": "params", "variant_name": "decode_steps_16", "tput": 3199.4},
    ]
    # Flags unchanged + snapshot_id == 2 -> back to standard N21 deny.
    denied = coord._proposal_denial_for_roofline()
    assert denied is not None


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------


def test_extract_executive_summary_normal(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(
        "# Qwen1.5-7B - MI300X Analysis\n\n"
        "## Executive Summary\n\n"
        "Compute time 77.83%, idle 22%.\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Total Time | 880 ms |\n\n"
        "## Compute Kernel Optimizations\n\n"
        "...details elided...\n",
        encoding="utf-8",
    )
    out = _extract_executive_summary(str(md))
    assert "Executive Summary" in out
    assert "Compute time 77.83%" in out
    # Must stop at the next ## heading.
    assert "Compute Kernel Optimizations" not in out
    assert "details elided" not in out


def test_extract_executive_summary_missing_file():
    out = _extract_executive_summary("/tmp/does-not-exist.md")
    assert "could not read" in out


def test_extract_executive_summary_empty_path():
    out = _extract_executive_summary("")
    assert "no analysis.md path" in out


def test_extract_executive_summary_no_section(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Title\n\n## Other\ncontent\n", encoding="utf-8")
    out = _extract_executive_summary(str(md))
    assert "does not contain" in out


def test_extract_executive_summary_strips_base64_images(tmp_path):
    md = tmp_path / "analysis.md"
    big_b64 = "X" * 50000  # mock huge inline image
    md.write_text(
        "## Executive Summary\n\n"
        f"![chart](data:image/png;base64,{big_b64})\n\n"
        "Real text after image.\n",
        encoding="utf-8",
    )
    out = _extract_executive_summary(str(md))
    assert "[image stripped]" in out
    assert big_b64[:100] not in out
    assert "Real text after image" in out


def test_extract_executive_summary_caps_at_2kb(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(
        "## Executive Summary\n\n"
        + "A" * 5000
        + "\n",
        encoding="utf-8",
    )
    out = _extract_executive_summary(str(md))
    assert len(out) <= 2050
    assert out.endswith("...")


def test_format_comparison_section_single_snapshot(tmp_path):
    """No re-roofline triggered -> render single snapshot with
    explanatory note (cumulative_gain stayed at 0, no point spending
    20min on a second snapshot just for the report)."""
    md = tmp_path / "baseline.md"
    md.write_text(
        "## Executive Summary\n\n"
        "Compute 80%, idle 20%.\n\n"
        "## Compute Kernel\n\n",
        encoding="utf-8",
    )
    cmp = {
        "baseline": {
            "snapshot_id": 1,
            "analysis_md_path": str(md),
            "ts": "2026-05-21T07:50:00+00:00",
        },
        "latest": {
            "snapshot_id": 1,
            "analysis_md_path": str(md),  # same as baseline
            "ts": "2026-05-21T07:50:00+00:00",
        },
    }
    lines = _format_roofline_comparison_section(cmp)
    text = "\n".join(lines)
    assert "## Roofline Comparison" in text
    assert "Only one roofline snapshot was captured" in text
    assert "Compute 80%" in text


def test_format_comparison_section_before_after(tmp_path):
    """Two distinct snapshots -> render both Executive Summaries with
    before/after section headings."""
    base = tmp_path / "baseline.md"
    base.write_text(
        "## Executive Summary\n\nBaseline: compute 60%\n\n## Next\n",
        encoding="utf-8",
    )
    after = tmp_path / "optimized.md"
    after.write_text(
        "## Executive Summary\n\nOptimized: compute 75%\n\n## Next\n",
        encoding="utf-8",
    )
    cmp = {
        "baseline": {
            "snapshot_id": 1,
            "analysis_md_path": str(base),
            "ts": "2026-05-21T07:50:00+00:00",
        },
        "latest": {
            "snapshot_id": 2,
            "analysis_md_path": str(after),
            "ts": "2026-05-21T08:30:00+00:00",
        },
    }
    lines = _format_roofline_comparison_section(cmp)
    text = "\n".join(lines)
    assert "## Roofline Comparison" in text
    assert "### Baseline snapshot #1" in text
    assert "### Post-optimization snapshot #2" in text
    assert "Baseline: compute 60%" in text
    assert "Optimized: compute 75%" in text


def test_format_comparison_section_no_snapshot_at_all():
    """When roofline never ran successfully (baseline empty) the
    section explains the gap rather than throwing."""
    cmp = {"baseline": {}, "latest": {"snapshot_id": None, "analysis_md_path": ""}}
    lines = _format_roofline_comparison_section(cmp)
    text = "\n".join(lines)
    assert "## Roofline Comparison" in text
    assert "No roofline snapshot was captured" in text
