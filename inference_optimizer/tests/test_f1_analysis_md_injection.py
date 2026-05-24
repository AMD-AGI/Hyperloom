"""F1-4 — TraceLens analysis.md verbatim injection into the prompt.

Verifies the contracts of :meth:`SharedState._format_analysis_md_full`
and the gating hook in :meth:`SharedState.to_prompt_summary`:

* Empty cache returns the propose-roofline hint.
* Populated cache renders verbatim between
  ``=== TraceLens Analysis ... ===`` bookends with snapshot id + gain.
* Base64 image payloads are stripped from the in-memory render.
* The injection only fires when ``use_roofline_composite=True``;
  default-off preserves the existing prompt surface (legacy
  ``last_select_kernels`` summary stays the only TraceLens hook).

Reference: ``plan_roofline_framework/F1_roofline_composite.MD`` §F1-4.
"""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


def test_format_analysis_md_full_empty_cache_returns_hint():
    s = SharedState()
    out = s._format_analysis_md_full()
    assert "(no TraceLens snapshot yet" in out
    assert "propose `roofline`" in out


def test_format_analysis_md_full_renders_with_bookends():
    s = SharedState()
    s.last_trace_analyze = {
        "analysis_md_text": "# TraceLens analysis\nbody body body.\n",
        "roofline_snapshot_id": 3,
        "roofline_baseline_gain_at_snapshot": 12.5,
    }
    out = s._format_analysis_md_full()
    assert "=== TraceLens Analysis (snapshot #3, gain at snapshot = 12.50%) ===" in out
    assert "# TraceLens analysis" in out
    assert "=== End TraceLens Analysis ===" in out


def test_format_analysis_md_full_handles_non_numeric_gain():
    s = SharedState()
    s.last_trace_analyze = {
        "analysis_md_text": "body",
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": "n/a",
    }
    out = s._format_analysis_md_full()
    # Defensive coercion keeps the bookend renderable.
    assert "snapshot #1" in out
    assert "gain at snapshot = ?%" in out


def test_format_analysis_md_full_strips_base64_data_urls():
    s = SharedState()
    payload = (
        "before\n"
        "![chart](data:image/png;base64,AAAA)\n"
        "after"
    )
    s.last_trace_analyze = {
        "analysis_md_text": payload,
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
    }
    out = s._format_analysis_md_full()
    assert "data:image/png" not in out
    assert "AAAA" not in out
    assert "before" in out
    assert "after" in out


def test_to_prompt_summary_omits_analysis_md_when_toggle_off():
    s = SharedState()
    s.use_roofline_composite = False
    s.last_trace_analyze = {
        "analysis_md_text": "# TraceLens body",
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
    }
    summary = s.to_prompt_summary()
    assert "=== TraceLens Analysis" not in summary
    assert "analysis_md=" not in summary


def test_to_prompt_summary_injects_analysis_md_when_toggle_on():
    s = SharedState()
    s.use_roofline_composite = True
    s.last_trace_analyze = {
        "analysis_md_text": "# TraceLens body line",
        "roofline_snapshot_id": 4,
        "roofline_baseline_gain_at_snapshot": 7.0,
    }
    summary = s.to_prompt_summary()
    assert "analysis_md=" in summary
    assert "=== TraceLens Analysis (snapshot #4," in summary
    assert "# TraceLens body line" in summary


def test_to_prompt_summary_renders_hint_when_toggle_on_and_cache_empty():
    s = SharedState()
    s.use_roofline_composite = True
    summary = s.to_prompt_summary()
    assert "analysis_md=(no TraceLens snapshot yet" in summary
