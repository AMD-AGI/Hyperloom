"""Roofline-v2 N5: `_format_analysis_md_full` analysis.md verbatim injection tests.

Pins the contract the main Orchestration LLM consumes per
orchestration.md "How to consume the TraceLens Analysis" section:

* When `last_trace_analyze.analysis_md_text` is empty → render
  a hint asking the LLM to propose `roofline`.
* When populated → render the full report between explicit
  `=== TraceLens Analysis (snapshot #N, gain at snapshot = X.XX%) ===`
  bookends so the LLM can syntactically distinguish report-content
  from surrounding SharedState dump lines.

The header `snapshot=N` + `gain_at_snapshot=X.XX%` fields are what
the LLM uses to detect "stale snapshot, gain has moved by ≥3%" per
the re-profile guidance, so those must be present and accurately
reflect cached values.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


def _state_with_snapshot(*, md: str, snapshot_id: int = 1,
                          gain: float = 0.0) -> SharedState:
    s = SharedState()
    s.last_trace_analyze = {
        "analysis_md_text": md,
        "analysis_md_path": "/tmp/analysis.md",
        "roofline_snapshot_id": snapshot_id,
        "roofline_baseline_gain_at_snapshot": gain,
    }
    return s


# ---------------------------------------------------------------------------
# Empty cache — render placeholder
# ---------------------------------------------------------------------------
def test_no_snapshot_renders_placeholder():
    s = SharedState()
    out = s._format_analysis_md_full()
    assert "no TraceLens snapshot yet" in out
    assert "propose `roofline`" in out
    assert "composite action" in out
    # No bookends — distinguishes from cached output
    assert "=== TraceLens Analysis" not in out


def test_empty_analysis_md_text_renders_placeholder():
    """Cache exists (e.g. trace_analyze succeeded but TraceLens emitted
    an empty report) — still placeholder, no bookends."""
    s = _state_with_snapshot(md="")
    out = s._format_analysis_md_full()
    assert "no TraceLens snapshot yet" in out
    assert "=== TraceLens Analysis" not in out


# ---------------------------------------------------------------------------
# Populated cache — verbatim injection with bookends
# ---------------------------------------------------------------------------
def test_renders_full_report_with_bookends():
    md = (
        "# Executive Summary\n"
        "Compute 51%, Idle 48%\n"
        "\n"
        "## Top Operations\n"
        "| aten::mm | 25.2% | efficiency 64.9% |\n"
        "| rcclAllreduce | 19.1% | comm |\n"
        "\n"
        "## Recommendations\n"
        "Try --enable-two-batch-overlap\n"
    )
    s = _state_with_snapshot(md=md, snapshot_id=3, gain=2.45)
    out = s._format_analysis_md_full()

    # Header with snapshot + gain (rendered with 2 decimals)
    assert "=== TraceLens Analysis (snapshot #3, gain at snapshot = 2.45%) ===" in out
    # Body is verbatim — every line preserved
    assert "# Executive Summary" in out
    assert "Compute 51%, Idle 48%" in out
    assert "## Top Operations" in out
    assert "| aten::mm | 25.2% | efficiency 64.9% |" in out
    assert "| rcclAllreduce | 19.1% | comm |" in out
    assert "## Recommendations" in out
    assert "Try --enable-two-batch-overlap" in out
    # End bookend
    assert "=== End TraceLens Analysis ===" in out
    # Leading newline so the surrounding prompt summary indents nicely
    assert out.startswith("\n")


def test_header_renders_zero_gain_with_two_decimals():
    s = _state_with_snapshot(md="x", gain=0.0)
    out = s._format_analysis_md_full()
    assert "gain at snapshot = 0.00%" in out


def test_header_renders_negative_gain():
    s = _state_with_snapshot(md="x", gain=-1.234)
    out = s._format_analysis_md_full()
    assert "gain at snapshot = -1.23%" in out


def test_header_falls_back_to_question_mark_on_garbage_gain():
    s = _state_with_snapshot(md="x")
    s.last_trace_analyze["roofline_baseline_gain_at_snapshot"] = "not-a-float"
    out = s._format_analysis_md_full()
    assert "gain at snapshot = ?%" in out


def test_header_falls_back_to_question_mark_on_missing_snapshot_id():
    s = _state_with_snapshot(md="x")
    del s.last_trace_analyze["roofline_snapshot_id"]
    out = s._format_analysis_md_full()
    assert "snapshot #?" in out


def test_large_report_round_trips_intact():
    """200 KB report (Case A-D scale) must inject without truncation
    (Decision A3 / B2 — see design §5.1)."""
    md = "# Analysis\n" + ("filler line\n" * 20000)
    s = _state_with_snapshot(md=md)
    out = s._format_analysis_md_full()
    # Body length preserved (modulo bookends + header)
    assert len(out) >= len(md)
    assert out.count("filler line") == 20000
    assert "=== End TraceLens Analysis ===" in out


# ---------------------------------------------------------------------------
# Integration with to_prompt_summary
# ---------------------------------------------------------------------------
def test_to_prompt_summary_includes_analysis_md_line():
    md = "# Report Body\n"
    s = _state_with_snapshot(md=md, snapshot_id=2, gain=1.5)
    out = s.to_prompt_summary()
    # Field name visible
    assert "analysis_md=" in out
    # Bookend + body
    assert "=== TraceLens Analysis (snapshot #2, gain at snapshot = 1.50%) ===" in out
    assert "# Report Body" in out


def test_to_prompt_summary_includes_placeholder_when_no_snapshot():
    s = SharedState()
    out = s.to_prompt_summary()
    assert "analysis_md=(no TraceLens snapshot yet" in out


# ---------------------------------------------------------------------------
# Snapshot stability for cache friendliness (N6 dependency)
# ---------------------------------------------------------------------------
def test_render_is_deterministic_for_same_cache():
    """Two consecutive renders of the same cache must produce identical
    output — this is the prerequisite for Claude Code automatic
    caching (N6) to recognise SECTION-B as cacheable. Any randomness
    or timestamp leakage would invalidate the cache key on every tick."""
    s = _state_with_snapshot(md="stable content", snapshot_id=1, gain=0.5)
    out1 = s._format_analysis_md_full()
    out2 = s._format_analysis_md_full()
    assert out1 == out2


def test_render_changes_when_md_content_changes():
    """Snapshot-stable means snapshot 1 vs snapshot 2 *must* differ;
    otherwise caching would serve stale content."""
    s = _state_with_snapshot(md="snapshot 1", snapshot_id=1)
    out1 = s._format_analysis_md_full()

    s.last_trace_analyze["analysis_md_text"] = "snapshot 2"
    s.last_trace_analyze["roofline_snapshot_id"] = 2
    out2 = s._format_analysis_md_full()

    assert out1 != out2
    assert "snapshot #1" in out1 and "snapshot 1" in out1
    assert "snapshot #2" in out2 and "snapshot 2" in out2


# ---------------------------------------------------------------------------
# orchestration.md guidance section presence (so the prompt has the
# instructions the LLM needs to consume the injected report)
# ---------------------------------------------------------------------------
def test_orchestration_md_includes_consumption_guidance():
    """The orchestration system prompt must include the new "How to
    consume the TraceLens Analysis" section so the LLM knows what to
    do with the verbatim injection from SECTION-B."""
    from inference_optimizer.paths import asset_system_prompts_dir
    text = (asset_system_prompts_dir() / "orchestration.md").read_text(encoding="utf-8")
    # Section header present
    assert "How to consume the TraceLens Analysis" in text
    # Key decision rules referenced
    assert "PRUNE_BRANCH" in text
    assert "discovered_flags" in text
    assert "[untested]" in text
    # Re-profile trigger documented
    assert "cumulative_gain_validated_pct" in text
    assert "3%" in text
    # Bookend reference for the LLM
    assert "=== TraceLens Analysis" in text
