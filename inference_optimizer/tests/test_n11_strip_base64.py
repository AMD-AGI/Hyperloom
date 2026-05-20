"""Roofline-v2 N11: strip base64 image data URLs from analysis.md.

GPU-empirical root cause (DeepSeek-R1 session 16:06-02:00 of
2026-05-19): the 200 KB analysis.md TraceLens emits has 184.5 KB of
base64 PNG payload (the "Performance Improvement" chart on line 17),
i.e. 92% noise. The main Orchestration LLM cannot see PNG pixels
through base64, so injecting the wholesale string dilutes the
prompt and obscures the high-signal P1 recommendation sections
that drive the entire roofline-v2 decision loop.

N11 strips the data-URL payload before injection. The on-disk
analysis.md stays intact for operator inspection — only the
in-memory string injected into the LLM prompt is modified.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Strip helper — direct unit tests
# ---------------------------------------------------------------------------
def test_strip_passes_text_through_when_no_base64_url():
    """Reports without any data: URL must pass through verbatim."""
    md = (
        "# Analysis\n\nNo images here, just text.\n"
        "Some markdown link: [foo](https://example.com/bar)\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert out == md


def test_strip_empty_input_returns_empty():
    assert SharedState._strip_base64_data_urls("") == ""


def test_strip_handles_none_gracefully():
    """Defence — a truthy-but-non-str ``data:image/`` shouldn't crash
    the renderer; falsy short-circuits before regex even runs."""
    assert SharedState._strip_base64_data_urls(None) is None or \
           SharedState._strip_base64_data_urls(None) == ""


def test_strip_replaces_data_image_png_payload():
    """The smoke-test PNG mirror of what TraceLens emits."""
    md = (
        "## Section\n"
        "![Performance Improvement](data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAACB8AAAN8CAYAAAAa5)\n"
        "Trailing text\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/png;base64" not in out
    assert "iVBORw0KGgo" not in out
    assert "stripped" in out.lower()
    assert "Performance Improvement" in out  # alt-text preserved
    assert "Trailing text" in out
    assert "## Section" in out


def test_strip_replaces_data_image_jpeg_too():
    """N11 is format-agnostic — any `data:image/<type>;base64,...`
    inside a markdown image gets stripped."""
    md = "![Foo](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "stripped" in out.lower()
    assert "Foo" in out


def test_strip_replaces_data_image_svg_xml_base64():
    md = "![chart](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "chart" in out


def test_strip_replaces_multiple_images_independently():
    md = (
        "![a](data:image/png;base64,AAA)\n"
        "text\n"
        "![b](data:image/png;base64,BBB)\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert "AAA" not in out and "BBB" not in out
    assert out.count("stripped") == 2
    # Both alt-texts preserved
    assert "stripped: base64 image — a" in out
    assert "stripped: base64 image — b" in out


def test_strip_does_not_touch_regular_image_urls():
    """Markdown images pointing at filesystem / http URLs must NOT
    be stripped — only base64 data URLs are pure-noise to the LLM."""
    md = (
        "![chart](https://example.com/perf.png)\n"
        "![fig](./local/figure.svg)\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert out == md


def test_strip_preserves_empty_alt_text():
    """Alt-text is optional; an empty `![]` must still get the
    placeholder, with the generic "image" label."""
    md = "![](data:image/png;base64,iVBOR)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "image" in out


# ---------------------------------------------------------------------------
# Real DeepSeek-R1 analysis.md reduction validation
# ---------------------------------------------------------------------------
def test_strip_reduces_real_r1_report_by_90pct_plus():
    """Round-trip the actual analysis.md that triggered N11 design.

    The file is a captured artefact from the GPU run; if it does not
    exist (CI environment) skip with reason. The 92% reduction is the
    central empirical claim of N11 — pin it so a future regex
    refactor cannot silently regress.
    """
    import os
    p = "/wekafs/xiaofei/sessions/kernel-agent/runs/sessions/tracelens/analysis.md"
    if not os.path.exists(p):
        pytest.skip(f"sample analysis.md not present at {p}")
    with open(p, encoding="utf-8") as f:
        md = f.read()
    before = len(md)
    stripped = SharedState._strip_base64_data_urls(md)
    after = len(stripped)
    reduction_pct = (1 - after / before) * 100
    assert reduction_pct > 80, (
        f"N11 must reduce real R1 analysis.md by ≥80%; "
        f"got before={before} after={after} reduction={reduction_pct:.1f}%"
    )
    # Critical signal preserved
    assert "Executive Summary" in stripped
    assert "Idle %" in stripped
    # Specific R1 finding from the report
    assert "fmoe_fp8_blockscale_g1u1" in stripped or \
           "MoE" in stripped


# ---------------------------------------------------------------------------
# Integration: _format_analysis_md_full applies the strip
# ---------------------------------------------------------------------------
def test_format_analysis_md_full_strips_base64_before_injection():
    """The strip must run as part of the prompt-render path; otherwise
    the LLM still sees the noise. Verify via the public renderer."""
    s = SharedState()
    s.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
        "analysis_md_text": (
            "# Test\n\n"
            "## Executive Summary\nGPU idle 64%\n"
            "![chart](data:image/png;base64,iVBORw0KGgoAAAANSUhEUg)\n"
            "## Recommendations\nP1: kernel_opt fmoe\n"
        ),
        "analysis_md_path": "/p/analysis.md",
    }
    rendered = s._format_analysis_md_full()
    assert "data:image/" not in rendered
    assert "iVBORw0KGgo" not in rendered
    # Critical content preserved
    assert "Executive Summary" in rendered
    assert "GPU idle 64%" in rendered
    assert "P1: kernel_opt fmoe" in rendered
    # Bookends still in place
    assert "=== TraceLens Analysis (snapshot #1" in rendered
    assert "=== End TraceLens Analysis ===" in rendered


def test_format_analysis_md_full_no_strip_when_no_image():
    """When the report has no base64 image, the rendered output must
    match what an un-stripped renderer would produce — pure
    pass-through."""
    s = SharedState()
    s.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
        "analysis_md_text": "# Report\nText only\n",
        "analysis_md_path": "/p/analysis.md",
    }
    rendered = s._format_analysis_md_full()
    assert "# Report" in rendered
    assert "Text only" in rendered


# ---------------------------------------------------------------------------
# Strip preserves surrounding markdown structure
# ---------------------------------------------------------------------------
def test_strip_keeps_surrounding_markdown_intact():
    """Stripping must not break neighbouring headings, tables, code
    blocks — the LLM relies on the markdown structure to find sections."""
    md = (
        "# H1\n\n"
        "## H2\n"
        "Some text\n\n"
        "![chart](data:image/png;base64,xxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\n\n"
        "## H2-2\n"
        "| col | val |\n"
        "|-----|-----|\n"
        "| a   | 1   |\n\n"
        "```python\nprint('hi')\n```\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    # Structure preserved
    assert "# H1" in out
    assert "## H2" in out
    assert "## H2-2" in out
    assert "| col | val |" in out
    assert "```python" in out
    assert "print('hi')" in out
    # Strip applied
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in out
