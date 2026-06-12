# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the kernel decision-path breakdown renderer."""

from __future__ import annotations

from inference_optimizer.breakdown.reporters._renderers import (
    kernel_decision_path as kdp,
)


# ---- _fmt_duration ----

def test_fmt_duration_none():
    assert kdp._fmt_duration(None) == "—"


def test_fmt_duration_non_numeric():
    assert kdp._fmt_duration("abc") == "—"


def test_fmt_duration_seconds():
    assert kdp._fmt_duration(12.34) == "12.3s"


def test_fmt_duration_minutes():
    assert kdp._fmt_duration(120) == "2.0min"


# ---- render ----

def test_render_absent_field_skipped():
    sec = kdp.render({})
    assert sec.skipped is True


def test_render_empty_entries_skipped():
    sec = kdp.render({"kernel_decision_path": []})
    assert sec.skipped is True
    assert sec.key_facts


def test_render_with_entries():
    bd = {
        "kernel_decision_path": [
            {
                "kid": "k1",
                "kernel_name": "attn",
                "summary": {
                    "total_steps": 2,
                    "backends_attempted": ["geak"],
                    "final_outcome": "kept",
                    "total_duration_seconds": 90,
                },
                "steps": [
                    {"ts": "t0", "step": "kernel_opt", "backend": "geak",
                     "outcome": "ok", "gain_pct": 5.0, "duration_seconds": 10,
                     "decision_note": "good"},
                    {"ts": "t1", "step": "integrate", "backend": "geak",
                     "outcome": "kept", "gain_pct": 3.0, "duration_seconds": 20},
                ],
            },
        ]
    }
    sec = kdp.render(bd)
    assert sec.skipped is False
    assert "Tracked 1 kernel" in sec.key_facts[0]
    assert "kernel_opt=1" in sec.key_facts[1]
    assert "integrate=1" in sec.key_facts[1]
    assert "k1" in sec.markdown_block
    assert "attn" in sec.markdown_block


def test_render_truncates_steps_and_kids():
    entries = []
    for i in range(10):  # more than _MAX_KIDS (8)
        entries.append({
            "kid": f"k{i}",
            "steps": [
                {"step": "kernel_opt", "ts": f"s{j}"} for j in range(15)  # > _MAX_STEPS_PER_KID
            ],
        })
    sec = kdp.render({"kernel_decision_path": entries})
    assert "Showing first 8 of 10 kernel(s)" in sec.markdown_block
    assert "Showing first 12 of 15 step(s)" in sec.markdown_block


def test_render_ignores_non_dict_entries():
    sec = kdp.render({"kernel_decision_path": ["bad", 123]})
    assert sec.skipped is True


def test_render_entry_without_steps():
    sec = kdp.render({"kernel_decision_path": [{"kid": "k", "steps": []}]})
    assert sec.skipped is False
    assert "k" in sec.markdown_block
