# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``§ 5b. RELATED LESSONS`` specialist prompt section + ``warm_start_lessons`` plumbing tests.

Locks the reader → prompt-render path for ``kind=lesson`` KB writes: the
warmer populates the task param, ``build_specialist_prompts`` renders the
section with metadata, and empty/malformed rows fall back gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_domains import (
    get_domain,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_lessons,
    _section_pitfalls,
    build_specialist_prompts,
)


# Coordinator-warmer integration
@dataclass
class _BareState:
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    gpu_type: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def find_gap(self, _cid: str):
        return None


def _make_coord(tmp_path: Path, *, state: _BareState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_warm_specialist_params_populates_warm_start_lessons(tmp_path: Path):
    """A non-empty SharedState.warm_start_lessons is copied to the task params dict."""
    lessons = [
        {
            "canonical_id": "lesson:abc",
            "kind": "lesson",
            "confidence": 0.9,
            "attrs": {
                "statement":       "--attention-backend AITER → +12.3%",
                "measured_impact": "gain_pct=12.30",
                "source_session_id": "session-A",
            },
        },
    ]
    coord = _make_coord(tmp_path, state=_BareState(warm_start_lessons=lessons))
    params: dict[str, Any] = {}
    await coord._warm_specialist_params(params)
    assert params["warm_start_lessons"] == lessons


@pytest.mark.asyncio
async def test_warm_specialist_params_omits_warm_start_lessons_when_empty(
    tmp_path: Path,
):
    """No lessons → no key in params (avoids leaking a misleading empty list)."""
    coord = _make_coord(tmp_path, state=_BareState(warm_start_lessons=[]))
    params: dict[str, Any] = {}
    await coord._warm_specialist_params(params)
    assert "warm_start_lessons" not in params


# Prompt section
def _make_inp(
    lessons: list[dict[str, Any]] | None = None,
    pitfalls: list[dict[str, Any]] | None = None,
) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        warm_start_lessons=lessons or [],
        warm_start_pitfalls=pitfalls or [],
    )


def test_section_lessons_empty_falls_back_to_placeholder():
    rows = _section_lessons(_make_inp())
    text = "\n".join(rows)
    assert "## 5b. RELATED LESSONS" in text
    assert "(none" in text


def test_section_lessons_renders_each_lesson_with_metadata():
    lessons = [
        {
            "canonical_id": "lesson:abc",
            "confidence": 0.85,
            "attrs": {
                "statement":          "--attention-backend AITER → +12.3%",
                "measured_impact":    "gain_pct=12.30 throughput_after=875.0",
                "source_session_id":  "session-A",
            },
        },
        {
            "canonical_id": "lesson:def",
            "confidence": 0.5,
            "attrs": {
                "statement":       "VLLM_ROCM_USE_AITER=1 → +9.5%",
                "measured_impact": "gain_pct=9.50",
            },
        },
    ]
    rows = _section_lessons(_make_inp(lessons))
    text = "\n".join(rows)
    # Both statements rendered as bullet items.
    assert "--attention-backend AITER → +12.3%" in text
    assert "VLLM_ROCM_USE_AITER=1 → +9.5%" in text
    # Metadata: confidence + source_session_id on the first.
    assert "conf=0.85" in text
    assert "src=session-A" in text
    # Measured impact on its own line (the "impact: ..." line).
    assert "gain_pct=12.30 throughput_after=875.0" in text


def test_section_lessons_renders_dict_measured_impact_as_human_readable_line(
):
    """GAP 3 — a dict ``measured_impact`` is rendered as a human-readable summary instead of raw JSON."""
    lessons = [
        {
            "canonical_id": "lesson:dict",
            "confidence": 0.85,
            "attrs": {
                "statement": "--attention-backend AITER → +12.3%",
                "measured_impact": {
                    "gain_pct": 12.3,
                    "throughput_after": 678.0,
                    "stack_depth_at_apply": 2,
                    "measured_at": "2026-05-26T08:48:00Z",
                },
            },
        },
    ]
    rows = _section_lessons(_make_inp(lessons))
    text = "\n".join(rows)
    assert "+12.30%" in text
    assert "tput=678.0" in text
    assert "depth=2" in text
    assert "2026-05-26" in text


def test_section_lessons_renders_validated_count_when_above_1():
    """GAP 4 — when multiple sessions validated a lesson, surface the count."""
    lessons = [
        {
            "canonical_id": "lesson:multi",
            "confidence": 0.85,
            "attrs": {
                "statement": "AITER+TileLang → +15%",
                "measured_impact": "gain_pct=15",
                "validated_count": 5,
                "source_session_ids": ["s-a", "s-b", "s-c", "s-d", "s-e"],
            },
        },
    ]
    rows = _section_lessons(_make_inp(lessons))
    text = "\n".join(rows)
    assert "validated=5" in text
    # Prefers the latest session id (tail of source_session_ids[]).
    assert "recent=s-e" in text


def test_section_lessons_singleton_validation_omits_validated_tag():
    """When validated_count == 1, the renderer skips the ``validated=N`` bit."""
    lessons = [
        {
            "canonical_id": "lesson:single",
            "confidence": 0.7,
            "attrs": {
                "statement": "x → +5%",
                "validated_count": 1,
                "source_session_ids": ["s-only"],
            },
        },
    ]
    rows = _section_lessons(_make_inp(lessons))
    text = "\n".join(rows)
    assert "validated=" not in text
    # ``recent=`` still surfaces so the operator knows which session produced it.
    assert "recent=s-only" in text


def test_section_lessons_skips_lessons_with_empty_statement():
    """Defensive: a KB row with empty ``statement`` is skipped."""
    lessons = [
        {"canonical_id": "lesson:empty", "attrs": {"statement": ""}},
        {"canonical_id": "lesson:real",
         "attrs": {"statement": "real one"}},
    ]
    rows = _section_lessons(_make_inp(lessons))
    text = "\n".join(rows)
    assert "real one" in text
    # No empty bullet row.
    assert "**** " not in text


def test_build_specialist_prompts_inserts_5b_between_recipe_and_pr_feed():
    """End-to-end: section 5b is inserted between section 5 (recipe) and 5c (pitfalls)."""
    inp = _make_inp([
        {"canonical_id": "lesson:x",
         "attrs": {"statement": "x → +1%"}},
    ])
    _system, user = build_specialist_prompts(inp)
    recipe_idx = user.index("## 5. WARM-START RECIPE SUMMARY")
    lessons_idx = user.index("## 5b. RELATED LESSONS")
    pitfalls_idx = user.index("## 5c. KNOWN PITFALLS")
    pr_idx = user.index("## 6. PR FEED")
    assert recipe_idx < lessons_idx < pitfalls_idx < pr_idx


# § 5c pitfalls section — symmetric mirror of § 5b lessons
def test_section_pitfalls_empty_falls_back_to_placeholder():
    rows = _section_pitfalls(_make_inp())
    text = "\n".join(rows)
    assert "## 5c. KNOWN PITFALLS" in text
    assert "do NOT repeat" in text  # negative framing is critical
    assert "(none" in text


def test_section_pitfalls_renders_each_pitfall_with_metadata():
    pitfalls = [
        {
            "canonical_id": "pitfall:abc",
            "confidence": 0.7,
            "attrs": {
                "description":       "VLLM_ROCM_USE_AITER_FP4BMM=1 → crash on gfx942",
                "severity":          "crash",
                "source_session_id": "session-A",
            },
        },
        {
            "canonical_id": "pitfall:def",
            "confidence": 0.4,
            "attrs": {
                "description": "--max-num-seqs 1024 on MoE → -8% regress",
                "severity":    "regress",
            },
        },
    ]
    rows = _section_pitfalls(_make_inp(pitfalls=pitfalls))
    text = "\n".join(rows)
    assert "VLLM_ROCM_USE_AITER_FP4BMM=1 → crash on gfx942" in text
    assert "severity=crash" in text
    assert "conf=0.70" in text
    assert "src=session-A" in text
    assert "--max-num-seqs 1024 on MoE → -8% regress" in text
    assert "severity=regress" in text


def test_section_pitfalls_skips_pitfalls_with_empty_description():
    """Defensive against partial / legacy rows lacking ``attrs.description``."""
    pitfalls = [
        {"canonical_id": "pitfall:empty", "attrs": {"description": ""}},
        # Legacy shape from the broken traps(symptom=) era.
        {"raw": "{\"points\":[...legacy json blob...]}"},
        {"canonical_id": "pitfall:real",
         "attrs": {"description": "real one", "severity": "crash"}},
    ]
    rows = _section_pitfalls(_make_inp(pitfalls=pitfalls))
    text = "\n".join(rows)
    assert "real one" in text
    assert "**** " not in text
    # No legacy-blob leakage.
    assert "legacy json blob" not in text
