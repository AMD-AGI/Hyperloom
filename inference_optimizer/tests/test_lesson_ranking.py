"""GAP 7+8 — client-side lesson / pitfall ranking tests.

Locks in:

* ``_lesson_rank_score`` composite formula (confidence × validated_count
  × shape similarity × version proximity × time decay).
* ``_version_proximity`` semver-ish bucketing.
* ``lessons()`` / ``pitfalls()`` actually USE the ranking instead of
  falling back to confidence-only sort.
* Specialist prompt ``_format_version_note`` annotates a version
  mismatch with the human-readable ``[from sglang@X, you're on Y]``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from inference_optimizer.cortex_kb_client import (
    _age_in_days,
    _lesson_rank_score,
    _rank_fact_points,
    _version_proximity,
)


# ===========================================================================
# _version_proximity
# ===========================================================================
def test_version_proximity_exact_match():
    assert _version_proximity("0.5.11", "0.5.11") == 1.0


def test_version_proximity_same_major_minor_different_patch():
    assert _version_proximity("0.5.11", "0.5.4") == 0.8


def test_version_proximity_same_major_different_minor():
    assert _version_proximity("0.5.11", "0.4.0") == 0.5


def test_version_proximity_different_major():
    assert _version_proximity("1.0.0", "0.5.11") == 0.2


def test_version_proximity_strips_v_prefix():
    """``v0.5.11`` and ``0.5.11`` are the same version under either spelling."""
    assert _version_proximity("v0.5.11", "0.5.11") == 1.0


def test_version_proximity_empty_returns_default():
    """Unparseable / empty inputs collapse to 0.2 (no one gets a bonus)."""
    assert _version_proximity("", "0.5.11") == 0.2
    assert _version_proximity("garbage", "0.5.11") == 0.2


# ===========================================================================
# _lesson_rank_score formula
# ===========================================================================
def _point(
    *,
    confidence: float = 0.5,
    validated_count: int | None = None,
    framework_version: str = "",
    last_validated_at: str = "",
    extra_attrs: dict | None = None,
) -> dict:
    attrs = dict(extra_attrs or {})
    if validated_count is not None:
        attrs["validated_count"] = validated_count
    if framework_version:
        attrs["framework_version"] = framework_version
    if last_validated_at:
        attrs["last_validated_at"] = last_validated_at
    return {"confidence": confidence, "attrs": attrs}


def test_rank_score_base_is_kb_confidence():
    """With everything else missing, score = confidence."""
    p = _point(confidence=0.75)
    assert _lesson_rank_score(p, {}, "") == pytest.approx(0.75)


def test_rank_score_validated_count_multiplier():
    """Each additional validator (above 1) adds 10%, capped at +50%
    (i.e. saturates at validated_count=6)."""
    p1 = _point(confidence=0.5, validated_count=1)
    p2 = _point(confidence=0.5, validated_count=3)
    p6 = _point(confidence=0.5, validated_count=6)
    p10 = _point(confidence=0.5, validated_count=10)  # capped at 6
    assert _lesson_rank_score(p1, {}, "") == pytest.approx(0.5)
    assert _lesson_rank_score(p2, {}, "") == pytest.approx(0.5 * 1.2)
    assert _lesson_rank_score(p6, {}, "") == pytest.approx(0.5 * 1.5)
    assert _lesson_rank_score(p10, {}, "") == pytest.approx(0.5 * 1.5)


def test_rank_score_shape_similarity_exact_tp_match():
    """Exact match on a high-weight shape field (tp) bumps the score
    by ~30%."""
    p = _point(confidence=0.5, extra_attrs={"tp": 8})
    s_match = _lesson_rank_score(p, {"tp": 8}, "")
    s_mismatch = _lesson_rank_score(p, {"tp": 4}, "")
    # match: 0.5 * 1.3 = 0.65; mismatch: 0.5 * (0.5 + 0.5 * 4/8) = 0.5 * 0.75
    assert s_match == pytest.approx(0.5 * 1.3)
    assert s_mismatch == pytest.approx(0.5 * 0.75)
    assert s_match > s_mismatch


def test_rank_score_shape_string_mismatch_downweights():
    """Non-numeric shape mismatch (precision="fp8" vs "bf16") gets a
    50% downweight."""
    p = _point(confidence=0.5, extra_attrs={"precision": "fp8"})
    s = _lesson_rank_score(p, {"precision": "bf16"}, "")
    assert s == pytest.approx(0.5 * 0.5)


def test_rank_score_framework_version_proximity():
    """Same patch is best; different major worst."""
    same_patch = _point(confidence=1.0, framework_version="0.5.11")
    same_minor = _point(confidence=1.0, framework_version="0.5.4")
    same_major = _point(confidence=1.0, framework_version="0.4.0")
    diff_major = _point(confidence=1.0, framework_version="1.0.0")
    assert _lesson_rank_score(same_patch, {}, "0.5.11") == pytest.approx(1.0)
    assert _lesson_rank_score(same_minor, {}, "0.5.11") == pytest.approx(0.8)
    assert _lesson_rank_score(same_major, {}, "0.5.11") == pytest.approx(0.5)
    assert _lesson_rank_score(diff_major, {}, "0.5.11") == pytest.approx(0.2)


def test_rank_score_time_decay_recent_lesson_outranks_old():
    """Fresh lesson > 90-day-old lesson (half-life 90)."""
    now = datetime.now(timezone.utc)
    fresh = _point(
        confidence=1.0,
        last_validated_at=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    old = _point(
        confidence=1.0,
        last_validated_at=(now - timedelta(days=90)).isoformat(
            timespec="seconds",
        ).replace("+00:00", "Z"),
    )
    s_fresh = _lesson_rank_score(fresh, {}, "")
    s_old = _lesson_rank_score(old, {}, "")
    # 90 days ago → exactly 0.5 weight (half-life).
    assert s_old == pytest.approx(0.5, abs=0.01)
    assert s_fresh > s_old


def test_rank_score_handles_malformed_inputs_gracefully():
    """None confidence / bad validated_count / unparsable date all
    fall back to neutral 1.0 multipliers without raising."""
    p = {
        "confidence": None,
        "attrs": {
            "validated_count": "not-a-number",
            "last_validated_at": "garbage",
        },
    }
    score = _lesson_rank_score(p, {}, "")
    # confidence None → 0.5 default; everything else collapses to 1.0.
    assert score == pytest.approx(0.5)


# ===========================================================================
# _rank_fact_points end-to-end
# ===========================================================================
def test_rank_fact_points_sorts_by_composite_score():
    """A lesson with same framework version + 5 validators outranks
    a lesson with same confidence but mismatched version + 1 validator."""
    # Recent vs old kept identical (so time decay is identical).
    now_iso = datetime.now(timezone.utc).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")
    p_better = _point(
        confidence=0.7,
        validated_count=5,
        framework_version="0.5.11",
        last_validated_at=now_iso,
    )
    p_worse = _point(
        confidence=0.7,
        validated_count=1,
        framework_version="0.4.0",
        last_validated_at=now_iso,
    )
    p_better["attrs"]["id"] = "better"
    p_worse["attrs"]["id"] = "worse"
    ranked = _rank_fact_points(
        [p_worse, p_better],
        current_workload_shape=None,
        current_framework_version="0.5.11",
    )
    assert [p["attrs"]["id"] for p in ranked] == ["better", "worse"]


def test_rank_fact_points_falls_back_to_confidence_when_no_extra_context():
    """When the caller doesn't supply workload_shape / framework_version,
    the ranking should still produce a sensible order (effectively
    confidence-only). Old code path back-compat."""
    p_low = {"confidence": 0.3, "attrs": {"id": "low"}}
    p_high = {"confidence": 0.9, "attrs": {"id": "high"}}
    ranked = _rank_fact_points(
        [p_low, p_high],
        current_workload_shape=None,
        current_framework_version="",
    )
    assert [p["attrs"]["id"] for p in ranked] == ["high", "low"]


def test_rank_fact_points_drops_non_dict_entries():
    """Defensive — KB may return None / string entries that we skip."""
    ranked = _rank_fact_points(
        [None, "garbage", {"confidence": 0.5, "attrs": {"id": "ok"}}],
    )
    assert len(ranked) == 1
    assert ranked[0]["attrs"]["id"] == "ok"


# ===========================================================================
# _age_in_days helper
# ===========================================================================
def test_age_in_days_handles_z_suffix():
    now = datetime.now(timezone.utc) - timedelta(days=30)
    iso = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    age = _age_in_days(iso)
    assert age is not None
    assert 29.5 < age < 30.5


def test_age_in_days_returns_none_on_garbage():
    assert _age_in_days("not-a-date") is None


# ===========================================================================
# Specialist prompt — version mismatch annotation
# ===========================================================================
def test_format_version_note_annotates_mismatch():
    """When the lesson's framework_version differs from the current
    session, the renderer adds ``[from sglang@X, you're on Y]``."""
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _format_version_note,
    )
    from inference_optimizer.orchestrator.specialist_domains import get_domain
    inp = SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        framework="sglang",
        framework_version="0.5.11",
    )
    note = _format_version_note(inp, {"framework_version": "0.4.0"})
    assert note == " [from sglang@0.4.0, you're on 0.5.11]"


def test_format_version_note_empty_when_versions_match():
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _format_version_note,
    )
    from inference_optimizer.orchestrator.specialist_domains import get_domain
    inp = SpecialistPromptInputs(
        task_id="t-2",
        domain=get_domain("serving_specialist"),
        framework="sglang",
        framework_version="0.5.11",
    )
    note = _format_version_note(inp, {"framework_version": "0.5.11"})
    assert note == ""


def test_format_version_note_empty_when_either_missing():
    """No annotation when either side doesn't know its version
    (legacy / pre-PR row)."""
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _format_version_note,
    )
    from inference_optimizer.orchestrator.specialist_domains import get_domain
    inp_no_current = SpecialistPromptInputs(
        task_id="t-3",
        domain=get_domain("serving_specialist"),
    )
    assert _format_version_note(inp_no_current, {"framework_version": "0.5.11"}) == ""
    inp_has_current = SpecialistPromptInputs(
        task_id="t-4",
        domain=get_domain("serving_specialist"),
        framework="sglang",
        framework_version="0.5.11",
    )
    assert _format_version_note(inp_has_current, {}) == ""


# ===========================================================================
# FIX-3 — KB over-fetch when ranking signals are supplied
# ===========================================================================
def _make_client(tmp_path):
    """Construct a real CortexKBClient pointed at a mock URL so we can
    assert on the wire body."""
    import os
    from inference_optimizer.cortex_kb_client import CortexKBClient
    return CortexKBClient(
        session_dir=tmp_path / "session",
        kb_url="http://kb-test.local",
    )


def test_lessons_over_fetches_when_ranking_active(tmp_path):
    """With ranking signals supplied, KB is asked for limit * 4 rows
    (capped at 200) so the client-side ranker can re-rank a meaningful
    superset before trimming back to ``limit``."""
    import httpx, respx, json
    from inference_optimizer.cortex_kb_client import CortexKBClient
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url="http://kb-test.local",
    )
    with respx.mock(base_url="http://kb-test.local") as router:
        route = router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        client.lessons(
            model="DeepSeek-R1", hardware="MI300X",
            framework="sglang", limit=20,
            current_workload_shape={"tp": 8},
            current_framework_version="0.5.11",
        )
    body = json.loads(route.calls.last.request.content)
    # Over-fetched: 20 × 4 = 80 (well under the 200 cap).
    assert body["limit"] == 80


def test_lessons_legacy_limit_when_no_ranking_signal(tmp_path):
    """Back-compat: callers that pre-date the PR don't supply ranking
    signals. KB receives the legacy ``limit`` so wire cost is
    unchanged for them."""
    import httpx, respx, json
    from inference_optimizer.cortex_kb_client import CortexKBClient
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url="http://kb-test.local",
    )
    with respx.mock(base_url="http://kb-test.local") as router:
        route = router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        client.lessons(
            model="DeepSeek-R1", hardware="MI300X",
            framework="sglang", limit=20,
        )
    body = json.loads(route.calls.last.request.content)
    assert body["limit"] == 20


def test_lessons_returns_top_limit_after_ranking(tmp_path):
    """When KB returns more rows than the caller asked for (because of
    the over-fetch), the client-side sort + slice picks the top
    ``limit`` by composite score."""
    import httpx, respx
    from inference_optimizer.cortex_kb_client import CortexKBClient
    from datetime import datetime, timezone
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url="http://kb-test.local",
    )
    now_iso = datetime.now(timezone.utc).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")
    points = [
        # Old framework_version, low confidence → low score.
        {"confidence": 0.5, "attrs": {
            "statement": "old", "framework_version": "0.3.0",
            "last_validated_at": now_iso,
        }},
        # Same framework_version, high validated_count → high score.
        {"confidence": 0.5, "attrs": {
            "statement": "winner", "framework_version": "0.5.11",
            "validated_count": 5,
            "last_validated_at": now_iso,
        }},
        # Same framework_version, low validated_count → mid score.
        {"confidence": 0.5, "attrs": {
            "statement": "ok", "framework_version": "0.5.11",
            "last_validated_at": now_iso,
        }},
    ]
    with respx.mock(base_url="http://kb-test.local") as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": points}),
        )
        result = client.lessons(
            model="DeepSeek-R1", hardware="MI300X",
            framework="sglang", limit=2,
            current_framework_version="0.5.11",
        )
    # KB returned 3 rows; we trim back to 2; top by composite score
    # is the validated_count=5 row, second is the same-version "ok" row.
    assert len(result) == 2
    assert result[0]["attrs"]["statement"] == "winner"
    assert result[1]["attrs"]["statement"] == "ok"


def test_pitfalls_over_fetches_when_ranking_active(tmp_path):
    """Same over-fetch behaviour for pitfalls (symmetric with lessons)."""
    import httpx, respx, json
    from inference_optimizer.cortex_kb_client import CortexKBClient
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url="http://kb-test.local",
    )
    with respx.mock(base_url="http://kb-test.local") as router:
        route = router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        client.pitfalls(
            model="DeepSeek-R1", hardware="MI300X",
            framework="sglang", limit=20,
            current_workload_shape={"tp": 8},
        )
    body = json.loads(route.calls.last.request.content)
    assert body["limit"] == 80


def test_section_lessons_emits_version_note_in_bullet():
    """End-to-end: a lesson with framework_version=0.4 rendered for a
    session on 0.5 carries the annotation in the bullet line."""
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _section_lessons,
    )
    from inference_optimizer.orchestrator.specialist_domains import get_domain
    inp = SpecialistPromptInputs(
        task_id="t-5",
        domain=get_domain("serving_specialist"),
        framework="sglang",
        framework_version="0.5.11",
        warm_start_lessons=[{
            "canonical_id": "lesson:old",
            "confidence": 0.7,
            "attrs": {
                "statement": "AITER backend wins on MI300X",
                "framework_version": "0.4.0",
            },
        }],
    )
    text = "\n".join(_section_lessons(inp))
    assert "AITER backend wins on MI300X" in text
    assert "[from sglang@0.4.0, you're on 0.5.11]" in text
