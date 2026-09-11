# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``run_t0_anchor`` after the RecipeKB cutover (local read-modify-write + warm-start lookup)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _cascade_warm_start_search,
    _experience_rows,
    _warm_recipe_source,
    run_t0_anchor,
)
from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)


def test_warm_recipe_source_is_local_recipe_kb() -> None:
    assert _warm_recipe_source({}, kb=object()) == "recipe-kb"
    assert _warm_recipe_source(None, kb=object()) == "recipe-kb"


# Fake SharedState — only the fields the anchor reads
@dataclass
class _FakeSharedState:
    recipe_kb_session_id: str = ""
    warm_start_ts: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    warm_start_context: dict[str, Any] = field(default_factory=dict)
    framework_name: str = "sglang"
    framework_version: str = "0.4.5"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    model_class: str = ""
    baseline_workload_extra: dict[str, Any] = field(default_factory=dict)
    compute_partition: dict[str, Any] = field(default_factory=dict)

    def save(self, _path: Path) -> None:  # noqa: D401
        """No-op save — tests don't care about disk persistence here."""


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "session"
    sd.mkdir()
    return sd


@pytest.fixture
def kb(tmp_path: Path) -> RecipeKB:
    """Local-only dispatcher (no remote)."""
    return RecipeKB(
        local=LocalRecipeStore(root=tmp_path / "kb"),
    )


def _expected_cid(state: _FakeSharedState, workload: str, hw: str) -> str:
    return recipe_canonical_id(
        model=workload,
        hardware=hw,
        framework_name=state.framework_name,
        framework_version=state.framework_version,
        precision=state.precision,
    )


def test_t0_anchor_accepts_legacy_framework_extra_attr(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """Legacy ``framework`` attrs should not produce unknown_framework keys."""
    state = _FakeSharedState(framework_version="0.5.11")
    state.framework_name = ""
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )

    cid = recipe_canonical_id(
        model="m",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.5.11",
        precision="fp8",
    )
    assert kb.local.get_recipe(canonical_id=cid) is not None
    assert "unknown_framework" not in state.warm_start_recipe.get("workload", "")


# happy path
def test_t0_anchor_writes_recipe_row_with_arbor_schema(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """First T0 anchor writes a recipe row whose on-disk JSON matches the arbor schema."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb,
        state,
        workload="DeepSeek-R1",
        hw="MI300X",
        image_digest="img-sha-abc",
        stack_fingerprint={"vllm": "0.6.0", "rocm": "7.2", "aiter": "abc1234"},
        extra_attrs={
            "framework_name": "sglang",
            "model_class": "moe",
        },
        session_dir=session_dir,
    )
    cid = _expected_cid(state, "DeepSeek-R1", "MI300X")
    row = kb.get_recipe(canonical_id=cid)
    assert row is not None
    # arbor-shape top-level identity
    assert row["model"] == "DeepSeek-R1"
    assert row["hardware"] == "MI300X"
    # extras splatted at the top level (arbor convention)
    assert row.get("model_class") == "moe"
    assert row.get("image_digest") == "img-sha-abc"
    assert row.get("tp") == 8


def test_t0_anchor_sets_warm_start_recipe_on_state(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """``warm_start_recipe`` is set on the shared state after anchor lookup."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    # Bare T0 anchor row is classified seed_only/conf 0.0 (not actionable).
    assert state.warm_start_recipe.get("tier") == "seed_only"
    assert state.warm_start_recipe.get("confidence") == 0.0
    assert state.warm_start_context.get("status") == "seed_only"


# warm-start surfacing pitfalls / lessons embedded in the recipe row
def test_t0_anchor_surfaces_pitfalls_and_lessons_from_existing_row(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """Pre-seeded pitfalls/lessons must surface on shared_state.warm_start_pitfalls/lessons."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "M", "MI300X")
    kb.put_recipe(
        canonical_id=cid,
        model="M",
        hardware="MI300X",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        pitfalls=[{"description": "watch for X"}],
        lessons=[{"statement": "Y is the answer", "measured_impact": "+15%"}],
        provenance={"source": "seed", "generator": "ut"},
    )
    run_t0_anchor(
        kb,
        state,
        workload="M",
        hw="MI300X",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    assert len(state.warm_start_pitfalls) == 1
    assert state.warm_start_pitfalls[0]["description"] == "watch for X"
    assert len(state.warm_start_lessons) == 1
    assert state.warm_start_lessons[0]["statement"] == "Y is the answer"
    # The snapshot is the contract the prompt renderers read; it is flat.
    assert "attrs" not in state.warm_start_pitfalls[0]
    assert "attrs" not in state.warm_start_lessons[0]
    assert state.warm_start_recipe["tier"] == "exact"
    assert state.warm_start_recipe["confidence"] == 1.0


def test_t0_anchor_drops_an_unparseable_row_on_disk_and_says_so(
    kb: RecipeKB,
    session_dir: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A row the schema cannot parse is dropped loudly, not rendered as nothing.

    ``Recipe.from_dict`` reads ``statement`` off each lesson, so a row on disk in
    any other shape comes back empty rather than wrapped — the local store cannot
    hand T0 something to unwrap. What matters here is the difference from before:
    the empty row is dropped with a log naming the field, instead of reaching the
    prompt and rendering as ``(none)`` with no signal anywhere.
    """
    state = _FakeSharedState()
    cid = _expected_cid(state, "M", "MI300X")
    kb.put_recipe(
        canonical_id=cid,
        model="M",
        hardware="MI300X",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        lessons=[{"statement": "placeholder", "measured_impact": "+1%"}],
        provenance={"source": "seed", "generator": "ut"},
    )
    recipe_path = next((tmp_path / "kb").rglob("recipe.json"))
    row = json.loads(recipe_path.read_text(encoding="utf-8"))
    row["lessons"] = [{"attrs": {"statement": "wrapped", "measured_impact": "+9%"}}]
    recipe_path.write_text(json.dumps(row), encoding="utf-8")

    with caplog.at_level("WARNING"):
        run_t0_anchor(
            kb,
            state,
            workload="M",
            hw="MI300X",
            extra_attrs={"framework_name": "sglang"},
            session_dir=session_dir,
        )
    assert state.warm_start_lessons == []
    assert "warm_start_lessons: dropped 1 row(s) missing 'statement'" in caplog.text


def test_experience_rows_passes_through_the_stored_flat_shape() -> None:
    """The stored shape is flat; normalisation must not restructure it."""
    rows = _experience_rows(
        [{"statement": "Y is the answer", "measured_impact": "+15%"}],
        "statement",
        "lessons",
    )
    assert rows == [{"statement": "Y is the answer", "measured_impact": "+15%"}]


def test_experience_rows_unwraps_a_wrapped_row_once() -> None:
    """The remote projection passes rows through unnormalised, so unwrap here.

    ``knowledge_to_warm_recipe`` copies ``lessons`` straight out of the remote
    record without going through ``Recipe.from_dict``, which is the one way a
    wrapped row reaches T0. Unwrapping at this boundary keeps every downstream
    reader on a single shape.
    """
    rows = _experience_rows(
        [{"canonical_id": "lesson:x", "attrs": {"statement": "wrapped", "measured_impact": "+1%"}}],
        "statement",
        "lessons",
    )
    assert rows == [{"statement": "wrapped", "measured_impact": "+1%"}]


def test_experience_rows_drops_unusable_rows_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """A row the renderer could not have used is dropped here, with a log.

    The renderer used to emit "(none)" off a non-empty list with no log and no
    error, which is why a wrong shape went unnoticed for so long.
    """
    with caplog.at_level("WARNING"):
        rows = _experience_rows(
            [
                {"statement": "kept"},
                {"measured_impact": "no statement"},
                "not-a-row",
            ],
            "statement",
            "lessons",
        )
    assert rows == [{"statement": "kept"}]
    assert "dropped 2 row(s)" in caplog.text
    assert "warm_start_lessons" in caplog.text


def test_t0_anchor_no_prior_recipe_means_warm_miss(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """Confirm pitfalls/lessons stay empty when the (self-written) row had none."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb,
        state,
        workload="cold-model",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    assert state.warm_start_pitfalls == []
    assert state.warm_start_lessons == []


# Resume short-circuits
def test_t0_anchor_short_circuits_when_already_anchored(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """``recipe_kb_session_id`` AND ``warm_start_ts`` set, no resume → anchor short-circuits."""
    state = _FakeSharedState(
        recipe_kb_session_id="prior-sid",
        warm_start_ts="2026-05-28T00:00:00Z",
    )
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
        resume=False,
    )
    cid = _expected_cid(state, "m", "mi300x")
    assert kb.get_recipe(canonical_id=cid) is None


def test_t0_anchor_resume_does_not_short_circuit(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """``resume=True`` bypasses the skipped-already short-circuit so warm-start gets refreshed."""
    state = _FakeSharedState(
        recipe_kb_session_id="prior-sid",
        warm_start_ts="2026-05-28T00:00:00Z",
    )
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
        resume=True,
    )
    cid = _expected_cid(state, "m", "mi300x")
    assert kb.get_recipe(canonical_id=cid) is not None


def test_t0_anchor_uses_existing_recipe_kb_session_id_when_present(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """A pre-existing ``recipe_kb_session_id`` survives the anchor."""
    state = _FakeSharedState(recipe_kb_session_id="prior-sid-from-resume")
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    assert state.recipe_kb_session_id == "prior-sid-from-resume"


def test_t0_anchor_falls_back_to_session_dir_basename_for_sid(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """With no recipe_kb_session_id, the anchor uses the session_dir basename as the local sid."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    assert state.recipe_kb_session_id == session_dir.name


# Read-modify-write correctness
def test_t0_anchor_preserves_existing_best_config_on_metadata_stamp(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """T0 only stamps metadata — best_config/best_throughput/sessions from a prior CLOSE survive."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "m", "mi300x")
    kb.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        best_config={"tp": "16", "ep": "8"},
        best_throughput=12345.6,
        sessions=[
            {"date": "2026-05-25", "throughput_before": 1.0, "throughput_after": 12345.6, "actions_taken": ["tp+ep"]}
        ],
        provenance={"source": "seed", "generator": "ut"},
    )
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        image_digest="new-img-digest",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    after = kb.get_recipe(canonical_id=cid)
    assert after is not None
    assert after["best_config"] == {"tp": "16", "ep": "8"}
    assert after["best_throughput"] == 12345.6
    assert len(after["sessions"]) == 1
    assert after["sessions"][0]["actions_taken"] == ["tp+ep"]
    assert after.get("image_digest") == "new-img-digest"


def test_t0_anchor_increments_version_on_existing_row(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    state = _FakeSharedState()
    cid = _expected_cid(state, "m", "mi300x")
    kb.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        provenance={"source": "seed", "generator": "ut"},
    )
    assert kb.get_recipe(canonical_id=cid)["version"] == 1  # type: ignore[index]
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    after = kb.get_recipe(canonical_id=cid)
    assert after is not None
    assert after["version"] == 2


# Defensive: anchor never raises on missing optional inputs
def test_t0_anchor_tolerates_missing_extra_attrs(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    state = _FakeSharedState()
    run_t0_anchor(
        kb,
        state,
        workload="m",
        hw="mi300x",
        session_dir=session_dir,
    )


def test_t0_anchor_requires_explicit_session_dir(
    kb: RecipeKB,
) -> None:
    state = _FakeSharedState()
    with pytest.raises(ValueError):
        run_t0_anchor(kb, state, workload="m", hw="mi300x")


# _cascade_warm_start_search: the L1-L4 warm-start tier resolution.
_ACTIONABLE = {
    "best_throughput": 100.0,
    "validated_gain_pct": 10.0,
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen",
    "precision": "fp8",
    "best_config": {"extra_server_args": "--trusted"},
}


class _FakeKB:
    """Minimal KB double exposing only get_recipe + search."""

    def __init__(self, *, get_result=None, search_by_labels=None, get_raises=False):
        self._get_result = get_result
        self._search = search_by_labels or []
        self._get_raises = get_raises
        self.search_calls: list[dict] = []

    def get_recipe(self, *, canonical_id, prefer=None):
        if self._get_raises:
            raise RuntimeError("boom")
        return self._get_result

    def search(self, *, label_match, limit=5):
        self.search_calls.append(dict(label_match))
        # Pop the next queued result list per search invocation.
        if self._search:
            return self._search.pop(0)
        return []


def _cascade(kb, cid="CID:target"):
    return _cascade_warm_start_search(
        kb,
        cid=cid,
        hw="mi300x",
        framework="sglang",
        model_type_val="qwen",
        architectures_val=["Qwen3ForCausalLM"],
        arch_slug="qwen3forcausallm",
        fw_version="0.4.5",
        precision="fp8",
        warm_prefer=None,
    )


def test_cascade_l1_get_recipe_exception_is_swallowed():
    # A raising get_recipe degrades to search-based tiers, not a crash.
    kb = _FakeKB(get_raises=True, search_by_labels=[[{"canonical_id": "CID:l2", **_ACTIONABLE}]])
    point, tier, conf = _cascade(kb)
    assert tier == "same_arch_class"
    assert conf == 0.95


def test_cascade_l2_skips_same_cid_and_nonactionable():
    # A row with the target cid is skipped; a bare non-actionable row is skipped; only the actionable distinct-cid row
    # is accepted.
    kb = _FakeKB(
        get_result={"canonical_id": "CID:other"},
        search_by_labels=[
            [
                {"canonical_id": "CID:target", **_ACTIONABLE},  # same cid: skip
                {"canonical_id": "CID:bare"},  # not actionable: skip
                {"canonical_id": "CID:l2", **_ACTIONABLE},  # accept
            ]
        ],
    )
    point, tier, conf = _cascade(kb)
    assert tier == "same_arch_class"
    assert point["canonical_id"] == "CID:l2"


# ---------------------------------------------------------------------------
# warm_start event wiring
# ---------------------------------------------------------------------------


def _warm_start_events(session_dir: Path) -> list[dict[str, Any]]:
    from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events

    return [event for event in read_timeline_events(session_dir) if event.get("type") == "warm_start"]


def test_t0_anchor_records_its_own_lookup_as_a_timeline_event(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """A cold KB is a completed lookup, not a failure.

    What it matches is its own freshly-stamped anchor row, which demotes to
    ``seed_only`` -- so this is also the case that pins the status apart from
    the finding: every first-ever session lands here.
    """
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    state = _FakeSharedState()
    with session_scope(session_dir):
        run_t0_anchor(
            kb,
            state,
            workload="DeepSeek-R1",
            hw="MI300X",
            extra_attrs={"framework_name": "sglang"},
            session_dir=session_dir,
        )

    events = _warm_start_events(session_dir)
    assert len(events) == 1
    event = events[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["match_status"] == "seed_only"
    # The identity is recorded as T0 queries it, not rebuilt afterwards.
    assert event["ext"]["request"]["canonical_id"] == _expected_cid(state, "DeepSeek-R1", "mi300x")


def test_t0_anchor_claims_only_the_reads_its_own_lookup_made(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """``_kb_amend_recipe`` reads the same store later through the same hook."""
    from hyperloom.inference_optimizer.breakdown.recorder import warm_start_event
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    kb.audit_hook = lambda event: warm_start_event.record_read(session_dir, event)
    state = _FakeSharedState()
    with session_scope(session_dir):
        run_t0_anchor(
            kb,
            state,
            workload="DeepSeek-R1",
            hw="MI300X",
            extra_attrs={"framework_name": "sglang"},
            session_dir=session_dir,
        )
        cid = _expected_cid(state, "DeepSeek-R1", "mi300x")
        # A mid-session amendment read, after the anchor settled.
        kb.get_authoritative_recipe(canonical_id=cid)

    reads = _warm_start_events(session_dir)[0]["ext"]["reads"]
    assert reads is not None
    # T0's exact-identity probe, then the degradation cascade behind it.
    assert reads["by_method"]["get_recipe"] == 1
    assert reads["by_method"]["search"] > 1
    # The row T0 stamps is a write. The audit log carries a successful write
    # with ``hit: True``, which is what let the projection count it twice over.
    assert "put_recipe" not in reads["by_method"]
    # T0 makes exactly one authority read, at the top of the anchor; the
    # amendment read issued after the event settled is not this one.
    assert reads["by_method"]["get_authoritative_recipe"] == 1
    assert reads["count"] == len(reads["rows"]) == sum(reads["by_method"].values())
    # Every read lands in the same second, so service order is carried
    # explicitly: the exact probe has to be readable as having missed before
    # the ladder was walked.
    assert [row["method"] for row in reads["rows"]][:2] == ["get_authoritative_recipe", "get_recipe"]


# --- shape guard: tp/ep/partitions are not canonical_id dimensions -----------


def _seed_row_with_shape(kb: RecipeKB, cid: str, shape: dict[str, Any]) -> None:
    """Seed an actionable row whose shape sits where the warm projection splats it: the row's top level."""
    kb.put_recipe(
        canonical_id=cid,
        model="M",
        hardware="MI300X",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        extras=dict(shape),
        pitfalls=[{"description": "watch for X"}],
        lessons=[{"statement": "Y is the answer", "measured_impact": "+15%"}],
        provenance={"source": "seed", "generator": "ut"},
    )


def test_strict_shape_demotes_a_row_recorded_under_a_different_partition_mode(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """A CPX-recorded row must not replay its config on an unpartitioned pod."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "M", "MI300X")
    _seed_row_with_shape(kb, cid, {"tp": 8, "partitions": 8})

    run_t0_anchor(
        kb,
        state,
        workload="M",
        hw="MI300X",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
        strict_shape=True,
    )

    assert state.warm_start_recipe["tier"] == "seed_only"
    assert state.warm_start_recipe["confidence"] == 0.0
    # Demoted, not dropped: the priors are what the prompt renders, and they cost nothing to honour.
    assert state.warm_start_lessons[0]["statement"] == "Y is the answer"
    assert state.warm_start_pitfalls[0]["description"] == "watch for X"


def test_strict_shape_keeps_a_row_whose_shape_matches_this_pod(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """The guard only fires on disagreement; a matching CPX pod still gets its exact hit."""
    state = _FakeSharedState(compute_partition={"mode": "CPX"})
    cid = _expected_cid(state, "M", "MI300X")
    _seed_row_with_shape(kb, cid, {"tp": 8, "partitions": 8})

    run_t0_anchor(
        kb,
        state,
        workload="M",
        hw="MI300X",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
        strict_shape=True,
    )

    assert state.warm_start_recipe["tier"] == "exact"
    assert state.warm_start_recipe["confidence"] == 1.0


def test_the_shape_guard_is_off_unless_asked_for(
    kb: RecipeKB,
    session_dir: Path,
) -> None:
    """Default is lenient: the same mismatched row keeps its exact hit and its config."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "M", "MI300X")
    _seed_row_with_shape(kb, cid, {"tp": 8, "partitions": 8})

    run_t0_anchor(
        kb,
        state,
        workload="M",
        hw="MI300X",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )

    assert state.warm_start_recipe["tier"] == "exact"
    assert state.warm_start_recipe["confidence"] == 1.0
