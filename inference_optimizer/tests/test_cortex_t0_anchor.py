# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``run_t0_anchor`` after the RecipeKB cutover.

The anchor is now a thin read-modify-write against the local
recipe-snapshot store + a single ``kb.get_recipe`` warm-start
lookup (no fallback ladder). These tests use a real
:class:`LocalRecipeStore` against a tmp dir so we exercise the
on-disk format end-to-end; the remote half is wired to ``None``
so no network call ever happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.cortex_t0 import (
    T0Result,
    run_t0_anchor,
)
from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)


# ===========================================================================
# Fake SharedState — only the fields the anchor reads
# ===========================================================================
@dataclass
class _FakeSharedState:
    cortex_session_id: str = ""
    warm_start_ts: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    framework: str = "sglang"
    framework_version: str = "0.4.5"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    model_class: str = ""

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
        remote=None,
    )


def _expected_cid(state: _FakeSharedState, workload: str, hw: str) -> str:
    return recipe_canonical_id(
        model=workload,
        hardware=hw,
        framework=state.framework,
        framework_version=state.framework_version,
        precision=state.precision,
    )


# ===========================================================================
# happy path
# ===========================================================================
def test_t0_anchor_writes_recipe_row_with_arbor_schema(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """First T0 anchor writes a recipe row whose on-disk JSON
    matches the arbor schema (top-level identity + extras-splatted
    operator-tracing tags)."""
    state = _FakeSharedState()
    result = run_t0_anchor(
        kb, state,
        workload="DeepSeek-R1",
        hw="MI300X",
        image_digest="img-sha-abc",
        stack_fingerprint={"vllm": "0.6.0", "rocm": "7.2", "aiter": "abc1234"},
        extra_attrs={
            "framework": "sglang",
            "model_class": "moe",
        },
        session_dir=session_dir,
    )
    assert isinstance(result, T0Result)
    assert result.workload == "DeepSeek-R1"
    cid = _expected_cid(state, "DeepSeek-R1", "MI300X")
    row = kb.get_recipe(canonical_id=cid)
    assert row is not None
    # arbor-shape top-level identity
    assert row["model"]    == "DeepSeek-R1"
    assert row["hardware"] == "MI300X"
    # extras splatted at the top level (arbor convention)
    assert row.get("model_class")          == "moe"
    assert row.get("image_digest")         == "img-sha-abc"
    assert row.get("tp")                   == 8


def test_t0_anchor_writes_warm_start_snapshot_to_disk(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """``warm_start_recipe`` snapshot lands at
    ``runtime/cortex/.kb_warm.json`` with the expected envelope."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    warm_path = session_dir / "runtime" / "cortex" / ".kb_warm.json"
    assert warm_path.is_file()
    import json
    payload = json.loads(warm_path.read_text())
    # Empty warm on first put — recipe just got created, but the
    # warm_start lookup happens AFTER the put_recipe so the row IS
    # present and tier == "exact".
    assert payload["tier"] == "exact"
    assert payload["confidence"] == 1.0


# ===========================================================================
# warm-start surfacing pitfalls / lessons embedded in the recipe row
# ===========================================================================
def test_t0_anchor_surfaces_pitfalls_and_lessons_from_existing_row(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """Pre-seeded recipe with embedded pitfalls / lessons must
    surface on shared_state.warm_start_pitfalls / lessons."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "M", "MI300X")
    # Pre-seed a recipe with embedded pitfalls + lessons.
    kb.put_recipe(
        canonical_id=cid,
        model="M", hardware="MI300X",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        pitfalls=[{"description": "watch for X"}],
        lessons=[{"statement": "Y is the answer", "measured_impact": "+15%"}],
        provenance={"source": "seed", "generator": "ut"},
    )
    run_t0_anchor(
        kb, state,
        workload="M", hw="MI300X",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    assert len(state.warm_start_pitfalls) == 1
    assert state.warm_start_pitfalls[0]["description"] == "watch for X"
    assert len(state.warm_start_lessons) == 1
    assert state.warm_start_lessons[0]["statement"] == "Y is the answer"
    assert state.warm_start_recipe["tier"] == "exact"
    assert state.warm_start_recipe["confidence"] == 1.0


def test_t0_anchor_no_prior_recipe_means_warm_miss(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """The anchor itself writes a row (so warm becomes 'exact' on
    the second pass); but the warm_start lookup happens AFTER the
    put, so we always see at least our own row.

    To test 'true miss' we'd have to inspect the state BEFORE the
    anchor runs. Here we just confirm pitfalls / lessons stay empty
    when the row had none."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb, state,
        workload="cold-model", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    assert state.warm_start_pitfalls == []
    assert state.warm_start_lessons  == []


# ===========================================================================
# Resume short-circuits
# ===========================================================================
def test_t0_anchor_short_circuits_when_already_anchored(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """``cortex_session_id`` AND ``warm_start_ts`` both set, no
    resume → anchor must short-circuit without re-running the
    read-modify-write."""
    state = _FakeSharedState(
        cortex_session_id="prior-sid",
        warm_start_ts="2026-05-28T00:00:00Z",
    )
    result = run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
        resume=False,
    )
    assert result.status == "skipped_already"
    # No row was written — the anchor short-circuited.
    cid = _expected_cid(state, "m", "mi300x")
    assert kb.get_recipe(canonical_id=cid) is None


def test_t0_anchor_resume_does_not_short_circuit(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """``resume=True`` bypasses the skipped-already short-circuit
    so warm-start gets refreshed (the local store may have grown
    new recipes since the original run)."""
    state = _FakeSharedState(
        cortex_session_id="prior-sid",
        warm_start_ts="2026-05-28T00:00:00Z",
    )
    result = run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
        resume=True,
    )
    assert result.status in ("ok", "resumed")
    cid = _expected_cid(state, "m", "mi300x")
    assert kb.get_recipe(canonical_id=cid) is not None


def test_t0_anchor_uses_existing_cortex_session_id_when_present(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """A pre-existing ``cortex_session_id`` survives the anchor —
    we don't fabricate a new one."""
    state = _FakeSharedState(cortex_session_id="prior-sid-from-resume")
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    assert state.cortex_session_id == "prior-sid-from-resume"


def test_t0_anchor_falls_back_to_session_dir_basename_for_sid(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """When SharedState has no cortex_session_id, the anchor uses
    the session_dir basename as the local sid (mirrors the prior
    cortex_kb behaviour now that the central session protocol is
    retired)."""
    state = _FakeSharedState()
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    assert state.cortex_session_id == session_dir.name


# ===========================================================================
# Read-modify-write correctness
# ===========================================================================
def test_t0_anchor_preserves_existing_best_config_on_metadata_stamp(
    kb: RecipeKB, session_dir: Path,
) -> None:
    """T0 only stamps metadata — best_config / best_throughput /
    sessions written by a prior CLOSE must survive the anchor."""
    state = _FakeSharedState()
    cid = _expected_cid(state, "m", "mi300x")
    # Seed a row with a populated best_config.
    kb.put_recipe(
        canonical_id=cid,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        best_config={"tp": "16", "ep": "8"},
        best_throughput=12345.6,
        sessions=[
            {"date": "2026-05-25", "throughput_before": 1.0,
             "throughput_after": 12345.6, "actions_taken": ["tp+ep"]}
        ],
        provenance={"source": "seed", "generator": "ut"},
    )
    # T0 anchor adds new metadata extras.
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        image_digest="new-img-digest",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    # best_config / best_throughput / sessions all preserved.
    after = kb.get_recipe(canonical_id=cid)
    assert after is not None
    assert after["best_config"]     == {"tp": "16", "ep": "8"}
    assert after["best_throughput"] == 12345.6
    assert len(after["sessions"])    == 1
    assert after["sessions"][0]["actions_taken"] == ["tp+ep"]
    # New metadata stamped.
    assert after.get("image_digest") == "new-img-digest"


def test_t0_anchor_increments_version_on_existing_row(
    kb: RecipeKB, session_dir: Path,
) -> None:
    state = _FakeSharedState()
    cid = _expected_cid(state, "m", "mi300x")
    kb.put_recipe(
        canonical_id=cid,
        model="m", hardware="mi300x",
        framework="sglang", framework_version="0.4.5", precision="fp8",
        provenance={"source": "seed", "generator": "ut"},
    )
    assert kb.get_recipe(canonical_id=cid)["version"] == 1  # type: ignore[index]
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=session_dir,
    )
    after = kb.get_recipe(canonical_id=cid)
    assert after is not None
    assert after["version"] == 2


# ===========================================================================
# Defensive: anchor never raises on missing optional inputs
# ===========================================================================
def test_t0_anchor_tolerates_missing_extra_attrs(
    kb: RecipeKB, session_dir: Path,
) -> None:
    state = _FakeSharedState()
    # No extra_attrs / stack_fingerprint / image_digest at all.
    result = run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        session_dir=session_dir,
    )
    assert result.status in ("ok", "resumed")


def test_t0_anchor_requires_explicit_session_dir(
    kb: RecipeKB,
) -> None:
    state = _FakeSharedState()
    with pytest.raises(ValueError):
        run_t0_anchor(kb, state, workload="m", hw="mi300x")
