"""Regression tests for the Coordinator -> recipe-snapshot KB write chain.

Guards the P0 that the retired ``test_kb_facts.py`` used to cover before
the RecipeKB cutover: KEEP / REVERT / CLOSE amend the recipe row via
``_kb_amend_recipe`` -> ``_workload_canonical_id``. If
``_workload_canonical_id`` is missing, every write silently no-ops (the
``AttributeError`` is swallowed by the best-effort ``except``), so
warm-start would forever read an empty KB.

Also pins the canonical_id consistency contract: the id the Coordinator
writes under MUST equal the id ``cortex_t0`` anchors under, otherwise
writes and warm-start reads diverge.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend, MockTurn, ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.recipe_kb import (
    LocalRecipeStore, RecipeKB, recipe_canonical_id,
)


_MODEL = "qwen3-30b-a3b"
_HW = "mi300x"
_FW = "sglang"
_FWV = "0.4.5"
_PREC = "fp8"


def _make_coordinator(tmp_path: Path) -> Coordinator:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "kernel":        MockBackend(idle),
        "critic":        MockBackend(idle),
        "robustness":    MockBackend(idle),
    }
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=kb,
        knowledge_plane=None,
    )
    ss = coord.shared_state
    ss.model_name = _MODEL
    ss.gpu_type = _HW
    ss.framework = _FW
    ss.framework_version = _FWV
    ss.precision = _PREC
    return coord


def _expected_cid() -> str:
    return recipe_canonical_id(
        model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
    )


def test_workload_canonical_id_defined_and_consistent(tmp_path: Path) -> None:
    """P0 regression: ``_workload_canonical_id`` must exist and agree
    with both ``recipe_canonical_id`` and the gap-anchor derivation."""
    coord = _make_coordinator(tmp_path)
    assert hasattr(coord, "_workload_canonical_id")
    assert coord._workload_canonical_id() == _expected_cid()
    assert coord._gap_anchor_canonical_id() == _expected_cid()


def test_kb_amend_recipe_persists_lesson(tmp_path: Path) -> None:
    """End-to-end: appending a lesson must actually land in the local
    KB. Before the fix this silently no-opped."""
    coord = _make_coordinator(tmp_path)
    coord._kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "lesson write silently no-opped (P0)"
    statements = [l.get("statement") for l in (row.get("lessons") or [])]
    assert "raise tp to 8" in statements


def test_kb_amend_recipe_persists_pitfall(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    coord._kb_amend_recipe(
        append_pitfall={"description": "ep=8 OOMs on 30B"},
    )
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    descs = [p.get("description") for p in (row.get("pitfalls") or [])]
    assert "ep=8 OOMs on 30B" in descs


def test_recipe_kb_enabled_is_true(tmp_path: Path) -> None:
    """P1-2: RecipeKB exposes ``enabled`` (always True) so callers that
    probe ``client.enabled`` (the coordinator T0 gate) don't silently
    skip the SDK-fallback anchor."""
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    assert kb.enabled is True


def test_sdk_fallback_t0_anchors_into_self_cortex_kb(tmp_path: Path) -> None:
    """P1-2: the SDK-fallback T0 anchor must run (it no longer skips on
    a missing ``enabled``) and write into the SAME dispatcher the
    Coordinator holds — not a throwaway with a possibly-different root."""
    coord = _make_coordinator(tmp_path)
    assert coord.cortex_kb.enabled is True
    # Coordinator construction already fired one fallback anchor while
    # model/hw were still empty (writing an ``unknown_*`` row INTO
    # self.cortex_kb — itself proof the client is reused, not a
    # throwaway). Clear the already-anchored short-circuit markers and
    # re-anchor now that the 5-tuple is seeded, to confirm the anchor
    # targets self.cortex_kb's store.
    coord.shared_state.warm_start_ts = ""
    coord.shared_state.cortex_session_id = ""
    coord._ensure_cortex_t0_anchored()
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "SDK-fallback T0 did not anchor into self.cortex_kb"


# ===========================================================================
# R-6: schema field fidelity — the on-disk row must preserve the fields the
# Coordinator actually writes (severity / dict measured_impact / session
# provenance), otherwise warm-start + session dedup silently lose data.
# ===========================================================================
def _put(store: LocalRecipeStore, **kw) -> None:
    store.put_recipe(
        canonical_id=_expected_cid(),
        model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        **kw,
    )


def test_local_store_preserves_pitfall_severity(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(store, pitfalls=[{"description": "ep=8 OOMs on 30B", "severity": "crash"}])
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    assert row["pitfalls"][0]["severity"] == "crash"


def test_local_store_preserves_lesson_dict_measured_impact(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(store, lessons=[{
        "statement": "raise tp to 8",
        "measured_impact": {"gain_pct": 12.0, "throughput_after": 1000.0},
    }])
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    mi = row["lessons"][0]["measured_impact"]
    assert isinstance(mi, dict), f"measured_impact got mangled to {type(mi)}"
    assert mi["gain_pct"] == 12.0


def test_local_store_preserves_session_provenance(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(store, sessions=[{"session_id": "sess-1", "gain_pct": 12.0, "stack_len": 3}])
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    s = row["sessions"][0]
    assert s["session_id"] == "sess-1"
    assert s["gain_pct"] == 12.0
    assert s["stack_len"] == 3


# ===========================================================================
# R-4: _kb_amend_recipe reads the LOCAL row and preserves T0-stamped extras
# + audit fields; appends accumulate instead of clobbering.
# ===========================================================================
def test_amend_preserves_t0_extras_and_audit(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    # Simulate a T0 anchor that stamped top-level extras + audit fields.
    coord.cortex_kb.put_recipe(
        canonical_id=cid, model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        extras={"model_class": "moe", "image_digest": "sha256:abc"},
        authority="AUTHORITATIVE", confidence=0.99,
    )
    coord._kb_amend_recipe(
        append_lesson={"statement": "x", "measured_impact": "+1%"},
    )
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert row["model_class"] == "moe"           # extras preserved
    assert row["image_digest"] == "sha256:abc"
    assert row["authority"] == "AUTHORITATIVE"   # audit preserved
    assert row["confidence"] == 0.99


def test_amend_appends_lessons_cumulatively(tmp_path: Path) -> None:
    """Local read-modify-write must accumulate, not overwrite."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord._kb_amend_recipe(append_lesson={"statement": "first", "measured_impact": "+1%"})
    coord._kb_amend_recipe(append_lesson={"statement": "second", "measured_impact": "+2%"})
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert [l["statement"] for l in row["lessons"]] == ["first", "second"]


# ===========================================================================
# R-5: CLOSE finalize must not clobber a better historical best_config with
# an empty/worse current result, and must merge (not replace) the fingerprint.
# ===========================================================================
def test_close_does_not_clobber_better_best_config(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.cortex_kb.put_recipe(
        canonical_id=cid, model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        best_config={"name": "good", "tput": "1000"}, best_throughput=1000.0,
        stack_fingerprint={"vllm_version": "0.6.0"},
    )
    # This session ended with no validated win (current_best empty).
    coord.shared_state.current_best = {}
    coord.cortex_finalize_recipe_and_journal()
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 1000.0, "empty CLOSE clobbered a better config"
    assert row["best_config"].get("name") == "good"
    # T0-written fingerprint version survives the CLOSE fingerprint merge.
    assert row["stack_fingerprint"].get("vllm_version") == "0.6.0"


# ===========================================================================
# Regression: recipe write-back must read the renamed canonical
# ``extra_server_args`` off current_best / optimization_stack (state is
# migrated on load) and still emit the KB-legacy ``extra_sglang_args``
# field. Reading the stale name silently dropped the server args and
# broke warm-replay reproduction in the next session.
# ===========================================================================
def test_recipe_attrs_read_canonical_extra_server_args(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.current_best = {
        "action": "explore",
        "name": "win",
        "tput": 900.0,
        # canonical key, as written by _lift_to_current_best post-rename
        "extra_server_args": "--page-size 16",
        "extra_envs": {"SGLANG_X": "1"},
    }
    ss.optimization_stack = [{
        "action": "explore",
        "name": "win",
        "extra_server_args": "--page-size 16",
        "extra_envs": {"SGLANG_X": "1"},
    }]
    ss.gain_per_stack_entry = [12.5]

    attrs = coord._build_recipe_attrs_from_state()

    # best_config carries the args under the KB-legacy field name, read
    # from the canonical state key.
    assert attrs["best_config"]["extra_sglang_args"] == "--page-size 16"
    # what_worked entries likewise carry the args (not an empty string).
    assert attrs["what_worked"], "what_worked should not be empty"
    assert attrs["what_worked"][0]["extra_sglang_args"] == "--page-size 16"
    assert attrs["what_worked"][0]["gain_pct"] == 12.5
