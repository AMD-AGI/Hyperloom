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
# R-7: KEEP'd kernel optimizations (incl. E2E-verified-but-no-gain) must be
# persisted into the recipe row. Repro for overnight session 20260529T132829Z
# where k006 (micro 1.32x, KEEP, E2E gain -0.094%) vanished from recipe.json
# entirely — because ``what_worked`` is built only from optimization_stack,
# which a no-E2E-gain kernel never enters.
# ===========================================================================
def _seed_kept_kernel(coord: Coordinator) -> None:
    """Populate SharedState the way a KEEP'd + E2E-integrated kernel leaves
    it, mirroring the real shapes:

    * ``kernel_opt_attempts[kid]`` — micro result (record_kernel_opt)
    * ``kernel_integrate_attempts[key]`` — E2E verification outcome
      (record_kernel_integrate_result), joined back by ``kernel_id``.
    """
    ss = coord.shared_state
    ss.kernel_opt_attempts = {
        "k006": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.3202,
            "last_artifact_path": "/x/optimized_versions/v1_n128_launch_tuning.cu",
            # record_kernel_opt persists the source under ``last_source_file``
            # (NOT ``source_file``); the recipe builder must read that key.
            "last_source_file": "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
        },
    }
    ss.kernel_integrate_attempts = {
        "k006|/x/optimized_versions/v1_n128_launch_tuning.cu|": {
            "kernel_id": "k006",
            "patch_path": "/x/optimized_versions/v1_n128_launch_tuning.cu",
            "best_gain_pct": -0.094,
            "last_decision": "NEEDS_REVIEW",
            "attempts": [
                {"decision": "NEEDS_REVIEW", "new_tput": 2477.82,
                 "gain_pct": -0.094},
            ],
        },
    }


def test_build_recipe_attrs_surfaces_kept_kernel(tmp_path: Path) -> None:
    """``_build_recipe_attrs_from_state`` must emit a ``kernel_optimizations``
    entry for the KEEP'd kernel, carrying BOTH micro_speedup and the E2E
    verification outcome. Before the fix this dict has no such key."""
    coord = _make_coordinator(tmp_path)
    _seed_kept_kernel(coord)
    attrs = coord._build_recipe_attrs_from_state()
    kopts = attrs.get("kernel_optimizations") or []
    assert kopts, "KEEP'd kernel k006 missing from recipe attrs"
    k = next((x for x in kopts if x.get("kernel_id") == "k006"), None)
    assert k is not None, f"k006 not in {kopts}"
    assert k["micro_speedup"] == 1.3202
    assert k["decision"] == "KEEP"
    # source_file is read from the ``last_source_file`` ledger key
    # (record_kernel_opt's actual field) so warm-start can relocate the
    # patched source; an empty string means the wrong key was read.
    assert k["source_file"] == (
        "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu"
    )
    # E2E verification outcome must be carried, not dropped.
    assert k["e2e_gain_pct"] == -0.094
    assert k["e2e_tput"] == 2477.82
    assert k["integrated"] is True


def test_close_finalize_persists_kept_kernel_to_kb(tmp_path: Path) -> None:
    """End-to-end: after CLOSE finalize, recipe.json must carry the KEEP'd
    kernel under ``kernel_optimizations`` so warm-start can reuse it."""
    coord = _make_coordinator(tmp_path)
    _seed_kept_kernel(coord)
    coord.cortex_finalize_recipe_and_journal()
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    kopts = row.get("kernel_optimizations") or []
    ids = [k.get("kernel_id") for k in kopts]
    assert "k006" in ids, f"k006 not persisted to KB: {row.get('kernel_optimizations')}"
    k006 = next(k for k in kopts if k.get("kernel_id") == "k006")
    assert k006["micro_speedup"] == 1.3202
    assert k006["e2e_gain_pct"] == -0.094
