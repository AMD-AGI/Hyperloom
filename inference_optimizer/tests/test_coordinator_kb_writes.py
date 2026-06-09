# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the Coordinator -> recipe-snapshot KB write chain.

P0: KEEP/REVERT/CLOSE amend the recipe row via ``_kb_amend_recipe`` ->
``_workload_canonical_id``; if that helper is missing every write silently
no-ops. Also pins the canonical_id consistency contract between Coordinator
writes and ``cortex_t0`` anchors.
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
    """P0 regression: ``_workload_canonical_id`` exists and agrees with ``recipe_canonical_id`` and the gap anchor."""
    coord = _make_coordinator(tmp_path)
    assert hasattr(coord, "_workload_canonical_id")
    assert coord._workload_canonical_id() == _expected_cid()
    assert coord._gap_anchor_canonical_id() == _expected_cid()


def test_kb_amend_recipe_persists_lesson(tmp_path: Path) -> None:
    """End-to-end: appending a lesson lands in the local KB (previously a silent no-op)."""
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


def test_kb_amend_recipe_stamps_architecture_tags(tmp_path: Path) -> None:
    """Amend stamps config.json architecture tags (``architectures`` + ``model_type``) into the recipe."""
    coord = _make_coordinator(tmp_path)
    coord.shared_state.model_architectures = ["LlamaForCausalLM"]
    coord.shared_state.model_type = "llama"
    coord._kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    assert row.get("architectures") == ["LlamaForCausalLM"]
    assert row.get("model_type") == "llama"


def test_kb_amend_recipe_skips_empty_architecture_tags(tmp_path: Path) -> None:
    """With no config.json tags the amend must NOT stamp empty ``architectures`` / ``model_type`` keys."""
    coord = _make_coordinator(tmp_path)
    coord._kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    assert "architectures" not in row
    assert "model_type" not in row


def test_recipe_kb_enabled_is_true(tmp_path: Path) -> None:
    """P1-2: RecipeKB exposes ``enabled`` (always True) so the T0 gate doesn't skip the SDK-fallback anchor."""
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    assert kb.enabled is True


def test_sdk_fallback_t0_anchors_into_self_cortex_kb(tmp_path: Path) -> None:
    """P1-2: the SDK-fallback T0 anchor runs and writes into the SAME dispatcher the Coordinator holds."""
    coord = _make_coordinator(tmp_path)
    assert coord.cortex_kb.enabled is True
    # Clear the already-anchored markers and re-anchor with the 5-tuple seeded.
    coord.shared_state.warm_start_ts = ""
    coord.shared_state.cortex_session_id = ""
    coord._ensure_cortex_t0_anchored()
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "SDK-fallback T0 did not anchor into self.cortex_kb"


# R-6: schema field fidelity — the on-disk row must preserve severity / dict
# measured_impact / session provenance, or warm-start + dedup lose data.
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


# R-4: _kb_amend_recipe reads the LOCAL row and preserves T0-stamped extras +
# audit fields; appends accumulate instead of clobbering.
def test_amend_preserves_t0_extras_and_audit(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
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
    assert row["model_class"] == "moe"
    assert row["image_digest"] == "sha256:abc"
    assert row["authority"] == "AUTHORITATIVE"
    assert row["confidence"] == 0.99


def test_amend_appends_lessons_cumulatively(tmp_path: Path) -> None:
    """Local read-modify-write accumulates, not overwrites."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord._kb_amend_recipe(append_lesson={"statement": "first", "measured_impact": "+1%"})
    coord._kb_amend_recipe(append_lesson={"statement": "second", "measured_impact": "+2%"})
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert [l["statement"] for l in row["lessons"]] == ["first", "second"]


# R-5: CLOSE finalize must not clobber a better historical best_config with an
# empty/worse current result, and must merge (not replace) the fingerprint.
def test_close_does_not_clobber_better_best_config(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.cortex_kb.put_recipe(
        canonical_id=cid, model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        best_config={"name": "good", "tput": "1000"}, best_throughput=1000.0,
        stack_fingerprint={"vllm_version": "0.6.0"},
    )
    coord.shared_state.current_best = {}
    coord.cortex_finalize_recipe_and_journal()
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 1000.0, "empty CLOSE clobbered a better config"
    assert row["best_config"].get("name") == "good"
    assert row["stack_fingerprint"].get("vllm_version") == "0.6.0"


# R-7: KEEP'd kernel optimizations (incl. E2E-verified-but-no-gain) must be
# persisted. Regression: k006 (micro 1.32x, KEEP, E2E -0.094%) vanished from
# recipe.json because ``what_worked`` is built only from optimization_stack.
def _seed_kept_kernel(coord: Coordinator) -> None:
    """Populate SharedState as a KEEP'd + E2E-integrated kernel leaves it."""
    ss = coord.shared_state
    ss.kernel_opt_attempts = {
        "k006": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.3202,
            "last_artifact_path": "/x/optimized_versions/v1_n128_launch_tuning.cu",
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
    """``_build_recipe_attrs_from_state`` emits a ``kernel_optimizations`` entry carrying micro_speedup + the E2E outcome."""
    coord = _make_coordinator(tmp_path)
    _seed_kept_kernel(coord)
    attrs = coord._build_recipe_attrs_from_state()
    kopts = attrs.get("kernel_optimizations") or []
    assert kopts, "KEEP'd kernel k006 missing from recipe attrs"
    k = next((x for x in kopts if x.get("kernel_id") == "k006"), None)
    assert k is not None, f"k006 not in {kopts}"
    assert k["micro_speedup"] == 1.3202
    assert k["decision"] == "KEEP"
    assert k["source_file"] == (
        "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu"
    )
    assert k["e2e_gain_pct"] == -0.094
    assert k["e2e_tput"] == 2477.82
    assert k["integrated"] is True
    assert k["e2e_decision"] == "NEEDS_REVIEW"


def test_close_finalize_persists_kept_kernel_to_kb(tmp_path: Path) -> None:
    """End-to-end: after CLOSE finalize, recipe.json carries the KEEP'd kernel under ``kernel_optimizations``."""
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


# R-8 (P0): a bare-baseline CLOSE whose tput happens to exceed a historical
# best must NOT overwrite the validated best_config. Regression: a flagless
# baseline clobbered the warm_replay recipe, dropping every extra_sglang_arg.
def test_close_does_not_clobber_with_bare_baseline_higher_tput(
    tmp_path: Path,
) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.cortex_kb.put_recipe(
        canonical_id=cid, model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        best_config={
            "name": "warm_replay",
            "extra_sglang_args": "--schedule-policy lpm --page-size 16",
            "tput": "2532",
        },
        best_throughput=2532.0,
    )
    # Bare baseline: no validated stack/gain, but a numerically higher tput
    # that the better-throughput guard alone would let through.
    ss = coord.shared_state
    ss.current_best = {"action": "baseline", "name": "baseline", "tput": 2813.5}
    ss.optimization_stack = []
    ss.cumulative_gain_validated = 0.0
    ss.cumulative_gain = 0.0
    coord.cortex_finalize_recipe_and_journal()
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 2532.0, (
        "bare-baseline CLOSE clobbered a validated best_throughput"
    )
    assert row["best_config"].get("extra_sglang_args") == (
        "--schedule-policy lpm --page-size 16"
    ), "warm_replay launch flags were dropped by a flagless baseline overwrite"


def test_best_config_reads_stack_args_from_canonical_server_key(
    tmp_path: Path,
) -> None:
    """Review #2 repro (fix for #332): best_config reads the canonical ``extra_server_args`` stack key, not the legacy ``*_sglang_args``."""
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.current_best = {
        "name": "tuned",
        "extra_server_args": "--page-size 16 --page-size 16",
        "tput": 2200.0,
    }
    ss.optimization_stack = [{
        "action": "explore",
        "variant_name": "page32",
        "candidate_extra_server_args": "--page-size 32 --schedule-policy lpm",
        "extra_server_args": "--page-size 32 --schedule-policy lpm",
    }]
    ss.cumulative_gain_validated = 10.0
    attrs = coord._build_recipe_attrs_from_state()
    assert attrs["best_config"]["extra_sglang_args"] == (
        "--page-size 32 --schedule-policy lpm"
    ), (
        "stack-layer launch args must be read from the canonical "
        "extra_server_args key; got "
        f"{attrs['best_config'].get('extra_sglang_args')!r}"
    )


def test_close_overwrites_best_when_validated_win(tmp_path: Path) -> None:
    """Counterpart guard: a genuine validated win DOES update best_config/best_throughput."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.cortex_kb.put_recipe(
        canonical_id=cid, model=_MODEL, hardware=_HW, framework=_FW,
        framework_version=_FWV, precision=_PREC,
        best_config={"name": "old", "extra_sglang_args": "--page-size 16",
                     "tput": "2000"},
        best_throughput=2000.0,
    )
    ss = coord.shared_state
    ss.current_best = {
        "name": "tuned",
        "extra_sglang_args": "--page-size 32 --schedule-policy lpm",
        "tput": 2200.0,
    }
    ss.optimization_stack = [{
        "action": "explore", "variant_name": "page32",
        "extra_sglang_args": "--page-size 32 --schedule-policy lpm",
    }]
    ss.cumulative_gain_validated = 10.0
    coord.cortex_finalize_recipe_and_journal()
    row = coord.cortex_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 2200.0
    assert "--page-size 32" in row["best_config"].get("extra_sglang_args", "")


# R-9 (P1): kernel_optimizations[].e2e_decision must carry the integrate
# verdict. Regression: k007 integrate REVERT'd (E2E -1.05%) but the row showed
# only the micro-layer decision=KEEP, hiding the rollback.
def test_kernel_e2e_decision_reflects_integrate_revert(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.kernel_opt_attempts = {
        "k007": {
            "last_decision": "KEEP",
            "last_micro_speedup": 2.42,
            "last_artifact_path": "/x/optimized_versions/v1_multirow.cu",
            "last_source_file": "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
        },
    }
    ss.kernel_integrate_attempts = {
        "k007|/x/optimized_versions/v1_multirow.cu|": {
            "kernel_id": "k007",
            "best_gain_pct": -1.0488865062914727,
            "last_decision": "REVERT",
            "attempts": [
                {"decision": "REVERT", "new_tput": 2784.0072254162687,
                 "gain_pct": -1.0488865062914727},
            ],
        },
    }
    attrs = coord._build_recipe_attrs_from_state()
    kopts = attrs.get("kernel_optimizations") or []
    k = next((x for x in kopts if x.get("kernel_id") == "k007"), None)
    assert k is not None, f"k007 missing from {kopts}"
    assert k["decision"] == "KEEP"
    assert k["e2e_decision"] == "REVERT"
    assert k["integrated"] is True
    assert round(k["e2e_gain_pct"], 2) == -1.05


def test_kernel_e2e_decision_micro_only_when_not_integrated(
    tmp_path: Path,
) -> None:
    """A micro-KEEP kernel that never reached integrate has empty e2e_decision and integrated is False."""
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.kernel_opt_attempts = {
        "k009": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.241,
            "last_artifact_path": "/x/optimized_versions/v1_multirow.cu",
            "last_source_file": "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
        },
    }
    ss.kernel_integrate_attempts = {}
    attrs = coord._build_recipe_attrs_from_state()
    kopts = attrs.get("kernel_optimizations") or []
    k = next((x for x in kopts if x.get("kernel_id") == "k009"), None)
    assert k is not None
    assert k["decision"] == "KEEP"
    assert k["integrated"] is False
    assert k["e2e_decision"] == ""


# R-10 (P3): sessions[] entry must carry throughput_before/after, a date, and
# stack actions. Regression: rows had 0.0 / "" / [] for these fields.
def test_session_entry_carries_throughput_date_and_actions(
    tmp_path: Path,
) -> None:
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.baseline_tput = 2000.0
    ss.current_best = {
        "name": "tuned", "tput": 2150.0,
        "extra_sglang_args": "--page-size 32",
    }
    ss.optimization_stack = [
        {"action": "explore", "variant_name": "page32"},
        {"action": "explore", "variant_name": "stream_interval_4"},
    ]
    ss.cumulative_gain_validated = 7.5
    ss.cumulative_gain_validated_stack_len = 2
    attrs = coord._build_recipe_attrs_from_state()
    sessions = attrs.get("sessions") or []
    assert sessions, "no session entry emitted"
    s = sessions[0]
    assert s["throughput_before"] == 2000.0
    assert s["throughput_after"] == 2150.0
    assert s["date"], "session date must not be empty"
    assert s["actions_taken"] == ["page32", "stream_interval_4"]


# R-11 (P4): a per-variant pitfall with an empty variant dict must still carry
# the variant NAME in its description, not collapse to the bare task kind.
def test_pitfall_description_uses_variant_name_not_bare_kind(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)
    task = SimpleNamespace(kind="explore", task_id="t-1")
    coord._record_fact_per_variant(
        task=task,
        source_session_id="sess-1",
        variant_outcome={
            "outcome": "REVERT",
            "variant_name": "page64_no_radix",
            "variant": {},
            "metrics": {"gain_pct": -10.0},
        },
    )
    row = coord.cortex_kb.get_recipe(canonical_id=_expected_cid())
    descs = [p.get("description") for p in (row.get("pitfalls") or [])]
    assert any("page64_no_radix" in (d or "") for d in descs), descs
    assert not any(
        (d or "") == f"[{_FW}] explore → regress on {_MODEL}/{_HW}"
        for d in descs
    ), descs
