# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for KEEP/REVERT recipe sedimentation and the warm-start closure."""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend, MockTurn, ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.recipe_kb import LocalRecipeStore, RecipeKB


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
    return Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=kb,
        knowledge_plane=None,
    )


def test_collect_attempt_provenance_maps_keep_and_revert(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.upsert_gap({
        "canonical_id": "gap.research_hint.0",
        "symptom": "enable MTP",
        "layer": "research_hint",
        "source": "research_scout",
        "provenance": "https://pr/123",
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "mtp_on", "outcome": "KEEP", "gain_pct": 4.2,
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "bad_flag", "outcome": "REVERT", "gain_pct": -1.0,
    })

    kept, kept_by_gap, reverted = coord._collect_attempt_provenance()
    assert kept == {"mtp_on": "https://pr/123"}
    assert kept_by_gap == {"gap.research_hint.0": "https://pr/123"}
    assert len(reverted) == 1
    assert reverted[0]["name"] == "bad_flag"
    assert reverted[0]["source"] == "https://pr/123"


def test_provenance_resolves_by_gap_id_when_name_mismatches(tmp_path):
    """A cross-stage KEEP whose stack name never matches the attempt still sediments its source via ``gap_canonical_id``."""
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.optimization_stack = [{
        "action": "integrate",
        "kernel_id": "k007",
        "extra_server_args": "--fused",
        "gap_canonical_id": "gap.research_hint.0",
    }]
    ss.gain_per_stack_entry = [3.1]
    ss.upsert_gap({
        "canonical_id": "gap.research_hint.0",
        "symptom": "fuse rmsnorm",
        "layer": "research_hint",
        "source": "research_scout",
        "provenance": "https://pr/777",
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "rmsnorm_fuse", "outcome": "KEEP", "gain_pct": 3.1,
    })

    attrs = coord._build_recipe_attrs_from_state()
    row = next(x for x in attrs["what_worked"] if x["name"] == "k007")
    assert row["source"] == "https://pr/777"


def test_build_recipe_attrs_sediments_source_and_revert(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.optimization_stack = [{"name": "mtp_on", "extra_server_args": "--mtp"}]
    ss.gain_per_stack_entry = [4.2]
    ss.upsert_gap({
        "canonical_id": "gap.research_hint.0",
        "symptom": "enable MTP",
        "layer": "research_hint",
        "source": "research_scout",
        "provenance": "https://pr/123",
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "mtp_on", "outcome": "KEEP", "gain_pct": 4.2,
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "bad_flag", "outcome": "REVERT", "gain_pct": -1.0,
    })

    attrs = coord._build_recipe_attrs_from_state()
    ww = attrs["what_worked"]
    mtp = next(x for x in ww if x["name"] == "mtp_on")
    assert mtp["source"] == "https://pr/123"
    assert mtp["gain_pct"] == 4.2
    wf = attrs["what_failed"]
    assert any(r["name"] == "bad_flag" and r.get("source") == "https://pr/123"
               for r in wf)


def test_sediment_toggle_off_keeps_recipe_ephemeral(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.recipe_sediment_enabled = False
    ss.optimization_stack = [{"name": "mtp_on", "extra_server_args": "--mtp"}]
    ss.gain_per_stack_entry = [4.2]
    ss.upsert_gap({
        "canonical_id": "gap.research_hint.0",
        "symptom": "enable MTP",
        "layer": "research_hint",
        "provenance": "https://pr/123",
    })
    ss.append_gap_attempt("gap.research_hint.0", {
        "variant_name": "mtp_on", "outcome": "KEEP", "gain_pct": 4.2,
    })

    attrs = coord._build_recipe_attrs_from_state()
    mtp = next(x for x in attrs["what_worked"] if x["name"] == "mtp_on")
    assert "source" not in mtp


def test_warm_recipe_proven_items(tmp_path):
    coord = _make_coordinator(tmp_path)
    coord.shared_state.warm_start_recipe = {
        "recipe": {
            "attrs": {
                "what_worked": [
                    {"name": "mtp_on", "source": "https://pr/123"},
                    {"name": "fp8_kv"},
                    {"bogus": 1},
                ],
            },
        },
    }
    proven = coord._warm_recipe_proven_items()
    names = {p["name"] for p in proven}
    assert names == {"mtp_on", "fp8_kv"}
    mtp = next(p for p in proven if p["name"] == "mtp_on")
    assert mtp["source"] == "https://pr/123"


def test_warm_recipe_proven_items_empty_without_recipe(tmp_path):
    coord = _make_coordinator(tmp_path)
    assert coord._warm_recipe_proven_items() == []


def test_gap_provenance_round_trips_through_serialization(tmp_path):
    from inference_optimizer.orchestrator.shared_state import SharedState
    ss = SharedState(session_id="s", model_name="m")
    ss.upsert_gap({
        "canonical_id": "gap.research_hint.0",
        "symptom": "x", "provenance": "https://pr/9",
    })
    restored = SharedState.from_dict(ss.to_dict())
    assert restored.gaps[0]["provenance"] == "https://pr/9"


# Cold-start prefers research hints over the domain-focus fallback (C9).
def test_research_hints_suppress_cold_start():
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _is_cold_start,
    )
    inp = SpecialistPromptInputs(
        task_id="t", domain="serving_specialist",
        research_hints="- enable MTP (source=https://pr/1)",
    )
    assert _is_cold_start(inp) is False


def test_kb_section_renders_research_hints_when_kb_empty():
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _section_kb_subgraph,
    )
    inp = SpecialistPromptInputs(
        task_id="t", domain="serving_specialist",
        research_hints="- enable MTP (source=https://pr/1)",
    )
    text = "\n".join(_section_kb_subgraph(inp))
    assert "research scout collected source-backed priors" in text
    assert "enable MTP" in text
    assert "COLD-START MODE" not in text


def test_bare_cold_start_still_uses_domain_focus_fallback():
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _section_kb_subgraph,
    )
    inp = SpecialistPromptInputs(task_id="t", domain="serving_specialist")
    text = "\n".join(_section_kb_subgraph(inp))
    assert "COLD-START MODE" in text


def test_scout_focus_lists_already_proven():
    from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
        SpecialistPromptInputs, _focus_research_scout_specialist,
    )
    inp = SpecialistPromptInputs(
        task_id="t", domain="research_scout_specialist",
        already_proven=[{"name": "mtp_on", "source": "https://pr/1"}],
    )
    text = "\n".join(_focus_research_scout_specialist(inp))
    assert "Already proven" in text
    assert "mtp_on" in text
    assert "https://pr/1" in text
