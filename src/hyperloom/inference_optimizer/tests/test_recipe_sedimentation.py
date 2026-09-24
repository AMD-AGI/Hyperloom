# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for KEEP/REVERT recipe sedimentation and the warm-start closure."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB


def _make_coordinator(tmp_path: Path) -> Coordinator:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "critic": MockBackend(idle),
    }
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    return Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=kb,
        knowledge_plane=None,
    )


def test_collect_attempt_provenance_maps_keep_and_revert(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.upsert_gap(
        {
            "canonical_id": "gap.research_hint.0",
            "symptom": "enable MTP",
            "layer": "research_hint",
            "source": "research_scout",
            "provenance": "https://pr/123",
        }
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "mtp_on",
            "outcome": "KEEP",
            "gain_pct": 4.2,
        },
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "bad_flag",
            "outcome": "REVERT",
            "gain_pct": -1.0,
        },
    )

    kept, kept_by_gap, reverted = coord._collect_attempt_provenance()
    assert kept == {"mtp_on": "https://pr/123"}
    assert kept_by_gap == {"gap.research_hint.0": "https://pr/123"}
    assert len(reverted) == 1
    assert reverted[0]["name"] == "bad_flag"
    assert reverted[0]["source"] == "https://pr/123"


def test_provenance_resolves_by_gap_id_when_name_mismatches(tmp_path):
    """A KEEP whose stack name never matches the attempt still sediments its source via ``gap_canonical_id``."""
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "k007",
            "extra_server_args": "--fused",
            "gap_canonical_id": "gap.research_hint.0",
        }
    ]
    ss.gain_per_stack_entry = [3.1]
    ss.upsert_gap(
        {
            "canonical_id": "gap.research_hint.0",
            "symptom": "fuse rmsnorm",
            "layer": "research_hint",
            "source": "research_scout",
            "provenance": "https://pr/777",
        }
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "rmsnorm_fuse",
            "outcome": "KEEP",
            "gain_pct": 3.1,
        },
    )

    attrs = coord._build_recipe_attrs_from_state()
    row = next(x for x in attrs["what_worked"] if x["name"] == "k007")
    assert row["source"] == "https://pr/777"


def test_build_recipe_attrs_sediments_source_and_revert(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.optimization_stack = [{"name": "mtp_on", "extra_server_args": "--mtp"}]
    ss.gain_per_stack_entry = [4.2]
    ss.upsert_gap(
        {
            "canonical_id": "gap.research_hint.0",
            "symptom": "enable MTP",
            "layer": "research_hint",
            "source": "research_scout",
            "provenance": "https://pr/123",
        }
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "mtp_on",
            "outcome": "KEEP",
            "gain_pct": 4.2,
        },
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "bad_flag",
            "outcome": "REVERT",
            "gain_pct": -1.0,
        },
    )

    attrs = coord._build_recipe_attrs_from_state()
    ww = attrs["what_worked"]
    mtp = next(x for x in ww if x["name"] == "mtp_on")
    assert mtp["source"] == "https://pr/123"
    assert mtp["gain_pct"] == 4.2
    wf = attrs["what_failed"]
    assert any(r["name"] == "bad_flag" and r.get("source") == "https://pr/123" for r in wf)


def test_sediment_toggle_off_keeps_recipe_ephemeral(tmp_path):
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.recipe_sediment_enabled = False
    ss.optimization_stack = [{"name": "mtp_on", "extra_server_args": "--mtp"}]
    ss.gain_per_stack_entry = [4.2]
    ss.upsert_gap(
        {
            "canonical_id": "gap.research_hint.0",
            "symptom": "enable MTP",
            "layer": "research_hint",
            "provenance": "https://pr/123",
        }
    )
    ss.append_gap_attempt(
        "gap.research_hint.0",
        {
            "variant_name": "mtp_on",
            "outcome": "KEEP",
            "gain_pct": 4.2,
        },
    )

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


def test_experience_rows_survive_the_kb_round_trip(tmp_path):
    """Every field the Coordinator writes onto an experience row is readable again after persistence.

    This is the seam between the writeback shape and the prelude readers: the
    surrounding tests prove writeback stamps these keys and that prelude reads
    them, but they hand-inject the rows, so the store sits between two proven
    ends without its own coverage.
    """
    store = LocalRecipeStore(root=tmp_path / "kb2")
    cid = "inference:m:h:f:text:a:1:fp4"
    store.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="h",
        framework_name="f",
        # Exactly what writeback._build_recipe_attrs_from_state emits.
        what_worked=[
            {
                "name": "mtp_on",
                "extra_server_args": "--speculative-num-steps 3",
                "extra_envs": {"VLLM_MTP": "1"},
                "gain_pct": 4.2,
                "source": "https://pr/123",
            }
        ],
        what_failed=[{"name": "bad_flag", "reason": "reverted", "gain_pct": -1.0}],
    )
    row = store.get_recipe(canonical_id=cid) or {}

    worked = row.get("what_worked") or []
    assert len(worked) == 1
    assert worked[0].get("name") == "mtp_on"
    assert worked[0].get("source") == "https://pr/123"
    assert worked[0].get("extra_server_args") == "--speculative-num-steps 3"
    assert worked[0].get("extra_envs") == {"VLLM_MTP": "1"}
    assert worked[0].get("gain_pct") == 4.2

    failed = row.get("what_failed") or []
    assert failed[0].get("name") == "bad_flag"
    assert failed[0].get("reason") == "reverted"
    # A REVERT's gain is negative, so a falsy-check would drop exactly the rows this column exists to record.
    assert failed[0].get("gain_pct") == -1.0


def test_an_experience_row_that_stored_a_description_still_names_its_variant(tmp_path):
    """A row whose producer really wrote ``description`` reads back under ``name``.

    This covers arbor-sourced rows only. A Coordinator row went to disk through a
    projection that wrote ``description: ""``, so its variant was destroyed at write
    time and no read-side fallback can recover it — the row stays nameless.
    """
    store = LocalRecipeStore(root=tmp_path / "kb_legacy")
    cid = "inference:m:h:f:text:a:1:fp4"
    store.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="h",
        framework_name="f",
        what_worked=[{"description": "mtp_on", "measured_impact": "+4.2%"}],
        what_failed=[{"description": "", "reason": "OOM"}],
    )
    row = store.get_recipe(canonical_id=cid) or {}
    assert (row.get("what_worked") or [])[0].get("name") == "mtp_on"
    assert not (row.get("what_failed") or [])[0].get("name")


def test_an_experience_row_keeps_a_key_this_module_never_heard_of(tmp_path):
    """A field the Coordinator starts sending reaches disk without being taught here first."""
    store = LocalRecipeStore(root=tmp_path / "kb_forward")
    cid = "inference:m:h:f:text:a:1:fp4"
    store.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="h",
        framework_name="f",
        what_worked=[{"name": "mtp_on", "gain_pct": 4.2, "fingerprint": "abc123"}],
        what_failed=[{"name": "bad_flag", "reason": "reverted", "error_class": "OOMError"}],
    )
    row = store.get_recipe(canonical_id=cid) or {}
    assert (row.get("what_worked") or [])[0] == {"name": "mtp_on", "gain_pct": 4.2, "fingerprint": "abc123"}
    assert (row.get("what_failed") or [])[0] == {
        "name": "bad_flag",
        "reason": "reverted",
        "error_class": "OOMError",
    }


def test_an_experience_row_serialises_to_exactly_what_it_parsed(tmp_path):
    """``Recipe.from_dict`` and ``to_dict`` are inverses, so a re-put cannot erode a row."""
    from hyperloom.orchestrator.knowledge.recipe_kb.schema import Recipe

    payload = {
        "what_worked": [
            {
                "name": "mtp_on",
                "extra_server_args": "--speculative-num-steps 3",
                "extra_envs": {"VLLM_MTP": "1"},
                "gain_pct": 4.2,
                "source": "https://pr/123",
            }
        ],
        "what_failed": [
            {
                "name": "bad_flag",
                "reason": "reverted",
                "extra_server_args": "",
                "extra_envs": {},
                "gain_pct": -1.0,
                "source": "",
            }
        ],
    }
    once = Recipe.from_dict(payload).to_dict()
    assert once["what_worked"] == payload["what_worked"]
    assert once["what_failed"] == payload["what_failed"]
    assert Recipe.from_dict(once).to_dict() == once


@pytest.mark.parametrize(
    "bad_row",
    [
        {"name": "v", "gain_pct": "not-a-number"},
        # A number the producer stringified is still the wrong type: ``writeback`` passes ``float | None``, and a
        # store that quietly parses this one would take the unparseable one above just as quietly.
        {"name": "v", "gain_pct": "4.2"},
        {"name": "v", "extra_envs": ["not", "a", "mapping"]},
    ],
)
def test_a_malformed_experience_row_fails_at_the_boundary(tmp_path, bad_row):
    """``put_recipe`` is the boundary: a bad type raises rather than landing a row with the value silently gone."""
    store = LocalRecipeStore(root=tmp_path / "kb_bad")
    with pytest.raises(TypeError):
        store.put_recipe(
            canonical_id="inference:m:h:f:text:a:1:fp4",
            model="m",
            hardware="h",
            framework_name="f",
            what_worked=[bad_row],
        )


def test_proven_items_reach_the_scout_through_a_stored_recipe(tmp_path):
    """End-to-end: a recipe written to the KB feeds prelude's ``already_proven``."""
    coord = _make_coordinator(tmp_path)
    store = LocalRecipeStore(root=tmp_path / "kb3")
    cid = "inference:m:h:f:text:a:1:fp4"
    store.put_recipe(
        canonical_id=cid,
        model="m",
        hardware="h",
        framework_name="f",
        what_worked=[
            {"name": "mtp_on", "source": "https://pr/123"},
            {"name": "fp8_kv"},
        ],
    )
    # The shape recipe_kb_t0 hands to SharedState on a hit.
    coord.shared_state.warm_start_recipe = {
        "workload": "m",
        "hw": "h",
        "recipe": store.get_recipe(canonical_id=cid) or {},
    }
    proven = coord._warm_recipe_proven_items()
    assert {p["name"] for p in proven} == {"mtp_on", "fp8_kv"}
    assert next(p for p in proven if p["name"] == "mtp_on")["source"] == "https://pr/123"


def test_gap_provenance_round_trips_through_serialization(tmp_path):
    from hyperloom.orchestrator.state.shared_state import SharedState

    ss = SharedState(session_id="s", model_name="m")
    ss.upsert_gap(
        {
            "canonical_id": "gap.research_hint.0",
            "symptom": "x",
            "provenance": "https://pr/9",
        }
    )
    restored = SharedState.from_dict(ss.to_dict())
    assert restored.gaps[0]["provenance"] == "https://pr/9"


def test_research_hints_suppress_cold_start():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        _is_cold_start,
    )

    inp = SpecialistPromptInputs(
        task_id="t",
        domain="serving_specialist",
        research_hints="- enable MTP (source=https://pr/1)",
    )
    assert _is_cold_start(inp) is False


def test_kb_section_renders_research_hints_when_kb_empty():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        _section_kb_subgraph,
    )

    inp = SpecialistPromptInputs(
        task_id="t",
        domain="serving_specialist",
        research_hints="- enable MTP (source=https://pr/1)",
    )
    text = "\n".join(_section_kb_subgraph(inp))
    assert "research scout collected source-backed priors" in text
    assert "enable MTP" in text
    assert "COLD-START MODE" not in text


def test_bare_cold_start_still_uses_domain_focus_fallback():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        _section_kb_subgraph,
    )

    inp = SpecialistPromptInputs(task_id="t", domain="serving_specialist")
    text = "\n".join(_section_kb_subgraph(inp))
    assert "COLD-START MODE" in text


def test_scout_focus_lists_already_proven():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        _focus_research_scout_specialist,
    )

    inp = SpecialistPromptInputs(
        task_id="t",
        domain="research_scout_specialist",
        already_proven=[{"name": "mtp_on", "source": "https://pr/1"}],
    )
    text = "\n".join(_focus_research_scout_specialist(inp))
    assert "Already proven" in text
    assert "mtp_on" in text
    assert "https://pr/1" in text
