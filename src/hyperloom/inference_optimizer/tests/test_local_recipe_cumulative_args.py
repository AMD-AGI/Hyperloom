# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe writers preserve the measured cumulative launch and repair tied rows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.grid_server_args import normalize_server_args
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from hyperloom.orchestrator.knowledge.recipe_journal import RecipeJournalCollaborator, recipe_attrs_from_state
from hyperloom.orchestrator.knowledge.recipe_kb import RecipeUpsert

from .test_coordinator_kb_writes import _FW, _FWV, _HW, _MODEL, _PREC, _expected_cid, _make_coordinator, _promote


MTP = '--speculative-config {"method":"mtp","num_speculative_tokens":1}'
INITIAL = MTP + " --max-num-batched-tokens 8192"
SLOTS = "--max-num-seqs 16"
FULL = INITIAL + " " + SLOTS


@pytest.mark.parametrize(
    "current,stack,expected",
    [
        (FULL, {"extra_server_args": FULL, "candidate_extra_server_args": SLOTS}, FULL),
        ("", {"extra_server_args": "--removed", "candidate_extra_server_args": "--removed"}, ""),
        (INITIAL, {"candidate_extra_server_args": SLOTS}, INITIAL),
        (FULL, {"extra_server_args": FULL, "candidate_extra_server_args": ""}, FULL),
    ],
)
def test_local_snapshot_preserves_the_promoted_launch(current, stack, expected):
    state = SimpleNamespace(
        current_best={"extra_server_args": current, "extra_envs": {"TUNE": "1"}, "tput": 1200.0},
        optimization_stack=[stack],
        recipe_sediment_enabled=False,
        gain_per_stack_entry=[],
        last_action_failures=[],
        cumulative_gain_validated_ts="",
        cumulative_gain_validated=20.0,
        cumulative_gain_validated_stack_len=1,
        recipe_kb_session_id="",
        baseline_tput=1000.0,
        framework="vllm",
        model_name="Qwen3.5-9B",
        model_class="Qwen3_5ForCausalLM",
        precision="bf16",
        stack_fingerprint_meta={},
        baseline_workload_extra={},
    )
    config = recipe_attrs_from_state(state, "snapshot")["best_config"]
    assert config["extra_server_args"] == expected
    assert config["extra_envs"] == {"TUNE": "1"}
    assert config["tput"] == 1200.0


@pytest.mark.parametrize(
    "live_tput,live_args,live_envs,replace",
    [
        (1200.0, SLOTS, {}, True),
        (1200.0, FULL, {"OLD": "1"}, True),
        (1200.0, FULL, {}, False),
        (1200.0, "  " + FULL + "  ", {}, False),
        (1200.001, SLOTS, {}, False),
        (1300.0, SLOTS, {}, False),
        (1100.0, FULL, {}, True),
    ],
)
def test_recipe_selection_repairs_only_a_tied_launch_difference(live_tput, live_args, live_envs, replace):
    live = {
        "best_throughput": live_tput,
        "best_config": {"extra_server_args": live_args, "extra_envs": live_envs, "name": "historical", "accuracy": 0.9},
    }
    overrides = RecipeJournalCollaborator.kb_best_config_overrides(
        live=live, best_config={"extra_server_args": FULL, "extra_envs": {}, "name": "current", "tput": 1200.0}
    )
    assert bool(overrides) is replace
    if replace:
        assert normalize_server_args(overrides["best_config"]["extra_server_args"]) == normalize_server_args(FULL)
        assert overrides["best_throughput"] == 1200.0


@pytest.mark.parametrize(
    "winner,expected",
    [
        ({"extra_server_args": SLOTS}, FULL),
        ({"extra_server_args": MTP, "remove_args": ["--max-num-batched-tokens"], "args_mode": "replace"}, MTP),
        (
            {
                "extra_server_args": "",
                "remove_args": ["--speculative-config", "--max-num-batched-tokens"],
                "args_mode": "replace",
            },
            "",
        ),
        ({"extra_server_args": "--no-enable-prefix-caching"}, INITIAL),
    ],
)
def test_real_promotions_preserve_replacement_and_publication_filter(tmp_path: Path, winner, expected):
    coord = _make_coordinator(tmp_path)
    _promote(coord, "initial", 1100.0, INITIAL)
    assert coord.writeback.lift_to_current_best("explore", 1200.0, {"name": "next", **winner})
    config = recipe_attrs_from_state(coord.shared_state, coord.session_dir.name)["best_config"]
    assert normalize_server_args(config["extra_server_args"]) == normalize_server_args(expected)
    assert config["extra_server_args"] == coord.shared_state.current_best["extra_server_args"]
    assert "--no-enable-prefix-caching" not in config["extra_server_args"]


def test_two_keep_amends_and_close_replay_the_complete_stack(tmp_path: Path):
    coord = _make_coordinator(tmp_path)
    for name, args, tput in [("initial", INITIAL, 1100.0), ("slots", SLOTS, 1200.0)]:
        _promote(coord, name, tput, args)
        coord.recipe_journal._record_fact_per_variant(
            task=SimpleNamespace(kind="explore", task_id=name, params={}),
            source_session_id="recipe-test",
            variant_outcome={
                "outcome": "KEEP",
                "variant_name": name,
                "variant": {"extra_server_args": args},
                "metrics": {"gain_pct": 10.0, "output_throughput": tput},
            },
            adopted=True,
        )
    cid = _expected_cid()
    row = coord.recipe_kb.get_authoritative_recipe(canonical_id=cid)
    assert normalize_server_args(row["best_config"]["extra_server_args"]) == normalize_server_args(FULL)
    assert row["best_throughput"] == 1200.0
    assert coord.writeback.validate(1200.0, {"output_throughput": 1200.0})
    assert coord.recipe_journal.finalize_recipe_and_journal()["result_type"] == "written"
    replay_row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert replay_row["best_config"] == row["best_config"]
    assert replay_row["best_throughput"] == 1200.0
    assert [item["name"] for item in replay_row["what_worked"]] == ["initial", "slots"]


def test_source_patch_keep_preserves_args_and_cumulative_envs(tmp_path: Path):
    coord = _make_coordinator(tmp_path)
    assert coord.writeback.lift_to_current_best(
        "explore", 1100.0, {"name": "initial", "extra_server_args": INITIAL, "extra_envs": {"TUNE": "1", "OLD": "1"}}
    )
    result = {"output_throughput": 1200.0, "delta_pct": 10.0, "extra_envs": {"PATCH": "1"}, "unset_envs": ["OLD"]}
    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, dict(result))
    coord.recipe_journal._record_fact_per_task(
        task=SimpleNamespace(kind="integrate_patch", task_id="patch", params={}),
        source_session_id="recipe-test",
        result_dict=result,
        verdict=Verdict.ADOPTED,
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert normalize_server_args(row["best_config"]["extra_server_args"]) == normalize_server_args(INITIAL)
    assert row["best_config"]["extra_envs"] == {"TUNE": "1", "PATCH": "1"}
    assert row["best_throughput"] == 1200.0


@pytest.mark.parametrize("validated", [True, False])
def test_close_repairs_a_legacy_tied_row_only_after_validation(tmp_path: Path, validated: bool):
    coord = _make_coordinator(tmp_path)
    _promote(coord, "initial", 1100.0, INITIAL)
    _promote(coord, "slots", 1200.0, SLOTS)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={"extra_server_args": SLOTS, "tput": 1200.0},
            best_throughput=1200.0,
            what_worked=[{"name": "old", "gain_pct": 20.0}],
        )
    )
    if validated:
        assert coord.writeback.validate(1200.0, {"output_throughput": 1200.0})
    outcome = coord.recipe_journal.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    if validated:
        assert outcome["result_type"] == "written"
        assert normalize_server_args(row["best_config"]["extra_server_args"]) == normalize_server_args(FULL)
        assert [item["name"] for item in row["what_worked"]] == ["initial", "slots"]
    else:
        assert outcome["result_type"] == "unvalidated_recipe"
        assert outcome["reason"] == "unvalidated_recipe_stack"
        assert row["best_config"]["extra_server_args"] == SLOTS
        assert [item["name"] for item in row["what_worked"]] == ["old"]
    assert row["best_throughput"] == 1200.0
