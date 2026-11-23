# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A change this session saw stop a server before it was ready is not launched again inside a bigger one."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.explore_prefilter.dead_changes import (
    KNOWN_DEAD_CHANGE,
    carried_dead_change,
    dead_changes,
)
from hyperloom.orchestrator.actions.executors.grid._grid_base import GridVariant

_WS = "ws-1"
_AITER_ON = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "1"}
_MOE_DEATH = (
    "(EngineCore_DP0 pid=7) ERROR [core.py:700] ValueError: Unquantized MoE backend 'ROCm AITER' does not support "
    "the deployment configuration.\n"
    "(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above."
)
_EMPTY_STACK: dict[str, Any] = {"extra_server_args": "", "extra_envs": {}}


def _failed(name: str, envs: dict[str, str], **row: Any) -> dict[str, Any]:
    return {
        "name": name,
        "round_id": "explore-001",
        "outcome": "FAILED",
        "workload_signature": _WS,
        "extra_server_args": "",
        "extra_envs": dict(envs),
        "error_class": "server_init_dead",
        "error_excerpt": _MOE_DEATH,
        **row,
    }


def _variant(envs: dict[str, str], args: str = "") -> dict[str, Any]:
    return {"extra_server_args": args, "extra_envs": dict(envs)}


def test_a_variant_carrying_a_dead_change_whole_is_named_with_the_cause():
    dead = dead_changes([_failed("v00_enable_aiter_moe", _AITER_ON)], workload_signature=_WS)
    found = carried_dead_change(_variant(_AITER_ON, "--cuda-graph-max-bs 64"), dead, stack=_EMPTY_STACK)
    assert found is not None
    assert "v00_enable_aiter_moe" in found.detail()
    assert "ValueError: Unquantized MoE backend 'ROCm AITER'" in found.detail()


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(_variant({"VLLM_ROCM_USE_AITER": "1"}), id="part-of-the-change"),
        pytest.param(_variant({**_AITER_ON, "VLLM_ROCM_USE_AITER_MOE": "0"}), id="another-value"),
        pytest.param(_variant({}, "--no-enforce-eager"), id="unrelated-change"),
    ],
)
def test_a_variant_that_does_not_carry_the_whole_change_launches(variant):
    dead = dead_changes([_failed("v00", _AITER_ON)], workload_signature=_WS)
    assert carried_dead_change(variant, dead, stack=_EMPTY_STACK) is None


def test_a_stack_that_already_serves_the_change_lifts_it():
    dead = dead_changes([_failed("v00", _AITER_ON)], workload_signature=_WS)
    stack = {"extra_server_args": "", "extra_envs": dict(_AITER_ON)}
    assert carried_dead_change(_variant(_AITER_ON, "--max-num-seqs 128"), dead, stack=stack) is None


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"error_excerpt": "torch.OutOfMemoryError: HIP out of memory"}, id="memory"),
        pytest.param({"error_excerpt": "a concurrent GPU holder remained"}, id="host"),
        pytest.param({"error_class": "engine_capacity"}, id="capacity"),
        pytest.param({"error_class": "round_timeout"}, id="not-a-start-failure"),
        pytest.param({"framework_changed": True}, id="code-change"),
        pytest.param({"workload_signature": "ws-2"}, id="other-workload"),
        pytest.param({"outcome": "REVERT"}, id="measured"),
        pytest.param({"extra_envs": {}}, id="no-change"),
    ],
)
def test_only_a_start_failure_of_the_variants_own_change_counts(row):
    assert dead_changes([_failed("v00", _AITER_ON, **row)], workload_signature=_WS) == []


def test_a_decision_round_failure_carries_its_log_text_as_the_reason():
    row = _failed("v00", _AITER_ON, error_excerpt=None, reason=_MOE_DEATH)
    (dead,) = dead_changes([row], workload_signature=_WS)
    assert dead.cause.startswith("ValueError: Unquantized MoE backend")


def _round(tested: dict[str, Any], ledger: dict[str, Any] | None = None) -> Any:
    from hyperloom.orchestrator.actions.executors.explore import _ExploreRound

    rnd = _ExploreRound.__new__(_ExploreRound)
    rnd.params = {}
    rnd.ss = SimpleNamespace(explore_search={"tested": ledger or {}})
    rnd.tested = dict(tested)
    rnd.skipped = []
    rnd.ws_sig = _WS
    rnd.round_id = "explore-002"
    rnd.running_tput = 100.0
    rnd.accuracy_reference = SimpleNamespace(score=None)
    rnd.args = ""
    rnd.envs = {}
    rnd.remove_args = []
    rnd.unset_envs = []
    rnd.args_mode = "append"
    return rnd


@pytest.mark.parametrize("earlier", ["this_round", "an_earlier_round"])
def test_explore_skips_siblings_of_a_variant_whose_server_died_at_init(earlier):
    """The #2751 run: after v00 died, v02 and v03 carried the same AITER change and died the same way."""
    death = {"fp-v00": _failed("v00_enable_aiter_moe", _AITER_ON)}
    rnd = _round(death) if earlier == "this_round" else _round({}, ledger=death)
    sibling = GridVariant("v02_aiter_moe_plus_no_eager", "--no-enforce-eager", dict(_AITER_ON))
    unrelated = GridVariant("v01_remove_eager_mode", "--no-enforce-eager")

    assert rnd._carries_dead_change(sibling) is True
    assert rnd._carries_dead_change(unrelated) is False
    assert [(s["name"], s["reason"]) for s in rnd.skipped] == [("v02_aiter_moe_plus_no_eager", KNOWN_DEAD_CHANGE)]
    assert "v00_enable_aiter_moe" in rnd.skipped[0]["detail"]
