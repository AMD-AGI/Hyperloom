# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the direction families of past sessions came to, and whether their odds tell tests apart."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.blackboard.families import Family, Mechanism
from hyperloom_multibranch.blackboard.rsi.archive import TreeDirection
from hyperloom_multibranch.blackboard.rsi.family_stats import (
    MAX_TRIALS,
    FamilyStat,
    direction_family,
    discrimination,
    family_stats,
)
from hyperloom_multibranch.blackboard.rsi.tests.trees import TASK, branch, tree

SOURCE, CONFIG = IntentLayer.SOURCE, IntentLayer.CONFIG
KV = TreeDirection("kv cache", "--kv-cache-dtype fp8_e4m3", CONFIG, 2, 1, 3.0)
SCHED = TreeDirection("scheduler", "--chunked-prefill-size 16384", CONFIG, 2, 0)
FP8_KV = Family("KV cache", Mechanism.PRECISION, CONFIG)
CHUNKS = Family("scheduling and batching", Mechanism.TUNING, CONFIG)


def _with_type(model_type: str, version: str = "0.5"):
    return dataclasses.replace(TASK, labels={**TASK.labels, "model_type": model_type, "framework_version": version})


def _session(name: str, *directions: TreeDirection, task=None):
    return tree(name, branch("b1", [], directions=directions), task=task or _with_type("qwen3_next"))


def test_an_archived_direction_keeps_the_family_close_read_off_its_change() -> None:
    kept = dataclasses.replace(KV, family_target="attention", family_mechanism="backend or implementation")

    assert direction_family(KV) == FP8_KV
    assert direction_family(kept) == Family("attention", Mechanism.BACKEND, CONFIG)


def test_a_familys_odds_start_at_its_layers_base_rate_worth_four_tests() -> None:
    assert FamilyStat().odds(SOURCE) == pytest.approx(0.24)
    assert FamilyStat(1.0, 1.0).odds(SOURCE) == pytest.approx((1 + 0.24 * 4) / 5)
    assert FamilyStat(gains=(1.0, 5.0, 2.0)).typical_gain == 2.0


def test_evidence_weighs_by_model_type_and_version_and_is_capped() -> None:
    like = _session("like")
    same, other, older = (
        _session("a", KV),
        _session("b", KV, task=_with_type("qwen3")),
        _session("c", KV, task=_with_type("qwen3_next", "0.4")),
    )
    crowded = _session("d", *(dataclasses.replace(SCHED, tests=4) for _ in range(5)))

    stats = family_stats([same, other, older, crowded], like=like.task)

    assert stats[FP8_KV] == FamilyStat(2.0 + 1.0 + 1.0, 1.0 + 0.5 + 0.5, (3.0, 3.0, 3.0))
    assert (stats[CHUNKS].tests, stats[CHUNKS].commits) == (MAX_TRIALS, 0.0)


def test_a_sessions_evidence_halves_for_every_five_archived_sessions_newer_than_it() -> None:
    once = dataclasses.replace(KV, tests=1, commits=0)
    sessions = [dataclasses.replace(_session(f"s{n}", once), started_at=float(n)) for n in range(6)]

    # s0 has five sessions newer than it: its test counts half.
    assert family_stats(sessions)[FP8_KV].tests == pytest.approx(5 * 1.0 + 0.5)


def test_the_offline_check_reads_how_well_the_families_tell_committing_tests_apart() -> None:
    paying = TreeDirection("kv cache", "--kv-cache-dtype fp8_e4m3", CONFIG, 2, 2, 3.0)
    idle = TreeDirection("scheduler", "--chunked-prefill-size 16384", CONFIG, 2, 0)
    sessions = [_session(f"s{n}", paying, idle) for n in range(3)]

    found = discrimination(sessions)

    assert (found.tests, found.commits) == (12, 6)
    assert (found.family_auc, found.layer_auc) == (1.0, 0.5) and found.tells_apart
    assert found.render().startswith("Over 12 archived tests, 6 of them committing")
