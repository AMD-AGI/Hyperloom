# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A champion that was never accuracy-checked must not become the KB's champion.

The recipe stores ``best_config`` and ``best_throughput`` and no accuracy at
all, so once a config is written there nothing downstream can tell a safe one
from a ruinous one — the next session simply replays it. The only place that
distinction still exists is the session that produced it, which makes CLOSE the
last point where it can be enforced.

Also covers the ledger side: ``explore_search.accepted`` recorded no accuracy at
all (188 of 188 high-risk variants in the retained pool), so after the fact
there was no way to tell whether a kept config had ever been judged.
"""

from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.state.shared_state import SharedState

RISKY = {"extra_server_args": "--kv-cache-dtype fp8", "extra_envs": {}}
SAFE = {"extra_server_args": "--max-num-seqs 2048", "extra_envs": {}}
BASELINE = 0.90


def _collab(baseline_accuracy: float, stack: list | None = None) -> WritebackCollaborator:
    c = WritebackCollaborator.__new__(WritebackCollaborator)
    state = SharedState()
    state.baseline_accuracy = baseline_accuracy
    state.optimization_stack = list(stack or [])
    c.shared_state = state
    return c


def _stack_entry(action: str, accuracy):
    return {
        "action": action,
        "candidate_extra_server_args": RISKY["extra_server_args"],
        "extra_envs": {},
        "accuracy": accuracy,
        "tput": 1000.0,
    }


class TestChampionAccuracyGate:
    def test_high_risk_champion_without_a_verdict_is_withheld(self):
        """The state that produced the 45 bad promotions: a high-risk config
        whose accuracy was never recorded."""
        c = _collab(BASELINE, stack=[_stack_entry("explore", None)])
        assert c._champion_accuracy_ok({"best_config": RISKY}) is False

    def test_high_risk_champion_with_a_collapsed_verdict_is_withheld(self):
        c = _collab(BASELINE, stack=[_stack_entry("explore", 0.20)])
        assert c._champion_accuracy_ok({"best_config": RISKY}) is False

    def test_high_risk_champion_with_a_passing_verdict_is_written(self):
        c = _collab(BASELINE, stack=[_stack_entry("explore", 0.89)])
        assert c._champion_accuracy_ok({"best_config": RISKY}) is True

    def test_a_warm_replay_champion_is_judged_on_the_same_evidence(self):
        """Reading the verdict off the stack rather than per lane is the point:
        a champion promoted by warm replay carries no explore ledger row, and
        reconstructing evidence lane by lane would withhold its write even
        though the replay gate had verified it."""
        c = _collab(BASELINE, stack=[_stack_entry("replay_warm_recipe", 0.89)])
        assert c._champion_accuracy_ok({"best_config": RISKY}) is True

    def test_only_the_promotion_that_produced_the_champion_counts(self):
        """An earlier passing layer does not vouch for a later unverified one."""
        c = _collab(
            BASELINE,
            stack=[_stack_entry("explore", 0.90), _stack_entry("explore", None)],
        )
        assert c._champion_accuracy_ok({"best_config": RISKY}) is False

    def test_a_champion_with_no_high_risk_knob_is_unaffected(self):
        """Those cannot change numerics; gating them would stall the KB."""
        c = _collab(BASELINE, stack=[])
        assert c._champion_accuracy_ok({"best_config": SAFE}) is True

    def test_without_a_baseline_there_is_no_verdict_to_demand(self):
        """Mirrors ``accuracy_passed``: a non-positive baseline skips the gate,
        so an eval-less setup is not locked out of the KB forever."""
        c = _collab(0.0, stack=[_stack_entry("explore", None)])
        assert c._champion_accuracy_ok({"best_config": RISKY}) is True

    def test_a_verdict_carried_on_best_config_is_accepted(self):
        """``_build_recipe_attrs_from_state`` copies an ``accuracy`` key off
        ``current_best`` when one is present; honour it as evidence."""
        c = _collab(BASELINE, stack=[])
        assert c._champion_accuracy_ok({"best_config": {**RISKY, "accuracy": 0.89}}) is True


class TestLedgerRecordsTheVerdict:
    def test_accepted_entry_carries_the_accuracy_it_was_judged_on(self):
        state = SharedState()
        state.record_explore_accepted(
            {
                "name": "kv-fp8",
                "extra_server_args": "--kv-cache-dtype fp8",
                "fingerprint": "aa" * 8,
                "gain_pct": 29.8,
                "accuracy": 0.8912,
            }
        )
        entry = state.explore_search["accepted"][0]
        assert entry["accuracy"] == 0.8912

    def test_an_ungated_variant_records_none_rather_than_omitting_the_key(self):
        """``None`` has to mean "never judged"; a missing key is indistinguishable
        from a zero score to anything reading the ledger later."""
        state = SharedState()
        state.record_explore_accepted(
            {
                "name": "seqs",
                "extra_server_args": "--max-num-seqs 2048",
                "fingerprint": "bb" * 8,
                "gain_pct": 1.2,
            }
        )
        entry = state.explore_search["accepted"][0]
        assert "accuracy" in entry
        assert entry["accuracy"] is None
