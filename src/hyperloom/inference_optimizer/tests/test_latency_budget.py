# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--max-latency-ms``: one veto on KEEP, and the contract that feeds it.

The gate fails closed, so what reaches it is as much the subject here as the
decision itself: a lane that does not carry ``e2el_mean_ms`` refuses every KEEP
it would ever have made, and that is a bug in the lane, not in the lookup. The
lane tests therefore drive each executor's real return dict rather than a
fixture already shaped the way the gate wants.
"""

from __future__ import annotations

import argparse

import pytest

from hyperloom.common.perf_metric import (
    VERDICT_KEEP,
    VERDICT_REVERT,
    latency_veto_reason,
)
from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState, resolve_graded_comparison


class TestPredicate:
    """The whole decision, in one function."""

    def test_a_candidate_under_the_ceiling_is_not_vetoed(self):
        assert latency_veto_reason(150.0, 200.0) == ""

    def test_a_candidate_over_the_ceiling_is_vetoed(self):
        assert latency_veto_reason(1211.0, 250.0) == "latency_budget_exceeded"

    def test_the_ceiling_itself_passes(self):
        """A ceiling is a maximum, so equality is inside it."""
        assert latency_veto_reason(250.0, 250.0) == ""

    @pytest.mark.parametrize("missing", [None, "", "183", float("nan"), 0.0, -1.0, True])
    def test_an_unmeasured_candidate_is_refused_not_admitted(self, missing):
        """Fail closed: a constraint nobody measured is not one anybody satisfied."""
        assert latency_veto_reason(missing, 250.0) == "latency_unmeasured"

    @pytest.mark.parametrize("off", [0.0, None])
    def test_no_budget_vetoes_nothing(self, off):
        """Off by default, including for a candidate that reported nothing."""
        assert latency_veto_reason(5000.0, off) == ""
        assert latency_veto_reason(None, off) == ""


class TestOneVetoChannel:
    """The SLA rides the verdict the gain gates already decide."""

    def _state(self, budget: float) -> SharedState:
        state = SharedState()
        state.baseline_tput = 1000.0
        state.latency_budget_ms = budget
        return state

    def test_the_motivating_case_a_throughput_win_that_breaks_the_sla(self):
        """+20% aggregate throughput at 1211 ms against a 250 ms SLA is a REVERT.

        The case the flag exists for: a lever that raises throughput *by* making
        each stream slower. Graded on throughput alone this is the best candidate
        on offer, which is exactly the problem.
        """
        graded = resolve_graded_comparison(
            self._state(250.0),
            {"output_throughput": 1200.0, "e2el_mean_ms": 1211.0},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == "latency_budget_exceeded"

    def test_the_same_win_inside_the_sla_still_keeps(self):
        graded = resolve_graded_comparison(
            self._state(250.0),
            {"output_throughput": 1200.0, "e2el_mean_ms": 183.0},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_KEEP
        assert graded.veto_reason == ""

    def test_an_untimed_winner_is_refused_under_a_budget(self):
        graded = resolve_graded_comparison(
            self._state(250.0),
            {"output_throughput": 1200.0},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == "latency_unmeasured"

    def test_an_untimed_winner_keeps_when_no_budget_is_set(self):
        """Off by default: KEEP behaviour is exactly as it was when unset."""
        graded = resolve_graded_comparison(
            self._state(0.0),
            {"output_throughput": 1200.0},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_KEEP
        assert graded.veto_reason == ""

    def test_a_veto_is_distinguishable_from_a_candidate_that_simply_did_not_gain(self):
        """Both are REVERT; only the reason says which, and they need opposite responses."""
        state = self._state(250.0)
        no_gain = resolve_graded_comparison(
            state,
            {"output_throughput": 900.0, "e2el_mean_ms": 100.0},
            keep_threshold_pct=1.0,
        )
        vetoed = resolve_graded_comparison(
            state,
            {"output_throughput": 1200.0, "e2el_mean_ms": 1211.0},
            keep_threshold_pct=1.0,
        )
        assert no_gain.verdict == vetoed.verdict == VERDICT_REVERT
        assert no_gain.veto_reason == ""
        assert vetoed.veto_reason == "latency_budget_exceeded"


class TestLaneResultShapes:
    """Every lane must hand the gate the field it grades on.

    Built from each executor's own return-dict construction rather than from a
    dict already carrying the canonical key: the gate fails closed, so a lane
    that forgets the field loses every KEEP it would have made, and a fixture
    that supplies it cannot catch that.
    """

    def _variant(self) -> VariantResult:
        return VariantResult(
            name="cpx-2-streams",
            extra_server_args="--tp 8",
            extra_envs={},
            status="succeeded",
            output_throughput=1200.0,
            ttft_mean_ms=40.0,
            e2el_mean_ms=1211.0,
            tpot_mean_ms=12.0,
        )

    def test_variant_result_carries_the_canonical_name(self):
        """Pins the attribute the lanes copy: a rename must not silently read None."""
        assert self._variant().e2el_mean_ms == 1211.0
        assert "e2el_mean_ms" in self._variant().to_dict()

    def test_the_explore_variant_dict_carries_it(self):
        """Explore promotes ``VariantResult.to_dict()`` rows directly."""
        assert self._variant().to_dict()["e2el_mean_ms"] == 1211.0

    def test_the_specialist_rebench_dict_carries_it(self, tmp_path, monkeypatch):
        """The lane's own return dict, produced by calling it with the benchmark stubbed."""
        import asyncio

        from hyperloom.orchestrator.specialists import rebench

        monkeypatch.setattr(rebench, "materialize_config_with_envs", lambda *a, **k: tmp_path / "cfg.yaml")
        monkeypatch.setattr(rebench, "_current_leased_cards", lambda: "0")

        async def _fake_run_grid(**_kwargs):
            return [self._variant()]

        monkeypatch.setattr(rebench, "run_grid", _fake_run_grid)
        result = asyncio.run(
            rebench.run_specialist_rebench(config_path=None, output_dir=tmp_path, port=8000),
        )
        assert result["e2el_mean_ms"] == 1211.0

    def test_the_integrate_patch_lift_carries_it(self):
        """``_integrate_measurement_fields`` builds the dict integrate_patch promotes."""
        from hyperloom.orchestrator.loop.writeback import _integrate_measurement_fields

        fields = _integrate_measurement_fields(self._variant().to_dict())
        assert fields["e2el_mean_ms"] == 1211.0

    def test_a_lane_that_forgets_the_field_loses_its_keep(self):
        """Why the lane tests exist: the gate cannot tell an untimed candidate
        from an unplumbed one, so the plumbing is part of the contract."""
        state = SharedState()
        state.baseline_tput = 1000.0
        state.latency_budget_ms = 250.0
        unplumbed = {k: v for k, v in self._variant().to_dict().items() if k != "e2el_mean_ms"}
        graded = resolve_graded_comparison(state, unplumbed, keep_threshold_pct=1.0)
        assert graded.veto_reason == "latency_unmeasured"


class TestLiftRefusesAndSaysWhy:
    """The promotion choke point honours the veto and leaves an operator a trail."""

    def _coord(self, tmp_path, budget: float):
        from hyperloom.orchestrator.loop.coordinator import Coordinator

        coord = Coordinator.__new__(Coordinator)
        coord.session_dir = tmp_path
        coord.shared_state = SharedState(
            baseline_tput=1000.0,
            latency_budget_ms=budget,
            model_path="/models/m",
            gpu_type="mi355x",
        )
        return coord

    def _winner(self, **over):
        return {
            "name": "cpx-2-streams",
            "extra_server_args": "--tp 8",
            "output_throughput": 1200.0,
            **over,
        }

    def test_an_over_budget_winner_does_not_reach_current_best(self, tmp_path):
        coord = self._coord(tmp_path, 250.0)
        assert coord._lift_to_current_best("explore", 1200.0, self._winner(e2el_mean_ms=1211.0)) is False
        assert not coord.shared_state.current_best
        assert not coord.shared_state.optimization_stack

    def test_the_refusal_is_recorded_with_what_it_measured(self, tmp_path):
        """A session that ends near baseline under an SLA must be distinguishable
        from one that found no headroom."""
        coord = self._coord(tmp_path, 250.0)
        coord._lift_to_current_best("explore", 1200.0, self._winner(e2el_mean_ms=1211.0))
        (refusal,) = coord.shared_state.latency_refusals
        assert refusal["reason"] == "latency_budget_exceeded"
        assert refusal["variant_name"] == "cpx-2-streams"
        assert refusal["e2el_mean_ms"] == 1211.0
        assert refusal["budget_ms"] == 250.0

    def test_an_untimed_winner_is_refused_as_untimed_not_as_slow(self, tmp_path):
        """The two reasons need opposite responses: one needs a different
        candidate, the other needs the benchmark to report latency at all."""
        coord = self._coord(tmp_path, 250.0)
        assert coord._lift_to_current_best("integrate_patch", 1200.0, self._winner()) is False
        assert coord.shared_state.latency_refusals[0]["reason"] == "latency_unmeasured"
        assert coord.shared_state.latency_refusals[0]["e2el_mean_ms"] is None

    def test_an_in_budget_winner_is_promoted_and_records_nothing(self, tmp_path):
        coord = self._coord(tmp_path, 250.0)
        assert coord._lift_to_current_best("explore", 1200.0, self._winner(e2el_mean_ms=183.0)) is True
        assert coord.shared_state.current_best
        assert coord.shared_state.latency_refusals == []

    def test_with_no_budget_an_untimed_winner_still_promotes(self, tmp_path):
        """Off by default: KEEP behaviour is exactly as it was when unset."""
        coord = self._coord(tmp_path, 0.0)
        assert coord._lift_to_current_best("explore", 1200.0, self._winner()) is True
        assert coord.shared_state.latency_refusals == []


class TestBaselineFailsClosedAtTheBoundary:
    """An over-budget baseline is knowable at launch; do not spend the run on it."""

    def test_the_stop_reason_is_in_the_closed_vocabulary(self):
        """PolicyGate rejects anything outside it, so an unregistered value would
        silently degrade into "the run did not stop"."""
        from hyperloom.orchestrator.phases.machine_state import is_valid_stop_reason

        assert is_valid_stop_reason("baseline_over_latency_budget")

    def test_setting_it_takes(self):
        state = SharedState()
        assert state.set_stop_reason("baseline_over_latency_budget") == "baseline_over_latency_budget"
        assert state.stop_reason == "baseline_over_latency_budget"


class TestCliValidation:
    """The switch on a fail-closed gate must not itself fail open."""

    def _parse(self, *argv: str) -> argparse.Namespace:
        from hyperloom.inference_optimizer.cli.parser import _build_parser

        return _build_parser().parse_args(["optimize", "--model", "/m", *argv])

    def test_a_budget_is_parsed(self):
        assert self._parse("--max-latency-ms", "250").max_latency_ms == 250.0

    def test_omitting_it_leaves_the_gate_off(self):
        assert self._parse().max_latency_ms is None

    @pytest.mark.parametrize("bad", ["250ms", "abc", "0", "-5", "nan", "inf"])
    def test_an_unusable_value_stops_the_launch_rather_than_disabling_the_sla(self, bad):
        """The old failure: a bad value resolved to "no budget", so the operator
        believed an SLA was enforced while every candidate passed."""
        with pytest.raises(SystemExit) as exc:
            self._parse("--max-latency-ms", bad)
        assert exc.value.code == 2


class TestSessionCarriesTheOnlyCopy:
    """One copy of the budget: state, written at launch, archived for resume."""

    def test_the_launch_flag_lands_on_state(self):
        from hyperloom.orchestrator.state.shared_state import SharedState as S

        assert S(latency_budget_ms=250.0).latency_budget_ms == 250.0

    def test_a_resume_restores_it_from_the_archived_state(self, tmp_path):
        """No second source to reconcile: the value round-trips through state.json."""
        state = SharedState(latency_budget_ms=250.0)
        state.save(tmp_path)
        assert SharedState.load_or_init(tmp_path).latency_budget_ms == 250.0

    def test_it_defaults_to_off(self):
        assert SharedState().latency_budget_ms == 0.0


class TestPromptBlock:
    """What the router sees. Rendered, not asserted against source text."""

    def test_no_block_when_no_budget(self):
        assert SharedState().to_latency_budget_summary() == ""

    def test_the_constraint_is_stated_when_set(self):
        state = SharedState(latency_budget_ms=250.0)
        block = state.to_latency_budget_summary()
        assert "250 ms" in block
        assert "refused   : none so far" in block

    def test_refusals_are_listed_so_a_binding_sla_is_visible(self):
        state = SharedState(latency_budget_ms=250.0)
        state.latency_refusals = [
            {"variant_name": "cpx-2-streams", "action": "explore", "e2el_mean_ms": 1211.0},
            {"variant_name": "qpx-4", "action": "integrate_patch", "e2el_mean_ms": None},
        ]
        block = state.to_latency_budget_summary()
        assert "2 winner(s)" in block
        assert "cpx-2-streams (explore): 1211 ms" in block
        # An untimed refusal must not read as a measured one.
        assert "qpx-4 (integrate_patch): not measured" in block

    def test_the_list_is_capped_and_says_so(self):
        state = SharedState(latency_budget_ms=250.0)
        state.latency_refusals = [
            {"variant_name": f"v{i}", "action": "explore", "e2el_mean_ms": 900.0} for i in range(8)
        ]
        block = state.to_latency_budget_summary()
        assert "8 winner(s)" in block
        assert "(+3 more elided" in block
