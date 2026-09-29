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

    def test_a_candidate_that_lost_on_throughput_carries_no_veto_even_over_budget(self):
        """The ledger must blame the gate that refused it; the budget never got a say."""
        graded = resolve_graded_comparison(
            self._state(250.0),
            {"output_throughput": 800.0, "e2el_mean_ms": 1211.0},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == ""


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
        """``integrate_measurement_fields`` builds the dict integrate_patch promotes."""
        from hyperloom.orchestrator.measurement.integrate_performance import integrate_measurement_fields

        fields = integrate_measurement_fields(self._variant().to_dict())
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

    def test_an_untimed_winner_is_refused(self, tmp_path):
        coord = self._coord(tmp_path, 250.0)
        assert coord._lift_to_current_best("integrate_patch", 1200.0, self._winner()) is False
        assert not coord.shared_state.current_best

    def test_an_in_budget_winner_is_promoted(self, tmp_path):
        coord = self._coord(tmp_path, 250.0)
        assert coord._lift_to_current_best("explore", 1200.0, self._winner(e2el_mean_ms=183.0)) is True
        assert coord.shared_state.current_best

    def test_with_no_budget_an_untimed_winner_still_promotes(self, tmp_path):
        """Off by default: KEEP behaviour is exactly as it was when unset."""
        coord = self._coord(tmp_path, 0.0)
        assert coord._lift_to_current_best("explore", 1200.0, self._winner()) is True


class TestIntegrateDecisionsHonourTheVeto:
    """The lanes decide KEEP themselves; each must revert its own over-budget change, not leave it for the lift."""

    def _state(self) -> SharedState:
        return SharedState(baseline_tput=1000.0, latency_budget_ms=250.0)

    def test_the_shared_integrate_decision_reverts_an_over_budget_gain(self):
        """``assess_integrate_performance`` decides kernel integration and GEMM tuning KEEPs."""
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        performance = assess_integrate_performance(
            self._state(),
            {"output_throughput": 1200.0, "e2el_mean_ms": 1211.0},
            base_tput=1000.0,
            keep_threshold_pct=1.0,
            stack_incremental_keep_threshold_pct=0.5,
        )
        assert performance.decision == "REVERT"
        assert performance.graded.veto_reason == "latency_budget_exceeded"

    def test_the_shared_integrate_decision_keeps_an_in_budget_gain(self):
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        performance = assess_integrate_performance(
            self._state(),
            {"output_throughput": 1200.0, "e2el_mean_ms": 183.0},
            base_tput=1000.0,
            keep_threshold_pct=1.0,
            stack_incremental_keep_threshold_pct=0.5,
        )
        assert performance.decision == "KEEP"


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


class TestResumeDoesNotSilentlyKeepTheOldBudget:
    """A resume-time ``--max-latency-ms`` must not fail open."""

    @pytest.mark.parametrize(
        ("archived", "requested"),
        [(0.0, 250.0), (500.0, 250.0), (250.0, 500.0)],
    )
    def test_a_different_value_is_refused(self, archived, requested):
        from hyperloom.inference_optimizer.cli.bootstrap import latency_budget_resume_conflict

        reason = latency_budget_resume_conflict(SharedState(latency_budget_ms=archived), requested)
        assert "--max-latency-ms" in reason
        assert f"{requested:g}" in reason

    @pytest.mark.parametrize(("archived", "requested"), [(250.0, None), (0.0, None), (250.0, 250.0)])
    def test_omitting_or_repeating_the_value_resumes(self, archived, requested):
        from hyperloom.inference_optimizer.cli.bootstrap import latency_budget_resume_conflict

        assert latency_budget_resume_conflict(SharedState(latency_budget_ms=archived), requested) == ""


class TestScope:
    """Only scriptable frameworks can carry a budget; everywhere else the flag is refused, not ignored."""

    @pytest.mark.parametrize("framework", ["sglang", "vllm", "atom", "", None])
    def test_a_serving_framework_is_refused(self, framework):
        from hyperloom.inference_optimizer.cli.bootstrap import latency_budget_scope_error

        assert "scriptable" in latency_budget_scope_error(framework, 250.0)

    @pytest.mark.parametrize("framework", ["xdit", "custom"])
    def test_a_scriptable_framework_is_accepted(self, framework):
        from hyperloom.inference_optimizer.cli.bootstrap import latency_budget_scope_error

        assert latency_budget_scope_error(framework, 250.0) == ""

    def test_omitting_the_flag_is_never_an_error(self):
        from hyperloom.inference_optimizer.cli.bootstrap import latency_budget_scope_error

        assert latency_budget_scope_error("sglang", None) == ""


class TestPromptBlock:
    """What the router sees. Rendered, not asserted against source text."""

    def test_no_block_when_no_budget(self):
        assert SharedState().to_latency_budget_summary() == ""

    def test_the_constraint_is_one_line_pointing_at_the_existing_ledgers(self):
        block = SharedState(latency_budget_ms=250.0).to_latency_budget_summary()
        assert "\n" not in block
        assert "250 ms" in block
        assert "latency_budget_exceeded" in block
        assert "explore_search" in block
