# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``--max-latency-ms``: resolution, the KEEP gate, and its reporting.

The gate's whole value is that it holds on paths nobody remembered to wire, so
the promotion tests here go through the real ``_lift_to_current_best`` for each
action kind rather than calling the predicate directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from hyperloom import inference_optimizer
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.actions.executors._latency_budget import (
    LATENCY_BUDGET_ENV,
    REASON_OVER_BUDGET,
    REASON_UNMEASURED,
    describe_latency_budget,
    latency_fields_from_result,
    latency_from_result,
    latency_keep_block,
    read_session_budget,
    resolve_latency_budget_ms,
)
from hyperloom.orchestrator.actions.executors.report import _format_latency_budget_section
from hyperloom.orchestrator.loop import conversation
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture(autouse=True)
def _clean_budget_env(monkeypatch):
    """Keep an inherited budget out of every test's resolution chain."""
    monkeypatch.delenv(LATENCY_BUDGET_ENV, raising=False)


def _coord(session_dir: Path) -> Coordinator:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="orch"),
            "critic": MockCriticBackend(),
            "robustness": MockRobustnessBackend(),
        },
    )


class TestBlockDecision:
    """The predicate itself."""

    def test_no_budget_never_blocks(self):
        assert latency_keep_block(9_999.0, budget_ms=0.0) == (False, "")

    def test_under_budget_passes(self):
        blocked, reason = latency_keep_block(150.0, budget_ms=200.0)
        assert (blocked, reason) == (False, "")

    def test_over_budget_blocks_and_says_by_how_much(self):
        blocked, reason = latency_keep_block(1211.0, budget_ms=200.0)
        assert blocked
        assert REASON_OVER_BUDGET in reason
        assert "6.05x" in reason

    def test_exactly_at_budget_is_allowed(self):
        """The budget is a ceiling the workload owner named, not an exclusive bound."""
        assert latency_keep_block(200.0, budget_ms=200.0) == (False, "")

    def test_unmeasured_fails_closed(self):
        blocked, reason = latency_keep_block(None, budget_ms=200.0)
        assert blocked
        assert REASON_UNMEASURED in reason

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "120"])
    def test_unusable_observations_fail_closed(self, bad):
        """A budget cannot be shown to hold against a non-measurement."""
        blocked, _ = latency_keep_block(bad, budget_ms=200.0)
        assert blocked

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0])
    def test_unusable_budget_disables_the_gate(self, bad):
        """An unusable budget is no budget, matching the CLI default."""
        assert latency_keep_block(5_000.0, budget_ms=bad) == (False, "")


class TestResolution:
    """Where the budget comes from, and in what order."""

    def test_params_win_over_state_and_env(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "300")
        state = type("S", (), {"latency_budget_ms": 200.0})()
        assert resolve_latency_budget_ms({"latency_budget_ms": 100.0}, state) == 100.0

    def test_state_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "300")
        state = type("S", (), {"latency_budget_ms": 200.0})()
        assert resolve_latency_budget_ms({}, state) == 200.0

    def test_env_is_the_last_resort(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "300")
        assert resolve_latency_budget_ms({}, None) == 300.0

    def test_absent_everywhere_is_off(self):
        assert resolve_latency_budget_ms({}, None) == 0.0

    def test_a_zero_tier_falls_through_rather_than_disabling(self, monkeypatch):
        """An unset flag arrives as 0, which must not mask a real budget below it."""
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "300")
        state = type("S", (), {"latency_budget_ms": 0.0})()
        assert resolve_latency_budget_ms({"latency_budget_ms": 0.0}, state) == 300.0

    def test_unparseable_env_is_ignored_not_fatal(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "soon")
        assert resolve_latency_budget_ms({}, None) == 0.0

    def test_session_seed_reads_the_published_env(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "250.5")
        assert read_session_budget() == 250.5

    @pytest.mark.parametrize("raw", ["", "   ", "later", "0", "-3"])
    def test_session_seed_of_an_unusable_value_is_off(self, monkeypatch, raw):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, raw)
        assert read_session_budget() == 0.0


class TestLatencyFromResult:
    """The alias lookup that keeps a fail-closed gate from misfiring."""

    @pytest.mark.parametrize("key", ["e2el_mean_ms", "mean_e2el_ms", "e2el_ms"])
    def test_each_spelling_in_use_is_found(self, key):
        assert latency_from_result({key: 183.0}) == 183.0

    def test_normalized_spelling_wins_when_several_are_present(self):
        payload = {"e2el_ms": 1.0, "mean_e2el_ms": 2.0, "e2el_mean_ms": 3.0}
        assert latency_from_result(payload) == 3.0

    def test_an_unusable_value_does_not_shadow_a_later_key(self):
        """A ``None`` under the preferred name must not hide a real measurement."""
        assert latency_from_result({"e2el_mean_ms": None, "mean_e2el_ms": 250.0}) == 250.0

    @pytest.mark.parametrize("payload", [None, "183", 183, [], {}, {"tput": 100.0}])
    def test_nothing_usable_is_none(self, payload):
        assert latency_from_result(payload) is None


class TestKeepGate:
    """The gate as promotion actually reaches it."""

    def test_an_over_budget_winner_does_not_become_current_best(self, session_dir):
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            "explore",
            1200.0,
            {"name": "cpx-2-streams", "extra_server_args": "--flag", "e2el_mean_ms": 1211.0},
        )

        assert lifted is False
        assert not s.optimization_stack
        assert not (s.current_best or {}).get("variant_name")

    def test_an_under_budget_winner_still_lifts(self, session_dir):
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            "explore",
            1200.0,
            {"name": "good", "extra_server_args": "--flag", "e2el_mean_ms": 183.0},
        )

        assert lifted is True
        assert s.current_best["variant_name"] == "good"

    def test_an_untimed_winner_is_refused_under_a_budget(self, session_dir):
        """Fail-closed: an unmeasured constraint is not a satisfied one."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            "explore",
            1200.0,
            {"name": "untimed", "extra_server_args": "--flag"},
        )

        assert lifted is False
        assert s.latency_refusals[0]["reason"].startswith(REASON_UNMEASURED)

    def test_an_untimed_winner_lifts_when_no_budget_is_set(self, session_dir):
        """The default stays exactly the throughput-only KEEP it was."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0

        lifted = coord._lift_to_current_best(
            "explore",
            1200.0,
            {"name": "untimed", "extra_server_args": "--flag"},
        )

        assert lifted is True
        assert not s.latency_refusals

    @pytest.mark.parametrize(
        "task_kind",
        ["explore", "geak_e2e", "integrate_patch", "framework", "gemm_tuning", "collective", "fusion"],
    )
    def test_the_gate_holds_on_every_promoting_action(self, session_dir, task_kind):
        """The gap this closes: only explore used to consult the budget."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            task_kind,
            1200.0,
            {"name": f"{task_kind}-winner", "extra_server_args": "--flag", "e2el_mean_ms": 1211.0},
        )

        assert lifted is False, f"{task_kind} promoted an over-budget winner"

    def test_a_refusal_is_recorded_with_what_it_cost(self, session_dir):
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        coord._lift_to_current_best(
            "explore",
            1200.0,
            {"name": "cpx", "extra_server_args": "--flag", "e2el_mean_ms": 1211.0},
        )

        (refusal,) = s.latency_refusals
        assert refusal["action"] == "explore"
        assert refusal["variant_name"] == "cpx"
        assert refusal["tput"] == 1200.0
        assert refusal["e2el_mean_ms"] == 1211.0
        assert refusal["budget_ms"] == 200.0
        assert REASON_OVER_BUDGET in refusal["reason"]

    def test_the_anchor_check_still_runs_first(self, session_dir):
        """A winner that beats nothing is refused for that, not for its latency."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            "explore",
            900.0,
            {"name": "slower", "extra_server_args": "--flag", "e2el_mean_ms": 1211.0},
        )

        assert lifted is False
        assert not s.latency_refusals

    def test_a_gate_refusal_leaves_the_prior_champion_intact(self, session_dir):
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        coord._lift_to_current_best(
            "explore",
            1100.0,
            {"name": "first", "extra_server_args": "--flag-a 1", "e2el_mean_ms": 150.0},
        )
        s.latency_budget_ms = 200.0

        coord._lift_to_current_best(
            "explore",
            1500.0,
            {"name": "second", "extra_server_args": "--flag-b 1", "e2el_mean_ms": 900.0},
        )

        assert s.current_best["variant_name"] == "first"
        assert s.current_best["extra_server_args"] == "--flag-a 1"
        assert [e["variant_name"] for e in s.optimization_stack] == ["first"]

    def test_the_gate_reads_a_result_that_used_another_spelling(self, session_dir):
        """A lane reporting ``mean_e2el_ms`` must not be refused as untimed."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0

        lifted = coord._lift_to_current_best(
            "integrate",
            1200.0,
            {"name": "aliased", "extra_server_args": "--flag", "mean_e2el_ms": 150.0},
        )

        assert lifted is True


def _variant_result(**over) -> VariantResult:
    """A succeeded bench result as the grid executors actually produce one."""
    fields = {
        "name": "cand",
        "extra_server_args": "--flag",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1200.0,
        "ttft_mean_ms": 40.0,
        "e2el_mean_ms": 150.0,
        "tpot_mean_ms": 12.0,
    }
    fields.update(over)
    return VariantResult(**fields)


class TestLaneResultShapes:
    """The lanes' real payloads, not a hand-written dict that already complies.

    A fail-closed gate is only as good as what feeds it. Every promotion test
    above hands ``_lift_to_current_best`` a dict with ``e2el_mean_ms`` already on
    it, which is exactly the shape that cannot catch a lane whose result never
    carried the field. The bench lanes did not: their dicts carried no
    end-to-end latency at all, and reported the other two under the GEAK
    spellings (``ttft_ms`` / ``itl_ms``) that the collectors read. Under a budget
    that is not a missing nicety, it is a lane whose every KEEP is refused as
    untimed whatever it measured.
    """

    def test_the_canonical_names_are_the_ones_the_bench_object_uses(self):
        """Reading a field an object does not have returns ``None`` silently, so
        the ``VariantResult`` side of the contract is pinned here."""
        vr = _variant_result()
        for canonical in ("ttft_mean_ms", "e2el_mean_ms", "tpot_mean_ms"):
            assert hasattr(vr, canonical), f"VariantResult no longer carries {canonical}"
        assert not hasattr(vr, "ttft_ms")
        assert not hasattr(vr, "itl_ms")

    def test_a_lane_that_nests_its_bench_payload_is_still_timed(self):
        """integrate_patch returns the measurement one layer down under
        ``bench_result``; the top level carries only throughput."""
        vr = _variant_result()
        result = {
            "status": "kept",
            "output_throughput": vr.output_throughput,
            "bench_result": {
                "name": vr.name,
                "status": vr.status,
                "output_throughput": vr.output_throughput,
                "ttft_mean_ms": vr.ttft_mean_ms,
                "e2el_mean_ms": vr.e2el_mean_ms,
                "tpot_mean_ms": vr.tpot_mean_ms,
            },
        }
        assert latency_from_result(result) == 150.0
        assert latency_fields_from_result(result) == {
            "ttft_mean_ms": 40.0,
            "e2el_mean_ms": 150.0,
            "tpot_mean_ms": 12.0,
        }

    def test_a_lane_that_reports_at_the_top_level_is_unaffected(self):
        result = {"output_throughput": 1200.0, "e2el_mean_ms": 150.0}
        assert latency_fields_from_result(result)["e2el_mean_ms"] == 150.0

    def test_the_top_level_wins_over_a_nested_copy(self):
        """A lane that surfaced its own figure is the authority on it."""
        result = {"e2el_mean_ms": 150.0, "bench_result": {"e2el_mean_ms": 999.0}}
        assert latency_from_result(result) == 150.0

    def test_the_geak_spellings_still_resolve(self):
        """Load-bearing, not defensive: the bench lanes deliberately emit these
        spellings because the breakdown collectors read them."""
        result = {"ttft_ms": 40.0, "e2el_ms": 150.0, "itl_ms": 12.0}
        assert latency_fields_from_result(result) == {
            "ttft_mean_ms": 40.0,
            "e2el_mean_ms": 150.0,
            "tpot_mean_ms": 12.0,
        }

    def test_a_lane_reporting_nothing_stays_none_rather_than_inventing_a_figure(self):
        result = {"status": "kept", "output_throughput": 1200.0, "bench_result": {"status": "succeeded"}}
        assert latency_from_result(result) is None
        assert latency_fields_from_result(result) == {
            "ttft_mean_ms": None,
            "e2el_mean_ms": None,
            "tpot_mean_ms": None,
        }

    def test_a_self_referential_payload_terminates(self):
        """Depth is capped, so a cycle cannot spin the lookup."""
        result: dict = {"status": "kept"}
        result["bench_result"] = result
        assert latency_from_result(result) is None

    @pytest.mark.parametrize(
        "task_kind",
        ["integrate_patch", "framework", "specialist_rebench"],
    )
    def test_a_nested_lane_promotes_under_a_budget_it_satisfies(self, session_dir, task_kind):
        """The reported failure: these lanes were refused whatever their latency."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0
        vr = _variant_result()
        result = {"output_throughput": vr.output_throughput, "bench_result": {"e2el_mean_ms": vr.e2el_mean_ms}}

        lifted = coord._lift_to_current_best(
            task_kind,
            1200.0,
            {
                "name": f"{task_kind}-winner",
                "extra_server_args": "--flag",
                **latency_fields_from_result(result),
            },
        )

        assert lifted is True, f"{task_kind} was refused despite reporting 150 ms under a 200 ms budget"
        assert s.current_best["e2el_mean_ms"] == 150.0
        assert not s.latency_refusals

    def test_a_nested_lane_over_budget_is_still_refused(self, session_dir):
        """Plumbing the field through must not soften the gate."""
        coord = _coord(session_dir)
        s = coord.shared_state
        s.baseline_tput = 1000.0
        s.latency_budget_ms = 200.0
        result = {"output_throughput": 1200.0, "bench_result": {"e2el_mean_ms": 1211.0}}

        lifted = coord._lift_to_current_best(
            "framework",
            1200.0,
            {"name": "slow", "extra_server_args": "--flag", **latency_fields_from_result(result)},
        )

        assert lifted is False
        assert s.latency_refusals[0]["e2el_mean_ms"] == 1211.0


class TestDescribe:
    """The log/report rendering."""

    def test_a_budget_is_named_with_its_unit(self):
        assert describe_latency_budget(200.0) == "latency budget 200 ms"

    @pytest.mark.parametrize("off", [0.0, -1.0, float("nan")])
    def test_no_budget_says_so(self, off):
        assert describe_latency_budget(off) == "no latency budget"


class TestReportSection:
    """What the report says about the kept configuration's own latency."""

    def test_a_kept_config_under_budget_is_called_within_budget(self):
        md = "\n".join(
            _format_latency_budget_section(
                {"latency_budget_ms": 200.0, "current_best": {"e2el_mean_ms": 183.0}},
            )
        )
        assert "183.0" in md
        assert "within budget" in md
        assert "over budget" not in md

    def test_a_kept_config_over_budget_is_not_called_within_budget(self):
        """The baseline is exempt from the gate, so this slot can hold a figure
        over budget -- and the report is where an operator would catch it."""
        md = "\n".join(
            _format_latency_budget_section(
                {"latency_budget_ms": 200.0, "current_best": {"e2el_mean_ms": 1211.0}},
            )
        )
        assert "within budget" not in md
        assert "over budget by 1011.0 ms" in md
        assert "no candidate has yet come in under" in md

    def test_an_untimed_kept_config_says_so(self):
        md = "\n".join(
            _format_latency_budget_section(
                {"latency_budget_ms": 200.0, "current_best": {"output_throughput": 1200.0}},
            )
        )
        assert "latency not measured" in md
        assert "within budget" not in md

    def test_no_section_without_a_budget(self):
        assert _format_latency_budget_section({"latency_budget_ms": 0.0}) == []

    def test_refusals_are_listed_with_what_they_measured(self):
        md = "\n".join(
            _format_latency_budget_section(
                {
                    "latency_budget_ms": 200.0,
                    "current_best": {"e2el_mean_ms": 183.0},
                    "latency_refusals": [
                        {"variant_name": "cpx-2-streams", "action": "explore", "e2el_mean_ms": 1211.0},
                    ],
                },
            )
        )
        assert "cpx-2-streams" in md
        assert "1211.0 ms" in md


class TestPromptSection:
    """``orchestration.md`` routes on the refusals, so a prompt must carry them."""

    def test_no_block_when_the_session_is_unconstrained(self, session_dir):
        s = _coord(session_dir).shared_state
        assert s.to_latency_budget_summary() == ""

    def test_the_budget_and_its_fail_closed_rule_reach_the_prompt(self, session_dir):
        s = _coord(session_dir).shared_state
        s.latency_budget_ms = 200.0

        block = s.to_latency_budget_summary()

        assert "200 ms" in block
        assert "refused" in block

    def test_refusals_reach_the_prompt_with_their_cost(self, session_dir):
        s = _coord(session_dir).shared_state
        s.latency_budget_ms = 200.0
        s.latency_refusals = [
            {"variant_name": "cpx-2-streams", "action": "explore", "e2el_mean_ms": 1211.0},
        ]

        block = s.to_latency_budget_summary()

        assert "cpx-2-streams" in block
        assert "1211 ms" in block
        assert "binding limit" in block

    def test_only_the_recent_refusals_are_rendered(self, session_dir):
        """The signal is the trend; the roster would crowd the prompt."""
        s = _coord(session_dir).shared_state
        s.latency_budget_ms = 200.0
        s.latency_refusals = [
            {"variant_name": f"v{i}", "action": "explore", "e2el_mean_ms": 300.0 + i} for i in range(12)
        ]

        block = s.to_latency_budget_summary()

        assert "refused   : 12" in block
        assert "v11" in block
        assert "v0 " not in block

    def test_an_untimed_refusal_renders_without_a_number(self, session_dir):
        s = _coord(session_dir).shared_state
        s.latency_budget_ms = 200.0
        s.latency_refusals = [{"variant_name": "untimed", "action": "framework"}]

        assert "not measured" in s.to_latency_budget_summary()

    def test_the_block_is_wired_into_the_prompt_assembly(self):
        """A renderer nothing calls is the bug this closes, so pin the call."""
        source = Path(conversation.__file__).read_text(encoding="utf-8")
        assert "to_latency_budget_summary()" in source
        assert "=== Latency budget (constraint) ===" in source


class TestResumePrecedence:
    """CLI > env > archived state, on the path a resume actually takes."""

    def test_the_launch_export_no_longer_clears_the_env_before_resume_reads_it(self, monkeypatch):
        """The reported failure: the unconditional export popped the variable, so
        the env tier could never be read and the documented chain was dead."""
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "250")
        args = argparse.Namespace(resume_from="s-1", max_latency_ms=None)
        state = SharedState(latency_budget_ms=400.0)

        if not args.resume_from:
            cli._export_latency_budget(args.max_latency_ms)
        cli._restore_latency_budget_from_state(args, state)

        assert args.max_latency_ms == 250.0

    def test_a_fresh_launch_still_clears_a_stale_variable(self, monkeypatch):
        """Without this a second session in the same shell inherits a budget the
        operator did not ask for."""
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "250")
        args = argparse.Namespace(resume_from=None, max_latency_ms=None)

        if not args.resume_from:
            cli._export_latency_budget(args.max_latency_ms)

        assert LATENCY_BUDGET_ENV not in os.environ

    def test_the_flag_outranks_both_lower_tiers(self, monkeypatch):
        monkeypatch.setenv(LATENCY_BUDGET_ENV, "250")
        args = argparse.Namespace(resume_from="s-1", max_latency_ms=100.0)

        cli._restore_latency_budget_from_state(args, SharedState(latency_budget_ms=400.0))

        assert args.max_latency_ms == 100.0

    def test_the_archive_is_used_when_the_env_is_unset(self, monkeypatch):
        monkeypatch.delenv(LATENCY_BUDGET_ENV, raising=False)
        args = argparse.Namespace(resume_from="s-1", max_latency_ms=None)

        cli._restore_latency_budget_from_state(args, SharedState(latency_budget_ms=400.0))

        assert args.max_latency_ms == 400.0

    @pytest.mark.parametrize("raw", ["abc", "", "0", "-5", "nan"])
    def test_an_unusable_env_tier_falls_through_instead_of_ending_the_resume(self, monkeypatch, raw):
        """A stale shell variable is a poor reason to refuse to resume a session
        whose own budget is archived in its state."""
        monkeypatch.setenv(LATENCY_BUDGET_ENV, raw)
        args = argparse.Namespace(resume_from="s-1", max_latency_ms=None)

        cli._restore_latency_budget_from_state(args, SharedState(latency_budget_ms=400.0))

        assert args.max_latency_ms == 400.0

    def test_an_unconstrained_resume_stays_unconstrained(self, monkeypatch):
        monkeypatch.delenv(LATENCY_BUDGET_ENV, raising=False)
        args = argparse.Namespace(resume_from="s-1", max_latency_ms=None)

        cli._restore_latency_budget_from_state(args, SharedState())

        assert args.max_latency_ms == 0.0


class TestOperatorSurfaces:
    """The flag has to be reachable from the surfaces that forward flags."""

    def test_the_skill_flag_table_lists_the_budget(self):
        """The table is documented as the source of truth for forwarding a
        prompt-stated value, so an absent row silently drops an operator's SLA."""
        skill = Path(inference_optimizer.__file__).parent / "SKILL.md"
        table_rows = [ln for ln in skill.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]

        assert any("--max-latency-ms" in row for row in table_rows)
