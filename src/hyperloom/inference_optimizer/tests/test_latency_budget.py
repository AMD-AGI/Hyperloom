# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``--max-latency-ms``: resolution, the KEEP gate, and its reporting.

The gate's whole value is that it holds on paths nobody remembered to wire, so
the promotion tests here go through the real ``_lift_to_current_best`` for each
action kind rather than calling the predicate directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.actions.executors._latency_budget import (
    LATENCY_BUDGET_ENV,
    REASON_OVER_BUDGET,
    REASON_UNMEASURED,
    describe_latency_budget,
    latency_from_result,
    latency_keep_block,
    read_session_budget,
    resolve_latency_budget_ms,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)


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


class TestDescribe:
    """The log/report rendering."""

    def test_a_budget_is_named_with_its_unit(self):
        assert describe_latency_budget(200.0) == "latency budget 200 ms"

    @pytest.mark.parametrize("off", [0.0, -1.0, float("nan")])
    def test_no_budget_says_so(self, off):
        assert describe_latency_budget(off) == "no latency budget"
