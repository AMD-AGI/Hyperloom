# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A benchmark-only launch cannot become the serving champion without remeasurement."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.grid_server_args import compose_server_args
from hyperloom.orchestrator.loop import _current_best_lift as lift


PUBLISHABLE = '--speculative-config {"method":"mtp","num_speculative_tokens":1} --max-num-batched-tokens 8192'
DIAGNOSTIC = PUBLISHABLE + " --no-enable-prefix-caching"


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.setattr(lift, "resolve_graded_comparison", lambda *_a, **_kw: SimpleNamespace(comparable=True))

    class Writer(lift.CurrentBestLiftMixin):
        def __init__(self):
            self.shared_state = SimpleNamespace(baseline_tput=100.0, current_best={"extra_server_args": ""})
            self.writes = []

        def _refuses_graded_winner(self, *_args):
            return False

        def _append_stack_entry(self, *_args):
            self.writes.append("stack")

        def _write_current_best(self, _request, launch, *_args):
            self.writes.append("current_best")
            self.shared_state.current_best = {"extra_server_args": launch.full_args}

    return Writer()


@pytest.mark.parametrize(
    "fields",
    [
        {"extra_server_args": DIAGNOSTIC},
        {"candidate_extra_server_args": DIAGNOSTIC},
        {"effective_extra_server_args": DIAGNOSTIC},
        {"extra_server_args": PUBLISHABLE, "launch_evidence": {"requested_server_args": DIAGNOSTIC}},
        {"extra_server_args": PUBLISHABLE, "launch_evidence": {"requested_server_args": DIAGNOSTIC.split()}},
        {"extra_server_args": PUBLISHABLE, "resolved_server_launch_flags": DIAGNOSTIC},
        {"extra_server_args": PUBLISHABLE, "launch_evidence": {"observed_server_launch_flags": DIAGNOSTIC}},
        {"extra_server_args": DIAGNOSTIC, "launch_evidence": {"observed_server_launch_flags": ""}},
        {"extra_server_args": DIAGNOSTIC, "resolved_server_launch_flags": ""},
        {
            "extra_server_args": PUBLISHABLE,
            "resolved_server_launch_flags": PUBLISHABLE,
            "launch_evidence": {"requested_server_args": DIAGNOSTIC},
        },
    ],
)
def test_benchmark_only_winner_is_not_promoted_or_rewritten(writer, fields, caplog):
    winner = {"name": "prefix-off", **fields}
    original = deepcopy(winner)

    assert writer.lift_to_current_best("explore", 120.0, winner) is False
    assert writer.writes == []
    assert winner == original
    assert "needs remeasurement under the publishable serving arguments" in caplog.text


def test_explicit_clean_remeasurement_can_be_promoted(writer):
    winner = {
        "name": "remeasured",
        "extra_server_args": DIAGNOSTIC,
        "launch_evidence": {"requested_server_args": PUBLISHABLE},
    }

    assert writer.lift_to_current_best("explore", 120.0, winner) is True
    assert writer.shared_state.current_best["extra_server_args"] == PUBLISHABLE


def test_engine_default_prefix_state_does_not_make_a_clean_request_diagnostic(writer):
    winner = {
        "name": "clean",
        "extra_server_args": PUBLISHABLE,
        "launch_evidence": {
            "requested_server_args": PUBLISHABLE,
            "observed_server_config": {"enable_prefix_caching": False},
        },
    }

    assert writer.lift_to_current_best("explore", 120.0, winner) is True


def test_prebaseline_enablement_retains_its_non_attributable_contract(writer):
    writer.shared_state.baseline_tput = 0.0
    winner = {
        "name": "enablement",
        "extra_server_args": DIAGNOSTIC,
        "baseline_enablement": True,
        "attribution_eligible": False,
    }

    assert writer.lift_to_current_best("integrate_patch", 0.0, winner) is True


def test_diagnostic_launch_remains_legal_to_benchmark():
    assert "--no-enable-prefix-caching" in compose_server_args(variant_extra_args=DIAGNOSTIC)


def test_real_writeback_waits_for_clean_launch_qualification(tmp_path):
    from .test_recipe_sedimentation import _make_coordinator

    coord = _make_coordinator(tmp_path)
    state = coord.shared_state
    state.baseline_tput = 100.0
    state.current_best = {"tput": 100.0, "extra_server_args": "", "extra_envs": {}}
    assert coord.writeback.lift_to_current_best("explore", 110.0, {"name": "initial", "extra_server_args": PUBLISHABLE})
    before = deepcopy(state.current_best)
    generation = state.working_recipe_generation
    stack_len = len(state.optimization_stack)
    winner = {
        "name": "diagnostic",
        "extra_server_args": DIAGNOSTIC,
        "launch_evidence": {"requested_server_args": DIAGNOSTIC},
    }
    original = deepcopy(winner)

    assert coord.writeback.lift_to_current_best("explore", 120.0, winner) is False
    assert state.current_best == before
    assert state.working_recipe_generation == generation
    assert len(state.optimization_stack) == stack_len
    assert winner == original

    winner["name"] = "remeasured"
    winner["launch_evidence"] = {"requested_server_args": PUBLISHABLE}
    assert coord.writeback.lift_to_current_best("explore", 120.0, winner) is True
    assert state.current_best["extra_server_args"] == PUBLISHABLE
