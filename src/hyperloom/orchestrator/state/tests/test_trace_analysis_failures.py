# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A trace analysis that failed in a way the same trace reproduces is served, not re-run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop.intent_router import IntentRouter
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.analysis_failures import remember_trace_failure, remembered_trace_failure
from hyperloom.orchestrator.state.shared_state import SharedState

TRACE = "/runs/roofline/ecb62ceb/torch_trace"
SPLIT_FAILURE = {
    "status": "failed",
    "error": "RuntimeError: trace_split_no_steady_state: TraceLens splitter produced no steady-state chunks",
}
TIMEOUT_FAILURE = {"status": "failed", "error_class": "subprocess_timeout", "error": "TimeoutExpired after 3600s"}


def _state() -> SharedState:
    return SharedState(session_id="s")


@pytest.mark.parametrize("failure", [SPLIT_FAILURE, TIMEOUT_FAILURE], ids=["trace-split", "timeout"])
def test_a_recurring_failure_is_served_for_the_same_trace(failure):
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE}, failure)

    served = remembered_trace_failure(state, {"trace_input": TRACE})

    assert served is not None
    assert served["status"] == "failed"
    assert served["error_class"] == "trace_analysis_failure_remembered"
    assert served["trace_input"] == TRACE


def test_a_different_trace_is_analysed():
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE}, SPLIT_FAILURE)

    assert remembered_trace_failure(state, {"trace_input": "/runs/roofline/other/torch_trace"}) is None


def test_a_timeout_under_a_different_budget_is_analysed_again():
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE, "budget_minutes": 60}, TIMEOUT_FAILURE)

    assert remembered_trace_failure(state, {"trace_input": TRACE, "budget_minutes": 60}) is not None
    assert remembered_trace_failure(state, {"trace_input": TRACE, "budget_minutes": 120}) is None


def test_a_request_naming_no_budget_runs_under_the_default_budget():
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE}, TIMEOUT_FAILURE)

    assert remembered_trace_failure(state, {"trace_input": TRACE, "budget_minutes": 60}) is not None


@pytest.mark.parametrize(
    "changed",
    [{"analysis_route": "bypass"}, {"split_conc": 32}, {"steady_state_mode": "decode_only"}],
    ids=["route", "split", "steady-state-mode"],
)
def test_a_request_that_asks_for_a_different_analysis_of_the_same_trace_runs(changed):
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE, "analysis_route": "agent"}, SPLIT_FAILURE)

    assert remembered_trace_failure(state, {"trace_input": TRACE, "analysis_route": "agent", **changed}) is None


def test_a_failure_the_roofline_recorded_is_served_to_a_kernel_agent_request_for_its_trace():
    """The roofline names its framework and a kernel agent's request does not; trace_analyze runs both under the
    session's framework, so they are the same analysis."""
    from hyperloom.orchestrator.actions.executors.roofline import _analysis_payload

    state = _state()
    state.framework = "sglang"
    roofline_payload = _analysis_payload({"reason": "cycle_start"}, trace_path=TRACE, framework="sglang")
    remember_trace_failure(state, roofline_payload, TIMEOUT_FAILURE)

    assert remembered_trace_failure(state, {"trace_input": TRACE, "top_k": 10}) is not None
    assert remembered_trace_failure(state, {"trace_input": TRACE, "framework": "vllm"}) is None


@pytest.mark.parametrize(
    "result",
    [
        {"status": "failed", "error_class": "tracelens_root_missing", "error": "TraceLens is not installed"},
        {"status": "ok", "hot_kernels": []},
    ],
    ids=["transient-failure", "success"],
)
def test_only_a_recurring_failure_is_remembered(result):
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE}, result)

    assert remembered_trace_failure(state, {"trace_input": TRACE}) is None


def test_the_kernel_request_cache_serves_the_remembered_failure_without_running_the_handler():
    state = _state()
    remember_trace_failure(state, {"trace_input": TRACE}, SPLIT_FAILURE)
    kernel = KernelPhase(SimpleNamespace(shared_state=state))

    served = kernel.cached_kernel_request("trace_analyze", {"trace_input": TRACE})

    assert served is not None and served["error_class"] == "trace_analysis_failure_remembered"
    assert kernel.cached_kernel_request("trace_analyze", {"trace_input": "/runs/roofline/other/torch_trace"}) is None


def test_the_router_remembers_a_failed_analysis_and_caches_a_successful_one():
    state = _state()
    router = IntentRouter(SimpleNamespace(shared_state=state))

    router._record_trace_analyze_outcome({"trace_input": TRACE}, SPLIT_FAILURE)
    assert remembered_trace_failure(state, {"trace_input": TRACE}) is not None

    other = "/runs/roofline/other/torch_trace"
    router._record_trace_analyze_outcome({"trace_input": other}, {"status": "ok", "hot_kernels": []})
    assert remembered_trace_failure(state, {"trace_input": other}) is None
    assert state.last_trace_analyze
