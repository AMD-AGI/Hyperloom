# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Failure classification + auto-retry prompt + freeform-patch coverage.

Covers the pure decision core behind the unified specialist's bounded
transient-failure auto-retry:

* ``classify_specialist_failure`` — the runner-status/error -> failure-type +
  retry-eligibility taxonomy (only infra flakes are retry-eligible);
* the prompt builder's auto-retry notice block (injected on a re-dispatch);
* the freeform + ``mode=patch`` prompt path (a freeform specialist may author
  patches just like a domain one).
* ``_maybe_auto_retry_specialist`` lane assignment — GPU specialists that set
  ``needs_gpu=true`` (including bench-enabled specialists) must acquire
  ``gpu_research_lane`` on retry, mirroring the first-dispatch logic in
  ``intent_router._handle_delegate``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hyperloom.orchestrator.specialists.domains import FREEFORM_DOMAIN
from hyperloom.orchestrator.specialists.runner import (
    RETRYABLE_SPECIALIST_FAILURES,
    SpecialistFailureType,
    classify_specialist_failure,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


# --------------------------------------------------------------------------- #
# classify_specialist_failure — the taxonomy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,error,expected_type,expected_retry",
    [
        ("succeeded", "", SpecialistFailureType.NONE, False),
        ("tool_violation", "emitted explore", SpecialistFailureType.TOOL_VIOLATION, False),
        # Transient infra failures (status == 'stale' with a backend_error).
        ("stale", "subprocess_timeout", SpecialistFailureType.TIMEOUT, True),
        ("stale", "subprocess_stale_heartbeat", SpecialistFailureType.STALE_HEARTBEAT, True),
        ("stale", "subprocess_exit_code:139", SpecialistFailureType.CRASH, True),
        ("stale", "subprocess_error: boom", SpecialistFailureType.CRASH, True),
        ("stale", "", SpecialistFailureType.CRASH, True),
        # Clean exit without a usable done — semantic, NOT retried.
        ("empty_synthesised", "max_turns_exhausted", SpecialistFailureType.NO_OUTPUT, False),
        ("empty_synthesised", "no_specialist_done_emitted", SpecialistFailureType.NO_OUTPUT, False),
        ("empty_synthesised", "", SpecialistFailureType.NO_OUTPUT, False),
        # Config errors — re-running verbatim can't help.
        ("empty_synthesised", "unknown_domain", SpecialistFailureType.CONFIG, False),
        ("empty_synthesised", "no_workspace", SpecialistFailureType.CONFIG, False),
        # Anything unrecognised is non-retryable by default.
        ("weird_status", "", SpecialistFailureType.UNKNOWN, False),
    ],
)
def test_classify_specialist_failure(status, error, expected_type, expected_retry):
    ftype, retry_eligible = classify_specialist_failure(status, error)
    assert ftype == expected_type
    assert retry_eligible is expected_retry


def test_only_infra_failures_are_in_the_retryable_set():
    assert RETRYABLE_SPECIALIST_FAILURES == frozenset(
        {
            SpecialistFailureType.TIMEOUT,
            SpecialistFailureType.STALE_HEARTBEAT,
            SpecialistFailureType.CRASH,
        }
    )


def test_classify_is_case_and_whitespace_insensitive():
    ftype, retry_eligible = classify_specialist_failure("  STALE ", "  Subprocess_Timeout  ")
    assert ftype == SpecialistFailureType.TIMEOUT
    assert retry_eligible is True


def test_retry_eligibility_matches_membership():
    """Every retry-eligible classification must be a member of the set, and
    every non-eligible one must not be — the two sources of truth agree."""
    cases = [
        ("succeeded", ""),
        ("tool_violation", "x"),
        ("stale", "subprocess_timeout"),
        ("stale", "subprocess_stale_heartbeat"),
        ("stale", "subprocess_exit_code:1"),
        ("empty_synthesised", "unknown_domain"),
        ("empty_synthesised", ""),
        ("weird", ""),
    ]
    for status, error in cases:
        ftype, retry_eligible = classify_specialist_failure(status, error)
        assert retry_eligible == (ftype in RETRYABLE_SPECIALIST_FAILURES)


# --------------------------------------------------------------------------- #
# Auto-retry notice block (prompt builder)
# --------------------------------------------------------------------------- #
def _freeform_inputs(**kwargs) -> SpecialistPromptInputs:
    base = dict(
        task_id="spec-1",
        domain=FREEFORM_DOMAIN,
        scope="freeform",
        task_description="Read the scheduler and find why prefill blocks decode.",
    )
    base.update(kwargs)
    return SpecialistPromptInputs(**base)


def test_auto_retry_notice_absent_on_first_attempt():
    system, _user = build_specialist_prompts(_freeform_inputs())
    assert "Auto-retry notice" not in system


def test_auto_retry_notice_rendered_with_reason():
    system, _user = build_specialist_prompts(
        _freeform_inputs(auto_retry_reason="timeout: subprocess_timeout"),
    )
    assert "### Auto-retry notice" in system
    assert "subprocess_timeout" in system
    # Framed as transient infra, not a rejection of the approach.
    assert "transient" in system.lower()


def test_auto_retry_notice_blank_reason_is_noop():
    system, _user = build_specialist_prompts(_freeform_inputs(auto_retry_reason="   "))
    assert "Auto-retry notice" not in system


# --------------------------------------------------------------------------- #
# Freeform + mode=patch prompt path
# --------------------------------------------------------------------------- #
def test_freeform_patch_prompt_carries_mandate_and_patch_protocol():
    """A freeform specialist dispatched with ``mode=patch`` still gets the
    free-form mandate AND the worktree patch-authoring protocol."""
    system, _user = build_specialist_prompts(_freeform_inputs(mode="patch"))
    assert "Free-form mandate (scope = freeform)" in system
    assert "prefill blocks decode" in system
    # Patch-authoring protocol (iron rules + output protocol) is present.
    assert "patches_written" in system
    assert "worktree" in system.lower()


# --------------------------------------------------------------------------- #
# _maybe_auto_retry_specialist — lane assignment mirrors first dispatch (#1 fix)
# --------------------------------------------------------------------------- #

def _make_explore_phase_stub(registry_lanes, registry_ttl, gpu_ttl, captured_tasks):
    """Return a minimal ExplorePhase-like stub with a fake TaskRegistry.

    ``captured_tasks`` is a list populated with each ``create_or_return_existing``
    call's ``(kind, params, requires_lanes, lease_ttl_sec)`` tuple.
    """
    from hyperloom.orchestrator.phases.explore import ExplorePhase

    # Fake Task returned by create_or_return_existing.
    fake_task = MagicMock()
    fake_task.task_id = "retry-task-1"

    async def _fake_create(kind, params, idempotency_key, requires_lanes, lease_ttl_sec):
        captured_tasks.append({
            "kind": kind,
            "requires_lanes": list(requires_lanes or []),
            "lease_ttl_sec": lease_ttl_sec,
        })
        return fake_task, False  # (task, was_existing=False)

    fake_tasks = MagicMock()
    fake_tasks.create_or_return_existing = _fake_create

    # Minimal coordinator stub.
    coord_stub = MagicMock()
    coord_stub.tasks = fake_tasks
    coord_stub._registry_lanes_ttl = MagicMock(return_value=(list(registry_lanes), registry_ttl))
    coord_stub._gpu_lease_ttl_sec = MagicMock(return_value=gpu_ttl)
    coord_stub._record_observation = AsyncMock()

    # Build ExplorePhase with __init__ bypassed (PhaseHandler only stores _coord).
    phase = ExplorePhase.__new__(ExplorePhase)
    phase._coord = coord_stub
    return phase


def _make_stale_task(params):
    """Minimal Task-like object for _maybe_auto_retry_specialist."""
    t = MagicMock()
    t.task_id = "orig-task-1"
    t.idempotency_key = "spec-key-1"
    t.params = params
    return t


def _make_stale_result(runner_status="stale", error="subprocess_timeout"):
    r = MagicMock()
    r.result = {"runner_status": runner_status}
    r.error = error
    return r


@pytest.mark.asyncio
async def test_auto_retry_needs_gpu_acquires_gpu_research_lane():
    """A specialist with needs_gpu=true must include gpu_research_lane in retry lanes.

    Mirrors the first-dispatch logic in intent_router._handle_delegate (High #1 fix).
    """
    captured = []
    phase = _make_explore_phase_stub(
        registry_lanes=["research_lane"],
        registry_ttl=600,
        gpu_ttl=7200,
        captured_tasks=captured,
    )

    task = _make_stale_task({"needs_gpu": True, "scope": "freeform", "task_description": "probe"})
    result = _make_stale_result()

    retried = await phase._maybe_auto_retry_specialist(task, result)

    assert retried is True, "infra failure + needs_gpu task must be retried"
    assert captured, "create_or_return_existing must have been called"
    lanes = captured[0]["requires_lanes"]
    assert "gpu_research_lane" in lanes, (
        f"retry must hold gpu_research_lane; got {lanes}"
    )
    assert captured[0]["lease_ttl_sec"] == 7200, (
        "retry TTL must be re-sourced via _gpu_lease_ttl_sec"
    )


@pytest.mark.asyncio
async def test_auto_retry_bench_specialist_acquires_both_lanes():
    """A bench-capable specialist (mode=patch & bench=true, needs_gpu defaulted)
    must hold both benchmark_lane and gpu_research_lane on retry."""
    captured = []
    phase = _make_explore_phase_stub(
        registry_lanes=["research_lane"],
        registry_ttl=600,
        gpu_ttl=7200,
        captured_tasks=captured,
    )

    task = _make_stale_task({
        "needs_gpu": True,
        "mode": "patch",
        "bench": True,
        "scope": "freeform",
        "task_description": "start a server and rebench",
    })
    result = _make_stale_result()

    retried = await phase._maybe_auto_retry_specialist(task, result)

    assert retried is True
    lanes = captured[0]["requires_lanes"]
    assert "benchmark_lane" in lanes, f"bench specialist retry must hold benchmark_lane; got {lanes}"
    assert "gpu_research_lane" in lanes, f"bench specialist retry must hold gpu_research_lane; got {lanes}"


@pytest.mark.asyncio
async def test_auto_retry_non_gpu_specialist_no_gpu_research_lane():
    """A non-GPU specialist (needs_gpu=false) must NOT acquire gpu_research_lane."""
    captured = []
    phase = _make_explore_phase_stub(
        registry_lanes=["research_lane"],
        registry_ttl=600,
        gpu_ttl=7200,
        captured_tasks=captured,
    )

    task = _make_stale_task({"needs_gpu": False, "scope": "freeform", "task_description": "read logs"})
    result = _make_stale_result()

    retried = await phase._maybe_auto_retry_specialist(task, result)

    assert retried is True
    lanes = captured[0]["requires_lanes"]
    assert "gpu_research_lane" not in lanes, (
        f"non-GPU specialist retry must NOT hold gpu_research_lane; got {lanes}"
    )
