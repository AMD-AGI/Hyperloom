# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Failure classification + auto-retry prompt + freeform-patch coverage.

Covers the pure decision core behind the unified specialist's bounded
transient-failure auto-retry:

* ``classify_specialist_failure`` — the runner-status/error -> failure-type +
  retry-eligibility taxonomy (only infra flakes are retry-eligible);
* the prompt builder's auto-retry notice block (injected on a re-dispatch);
* the freeform + ``mode=patch`` prompt path (a freeform specialist may author
  patches just like a domain one).
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.specialist_domains import FREEFORM_DOMAIN
from inference_optimizer.orchestrator.specialist_runner import (
    RETRYABLE_SPECIALIST_FAILURES,
    SpecialistFailureType,
    classify_specialist_failure,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
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
    assert RETRYABLE_SPECIALIST_FAILURES == frozenset({
        SpecialistFailureType.TIMEOUT,
        SpecialistFailureType.STALE_HEARTBEAT,
        SpecialistFailureType.CRASH,
    })


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
