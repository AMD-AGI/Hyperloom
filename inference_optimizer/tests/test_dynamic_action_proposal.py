# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the proposal validator and proposal-set builder."""

from __future__ import annotations

from typing import Any

import pytest

from inference_optimizer.orchestrator.dynamic_action_proposal import (
    ALLOWED_PROPOSAL_FIELDS,
    DynamicRunnerTerminalState,
    EXPECTED_PROVENANCE,
    FORBIDDEN_PROPOSAL_FIELDS,
    MAX_PROPOSAL_SET_LEN,
    REQUIRED_PROPOSAL_FIELDS,
    TERMINAL_REASONS,
    _normalise_diff_for_compare,
    build_proposal_set_payload,
    validate_proposal,
)


def _good_proposal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "combo",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": [
            "serving_specialist", "kernel_switch_specialist",
        ],
        "cross_domain_rationale": (
            "serving_specialist & kernel_switch_specialist coupled"
        ),
        "expected_qualitative_argument": (
            "should reduce contention without breaking accuracy"
        ),
    }
    base.update(overrides)
    return base


SCOPE = ["serving_specialist", "kernel_switch_specialist"]


# ===========================================================================
# Constants
# ===========================================================================
def test_constants_locked():
    assert ALLOWED_PROPOSAL_FIELDS == frozenset({
        "name", "provenance", "patch_text", "scope_domains",
        "cross_domain_rationale", "expected_qualitative_argument",
    })
    assert REQUIRED_PROPOSAL_FIELDS == (
        "name", "provenance", "patch_text", "scope_domains",
        "cross_domain_rationale", "expected_qualitative_argument",
    )
    assert FORBIDDEN_PROPOSAL_FIELDS >= {
        "expected_gain", "bench_evidence", "confidence", "score",
        "rank", "force_provenance",
    }
    assert EXPECTED_PROVENANCE == "dynamic"
    assert MAX_PROPOSAL_SET_LEN == 1


def test_terminal_state_reasons_closed():
    assert set(TERMINAL_REASONS) == {
        DynamicRunnerTerminalState.COMPLETED,
        DynamicRunnerTerminalState.COMPLETED_EMPTY,
        DynamicRunnerTerminalState.TIMED_OUT,
        DynamicRunnerTerminalState.FAILED,
        DynamicRunnerTerminalState.ABANDONED,
    }
    assert "wall_clock_exhausted" in TERMINAL_REASONS[
        DynamicRunnerTerminalState.TIMED_OUT
    ]


# ===========================================================================
# Happy path + reject branches
# ===========================================================================
def test_validate_proposal_happy_path():
    r = validate_proposal(_good_proposal(), spec_scope_domains=SCOPE)
    assert r.ok is True
    assert r.normalised["provenance"] == "dynamic"
    assert r.normalised["scope_domains"] == SCOPE


@pytest.mark.parametrize("field_name", sorted(FORBIDDEN_PROPOSAL_FIELDS))
def test_validate_proposal_rejects_each_forbidden_field(field_name: str):
    p = _good_proposal()
    p[field_name] = 1
    r = validate_proposal(p, spec_scope_domains=SCOPE)
    assert r.ok is False
    assert r.reason == "forbidden_field_present"


def test_validate_proposal_rejects_overlong_name():
    from inference_optimizer.orchestrator.dynamic_action_proposal import (
        MAX_PROPOSAL_NAME_CHARS,
    )
    r = validate_proposal(
        _good_proposal(name="x" * (MAX_PROPOSAL_NAME_CHARS + 1)),
        spec_scope_domains=SCOPE,
    )
    assert r.ok is False
    assert r.reason == "name_too_long"


def test_validate_proposal_rejects_unknown_field():
    p = _good_proposal()
    p["unexpected_extra"] = True
    r = validate_proposal(p, spec_scope_domains=SCOPE)
    assert r.reason == "unknown_field_present"


def test_validate_proposal_rejects_missing_required():
    p = _good_proposal()
    p.pop("cross_domain_rationale", None)
    r = validate_proposal(p, spec_scope_domains=SCOPE)
    assert r.reason == "missing_required_field"


@pytest.mark.parametrize("bad", ["", "specialist:foo", "dynamic:extra"])
def test_validate_proposal_provenance_must_be_dynamic(bad: str):
    r = validate_proposal(
        _good_proposal(provenance=bad), spec_scope_domains=SCOPE,
    )
    assert r.reason == "provenance_must_be_dynamic"


def test_validate_proposal_rejects_scope_superset():
    r = validate_proposal(
        _good_proposal(scope_domains=[*SCOPE, "compiler_specialist"]),
        spec_scope_domains=SCOPE,
    )
    assert r.reason == "scope_domains_not_subset"


def test_validate_proposal_rejects_patch_not_unified_diff():
    r = validate_proposal(
        _good_proposal(patch_text="not a diff"),
        spec_scope_domains=SCOPE,
    )
    assert r.reason == "patch_text_not_unified_diff"


def test_validate_proposal_allows_rationale_without_every_domain_mention():
    """The per-domain substring check was removed — the Critic judges
    cross-domain depth, not a mechanical keyword scan."""
    r = validate_proposal(
        _good_proposal(cross_domain_rationale="serving only"),
        spec_scope_domains=SCOPE,
    )
    assert r.ok is True


@pytest.mark.parametrize("claim", [
    "should give 20% gain",
    "expect a 1.5x speedup",
    "saves ~5 ms per step",
    "speedup of 30 over baseline",
])
def test_validate_proposal_flags_numeric_claims_as_advisory(claim: str):
    """Numeric speedup claims no longer reject the proposal — they are
    carried back as a non-blocking advisory warning for the Critic."""
    r = validate_proposal(
        _good_proposal(expected_qualitative_argument=claim),
        spec_scope_domains=SCOPE,
    )
    assert r.ok is True
    assert any("unverified_numeric_claim" in w for w in r.warnings)


# ===========================================================================
# build_proposal_set_payload
# ===========================================================================
def test_build_proposal_set_payload_completed():
    norm = {"name": "x", "provenance": "dynamic"}
    out = build_proposal_set_payload(
        dyn_id="dyn-1-1", normalised_proposal=norm, journal_path="/p",
    )
    assert out == {
        "dyn_id": "dyn-1-1",
        "proposal_set": [norm],
        "empty": False,
        "journal_path": "/p",
    }


def test_build_proposal_set_payload_empty():
    out = build_proposal_set_payload(
        dyn_id="dyn-1-1", normalised_proposal=None, journal_path="/p",
    )
    assert out == {
        "dyn_id": "dyn-1-1",
        "proposal_set": [],
        "empty": True,
        "journal_path": "/p",
    }


# ===========================================================================
# Cumulative-diff alignment check
# ===========================================================================
PATCH_A = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"


def test_cumulative_diff_skipped_when_worktree_clean():
    """Empty worktree diff (or None) skips the cumulative-diff check."""
    r = validate_proposal(
        _good_proposal(patch_text=PATCH_A),
        spec_scope_domains=SCOPE,
        worktree_cumulative_diff="",
    )
    assert r.ok is True


def test_cumulative_diff_check_disabled_when_arg_none():
    r = validate_proposal(
        _good_proposal(patch_text=PATCH_A),
        spec_scope_domains=SCOPE,
        worktree_cumulative_diff=None,
    )
    assert r.ok is True


def test_matching_cumulative_diff_passes():
    """A proposal whose ``patch_text`` matches ``git diff HEAD``
    (modulo git metadata) is accepted."""
    git_diff = (
        "diff --git a/x b/x\nindex 123..456 100644\n"
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
    )
    r = validate_proposal(
        _good_proposal(patch_text=PATCH_A),
        spec_scope_domains=SCOPE,
        worktree_cumulative_diff=git_diff,
    )
    assert r.ok is True


def test_mismatched_cumulative_diff_rejected():
    """A ``patch_text`` that disagrees with the worktree's actual diff
    is rejected so integrate_patch does not silently apply an
    incomplete patch."""
    worktree_diff = (
        "diff --git a/y b/y\nindex 999..aaa 100644\n"
        "--- a/y\n+++ b/y\n@@ -1 +1 @@\n-foo\n+bar\n"
    )
    r = validate_proposal(
        _good_proposal(patch_text=PATCH_A),
        spec_scope_domains=SCOPE,
        worktree_cumulative_diff=worktree_diff,
    )
    assert r.ok is False
    assert r.reason == "patch_text_not_cumulative_diff"


def test_diff_normaliser_strips_git_metadata():
    """``index`` / ``diff --git`` / mode lines drop out so a
    hand-crafted patch can match git's machine output."""
    with_meta = (
        "diff --git a/x b/x\nindex abc..def 100644\n"
        "new file mode 100644\n"
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    )
    without_meta = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert (
        _normalise_diff_for_compare(with_meta)
        == _normalise_diff_for_compare(without_meta)
    )
