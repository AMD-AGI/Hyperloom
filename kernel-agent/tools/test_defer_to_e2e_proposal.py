"""PART C — DEFER_TO_E2E disposition in ``make_proposal`` (kernel_optimization).

A HIGH-IMPACT (roofline/time-share), correctness-PASSED, artifact-valid kernel
whose MICRO score is inconclusive/low (~1.0x or below the 1.10x KEEP threshold,
but NOT a hard correctness / compile / E2E-regression / accuracy failure) must
be routed to integrate as ``DEFER_TO_E2E`` so the E2E ``gain_pct`` is the
authoritative KEEP/REVERT signal -- a self-measurement artifact (GEAK's
fused_moe self-score read 1.0x on a real +20.9% E2E win) must not silently
discard a real win.

These tests pin: the defer triggers only for high-impact + correctness-passed
inconclusive/low-micro cases (with the right ``fallback_decision``), and that
every other path (KEEP, hard correctness/compile/E2E/accuracy fail, and
low-impact kernels) keeps its current disposition bit-for-bit.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_TOOL_PATH = Path(__file__).resolve().parent / "kernel_optimization.py"


@pytest.fixture(scope="module")
def ko() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_kernel_optimization_defer_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verif(
    *,
    compile_passed=True,
    correctness_passed=True,
    artifact_valid=True,
    micro_speedup=1.0,
    micro_speedup_source="report_scan",
    e2e_gain_pct=None,
    accuracy_passed=None,
) -> dict:
    return {
        "compile_passed": compile_passed,
        "correctness_passed": correctness_passed,
        "artifact_valid": artifact_valid,
        "micro_speedup": micro_speedup,
        "micro_speedup_source": micro_speedup_source,
        "e2e_gain_pct": e2e_gain_pct,
        "accuracy_passed": accuracy_passed,
    }


# ---------------------------------------------------------------------------
# DEFER_TO_E2E fires for high-impact + correctness-passed inconclusive/low micro
# ---------------------------------------------------------------------------
def test_defer_on_measured_one_x_when_high_impact(ko) -> None:
    """The headline case: a real win whose micro self-score reads ~1.0x. With
    high_impact + correctness, route to E2E instead of REVERT."""
    v = _verif(micro_speedup=1.0, micro_speedup_source="report_scan")
    prop = ko.make_proposal(v, high_impact=True)
    assert prop["decision"] == "DEFER_TO_E2E"
    assert prop["fallback_decision"] == "REVERT"
    assert prop["defer_to_e2e"] is True


def test_defer_on_unmeasured_when_high_impact(ko) -> None:
    v = _verif(micro_speedup=1.0, micro_speedup_source="default_unmeasured")
    prop = ko.make_proposal(v, high_impact=True)
    assert prop["decision"] == "DEFER_TO_E2E"
    assert prop["fallback_decision"] == "PARTIAL"


def test_defer_on_below_keep_threshold_when_high_impact(ko) -> None:
    v = _verif(micro_speedup=1.05, micro_speedup_source="report_scan")
    prop = ko.make_proposal(v, high_impact=True)
    assert prop["decision"] == "DEFER_TO_E2E"
    assert prop["fallback_decision"] == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# NO defer: low-impact, or hard correctness/compile/E2E/accuracy failures
# ---------------------------------------------------------------------------
def test_no_defer_when_low_impact_keeps_revert(ko) -> None:
    v = _verif(micro_speedup=1.0, micro_speedup_source="report_scan")
    assert ko.make_proposal(v, high_impact=False)["decision"] == "REVERT"


def test_no_defer_when_low_impact_unmeasured_keeps_partial(ko) -> None:
    v = _verif(micro_speedup=1.0, micro_speedup_source="default_unmeasured")
    assert ko.make_proposal(v, high_impact=False)["decision"] == "PARTIAL"


def test_no_defer_on_correctness_fail_even_high_impact(ko) -> None:
    """Hard correctness failure must NEVER be deferred (preserve the
    hard-correctness-fail path)."""
    v = _verif(correctness_passed=False, micro_speedup=1.0)
    assert ko.make_proposal(v, high_impact=True)["decision"] == "REVERT"


def test_no_defer_on_compile_fail_even_high_impact(ko) -> None:
    v = _verif(compile_passed=False)
    assert ko.make_proposal(v, high_impact=True)["decision"] == "REVERT"


def test_no_defer_on_e2e_regression_even_high_impact(ko) -> None:
    """A measured E2E regression is a hard REVERT, not a defer."""
    v = _verif(micro_speedup=1.2, e2e_gain_pct=-3.0)
    assert ko.make_proposal(v, high_impact=True)["decision"] == "REVERT"


def test_no_defer_on_accuracy_fail_even_high_impact(ko) -> None:
    v = _verif(micro_speedup=1.2, accuracy_passed=False)
    assert ko.make_proposal(v, high_impact=True)["decision"] == "REVERT"


# ---------------------------------------------------------------------------
# KEEP path unaffected
# ---------------------------------------------------------------------------
def test_keep_unaffected_by_high_impact(ko) -> None:
    v = _verif(micro_speedup=1.3, e2e_gain_pct=2.0, accuracy_passed=True)
    assert ko.make_proposal(v, high_impact=True)["decision"] == "KEEP"
    assert ko.make_proposal(v, high_impact=False)["decision"] == "KEEP"


def test_default_high_impact_false_preserves_legacy(ko) -> None:
    """make_proposal(verification) with no high_impact kwarg is bit-for-bit
    the legacy disposition."""
    assert ko.make_proposal(_verif(micro_speedup=1.0))["decision"] == "REVERT"
    assert ko.make_proposal(
        _verif(micro_speedup=1.0, micro_speedup_source="default_unmeasured"),
    )["decision"] == "PARTIAL"


# ---------------------------------------------------------------------------
# _candidate_is_high_impact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate,expected",
    [
        ({"gpu_pct": 6.8}, True),
        ({"gpu_pct": 1.2}, False),
        ({"percent_of_total": 4.0}, True),
        ({"impact_high_e2e_pct": 10.4}, True),
        ({"impact_high_e2e_pct": 0.5}, False),
        ({}, False),
        ({"name": "x"}, False),
        ({"task_group": {"rows": [{"percent_of_total": 5.0}]}}, True),
        ({"task_group": {"rows": [{"gpu_pct": 0.4}]}}, False),
    ],
)
def test_candidate_is_high_impact(ko, candidate, expected) -> None:
    assert ko._candidate_is_high_impact(candidate) is expected


def test_candidate_high_impact_env_override(ko, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "10.0")
    assert ko._candidate_is_high_impact({"gpu_pct": 6.8}) is False
    assert ko._candidate_is_high_impact({"gpu_pct": 12.0}) is True
