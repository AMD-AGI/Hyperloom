# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for the workload-comparability guard on the GEAK candidate.

``KernelPhase._record_geak_candidate`` computes ``self_reported_gain_pct`` from
two numbers measured by DIFFERENT harnesses: ``final_throughput_tok_s`` is
GEAK's, ``shared_state.baseline_tput`` is ours. That percentage is meaningful
only when both harnesses measured the same workload.

In AgentX mode they do not. Our baseline replays real agentic traces (p50 ~89k
input tokens) while the handoff still carries the CLI's synthetic isl/osl
defaults of 1024/1024, so a GEAK run that believes those defaults measures a
~1k-token synthetic sweep. Measured on Kimi-K3: GEAK's 465.676 tok/s against our
168.998 tok/s agentic baseline computes to roughly +175% with no kernel changed.

The rebench downstream would eventually reject such a candidate, but the number
is written into ``geak_pending`` the moment the candidate is recorded, and
anything reading that record before the rebench lands would read a real-looking
gain. So the gain is refused at the point of computation instead.

GEAK states the verdict in ``baseline_basis.workload_comparability.comparable``.
The guard is deliberately asymmetric: only an explicit ``False`` suppresses.
An absent verdict is an older GEAK that only ever ran the synthetic path, and
must keep behaving exactly as before -- that is what makes this safe to ship
without touching the established fixed-ISL/OSL flow.
"""

from __future__ import annotations

import pytest

# The measured Kimi-K3 numbers, kept as the fixture so a regression reproduces
# the real false claim rather than an invented one.
HL_AGENTIC_BASELINE_TOK_S = 168.998
GEAK_SYNTHETIC_TOK_S = 465.676
FALSE_GAIN_PCT = ((GEAK_SYNTHETIC_TOK_S - HL_AGENTIC_BASELINE_TOK_S) / HL_AGENTIC_BASELINE_TOK_S) * 100.0  # ~175.55


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )

    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    return Coordinator(sd, backends=backends)


def _record(coordinator, *, baseline: float, geak_tput: float, comparability=None):
    """Record a GEAK candidate and return the resulting geak_pending."""
    st = coordinator.shared_state
    st.baseline_tput = baseline
    st.geak_result = {"status": "ok"}
    result: dict = {
        "status": "ok",
        "final_throughput_tok_s": geak_tput,
        "throughput_speedup": 1.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    if comparability is not None:
        result["baseline_basis"] = {"workload_comparability": comparability}
    coordinator.phase_kernel._record_geak_candidate(result)
    return st.geak_pending


# ───────────────────────────── the invariant ─────────────────────────────────


def test_absent_verdict_keeps_todays_gain(coordinator) -> None:
    """No comparability block at all must behave exactly as before.

    Every GEAK result written before the verdict existed lands here, so treating
    silence as a mismatch would retroactively blank a valid gain.
    """
    pending = _record(coordinator, baseline=100.0, geak_tput=116.0)

    assert pending["self_reported_gain_pct"] == pytest.approx(16.0)
    assert pending["workload_comparability"] is None


def test_comparable_true_keeps_the_gain(coordinator) -> None:
    """An explicit 'these matched' verdict computes the gain as usual."""
    pending = _record(
        coordinator,
        baseline=100.0,
        geak_tput=116.0,
        comparability={
            "comparable": True,
            "orchestrator_workload_kind": "synthetic_isl_osl",
            "geak_workload_kind": "synthetic_isl_osl",
        },
    )

    assert pending["self_reported_gain_pct"] == pytest.approx(16.0)
    assert pending["workload_comparability"]["comparable"] is True


# ─────────────────────────────── the guard ───────────────────────────────────


def test_mismatch_suppresses_the_measured_false_gain(coordinator) -> None:
    """The real Kimi-K3 mismatch must not record its ~+175%."""
    pending = _record(
        coordinator,
        baseline=HL_AGENTIC_BASELINE_TOK_S,
        geak_tput=GEAK_SYNTHETIC_TOK_S,
        comparability={
            "comparable": False,
            "orchestrator_workload_kind": "agentx_trace_replay",
            "geak_workload_kind": "synthetic_isl_osl",
            "suppressed_reason": "workload kinds differ",
        },
    )

    assert FALSE_GAIN_PCT == pytest.approx(175.55, abs=0.01)
    assert pending["self_reported_gain_pct"] is None
    # The reason is carried so an absent gain is never read as "no gain".
    wc = pending["workload_comparability"]
    assert wc["comparable"] is False
    assert wc["orchestrator_workload_kind"] == "agentx_trace_replay"
    assert wc["geak_workload_kind"] == "synthetic_isl_osl"


def test_suppression_keeps_the_evidence(coordinator) -> None:
    """Refuse the ratio, keep both measurements that would have formed it."""
    pending = _record(
        coordinator,
        baseline=HL_AGENTIC_BASELINE_TOK_S,
        geak_tput=GEAK_SYNTHETIC_TOK_S,
        comparability={"comparable": False},
    )

    assert pending["self_reported_gain_pct"] is None
    # GEAK's own measurement and its within-harness speedup are still valid
    # numbers -- they simply are not OUR baseline's counterpart.
    assert pending["self_reported_tput"] == pytest.approx(GEAK_SYNTHETIC_TOK_S)
    assert pending["self_reported_speedup"] == pytest.approx(1.0)
    assert pending["status"] == "awaiting_rebench"


def test_malformed_verdict_does_not_suppress(coordinator) -> None:
    """Only an explicit False suppresses; junk must not blank a valid gain."""
    for junk in ({}, {"comparable": None}, {"comparable": "no"}, {"other": 1}):
        pending = _record(coordinator, baseline=100.0, geak_tput=116.0, comparability=junk)
        assert pending["self_reported_gain_pct"] == pytest.approx(16.0), f"verdict {junk!r} must not suppress the gain"
