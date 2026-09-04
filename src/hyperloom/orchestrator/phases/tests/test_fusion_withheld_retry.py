# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round the lane ceiling could not fund in full has targets left untried.

One ceiling covers both fusion pipelines, so a slate of compile-pass claims can
spend it whole and leave the authoring loop nothing. That round answers only for
what it ran, but its record reads ``complete`` -- which satisfies the
KERNEL-entry gate and skips fusion for the rest of the session.

Retrying is capped: discovery is re-run on every retry, so an unfunded target
that keeps being re-discovered must not re-spend the gateway budget forever.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.phases.kernel import MAX_FUSION_WITHHELD_RETRIES, KernelPhase

REQUIRED = KernelPhase._fusion_required_before_kernel_opt
RECORD = KernelPhase._handle_fusion_result


def _phase(last_fusion, *, spent: int = 0, session_dir=None):
    """A stand-in carrying only what the two methods under test read."""
    state = SimpleNamespace(
        framework="sglang",
        last_profile_trace="/tmp/decode.trace.json.gz",
        last_fusion=last_fusion,
        fusion_infra_aborts=0,
        fusion_withheld_retries=spent,
        macro_cycle=1,
        save=lambda *a, **k: None,
    )
    bus = SimpleNamespace(posted=[])

    async def _append_and_seq(message):
        bus.posted.append(message)

    bus.append_and_seq = _append_and_seq

    async def _integrate_fusion(result):
        return None

    return SimpleNamespace(
        shared_state=state,
        bus=bus,
        session_dir=session_dir,
        _integrate_fusion=_integrate_fusion,
    )


def _round(*, withheld: int, status: str = "complete") -> dict[str, Any]:
    """A round that kept nothing, reporting what the ceiling left unfunded."""
    return {
        "status": status,
        "kept": False,
        "patches": [],
        "nomination": {"candidates_seen": 6, "resolved": 6, "selected": 0, "withheld": withheld},
    }


@pytest.fixture(autouse=True)
def _no_skip_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SKIP_FUSION", raising=False)


def test_a_round_with_unfunded_targets_is_retried():
    """The withheld targets were never tried, so the round answers for nothing."""
    assert REQUIRED(_phase(_round(withheld=3))) is True


def test_repeated_unfunded_rounds_stop_being_retried():
    """Every retry re-runs discovery, so a target that keeps being re-discovered
    and re-withheld must not re-spend the budget without bound."""
    assert REQUIRED(_phase(_round(withheld=3), spent=MAX_FUSION_WITHHELD_RETRIES)) is False


def test_a_fully_funded_round_still_latches():
    """Nothing was withheld, so the round answers for the whole slate."""
    assert REQUIRED(_phase(_round(withheld=0))) is False


def test_a_round_without_a_nomination_summary_is_unaffected():
    """The combine path and every pre-contract record carry no summary."""
    assert REQUIRED(_phase({"status": "complete", "micro_decision": "no_improvement"})) is False


@pytest.mark.parametrize("withheld", ["2", None, "", "not-a-number", -1])
def test_an_unreadable_withheld_count_latches_as_before(withheld):
    """The count round-trips through state.json, so its type is not guaranteed.

    An unreadable value must not re-arm fusion on a round that may have been
    fully funded; only a positive count re-arms it.
    """
    assert REQUIRED(_phase(_round(withheld=withheld))) is False


def test_a_kept_round_is_still_blocked_by_its_own_status():
    """A KEEP is a real result; withheld targets do not re-open it."""
    kept = _round(withheld=3, status="ok")
    kept["kept"] = True
    assert REQUIRED(_phase(kept)) is False


@pytest.mark.asyncio
async def test_each_unfunded_round_increments_the_counter(tmp_path):
    """The count lives on the session: ``last_fusion`` is replaced every run, so
    holding it there would hand the cap a clean slate."""
    phase = _phase(None, session_dir=tmp_path)

    for expected in (1, 2, 3):
        await RECORD(phase, _round(withheld=2))
        assert phase.shared_state.fusion_withheld_retries == expected


@pytest.mark.asyncio
async def test_a_fully_funded_round_does_not_increment_the_counter(tmp_path):
    phase = _phase(None, session_dir=tmp_path)

    await RECORD(phase, _round(withheld=0))

    assert phase.shared_state.fusion_withheld_retries == 0


@pytest.mark.asyncio
async def test_the_counter_survives_a_state_round_trip(tmp_path):
    """An unpersisted counter would leave the cap unreachable across entries,
    which is the unbounded retry the cap exists to stop."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(tmp_path)
    state.framework = "sglang"
    state.last_profile_trace = "/tmp/decode.trace.json.gz"
    phase = SimpleNamespace(shared_state=state, session_dir=tmp_path)
    phase.bus = SimpleNamespace()

    async def _append_and_seq(_message):
        return None

    phase.bus.append_and_seq = _append_and_seq

    async def _integrate_fusion(_result):
        return None

    phase._integrate_fusion = _integrate_fusion

    for _ in range(MAX_FUSION_WITHHELD_RETRIES):
        await RECORD(phase, _round(withheld=2))

    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.fusion_withheld_retries == MAX_FUSION_WITHHELD_RETRIES
    phase.shared_state = reloaded
    assert REQUIRED(phase) is False
