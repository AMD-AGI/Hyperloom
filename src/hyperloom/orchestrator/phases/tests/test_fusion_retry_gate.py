# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How KERNEL entry re-arms forge-fusion after it aborted on infrastructure.

Exercised against ``KernelPhase`` unbound, on a stand-in ``self``: the gate reads
two state fields and nothing else, while building a Coordinator would pull in the
whole orchestrator for no added coverage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.kernel import MAX_FUSION_INFRA_RETRIES, KernelPhase

REQUIRED = KernelPhase._fusion_required_before_kernel_opt
RECORD = KernelPhase._handle_fusion_result


def _phase(last_fusion, *, spent=0, session_dir=None):
    """A stand-in carrying only what the two methods under test read."""
    saved = []
    state = SimpleNamespace(
        framework="sglang",
        last_profile_trace="/tmp/decode.trace.json.gz",
        last_fusion=last_fusion,
        fusion_infra_aborts=spent,
        save=lambda *a, **k: saved.append(a),
    )
    bus = SimpleNamespace(posted=[])

    async def _append_and_seq(message):
        bus.posted.append(message)

    bus.append_and_seq = _append_and_seq
    return SimpleNamespace(shared_state=state, bus=bus, session_dir=session_dir)


def _abort(reason="no_git_workspace"):
    return {"status": "failed", "error_class": reason, "infrastructure_abort": True}


@pytest.fixture(autouse=True)
def _no_skip_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SKIP_FUSION", raising=False)


def test_an_infrastructure_abort_is_retried():
    """An abort judged nothing about the kernel, so the next entry must retry it.

    This is the whole point of reporting it as ``failed``: the pre-fix
    ``complete`` satisfied this gate and ended fusion for the session.
    """
    assert REQUIRED(_phase(_abort(), spent=1)) is True


@pytest.mark.parametrize("spent", [MAX_FUSION_INFRA_RETRIES, MAX_FUSION_INFRA_RETRIES + 3])
def test_repeated_infrastructure_aborts_stop_being_retried(spent):
    """Retrying is not free: every run re-does LLM discovery before failing in
    the same place, and a missing git workspace does not heal mid-session."""
    assert REQUIRED(_phase(_abort(), spent=spent)) is False


def test_giving_up_on_aborts_does_not_restore_the_no_improvement_report():
    """The cap bounds the retries; it must not bring back the bug it replaced.

    The old behaviour recorded an abort as ``complete``/``no_improvement`` -- a
    claim that the model has no fusion opportunity. Capping only stops re-running
    it; the record still has to read as infrastructure that gave up.
    """
    phase = _phase(_abort(), spent=MAX_FUSION_INFRA_RETRIES)

    assert REQUIRED(phase) is False
    assert phase.shared_state.last_fusion["status"] == "failed"
    assert phase.shared_state.last_fusion["status"] not in ("ok", "complete", "kept")
    assert phase.shared_state.last_fusion["error_class"] == "no_git_workspace"


@pytest.mark.parametrize("spent", ["1", None, "", "not-a-number"])
def test_a_non_numeric_counter_does_not_end_fusion(spent):
    """The counter round-trips through state.json, so its type is not guaranteed.

    Failing closed here would silently reproduce the bug under repair, so an
    unreadable count has to mean "nothing spent yet", not "give up".
    """
    assert REQUIRED(_phase(_abort(), spent=spent)) is True


def test_a_result_that_is_not_an_abort_is_unaffected():
    """Only aborts are capped; every other record still uses the status gate."""
    assert REQUIRED(_phase({"status": "failed", "error_class": "TimeoutExpired"})) is True
    assert REQUIRED(_phase({"status": "complete", "micro_decision": "no_improvement"})) is False
    assert REQUIRED(_phase(None)) is True


@pytest.mark.asyncio
async def test_each_abort_increments_the_counter(tmp_path):
    """The count has to survive the record being replaced, or the cap never
    triggers and the retries stay unbounded."""
    phase = _phase(None, session_dir=tmp_path)

    for expected in (1, 2, 3):
        await RECORD(phase, _abort("harness_author_failed"))
        assert phase.shared_state.fusion_infra_aborts == expected


@pytest.mark.asyncio
async def test_an_unrelated_failure_between_aborts_does_not_reset_the_cap(tmp_path):
    """The regression this counter's placement exists to prevent.

    ``last_fusion`` is replaced by every run, so holding the count there let any
    non-abort outcome -- a subprocess timeout, a handler crash -- hand the cap a
    clean slate. Alternating abort/timeout would then retry forever, which is
    exactly the unbounded discovery spend the cap was added to stop.
    """
    phase = _phase(None, session_dir=tmp_path)

    await RECORD(phase, _abort("no_git_workspace"))
    assert REQUIRED(phase) is True

    # A wholly unrelated failure. It must not block fusion, and must not forgive
    # the abort already spent.
    await RECORD(phase, {"status": "failed", "error_class": "TimeoutExpired"})
    assert phase.shared_state.fusion_infra_aborts == 1

    await RECORD(phase, _abort("no_git_workspace"))
    assert phase.shared_state.fusion_infra_aborts == MAX_FUSION_INFRA_RETRIES
    assert REQUIRED(phase) is False


@pytest.mark.asyncio
async def test_the_count_stays_off_the_run_record(tmp_path):
    """``last_fusion`` describes one run; the tally belongs to the session.

    Keeping them apart is what makes the count survive an unrelated result, and
    keeps a per-run record from reporting a number that is not about that run.
    """
    phase = _phase(None, session_dir=tmp_path)

    await RECORD(phase, _abort("harness_author_failed"))

    assert phase.shared_state.fusion_infra_aborts == 1
    assert "infrastructure_aborts" not in phase.shared_state.last_fusion


@pytest.mark.asyncio
async def test_a_real_result_does_not_forgive_spent_aborts(tmp_path):
    """A loop that ran is a result, not absolution.

    It blocks the gate on its own status anyway, so clearing the tally would buy
    nothing and would reopen the reset the placement above closes.
    """
    phase = _phase(_abort(), spent=2, session_dir=tmp_path)

    await RECORD(phase, {"status": "complete", "micro_decision": "no_improvement"})

    assert phase.shared_state.fusion_infra_aborts == 2
    assert phase.shared_state.last_fusion["status"] == "complete"
