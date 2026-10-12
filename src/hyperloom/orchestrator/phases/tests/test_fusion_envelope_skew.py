# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fusion KEEP whose nomination envelope cannot be read judged nothing."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.tests.conftest import make_coordinator
from hyperloom.orchestrator.phases.kernel import MAX_FUSION_INFRA_RETRIES, KernelPhase

REQUIRED = KernelPhase._fusion_required_before_kernel_opt
RECORD = KernelPhase._handle_fusion_result


def _phase(*, spent: int = 0, session_dir):
    """The real kernel phase over a real session, with the bus and integration recorded."""
    coord = make_coordinator(session_dir)
    state = coord.shared_state
    state.framework = "sglang"
    state.last_profile_trace = "/tmp/decode.trace.json.gz"
    state.fusion_infra_aborts = spent
    state.macro_cycle = 1
    posted: list = []
    integrated: list[dict] = []

    async def _append_and_seq(message):
        posted.append(message)

    async def _integrate_fusion(result):
        integrated.append(result)

    coord.bus.append_and_seq = _append_and_seq
    phase = coord.phase_kernel
    phase._integrate_fusion = _integrate_fusion
    phase.posted = posted
    phase.integrated = integrated
    return phase


def _kept(**extra: Any) -> dict[str, Any]:
    """A KEPT fusion awaiting the e2e re-baseline, as the producer reports it."""
    base = {
        "status": "ok",
        "engine": "forge_fusion",
        "kept": True,
        "requires_e2e_validation": True,
        "patch": "/out/fusion.patch",
        "source_file": "/repo/model.py",
        "kernel_repo": "/repo",
    }
    base.update(extra)
    return base


def _sibling() -> dict[str, Any]:
    return {
        "kernel_name": "llm:rmsnorm_fuse",
        "patch_path": "/out/fusion_rmsnorm.patch",
        "target_file": "/repo/model.py",
        "micro_speedup": 1.4,
    }


@pytest.fixture(autouse=True)
def _no_skip_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SKIP_FUSION", raising=False)


@pytest.mark.asyncio
async def test_an_unreadable_keep_envelope_is_not_recorded_as_a_success(tmp_path):
    """Recorded as ``ok`` it would satisfy the gate and end fusion for the session."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, _kept())

    assert phase.shared_state.last_fusion["status"] == "failed"
    assert phase.shared_state.last_fusion["error_class"] == "nomination_envelope_skew"


@pytest.mark.asyncio
async def test_an_unreadable_keep_envelope_leaves_fusion_retryable(tmp_path):
    """It judged nothing about the model, so the next entry must try again."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, _kept())

    assert REQUIRED(phase) is True


@pytest.mark.asyncio
async def test_repeated_unreadable_envelopes_stop_being_retried(tmp_path):
    """A producer too old to answer the contract does not heal mid-session, and every retry re-runs LLM discovery before failing the same way."""
    phase = _phase(spent=MAX_FUSION_INFRA_RETRIES - 1, session_dir=tmp_path)

    await RECORD(phase, _kept())

    assert phase.shared_state.fusion_infra_aborts == MAX_FUSION_INFRA_RETRIES
    assert REQUIRED(phase) is False


@pytest.mark.asyncio
async def test_the_bus_and_the_record_agree_on_the_failure(tmp_path):
    """Both are written from this result, so a skew must not leave them disagreeing."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, _kept())

    (message,) = phase.posted
    assert message.payload["status"] == "failed"
    assert message.payload["status"] == phase.shared_state.last_fusion["status"]


@pytest.mark.asyncio
async def test_a_clean_empty_nomination_still_latches_the_round(tmp_path):
    """An explicit empty ``patches`` is a real answer: the round kept nothing."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, _kept(patches=[]))

    assert phase.shared_state.last_fusion["status"] == "ok"
    assert REQUIRED(phase) is False


@pytest.mark.asyncio
async def test_a_readable_nomination_is_untouched(tmp_path):
    """The live multi-patch path must be byte-for-byte what it was."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, _kept(patches=[_sibling()]))

    assert phase.shared_state.last_fusion["status"] == "ok"
    assert "error_class" not in phase.shared_state.last_fusion
    assert phase.shared_state.fusion_infra_aborts == 0
    assert len(phase.integrated) == 1


@pytest.mark.asyncio
async def test_a_run_that_kept_nothing_is_never_inspected(tmp_path):
    """The contract is only read for a KEEP awaiting validation."""
    phase = _phase(session_dir=tmp_path)

    await RECORD(phase, {"status": "complete", "kept": False, "micro_decision": "no_improvement"})

    assert phase.shared_state.last_fusion["status"] == "complete"
    assert phase.shared_state.fusion_infra_aborts == 0
    assert phase.integrated == []
