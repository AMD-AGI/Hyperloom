# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A batch round records each kernel once.

``record_partial`` streams every sub-attempt into SharedState while the gather is
still running, so recording the aggregate afterwards charges the batch winner a
second attempt. The default partial cap is 2, so that second charge retires the
strongest kernel of the round after a single real attempt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState

_AUTO_ENV = "HYPERLOOM_FORGE_NOMINATION_AUTO"


class _Bus:
    """Serialises the row like the real bus, which is where a payload is rejected."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def append_and_seq(self, message: Any) -> None:
        message.to_db_row()
        self.sent.append(message)


def _row(kernel_id: str, root: Path) -> dict[str, Any]:
    source = root / f"{kernel_id}.py"
    source.write_text("def k(): return 1\n", encoding="utf-8")
    return {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "gpu_pct": 30.0,
        "source_file": str(source),
        "reusable_native_kernel": True,
        "skip_reason": "",
    }


@pytest.fixture
def phase(tmp_path, monkeypatch):
    """A KERNEL phase on the selector path with two routable candidates."""
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    monkeypatch.delenv(_AUTO_ENV, raising=False)
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "0.0")

    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = tmp_path / "kernel_candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "hot_kernels": [_row("k001", tmp_path), _row("k002", tmp_path)],
                "reusable_native_kernel_ids": ["k001", "k002"],
            }
        ),
        encoding="utf-8",
    )

    state = SharedState.load_or_init(tmp_path)
    state.max_minutes = 600.0
    state.last_profile_trace = str(trace)
    state.last_trace_analyze = {"candidates_path": str(candidates)}
    state.save(tmp_path)

    obj = KernelPhase.__new__(KernelPhase)
    object.__setattr__(obj, "shared_state", state)
    object.__setattr__(obj, "session_dir", tmp_path)
    object.__setattr__(obj, "bus", _Bus())
    obj._record_kernel_opt_partial = lambda result: state.record_kernel_opt(result)
    obj._record_kernel_opt_dispatch_skip = lambda _reason: None
    return obj


@pytest.fixture
def partial_backend(monkeypatch):
    """Every candidate comes back PARTIAL, so nothing is retired on its verdict."""
    from hyperloom.orchestrator.kernel import request_handlers as krh

    async def _sequence(base_payload, candidate, *, session_dir):
        kernel_id = str(candidate.get("kernel_id") or "")
        return {
            "status": "ok",
            "kernel_id": kernel_id,
            "source_file": str(candidate.get("source_file") or ""),
            "selected_backends": ["forge"],
            "verification": {"compile_passed": False, "correctness_passed": False, "micro_speedup": 1.0},
            "proposal": {"decision": "PARTIAL", "reasons": ["no measurable speedup found"]},
        }

    monkeypatch.setattr(krh, "_run_kernel_backend_sequence", _sequence)


def _ledger_by_kernel(state: Any, field: str) -> dict[str, int]:
    """The ledger keys on a composite task key, so read the counts by kernel id."""
    counts: dict[str, int] = {}
    for key, entry in (state.kernel_opt_task_attempts or {}).items():
        try:
            kernel_id = str(json.loads(key).get("kernel_id") or "")
        except (TypeError, ValueError):
            kernel_id = str(key)
        counts[kernel_id] = int((entry or {}).get(field, 0))
    return counts


@pytest.mark.asyncio
async def test_a_batch_charges_each_kernel_one_attempt(phase, partial_backend):
    """The aggregate carries the winner's kernel_id, so it must not be recorded again."""
    await phase._run_kernel_opt_nomination()

    assert _ledger_by_kernel(phase.shared_state, "attempts") == {"k001": 1, "k002": 1}


@pytest.mark.asyncio
async def test_a_batch_round_retires_nobody(phase, partial_backend):
    """One PARTIAL attempt is below the cap of 2, so the round blacklists nothing."""
    await phase._run_kernel_opt_nomination()

    state = phase.shared_state
    assert list(state.rejected_kernel_ids or []) == []
    assert _ledger_by_kernel(state, "partial_count") == {"k001": 1, "k002": 1}
