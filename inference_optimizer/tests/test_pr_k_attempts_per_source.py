"""PR-K — per-source attempts ledger unlocks device retry post-promotion.

Background
----------
Pre-PR-K, ``_batch_kernel_candidates._is_live`` consulted only the
cumulative ``attempts`` counter on each ``kernel_opt_attempts[kid]``
entry. With ``max_attempts=1`` (the default), a single attempt against
the python ``@compile_ops`` wrapper (e.g. ``aiter/ops/moe_op.py``) would
flip every member of that kernel's task_group to ``not_live`` on the
next batch — even though the wrapper rewrite was structurally a no-op
and the device source ``.cu`` had never been tried. Qwen3-30B-A3B-Base
session 20260523T162026Z burned its full 2h budget after one wrapper
attempt produced ``group_exhausted`` for every reusable hot kernel.

PR-K's promotion (see ``upgrade_aiter_compile_ops_launcher``) flips
``source_file`` to the device ``.cu`` on the next batch. This test file
pins the contract that the new ``attempts_per_source`` ledger plus
``_is_live`` per-source check actually unlock the device retry:

* ``record_kernel_opt`` writes the cumulative ``attempts`` AND the new
  ``attempts_per_source[source_file]`` counter.
* ``_is_live(kid, current_source)`` returns True when the entry has a
  per-source row showing ``attempts_per_source[current_source] <
  max_attempts``, even when the cumulative ``attempts`` exceeds the
  threshold (because all those attempts were against a different
  source path).
* ``_is_live`` falls back to cumulative ``attempts`` when the entry
  predates per-source tracking (resume from a v1 state.json) or when
  the caller passes an empty source — pre-PR-K behaviour preserved
  byte-for-byte for legacy callers.
* End-to-end: a candidate whose pre-batch ``source_file`` was promoted
  from wrapper to device source IS dispatched even though the kernel
  has a recorded wrapper attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = tmp_path
    (sd / "manifest.json").write_text("{}", encoding="utf-8")
    return sd


@pytest.fixture
def candidates_factory(tmp_path: Path):
    """Write a kernel_candidates.json fixture and return its path."""

    def _make(hot_kernels: list[dict], task_groups: list[dict] | None = None) -> str:
        path = tmp_path / "kernel_candidates.json"
        path.write_text(
            json.dumps({
                "hot_kernels": hot_kernels,
                "task_groups": task_groups or [],
                "reusable_native_kernel_ids": [],
            }),
            encoding="utf-8",
        )
        return str(path)

    return _make


# ---------------------------------------------------------------------------
# record_kernel_opt — writes the per-source ledger.
# ---------------------------------------------------------------------------
def test_record_kernel_opt_writes_attempts_per_source(
    session_dir: Path,
) -> None:
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok",
        "kernel_id": "k001",
        "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.02},
    })
    entry = state.kernel_opt_attempts["k001"]
    assert entry["attempts"] == 1
    assert entry["attempts_per_source"] == {
        "/sgl-workspace/aiter/aiter/ops/moe_op.py": 1,
    }


def test_record_kernel_opt_increments_per_source_independently(
    session_dir: Path,
) -> None:
    """Same kernel attempted against two distinct source paths produces
    a per-source dict with separate counters; cumulative ``attempts``
    sums across all sources."""
    state = SharedState.load_or_init(session_dir)
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 0.99},
    })
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": device,
        "proposal": {"decision": "REVERT"},
        "verification": {"micro_speedup": 0.95},
    })
    entry = state.kernel_opt_attempts["k001"]
    assert entry["attempts"] == 2
    assert entry["attempts_per_source"] == {wrapper: 1, device: 1}


def test_record_kernel_opt_normalizes_empty_source_file(
    session_dir: Path,
) -> None:
    """When the result lacks a source_file (legacy caller / failed
    attempt before resolution), the empty key ``""`` is used so the
    ledger remains a valid dict and resume from a v1 state.json with no
    per-source field is bit-for-bit transparent."""
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "failed", "kernel_id": "k042",
        # source_file omitted on purpose.
        "proposal": {"decision": "REVERT"},
    })
    entry = state.kernel_opt_attempts["k042"]
    assert entry["attempts_per_source"] == {"": 1}


# ---------------------------------------------------------------------------
# End-to-end: promotion unlocks a fresh attempt against the device source.
# ---------------------------------------------------------------------------
def test_batch_candidates_unlocks_promoted_device_source_after_wrapper_attempt(
    session_dir: Path, candidates_factory,
) -> None:
    """The PR-K core scenario.

    Round 1: kernel was dispatched against the python wrapper, came
    back PARTIAL → ``attempts_per_source[wrapper] = 1``, kernel is on
    the cumulative attempts list.

    Round 2: tracelens_analysis re-ran and promoted the source from
    wrapper to device .cu (kernel name like ``aiter::ck_moe_stage1``
    triggers ``upgrade_aiter_compile_ops_launcher``); the candidates
    file now lists the device path.

    Pre-PR-K: ``_is_live(kid)`` returned False because cumulative
    ``attempts >= 1`` → ``group_exhausted`` → integrate never fires.
    Post-PR-K: ``_is_live(kid, device_path)`` consults
    ``attempts_per_source[device_path]`` (zero) and returns True →
    candidate is dispatched.
    """
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state = SharedState.load_or_init(session_dir)
    # Simulate a wrapper attempt that PARTIAL'd.
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.01},
    })
    state.save(session_dir)

    # Round 2 candidates: source_file now points at device .cu.
    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": device,
            "name": "aiter::ck_moe_stage1",
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert [c["kernel_id"] for c in out] == ["k001"], (
        "promoted device source must unlock a fresh dispatch"
    )


def test_batch_candidates_skips_kernel_when_same_source_already_attempted(
    session_dir: Path, candidates_factory,
) -> None:
    """Mirror image of the previous test: when round 2's candidate
    source_file MATCHES the round-1 attempted source, the kernel is
    correctly skipped (no infinite-attempt loop on the same target)."""
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"

    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.0},
    })
    state.save(session_dir)

    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": wrapper,  # same as the recorded attempt
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert out == []


def test_batch_candidates_falls_back_to_cumulative_for_legacy_entry(
    session_dir: Path, candidates_factory,
) -> None:
    """Resume from a pre-PR-K state.json: the entry has cumulative
    ``attempts`` but no ``attempts_per_source``. ``_is_live`` must fall
    back to the cumulative counter so old sessions keep their pre-PR-K
    skip semantics (no surprise floods of re-attempts on resume)."""
    state = SharedState.load_or_init(session_dir)
    # Synthesize a v1 entry by writing kernel_opt_attempts directly,
    # WITHOUT going through record_kernel_opt (which would populate the
    # new field).
    state.kernel_opt_attempts = {
        "k001": {
            "attempts": 1, "partial_count": 1,
            "last_decision": "PARTIAL",
            "last_source_file": "/p/moe_op.py",
        },
    }
    state.save(session_dir)

    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": "/p/different.cu",  # different source on round 2
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    # Legacy entry: no per-source ledger → cumulative ``attempts >= 1``
    # → skipped, even though on-disk source differs. This is the
    # conservative resume contract the user requested for PR-K.
    assert out == []


def test_batch_candidates_task_group_promotion_unlocks_primary(
    session_dir: Path, candidates_factory,
) -> None:
    """Task_group flow: primary k002 was dispatched against the wrapper
    and PARTIAL'd. Round 2 promotes both members to the device source.
    The group must dispatch as primary again (k002), not skip with
    ``group_exhausted``."""
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k002",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.0},
    })
    state.save(session_dir)

    cpath = candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
             "source_file": device, "name": "aiter::ck_moe_stage1"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
             "source_file": device, "name": "aiter::ck_moe_stage2"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert len(out) == 1
    assert out[0]["kernel_id"] == "k002"
    assert out[0]["source_file"] == device
