# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""N36 — low-quality steady-state chunk auto-recovery.

Background — May 2026 DSR1-0528 (671B FP8 MoE) TP=8 / ISL=10240 OSL=1024
CONC=64 production trace:

The TraceLens splitter produced a ``mixed_steady_state`` chunk with
``num_gpu_events=160`` and ``gpu_busy_duration=2053us`` out of
``gpu_duration=3,257,719us`` (0.06% busy). The N25 gate passed
(structural emptiness check: both ``num_gpu_events > 0`` AND
``gpu_busy_duration > 0`` were satisfied). The N26 retry never fired
(``steady_state_chunk_empty`` warning code never emitted).
Downstream analysis.md reported "Compute %=0.09%, Idle %=99.90%"
and ``reusable_native_kernel_ids=[]`` -- the LLM had no kernel_opt
candidates to feed GEAK with.

Root cause: TraceLens' ``delay_iters`` formula in
``_workload_envs._materialize_config_with_envs`` only considers OSL,
so a 10k/1k workload (10x ISL prefill) and a 1k/1k workload land at
the same ``start_step=6016``. For high-ISL workloads, by step 6016
every batch has finished its single prefill iter and the profiler
captures only decode iters -- where CONC=64 streams produce sparse
small kernels that don't fill 8 MI300X. The chunk passes structural
checks but is substantively garbage.

N36 closes the gap: ``_check_selected_chunk_has_gpu_events_quality``
fires when the selected chunk's busy ratio falls below a threshold
AND at least one alternate mode has a meaningfully higher busy
ratio. The warning code (``steady_state_chunk_low_quality``) is in
the N26 auto-retry allowlist so coordinator re-issues trace_analyze
with the best alternate mode automatically.

Tests:

* Low busy_ratio with a high-quality alternate → warning with
  ``code=steady_state_chunk_low_quality`` listing the alternate.
* Low busy_ratio but no better alternate → NO warning (avoids
  infinite retry loop where every mode is equally bad; the failure
  surfaces via the existing ``roofline_failure_streak`` path).
* High busy_ratio everywhere → NO warning (happy path unchanged).
* Threshold is configurable via env
  ``INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO`` (default 0.05).
* N26 retry allowlist includes the new warning code so the existing
  retry-on-alternate-mode logic in ``roofline._extract_steady_state_
  retry_mode`` picks it up without further wiring.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parent
TL_PATH = TOOLS_DIR / "tracelens_analysis.py"


@pytest.fixture(scope="module")
def tl_module():
    spec = importlib.util.spec_from_file_location(
        "tracelens_analysis_n36_under_test", TL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_exec_details(
    split_dir: Path, rows: list[dict[str, object]],
) -> Path:
    path = split_dir / "execution_details.csv"
    cols = [
        "idx", "output_path", "event_count", "num_gpu_events",
        "gpu_duration", "gpu_busy_duration",
        "phase_num_prefill", "phase_num_prefilldecode", "phase_num_decode",
        "phase_avg_bs", "phase_avg_conc", "num_steps",
    ]
    with path.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            full = {c: "" for c in cols}
            full.update({k: str(v) for k, v in row.items()})
            w.writerow(full)
    return path


def _make_chunks(split_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for label in (
        "mixed_steady_state",
        "decode_only_steady_state",
        "prefilldecode_steady_state",
    ):
        p = split_dir / f"{label}_chunk.trace.json.gz"
        p.write_bytes(b"")
        out[label] = p
    return out


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "trace_split"
    sd.mkdir()
    return sd


# ===========================================================================
# Quality gate behaviour
# ===========================================================================
def test_dsr1_style_low_quality_chunk_emits_warning(tl_module, split_dir):
    """DSR1-0528 10k/1k case: mixed chunk has 160 events / 2053us busy /
    3.26s duration (0.063% busy). prefilldecode has 60% busy. The N36
    quality gate must fire with code ``steady_state_chunk_low_quality``
    listing ``prefilldecode`` as the alternate."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    pd = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed),
            "num_gpu_events": 160,
            "gpu_duration": 3257719.65,
            "gpu_busy_duration": 2053.09,  # 0.063% busy
        },
        {
            "output_path": str(pd),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,  # 60% busy
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd]),
    }
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir,
        selected_chunk=mixed,
        mode="mixed",
        available_modes=available,
    )
    assert result is not None
    assert result["code"] == "steady_state_chunk_low_quality"
    assert result["requested_mode"] == "mixed"
    assert result["busy_ratio"] < 0.01
    assert "prefilldecode" in result["non_empty_modes"]
    # Remediation message must point operator at the env knob the
    # coordinator's N26 retry consumes.
    assert "INFERENCE_OPTIMIZER_STEADY_STATE_MODE" in result["remediation"]
    assert "prefilldecode" in result["remediation"]


def test_no_better_alternate_emits_no_warning(tl_module, split_dir):
    """All modes are equally bad → don't emit a retry-warning.
    Surfacing here would create an infinite-retry loop because every
    retry target is also low-quality. The session should instead let
    ``roofline_failure_streak`` accumulate so the existing N27
    fallback can downgrade gates."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    pd = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed),
            "num_gpu_events": 160,
            "gpu_duration": 3257719.65,
            "gpu_busy_duration": 2053.09,  # 0.063%
        },
        {
            "output_path": str(pd),
            "num_gpu_events": 100,
            "gpu_duration": 3000000.0,
            "gpu_busy_duration": 1500.0,  # 0.05%
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd]),
    }
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir,
        selected_chunk=mixed,
        mode="mixed",
        available_modes=available,
    )
    assert result is None, (
        "every mode is low-quality; emitting a retry warning would "
        "spin the same bad workload forever"
    )


def test_high_quality_chunk_passes(tl_module, split_dir):
    """Qwen1.5-7B-style mixed chunk: 60% busy → quality gate must
    return None (happy path unchanged)."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed),
            "num_gpu_events": 38796,
            "gpu_duration": 3000000.0,
            "gpu_busy_duration": 1800000.0,  # 60%
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", []),
    }
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir,
        selected_chunk=mixed,
        mode="mixed",
        available_modes=available,
    )
    assert result is None


def test_quality_threshold_overridable_via_env(
    tl_module, split_dir, monkeypatch,
):
    """Operator can tighten / loosen the quality threshold via env."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    pd = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed),
            "num_gpu_events": 1000,
            "gpu_duration": 1000000.0,
            "gpu_busy_duration": 80000.0,  # 8% busy
        },
        {
            "output_path": str(pd),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,  # 60% busy
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd]),
    }
    # Default threshold 5%: 8% chunk passes.
    monkeypatch.delenv(
        "INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO", raising=False,
    )
    assert tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir, selected_chunk=mixed,
        mode="mixed", available_modes=available,
    ) is None
    # Tightened to 20%: same chunk now fails.
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO", "0.20",
    )
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir, selected_chunk=mixed,
        mode="mixed", available_modes=available,
    )
    assert result is not None
    assert result["code"] == "steady_state_chunk_low_quality"


def test_zero_duration_does_not_divide_by_zero(tl_module, split_dir):
    """Edge case: gpu_duration=0 must not crash the gate; treat as
    low-quality (no events to measure) when an alternate has events."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    pd = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed),
            "num_gpu_events": 5,
            "gpu_duration": 0.0,
            "gpu_busy_duration": 0.0,
        },
        {
            "output_path": str(pd),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd]),
    }
    # Should not raise ZeroDivisionError.
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir, selected_chunk=mixed,
        mode="mixed", available_modes=available,
    )
    # gpu_duration==0 → cannot compute ratio; downgrade to N25
    # structural-empty semantics: leave the existing gate to surface it.
    # Either None or a low-quality warning is acceptable as long as it
    # doesn't crash.
    if result is not None:
        assert result["code"] == "steady_state_chunk_low_quality"


def test_missing_execution_details_csv_returns_none(tl_module, split_dir):
    """Back-compat: older TraceLens builds without execution_details.csv
    keep the existing pass-through behaviour."""
    chunks = _make_chunks(split_dir)
    mixed = chunks["mixed_steady_state"]
    available = {
        "mixed": ("mixed_steady_state", [mixed]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", []),
    }
    assert tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir, selected_chunk=mixed,
        mode="mixed", available_modes=available,
    ) is None


# ===========================================================================
# N26 retry allowlist contract
# ===========================================================================
def test_n26_retry_allowlist_includes_low_quality_code():
    """Without this entry the coordinator's _extract_steady_state_retry_
    mode helper would ignore the new warning and the auto-retry never
    fires. Pin the contract explicitly so a future cleanup doesn't
    drop the code."""
    # Import roofline executor module — its module-level constant is
    # the contract surface.
    # The inference_optimizer package is installed editable on this
    # repo (pip install -e .[test] per inference_optimizer/scripts/
    # install.sh); the test is bookkeeping for the symbol contract.
    from inference_optimizer.orchestrator.action_executors import (
        roofline as ro,
    )
    assert "steady_state_chunk_low_quality" in ro._AUTO_RETRY_WARNING_CODES, (
        "N26 retry allowlist must include the N36 low-quality code; "
        "otherwise the chunk-quality gate has no recovery path and "
        "the optimization loop deadlocks on low-quality traces "
        "(see DSR1-0528 10k/1k case)"
    )
