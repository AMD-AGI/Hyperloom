# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""N36 — low-quality steady-state chunk auto-recovery (DSR1-0528 10k/1k case).

``_check_selected_chunk_has_gpu_events_quality`` emits ``steady_state_chunk_low_quality`` (N26 retry allowlist) when busy ratio is below threshold AND a meaningfully-better alternate exists; threshold via ``INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO`` (default 0.05).
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


# Quality gate behaviour
def test_dsr1_style_low_quality_chunk_emits_warning(tl_module, split_dir):
    """DSR1-0528 10k/1k case: 0.063%-busy mixed chunk fires ``steady_state_chunk_low_quality`` listing ``prefilldecode`` as alternate."""
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
    # Remediation must point at the env knob the N26 retry consumes.
    assert "INFERENCE_OPTIMIZER_STEADY_STATE_MODE" in result["remediation"]
    assert "prefilldecode" in result["remediation"]


def test_no_better_alternate_emits_no_warning(tl_module, split_dir):
    """All modes equally bad → no retry-warning (would loop forever); let ``roofline_failure_streak`` accumulate instead."""
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
    """60%-busy mixed chunk → quality gate returns None (happy path)."""
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
    """Edge case: gpu_duration=0 must not crash the gate."""
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
    result = tl_module._check_selected_chunk_has_gpu_events_quality(
        split_dir=split_dir, selected_chunk=mixed,
        mode="mixed", available_modes=available,
    )
    # gpu_duration==0 → either None or a low-quality warning, but must not crash.
    if result is not None:
        assert result["code"] == "steady_state_chunk_low_quality"


def test_missing_execution_details_csv_returns_none(tl_module, split_dir):
    """Back-compat: builds without execution_details.csv keep pass-through behaviour."""
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


# N26 retry allowlist contract
def test_n26_retry_allowlist_includes_low_quality_code():
    """Pin that the N26 retry allowlist includes the N36 code, else auto-retry never fires."""
    from inference_optimizer.orchestrator.action_executors import (
        roofline as ro,
    )
    assert "steady_state_chunk_low_quality" in ro._AUTO_RETRY_WARNING_CODES, (
        "N26 retry allowlist must include the N36 low-quality code; "
        "otherwise the chunk-quality gate has no recovery path and "
        "the optimization loop deadlocks on low-quality traces "
        "(see DSR1-0528 10k/1k case)"
    )
