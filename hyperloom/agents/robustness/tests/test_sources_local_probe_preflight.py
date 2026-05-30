"""LocalProbe tests for the C-section preflight probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.agents.robustness.sources.local_probe import (
    LocalProbeConfig,
    LocalProbeSource,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_manifest_loads_when_present(tmp_path):
    sd = tmp_path
    manifest = {
        "schema_version": 2,
        "model_name": "Qwen3-32B",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {"precision": "bf16", "max_model_len": 4096, "conc": 16},
    }
    _write_json(sd / "manifest.json", manifest)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == manifest


@pytest.mark.asyncio
async def test_manifest_empty_when_missing(tmp_path):
    sd = tmp_path
    # Add minimal other probe data so we don't trip SourceUnavailable.
    _write_json(sd / "results" / "ci_metrics.json", {"model": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}


@pytest.mark.asyncio
async def test_manifest_empty_on_malformed_json(tmp_path, monkeypatch):
    sd = tmp_path
    (sd / "manifest.json").write_text("not json at all", encoding="utf-8")
    # Also drop a small breakdown so the probe still returns data (the
    # rest of the test verifies graceful handling, not SourceUnavailable).
    _write_json(sd / "profiles" / "kernel_breakdown.json", [])
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    # Silence other probes that may pick up host data unrelated to the
    # malformed manifest under test.
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_gpu", lambda: {}
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_aiter_jit",
        lambda _d: {},
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}


@pytest.mark.asyncio
async def test_kernel_breakdown_aggregates_by_tier(tmp_path):
    sd = tmp_path
    rows = [
        {"name": "triton_red_x", "gpu_pct": 12.0, "tier": "T1_TRITON",
         "count": 100, "duration_us": 1000},
        {"name": "triton_poi_y", "gpu_pct": 8.5, "tier": "T1_TRITON",
         "count": 80, "duration_us": 800},
        {"name": "mha_fwd_kernel", "gpu_pct": 30.0, "tier": "T2_AITER_CK",
         "count": 50, "duration_us": 2000},
        {"name": "rccl_allreduce", "gpu_pct": 20.0, "tier": "T4_COMM",
         "count": 200, "duration_us": 500},
        {"name": "hipblaslt_gemm", "gpu_pct": 25.0, "tier": "T5_COMPILED",
         "count": 30, "duration_us": 1500},
    ]
    _write_json(sd / "profiles" / "kernel_breakdown.json", rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    breakdown = data.local_kernel_breakdown
    # Triton aggregates 12+8.5 = 20.5
    assert breakdown["tier_pcts"]["triton"] == pytest.approx(20.5, rel=1e-3)
    assert breakdown["tier_pcts"]["vendor"] == 30.0
    assert breakdown["tier_pcts"]["comm"] == 20.0
    assert breakdown["tier_pcts"]["compiled"] == 25.0
    assert breakdown["total_kernels"] == 5
    assert breakdown["total_gpu_pct"] == pytest.approx(95.5, rel=1e-3)
    assert breakdown["kernel_breakdown_path"].endswith("kernel_breakdown.json")
    assert isinstance(breakdown["mtime"], float)


@pytest.mark.asyncio
async def test_kernel_breakdown_empty_when_missing(tmp_path):
    sd = tmp_path
    _write_json(sd / "manifest.json", {"model_name": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_kernel_breakdown == {}


@pytest.mark.asyncio
async def test_preflight_disabled_skips_manifest_and_breakdown(
    tmp_path, monkeypatch,
):
    """When ``preflight_enabled=False`` both new slots stay empty even
    when the files exist on disk."""
    sd = tmp_path
    _write_json(sd / "manifest.json", {"model_name": "X"})
    _write_json(sd / "profiles" / "kernel_breakdown.json", [
        {"name": "k", "gpu_pct": 50.0, "tier": "T1_TRITON",
         "count": 1, "duration_us": 100},
    ])
    # Force the host probes to be empty so we're sure the only data
    # source SourceUnavailable would care about is the preflight slot.
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_gpu", lambda: {}
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_aiter_jit",
        lambda _d: {},
    )
    # Add a tiny disk sample so the probe has *some* data and doesn't
    # raise SourceUnavailable (we're testing preflight slot behaviour,
    # not the unavailable path).
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(str(sd),),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
        preflight_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}
    assert data.local_kernel_breakdown == {}


@pytest.mark.asyncio
async def test_kernel_breakdown_unknown_tier_falls_back_to_lowered_name(tmp_path):
    sd = tmp_path
    rows = [
        {"name": "k1", "gpu_pct": 50.0, "tier": "T6_NEW_TIER",
         "count": 1, "duration_us": 100},
        {"name": "k2", "gpu_pct": 50.0, "tier": "",
         "count": 1, "duration_us": 100},
    ]
    _write_json(sd / "profiles" / "kernel_breakdown.json", rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    # Unknown tier surfaces as lowered name; empty tier collapses to "unknown".
    assert data.local_kernel_breakdown["tier_pcts"]["t6_new_tier"] == 50.0
    assert data.local_kernel_breakdown["tier_pcts"]["unknown"] == 50.0
