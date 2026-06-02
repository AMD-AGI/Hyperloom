"""Unit tests for LocalProbe ``_sample_decision_audit`` (G data path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness_agent.sources.local_probe import (
    LocalProbeConfig,
    LocalProbeSource,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.mark.asyncio
async def test_decision_audit_empty_when_no_session_dir():
    cfg = LocalProbeConfig(session_dir=None)
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_decision_audit == {}


@pytest.mark.asyncio
async def test_decision_audit_scans_integrate_result_files(tmp_path):
    """Newest integrate ``result.json`` files are surfaced sorted by mtime."""
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate"
    patch_path = sd / "patch.diff"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(b"diff --git a/x b/x\n")
    # Three integrate result.json files.
    for i, decision in enumerate(["REVERT", "KEEP", "NEEDS_REVIEW"]):
        out = integrate_root / f"task-{i}" / "result.json"
        _write_json(out, {
            "kernel_id": f"k{i}",
            "decision": decision,
            "gain_pct": float(i),
            "base_tput": 100.0,
            "new_tput": 100.0 + i,
            "patch_path": str(patch_path),
        })

    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False,
        fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    entries = audit["recent_integrate"]
    assert len(entries) == 3
    # Each entry has the projected shape.
    decisions = {e["decision"] for e in entries}
    assert decisions == {"REVERT", "KEEP", "NEEDS_REVIEW"}
    # patch_size_bytes was file-stat'd from disk.
    for e in entries:
        assert e["patch_size_bytes"] == len(b"diff --git a/x b/x\n")


@pytest.mark.asyncio
async def test_decision_audit_patch_size_none_when_patch_missing(tmp_path):
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {
        "kernel_id": "k1",
        "decision": "KEEP",
        "gain_pct": 5.0,
        "patch_path": "/nonexistent/patch.diff",
    })
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    entry = data.local_decision_audit["recent_integrate"][0]
    assert entry["patch_size_bytes"] is None


@pytest.mark.asyncio
async def test_decision_audit_loads_ci_metrics_from_results_dir(tmp_path):
    sd = tmp_path
    ci_path = sd / "results" / "ci_metrics.json"
    _write_json(ci_path, {
        "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
        "baseline_tok_per_gpu": 1500.0,
        "optimized_tok_per_gpu": 1800.0,
        "gain_pct": 20.0,
    })
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"]["model"] == "X"
    assert audit["ci_metrics_path"].endswith("ci_metrics.json")


@pytest.mark.asyncio
async def test_decision_audit_prefers_final_over_inflight(tmp_path):
    sd = tmp_path
    inflight = sd / "results" / "ci_metrics.json"
    final = sd / "results" / "ci_metrics_final.json"
    _write_json(inflight, {"baseline_tput": 0.0})
    _write_json(final, {"baseline_tput": 1500.0, "model": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"]["baseline_tput"] == 1500.0
    assert audit["ci_metrics_path"].endswith("ci_metrics_final.json")


@pytest.mark.asyncio
async def test_decision_audit_ci_metrics_empty_when_missing(tmp_path):
    sd = tmp_path
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    # Make session_dir non-empty so SourceUnavailable isn't raised — give
    # it a tiny ``runs/integrate`` tree.
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {"decision": "KEEP"})
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"] == {}
    assert audit["ci_metrics_path"] == ""


@pytest.mark.asyncio
async def test_decision_audit_tails_oob_attempts(tmp_path):
    sd = tmp_path
    oob_path = (
        sd / "kernel-agent" / "runs" / "sess-1"
        / "optimization_attempts.jsonl"
    )
    rows = [
        {
            "kernel_id": "gemm_a8w8",
            "backend": "oob_claude",
            "report_text": "expected speedup ~1.1x",
            "microbench_speedup": None,
            "ts": "2026-05-11T09:30:19",
        },
        {
            "kernel_id": "rmsnorm",
            "backend": "geak",
            "report_text": "measured 1.18x",
            "microbench_speedup": 1.18,
            "ts": "2026-05-11T10:00:00",
        },
    ]
    _write_jsonl(oob_path, rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert len(audit["oob_attempts"]) == 2
    first = audit["oob_attempts"][0]
    assert first["backend"] == "oob_claude"
    assert first["microbench_speedup"] is None


@pytest.mark.asyncio
async def test_decision_audit_max_integrate_caps_count(tmp_path):
    sd = tmp_path
    # 30 result files; ``decision_audit_max_integrate=5`` should keep
    # only the 5 most-recent.
    for i in range(30):
        out = sd / "runs" / "integrate" / f"t{i}" / "result.json"
        _write_json(out, {"decision": "KEEP", "kernel_id": f"k{i}"})
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        decision_audit_max_integrate=5,
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert len(data.local_decision_audit["recent_integrate"]) == 5


@pytest.mark.asyncio
async def test_decision_audit_disable_returns_empty(tmp_path):
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {"decision": "KEEP"})
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        decision_audit_enabled=False,
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_decision_audit == {}


@pytest.mark.asyncio
async def test_decision_audit_handles_malformed_json_gracefully(tmp_path):
    sd = tmp_path
    out = sd / "runs" / "integrate" / "t1" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("this is not json", encoding="utf-8")
    # Also write a valid one to confirm it still gets picked up.
    good = sd / "runs" / "integrate" / "t2" / "result.json"
    _write_json(good, {"decision": "KEEP", "kernel_id": "k2"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    entries = data.local_decision_audit["recent_integrate"]
    # Only the valid one is included.
    kernels = {e["kernel_id"] for e in entries}
    assert kernels == {"k2"}
