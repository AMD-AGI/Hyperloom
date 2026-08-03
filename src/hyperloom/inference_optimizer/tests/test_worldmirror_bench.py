# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_BENCH_PATH = (
    Path(__file__).parents[1]
    / "assets"
    / "benchmark_scripts"
    / "worldmirror_bench.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location("worldmirror_bench", _BENCH_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worldmirror_bench_discovers_example_scene_dirs(tmp_path):
    bench = _load_bench()
    repo = tmp_path / "HY-World-2.0"
    office = repo / "examples" / "worldrecon" / "realistic" / "Office"
    office.mkdir(parents=True)
    (office / "Office.jpg").write_bytes(b"fake")

    scenes = bench.discover_scenes(repo, "")

    assert scenes == [office]


def test_worldmirror_depth_quality_ref_write_and_compare(tmp_path):
    bench = _load_bench()
    output_dir = tmp_path / "out"
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True)
    np.save(depth_dir / "depth_0000.npy", np.ones((2, 2), dtype=np.float32))
    ref = tmp_path / "baseline.ref"

    established = bench.evaluate_quality_gate(output_dir, "", str(ref), rel_max=0.2)
    compared = bench.evaluate_quality_gate(output_dir, str(ref), "", rel_max=0.2)

    assert established["skipped"] is True
    assert established["reason"] == "reference_established"
    assert compared["passed"] is True
    assert compared["heads"]["depth"]["rel_l1"] == 0.0


def test_metric_scope_defaults_to_inference(monkeypatch):
    """The forward is ~5% of a reconstruction; e2e would bury a real GPU win."""
    bench = _load_bench()
    monkeypatch.delenv("WM_METRIC_SCOPE", raising=False)
    assert bench.metric_scope() == "inference"
    monkeypatch.setenv("WM_METRIC_SCOPE", "  E2E ")
    assert bench.metric_scope() == "e2e"
    monkeypatch.setenv("WM_METRIC_SCOPE", "nonsense")
    assert bench.metric_scope() == "inference"


def test_read_stage_timings_parses_pipeline_output(tmp_path):
    bench = _load_bench()
    (tmp_path / "pipeline_timing.json").write_text(
        '{"inference": 0.11022, "compute_mask": 1.187, "case_total": 2.036, "note": "x"}',
        encoding="utf-8",
    )
    stages = bench.read_stage_timings(tmp_path)
    assert stages["inference"] == 0.11022
    assert stages["case_total"] == 2.036
    assert "note" not in stages, "non-numeric entries must be dropped"


def test_read_stage_timings_missing_or_corrupt_is_empty(tmp_path):
    bench = _load_bench()
    assert bench.read_stage_timings(tmp_path) == {}
    (tmp_path / "pipeline_timing.json").write_text("not json", encoding="utf-8")
    assert bench.read_stage_timings(tmp_path) == {}


def test_bench_driver_always_enables_log_time():
    """The reported latency is derived from the per-stage file, which the
    pipeline only writes when log_time is on."""
    src = _BENCH_PATH.read_text(encoding="utf-8")
    assert "log_time=bool(args.profile_dir)" not in src
    assert "log_time=True" in src


def test_bench_driver_captures_a_profiler_trace():
    """Without an exported trace, roofline and the kernel agent have no input."""
    src = _BENCH_PATH.read_text(encoding="utf-8")
    assert "from torch.profiler import ProfilerActivity, profile" in src
    assert "export_chrome_trace" in src
    # Profiling must not inflate the measurement it is meant to explain, so the
    # call site has to sit after the timed loop closes.
    call_site = src.index("_capture_profile_trace(pipeline, scenes[0]")
    assert call_site > src.index("duration = time.perf_counter() - start")
