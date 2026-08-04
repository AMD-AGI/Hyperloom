# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


_ASSETS = Path(__file__).parents[1] / "assets"
_BENCH_PATH = _ASSETS / "benchmark_scripts" / "worldmirror_bench.py"
_HEAVY_SCENES = {
    "Building",
    "Landmark",
    "Messy_Room",
    "Park_Stone",
    "Small_Room",
    "Statue_Face",
    "Tree_Building",
}


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


def test_discover_scenes_prefers_exact_name_over_substring(tmp_path):
    """A substring hit must not let "Building" time "Tree_Building" instead."""
    bench = _load_bench()
    repo = tmp_path / "HY-World-2.0"
    root = repo / "examples" / "worldrecon" / "realistic"
    for name in ("Building", "Tree_Building"):
        (root / name).mkdir(parents=True)
        (root / name / "a.jpg").write_bytes(b"fake")

    assert bench.discover_scenes(repo, "Building") == [root / "Building"]
    assert bench.discover_scenes(repo, "Tree_Building") == [root / "Tree_Building"]


def test_discover_scenes_rejects_unresolvable_token(tmp_path):
    """The pipeline swallows a bad path as a skip, which would silently shrink
    the timed set and downgrade the metric scope to e2e."""
    bench = _load_bench()
    repo = tmp_path / "HY-World-2.0"
    (repo / "examples" / "worldrecon" / "realistic" / "Building").mkdir(parents=True)

    with pytest.raises(SystemExit):
        bench.discover_scenes(repo, "Typo_Scene")


def test_warmup_covers_every_timed_scene(tmp_path):
    """An unmet shape costs 3-18s of kernel compile inside the timed forward, so
    partial coverage inflated iteration 0 to 82.5s against a 9.2s steady state."""
    bench = _load_bench()
    scenes = [tmp_path / f"scene_{i}" for i in range(7)]

    plan = bench.warmup_plan(scenes, 3)

    assert set(plan) == set(scenes), "every timed scene must be warmed"
    assert len(plan) == 7, "the requested count is a floor, not the total"


def test_warmup_honours_a_larger_operator_request(tmp_path):
    bench = _load_bench()
    scenes = [tmp_path / f"scene_{i}" for i in range(3)]

    plan = bench.warmup_plan(scenes, 8)

    assert len(plan) == 8
    assert set(plan) == set(scenes)
    assert bench.warmup_plan([], 5) == []


def test_save_artifacts_defaults_to_minimal(monkeypatch):
    """gs/points/normal writing costs 38% wall-clock and leaks ~75MB per
    reconstruction while leaving the forward unchanged."""
    bench = _load_bench()
    monkeypatch.delenv("WM_SAVE_ARTIFACTS", raising=False)
    assert bench.save_artifact_mode() == "minimal"
    monkeypatch.setenv("WM_SAVE_ARTIFACTS", "  FULL ")
    assert bench.save_artifact_mode() == "full"
    monkeypatch.setenv("WM_SAVE_ARTIFACTS", "nonsense")
    assert bench.save_artifact_mode() == "minimal"

    minimal = bench.save_flags("minimal")
    assert minimal == {
        "save_depth": True,
        "save_camera": True,
        "save_normal": False,
        "save_gs": False,
        "save_points": False,
    }
    assert all(bench.save_flags("full").values())
    assert set(bench.save_flags("full")) == set(minimal)


def test_summarize_latencies_reports_mean_and_spread():
    bench = _load_bench()
    stats = bench.summarize_latencies([100.0, 110.0, 90.0, 100.0])

    assert stats["count"] == 4
    assert stats["mean_ms"] == 100.0
    assert stats["stdev_ms"] == pytest.approx(8.164966, rel=1e-6)
    assert stats["cv_pct"] == pytest.approx(8.164966, abs=1e-3)
    assert bench.summarize_latencies([])["count"] == 0
    assert bench.summarize_latencies([5.0])["stdev_ms"] == 0.0


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


def test_bench_driver_warms_the_plan_and_shares_save_flags():
    """Warmup must cover the configured set and run the same save path as the
    timed loop, or it warms shapes the measurement never uses."""
    src = _BENCH_PATH.read_text(encoding="utf-8")
    assert "warmup_scenes = warmup_plan(scenes, args.warmup)" in src
    assert "for i in range(max(args.warmup, 0)):" not in src
    assert src.count("**save_kwargs,") == 3, "warmup, timed and profiled calls"


@pytest.mark.parametrize("name", ["baseline_worldmirror.yaml", "profile_worldmirror.yaml"])
def test_shipped_configs_time_only_the_resolvable_scenes(name):
    """Averaging all 22 scenes buries a 10% win: the light ones run at a 13.3%
    median CV against 0.2-3.8% for the seven 32-image scenes."""
    envs = yaml.safe_load((_ASSETS / "configs" / name).read_text(encoding="utf-8"))["benchmark"]["envs"]

    assert set(envs["WM_SCENES"].split()) == _HEAVY_SCENES
    assert envs["WM_SAVE_ARTIFACTS"] == "minimal"
