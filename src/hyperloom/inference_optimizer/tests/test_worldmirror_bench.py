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
