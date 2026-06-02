"""Unit tests for ``scripts/roofline_sweep.py``.

Real ``sglang_server`` / ``run_bench`` are mocked so the test suite stays
GPU-free. We exercise: template extraction guard, csv round-trip,
ceiling computation parity with ``roofline_ceiling``, sweep dispatch
with a fake server context manager, and the svg smoke path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import pytest

from inference_optimizer.scripts import roofline_sweep as rs


# ---------------------------------------------------------------------------
# extract_templates: both empty -> raise; optimized populated -> two templates
# ---------------------------------------------------------------------------
def test_extract_templates_refuses_when_current_best_empty() -> None:
    state = {"current_best": {"extra_sglang_args": "", "extra_envs": {}}}
    with pytest.raises(SystemExit, match="did not accept any optimization"):
        rs.extract_templates(state)


def test_extract_templates_uses_current_best() -> None:
    state = {
        "current_best": {
            "extra_sglang_args": "--enable-aiter-fmoe-permute-fusion",
            "extra_envs": {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"},
        }
    }
    base, opt = rs.extract_templates(state)
    assert base.label == "baseline" and base.extra_sglang_args == ""
    assert opt.label == "optimized"
    assert "permute-fusion" in opt.extra_sglang_args
    assert opt.extra_envs["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"


def test_extract_templates_optimized_via_envs_only() -> None:
    state = {
        "current_best": {"extra_sglang_args": "", "extra_envs": {"FOO": "1"}}
    }
    _, opt = rs.extract_templates(state)
    assert opt.extra_sglang_args == "" and opt.extra_envs == {"FOO": "1"}


# ---------------------------------------------------------------------------
# compute_ceiling: parity with the underlying roofline_ceiling helper
# ---------------------------------------------------------------------------
def test_compute_ceiling_matches_underlying_formula() -> None:
    from inference_optimizer.orchestrator.roofline_ceiling import (
        ModelMeta, compute_theoretical_peak_output_tok_per_sec,
    )
    meta = ModelMeta(
        weight_bytes=60_000_000_000, num_layers=48, num_kv_heads=4,
        head_dim=128, weight_dtype_bytes=2.0,
        active_weight_bytes=8_000_000_000,
    )
    got = rs.compute_ceiling(
        model_meta=meta, gpu_type="mi355x", num_gpus=1,
        conc=32, isl=1024, osl=1024,
    )
    want = compute_theoretical_peak_output_tok_per_sec(
        gpu_type="mi355x", num_gpus=1, weight_bytes=meta.weight_bytes,
        active_weight_bytes=meta.active_weight_bytes,
        num_layers=meta.num_layers, num_kv_heads=meta.num_kv_heads,
        head_dim=meta.head_dim, kv_dtype_bytes=meta.weight_dtype_bytes,
        isl=1024, osl=1024, concurrency=32,
    )
    assert got == pytest.approx(want)
    # Higher concurrency raises the ceiling (weight reuse).
    higher = rs.compute_ceiling(
        model_meta=meta, gpu_type="mi355x", num_gpus=1,
        conc=128, isl=1024, osl=1024,
    )
    assert higher > got


# ---------------------------------------------------------------------------
# write_csv: round-trip
# ---------------------------------------------------------------------------
def test_write_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        {"conc": 1, "config": "baseline", "measured_tps": 100.0,
         "ceiling_tps": 200.0, "target_70_tps": 140.0, "mbu_pct": 50.0,
         "status": "OK"},
        {"conc": 8, "config": "optimized", "measured_tps": 900.0,
         "ceiling_tps": 1200.0, "target_70_tps": 840.0, "mbu_pct": 75.0,
         "status": "OK"},
    ]
    p = tmp_path / "x.csv"
    rs.write_csv(rows, p)
    with p.open(encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    assert len(back) == 2
    assert back[0]["config"] == "baseline" and back[0]["status"] == "OK"
    assert float(back[1]["measured_tps"]) == 900.0


# ---------------------------------------------------------------------------
# sweep_one_template: mock SglangServer + run_bench
# ---------------------------------------------------------------------------
class _FakeServer:
    """In-memory replacement for the real SglangServer context manager."""

    def __init__(self, *args, **kwargs) -> None:
        self.label = kwargs.get("extra_args", "") or "baseline"

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    @property
    def base_url(self) -> str:
        return "http://fake:0"


def test_sweep_one_template_dispatches_per_conc(tmp_path: Path) -> None:
    from inference_optimizer.orchestrator.roofline_ceiling import ModelMeta
    meta = ModelMeta(
        weight_bytes=60_000_000_000, num_layers=48, num_kv_heads=4,
        head_dim=128, weight_dtype_bytes=2.0,
        active_weight_bytes=8_000_000_000,
    )
    tmpl = rs.LaunchTemplate(label="optimized", extra_sglang_args="--foo")
    measured_by_conc = {1: 500.0, 8: 4000.0, 64: 18000.0}

    def fake_bench(*, conc: int, **_kwargs):
        return {"output_throughput": measured_by_conc[conc]}

    with mock.patch.object(rs, "SglangServer", _FakeServer), \
         mock.patch.object(rs, "run_bench", side_effect=fake_bench):
        rows = rs.sweep_one_template(
            tmpl=tmpl, model_path="/tmp/nonexistent_model", model_meta=meta,
            gpu_type="mi355x", tp=1, gpu_id=0, port=30100,
            isl=1024, osl=1024, concs=[1, 8, 64], dataset="random",
            num_prompts_factor=4, output_dir=tmp_path,
        )
    assert [r["conc"] for r in rows] == [1, 8, 64]
    assert all(r["config"] == "optimized" for r in rows)
    assert rows[1]["measured_tps"] == 4000.0
    assert rows[1]["mbu_pct"] == pytest.approx(
        4000.0 / rows[1]["ceiling_tps"] * 100.0
    )
    assert all(r["status"] == "OK" for r in rows)


def test_sweep_one_template_marks_failed_bench_as_oom(tmp_path: Path) -> None:
    from inference_optimizer.orchestrator.roofline_ceiling import ModelMeta
    meta = ModelMeta(
        weight_bytes=60_000_000_000, num_layers=48, num_kv_heads=4,
        head_dim=128, weight_dtype_bytes=2.0,
        active_weight_bytes=8_000_000_000,
    )
    tmpl = rs.LaunchTemplate(label="baseline", extra_sglang_args="")
    with mock.patch.object(rs, "SglangServer", _FakeServer), \
         mock.patch.object(rs, "run_bench", return_value={}):
        rows = rs.sweep_one_template(
            tmpl=tmpl, model_path="/tmp/nonexistent_model", model_meta=meta,
            gpu_type="mi355x", tp=1, gpu_id=0, port=30100,
            isl=1024, osl=1024, concs=[128], dataset="random",
            num_prompts_factor=4, output_dir=tmp_path,
        )
    assert rows[0]["status"] == "FAILED_OR_OOM"
    assert rows[0]["measured_tps"] is None
    # Ceiling still computed (independent of bench).
    assert rows[0]["ceiling_tps"] > 0


# ---------------------------------------------------------------------------
# plot_svg smoke test
# ---------------------------------------------------------------------------
def test_plot_svg_writes_svg(tmp_path: Path) -> None:
    rows = []
    for label in ("baseline", "optimized"):
        for c in (1, 8, 64):
            rows.append({
                "conc": c, "config": label,
                "measured_tps": 1000.0 * c * (1.0 if label == "baseline" else 1.3),
                "ceiling_tps": 2000.0 * c,
                "target_70_tps": 1400.0 * c, "mbu_pct": 50.0,
                "status": "OK",
            })
    svg_path = tmp_path / "out.svg"
    rs.plot_svg(
        rows, svg_path, model_label="TestModel", gpu_label="MI355X",
        tp=1, isl=1024, osl=1024,
    )
    assert svg_path.is_file()
    content = svg_path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "TestModel" in content
    assert "MI355X" in content


# ---------------------------------------------------------------------------
# main --skip-bench end-to-end: csv → svg without launching anything
# ---------------------------------------------------------------------------
def test_main_skip_bench(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Minimal HF-style metadata so load_model_meta returns a real ModelMeta.
    (model_dir / "config.json").write_text(json.dumps({
        "num_hidden_layers": 4, "num_attention_heads": 8,
        "num_key_value_heads": 2, "hidden_size": 64, "head_dim": 8,
        "torch_dtype": "bfloat16",
    }))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1_000_000_000},
    }))
    (session_dir / "state.json").write_text(json.dumps({
        "model_path": str(model_dir),
        "model_name": "TestModel",
        "precision": "bf16",
        "current_best": {"extra_sglang_args": "--foo", "extra_envs": {}},
    }))
    csv_path = session_dir / "roofline_sweep.csv"
    csv_path.write_text(
        "conc,config,measured_tps,ceiling_tps,target_70_tps,mbu_pct,status\n"
        "1,baseline,100,200,140,50,OK\n"
        "1,optimized,130,200,140,65,OK\n"
    )
    rc = rs.main([
        "--session-dir", str(session_dir),
        "--concs", "1",
        "--isl", "1024", "--osl", "1024",
        "--tp", "1", "--gpu-type", "mi355x",
        "--skip-bench",
    ])
    assert rc == 0
    assert (session_dir / "roofline_sweep.svg").is_file()
