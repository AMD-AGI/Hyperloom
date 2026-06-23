# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the PerfSkills/GEAK-e2e breakdown collector and the
sweep ``benchmark_report.json`` writer.

These exercise the ``KERNEL_OPT_BACKEND_ORDER=perfskills`` paths in isolation:

* :func:`collect_perfskills` — the not-engaged short-circuit, the
  engaged-but-missing-result fallback, the full success mapping (including the
  ``accepted_kernels`` shape guard and the path-relativization warning branch).
* :func:`_write_benchmark_report` — the happy path plus the best-effort
  ``OSError`` branch (a failed write must never raise).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import collectors
from inference_optimizer.breakdown.collectors import collect_perfskills
from inference_optimizer.orchestrator.action_executors._perfskills_sweep import (
    _pareto_front,
    _parse_isl_osl,
    _read_json,
    _serving_gpus,
    _write_benchmark_report,
    sweep_via_perfskills,
)


def test_collect_perfskills_not_engaged_returns_empty(tmp_path: Path) -> None:
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, {"kernel_optimizer": "native"}, warnings)
    assert out == {}
    assert warnings == []


def test_collect_perfskills_native_with_empty_result_default(tmp_path: Path) -> None:
    # Regression: SharedState defaults perfskills_result to ``{}``. An empty dict
    # must NOT be treated as engaged, or every native session emits a spurious
    # perfskills section.
    state = {"kernel_optimizer": "native", "perfskills_result": {}}
    out = collect_perfskills(tmp_path, state, [])
    assert out == {}


def test_collect_perfskills_empty_result_no_flag(tmp_path: Path) -> None:
    # Empty result + no/blank optimizer flag → not engaged.
    out = collect_perfskills(tmp_path, {"perfskills_result": {}}, [])
    assert out == {}


def test_collect_perfskills_engaged_without_result(tmp_path: Path) -> None:
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, {"kernel_optimizer": "perfskills"}, warnings)
    assert out["engaged"] is True
    assert out["status"] == "missing"
    assert out["error_class"] == "no_result"
    assert out["accepted_kernels"] == []


def test_collect_perfskills_full_success_maps_fields(tmp_path: Path) -> None:
    eval_dir = tmp_path / "runs" / "perfskills" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "baseline_throughput_tok_s": 100.0,
            "final_throughput_tok_s": 300.0,
            "throughput_speedup": 3.0,
            "ttft_ms": 12.5,
            "tpot_ms": 4.5,
            "output_parity": "pass",
            "metric_basis": "aggregate_output_tok_s",
            "bench_client": "inferencex",
            "accepted_kernels": ["rmsnorm", "moe_gemm"],
            "accepted_heads": ["mla"],
            "accepted_config": {"tp": 8, "conc": 64},
            "validated_regimes": [{"isl": 8192, "osl": 1024, "conc": 64}],
            "eval_dir": str(eval_dir),
            "report_path": str(eval_dir / "final_report.md"),
            "returncode": 0,
        },
    }
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, state, warnings)

    assert out["engaged"] is True
    assert out["status"] == "ok"
    assert out["error_class"] is None
    assert out["baseline_throughput_tok_s"] == 100.0
    assert out["final_throughput_tok_s"] == 300.0
    assert out["throughput_speedup"] == 3.0
    assert out["gain_pct"] == 200.0
    assert out["kernels_optimized"] == 2
    assert out["accepted_config"] == {"tp": 8, "conc": 64}
    assert out["validated_regimes"] == [{"isl": 8192, "osl": 1024, "conc": 64}]
    # eval_dir lives under the session dir, so it is relativized.
    assert out["eval_dir"] == "runs/perfskills/eval"


def _write_kernel_journey(eval_dir: Path) -> Path:
    """Write a minimal kernel_journey.json with one integrated KEEP kernel."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    kj = {
        "schema_version": 1,
        "kernels": [
            {
                "kernel_id": "fused_moe_kernel_gptq_awq",
                "name": "fused_moe_kernel_gptq_awq",
                "gpu_pct": 50.64,
                "micro_speedup": 1.5902,
                "dispatch": {"dispatched": True, "backends": ["geak"]},
                "backend_result": {
                    "verification": {"micro_speedup": 1.5902, "best_backend": "geak"},
                },
                "e2e": {
                    "kernel_id": "fused_moe_kernel_gptq_awq",
                    "integrated": True,
                    "e2e_gain_pct": 16.049,
                    "validated": True,
                    "decision": "KEEP",
                    "target_file": "config/integrate_moe_tuned/E=384.json",
                    "extra_server_args": "--max-num-batched-tokens 16384",
                },
            },
            {
                # Cut-off kernel: dispatched but no e2e → must NOT be accepted.
                "kernel_id": "fwd_grouped_kernel_stage1",
                "name": "fwd_grouped_kernel_stage1",
                "gpu_pct": 15.55,
                "dispatch": {"dispatched": True, "backends": ["geak"]},
                "backend_result": {"attempts": [], "verification": {}},
            },
        ],
    }
    path = eval_dir / "kernel_journey.json"
    path.write_text(json.dumps(kj), encoding="utf-8")
    return path


def test_collect_perfskills_backfills_accepted_kernels_from_journey(tmp_path: Path) -> None:
    # result.json shipped the aggregate win but an empty accepted_kernels (a
    # recovered/intermediate flush). The sibling kernel_journey.json holds the
    # integrated KEEP kernel, so the collector must back-fill it.
    eval_dir = tmp_path / "perfskills" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "baseline_throughput_tok_s": 461.314,
            "final_throughput_tok_s": 535.352,
            "throughput_speedup": 1.1605,
            "accepted_kernels": [],
            "recovered_from_disk": True,
            "eval_dir": str(eval_dir),
        },
    }
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, state, warnings)

    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"
    only = out["accepted_kernels"][0]
    assert only["kernel_id"] == "fused_moe_kernel_gptq_awq"
    assert only["decision"] == "KEEP"
    # The GEAK-e2e GEAK is relabeled to the distinct geak_v4 variant.
    assert only["backend"] == "geak_v4"
    assert only["e2e_gain_pct"] == 16.049
    assert only["micro_speedup"] == 1.5902
    assert only["source"] == "kernel_journey_backfill"


def test_collect_perfskills_backfill_via_explicit_journey_path(tmp_path: Path) -> None:
    # kernel_journey_path takes precedence over deriving it from eval_dir.
    eval_dir = tmp_path / "perfskills" / "eval"
    kj_path = _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": [],
            "kernel_journey_path": str(kj_path),
        },
    }
    out = collect_perfskills(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"


def test_collect_perfskills_does_not_overwrite_populated_kernels(tmp_path: Path) -> None:
    # A producer-populated accepted_kernels list must be preserved verbatim and
    # marked as sourced from the result, never replaced by the journey.
    eval_dir = tmp_path / "perfskills" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": ["rmsnorm"],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_perfskills(tmp_path, state, [])
    assert out["accepted_kernels"] == ["rmsnorm"]
    assert out["accepted_kernels_source"] == "result"
    assert out["kernels_optimized"] == 1


def test_collect_perfskills_no_backfill_on_failure(tmp_path: Path) -> None:
    # A non-ok run must not back-fill (the e2e never landed a real win).
    eval_dir = tmp_path / "perfskills" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "error",
            "error_class": "timeout",
            "accepted_kernels": [],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_perfskills(tmp_path, state, [])
    assert out["accepted_kernels"] == []
    assert out["accepted_kernels_source"] is None
    assert out["kernels_optimized"] == 0


def test_collect_perfskills_backfill_missing_journey_is_noop(tmp_path: Path) -> None:
    # ok run but no kernel_journey.json on disk → empty list, no crash.
    eval_dir = tmp_path / "perfskills" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": [],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_perfskills(tmp_path, state, [])
    assert out["accepted_kernels"] == []
    assert out["accepted_kernels_source"] is None


def test_collect_perfskills_speedup_only_gain_and_bad_kernels(tmp_path: Path) -> None:
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {
            "status": "ok",
            "throughput_speedup": 1.5,
            # Non-list accepted_kernels must be coerced + warned.
            "accepted_kernels": "not-a-list",
        },
    }
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, state, warnings)
    # No baseline/final, so gain derives from the speedup.
    assert out["gain_pct"] == 50.0
    assert out["accepted_kernels"] == []
    assert any("accepted_kernels" in w for w in warnings)


def test_collect_perfskills_relativize_failure_warns(tmp_path: Path, monkeypatch) -> None:
    # Force the relativization helper to raise so the best-effort except branch
    # records a warning instead of swallowing it silently.
    def _boom(_path, _session_dir):
        raise OSError("relativize boom")

    monkeypatch.setattr(collectors, "_rel", _boom)

    eval_dir = tmp_path / "runs" / "eval"
    state = {
        "kernel_optimizer": "perfskills",
        "perfskills_result": {"status": "ok", "eval_dir": str(eval_dir)},
    }
    warnings: list[str] = []
    out = collect_perfskills(tmp_path, state, warnings)
    # Falls back to the absolute path and records the reason.
    assert out["eval_dir"] == str(eval_dir)
    assert any("failed to relativize path" in w for w in warnings)


def test_write_benchmark_report_happy_path(tmp_path: Path) -> None:
    _write_benchmark_report(
        tmp_path,
        conc=64,
        isl=8192,
        osl=1024,
        success=True,
        output_throughput_tok_s=300.0,
        mean_ttft_ms=12.5,
        mean_tpot_ms=4.5,
        mean_e2el_ms=900.0,
    )
    report = json.loads((tmp_path / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["output_throughput_tok_s"] == 300.0
    assert report["source"] == "perfskills"
    assert "error" not in report


def test_write_benchmark_report_oserror_is_swallowed(tmp_path: Path) -> None:
    # A file (not a directory) as out_dir makes the nested write raise
    # NotADirectoryError (an OSError); the best-effort writer must not raise.
    not_a_dir = tmp_path / "afile"
    not_a_dir.write_text("x", encoding="utf-8")
    _write_benchmark_report(
        not_a_dir,
        conc=64,
        isl=8192,
        osl=1024,
        success=False,
        output_throughput_tok_s=None,
        mean_ttft_ms=None,
        mean_tpot_ms=None,
        mean_e2el_ms=None,
        error="boom",
    )
    # No exception == pass; and nothing was written under the file path.
    assert not (not_a_dir / "benchmark_report.json").exists()


def test_serving_gpus() -> None:
    assert _serving_gpus(1) == "0"
    assert _serving_gpus(4) == "0,1,2,3"
    # Degenerate tp falls back to a single device.
    assert _serving_gpus(0) == "0"


def test_parse_isl_osl() -> None:
    assert _parse_isl_osl("8192:1024") == (8192, 1024)
    # Missing osl side defaults to 1024.
    assert _parse_isl_osl("2048") == (2048, 1024)
    # Empty defaults both sides.
    assert _parse_isl_osl("") == (1024, 1024)


def test_read_json_roundtrip_and_missing(tmp_path: Path) -> None:
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert _read_json(good) == {"a": 1}
    # Missing / unparseable files degrade to an empty dict (never raise).
    assert _read_json(tmp_path / "nope.json") == {}


def test_schema_has_perfskills_contract() -> None:
    # The exporter emits a top-level ``perfskills`` section, so the wire schema
    # must declare it (contract + TypedDict), else readers have no shape to rely on.
    from inference_optimizer.breakdown import schema

    assert hasattr(schema, "Perfskills")
    assert "perfskills" in schema.SessionBreakdown.__annotations__
    # ``from __future__ import annotations`` stores the type as a string ref.
    assert "Perfskills" in str(schema.SessionBreakdown.__annotations__["perfskills"])
    # A few representative fields must be part of the declared shape.
    for field in ("engaged", "status", "error_class", "throughput_speedup", "accepted_kernels"):
        assert field in schema.Perfskills.__annotations__


def test_pareto_front_drops_dominated_points() -> None:
    entries = [
        {"status": "succeeded", "output_throughput": 100.0, "ttft_mean_ms": 10.0},
        # Dominated: lower throughput AND higher latency than the first.
        {"status": "succeeded", "output_throughput": 80.0, "ttft_mean_ms": 20.0},
        # On the front: higher throughput at the cost of higher latency.
        {"status": "succeeded", "output_throughput": 150.0, "ttft_mean_ms": 25.0},
        # Failed points are ignored entirely.
        {"status": "failed", "output_throughput": 999.0, "ttft_mean_ms": 1.0},
    ]
    front = _pareto_front(entries)
    tputs = sorted(e["output_throughput"] for e in front)
    assert tputs == [100.0, 150.0]


@pytest.mark.asyncio
async def test_sweep_via_perfskills_reuses_script_and_records_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = tmp_path / "bench_e2e.sh"
    bench.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
if os.environ["CONC"] == "2":
    raise SystemExit(3)
summary = {
    "output_throughput_tok_s_median": 321.0,
    "ttft_ms_median": 12.0,
    "tpot_ms_median": 4.0,
    "e2el_ms_median": 99.0,
}
(out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
env_keys = [
    "BACKEND", "MODEL", "TP", "GPU", "ISL", "OSL", "CONC", "REPEATS",
    "OVERLAY_PYTHONPATH", "EXTRA_SERVER_ARGS", "EXTRA_ENV", "BENCH_CLIENT",
    "RANDOM_RANGE_RATIO", "NUM_WARMUPS", "SEED", "BENCH_TRUST_REMOTE_CODE",
]
(out / "env.json").write_text(
    json.dumps({k: os.environ.get(k) for k in env_keys}, sort_keys=True),
    encoding="utf-8",
)
PY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PATH", "/models/kimi")
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("TP", "2")

    result = await sweep_via_perfskills(
        result={
            "status": "ok",
            "bench_script": str(bench),
            "final_overlay": "/tmp/overlay",
            "bench_client": "inferencex",
            "accepted_config": {
                "flags": "--trust-remote-code --max-num-batched-tokens 16384",
                "env": "OPT_MOE_CONFIG=/tmp/moe.json",
            },
            "bench_protocol": {
                "random_range_ratio": 0.5,
                "num_warmups": 2,
                "seed": 1234,
            },
        },
        conc_values=[1, 2],
        isl_osl_configs=["8192:1024"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=2,
    )

    assert result["status"] == "succeeded"
    assert result["grid_size"] == 2
    assert result["source"] == "perfskills"
    assert result["best_for_each_conc"]["1"]["output_throughput"] == 321.0
    assert result["sweep_grid"][1]["status"] == "failed"
    assert result["sweep_grid"][1]["error"] == "no throughput"

    ok_dir = tmp_path / "sweep" / "variant_0_conc1_isl8192_osl1024"
    env = json.loads((ok_dir / "env.json").read_text(encoding="utf-8"))
    assert env["BACKEND"] == "vllm"
    assert env["MODEL"] == "/models/kimi"
    assert env["TP"] == "2"
    assert env["GPU"] == "0,1"
    assert env["REPEATS"] == "2"
    assert env["BENCH_CLIENT"] == "inferencex"
    assert env["RANDOM_RANGE_RATIO"] == "0.5"
    assert env["NUM_WARMUPS"] == "2"
    assert env["SEED"] == "1234"
    assert env["BENCH_TRUST_REMOTE_CODE"] == "1"
    assert env["OVERLAY_PYTHONPATH"] == "/tmp/overlay"
    assert env["EXTRA_ENV"] == "OPT_MOE_CONFIG=/tmp/moe.json"

    ok_report = json.loads((ok_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert ok_report["success"] is True
    assert ok_report["output_throughput_tok_s"] == 321.0

    fail_report = json.loads(
        (tmp_path / "sweep" / "variant_1_conc2_isl8192_osl1024" / "benchmark_report.json")
        .read_text(encoding="utf-8")
    )
    assert fail_report["success"] is False
    assert fail_report["source"] == "perfskills"


@pytest.mark.asyncio
async def test_sweep_via_perfskills_requires_existing_bench_script(tmp_path: Path) -> None:
    result = await sweep_via_perfskills(
        result={"status": "ok", "bench_script": str(tmp_path / "missing.sh")},
        conc_values=[1],
        isl_osl_configs=["8:8"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=1,
    )

    assert result["status"] == "failed"
    assert result["error_class"] == "missing_bench_script"
