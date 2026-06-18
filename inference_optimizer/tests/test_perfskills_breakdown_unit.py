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

from inference_optimizer.breakdown import collectors
from inference_optimizer.breakdown.collectors import collect_perfskills
from inference_optimizer.orchestrator.action_executors._perfskills_sweep import (
    _pareto_front,
    _parse_isl_osl,
    _read_json,
    _serving_gpus,
    _write_benchmark_report,
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
