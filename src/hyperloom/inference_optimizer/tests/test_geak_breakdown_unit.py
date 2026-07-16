# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the GEAK e2e breakdown collector and the
sweep ``benchmark_report.json`` writer.

These exercise the ``KERNEL_OPT_BACKEND_ORDER=geak`` paths in isolation:

* :func:`collect_geak` — the not-engaged short-circuit, the
  engaged-but-missing-result fallback, the full success mapping (including the
  ``accepted_kernels`` shape guard and the path-relativization warning branch).
* :func:`_write_benchmark_report` — the happy path plus the best-effort
  ``OSError`` branch (a failed write must never raise).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.inference_optimizer.breakdown.collectors import collect_geak
from hyperloom.orchestrator.actions.executors._geak_sweep import (
    _pareto_front,
    _parse_isl_osl,
    _serving_gpus,
    _write_benchmark_report,
    sweep_via_geak,
)


def test_collect_geak_not_engaged_returns_empty(tmp_path: Path) -> None:
    warnings: list[str] = []
    out = collect_geak(tmp_path, {"kernel_optimizer": "native"}, warnings)
    assert out == {}
    assert warnings == []


def test_collect_geak_native_with_empty_result_default(tmp_path: Path) -> None:
    # An empty geak_result dict must NOT be treated as engaged.
    state = {"kernel_optimizer": "native", "geak_result": {}}
    out = collect_geak(tmp_path, state, [])
    assert out == {}


def test_collect_geak_empty_result_no_flag(tmp_path: Path) -> None:
    # Empty result + no/blank optimizer flag → not engaged.
    out = collect_geak(tmp_path, {"geak_result": {}}, [])
    assert out == {}


def test_collect_geak_engaged_without_result(tmp_path: Path) -> None:
    # No on-disk geak/ tree → nothing to reconstruct → legacy ``missing``.
    warnings: list[str] = []
    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, warnings)
    assert out["engaged"] is True
    assert out["status"] == "missing"
    assert out["error_class"] == "no_result"
    assert out["accepted_kernels"] == []
    assert "recovered_from_disk" not in out


def test_collect_geak_reconstructs_from_disk_when_result_missing(
    tmp_path: Path,
) -> None:
    # geak engaged with an on-disk tree but no ``geak_result`` in state: the
    # collector reconstructs the run from disk instead of a bare miss.
    pf = tmp_path / "geak"
    pf.mkdir()
    (pf / "handoff.json").write_text(
        json.dumps(
            {
                "model_path": "/path/models/Qwen-Qwen3-0.6B",
                "framework": "vllm",
                "gpu_type": "mi300x",
                "tp": 1,
                "workload": {"isl": 1024, "osl": 1024, "conc": 64},
                "accepted_flags": "--kv-cache-dtype fp8",
                "raw_baseline_tput": 10434.27,
            }
        ),
        encoding="utf-8",
    )
    exp = pf / "e2e_Qwen-Qwen3-0.6B_20260629T174250Z"
    (exp / "baseline").mkdir(parents=True)
    (exp / "kernels" / "paged_attention_task").mkdir(parents=True)
    (exp / "kernels" / "_exp").mkdir(parents=True)
    (exp / "strategy.md").write_text("# strategy", encoding="utf-8")
    (exp / "kernel_journey.json").write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "kernel_id": "k002",
                        "name": "rocm_unquantized_gemm",
                        "gpu_pct": 17.0,
                        "e2e": {
                            "decision": "KEEP",
                            "integrated": True,
                            "e2e_gain_pct": 2.2,
                            "validated": True,
                            "target_file": "gemm.cu",
                            "extra_server_args": "",
                        },
                        "backend_result": {"verification": {"best_backend": "geak"}},
                        "dispatch": {"backends": ["geak"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    warnings: list[str] = []
    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, warnings)

    assert out["engaged"] is True
    assert out["status"] == "no_result_recovered_from_disk"
    assert out["error_class"] == "no_result"
    assert out["recovered_from_disk"] is True
    # Handoff evidence proves HL handed off (vs a silent miss).
    assert out["handoff"]["framework"] == "vllm"
    assert out["handoff"]["workload"] == {"isl": 1024, "osl": 1024, "conc": 64}
    # exp_root is relativized under the session dir.
    assert out["exp_root"] == "geak/e2e_Qwen-Qwen3-0.6B_20260629T174250Z"
    # Stages the runner reached are surfaced for forensics.
    for stage in ("handoff", "baseline", "kernels", "opbench", "strategy",
                  "kernel_journey"):
        assert stage in out["stages_reached"]
    assert out["kernels_attempted"] == [{"name": "paged_attention_task"}]
    # Per-kernel attribution is backfilled from the surviving journey.
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"
    assert [k["kernel_id"] for k in out["accepted_kernels"]] == ["k002"]
    assert out["accepted_kernels"][0]["backend"] == "geak"
    assert out["kernels_optimized"] == 1
    assert out["last_artifact_ts"] is not None


def test_collect_geak_reconstruct_surfaces_opbench_logs_and_cause(
    tmp_path: Path,
) -> None:
    # e2e ran the op-bench bake-off (no editable winner) then was killed before
    # flushing a result/journey: reconstruction surfaces the op-bench verdicts,
    # runner log tails, and a classified cause.
    pf = tmp_path / "geak"
    exp = pf / "e2e_run"
    (exp / "baseline").mkdir(parents=True)
    task = exp / "kernels" / "rocm_unquantized_gemm_decode_family_task"
    task.mkdir(parents=True)
    (exp / "kernels" / "_exp").mkdir(parents=True)
    (pf / "handoff.json").write_text(json.dumps({"framework": "vllm"}), encoding="utf-8")
    (task / "opbench_result.json").write_text(
        json.dumps(
            {
                "task": "rocm_unquantized_gemm_decode_family_task",
                "winner_backend": "hipblaslt",
                "isolated_speedup": 1.0,
                "winner_editable": False,
                "winner_kind": "none",
            }
        ),
        encoding="utf-8",
    )
    (exp / "logs").mkdir()
    (exp / "logs" / "opbench_gemm.log").write_text(
        "OPBENCH winner=hipblaslt speedup=1.0x editable=False\n", encoding="utf-8"
    )

    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, [])

    assert out["status"] == "no_result_recovered_from_disk"
    assert out["opbench_results"][0]["winner_backend"] == "hipblaslt"
    assert out["opbench_results"][0]["isolated_speedup"] == 1.0
    assert out["opbench_results"][0]["winner_editable"] is False
    assert "opbench_gemm.log" in out["runner_log_tails"]
    assert "OPBENCH" in out["runner_log_tails"]["opbench_gemm.log"]
    # op-bench ran, no editable winner, no journey/result → ran-no-winner.
    assert out["likely_cause"] == "ran_no_deployable_winner"


def test_collect_geak_reconstruct_cause_killed_before_flush(tmp_path: Path) -> None:
    # Stages reached but no journey/result.json/op-bench verdict → killed before flush.
    pf = tmp_path / "geak"
    exp = pf / "e2e_run"
    (exp / "baseline").mkdir(parents=True)
    (exp / "kernels" / "paged_attention_task").mkdir(parents=True)
    (pf / "handoff.json").write_text(json.dumps({"framework": "vllm"}), encoding="utf-8")

    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, [])

    assert out["status"] == "no_result_recovered_from_disk"
    assert out["opbench_results"] == []
    assert out["likely_cause"] == "killed_before_flush"


def test_collect_geak_reconstruct_cause_runner_reported_failure(tmp_path: Path) -> None:
    # A non-ok result.json was flushed → cause is the reported failure, not a silent kill.
    pf = tmp_path / "geak"
    pf.mkdir()
    (pf / "handoff.json").write_text(json.dumps({"framework": "vllm"}), encoding="utf-8")
    (pf / "result.json").write_text(
        json.dumps({"status": "error", "error_class": "timeout"}), encoding="utf-8"
    )

    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, [])

    assert out["status"] == "no_result_recovered_from_disk"
    assert out["flushed_result_status"] == "error"
    assert out["likely_cause"] == "runner_reported_failure"


def test_collect_geak_reconstruct_handoff_only(tmp_path: Path) -> None:
    # Only the handoff landed → still recoverable without an e2e exp_root.
    pf = tmp_path / "geak"
    pf.mkdir()
    (pf / "handoff.json").write_text(
        json.dumps({"framework": "sglang", "workload": {"conc": 32}}),
        encoding="utf-8",
    )
    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, [])
    assert out["status"] == "no_result_recovered_from_disk"
    assert out["recovered_from_disk"] is True
    assert out["stages_reached"] == ["handoff"]
    assert out["exp_root"] is None
    assert out["accepted_kernels"] == []


def test_collect_geak_empty_dir_falls_back_to_missing(tmp_path: Path) -> None:
    # An empty geak/ dir carries no evidence → legacy ``missing`` section.
    (tmp_path / "geak").mkdir()
    out = collect_geak(tmp_path, {"kernel_optimizer": "geak"}, [])
    assert out["status"] == "missing"
    assert "recovered_from_disk" not in out


def test_collect_geak_full_success_maps_fields(tmp_path: Path) -> None:
    eval_dir = tmp_path / "runs" / "geak" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
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
    out = collect_geak(tmp_path, state, warnings)

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
    assert out["eval_dir"] == "runs/geak/eval"


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


def test_collect_geak_backfills_accepted_kernels_from_journey(tmp_path: Path) -> None:
    # result.json shipped the aggregate win with empty accepted_kernels; the
    # collector back-fills the integrated KEEP kernel from kernel_journey.json.
    eval_dir = tmp_path / "geak" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
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
    out = collect_geak(tmp_path, state, warnings)

    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"
    only = out["accepted_kernels"][0]
    assert only["kernel_id"] == "fused_moe_kernel_gptq_awq"
    assert only["decision"] == "KEEP"
    # The GEAK e2e optimizer's own kernel backend is kept verbatim as ``geak``.
    assert only["backend"] == "geak"
    assert only["e2e_gain_pct"] == 16.049
    assert only["micro_speedup"] == 1.5902
    assert only["source"] == "kernel_journey_backfill"


def test_collect_geak_backfill_via_explicit_journey_path(tmp_path: Path) -> None:
    # kernel_journey_path takes precedence over deriving it from eval_dir.
    eval_dir = tmp_path / "geak" / "eval"
    kj_path = _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": [],
            "kernel_journey_path": str(kj_path),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"


def test_collect_geak_does_not_overwrite_populated_kernels(tmp_path: Path) -> None:
    # A populated accepted_kernels list is preserved verbatim, never replaced by the journey.
    eval_dir = tmp_path / "geak" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": ["rmsnorm"],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["accepted_kernels"] == ["rmsnorm"]
    assert out["accepted_kernels_source"] == "result"
    assert out["kernels_optimized"] == 1


def test_collect_geak_no_backfill_on_failure(tmp_path: Path) -> None:
    # A non-ok run must not back-fill (the e2e never landed a real win).
    eval_dir = tmp_path / "geak" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "error",
            "error_class": "timeout",
            "accepted_kernels": [],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["accepted_kernels"] == []
    assert out["accepted_kernels_source"] is None
    assert out["kernels_optimized"] == 0


def test_collect_geak_backfill_missing_journey_is_noop(tmp_path: Path) -> None:
    # ok run but no kernel_journey.json on disk → empty list, no crash.
    eval_dir = tmp_path / "geak" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "throughput_speedup": 1.16,
            "accepted_kernels": [],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["accepted_kernels"] == []
    assert out["accepted_kernels_source"] is None


def test_collect_geak_speedup_only_gain_and_bad_kernels(tmp_path: Path) -> None:
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "throughput_speedup": 1.5,
            # Non-list accepted_kernels must be coerced + warned.
            "accepted_kernels": "not-a-list",
        },
    }
    warnings: list[str] = []
    out = collect_geak(tmp_path, state, warnings)
    # No baseline/final, so gain derives from the speedup.
    assert out["gain_pct"] == 50.0
    assert out["accepted_kernels"] == []
    assert any("accepted_kernels" in w for w in warnings)


def test_collect_geak_relativize_failure_warns(tmp_path: Path, monkeypatch) -> None:
    # Force the relativization helper to raise so the except branch records a warning.
    def _boom(_path, _session_dir):
        raise OSError("relativize boom")

    # collect_geak binds ``_rel`` in the ``geak`` submodule; patch it there.
    monkeypatch.setattr(collectors.geak, "_rel", _boom)

    eval_dir = tmp_path / "runs" / "eval"
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "ok", "eval_dir": str(eval_dir)},
    }
    warnings: list[str] = []
    out = collect_geak(tmp_path, state, warnings)
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
    assert report["source"] == "geak"
    assert "error" not in report


def test_write_benchmark_report_oserror_is_swallowed(tmp_path: Path) -> None:
    # A file as out_dir makes the write raise NotADirectoryError; the writer must not raise.
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


def test_schema_has_geak_contract() -> None:
    # The wire schema must declare the top-level ``geak`` section the exporter emits.
    from hyperloom.inference_optimizer.breakdown import schema

    assert hasattr(schema, "Geak")
    assert "geak" in schema.SessionBreakdown.__annotations__
    # ``from __future__ import annotations`` stores the type as a string ref.
    assert "Geak" in str(schema.SessionBreakdown.__annotations__["geak"])
    # A few representative fields must be part of the declared shape.
    for field in ("engaged", "status", "error_class", "throughput_speedup", "accepted_kernels"):
        assert field in schema.Geak.__annotations__


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
async def test_sweep_via_geak_reuses_script_and_records_variants(
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

    result = await sweep_via_geak(
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
    assert result["source"] == "geak"
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
    assert fail_report["source"] == "geak"


@pytest.mark.asyncio
async def test_sweep_via_geak_requires_existing_bench_script(tmp_path: Path) -> None:
    result = await sweep_via_geak(
        result={"status": "ok", "bench_script": str(tmp_path / "missing.sh")},
        conc_values=[1],
        isl_osl_configs=["8:8"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=1,
    )

    assert result["status"] == "failed"
    assert result["error_class"] == "missing_bench_script"
