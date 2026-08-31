# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
from hyperloom.inference_optimizer.breakdown.collectors.geak import (
    _geak_accepted_kernels_from_journey,
    _geak_kind_index,
)
from hyperloom.orchestrator.actions.executors._geak_sweep import (
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
    for stage in ("handoff", "baseline", "kernels", "opbench", "strategy", "kernel_journey"):
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
    (pf / "result.json").write_text(json.dumps({"status": "error", "error_class": "timeout"}), encoding="utf-8")

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


def test_collect_geak_preserves_failed_revalidation_diagnostic(tmp_path: Path) -> None:
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "baseline_throughput_tok_s": 100.0,
            "final_throughput_tok_s": 103.2,
            "revalidation_status": "failed",
            "revalidation_error_class": "subprocess_nonzero",
            "revalidation_error": "same-harness rebench exited 1",
        },
    }

    out = collect_geak(tmp_path, state, [])

    assert out["revalidation_status"] == "failed"
    assert out["revalidation_error_class"] == "subprocess_nonzero"
    assert out["revalidation_error"] == "same-harness rebench exited 1"


def test_collect_geak_result_reads_accepted_heads_lane(tmp_path: Path) -> None:
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "accepted_heads": [
                {
                    "short_name": "dsa_sparse_attn_prefill_main_kernel",
                    "kind": "authored",
                    "e2e_delta_pct": 12.5,
                }
            ],
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels_source"] == "result"
    assert out["accepted_kernels"][0]["short_name"] == "dsa_sparse_attn_prefill_main_kernel"


def test_collect_geak_result_excludes_env_kind(tmp_path: Path) -> None:
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [
                {
                    "short_name": "ck_gemm_selection",
                    "kind": "env",
                    "e2e_delta_pct": 8.0,
                }
            ],
            "accepted_heads": [],
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 0
    assert out["accepted_kernels"] == []


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


def test_schema_has_optimizations_contract() -> None:
    # V5 exposes adopted GEAK results through the canonical optimizations
    # section and route diagnostics through the dedicated GEAK section.
    from hyperloom.inference_optimizer.breakdown import schema

    assert hasattr(schema, "Optimizations")
    assert "optimizations" in schema.SessionBreakdown.__annotations__
    # ``from __future__ import annotations`` stores the type as a string ref.
    assert "Optimizations" in str(schema.SessionBreakdown.__annotations__["optimizations"])
    assert "Geak" in str(schema.SessionBreakdown.__annotations__["geak"])
    # A few representative fields must be part of the declared shape.
    for field in ("entries", "backend_attempts", "summary_by_source", "validation"):
        assert field in schema.Optimizations.__annotations__


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
            "output_dir": str(tmp_path),
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
        (tmp_path / "sweep" / "variant_1_conc2_isl8192_osl1024" / "benchmark_report.json").read_text(encoding="utf-8")
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


def test_collect_gemm_tuning_prefers_e2e_gain_over_micro_speedup() -> None:
    from hyperloom.inference_optimizer.breakdown.collectors.kernels import collect_gemm_tuning

    out = collect_gemm_tuning(
        {
            "baseline_tput": 1000.0,
            "gemm_tuning_attempts": [
                {
                    "engine": "forge",
                    "status": "complete",
                    "decision": "KEEP",
                    "best_speedup": 1.5,
                    "e2e_gain_pct": 9.26,
                    "e2e_validated": True,
                    "tuned_file": "/tmp/tuned.csv",
                }
            ],
        }
    )

    run = out["runs"][0]
    assert run["gain_pct"] == pytest.approx(9.26)
    assert run["tuned_tput"] == pytest.approx(1092.6)
    assert run["best_speedup"] == pytest.approx(1.5)


def _gemm_state_with_keep(*, attempt_tuned_file: str) -> dict:
    """A session whose forge run was kept and lifted onto the stack."""
    return {
        "baseline_tput": 1000.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "gemm_tuning",
                "tuned_file": "/ws/merged_tuned_fmoe.csv",
                "gain_pct": 9.26,
            }
        ],
        "gemm_tuning_attempts": [
            {
                "engine": "forge",
                "status": "complete",
                "decision": "KEEP",
                "e2e_gain_pct": 9.26,
                "e2e_validated": True,
                "tuned_file": attempt_tuned_file,
            }
        ],
    }


def test_collect_gemm_tuning_marks_a_kept_run_adopted() -> None:
    """A forge run whose artifact reached the stack must read as adopted.

    Across 419 real forge attempts this was never true: the attempt row carried
    no ``tuned_file`` at all, so the stack lookup matched on the empty string
    and every KEEP -- including ones measuring +49% -- was reported as not
    adopted.
    """
    from hyperloom.inference_optimizer.breakdown.collectors.kernels import collect_gemm_tuning

    out = collect_gemm_tuning(_gemm_state_with_keep(attempt_tuned_file="/ws/merged_tuned_fmoe.csv"))

    run = out["runs"][0]
    assert run["adopted"] is True
    assert run["gain_pct"] == pytest.approx(9.26)


def test_collect_gemm_tuning_leaves_an_unlifted_run_unadopted() -> None:
    """Fail-open on the empty case must not make every run look adopted."""
    from hyperloom.inference_optimizer.breakdown.collectors.kernels import collect_gemm_tuning

    out = collect_gemm_tuning(_gemm_state_with_keep(attempt_tuned_file=""))

    assert out["runs"][0]["adopted"] is False


class TestCandidateTunedFile:
    """The artifact a KEEP adopted, named from the candidate's own env.

    One KEEP is described by three different path strings -- the durable copy in
    aiter's config dir, the tuner-workspace original, and the E2E merge product
    -- so the attempt row cannot re-derive the one the stack holds. The way out
    is not to read the stack back either: reading it back picks up whatever entry
    is newest, and ``_lift_to_current_best`` skips the append when
    ``(action, variant_name)`` already matches, which a second macro cycle
    re-tuning the same tuner does. The attempt would then claim the previous
    round's artifact and its gain. Both sides take the value from this one
    function instead, so they are the same string by construction.
    """

    def test_prefers_the_candidate_env_var(self) -> None:
        """The candidate's own key wins over whatever the env happens to list
        first -- a stacked env carries the earlier KEEPs' vars too."""
        from hyperloom.orchestrator.phases.kernel import _candidate_tuned_file

        # Deliberately not first: falling back to insertion order would pick
        # the wrong artifact and still look right if the target led the dict.
        env = {
            "AITER_CONFIG_GEMM_BF16": "/ws/earlier_keep.csv",
            "AITER_CONFIG_FMOE": "/ws/merged_tuned_fmoe.csv",
        }
        assert _candidate_tuned_file(env, "AITER_CONFIG_FMOE") == "/ws/merged_tuned_fmoe.csv"

    def test_falls_back_to_the_only_value_present(self) -> None:
        """A candidate whose env_var is not the key its env carries."""
        from hyperloom.orchestrator.phases.kernel import _candidate_tuned_file

        env = {"AITER_CONFIG_GEMM_A8W8": "/ws/tuned_a8w8.csv"}
        assert _candidate_tuned_file(env, "AITER_CONFIG_FMOE") == "/ws/tuned_a8w8.csv"

    def test_empty_env_yields_no_claim(self) -> None:
        from hyperloom.orchestrator.phases.kernel import _candidate_tuned_file

        assert _candidate_tuned_file({}, "AITER_CONFIG_FMOE") == ""

    def test_tolerates_malformed_input(self) -> None:
        from hyperloom.orchestrator.phases.kernel import _candidate_tuned_file

        assert _candidate_tuned_file({"AITER_CONFIG_FMOE": None}, "AITER_CONFIG_FMOE") == ""
        assert _candidate_tuned_file({"AITER_CONFIG_FMOE": ""}, "AITER_CONFIG_FMOE") == ""
        assert _candidate_tuned_file(None, "AITER_CONFIG_FMOE") == ""
        assert _candidate_tuned_file({"k": 42}, "k") == "42"

    def test_the_stack_reader_is_gone(self) -> None:
        """Reading the newest stack entry is what allowed the false claim."""
        from hyperloom.orchestrator.phases import kernel as kernel_phase

        assert not hasattr(kernel_phase, "_adopted_tuned_file")


def test_collect_geak_backfill_fires_on_no_gain(tmp_path: Path) -> None:
    # A run stamped ``no_gain`` on the COLD basis can still hold a measured hot
    # win and genuine KEEP rows in the journey. Attribution must not be dropped.
    eval_dir = tmp_path / "geak" / "eval"
    _write_kernel_journey(eval_dir)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "no_gain",
            "throughput_speedup": 0.9877,
            "alignment_metrics": {"hot_geak_speedup": 2.5722, "final_basis": "cold"},
            "accepted_kernels": [],
            "accepted_heads": [{"short_name": "fused_moe_kernel_gptq_awq"}],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels"][0]["kernel_id"] == "fused_moe_kernel_gptq_awq"


def test_collect_geak_backfill_still_skipped_on_error(tmp_path: Path) -> None:
    # ``error`` / ``timeout`` runs never produced a trustworthy workflow return;
    # the gate must stay closed for them.
    eval_dir = tmp_path / "geak" / "eval"
    _write_kernel_journey(eval_dir)
    for bad in ("error", "timeout", "missing"):
        state = {
            "kernel_optimizer": "geak",
            "geak_result": {
                "status": bad,
                "accepted_kernels": [],
                "eval_dir": str(eval_dir),
            },
        }
        out = collect_geak(tmp_path, state, [])
        assert out["accepted_kernels"] == [], bad
        assert out["kernels_optimized"] == 0, bad


def test_collect_geak_backfill_scans_earlier_cycles(tmp_path: Path) -> None:
    # ``kernel_journey_path`` names the LAST e2e cycle. A run that keeps a kernel
    # in cycle 0 and then opens a cycle 1 that keeps nothing must still attribute
    # the cycle-0 kernel. Observed on two campaign runs.
    geak_dir = tmp_path / "geak"
    _write_kernel_journey(geak_dir / "e2e_cycle0")
    last = geak_dir / "e2e_cycle1"
    last.mkdir(parents=True, exist_ok=True)
    (last / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {
                        "kernel_id": "paged_attention_decode_sliding_window_c0_triton",
                        "name": "paged_attention_decode_sliding_window_c0_triton",
                        "e2e": {"decision": "REJECTED", "integrated": False, "e2e_gain_pct": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "no_gain",
            "accepted_kernels": [],
            "kernel_journey_path": str(last / "kernel_journey.json"),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["accepted_kernels_source"] == "kernel_journey_backfill"
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels"][0]["kernel_id"] == "fused_moe_kernel_gptq_awq"


def test_collect_geak_backfill_dedupes_repeated_kernel_across_cycles(tmp_path: Path) -> None:
    # The same kernel_id present in two cycles must be credited once, and the
    # pointer (last) cycle wins.
    geak_dir = tmp_path / "geak"
    _write_kernel_journey(geak_dir / "e2e_cycle0")
    _write_kernel_journey(geak_dir / "e2e_cycle1")
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "kernel_journey_path": str(geak_dir / "e2e_cycle1" / "kernel_journey.json"),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1


def _alias_journey(
    eval_dir: Path,
    *,
    primary_gain: float,
    twin_gain: float,
    op_kind: str = "sparse_attn",
) -> None:
    """Journey holding one accepted kernel written twice (candidate + symbol)."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {
                        "kernel_id": "c0_triton",
                        "name": "c0_triton",
                        "op_kind": op_kind,
                        "gpu_pct": 20.2,
                        "micro_speedup": 2.2024,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": primary_gain},
                    },
                    {
                        "kernel_id": "dsa_sparse_attn_prefill_main_kernel",
                        "name": "dsa_sparse_attn_prefill_main_kernel",
                        "op_kind": op_kind,
                        "gpu_pct": None,
                        "micro_speedup": 1.13,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": twin_gain},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_journey_backfill_projects_op_kind(tmp_path: Path) -> None:
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {
                        "kernel_id": "k1",
                        "name": "k1",
                        "op_kind": "sparse_attn",
                        "gpu_pct": 10.0,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": 12.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = _geak_accepted_kernels_from_journey({"eval_dir": str(eval_dir)}, [])
    assert rows[0]["op_kind"] == "sparse_attn"


def test_journey_backfill_keeps_distinct_op_kinds_at_same_gain(tmp_path: Path) -> None:
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {
                        "kernel_id": "kernel_a",
                        "name": "kernel_a",
                        "op_kind": "prefill_attn",
                        "gpu_pct": 10.0,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": 5.0},
                    },
                    {
                        "kernel_id": "kernel_b",
                        "name": "kernel_b",
                        "op_kind": "decode_attn",
                        "gpu_pct": 12.0,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": 5.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "ok", "accepted_kernels": [], "eval_dir": str(eval_dir)},
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 2
    assert {k["kernel_id"] for k in out["accepted_kernels"]} == {"kernel_a", "kernel_b"}


def test_collect_geak_backfill_collapses_alias_twin(tmp_path: Path) -> None:
    # The journey records one acceptance twice: the candidate id carries the
    # measurement, the resolved profiler symbol carries gpu_pct=None. One kernel.
    #
    # Which row survives and which id names it are two separate questions. The
    # measured row survives, because it is the only one holding ``gpu_pct``; it
    # is then named by the *symbol*, because that is the id the acceptance
    # ledger keeps for the same kernel. Naming it ``c0_triton`` here put one
    # kernel under two names in two tables of the same report.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=29.994, twin_gain=29.994)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "no_gain", "accepted_kernels": [], "eval_dir": str(eval_dir)},
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    kernel = out["accepted_kernels"][0]
    assert kernel["kernel_id"] == "dsa_sparse_attn_prefill_main_kernel"
    # The measurement survives the rename: it came from the row that had it.
    assert kernel["gpu_pct"] == 20.2
    assert kernel["aliases"] == ["c0_triton"]


def test_collect_geak_backfill_collapses_rounded_alias_twin(tmp_path: Path) -> None:
    # The twin sometimes carries the rounded gain, so exact equality is too strict.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=10.38337292749906, twin_gain=10.383)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "ok", "accepted_kernels": [], "eval_dir": str(eval_dir)},
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    assert out["accepted_kernels"][0]["kernel_id"] == "dsa_sparse_attn_prefill_main_kernel"
    assert out["accepted_kernels"][0]["aliases"] == ["c0_triton"]


def test_collect_geak_backfill_excludes_declared_env_selection(tmp_path: Path) -> None:
    # Real shape, Qwen3-14B-FP8/20260814T163051Z: the journey holds an alias
    # twin whose resolved symbol is a CK library GEMM. ``accepted_kernels`` is
    # empty and the win sits in ``accepted_heads`` declaring ``kind: env``.
    #
    # The collapse must run before the kind join, or the join has only the slot
    # tag ``c1_ck`` to look up and finds nothing. Once the row is named by the
    # symbol, ``result.json`` answers the question GEAK already answered: this
    # is a library selection, not an authored kernel. It belongs to the config
    # bucket, so ``kernels_optimized`` is 0 -- the run's e2e gain is unaffected.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=14.924, twin_gain=14.924)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "accepted_heads": [
                {
                    "short_name": "dsa_sparse_attn_prefill_main_kernel",
                    "kind": "env",
                    "backend": "ck",
                    "e2e_delta_pct": 14.924,
                }
            ],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 0
    assert out["accepted_kernels"] == []


def test_collect_geak_backfill_keeps_authored_after_collapse(tmp_path: Path) -> None:
    # The converse, so the exclusion above can never be widened into a drop:
    # the same twin declared ``authored`` survives, named by the symbol, and
    # records where its kind came from. Only a *declared* env is excluded --
    # a row no lane names stays admitted with ``kind_source: absent``, because
    # guessing "env" would delete real kernels from dead runs.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=14.924, twin_gain=14.924)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "accepted_heads": [{"short_name": "dsa_sparse_attn_prefill_main_kernel", "kind": "authored"}],
            "eval_dir": str(eval_dir),
        },
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    kernel = out["accepted_kernels"][0]
    assert kernel["kernel_id"] == "dsa_sparse_attn_prefill_main_kernel"
    assert kernel["kind"] == "authored"
    assert kernel["kind_source"] == "result_json"


def test_collect_geak_backfill_reads_kind_from_the_stack_when_result_json_is_empty(
    tmp_path: Path,
) -> None:
    # ``result.json`` is rewritten once per cycle and the last write wins, so a
    # later cycle that accepts nothing blanks the lanes an earlier one declared.
    # The ``geak_e2e`` optimization_stack entry is append-only and keeps them
    # (KernelPhase copies both lanes into it verbatim). Reading only
    # ``result.json`` here left every recovered row ``kind_source: absent`` and
    # the ``env`` exclusion could not run at all -- the collector counted a CK
    # library selection as an authored kernel.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=14.924, twin_gain=14.924)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "accepted_heads": [],
            "eval_dir": str(eval_dir),
        },
        "optimization_stack": [
            {"action": "integrate", "kernel_id": "unrelated"},
            {
                "action": "geak_e2e",
                "accepted_kernels": [],
                "accepted_heads": [
                    {
                        "short_name": "dsa_sparse_attn_prefill_main_kernel",
                        "kind": "env",
                        "backend": "ck",
                    }
                ],
            },
        ],
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 0
    assert out["accepted_kernels"] == []


def test_collect_geak_backfill_stack_kind_is_labelled_as_from_the_stack(
    tmp_path: Path,
) -> None:
    # The converse, and the provenance. An authored declaration in the stack
    # keeps the row, and ``kind_source`` says *which* artifact declared it, so a
    # stack-sourced kind is never reported as something ``result.json`` said.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    _alias_journey(eval_dir, primary_gain=14.924, twin_gain=14.924)
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {
            "status": "ok",
            "accepted_kernels": [],
            "accepted_heads": [],
            "eval_dir": str(eval_dir),
        },
        "optimization_stack": [
            {
                "action": "geak_e2e",
                "accepted_kernels": [{"short_name": "dsa_sparse_attn_prefill_main_kernel", "kind": "authored"}],
                "accepted_heads": [],
            }
        ],
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 1
    kernel = out["accepted_kernels"][0]
    assert kernel["kernel_id"] == "dsa_sparse_attn_prefill_main_kernel"
    assert kernel["kind"] == "authored"
    assert kernel["kind_source"] == "stack"


def test_geak_kind_index_prefers_the_run_s_own_result_json_over_the_stack(
    tmp_path: Path,
) -> None:
    # Adding a second source must not let it overwrite what the run published.
    # A *declared* kind beats an undeclared one whichever artifact holds it;
    # between two declarations ``result.json`` wins.
    result = {
        "accepted_kernels": [{"short_name": "a", "kind": "authored"}, {"short_name": "b"}],
    }
    stack = [
        {
            "action": "geak_e2e",
            "accepted_heads": [
                {"short_name": "a", "kind": "env"},
                {"short_name": "b", "kind": "env"},
                {"short_name": "c", "kind": "env"},
            ],
        }
    ]
    index = _geak_kind_index(result, stack)
    assert index["a"] == ("authored", "result_json")  # published kind untouched
    assert index["b"] == ("env", "stack")  # undeclared lane filled in
    assert index["c"] == ("env", "stack")  # name only the stack knows
    assert _geak_kind_index(result) == {"a": ("authored", "result_json"), "b": (None, "result_json")}


def test_collect_geak_backfill_keeps_two_measured_kernels_of_equal_gain(tmp_path: Path) -> None:
    # Two genuinely distinct kernels that happen to share a gain must both stay:
    # the collapse needs a measured row AND an unmeasured row to fire.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {
                        "kernel_id": f"kernel_{i}",
                        "gpu_pct": 10.0 + i,
                        "e2e": {"decision": "KEEP", "integrated": True, "e2e_gain_pct": 5.0},
                    }
                    for i in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "ok", "accepted_kernels": [], "eval_dir": str(eval_dir)},
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 2
    assert all("aliases" not in k for k in out["accepted_kernels"])


def test_collect_geak_backfill_keeps_unmeasured_kernels_of_distinct_gain(tmp_path: Path) -> None:
    # Two unmeasured shape-split kernels are not aliases of the measured parent:
    # their gains differ, so all three survive.
    eval_dir = tmp_path / "geak" / "e2e_cycle0"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "kernel_journey.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernels": [
                    {"kernel_id": "c0_aiter", "gpu_pct": 0.0, "e2e": {"decision": "KEEP", "e2e_gain_pct": 7.53}},
                    {"kernel_id": "ck#1", "gpu_pct": None, "e2e": {"decision": "KEEP", "e2e_gain_pct": 6.779}},
                    {"kernel_id": "ck#2", "gpu_pct": None, "e2e": {"decision": "KEEP", "e2e_gain_pct": 0.705}},
                ],
            }
        ),
        encoding="utf-8",
    )
    state = {
        "kernel_optimizer": "geak",
        "geak_result": {"status": "no_gain", "accepted_kernels": [], "eval_dir": str(eval_dir)},
    }
    out = collect_geak(tmp_path, state, [])
    assert out["kernels_optimized"] == 3
