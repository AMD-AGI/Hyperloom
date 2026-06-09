#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run GEAK's FP8 GEMM tuning workflow as a Hyperloom kernel-agent tool.

Passes the task via file/string never argv, so GEAK's ``ps aux`` cleanup can't
match the task text and SIGKILL itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _safe_cleanup_clause() -> str:
    return (
        "SAFETY: do NOT run global process cleanup. In particular, never run "
        "`ps aux | grep ... | xargs kill`, `pgrep -f ... | xargs kill`, "
        "or `killall` for sglang / bench_serving / tuner processes. This "
        "workflow may run inside a live Hyperloom optimizer session; global "
        "cleanup can kill the main optimizer's benchmark server. Use the "
        "provided benchmark script as-is; it owns its server PID and port and "
        "cleans up only the process it starts. If a process must be stopped, "
        "stop only an explicit PID file created inside this workspace.\n"
    )


def _build_task(args: argparse.Namespace, workspace: Path) -> str:
    baseline = (
        f"Baseline output_throughput is already known: {args.baseline_tput:g} tok/s."
        if args.baseline_tput and args.baseline_tput > 0
        else "Run baseline at most once if needed to compute speedup."
    )
    return f"""Optimize the end-to-end performance of the workload via FP8 block-scale GEMM tuning, following the GEAK PR #228 contract on AMD MI355X / gfx950-class AMD GPUs.

CONSTRAINTS (do not deviate):
1) This action is valid only for SGLang FP8 A8W8 block-scale GEMM workloads. If the workload is not FP8, write final_report.json with status=\"skipped\" and explain why.
2) Use the aiter CK tuner at /sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py with --libtype ck, cktile, or all, and --mp 1. Do NOT write a custom Triton autotuner.
3) The deliverable CSV must be named a8w8_blockscale_tuned_gemm.csv. Canonical schema: gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio.
4) Modify /sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py so aiter_w8a8_block_fp8_linear dispatches via the CSV pointed to by AITER_CONFIG_GEMM_A8W8_BLOCKSCALE, falling back to existing logic for shapes missing from the CSV. Keep a copy of the patched file as fp8_utils_tuned.py in your workspace.
5) Benchmark script: {args.benchmark_script}. Workload knobs are locked: TP={args.tp}, CONC={args.conc}, ISL={args.isl}, OSL={args.osl}. Do not change these knobs. Do not edit the benchmark script and do not add global cleanup around it.
6) Model: {args.model_path}. Framework: {args.framework}. GPU type: {args.gpu_type}. Precision: {args.precision}.
7) {baseline}
8) Final report at final_report.json: status, tuned_file (absolute path to CSV), best_speedup (tuned/baseline), summary, tuned_file_usage.

{_safe_cleanup_clause()}

SEARCH SAFETY: do NOT run `find /`, `grep -R /`, or any unbounded root-filesystem
search. If you need to inspect skills or tuner files, use only these bounded
paths: /root/.claude/skills, ~/.claude/skills,
/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale,
/sgl-workspace/sglang/python/sglang/srt/layers/quantization,
and this workspace. Root-wide scans can hang the Hyperloom session.

The shell working directory is your workspace; keep benchmarks, tuner output, logs, CSVs, patched files, and final_report.json there.
Hyperloom session workspace root for this action: {workspace}
"""


def _latest_gemm_workspace(cwd: Path) -> Path | None:
    base = cwd / "optimization_logs"
    if not base.is_dir():
        return None
    candidates = [p for p in base.glob("gemm_tuning_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_report(workspace: Path | None) -> dict[str, Any]:
    if workspace is None:
        return {}
    path = workspace / "final_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"final_report_path": str(path), "error": "final_report.json is not valid JSON"}
    if isinstance(data, dict):
        data["final_report_path"] = str(path)
        return data
    return {"final_report_path": str(path), "error": "final_report.json is not a JSON object"}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperloom GEAK GEMM tuning wrapper")
    p.add_argument("--input-json", default="")
    p.add_argument("--config", default=os.environ.get("GEAK_CONFIG", ""))
    p.add_argument("--cwd", default="")
    p.add_argument("--model-path", default="")
    p.add_argument("--benchmark-script", default="")
    p.add_argument("--framework", default="sglang")
    p.add_argument("--gpu-type", default="")
    p.add_argument("--precision", default="")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--conc", type=int, default=0)
    p.add_argument("--isl", type=int, default=0)
    p.add_argument("--osl", type=int, default=0)
    p.add_argument("--baseline-tput", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _apply_input_json(args: argparse.Namespace) -> argparse.Namespace:
    if not args.input_json:
        return args
    path = Path(args.input_json)
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {path}")
    for key, value in data.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _apply_input_json(_parse_args(list(argv or sys.argv[1:])))
    if not args.cwd:
        _json_line({"status": "failed", "error_class": "cwd_missing", "error": "cwd is required"})
        return 2
    if not args.model_path:
        _json_line({"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"})
        return 2
    if not args.benchmark_script:
        _json_line({
            "status": "failed",
            "error_class": "benchmark_script_missing",
            "error": "benchmark_script is required",
        })
        return 2
    cwd = Path(args.cwd).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    task = _build_task(args, cwd)
    task_file = cwd / "gemm_tuning_task.txt"
    task_file.write_text(task, encoding="utf-8")

    if args.dry_run:
        _json_line({
            "status": "ok",
            "dry_run": True,
            "workspace": str(cwd),
            "task_file": str(task_file),
            "argv_task_safe": True,
        })
        return 0

    if not args.config:
        _json_line({
            "status": "failed",
            "error_class": "geak_config_missing",
            "error": "GEAK config is required via --config or GEAK_CONFIG",
            "workspace": str(cwd),
            "task_file": str(task_file),
        })
        return 2

    exit_code: int | str = 0
    try:
        from minisweagent.run.gemm_tuning import run as geak_gemm_run

        geak_gemm_run(
            task=task,
            config=str(args.config),
            cwd=cwd,
            model_name=None,
            log_dir=None,
        )
    except SystemExit as exc:
        exit_code = exc.code if exc.code is not None else 0
    except BaseException as exc:  # noqa: BLE001 - return structured failure
        _json_line({
            "status": "failed",
            "error_class": exc.__class__.__name__,
            "error": repr(exc),
            "workspace": str(cwd),
            "task_file": str(task_file),
        })
        return 1

    workspace = _latest_gemm_workspace(cwd)
    report = _load_report(workspace)
    status_raw = str(report.get("status") or "").strip().lower()
    ok = status_raw in {"complete", "completed", "ok", "succeeded", "success"}
    if not report:
        _json_line({
            "status": "failed",
            "error_class": "final_report_missing",
            "error": "GEAK completed without writing final_report.json",
            "returncode": exit_code,
            "workspace": str(workspace or cwd),
            "task_file": str(task_file),
        })
        return 1

    out = {
        "status": "ok" if ok else (status_raw or "failed"),
        "decision": "KEEP" if ok and float(report.get("best_speedup") or 0.0) > 1.0 else "REVERT",
        "returncode": exit_code,
        "workspace": str(workspace or cwd),
        "task_file": str(task_file),
        "argv_task_safe": True,
        **report,
    }
    _json_line(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
