#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run forge-gemm-tune as a Hyperloom kernel-agent tool.

This wrapper mirrors ``gemm_tuning.py`` (GEAK) at the kernel-agent tool layer:
the orchestrator writes an input JSON file and calls this script; the actual
deterministic tuning implementation remains in the standalone
``forge_gemm_tune`` package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_input_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {p}")
    return data


def _add_opt(cmd: list[str], args: dict[str, Any], key: str, flag: str, *, required: bool = False) -> None:
    val = args.get(key)
    if val in (None, ""):
        if required:
            raise ValueError(f"{key} is required")
        return
    cmd.extend([flag, str(val)])


def _build_cmd(args: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "forge_gemm_tune.cli", "run"]
    _add_opt(cmd, args, "model_path", "--model-path", required=True)
    _add_opt(cmd, args, "framework", "--framework", required=True)
    _add_opt(cmd, args, "precision", "--precision", required=True)
    _add_opt(cmd, args, "quant_type", "--quant-type")
    _add_opt(cmd, args, "gpu_type", "--gpu-type")
    _add_opt(cmd, args, "tp", "--tp")
    _add_opt(cmd, args, "conc", "--conc")
    _add_opt(cmd, args, "mp", "--mp")
    _add_opt(cmd, args, "output_dir", "--output-dir", required=True)
    _add_opt(cmd, args, "iters", "--iters")
    _add_opt(cmd, args, "warmup", "--warmup")
    _add_opt(cmd, args, "min_improvement_pct", "--min-improvement-pct")
    _add_opt(cmd, args, "timeout", "--timeout")
    _add_opt(cmd, args, "global_timeout", "--global-timeout")
    _add_opt(cmd, args, "tuner", "--tuner")
    _add_opt(cmd, args, "untuned_csv", "--untuned-csv")
    _add_opt(cmd, args, "shapes_json", "--shapes-json")
    _add_opt(cmd, args, "tunableop_input", "--tunableop-input")
    _add_opt(cmd, args, "kernel_signature_log", "--kernel-signature-log")
    _add_opt(cmd, args, "gpu_ids", "--gpu-ids")
    if bool(args.get("skip_gpu_check", True)):
        cmd.append("--skip-gpu-check")
    if bool(args.get("verbose", False)):
        cmd.append("--verbose")
    if bool(args.get("thorough", False)):
        cmd.append("--thorough")
    tokens = str(args.get("tokens") or "").strip()
    if tokens:
        cmd.extend(["--tokens", tokens])
    return cmd


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperloom wrapper for forge-gemm-tune")
    p.add_argument("--input-json", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(list(argv or sys.argv[1:]))
        payload = _load_input_json(args.input_json)
        cmd = _build_cmd(payload)
    except Exception as exc:  # noqa: BLE001 - structured wrapper failure
        print(
            json.dumps(
                {
                    "status": "failed",
                    "micro_decision": "failed",
                    "error_class": exc.__class__.__name__,
                    "error": repr(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
