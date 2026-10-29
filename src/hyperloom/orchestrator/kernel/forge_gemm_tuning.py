# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``kernelforge gemm-tune run`` command line for a Hyperloom GEMM tuning request."""

from __future__ import annotations

import os
import sys
from typing import Any


def _add_opt(cmd: list[str], args: dict[str, Any], key: str, flag: str, *, required: bool = False) -> None:
    val = args.get(key)
    if val in (None, ""):
        if required:
            raise ValueError(f"{key} is required")
        return
    cmd.extend([flag, str(val)])


def build_command(args: dict[str, Any]) -> list[str]:
    """Return the gemm-tune argv for ``args``; raises ``ValueError`` when a required field is missing."""
    cmd = [sys.executable, "-m", "kernelforge.cli", "gemm-tune", "run"]
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
    # Omitted by the caller when no lane ceiling was derived; gemm-tune reads both an absent flag and an explicit 0 as
    # "run every routed tuner".
    _add_opt(cmd, args, "max_tuners", "--max-tuners")
    _add_opt(cmd, args, "untuned_csv", "--untuned-csv")
    _add_opt(cmd, args, "moe_untuned_csv", "--moe-untuned-csv")
    _add_opt(cmd, args, "shapes_json", "--shapes-json")
    _add_opt(cmd, args, "shapes_manifest", "--shapes-manifest")
    _add_opt(cmd, args, "demand_json", "--demand")
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
    _add_kb_opts(cmd, args)
    return cmd


def _add_kb_opts(cmd: list[str], args: dict[str, Any]) -> None:
    """Forward the backend lib version recorded as artifact provenance."""
    cur_lib = args.get("kb_current_lib")
    if cur_lib in (None, ""):
        cur_lib = os.environ.get("FORGE_GEMM_TUNE_KB_CURRENT_LIB", "")
    cur_lib = str(cur_lib).strip()
    if cur_lib:
        cmd.extend(["--kb-current-lib", cur_lib])
