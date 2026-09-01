#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the forge GEMM tuner as a Hyperloom kernel-agent tool.

The orchestrator writes an input JSON file and calls this script; the
deterministic tuning implementation lives in ``kernelforge.gemm_tune``, reached
through the one forge CLI as ``python -m kernelforge.cli gemm-tune run``.
"""

from __future__ import annotations

import argparse
import json
import os
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
    _add_opt(cmd, args, "untuned_csv", "--untuned-csv")
    _add_opt(cmd, args, "moe_untuned_csv", "--moe-untuned-csv")
    _add_opt(cmd, args, "shapes_json", "--shapes-json")
    # forge calls the manifest its preferred dense-shape source, and Hyperloom
    # has produced one since WP-1 -- but nothing forwarded it, so the file was
    # written and never read. Same for --demand: forge can reconstruct demand
    # from --kernel-signature-log, and does, but a demand.json the evidence
    # step already computed is both cheaper and not subject to picking the
    # wrong log. Both degrade safely: forge's ``_safe_is_file`` drops a path
    # that is not there, with a warning.
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
    """Forward the backend lib version recorded as artifact provenance.

    Tuning no longer carries a knowledge base, so the options that read from one
    or judged a candidate against it are gone: forwarding them would abort the
    run on an unrecognised argument. What remains records which backend build
    produced an artifact, which is provenance rather than knowledge.
    """
    cur_lib = args.get("kb_current_lib")
    if cur_lib in (None, ""):
        cur_lib = os.environ.get("FORGE_GEMM_TUNE_KB_CURRENT_LIB", "")
    cur_lib = str(cur_lib).strip()
    if cur_lib:
        cmd.extend(["--kb-current-lib", cur_lib])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperloom wrapper for kernelforge gemm-tune")
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
