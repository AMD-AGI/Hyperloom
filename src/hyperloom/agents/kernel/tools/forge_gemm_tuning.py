#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run forge-gemm-tune as a Hyperloom kernel-agent tool.

The orchestrator writes an input JSON file and calls this script; the
deterministic tuning implementation lives in the standalone ``forge_gemm_tune``
package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Sibling import: kernel-agent tools cannot rely on the ``hyperloom`` import root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _io_utils import truthy as _truthy  # noqa: E402

sys.path.pop(0)


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
    _add_kb_read_opts(cmd, args)
    return cmd


def _add_kb_read_opts(cmd: list[str], args: dict[str, Any]) -> None:
    """Forward gemm-tune-kb read-side flags from input-json or env.

    The input-json takes precedence; env vars let the pipeline enable KB read
    without populating input-json. Disabled by default.
    """
    kb_read = args.get("kb_read")
    if kb_read is None:
        kb_read = os.environ.get("FORGE_GEMM_TUNE_KB_READ", "")
    if not _truthy(kb_read):
        return
    cmd.append("--kb-read")

    accept = args.get("kb_accept_candidate")
    if accept is None:
        accept = os.environ.get("FORGE_GEMM_TUNE_KB_ACCEPT_CANDIDATE", "")
    if _truthy(accept):
        cmd.append("--kb-accept-candidate")

    strict = args.get("kb_strict_lib")
    if strict is None:
        strict = os.environ.get("FORGE_GEMM_TUNE_KB_STRICT_LIB", "")
    if _truthy(strict):
        cmd.append("--kb-strict-lib")

    cur_lib = args.get("kb_current_lib")
    if cur_lib in (None, ""):
        cur_lib = os.environ.get("FORGE_GEMM_TUNE_KB_CURRENT_LIB", "")
    cur_lib = str(cur_lib).strip()
    if cur_lib:
        cmd.extend(["--kb-current-lib", cur_lib])


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
