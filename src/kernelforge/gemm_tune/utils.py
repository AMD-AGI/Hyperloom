# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utility functions for kernelforge gemm-tune: GPU detection, subprocess, constants."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aiter_script_map import TUNER_SCRIPT_HINTS as _TUNER_SCRIPT_HINTS

# Re-exported: these used to live here, and both callers and tests import them
# from this module. They moved to the leaf so ``script_discovery`` can reach
# them without importing this module back.
from .aiter_script_map import resolve_aiter_csrc, resolve_aiter_root  # noqa: F401

log = logging.getLogger(__name__)


def sha256_file(path: str | Path | None) -> str:
    """Return the sha256 hex digest of a file, or ``""`` on any I/O error.

    Used to fingerprint produced tuned CSVs in the TuningArtifactManifest so a
    consumer can verify the artifact it applies matches what was tuned.
    """
    if not path:
        return ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# Sentinel markers for stdout JSON output (Hyperloom parses between these)
RESULT_SENTINEL_BEGIN = "FORGE_GEMM_TUNE_RESULT_BEGIN"
RESULT_SENTINEL_END = "FORGE_GEMM_TUNE_RESULT_END"

# Preferred relative path per tuner, derived from the discovery hints so there is
# one source of truth. Kept as a plain mapping for callers that only want the
# expected location; resolution itself goes through script_discovery, which falls
# back to searching when aiter has moved the file.
AITER_TUNER_SCRIPTS = {name: rels[0] for name, rels in _TUNER_SCRIPT_HINTS.items() if rels}

# Environment variable names for tuned config outputs
TUNER_ENV_VARS = {
    "fmoe_ck": "AITER_CONFIG_FMOE",
    "a8w8": "AITER_CONFIG_GEMM_A8W8",
    "a8w8_blockscale": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    "a8w8_bpreshuffle": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    "a8w8_blockscale_bpreshuffle": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    # aiter reads the a4w4 (fp4/mxfp4, gfx950-only) config via AITER_CONFIG_GEMM_A4W4
    # (jit/core.py); the "_BLOCKSCALE" suffix here was a dead key aiter never reads,
    # which silently dropped all tuned fp4 GEMM configs at serving. Runtime filename
    # (a4w4_blockscale_tuned_gemm.csv) matches aiter's default and is unchanged.
    "a4w4_blockscale": "AITER_CONFIG_GEMM_A4W4",
    "sglang_dense_bf16": "AITER_CONFIG_GEMM_BF16",
    "vllm_moe_triton": "VLLM_TUNED_CONFIG_FOLDER",
    "vllm_dense_tunableop": "PYTORCH_TUNABLEOP_FILENAME",
}


@dataclass
class GpuInfo:
    """GPU status from rocm-smi."""

    gpu_id: int
    temperature: str
    power: str
    utilization: str
    memory_used: str
    memory_total: str
    busy: bool


def find_tuner_script(tuner_name: str) -> Path | None:
    """Locate a specific aiter tuner script by name.

    Delegates to script_discovery: the hinted path is tried first, then the file
    is searched for. A hardcoded path is what left the bf16 tuner pointing at
    ``gradlib/`` after aiter moved it.
    """
    from .script_discovery import discover_tuner_script

    return discover_tuner_script(tuner_name)


def check_gpu_status(skip: bool = False) -> list[GpuInfo]:
    """Run rocm-smi and return GPU status list. Returns empty if skip=True or rocm-smi unavailable."""
    if skip:
        return []
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showpower", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            log.warning("rocm-smi returned %d", proc.returncode)
            return []
        data = json.loads(proc.stdout)
        gpus = []
        for key, info in data.items():
            if not key.startswith("card"):
                continue
            gpu_id = int(key.replace("card", ""))
            util_str = str(info.get("GPU use (%)", info.get("GPU Utilization (%)", "0")))
            util_val = float(util_str.replace("%", "").strip() or "0")
            gpus.append(
                GpuInfo(
                    gpu_id=gpu_id,
                    temperature=str(info.get("Temperature (Sensor edge) (C)", "")),
                    power=str(info.get("Average Graphics Package Power (W)", "")),
                    utilization=util_str,
                    memory_used=str(info.get("VRAM Total Used Memory (B)", "")),
                    memory_total=str(info.get("VRAM Total Memory (B)", "")),
                    busy=util_val > 50.0,
                )
            )
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning("GPU check failed: %s", exc)
        return []


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int = 3600,
    log_file: Path | None = None,
    env_override: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess, optionally logging output to a file.

    Uses Popen with start_new_session=True so that on timeout we can kill
    the entire process group (including forked GPU workers, hipcc, etc).

    Returns (returncode, stdout, stderr).
    """
    import signal

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    log.info("Running: %s", " ".join(cmd))
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        elapsed = time.time() - started
        log.info("Command finished in %.1fs with rc=%d", elapsed, proc.returncode)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", encoding="utf-8") as fh:
                fh.write(f"# Command: {' '.join(cmd)}\n")
                fh.write(f"# CWD: {cwd}\n")
                fh.write(f"# Elapsed: {elapsed:.1f}s\n")
                fh.write(f"# Exit code: {proc.returncode}\n\n")
                fh.write("=== STDOUT ===\n")
                fh.write(stdout or "")
                fh.write("\n=== STDERR ===\n")
                fh.write(stderr or "")

        return proc.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        msg = f"TimeoutExpired after {timeout_s}s"
        log.error(msg)
        # Kill entire process group (child + all its descendants)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        # Reap zombie
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(f"# TIMEOUT after {elapsed:.1f}s\n# Command: {' '.join(cmd)}\n")
        return 124, "", msg


def emit_result_json(result: dict[str, Any]) -> None:
    """Print the sentinel-wrapped result JSON to stdout."""
    print(RESULT_SENTINEL_BEGIN)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(RESULT_SENTINEL_END)
