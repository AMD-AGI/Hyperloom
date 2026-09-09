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

# Re-exported: these used to live here, and both callers and tests import them from this module.
from .aiter_script_map import resolve_aiter_csrc, resolve_aiter_root  # noqa: F401

log = logging.getLogger(__name__)


def sha256_file(path: str | Path | None) -> str:
    """Return the sha256 hex digest of a file, or ``\"\"`` on any I/O error."""
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

# Preferred relative path per tuner, derived from the discovery hints so there is one source of truth.
AITER_TUNER_SCRIPTS = {name: rels[0] for name, rels in _TUNER_SCRIPT_HINTS.items() if rels}

# Environment variable names for tuned config outputs
TUNER_ENV_VARS = {
    "fmoe_ck": "AITER_CONFIG_FMOE",
    "a8w8": "AITER_CONFIG_GEMM_A8W8",
    "a8w8_blockscale": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    "a8w8_bpreshuffle": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    "a8w8_blockscale_bpreshuffle": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    # aiter reads the a4w4 (fp4/mxfp4, gfx950-only) config via AITER_CONFIG_GEMM_A4W4 (jit/core.py); the "_BLOCKSCALE"
    # suffix here was a dead key aiter never reads, which silently dropped all tuned fp4 GEMM configs at serving.
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
    """Locate a specific aiter tuner script by name."""
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


#: Peak resident memory of one aiter ``hipcc`` job. Measured on an MI355X under
#: ROCm 7.2: three CK-tile fused-MoE compiles from
#: ``module_moe_cktile2stages``, 30s each, 1.64 GiB every time to within
#: 0.01 GiB. Rounded up, because the sample is one module's worth of templates
#: and a heavier one is likelier than a lighter one.
#:
#: Rounding up is also what makes the answer safe rather than merely plausible.
#: Running the resulting 44 jobs at once on the same box took the cgroup's
#: ``memory.current`` from 1.0 to 58.4 GiB -- 1.30 GiB a job in aggregate, since
#: the peaks do not coincide -- and left 69.6 GiB under the 128 GiB ceiling,
#: with all 44 compiling clean. The same 1.30 GiB against aiter's own ``-j 188``
#: is about 245 GiB, which is the OOM.
HIPCC_JOB_BYTES = 2 * 1024**3

#: How much of the container's memory allowance a build may claim. The rest is
#: the tuner process that launched it -- torch, a loaded model, GPU buffers --
#: all of it already resident and none of it accounted for by the compiler.
BUILD_MEMORY_SHARE = 0.7

#: Where the kernel publishes the container's memory ceiling, cgroup v2 first.
_CGROUP_MEMORY_LIMITS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def cgroup_memory_limit() -> int | None:
    """The container's memory ceiling in bytes, or None if it has none.

    cgroup v2 writes the literal ``max`` when unlimited; v1 writes a sentinel
    near 2**63. Both mean "ask the host instead", and so does an unreadable or
    unparseable file -- in every one of those cases this returns None and the
    caller changes nothing.
    """
    for path in _CGROUP_MEMORY_LIMITS:
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1's "no limit" is PAGE_COUNTER_MAX rounded to a page; anything past
        # a petabyte is that sentinel rather than a real allowance.
        return value if 0 < value < 2**50 else None
    return None


def build_job_limit(cpus: int | None = None, memory_bytes: int | None = None) -> int | None:
    """How many compile jobs this container's memory can actually hold.

    Returns None when there is nothing to correct: no cgroup ceiling, or one
    roomy enough for a job per CPU.

    The bug this exists for: aiter's JIT sizes its ``ninja -j`` from the CPU
    count, which on a fleet box is the *host's* -- 236 here -- while the memory
    those jobs consume is capped by the *container's* cgroup, 128 GiB here.
    Nothing reconciles the two. Measured, that is ``-j 188`` at 1.64 GiB a job,
    or about 308 GiB against a 128 GiB ceiling, and the build is OOM-killed
    partway through. It is not a slow build or a flaky one; it does not finish.
    """
    if cpus is None:
        # The affinity mask, not the host's core count: a container pinned to a
        # subset is the case where the two differ and the smaller one is true.
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if memory_bytes is None:
        memory_bytes = cgroup_memory_limit()
    if not memory_bytes:
        return None
    fits = int(memory_bytes * BUILD_MEMORY_SHARE) // HIPCC_JOB_BYTES
    capped = max(1, min(cpus, fits))
    return capped if capped < cpus else None


def cap_build_parallelism(env: dict[str, str]) -> dict[str, str]:
    """Set ``MAX_JOBS`` in ``env`` when the container cannot afford one per CPU.

    Deliberately narrow. It does nothing when ``MAX_JOBS`` is already set --
    an operator who chose a number keeps it -- and nothing on a box with no
    cgroup ceiling or a generous one. Mutates and returns ``env`` so callers
    can chain it onto the environment they were building anyway.
    """
    if env.get("MAX_JOBS"):
        return env
    limit = build_job_limit()
    if limit is None:
        return env
    log.info(
        "capping MAX_JOBS at %d: %d CPUs visible but the cgroup allows %.0f GiB, "
        "and one hipcc job holds about %.1f GiB",
        limit,
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
        (cgroup_memory_limit() or 0) / 1024**3,
        HIPCC_JOB_BYTES / 1024**3,
    )
    env["MAX_JOBS"] = str(limit)
    return env


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int = 3600,
    log_file: Path | None = None,
    env_override: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess, optionally logging output to a file."""
    import signal

    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    # Every tuner here shells out to something that ends in an aiter JIT build.
    cap_build_parallelism(env)

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
