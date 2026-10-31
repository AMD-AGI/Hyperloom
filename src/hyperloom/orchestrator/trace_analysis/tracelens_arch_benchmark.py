#!/usr/bin/env python3
"""Ensure TraceLens GPU arch JSON exists before running the TL report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from TraceLens.Agent.Analysis.utils.arch_utils import _collect_arch_jsons

from hyperloom.common.visible_devices import (
    HIP_LEVEL_VARS,
    ROCR_LEVEL_VARS,
    effective_mask_tokens,
    parse_device_list,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment,misc]

MICROBENCH_WARMUP = 20
MICROBENCH_REP = 50
#: Exit status of ``TraceLens.PerfModel.benchmarking.microbench`` when its idle gate rejects the GPU.
_MICROBENCH_BUSY_EXIT = 2


def normalize_platform(platform: str) -> str:
    """Normalize a platform/arch name to its canonical upper-case form."""
    return (platform or "").strip().upper()


def _first_mask_ids(level_vars: tuple[str, ...]) -> list[int] | None:
    """Ids named by the first non-empty mask in ``level_vars``; a UUID mask maps to its logical positions."""
    for var in level_vars:
        tokens = effective_mask_tokens(os.environ.get(var))
        if tokens:
            return parse_device_list(tokens) or list(range(len(tokens)))
    return None


def list_candidate_physical_gpus() -> list[int]:
    """List the physical GPU ids currently visible to this process."""
    rocr_ids = _first_mask_ids(ROCR_LEVEL_VARS)
    hip_ids = _first_mask_ids(HIP_LEVEL_VARS)
    if rocr_ids is not None and hip_ids is not None:
        return [rocr_ids[i] for i in hip_ids if i < len(rocr_ids)]
    ids = rocr_ids if rocr_ids is not None else hip_ids
    if ids is not None:
        return ids
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return list(range(int(torch.cuda.device_count())))
        except RuntimeError as exc:
            print(
                f"[tracelens_arch_benchmark] Failed to query CUDA devices via torch: {exc}",
                file=sys.stderr,
            )
    return []


def single_physical_gpu_env(physical_id: int, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess env whose device 0 is ``physical_id``.

    TraceLens' idle gate reads HIP/CUDA masks as absolute physical ids and ignores ROCr-level masks, so the
    child is pinned at the HIP level with every ROCr-level mask cleared.
    """
    env = dict(base_env if base_env is not None else os.environ)
    for var in (*ROCR_LEVEL_VARS, *HIP_LEVEL_VARS):
        env.pop(var, None)
    env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = str(physical_id)
    return env


def resolve_arch_json_path(platform: str) -> Path | None:
    """Resolve the bundled arch JSON path for a platform."""
    canonical = normalize_platform(platform)
    if not canonical:
        return None
    for name, path in _collect_arch_jsons().items():
        if name.upper() == canonical:
            return Path(path)
    return None


#: hyperloom achievable-TFLOPS precision tag -> TraceLens
#: ``max_achievable_tflops`` matrix key.
_HYPERLOOM_DTYPE_TO_MATRIX_KEY: dict[str, str] = {
    "bf16": "matrix_bf16",
    "fp16": "matrix_fp16",
    "fp32": "matrix_fp32",
    "fp8": "matrix_fp8",
    "fp4": "matrix_fp4",
    "int8": "matrix_int8",
}


def build_hyperloom_arch_spec(platform: str) -> dict | None:
    """Build a TraceLens arch spec from hyperloom's own hardware tables."""
    from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS, HW_SPECS_ACHIEVABLE

    key = (platform or "").strip().lower()
    # Achievable first, vendor theoretical when a GPU has no measured entry: the same order the
    # model-level ceiling already resolves peaks in.
    spec = HW_SPECS_ACHIEVABLE.get(key)
    if not isinstance(spec, dict):
        spec = HW_SPECS.get(key)
    if not isinstance(spec, dict):
        return None
    table = spec.get("peak_tflops")
    if not isinstance(table, dict):
        return None
    maf: dict[str, float] = {}
    for tag, matrix_key in _HYPERLOOM_DTYPE_TO_MATRIX_KEY.items():
        val = table.get(tag)
        if isinstance(val, (int, float)) and val > 0:
            maf[matrix_key] = float(val)
    mem_bw = spec.get("hbm_bw_gbps")
    if not maf or not isinstance(mem_bw, (int, float)) or mem_bw <= 0:
        return None
    out: dict = {"name": normalize_platform(platform), "mem_bw_gbps": float(mem_bw)}
    mem_gb = spec.get("hbm_gb")
    if isinstance(mem_gb, (int, float)) and mem_gb > 0:
        out["memory_gb"] = float(mem_gb)
    out["max_achievable_tflops"] = maf
    return out


def write_hyperloom_arch_spec(tracelens_root: Path, platform: str, log: Callable[[str], None]) -> Path | None:
    """Write hyperloom's achievable arch spec into the TraceLens arch dir."""
    spec = build_hyperloom_arch_spec(platform)
    if spec is None:
        return None
    out_path = default_arch_output_path(tracelens_root, platform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=4) + "\n", encoding="utf-8")
    log(
        f"gpu_arch_json: wrote hyperloom achievable spec for {spec['name']} "
        f"-> {out_path} (mem_bw={spec['mem_bw_gbps']} GB/s, "
        f"dtypes={sorted(spec['max_achievable_tflops'])})"
    )
    return out_path


def default_arch_output_path(tracelens_root: Path, platform: str) -> Path:
    """Return the default arch-spec JSON path for a platform."""
    canonical = normalize_platform(platform)
    return tracelens_root / "TraceLens/Agent/Analysis/utils/arch" / f"{canonical}.json"


def _sanitize_measured_arch_spec(
    payload: dict,
    *,
    platform: str,
    out_path: Path,
    log: Callable[[str], None],
) -> bool:
    """Drop non-positive MAF entries and reject a structurally-broken spec."""
    maf = payload.get("max_achievable_tflops")
    if not isinstance(maf, dict) or not maf:
        raise RuntimeError(
            f"measured arch spec {out_path} has no max_achievable_tflops; the GPU "
            "microbenchmark produced a spec roofline cannot use"
        )

    kept: dict = {}
    dropped: list[str] = []
    for key, value in maf.items():
        try:
            positive = float(value) > 0.0
        except (TypeError, ValueError):
            positive = False
        if positive:
            kept[key] = value
        else:
            dropped.append(key)

    if not kept:
        raise RuntimeError(
            f"measured arch spec {out_path} has no positive max_achievable_tflops "
            f"values (all of {sorted(maf)} measured as 0); the GPU microbenchmark "
            "likely failed -- refusing to emit a spec roofline cannot use"
        )

    mem_bw = payload.get("mem_bw_gbps")
    try:
        mem_bw_ok = mem_bw is not None and float(mem_bw) > 0.0
    except (TypeError, ValueError):
        mem_bw_ok = False
    if not mem_bw_ok:
        raise RuntimeError(
            f"measured arch spec {out_path} has non-positive mem_bw_gbps "
            f"({mem_bw!r}); roofline's memory ceiling would divide by zero -- "
            "the HBM bandwidth benchmark likely failed"
        )

    if dropped:
        payload["max_achievable_tflops"] = kept
        log(
            f"gpu_arch_json: dropped non-positive MAF keys {sorted(dropped)} from "
            f"{platform} spec (roofline skips these dtypes rather than dividing by 0)"
        )
        return True
    return False


def _run_microbench_on_idle_gpu(
    tracelens_root: Path,
    out_path: Path,
    *,
    log: Callable[[str], None],
    run_command: Callable[..., int],
    timeout_s: int,
) -> None:
    """Run the arch microbenchmark on the first candidate GPU its own idle gate accepts."""
    candidates = list_candidate_physical_gpus()
    if not candidates:
        raise RuntimeError("gpu_arch_benchmark found no GPUs; cannot run arch microbenchmark")
    cmd = [
        sys.executable,
        "-m",
        "TraceLens.PerfModel.benchmarking.microbench",
        "--device",
        "0",
        "--warmup",
        str(MICROBENCH_WARMUP),
        "--rep",
        str(MICROBENCH_REP),
        "--output",
        str(out_path),
    ]
    for physical_id in candidates:
        rc = run_command(cmd, cwd=tracelens_root, timeout_s=timeout_s, env=single_physical_gpu_env(physical_id))
        if rc == 0:
            log(f"gpu_arch_json: microbenchmark ran on GPU {physical_id}")
            return
        if rc != _MICROBENCH_BUSY_EXIT:
            raise RuntimeError(f"gpu arch microbenchmark failed with exit code {rc}; see log for details")
        log(f"gpu_arch_json: GPU {physical_id} is busy; trying the next candidate")
    raise RuntimeError(f"gpu_arch_benchmark found no unoccupied GPU among {candidates}")


def internal_extension_arch_json(platform: str, *, log: Callable[[str], None]) -> Path | None:
    """The bundled GPU arch JSON when the internal extension backfills MAF, or ``None`` when none is bundled."""
    existing = resolve_arch_json_path(platform)
    if existing is not None and existing.is_file():
        log(f"gpu_arch_json: internal extension enabled; using bundled spec {existing} (MAF backfilled by extension)")
        return existing
    log("gpu_arch_json: internal extension enabled; MAF backfilled by extension, skipping microbenchmark")
    return None


def populate_gpu_arch_json(
    *,
    tracelens_root: Path,
    platform: str,
    log: Callable[[str], None],
    run_command: Callable[..., int],
    timeout_s: int = 3600,
) -> Path:
    """Ensure a GPU arch JSON is available for roofline without the internal extension, returning its path."""
    existing = resolve_arch_json_path(platform)
    if existing is not None and existing.is_file():
        log(f"gpu_arch_json: using bundled spec {existing}")
        return existing

    canonical = normalize_platform(platform)
    if not canonical:
        raise RuntimeError("target platform is empty; cannot resolve or generate gpu arch JSON")

    # Prefer a live microbenchmark (measured MAF); fall back to hyperloom's in-repo achievable spec when no idle GPU
    # is available or the microbenchmark cannot produce a usable spec.
    out_path = default_arch_output_path(tracelens_root, canonical)
    mb_error: RuntimeError | None = None
    try:
        log(
            "gpu_arch_json: no bundled spec for "
            f"{canonical} and internal extension disabled; running TraceLens "
            f"microbenchmark -> {out_path}"
        )
        _run_microbench_on_idle_gpu(tracelens_root, out_path, log=log, run_command=run_command, timeout_s=timeout_s)
        if not out_path.is_file():
            raise RuntimeError(f"gpu arch microbenchmark finished but output is missing: {out_path}")

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"gpu arch microbenchmark output is not a JSON object: {out_path}")
        changed = False
        if payload.get("name") != canonical:
            payload["name"] = canonical
            changed = True
            log(f"gpu_arch_json: patched name field -> {canonical}")

        # Reject / sanitize a spec with 0 (unmeasured) MAF or bandwidth before roofline consumes it as a divisor.
        changed = _sanitize_measured_arch_spec(payload, platform=canonical, out_path=out_path, log=log) or changed

        if changed:
            out_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")

        log(f"gpu_arch_json: measured spec ready at {out_path}")
        return out_path
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        # A microbenchmark that overruns ``timeout_s`` surfaces as subprocess.TimeoutExpired, not RuntimeError, so
        # catching RuntimeError alone let the most common failure escape and killed the whole trace_analyze instead of
        # taking the documented fallback below.
        mb_error = exc if isinstance(exc, RuntimeError) else RuntimeError(f"{type(exc).__name__}: {exc}")
        log(f"gpu_arch_json: microbenchmark unusable ({exc}); falling back to hyperloom achievable spec")

    hyperloom_spec = write_hyperloom_arch_spec(tracelens_root, canonical, log)
    if hyperloom_spec is not None:
        return hyperloom_spec
    assert mb_error is not None
    raise mb_error
