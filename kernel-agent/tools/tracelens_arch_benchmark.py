#!/usr/bin/env python3
"""Ensure TraceLens GPU arch JSON exists before running the TL report.

When ``TraceLens/Agent/Analysis/utils/arch/<platform>.json`` (and any
``TL_EXTENSION`` override) is missing, run the TraceLens GPU microbenchmark
suite to produce a measured arch spec. Selects an unoccupied GPU (pinning it
via ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` when multiple are
present) before launching the benchmark. The microbenchmark runs with
``--warmup 20 --rep 50 --allow-busy``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable


RunCommandFn = Callable[..., int]
LogFn = Callable[[str], None]

VISIBLE_DEVICE_VARS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def normalize_platform(platform: str) -> str:
    return (platform or "").strip().upper()


def count_visible_gpus() -> int:
    """Return the number of GPUs visible to this process."""
    candidates = list_candidate_physical_gpus()
    if candidates:
        return len(candidates)
    return 0


def list_candidate_physical_gpus() -> list[int]:
    """Return physical GPU ids currently visible to this process."""
    for var in VISIBLE_DEVICE_VARS:
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        parts = [part.strip() for part in val.split(",") if part.strip()]
        if not parts:
            continue
        try:
            return [int(part) for part in parts]
        except ValueError:
            return list(range(len(parts)))
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def pin_single_physical_gpu(physical_id: int) -> None:
    """Expose exactly one physical GPU to downstream CUDA/HIP consumers."""
    value = str(physical_id)
    os.environ["HIP_VISIBLE_DEVICES"] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    os.environ["ROCR_VISIBLE_DEVICES"] = value


def select_idle_gpu(*, log: LogFn | None = None, util_threshold: int = 5) -> int:
    """Pick an unoccupied GPU and pin it as the only visible device."""
    from TraceLens.PerfModel.benchmarking.microbench_utils import check_gpu_idle

    candidates = list_candidate_physical_gpus()
    if not candidates:
        raise RuntimeError(
            "gpu_arch_benchmark found no GPUs; cannot run arch microbenchmark"
        )

    busy_reports: list[str] = []
    for logical_idx, physical_id in enumerate(candidates):
        idle, msg = check_gpu_idle(logical_idx, util_threshold=util_threshold)
        if idle:
            pin_single_physical_gpu(physical_id)
            if log is not None:
                if len(candidates) == 1:
                    log(f"gpu_arch_json: using idle GPU {physical_id} ({msg})")
                else:
                    log(
                        "gpu_arch_json: selected idle GPU "
                        f"{physical_id} from candidates {candidates} ({msg})"
                    )
            return physical_id
        busy_reports.append(f"GPU {physical_id}: {msg}")

    raise RuntimeError(
        "gpu_arch_benchmark found no unoccupied GPU among "
        f"{candidates}. {'; '.join(busy_reports)}"
    )


def resolve_arch_json_path(platform: str) -> Path | None:
    """Return the bundled arch JSON path for ``platform``, if present."""
    from TraceLens.Agent.Analysis.utils.arch_utils import _collect_arch_jsons

    canonical = normalize_platform(platform)
    if not canonical:
        return None
    for name, path in _collect_arch_jsons().items():
        if name.upper() == canonical:
            return Path(path)
    return None


def default_arch_output_path(tracelens_root: Path, platform: str) -> Path:
    canonical = normalize_platform(platform)
    return (
        tracelens_root
        / "TraceLens/Agent/Analysis/utils/arch"
        / f"{canonical}.json"
    )


MICROBENCH_WARMUP = 20
MICROBENCH_REP = 50


def ensure_gpu_arch_json(
    *,
    tracelens_root: Path,
    platform: str,
    log: LogFn,
    run_command: RunCommandFn,
    timeout_s: int = 3600,
    device: int = 0,
) -> Path:
    """Return arch JSON path, running the microbenchmark when bundled spec is missing."""
    existing = resolve_arch_json_path(platform)
    if existing is not None and existing.is_file():
        log(f"gpu_arch_json: using bundled spec {existing}")
        return existing

    canonical = normalize_platform(platform)
    if not canonical:
        raise RuntimeError(
            "target platform is empty; cannot resolve or generate gpu arch JSON"
        )

    out_path = default_arch_output_path(tracelens_root, canonical)
    log(
        "gpu_arch_json: no bundled spec for "
        f"{canonical}; running TraceLens microbenchmark -> {out_path}"
    )

    select_idle_gpu(log=log)

    rc = run_command(
        [
            sys.executable,
            "-m",
            "TraceLens.PerfModel.benchmarking.microbench",
            "--device",
            str(device),
            "--warmup",
            str(MICROBENCH_WARMUP),
            "--rep",
            str(MICROBENCH_REP),
            "--allow-busy",
            "--output",
            str(out_path),
        ],
        cwd=tracelens_root,
        timeout_s=timeout_s,
    )
    if rc != 0:
        raise RuntimeError(
            f"gpu arch microbenchmark failed with exit code {rc}; see log for details"
        )
    if not out_path.is_file():
        raise RuntimeError(
            f"gpu arch microbenchmark finished but output is missing: {out_path}"
        )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("name") != canonical:
        payload["name"] = canonical
        out_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
        log(f"gpu_arch_json: patched name field -> {canonical}")

    log(f"gpu_arch_json: measured spec ready at {out_path}")
    return out_path
