#!/usr/bin/env python3
"""Ensure TraceLens GPU arch JSON exists before running the TL report.

The public (open-source) TraceLens does not carry MAF values; those live
only in the TraceLens-internal extension, which backfills MAF and adjusts
the final report. On MI355+ the missing MAF makes roofline (and therefore
the whole kernel-optimization loop) fail (#390).

To close that gap for open-source users, when the TraceLens-internal
extension is *not* enabled we run the TraceLens GPU microbenchmark suite to
produce a measured arch spec
(``TraceLens/Agent/Analysis/utils/arch/<platform>.json``). When the internal
extension *is* enabled it backfills MAF itself, so the microbenchmark is
skipped entirely. The benchmark selects an unoccupied GPU and pins
``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES``
only on the microbenchmark subprocess. The microbenchmark runs with
``--warmup 20 --rep 50``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment,misc]

# TraceLens supplies these, but tracelens_analysis.main() pip-installs TraceLens
# *after* this module is first imported. Binding them at import time would cache
# ``None`` on a fresh runtime and make the open-source MI355X path fail even
# after the install succeeds (#390). Resolve them lazily (and cache on first
# success) via the _get_* helpers below. They stay module globals so tests can
# still monkeypatch them.
_collect_arch_jsons = None  # type: ignore[assignment,misc]
check_gpu_idle = None  # type: ignore[assignment,misc]


def _get_collect_arch_jsons():
    """Return TraceLens' arch-JSON collector, importing lazily post-install."""
    global _collect_arch_jsons
    if _collect_arch_jsons is None:
        try:
            from TraceLens.Agent.Analysis.utils.arch_utils import (
                _collect_arch_jsons as _fn,
            )
        except ImportError:
            return None
        _collect_arch_jsons = _fn
    return _collect_arch_jsons


def _get_check_gpu_idle():
    """Return TraceLens' check_gpu_idle, importing lazily post-install."""
    global check_gpu_idle
    if check_gpu_idle is None:
        try:
            from TraceLens.PerfModel.benchmarking.microbench_utils import (
                check_gpu_idle as _fn,
            )
        except ImportError:
            return None
        check_gpu_idle = _fn
    return check_gpu_idle

MICROBENCH_WARMUP = 20
MICROBENCH_REP = 50

_VISIBLE_DEVICE_VARS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def normalize_platform(platform: str) -> str:
    return (platform or "").strip().upper()


def list_candidate_physical_gpus() -> list[int]:
    """Return physical GPU ids currently visible to this process."""
    for var in _VISIBLE_DEVICE_VARS:
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
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return list(range(int(torch.cuda.device_count())))
        except Exception as exc:
            print(
                f"[tracelens_arch_benchmark] Failed to query CUDA devices via torch: {exc}",
                file=sys.stderr,
            )
    return []


def single_physical_gpu_env(
    physical_id: int, *, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    """Return a subprocess env that exposes exactly one physical GPU."""
    env = dict(base_env if base_env is not None else os.environ)
    value = str(physical_id)
    for var in _VISIBLE_DEVICE_VARS:
        env[var] = value
    return env


def select_idle_gpu(
    *, log: Callable[[str], None] | None = None, util_threshold: int = 5
) -> int:
    """Pick an unoccupied GPU for the arch microbenchmark subprocess."""
    check_idle = _get_check_gpu_idle()
    if check_idle is None:
        raise RuntimeError("TraceLens is not installed; cannot check GPU idle state")

    candidates = list_candidate_physical_gpus()
    if not candidates:
        raise RuntimeError(
            "gpu_arch_benchmark found no GPUs; cannot run arch microbenchmark"
        )

    busy_reports: list[str] = []
    for logical_idx, physical_id in enumerate(candidates):
        idle, msg = check_idle(logical_idx, util_threshold=util_threshold)
        if idle:
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
    collect = _get_collect_arch_jsons()
    if collect is None:
        return None

    canonical = normalize_platform(platform)
    if not canonical:
        return None
    for name, path in collect().items():
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


def _sanitize_measured_arch_spec(
    payload: dict,
    *,
    platform: str,
    out_path: Path,
    log: Callable[[str], None],
) -> bool:
    """Drop non-positive MAF entries and reject a structurally-broken spec.

    The TraceLens microbenchmark writes ``0`` for any dtype it could not measure
    (FP8/INT8/MX unsupported on the stack, or a bench that was skipped) and a
    ``0`` bandwidth when the HBM sweep failed. Roofline consumes
    ``max_achievable_tflops[<spec>]`` as a divisor and ``mem_bw_gbps`` as the
    memory ceiling, so a ``0`` would divide-by-zero or yield garbage. Keep only
    positive MAF values (roofline's lookup then returns ``None`` and skips that
    dtype) and hard-fail when the spec is unusable. Returns True if ``payload``
    was modified (caller persists it).
    """
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


def populate_gpu_arch_json(
    *,
    tracelens_root: Path,
    platform: str,
    internal_extension_enabled: bool,
    log: Callable[[str], None],
    run_command: Callable[..., int],
    timeout_s: int = 3600,
    device: int = 0,
) -> Path | None:
    """Ensure a GPU arch JSON is available for roofline, returning its path.

    The microbenchmark is gated on whether the TraceLens-internal extension
    is enabled (#390):

    - When ``internal_extension_enabled`` is True the internal extension
      backfills MAF itself, so we never run the microbenchmark. Any bundled
      spec already on disk is returned as an artifact; otherwise ``None`` is
      returned and the internal extension supplies MAF at report time.
    - When it is False (open-source path) a bundled spec short-circuits, and
      a missing spec triggers the TraceLens microbenchmark on an idle GPU.
    """
    if internal_extension_enabled:
        existing = resolve_arch_json_path(platform)
        if existing is not None and existing.is_file():
            log(
                "gpu_arch_json: internal extension enabled; using bundled spec "
                f"{existing} (MAF backfilled by extension)"
            )
            return existing
        log(
            "gpu_arch_json: internal extension enabled; MAF backfilled by "
            "extension, skipping microbenchmark"
        )
        return None

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
        f"{canonical} and internal extension disabled; running TraceLens "
        f"microbenchmark -> {out_path}"
    )

    physical_id = select_idle_gpu(log=log)

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
            "--output",
            str(out_path),
        ],
        cwd=tracelens_root,
        timeout_s=timeout_s,
        env=single_physical_gpu_env(physical_id),
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
    changed = False
    if payload.get("name") != canonical:
        payload["name"] = canonical
        changed = True
        log(f"gpu_arch_json: patched name field -> {canonical}")

    # Reject / sanitize a spec with 0 (unmeasured) MAF or bandwidth before
    # roofline consumes it as a divisor (#390).
    changed = _sanitize_measured_arch_spec(
        payload, platform=canonical, out_path=out_path, log=log
    ) or changed

    if changed:
        out_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")

    log(f"gpu_arch_json: measured spec ready at {out_path}")
    return out_path
