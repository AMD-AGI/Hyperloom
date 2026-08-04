#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""WorldMirror scriptable benchmark driver.

The driver loads HY-World-2.0 once, warms every scene it is about to time,
measures reconstruction latency across the selected scenes, and writes an
InferenceX-shaped result JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_scenes(worldmirror_dir: Path, scenes_spec: str) -> list[Path]:
    """Return selected WorldMirror scene directories or input paths."""
    root = Path(worldmirror_dir)
    if scenes_spec.strip():
        scenes: list[Path] = []
        for token in scenes_spec.replace(",", " ").split():
            p = Path(token)
            if not p.is_absolute():
                candidate = root / "examples" / "worldrecon"
                matches = sorted(candidate.glob(f"*/*{token}*"))
                # Exact name wins: on substring order alone "Building" could
                # resolve to "Tree_Building" and time the wrong scene.
                exact = [m for m in matches if m.name == token]
                p = (exact or matches or [root / token])[0]
            if not p.exists():
                # The pipeline swallows an invalid path as a skip, which would
                # shrink the timed set and silently downgrade the metric scope.
                raise SystemExit(f"WorldMirror scene not found for '{token}': {p}")
            scenes.append(p)
        return scenes

    examples = root / "examples" / "worldrecon"
    found: list[Path] = []
    for scene_dir in sorted(examples.glob("*/*")):
        if scene_dir.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in scene_dir.iterdir()):
            found.append(scene_dir)
    return found


def _depth_arrays(output_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(Path(output_dir).glob("**/depth/*.npy")):
        arrays[str(path.relative_to(output_dir))] = np.asarray(np.load(path), dtype=np.float32)
    return arrays


def _write_quality_ref(output_dir: Path, ref_path: Path) -> dict[str, Any]:
    arrays = _depth_arrays(output_dir)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    return {
        "passed": True,
        "skipped": True,
        "reason": "reference_established",
        "heads": {"depth": {"count": len(arrays)}},
    }


def _compare_quality_ref(output_dir: Path, ref_path: Path, rel_max: float) -> dict[str, Any]:
    arrays = _depth_arrays(output_dir)
    if not ref_path.is_file():
        return {"passed": False, "skipped": True, "reason": "reference_missing"}
    if not arrays:
        return {"passed": False, "skipped": True, "reason": "depth_outputs_missing"}

    rel_values: list[float] = []
    with np.load(ref_path) as ref:
        for key, cur in arrays.items():
            if key not in ref.files:
                return {"passed": False, "reason": f"reference_key_missing:{key}"}
            base = np.asarray(ref[key], dtype=np.float32)
            if base.shape != cur.shape:
                return {"passed": False, "reason": f"shape_mismatch:{key}"}
            denom = float(np.mean(np.abs(base))) + 1e-6
            rel_values.append(float(np.mean(np.abs(cur - base)) / denom))

    max_rel = max(rel_values) if rel_values else float("inf")
    return {
        "passed": bool(max_rel <= rel_max),
        "rel_l1_max": rel_max,
        "heads": {
            "depth": {
                "count": len(rel_values),
                "rel_l1": round(max_rel, 8),
            }
        },
    }


def evaluate_quality_gate(output_dir: Path | str, quality_ref: str, quality_ref_write: str, rel_max: float) -> dict[str, Any]:
    """Establish or compare a depth relative-L1 quality reference."""
    out = Path(output_dir)
    if quality_ref_write:
        return _write_quality_ref(out, Path(quality_ref_write))
    if quality_ref:
        return _compare_quality_ref(out, Path(quality_ref), rel_max)
    return {"passed": True, "skipped": True, "reason": "quality_ref_unset"}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return sorted(values)[idx]


_METRIC_SCOPES = ("inference", "e2e")


def metric_scope() -> str:
    """Which slice of a reconstruction the reported latency covers.

    ``inference`` (default) reports the model forward only. A reconstruction
    also runs sky/edge masking on CPU and serialises point clouds to disk,
    which together dwarf the forward (~1.9s vs ~0.11s) and are untouchable by
    serving-side tuning -- measuring them would dilute a real GPU win below
    run-to-run noise. ``e2e`` restores whole-pipeline wall-clock.
    """
    scope = os.environ.get("WM_METRIC_SCOPE", "inference").strip().lower()
    return scope if scope in _METRIC_SCOPES else "inference"


_SAVE_ARTIFACT_MODES = ("minimal", "full")
_SAVE_FLAG_KEYS = ("save_depth", "save_camera", "save_normal", "save_gs", "save_points")
_MINIMAL_SAVE_FLAGS = {
    "save_depth": True,
    "save_camera": True,
    "save_normal": False,
    "save_gs": False,
    "save_points": False,
}


def save_artifact_mode() -> str:
    """Which reconstruction artifacts reach disk.

    ``minimal`` (default) writes only what is consumed downstream: depth (the
    quality gate) and camera (measured 1.2ms). Writing gs/points/normal costs
    38% of wall-clock and leaks ~75MB per reconstruction without moving the
    forward. ``full`` restores every artifact.
    """
    mode = os.environ.get("WM_SAVE_ARTIFACTS", "minimal").strip().lower()
    return mode if mode in _SAVE_ARTIFACT_MODES else "minimal"


def save_flags(mode: str) -> dict[str, bool]:
    """Pipeline save_* kwargs for a mode, shared by warmup and timed calls."""
    if mode == "full":
        return dict.fromkeys(_SAVE_FLAG_KEYS, True)
    return dict(_MINIMAL_SAVE_FLAGS)


def warmup_plan(scenes: list[Path], requested: int) -> list[Path]:
    """Scenes to warm before timing: every timed scene at least once.

    A shape the process has not met pays a one-time 3-18s ROCm kernel
    compile/autotune inside the timed forward (MIOpen user db + comgr), so
    partial coverage inflates iteration 0: measured 82.5s against a 9.2s steady
    state over 22 scenes. ``requested`` is therefore a floor, not the total.
    """
    if not scenes:
        return []
    plan = list(scenes)
    while len(plan) < max(requested, 0):
        plan.append(scenes[len(plan) % len(scenes)])
    return plan


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    """Mean plus a spread measure; CV is what says whether a 10% win resolves."""
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "stdev_ms": 0.0, "cv_pct": 0.0}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_ms": round(mean, 6),
        "median_ms": round(statistics.median(values), 6),
        "stdev_ms": round(stdev, 6),
        "cv_pct": round(100.0 * stdev / mean, 4) if mean > 0 else 0.0,
    }


def read_stage_timings(run_output: Path | str) -> dict[str, float]:
    """Read the per-stage timings the pipeline writes when log_time is on."""
    path = Path(run_output) / "pipeline_timing.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _capture_profile_trace(
    pipeline,
    scene: Path,
    output_root: Path,
    target_size: int,
    profile_dir: str,
    save_kwargs: dict[str, bool],
) -> None:
    """Capture one reconstruction under torch.profiler for TraceLens.

    Runs after the timed loop so profiling overhead never enters the
    measurement. Best-effort throughout: roofline and the kernel agent need a
    trace, but losing one must not sink results that already succeeded.
    """
    prof_dir = Path(profile_dir)
    try:
        prof_dir.mkdir(parents=True, exist_ok=True)
        import torch
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001 - profiling is never load-bearing
        print(f"[worldmirror][warn] torch.profiler unavailable: {type(exc).__name__}: {exc}")
        return

    # TraceLens resolves each kernel's editable source from the CPU-side Python
    # stack, so with_stack must stay on for the kernel agent to find sources.
    with_stack = os.environ.get("WM_PROFILER_WITH_STACK", "1") != "0"
    prof = None
    try:
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=with_stack,
            with_modules=with_stack,
        ) as prof:
            pipeline(
                str(scene),
                output_path=str(output_root),
                strict_output_path=str(output_root / "profile_capture"),
                target_size=target_size,
                log_time=False,
                **save_kwargs,
            )
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[worldmirror][warn] profiled reconstruction failed: {type(exc).__name__}: {exc}")
        return

    out_trace = prof_dir / "worldmirror-TP-0.pt.trace.json.gz"
    try:
        prof.export_chrome_trace(str(out_trace))
        print(f"[worldmirror] wrote torch profiler trace -> {out_trace}")
    except Exception as exc:  # noqa: BLE001
        print(f"[worldmirror][warn] export_chrome_trace failed: {type(exc).__name__}: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldMirror reconstruction benchmark")
    parser.add_argument("--worldmirror-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-filename", default="inferencex_result")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=3, help="floor; every timed scene is warmed regardless")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--quality-ref", default="")
    parser.add_argument("--quality-ref-write", default="")
    parser.add_argument("--quality-rel-max", type=float, default=0.2)
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--enable-bf16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    worldmirror_dir = Path(args.worldmirror_dir).resolve()
    sys.path.insert(0, str(worldmirror_dir))

    scenes = [Path(args.input_path)] if args.input_path else discover_scenes(worldmirror_dir, args.scenes)
    if not scenes:
        raise SystemExit("No WorldMirror input scenes found")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_root = result_dir / "worldmirror_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    pipeline = WorldMirrorPipeline.from_pretrained(
        args.model_path,
        subfolder=os.environ.get("WM_SUBFOLDER", "HY-WorldMirror-2.0"),
        config_path=os.environ.get("WM_CONFIG_PATH") or None,
        ckpt_path=os.environ.get("WM_CKPT_PATH") or None,
        use_fsdp=bool(args.use_fsdp),
        enable_bf16=bool(args.enable_bf16),
    )

    artifact_mode = save_artifact_mode()
    save_kwargs = save_flags(artifact_mode)

    warmup_scenes = warmup_plan(scenes, args.warmup)
    for i, scene in enumerate(warmup_scenes):
        pipeline(
            str(scene),
            output_path=str(output_root),
            strict_output_path=str(output_root / f"warmup_{i:03d}"),
            target_size=args.target_size,
            log_time=False,
            **save_kwargs,
        )

    scope = metric_scope()
    e2e_ms: list[float] = []
    inference_ms: list[float] = []
    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    per_scene_e2e: dict[str, list[float]] = {}
    per_scene_inference: dict[str, list[float]] = {}
    per_iteration_e2e: list[list[float]] = []
    per_iteration_inference: list[list[float]] = []
    completed = 0
    last_output = output_root
    start = time.perf_counter()
    for iteration in range(max(args.iterations, 1)):
        per_iteration_e2e.append([])
        per_iteration_inference.append([])
        for scene in scenes:
            run_output = output_root / f"iter_{iteration:03d}_{scene.name}"
            t0 = time.perf_counter()
            pipeline(
                str(scene),
                output_path=str(output_root),
                strict_output_path=str(run_output),
                target_size=args.target_size,
                # Always on: the per-stage breakdown is what the reported
                # latency is derived from, and it costs a few timestamps.
                log_time=True,
                **save_kwargs,
            )
            scene_e2e = (time.perf_counter() - t0) * 1000.0
            e2e_ms.append(scene_e2e)
            per_scene_e2e.setdefault(scene.name, []).append(scene_e2e)
            per_iteration_e2e[-1].append(scene_e2e)
            stages = read_stage_timings(run_output)
            for key, value in stages.items():
                stage_totals[key] = stage_totals.get(key, 0.0) + value
                stage_counts[key] = stage_counts.get(key, 0) + 1
            if "inference" in stages:
                scene_inference = stages["inference"] * 1000.0
                inference_ms.append(scene_inference)
                per_scene_inference.setdefault(scene.name, []).append(scene_inference)
                per_iteration_inference[-1].append(scene_inference)
            completed += 1
            last_output = run_output
    duration = time.perf_counter() - start

    # Fall back to wall-clock rather than reporting a partial sample: a mix of
    # the two scopes would be silently incomparable across variants.
    if scope == "inference" and completed and len(inference_ms) == completed:
        latencies_ms, scope_applied = inference_ms, "inference"
        per_scene, per_iteration = per_scene_inference, per_iteration_inference
    else:
        latencies_ms = e2e_ms
        scope_applied = "e2e" if scope == "e2e" else "e2e (pipeline_timing.json unavailable)"
        per_scene, per_iteration = per_scene_e2e, per_iteration_e2e

    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    # Throughput must invert the reported latency, otherwise a scoped metric
    # and a wall-clock rate would disagree about the same run.
    throughput = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
    if args.profile_dir:
        _capture_profile_trace(pipeline, scenes[0], output_root, args.target_size, args.profile_dir, save_kwargs)
    quality_gate = evaluate_quality_gate(
        last_output,
        args.quality_ref,
        args.quality_ref_write,
        args.quality_rel_max,
    )

    result = {
        "framework": "worldmirror",
        "model": args.model_path,
        "workload_kind": "scriptable",
        "throughput_unit": "recon/s",
        "request_throughput": round(throughput, 8),
        "output_throughput": round(throughput, 8),
        "total_token_throughput": 0.0,
        "completed": completed,
        "duration": round(duration, 6),
        "mean_e2el_ms": round(mean_ms, 6),
        "median_e2el_ms": round(statistics.median(latencies_ms), 6) if latencies_ms else 0.0,
        "p99_e2el_ms": round(_percentile(latencies_ms, 99), 6) if latencies_ms else 0.0,
        "std_e2el_ms": round(statistics.pstdev(latencies_ms), 6) if len(latencies_ms) > 1 else 0.0,
        "quality_gate": quality_gate,
        # What the headline latency covers, plus the whole-pipeline number and
        # the stage split, so a report can say "forward -X%, end-to-end -Y%"
        # instead of implying the whole pipeline moved.
        "metric_scope": scope_applied,
        "e2e_mean_ms": round(statistics.fmean(e2e_ms), 6) if e2e_ms else 0.0,
        "stage_timings_mean_s": {
            key: round(total / stage_counts[key], 6)
            for key, total in sorted(stage_totals.items())
        },
        # Per-scene spread decides which scenes can resolve a 10% win: the seven
        # 32-image scenes sit at CV 0.2-3.8%, the light ones at a 13.3% median.
        "per_scene_latency_ms": {
            name: summarize_latencies(values) for name, values in sorted(per_scene.items())
        },
        # A residual cold-start shows up here as an outlying iteration 0.
        "per_iteration_mean_ms": [
            round(statistics.fmean(values), 6) for values in per_iteration if values
        ],
        "bench_config": {
            "target_size": args.target_size,
            "warmup": args.warmup,
            "warmup_requested": args.warmup,
            "warmup_effective": len(warmup_scenes),
            "iterations": args.iterations,
            "scene_count": len(scenes),
            "scenes": [scene.name for scene in scenes],
            "save_artifacts": artifact_mode,
            "save_flags": save_kwargs,
        },
    }
    out = result_dir / f"{args.result_filename}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
