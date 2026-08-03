#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""WorldMirror scriptable benchmark driver.

The driver loads HY-World-2.0 once, runs warmup calls, measures reconstruction
latency across selected scenes, and writes an InferenceX-shaped result JSON.
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
                p = matches[0] if matches else root / token
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldMirror reconstruction benchmark")
    parser.add_argument("--worldmirror-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-filename", default="inferencex_result")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=3)
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

    for i in range(max(args.warmup, 0)):
        scene = scenes[i % len(scenes)]
        pipeline(
            str(scene),
            output_path=str(output_root),
            strict_output_path=str(output_root / f"warmup_{i:03d}"),
            target_size=args.target_size,
            log_time=False,
        )

    latencies_ms: list[float] = []
    completed = 0
    last_output = output_root
    start = time.perf_counter()
    for iteration in range(max(args.iterations, 1)):
        for scene in scenes:
            run_output = output_root / f"iter_{iteration:03d}_{scene.name}"
            t0 = time.perf_counter()
            pipeline(
                str(scene),
                output_path=str(output_root),
                strict_output_path=str(run_output),
                target_size=args.target_size,
                log_time=bool(args.profile_dir),
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            completed += 1
            last_output = run_output
    duration = time.perf_counter() - start

    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    throughput = (completed / duration) if duration > 0 else 0.0
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
        "bench_config": {
            "target_size": args.target_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "scene_count": len(scenes),
        },
    }
    out = result_dir / f"{args.result_filename}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
