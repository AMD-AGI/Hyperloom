#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark Origami picks and create an AITER overlay for proven wins.

The tool is intentionally a standalone process. AITER caches both parsed CSVs
and per-shape lookups, so running out-of-process guarantees that provenance is
computed from the CSV path supplied by the caller. A fallback row is emitted
only after the selected template directly beats AITER's default on that shape.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable


CSV_FIELDS = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
)
DEFAULT_KERNEL_ID = 7


def _origami_enabled() -> bool:
    return os.environ.get("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _load_input(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--input-json must contain a JSON object")
    return data


def _shape_records(value: Any) -> list[dict[str, int]]:
    """Normalize supported inline/path shape payloads to unique M/N/K rows."""
    if isinstance(value, (str, os.PathLike)):
        data = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        data = value
    if isinstance(data, dict):
        data = data.get("shapes", data.get("rows", []))
    if not isinstance(data, list):
        raise ValueError("shape source must be a JSON list or an object with 'shapes'")

    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw in data:
        if not isinstance(raw, dict):
            continue
        try:
            m = int(raw.get("M", raw.get("m")))
            n = int(raw.get("N", raw.get("n")))
            k = int(raw.get("K", raw.get("k")))
        except (TypeError, ValueError):
            continue
        key = (m, n, k)
        if min(key) <= 0 or key in seen:
            continue
        seen.add(key)
        rows.append({"M": m, "N": n, "K": k})
    return rows


def classify_dispatch(
    config: Any,
    *,
    default_kernel_name: str,
) -> tuple[str, str]:
    """Return dispatch provenance and kernel name from an AITER lookup result."""
    if config is None:
        return "fallback", ""
    if not isinstance(config, dict):
        return "invalid_csv", ""
    raw_name = config.get("kernelName", "")
    kernel_name = raw_name if isinstance(raw_name, str) else str(raw_name or "")
    if kernel_name == "":
        return "fallback", ""
    if kernel_name.lower() == "nan":
        # pandas represents an empty CSV cell as NaN; AITER would stringify it
        # to "nan" and fail registry lookup, not take the empty-name fallback.
        return "invalid_csv", kernel_name
    if kernel_name == default_kernel_name:
        return "csv_default_template", kernel_name
    return "csv", kernel_name


def _resolve_aiter_root(explicit: str = "") -> Path:
    candidates: list[Path] = []
    for value in (explicit, os.environ.get("AITER_ROOT_DIR", ""), os.environ.get("AITER_ROOT", "")):
        if value:
            candidates.append(Path(value))
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        candidates.extend((package_dir.parent, package_dir))
    for root in candidates:
        if (root / "csrc" / "ck_gemm_a8w8_blockscale").is_dir():
            return root
        if (root / "aiter" / "csrc" / "ck_gemm_a8w8_blockscale").is_dir():
            return root / "aiter"
    raise FileNotFoundError("could not locate AITER csrc; set AITER_ROOT_DIR")


def _load_kernel_table(aiter_root: Path) -> dict[int, Any]:
    module_path = (
        aiter_root
        / "csrc"
        / "ck_gemm_a8w8_blockscale"
        / "gemm_a8w8_blockscale_instance.py"
    )
    spec = importlib.util.spec_from_file_location("_hyperloom_a8w8_blockscale_instances", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AITER kernel table: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    table = getattr(module, "candidate_kernels_dict", None)
    if not isinstance(table, dict) or DEFAULT_KERNEL_ID not in table:
        raise ValueError("AITER blockscale candidate table is missing or incompatible")
    return {int(key): value for key, value in table.items()}


def _mi_k_for(origami: Any, hardware: Any, mi_m: int, mi_n: int) -> int:
    dtype = origami.string_to_datatype("f8")
    try:
        for mi in hardware.get_valid_matrix_instructions(dtype):
            if int(mi.m) == mi_m and int(mi.n) == mi_n:
                return int(mi.k)
    except Exception:
        pass
    return int(hardware.get_recommended_matrix_instruction(dtype).k)


def _build_configs(
    origami: Any,
    hardware: Any,
    table: dict[int, Any],
    occupancy: int,
) -> list[tuple[int, Any]]:
    configs: list[tuple[int, Any]] = []
    for kernel_id, kernel in table.items():
        mt = (int(kernel.MPerBLOCK), int(kernel.NPerBLOCK), int(kernel.KPerBLOCK))
        mi_m = int(kernel.MPerXDL)
        mi_n = int(kernel.NPerXDL)
        cfg = origami.config_t()
        cfg.mt = origami.dim3_t(*mt)
        cfg.mi = origami.dim3_t(mi_m, mi_n, _mi_k_for(origami, hardware, mi_m, mi_n))
        cfg.occupancy = occupancy
        configs.append((kernel_id, cfg))
    return configs


def _make_problem(origami: Any, m: int, n: int, k: int) -> Any:
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    f8 = origami.string_to_datatype("f8")
    bf16 = origami.string_to_datatype("bf16")
    problem.a_dtype = f8
    problem.b_dtype = f8
    problem.c_dtype = bf16
    problem.d_dtype = bf16
    problem.mi_dtype = f8
    return problem


def rank_shape(
    origami: Any,
    hardware: Any,
    problem: Any,
    configs: list[tuple[int, Any]],
) -> tuple[list[tuple[int, float]], str]:
    """Rank candidates, retrying only when the previous cache policy has none."""
    for hint_mode, cache_a, cache_b in (
        ("base", 0, 0),
        ("nt_b", 0, 4),
        ("nt_a", 4, 0),
    ):
        scored: list[tuple[int, float]] = []
        for kernel_id, cfg in configs:
            cfg.cache_hints_a = cache_a
            cfg.cache_hints_b = cache_b
            try:
                latency = float(origami.compute_total_latency(problem, hardware, cfg))
            except Exception:
                continue
            if math.isfinite(latency) and 0 < latency < 1e300:
                scored.append((kernel_id, latency))
        if scored:
            scored.sort(key=lambda item: item[1])
            return scored, hint_mode
    return [], "none"


def benchmark_is_faster(
    origami_us: float,
    default_us: float,
    *,
    min_speedup: float = 1.0,
) -> bool:
    """Return whether the paired median timing supports selecting Origami."""
    return (
        math.isfinite(origami_us)
        and math.isfinite(default_us)
        and origami_us > 0
        and default_us > 0
        and default_us / origami_us > max(1.0, min_speedup)
    )


def _event_batch_us(torch: Any, fn: Callable[[], Any], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iterations


def _sampled_error(torch: Any, actual: Any, expected: Any, max_elements: int = 65536) -> float:
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    count = min(int(actual_flat.numel()), max_elements)
    if count <= 0:
        return 1.0
    if int(actual_flat.numel()) == count:
        a_sample, e_sample = actual_flat, expected_flat
    else:
        indices = torch.linspace(
            0,
            int(actual_flat.numel()) - 1,
            count,
            device=actual.device,
            dtype=torch.float64,
        ).long()
        a_sample, e_sample = actual_flat[indices], expected_flat[indices]
    close = torch.isclose(
        a_sample.float(),
        e_sample.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    return float(1.0 - close.float().mean().item())


def _benchmark_shape(
    m: int,
    n: int,
    k: int,
    kernel_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Measure Origami's selected kernel and AITER's default on one shape."""
    import aiter
    import torch
    from aiter import dtypes

    warmup = max(1, int(payload.get("benchmark_warmup") or 3))
    iterations = max(1, int(payload.get("benchmark_iterations") or 10))
    rounds = max(3, int(payload.get("benchmark_rounds") or 5))
    max_error = float(payload.get("benchmark_max_error_ratio") or 0.05)
    min_speedup = float(payload.get("benchmark_min_speedup") or 1.0)

    torch.manual_seed(int(payload.get("benchmark_seed") or 42))
    scale_k = (k + 127) // 128
    scale_n = (n + 127) // 128
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(
        dtypes.fp8
    )
    w = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand((m, scale_k), dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand((scale_n, scale_k), dtype=dtypes.fp32, device="cuda")
    selected_out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")
    default_out = torch.empty_like(selected_out)

    def selected_fn():
        return aiter.gemm_a8w8_blockscale_tune(
            x,
            w,
            x_scale,
            w_scale,
            selected_out,
            kernel_id,
            0,
        )

    def default_fn():
        return aiter.gemm_a8w8_blockscale_ck(
            x,
            w,
            x_scale,
            w_scale,
            default_out,
            splitK=0,
            kernelName="",
        )

    try:
        for _ in range(warmup):
            default_fn()
            selected_fn()
        torch.cuda.synchronize()
        default_fn()
        selected_fn()
        torch.cuda.synchronize()
        error_ratio = _sampled_error(torch, selected_out, default_out)
        if error_ratio > max_error:
            return {
                "benchmark_status": "incorrect",
                "benchmark_error_ratio": error_ratio,
                "benchmark_max_error_ratio": max_error,
                "benchmark_use_origami": False,
            }

        selected_samples: list[float] = []
        default_samples: list[float] = []
        for round_index in range(rounds):
            if round_index % 2 == 0:
                default_samples.append(_event_batch_us(torch, default_fn, iterations))
                selected_samples.append(_event_batch_us(torch, selected_fn, iterations))
            else:
                selected_samples.append(_event_batch_us(torch, selected_fn, iterations))
                default_samples.append(_event_batch_us(torch, default_fn, iterations))
        selected_us = float(statistics.median(selected_samples))
        default_us = float(statistics.median(default_samples))
        speedup = default_us / selected_us if selected_us > 0 else 0.0
        return {
            "benchmark_status": "ok",
            "benchmark_selected_us": selected_us,
            "benchmark_default_us": default_us,
            "benchmark_speedup": speedup,
            "benchmark_error_ratio": error_ratio,
            "benchmark_max_error_ratio": max_error,
            "benchmark_min_speedup": max(1.0, min_speedup),
            "benchmark_use_origami": benchmark_is_faster(
                selected_us,
                default_us,
                min_speedup=min_speedup,
            ),
            "benchmark_rounds": rounds,
            "benchmark_iterations": iterations,
        }
    finally:
        del selected_fn, default_fn
        del selected_out, default_out, x, w, x_scale, w_scale
        torch.cuda.empty_cache()


def _aiter_runtime() -> tuple[Callable[..., Any], str, int, str]:
    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

    return (
        get_CKGEMM_config,
        str(get_gfx_runtime()),
        int(get_cu_num()),
        str(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_merged_csv(
    path: Path,
    active_csv: Path,
    candidate_rows: list[dict[str, Any]],
) -> None:
    """Write the complete active config with Origami rows filling missing keys."""
    with active_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        active_fields = list(reader.fieldnames or [])
        active_rows = [dict(row) for row in reader]
    fields = list(active_fields)
    for field in CSV_FIELDS:
        if field not in fields:
            fields.append(field)
    key_fields = [
        field
        for field in ("gfx", "cu_num", "M", "N", "K")
        if field in fields
    ]
    existing = {
        tuple(str(row.get(field, "")) for field in key_fields)
        for row in active_rows
    }
    for row in candidate_rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in existing:
            active_rows.append({field: row.get(field, "") for field in fields})
            existing.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(active_rows)


def select(payload: dict[str, Any]) -> dict[str, Any]:
    if not _origami_enabled():
        return {
            "status": "skipped",
            "candidate": False,
            "reason": "disabled",
        }
    output_value = str(payload.get("output_dir") or "").strip()
    tuned_value = str(payload.get("tuned_csv") or "").strip()
    shape_source = payload.get("shapes", payload.get("shapes_json"))
    if not output_value:
        raise ValueError("output_dir is required")
    output_dir = Path(output_value).resolve()
    shapes = _shape_records(shape_source)
    if not shapes:
        return {"status": "skipped", "reason": "no_shapes", "selected_shapes": 0}
    if str(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"status": "skipped", "reason": "operator_bypass", "selected_shapes": 0}

    import origami

    aiter_root = _resolve_aiter_root(str(payload.get("aiter_root") or ""))
    kernel_table = _load_kernel_table(aiter_root)
    default_name = str(kernel_table[DEFAULT_KERNEL_ID].name)
    resolver, gfx, cu_num, runtime_tuned_csv = _aiter_runtime()
    tuned_csv = Path(tuned_value or runtime_tuned_csv).resolve()
    if not tuned_csv.is_file():
        raise FileNotFoundError(f"active tuned CSV not found: {tuned_csv}")
    hardware = origami.get_hardware_for_device(int(payload.get("device_index") or 0))
    configs = _build_configs(
        origami,
        hardware,
        kernel_table,
        int(payload.get("occupancy") or 2),
    )

    report_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for shape in shapes:
        m, n, k = shape["M"], shape["N"], shape["K"]
        config = resolver(m, n, k, str(tuned_csv))
        source, existing_name = classify_dispatch(
            config,
            default_kernel_name=default_name,
        )
        record: dict[str, Any] = {
            **shape,
            "dispatch_source": source,
            "existing_kernelName": existing_name,
        }
        if source != "fallback":
            report_rows.append(record)
            continue
        problem = _make_problem(origami, m, n, k)
        scored, hint_mode = rank_shape(origami, hardware, problem, configs)
        if not scored:
            record.update({"status": "skipped", "reason": "no_feasible_config"})
            report_rows.append(record)
            continue
        kernel_id, latency = scored[0]
        kernel_name = str(kernel_table[kernel_id].name)
        record.update(
            {
                "status": "ranked",
                "kernelId": kernel_id,
                "kernelName": kernel_name,
                "splitK": 0,
                "hint_mode": hint_mode,
                "pred_latency_cycles": latency,
                "topk": [
                    {
                        "kernelId": kid,
                        "kernelName": str(kernel_table[kid].name),
                        "pred_latency_cycles": pred,
                    }
                    for kid, pred in scored[:5]
                ],
            }
        )
        if kernel_id == DEFAULT_KERNEL_ID:
            record.update(
                {
                    "status": "skipped",
                    "reason": "origami_selected_default",
                    "benchmark_use_origami": False,
                }
            )
            report_rows.append(record)
            continue
        try:
            benchmark = _benchmark_shape(m, n, k, kernel_id, payload)
        except Exception as exc:  # noqa: BLE001 - one shape must not abort the run
            record.update(
                {
                    "status": "skipped",
                    "reason": "benchmark_failed",
                    "benchmark_status": "error",
                    "benchmark_error": f"{exc.__class__.__name__}: {exc}",
                    "benchmark_use_origami": False,
                }
            )
            report_rows.append(record)
            continue
        record.update(benchmark)
        if not benchmark.get("benchmark_use_origami"):
            record.update(
                {
                    "status": "skipped",
                    "reason": (
                        "benchmark_incorrect"
                        if benchmark.get("benchmark_status") == "incorrect"
                        else "not_faster_than_default"
                    ),
                }
            )
            report_rows.append(record)
            continue
        record["status"] = "selected"
        report_rows.append(record)
        selected_us = float(benchmark["benchmark_selected_us"])
        tflops = (2.0 * m * n * k) / (selected_us * 1e6)
        candidate_rows.append(
            {
                "gfx": gfx,
                "cu_num": cu_num,
                "M": m,
                "N": n,
                "K": k,
                "libtype": "ck",
                "kernelId": kernel_id,
                "splitK": 0,
                "us": selected_us,
                "kernelName": kernel_name,
                "tflops": tflops,
                "bw": "",
                "errRatio": benchmark["benchmark_error_ratio"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "origami_a8w8_blockscale.csv"
    merged_csv_path = output_dir / "origami_a8w8_blockscale_merged.csv"
    report_path = output_dir / "origami_a8w8_blockscale_report.json"
    _write_csv(csv_path, candidate_rows)
    _write_merged_csv(merged_csv_path, tuned_csv, candidate_rows)
    fallback_shapes = sum(
        row.get("dispatch_source") == "fallback" for row in report_rows
    )
    report = {
        "status": "ok" if candidate_rows else "skipped",
        "reason": (
            ""
            if candidate_rows
            else (
                "no_measured_origami_wins"
                if fallback_shapes
                else "no_fallback_shapes"
            )
        ),
        "family": "a8w8_blockscale",
        "gfx": gfx,
        "cu_num": cu_num,
        "active_tuned_csv": str(tuned_csv),
        "candidate_csv": str(csv_path),
        "merged_csv": str(merged_csv_path),
        "observed_shapes": len(shapes),
        "fallback_shapes": fallback_shapes,
        "benchmarked_shapes": sum(
            row.get("benchmark_status") in {"ok", "incorrect"}
            for row in report_rows
        ),
        "selected_shapes": len(candidate_rows),
        "rows": report_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **report,
        "report_path": str(report_path),
        "candidate": bool(candidate_rows),
        "tuner": "origami_a8w8_blockscale",
        "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        "env_value": str(merged_csv_path) if candidate_rows else "",
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if not _origami_enabled():
        _json_line(
            {
                "status": "skipped",
                "candidate": False,
                "reason": "disabled",
            }
        )
        return 0
    try:
        result = select(_load_input(args.input_json))
    except Exception as exc:  # noqa: BLE001 - standalone tool returns structured failure
        result = {
            "status": "skipped",
            "candidate": False,
            "reason": "selector_unavailable",
            "error_class": exc.__class__.__name__,
            "error": str(exc),
        }
    _json_line(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
