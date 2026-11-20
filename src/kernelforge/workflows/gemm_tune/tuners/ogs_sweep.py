# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure GPT-OSS routed BF16 × MXFP4 projections on the bundled matmul_ogs kernel."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

from .ogs_runtime import _vendor_triton_kernels_path

_DEVICE = "cuda"
_TEXT_ENCODING = "utf-8"
_ROCM_GFX1201_RDNA_VERSION = 4
_MXFP4_PACKED_ELEMENTS_PER_BYTE = 2
_MXFP4_SCALE_BLOCK = 32
_CORRECTNESS_TOLERANCE = 0.02


class _SweepRuntime(NamedTuple):
    torch: Any
    opt_flags: Any
    matmul_ogs: Callable[..., Any]
    make_routing_data: Callable[..., Any]
    wrap_torch_tensor: Callable[..., Any]
    fp4: Any
    precision_config: Any
    fused_activation: Any
    fn_specs: Any
    swiglu_fn: Any


def _runtime() -> _SweepRuntime:
    _vendor_triton_kernels_path()
    import torch
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, PrecisionConfig, matmul_ogs
    from triton_kernels.matmul_ogs_details import opt_flags
    from triton_kernels.swiglu import swiglu_fn
    from triton_kernels.tensor import FP4, wrap_torch_tensor
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import make_routing_data

    return _SweepRuntime(
        torch,
        opt_flags,
        matmul_ogs,
        make_routing_data,
        wrap_torch_tensor,
        FP4,
        PrecisionConfig,
        FusedActivation,
        FnSpecs,
        swiglu_fn,
    )


def _projection_call(
    runtime: _SweepRuntime, case: dict[str, Any], tokens: int, experts: int, top_k: int
) -> Callable[[], Any]:
    """Build the same routing, packed weights, and projection arguments used by GPT-OSS serving."""
    torch = runtime.torch
    n, k = case["n"], case["k"]
    ids = torch.rand((tokens, experts), device=_DEVICE).topk(top_k, dim=-1).indices
    gates = torch.softmax(torch.randn((tokens, top_k), device=_DEVICE), dim=-1)
    routing, gather, scatter = runtime.make_routing_data(ids, gates, experts)
    packed = torch.randint(
        0, 256, (experts, n, k // _MXFP4_PACKED_ELEMENTS_PER_BYTE), dtype=torch.uint8, device=_DEVICE
    )
    scales = torch.randint(120, 125, (experts, n, k // _MXFP4_SCALE_BLOCK), dtype=torch.uint8, device=_DEVICE)
    weights = runtime.wrap_torch_tensor(packed.transpose(-1, -2), dtype=runtime.fp4)
    weight_scale = runtime.wrap_torch_tensor(scales.transpose(-1, -2))
    precision = runtime.precision_config(weight_scale=weight_scale)
    up = case["projection"] == "up"
    input_rows = tokens if up else tokens * top_k
    inputs = torch.randn((input_rows, k), dtype=torch.bfloat16, device=_DEVICE) * 0.1
    bias = torch.zeros((experts, n), dtype=torch.float32, device=_DEVICE)
    options: dict[str, Any] = {"routing_data": routing, "precision_config": precision}
    if up:
        options["gather_indx"] = gather
        options["fused_activation"] = runtime.fused_activation(
            runtime.fn_specs("swiglu", runtime.swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (1.702, 7.0),
        )
    else:
        options["scatter_indx"] = scatter
        options["gammas"] = routing.gate_scal
    return lambda: runtime.matmul_ogs(inputs, weights, bias, **options)


@contextmanager
def _override_amd_flags(module: Any, flags: dict[str, int]) -> Iterator[None]:
    original = module.make_default_opt_flags_amd

    def replacement(*args: Any, **kwargs: Any) -> Any:
        return replace(original(*args, **kwargs), **flags)

    module.make_default_opt_flags_amd = replacement
    try:
        yield
    finally:
        module.make_default_opt_flags_amd = original


def _bench_ms(torch: Any, call: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            call()
    timings = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iters)
    return statistics.median(timings)


def _tensor(value: Any) -> Any:
    return value.data if hasattr(value, "data") else value


def _measure_case(runtime: _SweepRuntime, case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    experts, top_k = config["experts"], config["top_k"]
    call = _projection_call(runtime, case, case["tokens"], experts, top_k)
    expected = _tensor(call())
    baseline_ms = _bench_ms(runtime.torch, call, warmup=config["warmup"], iters=config["iters"])
    best: dict[str, Any] | None = None
    failures: list[dict[str, str]] = []
    for flags in config["search_space"]:
        try:
            with _override_amd_flags(runtime.opt_flags, flags):
                runtime.torch.testing.assert_close(
                    _tensor(call()), expected, rtol=_CORRECTNESS_TOLERANCE, atol=_CORRECTNESS_TOLERANCE
                )
                tuned_ms = _bench_ms(runtime.torch, call, warmup=config["warmup"], iters=config["iters"])
        except Exception as exc:  # noqa: BLE001 - one invalid config is recorded without losing the sweep
            traceback.print_exc()
            failures.append({"flags": str(flags), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if best is None or tuned_ms < best["tuned_ms"]:
            best = {"flags": flags, "tuned_ms": tuned_ms}
    return {**case, "default_ms": baseline_ms, "best": best, "failures": failures}


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding=_TEXT_ENCODING))
    gpu_id = str(config.get("gpu_id") or "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    os.environ.setdefault("HIP_VISIBLE_DEVICES", gpu_id)
    runtime = _runtime()
    if not runtime.torch.cuda.is_available() or runtime.opt_flags.get_rdna_version() != _ROCM_GFX1201_RDNA_VERSION:
        raise RuntimeError("matmul_ogs BF16 × MXFP4 tuning requires an RDNA4 GPU")
    gpu_name = runtime.torch.cuda.get_device_name()
    if "R9700" not in gpu_name.upper():
        raise RuntimeError("matmul_ogs BF16 × MXFP4 tuning requires an R9700 product")
    source_sha = hashlib.sha256(Path(runtime.opt_flags.__file__).read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(1, int(config["timeout_s"]) - 30)
    for case in config["cases"]:
        if time.monotonic() >= deadline:
            break
        outcome = _measure_case(runtime, case, config)
        measurements.append(outcome)
        best = outcome["best"]
        if best is None:
            continue
        speedup = outcome["default_ms"] / best["tuned_ms"]
        if speedup <= 1.0 + float(config["min_improvement_pct"]) / 100.0:
            continue
        rows.append(
            {key: outcome[key] for key in ("m", "n", "k", "tokens", "projection")}
            | {
                "flags": best["flags"],
                "default_ms": outcome["default_ms"],
                "tuned_ms": best["tuned_ms"],
                "speedup": speedup,
            }
        )
    result = {
        "status": "ok" if rows else "no_improvement",
        "schema_version": 1,
        "gpu_target": "gfx1201",
        "gpu_name": gpu_name,
        "opt_flags_sha256": source_sha,
        "rows": rows,
        "measurements": measurements,
    }
    (config_path.parent / "sweep_results.json").write_text(json.dumps(result, indent=2), encoding=_TEXT_ENCODING)
    return result


if __name__ == "__main__":
    try:
        output = run(Path(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001 - process boundary records the failure and traceback
        traceback.print_exc()
        output = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(output), flush=True)
    raise SystemExit(0 if output["status"] in ("ok", "no_improvement") else 1)
