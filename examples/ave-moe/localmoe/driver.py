# SPDX-FileCopyrightText: 2026 CausalFlow Inc.
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed LocalMoE correctness and graph timing for the historical prompt replay.

Input generation and timing are adapted from AveLang's
benchmark/moe/bench_dynamic_mxfp4_moe.py at f2c2496e227d838e0edbc6edf878d9799239af85.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from avelang_kernels.amdgpu.local_moe import ExpertWeights, MoeWorkspace, Routing, dynamic_mxfp4_moe, get_2stage_cfgs

TOKENS = (8, 128, 1024)

BENCH_CASES = {
    "gptoss": {
        "model_dim": 3072,
        "inter_dim": 3072,
        "expert": 16,
        "topk": 4,
        "activation": "swiglu",
        "bias_dtype": "bf16",
    },
    "dsv3": {"model_dim": 7168, "inter_dim": 2048, "expert": 33, "topk": 9, "activation": "silu", "bias_dtype": "none"},
    "dsv4": {"model_dim": 7168, "inter_dim": 3072, "expert": 49, "topk": 7, "activation": "silu", "bias_dtype": "none"},
}


def make_inputs(config, tokens, seed):
    torch.manual_seed(seed)
    e, d, i, k = config.experts, config.hidden, config.intermediate, config.topk
    x = torch.randn((tokens, d), dtype=torch.bfloat16, device="cuda")
    w13 = torch.randint(0, 256, (e, 2 * i, d // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, d, i // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(117, 123, (e, 2 * i, d // 32), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(117, 123, (e, d, i // 32), dtype=torch.uint8, device="cuda")
    b1 = torch.randn((e, 2, i), dtype=torch.bfloat16, device="cuda") if config.bias else None
    b2 = torch.randn((e, d), dtype=torch.bfloat16, device="cuda") if config.bias else None
    weights = ExpertWeights.pack(w13, w2, s13, s2, b1, b2)
    top_ids = torch.rand((tokens, e)).topk(k, dim=1).indices
    top_weights = torch.rand((tokens, k)).softmax(-1)
    bm = config.stage1_tile_m
    capacity = ((tokens * k + e * bm - k + bm - 1) // bm) * bm
    ids = torch.full((capacity,), (k << 24) | tokens, dtype=torch.int32)
    rw = torch.zeros(capacity, dtype=torch.float32)
    experts = torch.full((capacity // bm,), -1, dtype=torch.int32)
    begin = 0
    for expert in range(e):
        matches = (top_ids == expert).nonzero()
        count = len(matches)
        ids[begin : begin + count] = matches[:, 0].int() | (matches[:, 1].int() << 24)
        rw[begin : begin + count] = top_weights[matches[:, 0], matches[:, 1]]
        blocks = (count + bm - 1) // bm
        experts[begin // bm : begin // bm + blocks] = expert
        begin += blocks * bm
    routing = Routing(
        ids.cuda(), rw.cuda(), experts.cuda(), torch.tensor([begin, tokens], dtype=torch.int32, device="cuda")
    )
    return x, weights, routing


def measure(fn, warmup, repeat, graph_iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / graph_iters)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def benchmark(warmup: int, iters: int) -> dict[str, float]:
    """Time the fixed cases and optionally capture a separate profiling invocation."""
    case_ms = {}
    for model, parameters in BENCH_CASES.items():
        for tokens in TOKENS:
            config = get_2stage_cfgs(token=tokens, **parameters, arch="gfx950")
            x, weights, routing = make_inputs(config, tokens, seed=42)
            workspace = MoeWorkspace.allocate(x, routing, config)

            def step():
                return dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)

            case = f"{model}_M{tokens}"
            timing = measure(step, warmup, iters, graph_iters=16)
            case_ms[case] = timing["median_us"] / 1000
            print(f"case_ms: {case} {case_ms[case]:.9f}", flush=True)
            if os.environ.get("PROFILE") == "1":
                trace_dir = Path(os.environ["VLLM_TORCH_PROFILER_DIR"])
                trace_dir.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profile:
                    with torch.profiler.record_function(f"avelang.local_moe.{case}"):
                        step()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(trace_dir / f"localmoe_{case}.pt.trace.json.gz"))
    return case_ms


def write_report(path: Path, case_ms: dict[str, float]) -> None:
    """Publish the measured workload and the exact kernel sources used by this process."""
    kernel_dir = Path(dynamic_mxfp4_moe.__code__.co_filename).resolve().parent
    latency_ms = statistics.fmean(case_ms.values())
    report = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "throughput_unit": "unit/s",
        "output_throughput": 1000 / latency_ms,
        "mean_e2el_ms": latency_ms,
        "precision_locked": True,
        "quality_gate": {"passed": True, "reference": "test/examples/gemm/amdgpu/test_moe.py"},
        "bench_summary": {"case_ms": case_ms},
        "bench_config": {
            "operator": "dynamic_mxfp4_moe",
            "kernel_directory": str(kernel_dir),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(kernel_dir.glob("*.py"))
            },
            "driver": str(Path(__file__).resolve()),
            "seed": 42,
            "tokens": TOKENS,
            "cases": BENCH_CASES,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run Ave's driver contract or a complete Hyperloom benchmark with a quality gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-mode", action="store_true")
    mode.add_argument("--report", type=Path, help="Run correctness and timing, then write a Hyperloom workload report")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.report:
        args.report.unlink(missing_ok=True)
    if min(args.warmup, args.iters) < 1:
        parser.error("warmup and iters must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("LocalMoE requires ROCm and a gfx950 GPU")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        raise RuntimeError("Native MXFP4 requires a gfx950 GPU")
    torch.set_num_threads(4)
    if not args.bench_mode:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "test/examples/gemm/amdgpu/test_moe.py"],
            cwd=Path.cwd(),
            check=True,
        )
        print("allclose: True", flush=True)
    if args.bench_mode or args.report:
        case_ms = benchmark(args.warmup, args.iters)
        if args.report:
            write_report(args.report, case_ms)


if __name__ == "__main__":
    main()
