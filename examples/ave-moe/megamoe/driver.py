# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed Kimi K3 MegaMoE EP8 workload, preserving the original suite's cases and timing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch.distributed as dist
from aiter.fused_moe import fused_topk
from avelang_kernels.amdgpu.mega_moe import (
    MegaMoeWorkspace,
    dynamic_mxfp4_mega_moe,
    get_2stage_cfgs,
    pack_expert_weights,
)

WORLD_SIZE = 8
SEED = 20260920
TOKENS = (8, 32, 128)
EXPERTS, TOPK, HIDDEN, INTERMEDIATE = 896, 16, 3584, 3072
REFERENCE = "test/examples/gemm/amdgpu/mega_moe_worker.py"


@dataclass
class Inputs:
    """One rank's activations and routing buffers."""

    x: torch.Tensor
    logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor

    def route(self) -> None:
        fused_topk(self.x, self.logits, TOPK, True, topk_ids=self.ids, topk_weights=self.weights)


class Workload:
    """One rank's fixed weights and shared workspace for all three token counts."""

    def __init__(self) -> None:
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        spec = importlib.util.spec_from_file_location("hyperloom_mega_moe_reference", Path.cwd() / REFERENCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {REFERENCE}")
        self.reference_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.reference_module
        spec.loader.exec_module(self.reference_module)
        config = get_2stage_cfgs(
            8, WORLD_SIZE, EXPERTS, TOPK, HIDDEN, INTERMEDIATE, activation="kimi_situ", bias_dtype="none"
        )
        self.raw = self.reference_module.make_raw_weights(self.profile(0), config.local_experts, self.rank, self.device)
        self.packed = pack_expert_weights(config, *self.raw)
        self.workspace = MegaMoeWorkspace.allocate(config, group=dist.new_group(backend="gloo"))

    def profile(self, tokens: int):
        return self.reference_module.Profile("kimi", tokens, EXPERTS, TOPK, HIDDEN, INTERMEDIATE, "kimi_situ", False)

    def inputs(self, tokens: int, seed: int = SEED) -> Inputs:
        torch.manual_seed(seed + self.rank)
        torch.cuda.manual_seed_all(seed + self.rank)
        return Inputs(
            torch.randn((tokens, HIDDEN), dtype=torch.float32, device=self.device).to(torch.bfloat16),
            torch.randn((tokens, EXPERTS), dtype=torch.float32, device=self.device),
            torch.empty((tokens, TOPK), dtype=torch.int32, device=self.device),
            torch.empty((tokens, TOPK), dtype=torch.float32, device=self.device),
        )

    def candidate(self, inputs: Inputs) -> torch.Tensor:
        inputs.route()
        output = dynamic_mxfp4_mega_moe(inputs.x, self.packed, inputs.ids, inputs.weights, workspace=self.workspace)
        # The next call reuses the workspace's output buffer. Preserve both parity outputs.
        return output.clone()

    def reference(self, inputs: Inputs) -> torch.Tensor:
        inputs.route()
        return self.reference_module.reference(
            self.profile(len(inputs.x)), inputs.x, inputs.ids, inputs.weights, self.raw, WORLD_SIZE
        )

    def check(self, tokens: int) -> float:
        first, second = self.inputs(tokens), self.inputs(tokens, SEED + 7919)
        first_got, second_got = self.candidate(first), self.candidate(second)
        # Rebuild inputs so an in-place candidate cannot change the reference's inputs.
        first_want = self.reference(self.inputs(tokens))
        second_want = self.reference(self.inputs(tokens, SEED + 7919))
        torch.cuda.synchronize()
        snr = torch.tensor(
            min(snr_db(first_got, first_want), snr_db(second_got, second_want)),
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        value = snr.item()
        if self.rank == 0:
            print(f"kimi_M{tokens} SNR: {value:.2f} dB", flush=True)
        if value < 30:
            raise RuntimeError(f"kimi_M{tokens}: worst-rank SNR {value:.2f} dB is below 30 dB")
        return value

    def benchmark(self, tokens: int, warmup: int, iters: int) -> float:
        inputs = self.inputs(tokens)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.candidate(inputs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.candidate(inputs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iters):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            # Collective bookkeeping stays outside the timed region.
            elapsed = torch.tensor(start.elapsed_time(end), dtype=torch.float64, device=self.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        if os.environ.get("PROFILE") == "1":
            self.capture_trace(inputs, tokens)
        return statistics.median(samples)

    def capture_trace(self, inputs: Inputs, tokens: int) -> None:
        trace_dir = Path(os.environ["VLLM_TORCH_PROFILER_DIR"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profile:
            with torch.profiler.record_function(f"avelang.mega_moe.kimi_M{tokens}"):
                self.candidate(inputs)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(trace_dir / f"megamoe_M{tokens}_rank{self.rank}.pt.trace.json.gz"))


def snr_db(candidate: torch.Tensor, expected: torch.Tensor) -> float:
    """Reject nonfinite or mismatched outputs before computing signal-to-noise ratio."""
    if candidate.shape != expected.shape:
        return -math.inf
    got, want = candidate.float(), expected.float()
    if not torch.isfinite(got).all() or not torch.isfinite(want).all():
        return -math.inf
    noise, signal = torch.sum((want - got) ** 2).item(), torch.sum(want**2).item()
    if noise == 0:
        return math.inf
    return 10 * math.log10(signal / noise) if signal > 0 else -math.inf


def write_report(path: Path, case_ms: dict[str, float], worst_snr: float) -> None:
    """Publish the collective workload's quality gate, timing and source identity."""
    kernel_dir = Path(dynamic_mxfp4_mega_moe.__code__.co_filename).resolve().parent
    latency_ms = statistics.fmean(case_ms.values())
    report = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "throughput_unit": "unit/s",
        "output_throughput": 1000 / latency_ms,
        "mean_e2el_ms": latency_ms,
        "precision_locked": True,
        "quality_gate": {
            "passed": True,
            "reference": REFERENCE,
            "minimum_snr_db": 30,
            "worst_snr_db": worst_snr if math.isfinite(worst_snr) else "inf",
        },
        "bench_summary": {"case_ms": case_ms},
        "bench_config": {
            "operator": "dynamic_mxfp4_mega_moe",
            "kernel_directory": str(kernel_dir),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(kernel_dir.glob("*.py"))
            },
            "driver": str(Path(__file__).resolve()),
            "seed": SEED,
            "tokens": TOKENS,
            "expert_parallel_size": WORLD_SIZE,
            "experts": EXPERTS,
            "topk": TOPK,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "activation": "kimi_situ",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_worker(args: argparse.Namespace) -> None:
    """Own the process group lifetime and publish results only from rank zero."""
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError(f"MegaMoE must run on {WORLD_SIZE} ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        workload = Workload()
        worst_snr = math.inf
        if not args.bench_mode:
            worst_snr = min(workload.check(tokens) for tokens in TOKENS)
            if workload.rank == 0:
                print("allclose: True", flush=True)
        if args.bench_mode or args.report:
            case_ms = {}
            for tokens in TOKENS:
                case = f"kimi_M{tokens}"
                case_ms[case] = workload.benchmark(tokens, args.warmup, args.iters)
                if workload.rank == 0:
                    print(f"case_ms: {case} {case_ms[case]:.9f}", flush=True)
            if args.report and workload.rank == 0:
                write_report(args.report, case_ms, worst_snr)
    finally:
        dist.destroy_process_group()


def main() -> int:
    """Launch all eight ranks for Ave validation/timing or a Hyperloom benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-mode", action="store_true")
    mode.add_argument("--report", type=Path, help="Run correctness and timing, then write a Hyperloom workload report")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if min(args.warmup, args.iters) < 1:
        parser.error("warmup and iters must be positive")
    if not torch.version.hip or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"MegaMoE requires {WORLD_SIZE} visible ROCm GPUs")
    for index in range(WORLD_SIZE):
        if not torch.cuda.get_device_properties(index).gcnArchName.startswith("gfx950"):
            raise RuntimeError(f"GPU {index} must be gfx950 (MI350/MI355)")
    torch.set_num_threads(4)
    if "LOCAL_RANK" not in os.environ:
        if args.report:
            args.report.unlink(missing_ok=True)
        # Keep torchrun in the driver's process group so Ave cancellation reaches every rank.
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={WORLD_SIZE}",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            check=False,
        ).returncode
    run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
