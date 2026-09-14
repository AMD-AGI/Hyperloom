# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Independent vector-add oracle, graph checks and fixed-input GPU timings."""

from __future__ import annotations

import argparse
import statistics

import torch

from kernel import N, VectorAdd

CASES = [("random", 42, 0.1), ("unit", 19, 1.0), ("near_zero", 29, 1e-5), ("zero", 31, 0.0)]


def inputs(seed, scale):
    torch.manual_seed(seed)
    a = torch.randn(N, device="cuda", dtype=torch.float32) * scale
    b = torch.randn(N, device="cuda", dtype=torch.float32) * scale
    return a, b, torch.full_like(a, float("nan"))


def check(args):
    a, b, output = args
    torch.testing.assert_close(output, a + b, rtol=0, atol=0)
    return 300.0


def correctness(kernel):
    snrs = []
    for _, seed, scale in CASES:
        args = inputs(seed, scale)
        saved = [value.clone() for value in args[:2]]
        kernel(*args)
        torch.cuda.synchronize()
        snrs.append(check(args))
        for value, original in zip(args[:2], saved):
            assert torch.equal(value, original), "input changed"

    args = inputs(8128, 0.1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(*args)
    stream.synchronize()
    snrs.append(check(args))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        kernel(*args)
    stream.synchronize()
    args[0].mul_(0.5)
    args[1].mul_(0.25)
    args[2].fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    snrs.append(check(args))
    del graph
    print(f"SNR: {min(snrs):.3f} dB")
    print("allclose: True")
    print("graph_capture: PASS")


def benchmark(kernel, warmup, iters, case="", repeat=1):
    medians = []
    for name, seed, scale in CASES:
        if case and name != case:
            continue
        args = inputs(seed, scale)
        for _ in range(warmup):
            kernel(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                kernel(*args)
        samples = []
        for _ in range(5 * repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iters)
        median = statistics.median(samples)
        medians.append(median)
        print(f"case_ms: {name} {median:.9f}")
        del graph
    print(f"mean_ms: {statistics.mean(medians):.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "bench", "profile"), default="test")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--bench-case", choices=[name for name, _, _ in CASES], default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 1 or args.repeat < 1:
        parser.error("warmup, iters and repeat must be positive")
    kernel = VectorAdd()
    if args.profile_run or args.mode == "profile":
        tensors = inputs(42, 0.1)
        for _ in range(5):
            kernel(*tensors)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        kernel(*tensors)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    elif args.bench_mode or args.mode == "bench":
        benchmark(kernel, args.warmup, args.iters, args.bench_case, args.repeat)
    else:
        correctness(kernel)
    torch.cuda.synchronize()
    kernel.close()


if __name__ == "__main__":
    main()
