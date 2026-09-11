# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed AttnRes oracle, graph checks and per-case GPU timing for Forge."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from kernel import Score

CASES = [("random", 42, 0.1), ("unit", 19, 1.0), ("near_zero", 29, 1e-5), ("zero", 31, 0.0)]


def inputs(seed, scale):
    torch.manual_seed(seed)
    prefix = torch.randn(64, 7168, device="cuda", dtype=torch.bfloat16) * scale
    bank = torch.randn(64, 8, 7168, device="cuda", dtype=torch.bfloat16) * scale
    weight = torch.randn(7168, device="cuda", dtype=torch.float32) * 0.1
    output = torch.full((64, 16), 12345.0, device="cuda")
    return prefix, bank, weight, output


def check(args):
    prefix, bank, weight, output = args
    vectors = torch.cat((bank.double(), prefix.double().unsqueeze(1)), dim=1)
    expected = (vectors * weight.double()).sum(-1) * torch.rsqrt(vectors.square().mean(-1) + 1e-6)
    actual = output[:, :9].double()
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
    assert torch.all(output[:, 9:] == 12345.0), "inactive output columns changed"
    noise = (actual - expected).norm().item()
    signal = expected.norm().item()
    return 300.0 if noise == 0 else 20 * math.log10(max(signal, 1e-300) / noise)


def correctness(kernel):
    snrs = []
    for _, seed, scale in CASES:
        args = inputs(seed, scale)
        saved = [value.clone() for value in args[:3]]
        kernel(*args)
        torch.cuda.synchronize()
        snrs.append(check(args))
        for value, original in zip(args[:3], saved):
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
    args[2].mul_(0.75)
    args[3].fill_(12345.0)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    snrs.append(check(args))
    del graph
    print(f"SNR: {min(snrs):.3f} dB")
    print("allclose: True")
    print("graph_capture: PASS")


def benchmark(kernel, warmup, iters):
    medians = []
    for name, seed, scale in CASES:
        args = inputs(seed, scale)
        for _ in range(warmup):
            kernel(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                kernel(*args)
        samples = []
        for _ in range(5):
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 1:
        parser.error("warmup and iters must be positive")
    kernel = Score()
    if args.mode == "test":
        correctness(kernel)
    elif args.mode == "bench":
        benchmark(kernel, args.warmup, args.iters)
    else:
        tensors = inputs(42, 0.1)
        kernel(*tensors)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    kernel.close()


if __name__ == "__main__":
    main()
