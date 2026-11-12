# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Bench all GEMV variants in experiments/w4_gemv/variants/ cache-cold at real model shapes,
with a correctness check vs a torch reference. Picks the best per shape. Run on the GPU (parent).
"""
import glob
import importlib.util
import os
import time
import traceback

import torch

G = 128
SHAPES = {
    "o    K5120 N5120":  (5120, 5120),
    "qkv  K5120 N7168":  (5120, 7168),
    "down K13824 N5120": (13824, 5120),
    "gate K5120 N27648": (5120, 27648),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def ref_dequant_matmul(a, w_q, w_s, w_zp, group_size):
    K8, N = w_q.shape
    K = K8 * 8
    shifts = (torch.arange(8, device=w_q.device, dtype=torch.int32) * 4)
    q = ((w_q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N).float()
    gidx = torch.arange(K, device=w_q.device) // group_size
    w = (q - w_zp.float()[gidx]) * w_s.float()[gidx]
    return a.float() @ w


def load_gemv(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemv


def bench_cold(gemv, K, N):
    wbytes = K * N * 0.5
    ncopies = max(3, int(160e6 // wbytes) + 1)
    a = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.1
    copies = [(
        torch.randint(0, 2**31 - 1, (K // 8, N), device="cuda", dtype=torch.int32),
        torch.randn(K // G, N, device="cuda", dtype=torch.float16) * 0.01,
        torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.uint8),
    ) for _ in range(ncopies)]
    wq, ws, wz = copies[0]
    ref = ref_dequant_matmul(a, wq, ws, wz, G)
    c = gemv(a, wq, ws, wz, G)
    rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    if rel > 1e-2:
        return None, rel
    for i in range(ncopies):
        gemv(a, *copies[i % ncopies], G)
    torch.cuda.synchronize()
    iters = ncopies * 8
    t0 = time.perf_counter()
    for i in range(iters):
        gemv(a, *copies[i % ncopies], G)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return wbytes / dt / 1e9, rel


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "variants", "*.py")))
    print(f"Found {len(files)} variants. Cache-cold GB/s (% of 800 DRAM peak):")
    print(f"{'variant':28} " + " ".join(f"{k.split()[0]:>9}" for k in SHAPES))
    results = {}
    for f in files:
        name = os.path.basename(f)
        try:
            gemv = load_gemv(f)
        except Exception as e:
            print(f"{name:28} LOAD FAIL: {repr(e)[:60]}")
            continue
        row = []
        for label, (K, N) in SHAPES.items():
            try:
                gbps, rel = bench_cold(gemv, K, N)
                if gbps is None:
                    row.append(f"WRONG({rel:.0e})")
                else:
                    row.append(f"{gbps:.0f}({gbps/800*100:.0f}%)")
                    results.setdefault(name, {})[label] = gbps
            except Exception:
                row.append("ERR")
        print(f"{name:28} " + " ".join(f"{c:>9}" for c in row))
    print("VARIANTS_BENCH_DONE")
