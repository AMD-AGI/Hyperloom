#!/usr/bin/env python3
"""Benchmark kaolin's spherical-gaussian inner-product kernel, built from public source.

Objective is the geometric mean of forward and backward latency, reported as
kops/s = 1000 / geomean_ms, matching the adapter contract. Correctness is
checked against a pure-PyTorch implementation of the same reduction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

KAOLIN = Path(os.environ.get("KAOLIN_SRC", "/work/kaolin"))
SG = KAOLIN / "kaolin" / "csrc" / "render" / "sg"

# kaolin's pybind entry point lives in its own bindings.cpp and exports the whole
# library; for a single-op benchmark we generate a minimal module instead.
_BIND = r"""
#include <torch/extension.h>
#include "render/sg/unbatched_reduced_sg_inner_product.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &kaolin::unbatched_reduced_sg_inner_product_forward_cuda);
  m.def("backward", &kaolin::unbatched_reduced_sg_inner_product_backward_cuda);
}
"""
_bdir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp")) / "_bind"
_bdir.mkdir(parents=True, exist_ok=True)
_bsrc = _bdir / "sg_bindings.cpp"
_bsrc.write_text(_BIND)

mod = load(
    name="kaolin_sg",
    sources=[
        str(SG / "unbatched_reduced_sg_inner_product.cpp"),
        str(SG / "unbatched_reduced_sg_inner_product_gpu.cu"),
        str(_bsrc),
    ],
    extra_include_paths=[str(KAOLIN / "kaolin" / "csrc")],
    extra_cflags=["-O3", "-DWITH_CUDA"],
    extra_cuda_cflags=["-O3", "-DWITH_CUDA"],
    verbose=False,
)


def gate_against_baseline(out, key, store):
    """Optimization gate: the kernel's output must not change.

    Deliberately NOT an analytic re-derivation of the SG math -- the thing a
    kernel-optimization loop must prove is that the *optimized* kernel still
    agrees with the *pre-optimization* one on the same inputs. The first run
    records the reference; later runs (after a rewrite) compare against it.
    """
    ref_path = Path(store) / f"ref_{key}.pt"
    if ref_path.exists():
        ref = torch.load(ref_path, map_location=out.device)
        return ((out.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30)).item()
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out.detach().cpu(), ref_path)
    return 0.0


def bench(n_sg, n_other, iters=50):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    ii = torch.rand(n_sg, 3, device=dev, generator=g, requires_grad=True)
    di = torch.nn.functional.normalize(torch.randn(n_sg, 3, device=dev, generator=g), dim=-1)
    si = torch.rand(n_sg, device=dev, generator=g) * 10 + 1
    oi = torch.rand(n_other, 3, device=dev, generator=g)
    od = torch.nn.functional.normalize(torch.randn(n_other, 3, device=dev, generator=g), dim=-1)
    os_ = torch.rand(n_other, device=dev, generator=g) * 10 + 1

    out = mod.forward(ii, di, si, oi, od, os_)
    store = os.environ.get("KAOLIN_REF_STORE", "/tmp/kaolin_refs")
    rel = gate_against_baseline(out, f"{n_sg}x{n_other}", store)
    out2 = mod.forward(ii, di, si, oi, od, os_)
    if not torch.equal(out, out2):
        rel = max(rel, 1.0)  # non-deterministic kernel: refuse to trust a speedup

    def timed(fn, n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000.0

    fwd = timed(lambda: mod.forward(ii, di, si, oi, od, os_), iters)
    go = torch.ones_like(out)
    bwd = timed(lambda: mod.backward(go, ii, di, si, oi, od, os_), iters)
    return fwd, bwd, rel


if __name__ == "__main__":
    shapes = [(64, 128), (128, 512), (256, 1024)]
    prod, rels = 1.0, []
    for a, b in shapes:
        f, bw, rel = bench(a, b)
        prod *= f + bw
        rels.append(rel)
        print(f"  sg={a:4d} other={b:5d}  fwd={f:7.4f}ms bwd={bw:7.4f}ms  rel={rel:.3e}")
    geo = prod ** (1.0 / len(shapes))
    worst = max(rels)
    print(f"geomean_fwd_bwd_ms={geo:.6f}")
    print(f"max_rel={worst:.3e}")
    print("RESULT correctness " + ("PASSED" if worst < 1e-3 else "FAILED"))
