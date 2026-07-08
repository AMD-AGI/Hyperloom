#!/usr/bin/env python3
"""Reproduce the wrapper exactly (reshape/view/empty_like) under realistic
conditions: near-unity weights, 3D input, non-contiguous input."""
import torch, triton, triton.language as tl

H = 2880; EPS = 1e-5; DT = torch.bfloat16; dev = "cuda"


@triton.jit
def _k(x_ptr, res_ptr, w_ptr, out_ptr, resout_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, BLOCK); mask = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(resout_ptr + off, s, mask=mask)
    var = tl.sum(s * s, axis=0) / n_cols
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + off, s * inv * w, mask=mask)


def fused(x, weight, residual):          # mirrors _fused_add_rmsnorm exactly
    orig_shape = x.shape
    x2d = x.reshape(-1, H).contiguous(); r2d = residual.reshape(-1, H).contiguous(); weight = weight.contiguous()
    n = x2d.shape[0]
    out = torch.empty_like(x2d); rout = torch.empty_like(r2d)
    BLOCK = triton.next_power_of_2(H)
    _k[(n,)](x2d, r2d, weight, out, rout, H, EPS, BLOCK=BLOCK, num_warps=8)
    return out.view(orig_shape), rout.view(orig_shape)


def ref(x, weight, residual):
    xf = x.to(torch.float32) + residual.to(torch.float32)
    rr = xf.to(DT)
    var = xf.pow(2).mean(-1, keepdim=True)
    o = (xf * torch.rsqrt(var + EPS)).to(DT) * weight
    return o, rr


def check(name, x, w, res):
    of, rf = fused(x, w, res); orf, rrf = ref(x, w, res)
    rel = (of.float() - orf.float()).abs().max().item() / (orf.float().abs().max().item() + 1e-6)
    rd = (rf.float() - rrf.float()).abs().max().item()
    print(f"{name:28s} out_rel={rel:.4f} resid_abs={rd:.5f} {'OK' if rel<0.02 and rd<0.05 else '**FAIL**'}")


torch.manual_seed(0)
w_unity = (torch.ones(H, dtype=DT, device=dev) + 0.02 * torch.randn(H, dtype=DT, device=dev))
x = torch.randn(256, H, dtype=DT, device=dev)
res = torch.randn(256, H, dtype=DT, device=dev)
check("near-unity weight 2D", x, w_unity, res)

x3 = torch.randn(4, 64, H, dtype=DT, device=dev)
res3 = torch.randn(4, 64, H, dtype=DT, device=dev)
check("3D input [4,64,H]", x3, w_unity, res3)

# non-contiguous x (slice of a wider tensor)
big = torch.randn(256, H * 2, dtype=DT, device=dev)
xnc = big[:, :H]                         # non-contiguous view, stride 2H
check("non-contiguous x", xnc, w_unity, res)

# large prefill-like row count
xL = torch.randn(4096, H, dtype=DT, device=dev)
resL = torch.randn(4096, H, dtype=DT, device=dev)
check("large N=4096", xL, w_unity, resL)

# ---- op-level speedup: fused Triton kernel vs the eager fp32 reference ----
import time
N = 2048
xt = torch.randn(N, H, dtype=DT, device=dev)
rt = torch.randn(N, H, dtype=DT, device=dev)
print()
for fn, name in ((fused, "fused_triton"), (ref, "eager_ref")):
    for _ in range(10):
        fn(xt, w_unity, rt)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        fn(xt, w_unity, rt)
    torch.cuda.synchronize()
    print(f"{name:14s} {1e6 * (time.time() - t0) / 100:8.1f} us/call  (N={N}, H={H})")
