# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Validate the new sliding_window param in native paged_attention_v1 (S5 correctness gate).
(a) sliding_window=0 must equal full attention (no regression).
(b) sliding_window=W must equal a windowed reference (query at pos seq_len-1 keeps tokens with
    distance (seq_len-1 - token) < W). Also re-check perf with sw=0 vs sw=W."""
import os, sys, math, time
_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp
_ATTN_DIR = wp.build_dir("vw_attn_build", "VLLM_WIN_ATTN_DIR")   # .pyd from build_attn_c.py
wp.add_dll_dirs(_ATTN_DIR)
import torch
torch.ops.load_library(os.path.join(_ATTN_DIR, "vllm_win_attn_C.pyd"))
print("native pyd loaded")

dev, dt = "cuda", torch.float16
NQ, NKV, HD, BS = 16, 8, 256, 16
GRP = NQ // NKV
X = 16 // dt.itemsize
scale = 1.0 / math.sqrt(HD)
one = torch.tensor(1.0, device=dev, dtype=torch.float32)
torch.manual_seed(0)

def pack_and_run(q, K, V, SEQ, W):
    NBLK = (SEQ + BS - 1) // BS
    kc = torch.zeros(NBLK, NKV, HD // X, BS, X, device=dev, dtype=dt)
    vc = torch.zeros(NBLK, NKV, HD, BS, device=dev, dtype=dt)
    for t in range(SEQ):
        b, off = t // BS, t % BS
        for kvh in range(NKV):
            kc[b, kvh, :, off, :] = K[t, kvh, :].view(HD // X, X)
            vc[b, kvh, :, off] = V[t, kvh, :]
    out_n = torch.empty(1, NQ, HD, device=dev, dtype=dt)
    btab = torch.arange(NBLK, device=dev, dtype=torch.int32).view(1, NBLK)
    slens = torch.tensor([SEQ], device=dev, dtype=torch.int32)
    torch.ops._C.paged_attention_v1(out_n, q.view(1, NQ, HD), kc, vc, NKV, scale, btab, slens,
        BS, SEQ, None, "auto", one, one, 0, 0, 0, 0, 0, W)  # last arg = sliding_window
    return out_n.view(NQ, HD).float(), (kc, vc, btab, slens)

def ref(q, K, V, SEQ, W):
    qf, Kf, Vf = q.float(), K.float(), V.float()
    o = torch.empty(NQ, HD, device=dev, dtype=torch.float32)
    pos = torch.arange(SEQ, device=dev)
    for h in range(NQ):
        kvh = h // GRP
        s = (qf[h] @ Kf[:, kvh, :].T) * scale
        if W > 0:
            s = s.masked_fill((SEQ - 1 - pos) >= W, float("-inf"))
        o[h] = torch.softmax(s, dim=-1) @ Vf[:, kvh, :]
    return o

for SEQ, W in [(1000, 0), (1000, 256), (2000, 512), (300, 128), (1023, 1024)]:
    q = torch.randn(NQ, HD, device=dev, dtype=dt) * 0.5
    K = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt) * 0.5
    V = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt) * 0.5
    on, _ = pack_and_run(q, K, V, SEQ, W)
    orf = ref(q, K, V, SEQ, W)
    rel = (on - orf).abs().max() / (orf.abs().max() + 1e-6)
    print(f"seq={SEQ:5d} W={W:5d}: rel={rel.item():.3e}  {'OK' if rel < 2e-2 else 'MISMATCH'}")

# perf: sw=0 vs sw=W should be ~identical (mask is a couple ALU ops)
SEQ = 1024
q = torch.randn(NQ, HD, device=dev, dtype=dt); K = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt)
V = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt)
_, (kc, vc, btab, slens) = pack_and_run(q, K, V, SEQ, 0)
out_n = torch.empty(1, NQ, HD, device=dev, dtype=dt)
def run(W):
    torch.ops._C.paged_attention_v1(out_n, q.view(1, NQ, HD), kc, vc, NKV, scale, btab, slens,
        BS, SEQ, None, "auto", one, one, 0, 0, 0, 0, 0, W)
def bench(W, it=300, wu=60):
    for _ in range(wu): run(W)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): run(W)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6
print(f"perf seq=1024: sw=0 {bench(0):.1f}us  sw=512 {bench(512):.1f}us")
print("MB_SLIDING_DONE")
