# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Which K/V cache format keeps the attention output intact? (issue #28)

Usage: python kv_quant_schemes.py <dump.pt>   (a layer dump written by kv_scale_probe.py with
VLLM_KV_PROBE_DUMP=<path>, VLLM_KV_PROBE_DUMP_LAYER=<n>)

Which K/V cache format keeps Qwen2.5 attention intact? Real q/k/v of one layer (dumped by
run/kv_scale_probe.py) through fp32 reference attention vs the same with K (and/or V) run
through a quantize -> dequantize round trip. Error = attention output of the LAST M query
positions (decode-like, every key visible) vs the fp32 reference, max-rel / rms-rel."""
import math, os, sys, torch

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "kv_dump.pt")
d = torch.load(path)
q, k, v = d["q"].cuda().float(), d["k"].cuda().float(), d["v"].cuda().float()   # [T,Hq,D] [T,Hk,D]
T, Hq, D = q.shape; Hk = k.shape[1]; G = Hq // Hk
M = 256
scale = 1.0 / math.sqrt(D)
print(f"layer {d['layer']} ctx {d['ctx']} | q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} | "
      f"max|Q| {q.abs().max():.1f} max|K| {k.abs().max():.1f} max|V| {v.abs().max():.1f} | median |K| {k.abs().median():.2f}")

FP8 = torch.float8_e4m3fn
def to_fp8(x, sat=448.0):            # Triton store semantics: saturate, never NaN
    return x.clamp(-sat, sat).to(FP8).float()

def attn(kq, vq):
    """fp32 attention of the last M queries over all T keys (causal)."""
    qh = q[-M:].transpose(0, 1)                          # [Hq, M, D]
    kh = kq.repeat_interleave(G, dim=1).transpose(0, 1)  # [Hq, T, D]
    vh = vq.repeat_interleave(G, dim=1).transpose(0, 1)
    s = torch.matmul(qh, kh.transpose(1, 2)) * scale     # [Hq, M, T]
    qpos = torch.arange(T - M, T, device=q.device).unsqueeze(1)
    kpos = torch.arange(T, device=q.device).unsqueeze(0)
    s = s.masked_fill(kpos > qpos, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), vh)   # [Hq, M, D]

ref = attn(k, v)
def err(o):
    diff = (o - ref)
    return f"max-rel {diff.abs().max() / ref.abs().max():.3f}  rms-rel {diff.norm() / ref.norm():.3f}"

# ---- K schemes (V kept exact) ----
schemes = {}
schemes["fp8 e4m3, per-tensor scale 1.0 (vLLM default)"] = (to_fp8(k), v)
s_t = k.abs().max() / 448.0
schemes["fp8 e4m3, per-tensor full-range scale"] = (to_fp8(k / s_t) * s_t, v)
s_c = k.abs().amax(dim=0, keepdim=True) / 448.0 + 1e-12                      # [1,Hk,D]
schemes["fp8 e4m3, per-(head,channel) scale"] = (to_fp8(k / s_c) * s_c, v)
mu = k.mean(dim=0, keepdim=True)
schemes["fp8 e4m3, per-(head,channel) mean removed, scale 1.0"] = (to_fp8(k - mu) + mu, v)
s_r = k.abs().amax(dim=2, keepdim=True) / 448.0 + 1e-12                      # per token, per head
schemes["fp8 e4m3, per-token scale"] = (to_fp8(k / s_r) * s_r, v)
r = k - mu
s_rc2 = r.abs().amax(dim=0, keepdim=True) / 448.0 + 1e-12
schemes["fp8 e4m3, per-(head,channel) mean removed + per-(head,channel) scale"] = (to_fp8(r / s_rc2) * s_rc2 + mu, v)
s_rt = r.abs().amax(dim=2, keepdim=True) / 448.0 + 1e-12
schemes["fp8 e4m3, per-(head,channel) mean removed + per-token scale"] = (to_fp8(r / s_rt) * s_rt + mu, v)
def int8_asym(x, dim):
    lo = x.amin(dim=dim, keepdim=True); hi = x.amax(dim=dim, keepdim=True)
    s = (hi - lo) / 255.0 + 1e-12
    return torch.round((x - lo) / s).clamp(0, 255) * s + lo
schemes["int8 asym, per-(head,channel) scale+zero"] = (int8_asym(k, 0), v)
s_rt1 = r.abs().max() / 448.0
schemes["fp8 e4m3, per-(head,channel) mean removed + per-tensor full-range scale (store-side only)"] = (to_fp8(r / s_rt1) * s_rt1 + mu, v)
schemes["fp8 e4m3, mean removed (not added back) + per-tensor full-range scale"] = (to_fp8(r / s_rt1) * s_rt1, v)
schemes["fp8 e5m2, per-tensor scale 1.0"] = (k.clamp(-57344, 57344).to(torch.float8_e5m2).float(), v)
def int8_sym(x, dim):
    s = x.abs().amax(dim=dim, keepdim=True) / 127.0 + 1e-12
    return torch.round(x / s).clamp(-127, 127) * s
schemes["int8, per-(head,channel) scale"] = (int8_sym(k, 0), v)
schemes["int8, per-token scale"] = (int8_sym(k, 2), v)
def int4_asym(x, dim):
    lo = x.amin(dim=dim, keepdim=True); hi = x.amax(dim=dim, keepdim=True)
    s = (hi - lo) / 15.0 + 1e-12
    return torch.round((x - lo) / s).clamp(0, 15) * s + lo
schemes["int4 asym, per-(head,channel) scale+zero (KIVI-style K)"] = (int4_asym(k, 0), v)
# Hadamard rotation (KVarN's first ingredient): rotate K and Q by the same orthonormal H,
# dot products are invariant, outliers get spread over all channels before quantizing.
def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).cuda()
H = hadamard(D)
k_rot = k @ H
def attn_rot(kq_rot, vq):
    global q
    q0 = q; q = q @ H
    try:
        return attn(kq_rot, vq)
    finally:
        q = q0
schemes["fp8 e4m3 after Hadamard rotation, per-tensor scale 1.0"] = ("rot", to_fp8(k_rot), v)
s_rc = k_rot.abs().amax(dim=0, keepdim=True) / 448.0 + 1e-12
schemes["fp8 e4m3 after Hadamard rotation, per-(head,channel) scale"] = ("rot", to_fp8(k_rot / s_rc) * s_rc, v)
schemes["int4 asym per-(head,channel) after Hadamard rotation"] = ("rot", int4_asym(k_rot, 0), v)
schemes["K fp16 exact, V fp8 e4m3 per-tensor 1.0"] = (k, to_fp8(v))
schemes["fp8 e4m3 both K and V, per-tensor 1.0 (vLLM --kv-cache-dtype fp8)"] = (to_fp8(k), to_fp8(v))

print(f"\nattention output error vs fp32 reference (last {M} queries over {T} keys):")
for name, val in schemes.items():
    if val[0] == "rot":
        o = attn_rot(val[1], val[2])
    else:
        o = attn(val[0], val[1])
    print(f"  {name:62}: {err(o)}")
