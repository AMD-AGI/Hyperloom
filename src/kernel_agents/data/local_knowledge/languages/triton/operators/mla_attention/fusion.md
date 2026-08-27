---
title: mla_attention — fusion
kind: operator_overview
operator: mla_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
---

# mla_attention — fusion

MLA's biggest "fusion" is structural: **matrix absorption** folds the KV up-projection weights into Q and
the output so two projection GEMMs disappear into the attention. Beyond that, the wins are launch
amortization and the pre-attention latent-write.

## Matrix absorption = weight-into-attention fusion
Standard MLA decode would: down-project to latent → up-project latent to full K/V (`Wuk`,`Wuv`) → MHA.
Absorption folds `Wuk` into `q_nope` and `Wuv` into the attention output, so the up-projection GEMMs
vanish and the layer runs as MQA on the latent. This is the fusion that delivers the 17× — it removes both
the up-projection FLOPs and the full-K/V bandwidth. See [numerics.md](numerics.md) for why it's exact.

## The MLA decode pipeline
```
new token ─► [down-proj to latent + RoPE] ─► [latent-cache write + quant] ─► [absorbed MLA decode (splitKV)] ─► [reduce] ─► [out-proj (absorbed Wuv folded)]
                          └──── fuse pre ────┘                                 └──── persistent / fused-decode ────┘
```

### Pre-attention: fused latent write
The new token's compressed latent + decoupled RoPE key must be written to the cache. aiter fuses the
**KV-down-projection-adjacent RoPE + latent-cache write + (fp8) quant**; `SGLANG_ROCM_FUSED_DECODE_MLA=1`
fuses the **MLA decode with RoPE** so the rotary is applied inside the decode kernel rather than as a
separate pass.

### Launch fusion: persistent decode
`mla_decode_fwd` has a **persistent mode** (`work_meta_data`/`work_indptr`/`work_info_set` set, or
`SGLANG_AITER_MLA_PERSIST=1`) — the kernel stays resident across the decode loop, amortizing launch at
small batch. This is the decode-side analog of unified attention.

### splitKV + reduce
The decode kernel produces per-split partials combined by a reduce (Triton stage-2
`_fwd_kernel_stage2_asm` / `mla_prefill_reduce` for prefill). The split count is auto-tuned (see
[tuning.md](tuning.md)).

## Backend support
| fusion | aiter (asm) | Triton MLA | CK MLA |
|---|---|---|---|
| matrix absorption (MQA on latent) | **yes** | yes | yes |
| fused decode + RoPE | yes (`SGLANG_ROCM_FUSED_DECODE_MLA`) | partial | — |
| persistent decode | **yes** (`SGLANG_AITER_MLA_PERSIST`) | no | — |
| fp8 latent/KV quant on write | yes (`q_scale`/`kv_scale`) | yes | — |

## Where fusion moves e2e
Absorption is the FLOP+bandwidth win (the 17× kernel); persistent + fused-decode + the latent-write
fusion are the launch/bandwidth wins that give the 1.2–1.6× TPOT over Triton MLA at serving. fp8 latent
halves bandwidth but is an accuracy gate.

## Sources
- absorption / fused decode / persistent mode / splitKV reduce: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py` ; https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- `SGLANG_ROCM_FUSED_DECODE_MLA` / `SGLANG_AITER_MLA_PERSIST`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py ; `backends/sglang_kernels/attention_backends.md`
