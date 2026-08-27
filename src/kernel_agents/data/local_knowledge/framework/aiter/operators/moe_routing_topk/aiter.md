---
title: moe_routing_topk on aiter — SOTA card
kind: sota_card
operator: moe_routing_topk
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/topk.py
  - ROCm/aiter@b467ce342:aiter/ops/moe_op.py
  - ROCm/aiter@b467ce342:csrc/kernels/topk_softmax_kernels_group.cu
  - ROCm/aiter@b467ce342:csrc/kernels/moe_align_block_size_kernels.cu
  - https://github.com/ROCm/aiter/pull/1909
  - https://github.com/ROCm/aiter/issues/2153
---

# moe_routing_topk × aiter

## TL;DR
> aiter owns the **fused router** on AMD serving stacks: `biased_grouped_topk` / `moe_fused_gate`
> (sigmoid + bias + grouped select in one kernel) and `moe_sorting` (align&sort). On gfx9 it dispatches to a
> **DPP** cross-lane kernel that is ~1.66× faster than CK and supports E=256 / fp32 (the CK path silently
> failed at 256 experts / fp32). Use it for DeepSeek-V3/R1, Kimi-K2, Qwen-MoE — but **assert the scoring
> function matches**: the biased grouped path hardcodes **sigmoid** (aiter #2153), and `num_expert_group`
> must be one of {1,2,4,8} or the group reduction is incomplete.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `biased_grouped_topk` (HIP/DPP) | `aiter/ops/topk.py` → `csrc/kernels/topk_softmax_kernels_group.cu` | gfx942/950, bf16/fp16/fp32 logits | DPP **1.42–1.94× (avg 1.66×)** vs CK @ MI300X gfx942, ROCm 7.2.0, aiter PR #1909, 2026 (vendor/PR-reported) | DeepSeek/Kimi grouped sigmoid routing, E≤256 |
| `moe_fused_gate` | `aiter/ops/topk.py` (`moe_fused_gate`) | gfx942/950 | — | large-token fused gate+select (renorm required) |
| `grouped_topk` (softmax/sigmoid switchable) | `aiter/ops/topk.py:29` (`is_softmax`) | gfx942/950 | — | non-biased grouped routing |
| `topk_softmax` (plain) | `aiter/ops/moe_op.py` (`topk_softmax`); a separate plain topk in `aiter/ops/topk_plain.py` | gfx942/950 | — | non-grouped softmax (Mixtral/Qwen-MoE) |
| `moe_sorting` (align&sort) | `aiter/ops/moe_sorting.py`, `csrc/kernels/moe_align_block_size_kernels.cu` | gfx942/950 | **7× MI300X** (SGLang multi-block rewrite, same algo) | building the grouped-GEMM layout |

Recommend: **DPP path on gfx942/950** for grouped/biased routing; plain `topk_softmax` for non-grouped
softmax models.

### The actual dispatch (on-box `b467ce342`, `aiter/ops/topk.py`)
```python
def biased_grouped_topk(gating_output, correction_bias, topk_weights, topk_ids,
                        num_expert_group, topk_group, need_renorm, routed_scaling_factor=1.0):
    token_num   = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    cu_num = get_cu_num()
    if token_num <= cu_num * 212 or num_experts // num_expert_group > 32:
        return biased_grouped_topk_hip(...)          # one-block-per-token HIP/DPP kernel
    else:
        assert need_renorm, "Renormalization is required for moe_fused_gate."
        return moe_fused_gate(...)                    # multi-token-per-block fused gate
```
So the **small/decode** regime (`token_num ≤ cu_num*212`, i.e. ≤ ~64.4k tokens on 304-CU MI300X, or
`experts/group > 32`) uses the asm/DPP `biased_grouped_topk_hip`; the large-token regime uses
`moe_fused_gate` (which **requires `need_renorm`**).

### Why sigmoid is hardcoded (the #2153 trap)
```cpp
// csrc/kernels/topk_softmax_kernels_group.cu — group kernel inner loop
gating[i] = static_cast<float>(tmp[i]);
gating[i] = 1.0f / (1.0f + expf(-gating[i]));     // <-- sigmoid, ALWAYS
if constexpr(isBiased) { gating[i] += tmp2_f32[i]; }   // + correction_bias
```
The score is unconditionally sigmoid-ed before the bias add — correct for DeepSeek-V3/R1/Kimi (sigmoid
gating), **wrong** for any softmax-gated model routed through this entry point.

## Config space / knobs
- **Arch dispatch**: gfx9 → DPP (no LDS round-trip, `__builtin_amdgcn_readlane` wave reduce), else CK. Auto;
  verify DPP fired for E=256.
- **Token-count dispatch**: `cu_num*212` threshold picks `biased_grouped_topk_hip` vs `moe_fused_gate`.
- **`num_expert_group`**: must be {1,2,4,8} — the kernel uses `THREAD_PER_GRP = ceil(WARP_SIZE/NUM_GRP)` and a
  `switch(num_expert_group){case 8/4/2/1; default: TORCH_CHECK(false)}` launcher; `static_assert(NUM_GRP <=
  WARP_SIZE)`. Off-list values are rejected (or, in older builds, silently incomplete — #2153).
- **`topk_group` / `topk`**: `topk_group ≤ num_expert_group`; selected groups per token.
- **`n_share_experts_fusion`** (`moe_fused_gate`): shared-expert-fusion hook (synthetic shared experts
  appended to the gate); `biased_grouped_topk` passes `0` by default.
- **`need_renorm`, `routed_scaling_factor`, `correction_bias`**: folded into the kernel epilogue.
- **`block_size` (BLOCK_M)** for `moe_sorting`: match the grouped-GEMM tile.

## Numerics / parity
fp32 reduction (the DPP path adds fp32 support the CK path lacked). argmax ties flip benignly. **Hard gates**:
(1) biased path = sigmoid only — assert your model's gating; (2) `num_expert_group ∈ {1,2,4,8}`. `err`-checked
against torch refs in `op_tests`. See [numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `torch.ops.vllm.rocm_aiter_biased_grouped_topk` (registered, PR #17955) selected when
  `is_rocm_aiter_moe_enabled()`; gated by `VLLM_ROCM_USE_AITER=1` (+ `_MOE`).
- SGLang: aiter routing under `SGLANG_USE_AITER=1`.
- aiter custom-op: `moe_fused_gate` has a `gen_moe_fused_gate_fake_tensor` fake impl so it survives
  `torch.compile`.
- `moe_sorting` is called inside `fused_moe` — no separate wiring; tune its `block_size` to the GEMM tile.

## Pitfalls & anti-patterns
- ⚠ **sigmoid hardcode** in the biased path → softmax models get wrong weights (#2153). Assert the gating.
- ⚠ unsupported `num_expert_group` (not 1/2/4/8) → `TORCH_CHECK` abort or silent incomplete group reduction
  (#2153, older builds).
- ⚠ large-token path requires `need_renorm=True` (assert in code) — passing `False` aborts.
- CK fallback caps at E=192 / no fp32 — confirm DPP is engaged for E=256 / fp32.
- `moe_sorting` grid not XCD-aligned → leaves dies idle (the SGLang 7× rewrite is exactly this fix).

## How to verify
`op_tests/test_moe_topk_gating.py` + `test_moeTopkSoftmax.py` parity vs torch; vLLM
`tests/kernels/moe/test_routing.py::test_grouped_topk` with `VLLM_ROCM_USE_AITER=1`; rocprof to confirm the
DPP kernel ran and that `moe_sorting`'s grid is a multiple of XCD=8.

## Worked example (DeepSeek-V3 router, MI300X decode)
E=256, num_expert_group=8, topk_group=4, top-8, sigmoid + correction_bias, M=128 tokens (decode).
1. `token_num=128 ≤ 304*212` → `biased_grouped_topk_hip` (DPP) fires. fp32 scores, E=256 supported.
2. `THREAD_PER_GRP = 64/8 = 8`; 8 lanes/group, 8 groups reduced in-wave.
3. Output `topk_weights/topk_ids` → `moe_sorting(block_size=BLOCK_M)` → fused MoE GEMM.
4. Verify: rocprof shows the DPP kernel (not CK `topk_softmax`); parity vs torch sigmoid+bias reference.
Anti-example: routing a softmax model (Qwen-MoE non-biased) through `biased_grouped_topk` → silently wrong;
use plain `topk_softmax` instead.

## Alternatives / cross-links
moe_routing_topk · hip (raw kernels) · triton ·
vllm_kernels · fused_moe_grouped_gemm (consumer) · moe_dispatch_combine (shares
`moe_sorting`) · `backends/aiter/fmoe.md` · [overview.md](overview.md) ·
[numerics.md](numerics.md).

## Sources
- on-box: `ROCm/aiter@b467ce342:aiter/ops/topk.py` (`biased_grouped_topk` dispatch, `moe_fused_gate` +
  `n_share_experts_fusion`), `aiter/ops/moe_op.py` (`topk_softmax`), `aiter/ops/moe_sorting.py`,
  `csrc/kernels/topk_softmax_kernels_group.cu` (sigmoid + THREAD_PER_GRP),
  `csrc/kernels/moe_align_block_size_kernels.cu`.
- DPP 1.66× / E=256 / fp32 (MI300X, ROCm 7.2.0): https://github.com/ROCm/aiter/pull/1909
- vLLM registration: https://github.com/vllm-project/vllm/pull/17955
- sigmoid hardcode / `num_expert_group` reduction bug: https://github.com/ROCm/aiter/issues/2153
- align&sort 7× MI300X / XCD grid: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
