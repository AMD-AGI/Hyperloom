# Generic — Correctness-critical flags concept + operator options (all frameworks)

## Set correctness-critical flags BEFORE any perf work
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#1.3
- impact: correctness
- domain_tags: framework

Some flags must be set before tuning or the model produces wrong results — which makes every perf
number meaningless. The specific flags are framework-specific (see each framework's
`correctness_flags.md`). Process: read the framework's model file for custom ops (aiter_ops,
rocm_aiter, triton, flash_attn, ck_ops) and add any model-specific correctness flags found; verify
accuracy per operator with a quick prompt check + a GSM8K/MMLU subset for MoE-heavy models.

## Operator → library option map (enumerate before assuming AITER)
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#2.1
- domain_tags: kernel_switch_specialist

For each hot operator, enumerate all kernel/library options rather than assuming one path:
RMSNorm (AITER fused HIP/ASM, Triton, torch); Attention (FA-2, FA-3, AITER HIP ASM, Triton ATTN, CK,
torch SDPA); KV-cache quant (AITER fp8 cache-write, Triton, torch); MoE routing (AITER HIP, Triton
tuned per-model tiles, Triton generic, CK grouped GEMM); Shared expert (AITER fusion, separate linear,
torch); GEMM (AITER, CK grouped GEMM, split-K skinny GEMM for small M, Triton, torch); GDN /
GatedDeltaNet (FlyDSL Triton kernel — the only option, torch fallback); SiLU×gate (AITER SwiGLU,
Triton, torch); RoPE (AITER fused+cache-quant, Triton, torch); all-reduce (RCCL, vendor custom AR
kernels).

## Tuned MoE tile configs — key format and how to generate
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#2.2
- impact: throughput
- domain_tags: kernel_switch_specialist

Tuned fused-MoE tile configs are keyed `E=<num_experts>,N=<intermediate_size>,device_name=<GPU>,
dtype=<quant_type>`. A missing config for the current model+GPU falls back to generic Triton tiles —
a major throughput gap. Generate with the framework's `benchmark_moe.py --num-experts <E>
--intermediate-size <N> --dtype <fp8|fp4|bf16> --device <GPU>` and commit it.
