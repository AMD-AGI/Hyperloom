# ATOM — Correctness-critical flags (ATOM runs only)

These MUST be set correctly before perf work or accuracy breaks.

## ATOM_USE_TRITON_MOE=1 for DeepSeek-V4
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#1.3
- impact: correctness
- accuracy_risk: without it GSM8K drops 95.5% -> ~60%
- domain_tags: moe

Set `ATOM_USE_TRITON_MOE=1` for DeepSeek-V4. Without it, GSM8K accuracy collapses from 95.5% to ~60%.

## AITER_BF16_FP8_MOE_BOUND=0 for PD disaggregation
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#1.3
- impact: correctness
- domain_tags: moe

Set `AITER_BF16_FP8_MOE_BOUND=0` when using prefill/decode disaggregation — it prevents incorrect MoE
boundary handling.

## ATOM_MOE_GU_ITLV=1 for ATOM MoE gate/up interleave
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#1.3
- impact: correctness
- domain_tags: moe

Set `ATOM_MOE_GU_ITLV=1` for correct gate/up interleave in ATOM MoE.

## ATOM_DISABLE_MMAP=true for large models
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#3.3
- impact: enablement
- domain_tags: systems

Set `ATOM_DISABLE_MMAP=true` to disable memory-mapped weights for large models.
