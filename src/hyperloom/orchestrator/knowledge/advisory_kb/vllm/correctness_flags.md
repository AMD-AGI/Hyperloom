# vLLM — Correctness-critical flags (vLLM runs only)

These must be set correctly before perf work, or results/startup break.

## VLLM_ROCM_MOE_PADDING=0 required for ROCm MoE
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#1.3
- impact: correctness
- domain_tags: moe

Set `VLLM_ROCM_MOE_PADDING=0` for ROCm MoE. It conflicts with EPLB if set to 1.

## VLLM_ROCM_USE_AITER_FP4BMM=0 on gfx942 + MXFP4 (startup-crash footgun)
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: correctness/enablement (avoids startup crash)
- domain_tags: quant

With `VLLM_ROCM_USE_AITER=1` on gfx942 (MI300X), an MXFP4 model can crash at startup
(`RuntimeError: MXFP4 quantization is not supported on gfx942` — vLLM has no hardware gate on the FP4
BMM path). Set `VLLM_ROCM_USE_AITER_FP4BMM=0` to disable just the FP4 BMM path while keeping AITER on.

## Keep --quantization mxfp4 for native-MXFP4 models
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: enablement (OOM otherwise)
- accuracy_risk: stripping baked-in mxfp4 -> 4x memory -> OOM
- domain_tags: quant

Models shipped natively in MXFP4 (e.g. gpt-oss-120b) must keep `--quantization mxfp4` — it is how the
model fits. Do not strip it or switch to BF16.

## HF_HUB_OFFLINE=1 for vLLM 0.20.x
- kind: hint
- source: cph-perf-tuning:SKILL.md#gotchas
- impact: enablement
- domain_tags: env

Set `HF_HUB_OFFLINE=1` in the env for vLLM 0.20.x images.
