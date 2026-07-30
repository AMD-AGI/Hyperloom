# vLLM — Tunable levers (vLLM runs only)

Exact knobs for vLLM (ROCm). Reasoning for *when* to pull each is in
`generic/symptom_levers.md`. Values/catalog distilled from cph-perf-tuning `conf/legal_axes.yaml`
(vLLM section) + KNOWLEDGE.md/SKILL.md + the gpt-oss-120b session.

## Throughput-bound: raise batch/scheduling knobs + AITER master switch
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: throughput
- accuracy_risk: fp8 KV cache — validate per model
- domain_tags: framework

Raise `--max-num-batched-tokens` (>2048; try 8192-65536), raise `--max-num-seqs`, raise
`--gpu-memory-utilization` toward 0.95, set `--kv-cache-dtype fp8`. Set `VLLM_ROCM_USE_AITER=1` — the
master switch (off by default) that enables AITER GEMM/RMSNorm/MoE kernels; required even when
`--attention-backend` is set explicitly (that flag alone only changes the attention kernel).

## TPOT-bound: AITER attention backend (use UNIFIED_ATTN for gpt-oss)
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1; session:gpt-oss-120b/20260729T193315Z
- impact: tpot
- accuracy_risk: none expected (kernel swap); gate accuracy after change
- domain_tags: kernel_switch_specialist

Set `--attention-backend`: `ROCM_AITER_FA` is 2.8-4.6x faster TPOT than the legacy `ROCM_ATTN`;
`ROCM_AITER_UNIFIED_ATTN` is within ~5% of FA. Requires `VLLM_ROCM_USE_AITER=1`. IMPORTANT:
`ROCM_AITER_FA` is INCOMPATIBLE with gpt-oss attention-sink models (it crashes / legal_axes rejects
it) — use `ROCM_AITER_UNIFIED_ATTN` there. There is no fixed default: with AITER off vLLM uses the
Triton unified fallback.

## MoE: expert-parallel + AITER MoE + tuned tiles
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1+2.2
- impact: throughput
- accuracy_risk: none expected; gate accuracy after backend change
- domain_tags: kernel_switch_specialist

Set `--enable-expert-parallel` (EP ≤ TP), select `--moe-backend` (`aiter` for the AITER MoE path,
`triton_unfused` for some models), set `VLLM_ROCM_USE_AITER_MOE=1`, and ensure a tuned `fused_moe`
tile config exists (missing → generic Triton → big gap). Consider `--enable-eplb` /`--eplb-config`
and quantized all-reduce `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`. NOTE: `VLLM_ROCM_USE_AITER_MOE`
is NOT always a win — for MXFP4/Triton-MoE models (gpt-oss on MI300X) `AITER_MOE=0` beat `=1` at
conc=4; sweep it.

## Speculative decoding: ngram prompt-lookup (the validated +46% lever)
- kind: hint
- source: session:gpt-oss-120b/20260729T193315Z
- impact: throughput+tpot (validated +46% on gpt-oss-120b vllm mi355x)
- accuracy_risk: none (spec-decode verifies against the base model)
- domain_tags: framework

Use `--speculative-config` with `method=ngram` (e.g. `num_speculative_tokens=3-5`,
`prompt_lookup_max=4-5`, `prompt_lookup_min=1-3`) for low-concurrency latency-bound decode. Pair with
`--no-enable-prefix-caching` (spec-decode is incompatible with prefix caching).

## Scheduling & prefill knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#vllm; SKILL.md
- impact: ttft/tpot
- domain_tags: scheduling

`--max-num-batched-tokens` (central TTFT↔ITL knob), `--max-num-seqs`, `--enable-chunked-prefill`
(V1: on-by-default, effectively non-optional — tune batched-tokens instead of toggling),
`--async-scheduling` (for GPU idle >10%), `--num-scheduler-steps`, `--max-num-partial-prefills`,
`--scheduling-policy`, `--max-model-len`.

## KV-cache knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#vllm
- impact: memory/throughput
- domain_tags: kv-cache

`--kv-cache-dtype` (fp8 halves KV mem), `--gpu-memory-utilization` (→0.95), `--block-size` (AMD
known-good 64; but gpt-oss MI300X empirically `--block-size 16` beat 64 — sweep it),
`--kv-cache-memory-bytes`, `--kv-offloading-size` + `--kv-offloading-backend` (long context, adds
PCIe latency).

## Compilation / CUDA-graph knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#vllm
- impact: ttft/host-overhead
- domain_tags: compilation

`--enforce-eager` (only if required, e.g. DeepSeek-V4), `--compilation-config` (torch.compile /
cudagraph), `--cudagraph-capture-sizes`, `--max-seq-len-to-capture`.

## GEMM auto-tuning env
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: throughput
- accuracy_risk: TunableOp first-run tuning cost; results cached
- domain_tags: kernel_switch_specialist

`TORCH_BLAS_PREFER_HIPBLASLT=1` and `PYTORCH_TUNABLEOP_ENABLED=1` for GEMM-bound linear layers; prefer
tuned AITER GEMM configs.

## Quantization / model knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#vllm; KNOWLEDGE.md#5A.1
- domain_tags: quant

`--quantization` (omit for baked-in quant; only set for non-auto-detectable schemes e.g.
`deepseek_v4_fp8`; keep `mxfp4` for native-MXFP4 models), `--dtype`, `--tokenizer-mode`,
`--reasoning-parser` (`deepseek_v4` for DeepSeek-V4). Collective env: `VLLM_ROCM_QUICK_REDUCE_*`,
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` (model-specific, e.g. MiniMax).
