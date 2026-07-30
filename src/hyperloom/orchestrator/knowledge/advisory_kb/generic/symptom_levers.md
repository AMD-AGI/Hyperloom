# Generic — Symptom → Lever model (all frameworks)

Framework-agnostic tuning *reasoning*: diagnose the binding constraint first, then pull the
matching lever. The exact knob names live in each framework's `levers.md`; this file is the
"which category to reach for" map. Sweep, don't assume — the optimum is workload-specific.

## Throughput-bound: GPU saturated but tokens/sec below target
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: throughput
- accuracy_risk: fp8 KV cache — validate accuracy per model (some fp8-KV bugs)
- domain_tags: framework

Raise the batch-size / scheduling knobs (max batched tokens, max sequences), raise GPU memory
utilization toward 0.95, and consider fp8 KV cache. Enable the vendor kernel path (on ROCm/AITER
that is a master switch — see the framework's levers). For aggregate batch throughput prefer
multiple single-GPU instances over high TP (maximizes throughput, not per-request latency; keep TP
within one ≤8-GPU XGMI island).

## TTFT-bound: prefill latency high
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: ttft
- domain_tags: framework

Raise max-batched-tokens (more prefill per batch), ensure no queueing/preemption, tune the
compilation / CUDA-graph capture, and consider multi-step scheduling. Note: enabling AITER + fp8-KV
recovers throughput but can REGRESS TTFT — watch the tradeoff.

## TPOT / ITL-bound: per-token decode latency high
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: tpot
- domain_tags: kernel_switch_specialist

Lower max-batched-tokens (fewer prefills interrupting decode), pick a faster decode attention
backend, and lower concurrency. On ROCm the AITER attention backends are 2.8-4.6x faster TPOT than
the legacy path — see the framework's levers for the exact backend name (and its caveats).

## OOM / preemption: KV cache exhausted, requests recomputed
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: stability
- domain_tags: framework

Raise GPU-memory-utilization or TP; lower max-sequences / max-model-len. Detect via the engine's
preemption metrics.

## MoE models: routed-expert dispatch dominates
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1+2.2
- impact: throughput
- domain_tags: kernel_switch_specialist

Enable expert-parallel (EP ≤ TP), select the MoE backend, and ensure a tuned fused-MoE tile config
exists for (num_experts, intermediate_size, GPU, dtype) — a missing config silently falls back to
generic Triton tiles, a major throughput gap. Consider quantized all-reduce for collective-skew.

## GEMM-bound: large linear layers dominate
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1; session:gpt-oss-120b/20260729T193315Z
- impact: throughput
- accuracy_risk: TunableOp first-run tuning cost; results cached
- domain_tags: kernel_switch_specialist

Prefer hipBLASLt over hipBLAS and enable PyTorch TunableOp GEMM auto-tuning; prefer tuned vendor GEMM
configs. Note the empirical ceiling: plain Triton GEMM loses to tuned Tensile/asm on bf16 decode-GEMV
(single-launch cross-CU split-K in asm is not expressible in Triton), so do NOT expect a hand-authored
Triton dense GEMM to beat the vendor baseline on memory-bound decode shapes.

## Speculative decoding is the biggest lever for low-concurrency decode
- kind: hint
- source: session:gpt-oss-120b/20260729T193315Z; cph-perf-tuning:KNOWLEDGE.md#parallelism
- impact: throughput+tpot
- accuracy_risk: none (spec-decode verifies drafts against the base model)
- domain_tags: framework

When decode is latency-bound (small batch, memory-bound), speculative decoding validates several
drafted tokens per forward pass — often the single largest win. Prefer prompt-lookup / ngram methods
when no draft model is available. CAVEAT: MTP / speculative decoding is incompatible with prefix
caching — disable prefix caching when using it. Each framework spells the knob differently (see its
levers.md).

## Bottleneck → axis quick table (from trace symptoms)
- kind: hint
- source: cph-perf-tuning:SKILL.md#bottleneck-heuristics
- domain_tags: freeform

MoE-heavy + low TFLOP/s + memory-bound → MoE backend + generate tuned MoE tile configs. AllReduce
skew (multi-rank) → quantized all-reduce, check the all-to-all backend. Prefill-bound (high TTFT, low
TPOT) → raise max-batched-tokens, switch attention backend. Decode-bound (high TPOT, low TTFT) → lower
max-sequences, enable speculative decoding if prefix caching is already off. GPU idle > 10% → async
scheduling, more CUDA-graph capture sizes. Short-kernel overhead spike → chunked prefill, larger
max-batched-tokens. Memory-bound (HBM saturated) → fused ops, fp8 KV cache.
