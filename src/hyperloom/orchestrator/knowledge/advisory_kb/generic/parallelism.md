# Generic — Parallelism & compatibility rules (all frameworks)

## EP ≤ TP always
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#parallelism
- impact: correctness (invalid config otherwise)
- domain_tags: moe

Expert parallelism must never exceed tensor parallelism. Total GPUs = TP × DP; within one DP replica
only TP GPUs are available to shard experts, so EP ≤ TP. Valid: TP=8, DP=2, EP=8. Invalid: TP=2,
DP=4, EP=8.

## MTP / speculative decoding is incompatible with prefix caching
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#parallelism; session:gpt-oss-120b/20260729T193315Z
- impact: correctness/enablement
- domain_tags: spec-decode

When using MTP or speculative decoding, disable prefix caching — they are incompatible. This is why
the validated gpt-oss recipe pairs the speculative config with prefix caching turned off.

## Keep TP within one XGMI island
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: throughput/latency
- domain_tags: parallelism

Keep TP within a single ≤8-GPU XGMI island. For aggregate batch throughput, multiple single-GPU
instances beat high TP (but that maximizes throughput, not per-request latency).
