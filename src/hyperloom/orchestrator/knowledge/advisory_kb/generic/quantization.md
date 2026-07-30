# Generic — Quantization tradeoffs (all frameworks)

## Quantization is the highest-leverage memory/throughput lever
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: throughput+memory
- accuracy_risk: validate per model; prefer a pre-quantized AMD variant when accuracy holds
- domain_tags: quant

FP8 ≈ 2x memory cut; FP4 / MXFP4 ≈ 4x; up to ~1.6x throughput. Prefer a pre-quantized AMD variant
when accuracy holds rather than electively quantizing (and mind the gfx942 MXFP4 emulation caveat in
hardware.md).

## fp8 KV cache halves KV memory but validate accuracy
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: memory (more concurrency / longer context)
- accuracy_risk: some models have fp8-KV bugs — validate
- domain_tags: kv-cache

Setting the KV-cache dtype to fp8 roughly halves KV memory, enabling higher concurrency or longer
context. Validate accuracy per model — some have fp8-KV correctness bugs.
