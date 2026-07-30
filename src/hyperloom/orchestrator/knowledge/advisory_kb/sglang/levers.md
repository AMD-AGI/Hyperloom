# SGLang — Tunable levers (SGLang runs only)

Exact knobs for SGLang on ROCm. Reasoning for *when* to pull each is in
`generic/symptom_levers.md`. Catalog from cph-perf-tuning `conf/legal_axes.yaml` (sglang section).
Doc-sourced — not yet run-validated on a measured SGLang session; grow with empirical findings.

## KV-cache / memory knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: memory/throughput
- domain_tags: kv-cache

`--mem-fraction-static` (SGLang's GPU-mem analogue of vLLM `--gpu-memory-utilization`; default
0.85, raise toward 0.95 for throughput), `--kv-cache-dtype` (fp8 halves KV mem — validate accuracy),
`--page-size` (KV page size), `--max-total-tokens` (max tokens across all running sequences).

## Attention backends
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: tpot
- domain_tags: attention

`--attention-backend` and `--decode-attention-backend` (decode-phase-specific kernel). On ROCm prefer
the AITER attention path when available (see generic symptom→lever reasoning for TPOT-bound).

## MoE / expert-parallel knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: throughput
- domain_tags: moe

`--ep-size` (expert parallelism, EP ≤ TP), `--moe-backend`, `--moe-a2a-backend` (all-to-all),
`--deepep-mode` (DeepEP dispatch), `--enable-deepep-waterfill` (route shared expert as an extra MoE
slot to the least-loaded rank), `--elastic-ep-backend` (dynamic expert redistribution),
`--enable-dp-attention` (data-parallel attention), `--moe-data-parallel-size`, `--enable-eplb` +
`--eplb-algorithm` + `--eplb-rebalance-num-iterations` (expert load balancing).

## Scheduling / prefill knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: ttft/tpot
- domain_tags: scheduling

`--max-prefill-tokens` (max prefill tokens per batch), `--chunked-prefill-size`, `--schedule-policy`
(request ordering), `--disable-overlap-schedule` (disable compute/comm overlap — usually leave
overlap on), `--enable-prefill-delayer` (defer prefill to improve decode batching).

## Speculative decoding
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: throughput+tpot
- accuracy_risk: none (verifies drafts); incompatible with radix/prefix cache
- domain_tags: spec-decode

`--speculative-algorithm` (e.g. EAGLE/EAGLE3), `--speculative-num-steps`,
`--speculative-draft-model-path`. As with all frameworks, speculative decoding is incompatible with
prefix caching — disable the radix cache when using it.

## Prefix cache
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: throughput (shared-prefix workloads)
- domain_tags: prefix-cache

`--disable-radix-cache` (SGLang's prefix cache is the radix cache; disable when using speculative
decoding), `--radix-eviction-policy`.

## Compilation / CUDA-graph
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#sglang (doc, not run-validated)
- impact: host-overhead
- domain_tags: compilation

`--disable-cuda-graph` (eager fallback), `--cuda-graph-max-bs` (max batch size for capture),
`--enable-torch-compile` (or `SGLANG_TORCH_COMPILE` env).
