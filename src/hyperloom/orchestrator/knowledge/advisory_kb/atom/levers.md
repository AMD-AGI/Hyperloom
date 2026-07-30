# ATOM — Tunable levers (ATOM runs only)

Exact knobs for ATOM on ROCm (note ATOM uses underscore flag names, e.g. `--kv_cache_dtype`).
Reasoning for *when* to pull each is in `generic/symptom_levers.md`. Catalog from cph-perf-tuning
`conf/legal_axes.yaml` (atom section). Doc-sourced — grow with measured ATOM findings.

## KV-cache / memory knobs
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: memory/throughput
- domain_tags: kv-cache

`--kv_cache_dtype` (fp8 halves KV mem — validate accuracy), `--gpu_memory_utilization` (raise toward
0.95 for throughput), `--block_size`.

## Attention
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: tpot
- domain_tags: attention

`--attention` (attention path). ATOM also supports the fused QK-norm+RoPE+cache-quant path via
`ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` (see correctness_flags for the fused-op env).

## MoE / expert-parallel
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: throughput
- domain_tags: moe

`--enable_expert_parallel` (EP ≤ TP), `--enable_dp_attention` (data-parallel attention),
`--all2all_backend`. ATOM has custom all-reduce kernels and MoE shared-expert fusion.

## Scheduling
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: throughput/ttft
- domain_tags: scheduling

`--max_num_batched_tokens`, `--max_num_seqs`, `--enable_chunked_prefill`, `--enable_tbo`
(two-batch-overlap scheduling).

## Speculative decoding
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: throughput+tpot
- accuracy_risk: none (verifies drafts); incompatible with prefix caching
- domain_tags: spec-decode

`--method` (speculative method), `--num_speculative_tokens`, `--draft_model` (EAGLE3). Disable prefix
caching when using speculative decoding.

## Prefix cache & compilation
- kind: hint
- source: cph-perf-tuning:legal_axes.yaml#atom (doc, not run-validated)
- impact: throughput/host-overhead
- domain_tags: compilation

`--enable_prefix_caching` (disable when using speculative decoding), `--level` (compilation level),
`--cudagraph_capture_sizes`. Large models: `ATOM_DISABLE_MMAP=true` to disable memory-mapped weights.
