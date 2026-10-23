---
myst:
  html_meta:
    "description": "Optimizations Hyperloom found or produced on AMD Instinct GPUs that vLLM or SGLang has merged, each with the end-to-end measurement of the upstream PR, and the rule and check that keep the list current."
    "keywords": "Hyperloom, upstream, vLLM, SGLang, merged pull requests, AMD Instinct, MI300X, MI325X, MI355X, LLM inference, optimization results"
---
# Upstream optimizations merged into vLLM and SGLang

Optimizations Hyperloom found or produced in a real session on AMD Instinct GPUs that vLLM or SGLang
has merged. The README carries the eight with the largest measured effect; this is the full list.

Each row quotes the end-to-end measurement of the linked PR, on the model and GPU that PR tested,
which is not always the workload the session was optimizing. The numbers were measured on the
upstream revision that PR names, so a row says what held at merge time; later upstream work can
move it. No micro-benchmark figure is reported here as an end-to-end gain.

## What enters this list

Every PR in `vllm-project/vllm` or `sgl-project/sglang` whose body credits Hyperloom (`Found by
Hyperloom`, `Powered by Hyperloom`, or the older `Found and validated with ROCm Hyperloom`) **and**
is merged. Attributed PRs that are still open, or that were closed without merging, are not listed:
a change nobody upstream accepted is not evidence.

As of 2026-10-23 that is 9 of 45 attributed PRs, with 25 still open and 11 closed unmerged.

`python scripts/report_upstream_results.py` names every merged attributed PR missing from this page
and prints the attributed, merged, open and closed counts for the line above. Set `GITHUB_TOKEN` to
raise the GitHub search rate limit.

## Merged

| Upstream PR | Merged | Model / workload | GPU | Measured effect |
|-------|------|-------|------|-------|
| [vllm#58539](https://github.com/vllm-project/vllm/pull/58539) | 2026-09-30 | DeepSeek-V4.1-Flash, 307 in / 23 out, TP=2 | MI355X | +53.9 % total tok/s, −37.0 % mean TTFT. At a 0.25 length ratio, where few steps carry more than four prefills, throughput and median latency are unchanged |
| [sglang#41071](https://github.com/sgl-project/sglang/pull/41071) | 2026-10-10 | Qwen3-32B-FP8, 1024 in / 512 out, concurrency 4 | MI300X | +314 % output tok/s against the untuned Triton default, and +50 % at 8192 in / 1024 out, concurrency 128. Opt-in behind a flag: CK costs about 2 GSM8K points on this model, and the cause is unresolved |
| [vllm#58897](https://github.com/vllm-project/vllm/pull/58897) | 2026-10-14 | gpt-oss-120b, 1033 in / 523 out, concurrency 61 | MI355X | +9.7 % total tok/s, −11.1 % median TPOT. The gain is in decode; TTFT is flat |
| [vllm#58405](https://github.com/vllm-project/vllm/pull/58405) | 2026-09-30 | DeepSeek-V4-Flash, 307 in / 23 out, TP=1 | MI355X | +6.8 % total tok/s, −7.1 % mean TTFT. No regression at concurrency 96, 48 or 3 |
| [vllm#55991](https://github.com/vllm-project/vllm/pull/55991) | 2026-09-16 | Qwen3-30B-A3B, 1024 in / 128 out | MI325X | +6.9 % throughput, +5.2 % on MI355X. Reaches any model whose per-head QK norm slices a fused QKV projection; a no-op elsewhere |
| [sglang#41229](https://github.com/sgl-project/sglang/pull/41229) | 2026-10-19 | gemma-4-26B-A4B-it, 64 ragged 32K requests, concurrency 16 | MI325X | +6 % output tok/s. Balanced batches are unchanged by construction; outputs are bitwise identical |
| [vllm#58354](https://github.com/vllm-project/vllm/pull/58354) | 2026-10-14 | GLM-5.3-Flash, sparse-MLA metadata build | MI355X | 10.5 µs less per build call, +11–15 % on the build itself. At one or two builds per decode step this is under 0.1 % of a step, and the PR claims no end-to-end speedup |
| [vllm#57108](https://github.com/vllm-project/vllm/pull/57108) | 2026-10-10 | Block-FP8 checkpoint missing a weight scale | MI300X, MI325X | Correctness: a repack missing its `wo_a` block scale in all 43 layers used to boot and serve at GSM8K 0.00; the load now fails and names the 43 parameters |
| [vllm#57568](https://github.com/vllm-project/vllm/pull/57568) | 2026-09-28 | DeepSeek-V4 MTP drafter, `use_local_argmax_reduction` | MI355X | Correctness: the flag used to fail engine startup on every `deepseek_v4` checkpoint. No measurable end-to-end or GSM8K change once it works |
