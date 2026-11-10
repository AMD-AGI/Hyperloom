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

As of 2026-11-02 that is 27 of 61 attributed PRs: the 26 optimizations below, and one CI-permissions
change reviewed and not listed. Another 19 are still open and 15 were closed unmerged.

`python scripts/report_upstream_results.py` names every merged attributed PR missing from this page
and prints the attributed, merged, open and closed counts for the line above. Set `GITHUB_TOKEN` to
raise the GitHub search rate limit.

## Merged: throughput and latency

| Upstream PR | Merged | Model / workload | GPU | Measured effect |
|-------|------|-------|------|-------|
| [sglang#41071](https://github.com/sgl-project/sglang/pull/41071) | 2026-10-10 | Qwen3-32B-FP8, 1024 in / 512 out, concurrency 4 | MI300X | +314 % output tok/s against the untuned Triton default, and +50 % at 8192 in / 1024 out, concurrency 128. Shipped behind a flag over a 2-point GSM8K drop; sglang#43815 traced that drop to the 5-shot evaluation setting and made the CK GEMM the default |
| [sglang#43815](https://github.com/sgl-project/sglang/pull/43815) | 2026-10-27 | Qwen3-235B-A22B-Instruct-2507-FP8, 1024 in / 512 out, concurrency 4, TP=4 | MI325X | +118 % output tok/s, and +28 % at 8192 in / 1024 out, concurrency 128. Makes sglang#41071's CK GEMM the gfx942 default; 8-shot GSM8K stays within a point and MMLU level across eight models |
| [vllm#58539](https://github.com/vllm-project/vllm/pull/58539) | 2026-09-30 | DeepSeek-V4.1-Flash, 307 in / 23 out, TP=2 | MI355X | +53.9 % total tok/s, −37.0 % mean TTFT. At a 0.25 length ratio, where few steps carry more than four prefills, throughput and median latency are unchanged |
| [sglang#45295](https://github.com/sgl-project/sglang/pull/45295) | 2026-10-29 | Gemma 3 12B it, 12288 in / 2048 out, concurrency 128 | MI325X | +14 % output tok/s, from 11–20 % more KV capacity for the full-attention layers. Found while optimizing Olmo-3-7B; Gemma 2, Gemma 3, EXAONE 4 and Olmo 3 now get a hybrid sliding-window KV pool. EXAONE 4 32B is compute-bound and does not speed up |
| [vllm#58897](https://github.com/vllm-project/vllm/pull/58897) | 2026-10-14 | gpt-oss-120b, 1033 in / 523 out, concurrency 61 | MI355X | +9.7 % total tok/s, −11.1 % median TPOT. The gain is in decode; TTFT is flat |
| [sglang#46731](https://github.com/sgl-project/sglang/pull/46731) | 2026-10-29 | gemma-4-26B-A4B-it, 8192 in / 1024 out, concurrency 256 | MI355X | +9.0 % total tok/s, −15.9 % mean TTFT, from a tuned Triton fused-MoE config where main had none. +2–9 % across five shapes, none slower |
| [sglang#46832](https://github.com/sgl-project/sglang/pull/46832) | 2026-10-29 | Qwen3-14B-FP8, 1024 in / 512 out, concurrency 8, fp8 KV cache | MI355X | +7.3 % output tok/s, −7.2 % median TPOT, by fusing QK-norm, RoPE and the fp8 KV write into one decode kernel. TTFT within ±3 %; gfx942 not measured |
| [vllm#55991](https://github.com/vllm-project/vllm/pull/55991) | 2026-09-16 | Qwen3-30B-A3B, 1024 in / 128 out | MI325X | +6.9 % throughput, +5.2 % on MI355X. Reaches any model whose per-head QK norm slices a fused QKV projection; a no-op elsewhere |
| [vllm#58405](https://github.com/vllm-project/vllm/pull/58405) | 2026-09-30 | DeepSeek-V4-Flash, 307 in / 23 out, TP=1 | MI355X | +6.8 % total tok/s, −7.1 % mean TTFT. No regression at concurrency 96, 48 or 3 |
| [sglang#41229](https://github.com/sgl-project/sglang/pull/41229) | 2026-10-19 | gemma-4-26B-A4B-it, 64 ragged 32K requests, concurrency 16 | MI325X | +6 % output tok/s. Balanced batches are unchanged by construction; outputs are bitwise identical |
| [sglang#47236](https://github.com/sgl-project/sglang/pull/47236) | 2026-11-01 | Qwen3-Next-80B-A3B-Instruct-FP8, 1024 in / 1024 out, concurrency 16, TP=2 | MI355X | +2.7 % output tok/s, −2.6 % median TPOT, from routing decode through the Triton top-k router. TTFT moves less than run-to-run drift |
| [vllm#58354](https://github.com/vllm-project/vllm/pull/58354) | 2026-10-14 | GLM-5.3-Flash, sparse-MLA metadata build | MI355X | 10.5 µs less per build call, +11–15 % on the build itself. At one or two builds per decode step this is under 0.1 % of a step, and the PR claims no end-to-end speedup |

## Merged: correctness

Each of these made a model load, run or produce correct output where it did not before.

| Upstream PR | Merged | Model / workload | GPU | Measured effect |
|-------|------|-------|------|-------|
| [sglang#47029](https://github.com/sgl-project/sglang/pull/47029) | 2026-11-01 | Llama-3.1-8B-Instruct-FP8-KV with NGRAM speculative decoding, 3.5M-token KV pool | MI355X | A long-running aiter server crashed once KV slot offsets passed 2^31, about 2.1M tokens: main faulted 430–485 questions into GSM8K in three runs of three. It now serves 600 requests and 2.46M output tokens without a fault |
| [vllm#63116](https://github.com/vllm-project/vllm/pull/63116) | 2026-10-31 | Kimi-K3 with its DSpark draft, 7 speculative tokens, TP=4 | MI355X | Warmup hit a memory fault from undersized AITER MLA metadata; the server now starts and serves 24 concurrent requests. Validated on dummy weights only, with no real-weight run |
| [sglang#46833](https://github.com/sgl-project/sglang/pull/46833) | 2026-10-30 | Granite 3.x and 4.0, and any model whose softmax scale is not `head_dim**-0.5` | MI355X | aiter prefill and decode ignored the layer's softmax scale: Granite 3.3-2B scored 0.0023 on 5-shot GSM8K and now 0.6187, against 0.6293 on triton. No change for default-scale models |
| [sglang#46800](https://github.com/sgl-project/sglang/pull/46800) | 2026-10-30 | MiniMax-M3.1-Flash-Quark-MXFP4, TP=4 | MI355X | Crashed at load from fusing MXFP8 shared experts into the MXFP4 slot; it now loads unfused at GSM8K 0.963–0.967. Correct output also needs the MXFP8 scale handling of sglang#46693, #46587 and #44080 |
| [sglang#45486](https://github.com/sgl-project/sglang/pull/45486) | 2026-10-30 | Qwen3-30B-A3B-FP8-dynamic, explicit Triton MoE runner with AITER on | MI355X | Compressed-tensors channel FP8 MoE produced wrong output: 0/200 on GSM8K, now 181/200 against 180/200 on the AITER runner. A 200-question sanity check, not a full-dataset result |
| [sglang#44268](https://github.com/sgl-project/sglang/pull/44268) | 2026-10-30 | Qwen3.6-35B-A3B, Kimi-Linear-48B, MiniMax-M2.7 and Qwen3-Next with `--enable-fused-moe-sum-all-reduce` | MI355X | The flag zeroed every Triton-runner MoE layer with top-k above 2 on ROCm, GSM8K 0 %; Qwen3.6-35B-A3B now scores 94.3 %. The flag still costs about 60 % throughput on gfx950 and should stay off there |
| [sglang#45802](https://github.com/sgl-project/sglang/pull/45802) | 2026-10-29 | Online Quark MXFP4 MoE loading under expert parallelism, including the Qwen3.5 MTP path | MI355X | A nonlocal expert arriving after the local weights tripped a size assertion at load; the no-copy load now returns cleanly. Verified with unit and loading tests, with no model accuracy run |
| [sglang#45800](https://github.com/sgl-project/sglang/pull/45800) | 2026-10-29 | GLM NextN with `SGLANG_GLM_NEXTN_MOE_PTPC=1` | — | Per-expert excludes stopped Quark from selecting a quantization method for the routed experts; it now selects the per-channel FP8 scheme. Verified with unit and CPU component tests only, with no GPU end-to-end run |
| [sglang#46528](https://github.com/sgl-project/sglang/pull/46528) | 2026-10-27 | Llama-3.1-8B-Instruct-FP8-KV and other fp8-KV checkpoints with calibrated KV scales | MI355X, MI325X | With the default aiter backend, three attention paths misused the KV scales and the output was garbage: GSM8K 0.000–0.005, now 0.755–0.760 against 0.765 with a bf16 cache. Reached without any flag; no overhead for models without KV scales |
| [vllm#63099](https://github.com/vllm-project/vllm/pull/63099) | 2026-10-26 | DeepSeek-V4-Flash with `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` | MI355X | Failed at weight loading because the V4.1 FP8 config skipped the shared-expert compatibility check; it now serves with the shared expert unfused. Throughput and GSM8K unchanged |
| [sglang#45531](https://github.com/sgl-project/sglang/pull/45531) | 2026-10-26 | Gemma 2 27B it with aiter unified-attention decode | MI325X | Decode dropped the layer's logit softcap: GSM8K fell to 0.213 against 0.878 on triton, and now scores 0.879. Applying the cap costs 3–6 % of decode time |
| [vllm#63082](https://github.com/vllm-project/vllm/pull/63082) | 2026-10-23 | MiniMax-M3 IndexCache through a top-level `--hf-overrides` | MI355X | The documented override form was silently ignored, leaving IndexCache off with no log; it now fails at startup and names the nested form. The PR also measured IndexCache's long-context NLL cost and dropped the docstring's "negligible accuracy impact" claim |
| [vllm#57108](https://github.com/vllm-project/vllm/pull/57108) | 2026-10-10 | Block-FP8 checkpoint missing a weight scale | MI300X, MI325X | A repack missing its `wo_a` block scale in all 43 layers used to boot and serve at GSM8K 0.00; the load now fails and names the 43 parameters |
| [vllm#57568](https://github.com/vllm-project/vllm/pull/57568) | 2026-09-28 | DeepSeek-V4 MTP drafter, `use_local_argmax_reduction` | MI355X | The flag used to fail engine startup on every `deepseek_v4` checkpoint. No measurable end-to-end or GSM8K change once it works |

## Reviewed and not listed

The check searches for any upstream PR body naming Hyperloom, so it also finds merged PRs that
mention the project without being an optimization it found. Those are recorded here so they are
not reviewed again.

| Upstream PR | Merged | Why it is not listed |
|-------|------|-------|
| [sglang#46969](https://github.com/sgl-project/sglang/pull/46969) | 2026-10-29 | Grants three Hyperloom contributors CI permissions in SGLang. A configuration change that names the team, not an optimization Hyperloom found |
