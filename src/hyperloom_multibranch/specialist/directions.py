# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directions a session hands its specialist branches, so its first branches do not all sweep the same settings.

A branch started with no direction of its own and no repair takes the first one no branch of the session has held
yet, so the first four are the ones a four-branch session works on; once each was held, the one the fewest branches
held, then the one the fewest running branches hold, the earlier one in this order on a tie. A direction on an
operator chain the model may lack (experts, a recurrent layer) sends a branch on a model without it to another.
"""

from __future__ import annotations

DIRECTIONS = (
    "Fast path checks: make sure every setting the recipe asks for and every patch really takes its fast path. Read "
    "the server.log of your branch's tests for what fell back: an eager or plain torch path, CUDA graphs off, a kernel "
    "or a feature the engine turned off. Where the log cannot tell, add a one-time log line on the path to prove it "
    "ran. For each operator the profile ranks highest, read the branch conditions on its call path (is_cuda or is_hip, "
    "an SM, gfx or capability check, a table keyed by TP size or cu_num): a tuned constant or a fused path that sits "
    "only in another branch never runs here, and the log does not say so. A fast path behind a whitelist or table "
    "keyed by the model's shape (expert count, top-k, head size) falls back silently for a shape it does not list, "
    "and a fused helper the framework already has runs only for the models that ask for it: grep where each fused "
    "helper is called and which model files pass its switch. A model feature its config leaves off (an index or KV "
    "shared across layers, a fused path) is set with --hf-overrides, nested under text_config for a multimodal "
    "wrapper: read which config keys the model file reads and the published MI355X recipes for this model "
    "(InferenceX, AMD); an approximate one must pass the accuracy gate. Fix the cause with a source patch or the "
    "setting that removes it; errors and warnings in the log are leads, and untuned GEMM shapes belong to the GEMM "
    "tuning branch.",
    "GEMM tuning: for the GEMM shapes the profile shows for this workload (decode and prefill), first check which "
    "table each shape actually reads in the server.log's tuned-config lookups; a shape already on its best row in the "
    "image's table needs nothing. When start_analysis lists kernel_tune, tune the untuned shapes with it before "
    "anything else. For a shape the log reports as untuned, first switch it to a kernel family with tuned rows or a "
    "stronger default; rows borrowed from tuned tables of the same kernel family at nearby shapes are an untimed "
    "guess, so say so in your branch log. Put only your rows with put_kernel_artifact and rows_only=true, carrying the "
    "rows of a table your head already has for that variable, and test several shapes in one test rather than one "
    "shape per test. A peer's report that a kernel family is missing from the image closes nothing for your head: read "
    "the failing test's log or the image's kernel registry yourself first. A split-K row's time counts in a CUDA graph "
    "with its memset (a dev sandbox times it there; without one, a test is the measurement): aiter's tuner times the "
    "kernel alone, so its split-K picks never pay the hipMemsetAsync the launcher adds before each call. That memset "
    "is not out of reach: one branch launched the same asm .co directly through ctypes, so o_proj and down_proj "
    "accumulate into the residual and the norm before qkv and gate_up zeroes their outputs; that removed 160 memsets "
    "and the residual adds per step, +4.2 to 5.1%. A tuned row that crashes at a large M (the padded-M lookup reuses a "
    "row whose kernel rejects outputs over 2^30 elements) is fixed by splitting the call along M, not by dropping the "
    "table.",
    "Fusion: fuse the elementwise work around the GEMMs and the attention (rmsnorm with the residual add, silu with "
    "mul, rope) with a source patch or a kernel, starting from the operators the profile ranks highest. Carry the "
    "fusion to the quantize: a norm, gate or attention output that an FP8 GEMM quantizes again should write the FP8 "
    "values and scales itself, and the all-reduce before a norm can take the residual add, the norm and the quantize "
    "in the same kernel. Fold an activation into the GEMM epilogue that consumes it, and profile again once a large "
    "kernel on the path got faster: the small operators' shares then grow past the bar. Take the fused kernels aiter "
    "ships first: fused_qk_norm_rope_cache_pts_quant_shuffle replaced six per-layer kernels (q/k split, QK-norm, "
    "RoPE, KV write; 26 to 4.5 us a layer, +6.9%), and fused_rms_fp8_group_quant with a fused silu_mul and fp8 group "
    "quant removed 120 quant kernels per step, +4.8%. A fused kernel that rejects the model's size (aiter's "
    "silu_and_mul_quant above d 8192) or sits inside an aiter op is copied into the framework tree and edited there. "
    "Fusions each below the bar go together as one intent.",
    "Server arguments and environment variables: yours is the one branch that tests settings alone; test the settings "
    "the profile or server.log point at, not a blind sweep. Before testing a setting, grep where it is read: one with "
    "no reader on this model's path, this backend or this phase (decode or prefill) is dead, and its test measures "
    "nothing. Bound a scheduler setting (chunked or mixed prefill, prefill and decode interleaving, running-request or "
    "cache-pool caps) from server.log first: what the run loses to the steady decode rate, less the prefill tokens at "
    "the prefill rate, is all scheduling can win, and under a closed-loop client a cap above the concurrency divided "
    "by the DP size is never reached. On a multi-turn workload, compare each follow-up turn's computed prefill tokens "
    "with its new tokens: the excess is a prefix the cache drops every turn (a trailing block a draft path discards, a "
    "block alignment, an eviction), and recovering it beats any prefill kernel.",
    "MoE chain: on a model with experts, take the router GEMV, the softmax and top-k, the sort and the expert GEMMs as "
    "one chain, past the tuned tables and environment switches: together they can hold half of a decode step, and "
    "each launch between them is a boundary to remove. Write the fused kernel, Triton or HIP, that replaces several "
    "of them (the router GEMV with the softmax and top-k, or the top-k with the sort), or fold the sort into the "
    "router: one branch measured 3.2 to 3.9 us a layer saved by emitting the sorted ids from the router. Time the "
    "expert GEMMs against the bytes of the experts the batch actually routes to, counted from the sort's output or "
    "the routing ids, not from a guessed expert count: an fmoe left as at the HBM bound ran 53 us a layer, 80% of "
    "the 6.26 TB/s a copy measured, so a new kernel still had about 6% of the step. A model without experts has no "
    "such chain: take another of the session's directions.",
    "Small-M GEMM kernels: for a decode GEMM far below its roofline (a GEMV-shaped router or a narrow projection, a "
    "block-scale FP8 GEMM at M 32 to 64), tuning rows is done; change the kernel. One session's CK block-scale GEMMs "
    "ran 7.7 to 8.5 us at N 2048 to 6144 whatever M, where their 4 to 13 MB of weights take 0.7 to 2 us at HBM "
    "bandwidth, and a bf16 GEMV read 128 KB in 8.8 us. Swap the kernel family (aiter's Triton block-scale GEMM with "
    "split-K, an asm kernel, a per-token FP8 GEMM on a requantized copy), merge sibling projections that read the "
    "same input into one GEMM, or fold a GEMV into the kernel that consumes its output (a router into the top-k, a "
    "gate projection into the conv or norm after it).",
    "ROCm code gaps: find the code paths where ROCm runs a slower or missing implementation where CUDA has a fast one "
    "(a fallback, a missing fused or tuned path), and port or align the CUDA implementation.",
    "Attention and decode kernels: speed up the hottest attention and decode kernels the profile shows, with a kernel "
    "artifact or a source patch. Check which attention kernel the head runs, in server.log and the profile: a newer "
    "kernel family (such as aiter's gluon paged attention) can read the KV cache in another layout, and then every "
    "kernel that writes the KV cache must write the layout the new one reads (a fused norm, rope and cache-write "
    "kernel included); a mismatch shows as an accuracy failure, not as a slowdown. Where the writer is already "
    "patched, revise the patch your head already has for that writer instead of adding a second one. Compare "
    "candidate kernels per layer at the graded batch and context, page size included, in a CUDA graph when you have "
    "a dev sandbox and by a profile test otherwise: aiter's pa_decode_gluon on the SHUFFLE 5D fp8 layout at page 16 "
    "read the KV cache at 5.6 TB/s where ll4mi read 3.5 to 4.1, while page 64 lost below context 2048 (one branch "
    "went from 7779-8105 to 8467-8646 tok/s at page 16). A kernel aiter ships is a starting "
    "point: one branch copied pa_decode_gluon into the framework tree and wrote the o_proj input's fp8 values and "
    "scales from its epilogue, +1.3 to 1.8%.",
    "Recurrent decode chain: on a model with linear attention or a state-space layer (GDN, Mamba), take its decode "
    "path as one chain: the input projection's split, the conv1d, the recurrent state update and the gated norm "
    "with its quantize. Measure the state bytes each step reads and writes against the time: one session's GDN "
    "recurrent kernel ran at 3.9 TB/s where a copy reached 6.26, and the chain held 12% of the step in three "
    "kernels. Fuse the chain into one or two kernels, keeping the state in its layout and precision, and port an "
    "upstream fused kernel first: sglang PR 39554 fuses it into aiter's fused_gdn_decode_qkvz, 830 to 279 us a "
    "step and +8.6% at TP4. A "
    "model without such a layer has no such chain: take another of the session's directions.",
    "Existing PRs and patches: scan the GitHub PRs of aiter and of this session's framework (sglang, vllm or atom), "
    "merged or open, for ones on this model's architecture, its hot operators or ROCm and gfx950; check each against "
    "the version installed in the image, which can predate a merged PR, and port the ones it lacks as a patch to test. "
    "Search with WebSearch, once per hot operator at least (the architecture, ROCm or gfx950 and the operator, on "
    "site:github.com/sgl-project/sglang or site:github.com/ROCm/aiter), read a PR's diff with WebFetch, tell from the "
    "installed version's commit whether the image has it, and record each PR's verdict with notebook_learn (in the "
    "image, missing, or not for this model) so no branch searches it again. A PR that needs a newer aiter or Triton "
    "than the image is vendored, not ruled out: copy its kernel into the patch with the helpers it imports (a "
    "Triton or gluon file, a ctypes launch of a .co), drop the version check that gates it and let the accuracy "
    "gate catch a miscompile. One session ruled a fused GDN decode PR unportable three times for its aiter and "
    "Triton versions, and a branch vendored it in minutes at the session's end.",
    "Communication: on a tensor-parallel workload, the all-reduces after the attention and the MoE are one "
    "operator. Time each at decode against its bytes: a message of a few hundred KB is latency-bound, so the "
    "levers are fewer all-reduces, an all-reduce that does the next op's work, and overlap. Routes already tried on "
    "one session (Qwen3-Next TP2, concurrency 64), each flat or 1.7 to 4.1% slower: --enable-aiter-allreduce-fusion "
    "(decode-only too), SGLANG_USE_1STAGE_ALLREDUCE, aiter's old small all-reduce path, --enable-quant-communications "
    "and --enable-fused-moe-sum-all-reduce (read only by the Triton MoE); a larger AITER_CUSTOM_AR_MAX_SIZE gained "
    "+2.1%. What was left is source: fuse the all-reduce with the residual add, norm and FP8 quantize the head "
    "already does after it, in one kernel that keeps that readout (sglang PR 44566 fuses the three); reduce the "
    "shared and routed experts' outputs in one all-reduce; or overlap an all-reduce with work that does not read "
    "its result.",
)


def direction_name(direction: str) -> str:
    """The short name a direction opens with, as prompts and notices cite it."""
    return direction.split(":", 1)[0]


__all__ = ["DIRECTIONS", "direction_name"]
