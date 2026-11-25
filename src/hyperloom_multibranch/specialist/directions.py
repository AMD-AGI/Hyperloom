# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directions a session hands its specialist branches, so its first branches do not all sweep the same settings.

A branch started with no direction of its own and no repair takes the first one no branch of the session has held
yet, so the first four are the ones a four-branch session works on; once each was held, the one the fewest branches
held, then the one the fewest running branches hold, the earlier one in this order on a tie. A direction on an
operator chain the model may lack (experts, a recurrent layer) sends a branch on a model without it to another.

A session deals one tuple (:func:`session_directions`): :data:`DIRECTIONS`, written for a framework that runs as its
image installs it, under a torch profile, with tuned kernel tables; or :data:`CHECKOUT_DIRECTIONS` when its server is
built from the checkout its worktrees hold, which has no profile and no tables to read, and whose levers are the
options its launch leaves unset, the upstream's open pull requests, the kernels' launch choices, the auxiliary model
files beside the served one and the host work between GPU launches. Those name a method and the evidence it asks for,
never an option, a file or a constant: which one pays is what the branch finds.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    "wrapper: read which config keys the model file reads and the published recipes for this model on this GPU "
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
    "accumulate into the residual and the norm before qkv and gate_up zeroes their outputs; that removed the memsets "
    "and the separate residual adds from every step. A tuned row that crashes at a large M (the padded-M lookup "
    "reuses a row whose kernel rejects outputs over 2^30 elements) is fixed by splitting the call along M, not by "
    "dropping the table.",
    "Fusion: fuse the elementwise work around the GEMMs and the attention (rmsnorm with the residual add, silu with "
    "mul, rope) with a source patch or a kernel, starting from the operators the profile ranks highest. Carry the "
    "fusion to the quantize: a norm, gate or attention output that an FP8 GEMM quantizes again should write the FP8 "
    "values and scales itself, and the all-reduce before a norm can take the residual add, the norm and the quantize "
    "in the same kernel. Fold an activation into the GEMM epilogue that consumes it, and profile again once a large "
    "kernel on the path got faster: the small operators' shares then grow past the bar. Take the fused kernels aiter "
    "ships first: fused_qk_norm_rope_cache_pts_quant_shuffle replaced six per-layer kernels (q/k split, QK-norm, "
    "RoPE, KV write) with one, and fused_rms_fp8_group_quant with a fused silu_mul and fp8 group quant removed the "
    "separate quant kernels from every layer. A fused kernel that rejects the model's size (aiter's "
    "silu_and_mul_quant above d 8192) or sits inside an aiter op is copied into the framework tree and edited there. "
    "Fusions that each read below the bar are tested again only together, as one intent.",
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
    "router by emitting the sorted ids from it. Time the expert GEMMs against the bytes of the experts the batch "
    "actually routes to, counted from the sort's output or the routing ids, not from a guessed expert count: an fmoe "
    "once left as at the HBM bound on such a guess ran at about four fifths of a copy's rate, and a new kernel still "
    "had a share of the step to win. A model without experts has no such chain: take another of the session's "
    "directions.",
    "Small-M GEMM kernels: for a decode GEMM far below its roofline (a GEMV-shaped router or a narrow projection, a "
    "block-scale FP8 GEMM at M 32 to 64), tuning rows is done; change the kernel. The sign is a time that stays flat "
    "across N and M at several times what reading the kernel's weights at HBM bandwidth takes, as CK block-scale "
    "GEMMs and a small bf16 GEMV showed on one session. Swap the kernel family (aiter's Triton block-scale GEMM with "
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
    "a dev sandbox and by a profile test otherwise: aiter's pa_decode_gluon on the SHUFFLE 5D fp8 layout read the KV "
    "cache faster than ll4mi at page 16 on one session, while at page 64 it lost at short contexts. A kernel aiter "
    "ships is a starting point: one branch copied pa_decode_gluon into the framework tree and wrote the o_proj "
    "input's fp8 values and scales from its epilogue.",
    "Recurrent decode chain: on a model with linear attention or a state-space layer (GDN, Mamba), take its decode "
    "path as one chain: the input projection's split, the conv1d, the recurrent state update and the gated norm "
    "with its quantize. Measure the state bytes each step reads and writes against the time: one session's GDN "
    "recurrent kernel ran well below what a plain copy reaches, and its chain of a few kernels held a large share "
    "of the step. Fuse the chain into one or two kernels, keeping the state in its layout and precision, and port an "
    "upstream fused kernel first: sglang PR 39554 fuses it into aiter's fused_gdn_decode_qkvz. A model "
    "without such a layer has no such chain: take another of the session's directions.",
    "Existing PRs and patches: scan the GitHub PRs of aiter and of this session's framework (sglang, vllm or atom), "
    "merged or open, for ones on this model's architecture, its hot operators or ROCm and gfx950; check each against "
    "the version installed in the image, which can predate a merged PR, and port the ones it lacks as a patch to test. "
    "Search with WebSearch, once per hot operator at least (the architecture, ROCm or gfx950 and the operator, on "
    "site:github.com/sgl-project/sglang or site:github.com/ROCm/aiter), read a PR's diff with WebFetch, tell from the "
    "installed version's commit whether the image has it, and record each PR's verdict with blackboard_learn (in the "
    "image, missing, or not for this model) so no branch searches it again. A PR that needs a newer aiter or Triton "
    "than the image is vendored, not ruled out: copy its kernel into the patch with the helpers it imports (a "
    "Triton or gluon file, a ctypes launch of a .co), drop the version check that gates it and let the accuracy "
    "gate catch a miscompile. One session ruled a fused GDN decode PR unportable three times for its aiter and "
    "Triton versions, and a branch vendored it in minutes at the session's end.",
    "Communication: on a tensor-parallel workload, the all-reduces after the attention and the MoE are one "
    "operator. Time each at decode against its bytes: a message of a few hundred KB is latency-bound, so the "
    "levers are fewer all-reduces, an all-reduce that does the next op's work, and overlap. Routes already tried on "
    "one session, each flat or slower on its model and TP size: --enable-aiter-allreduce-fusion "
    "(decode-only too), SGLANG_USE_1STAGE_ALLREDUCE, aiter's old small all-reduce path, --enable-quant-communications "
    "and --enable-fused-moe-sum-all-reduce (read only by the Triton MoE); a larger AITER_CUSTOM_AR_MAX_SIZE helped. "
    "What was left is source: fuse the all-reduce with the residual add, norm and FP8 quantize the head "
    "already does after it, in one kernel that keeps that readout (sglang PR 44566 fuses the three); reduce the "
    "shared and routed experts' outputs in one all-reduce; or overlap an all-reduce with work that does not read "
    "its result.",
)

CHECKOUT_DIRECTIONS = (
    "Server options: account for every option the server takes and the launch leaves unset. Your role says where "
    "they are listed, the inventory the session wrote from the server's own help where it names one and the option "
    "table of the checkout otherwise: go through all of them, not the ones you remember. Where your role names "
    "option_review, record a verdict on each unset option with it before your first test that changes server "
    "arguments: such a test is refused until none is left. For each option that moves work off the largest line of "
    "your step budget, or removes it, read the code path it switches on, state the budget line it acts on and the "
    "observation that would prove it wrong, and test it: an unset option is a code path no test of this session ran. "
    "For a numeric option the launch already sets, measure the nearest value on each "
    "side of the launch's value before you call the launch's value best: one distant value says nothing about its "
    "neighbours.",
    "Upstream pull requests: before anyone writes a change the upstream repository may already hold, where "
    "start_analysis lists upstream_pr_search, search its open pull requests with it, once for each large line of "
    "your step budget, in the words of the mechanism (the operation, the model feature, the backend). A search "
    "shows you the number and title of its best match, not its diff: search again with more specific words to see "
    "another. For each hit whose change is on a path this launch runs, check that your worktree lacks the change, "
    "write it as a source change with any setting that enables it, and test it. Record each pull "
    "request's verdict with blackboard_learn (already in the checkout, ported, or not on this launch's path) so no "
    "branch reads it twice.",
    "GPU backend kernels: the kernels in the directories your role names run on every decode step. Bound what they "
    "can still gain from the timing lines and the bytes each pass reads. Aim at the kernels that hold the most time "
    "of a step: where start_analysis lists knob_tune, the time its default build gives a weight type and column count "
    "at one weight shape of your model, divided by the copies a case multiplies and multiplied by the weights of that "
    "type and shape the step's passes of that column count read, is that kernel's time per step. Before you write a "
    "kernel change, where start_analysis lists upstream_pr_search, search the component's open pull requests with it "
    "in the words this session's "
    "evidence uses for the GPU backend the server runs and for this card's architecture (the server's log, the "
    "per-architecture branches of the kernels), not only in those of the model or the feature; when a result says "
    "its list is cut, search again with fewer, more specific words; and judge a pull request by its diff, not its "
    "title, which can name another architecture than the code gates. Read how each kernel that this model's "
    "weight format takes is launched for this card's architecture: the constants and the branches chosen per "
    "architecture. Where one was chosen for another card or shape than this workload's, measure the alternatives, "
    "with knob_tune where start_analysis lists it, else as a source change with a one-time log line on the path. Two "
    "kernels that can be one are the next lever.",
    "Auxiliary model files: list the directory of every model file the launch names, following symlinks, for other "
    "builds of an auxiliary file, and test each as the launch's auxiliary "
    "model. The served model's file never changes. Judge each by the acceptance on the timing lines and by the "
    "drafts' share of the step, then measure again, on the file that won, the settings that depend on it: the best "
    "settings move with the file.",
    "Host work between GPU launches: the part of a step that neither the bandwidth bound nor the kernels' time "
    "explains is host work while the GPU idles. Read the "
    "host side of the decode loop (your role names where it is), put one-time timing log lines around its stages in "
    "a test, and remove or move the largest: first with an option the server already has for that stage, else with a "
    "source change.",
)


def session_directions(checkouts: Sequence[str]) -> tuple[str, ...]:
    """The directions a session deals whose worktrees hold ``checkouts``, the components built from their checkout:
    those of a server built from its checkout when there is one, else those of a framework run as installed."""
    return CHECKOUT_DIRECTIONS if checkouts else DIRECTIONS


def direction_name(direction: str) -> str:
    """The short name a direction opens with, as prompts and notices cite it."""
    return direction.split(":", 1)[0]


def session_direction(named: str, directions: Sequence[str]) -> str | None:
    """The one of the session's ``directions`` an exhausted record's ``direction`` names: its name alone, or followed
    by a qualifier."""

    named = named.strip().lower()
    for direction in sorted(directions, key=lambda d: -len(direction_name(d))):
        name = direction_name(direction).lower()
        if named == name or (named.startswith(name) and not named[len(name)].isalnum()):
            return direction
    return None


__all__ = ["CHECKOUT_DIRECTIONS", "DIRECTIONS", "direction_name", "session_direction", "session_directions"]
