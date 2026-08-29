# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist sub-agent prompt assembler.

Returns ``(system_prompt, user_prompt)``: the system prompt carries the
immutable contract (identity / output protocol / iron rules) so the
backend can cache it across specialists; the user prompt carries per-task
context (hardware / gap / KB / recipe / PR / source hint). Most sections
render a ``(none)`` placeholder when empty; the execution-budget and
PD-disaggregation sections are omitted entirely. Not pure: the PD section
reads multi-node state (env + on-disk ``multi_node_state.json``) and the
enablement section builds the enablement mandate, which probes framework
source roots on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from hyperloom.common.prompt_safety import defang_prompt_structure

from ..specialists.domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SpecialistDomain,
    domain_for_tag,
)
from ..specialists.profile import MODE_PATCH


_NONE_PLACEHOLDER = "(none)"
# GPU share below which re-dispatching an op to another backend is not worth the
# equivalence work; keeps per-op glue out of the substitution directive.
_VENDOR_SUBSTITUTION_MIN_GPU_PCT = 5.0


# Curated launch-recipe sites the research scout mines for verified serve
# flags / envs, keyed by (model x hardware x quant x strategy). Overridable
# via HYPERLOOM_RECIPE_SITES (comma/space separated); values are advisory
# templates, not fetched by the Coordinator.
DEFAULT_RECIPE_SITES: tuple[str, ...] = (
    "https://recipes.vllm.ai/<org>/<model>?hardware=<gpu>",
    "https://lmsysorg.mintlify.app/cookbook/autoregressive/<family>/<model>",
)

# Operator sentinels (via HYPERLOOM_RECIPE_SITES) that disable recipe-site guidance.
RECIPE_SITES_DISABLED_VALUES: frozenset[str] = frozenset({"none", "off", "disable", "disabled"})

# Per-task-kind brief appended by _section_mandate; only the per-dispatch dynamic
# context (PR lead, critic feedback, apply errors, residual questions) lives in
# ``inp.notes``. Boilerplate that is the same for every dispatch of a given kind
# lives here so it is maintained in one place.
_TASK_KIND_BRIEFS: dict[str, str] = {
    "framework_authoring": (
        "Read the upstream PR as inspiration, then deliver the best win for this"
        " model / hardware / workload — go beyond the diff where the live source justifies it."
        " Deliverable: a unified-diff patch in your worktree (``patches_written``)"
        " OR a config/env ``proposal_set`` entry when the win is a flag toggle."
        " The Coordinator applies and benches it; you do not benchmark."
    ),
    "framework_local_explore": (
        "No upstream PR was found. Author the best throughput win directly from"
        " the live source + profiling evidence. Read ``framework_source_roots``"
        " and the roofline (Section 4a) to locate the hot path."
        " You MAY use WebSearch / WebFetch to compare the local checkout against"
        " the latest upstream code and port a newer optimisation when behind."
        " Deliverable: a unified-diff patch (``patches_written``)"
        " OR a config/env ``proposal_set`` entry. You do not benchmark."
    ),
    "explore_apply_retry": (
        "A previous patch failed to apply against the live source tree."
        " Study the apply errors in the notes, produce a corrected patch."
    ),
    "framework_config_generation": (
        "Propose a GRID of runtime config variants (server flags and/or env vars)"
        " that may raise throughput WITHOUT changing source. Return a"
        " ``proposal_set`` — each entry with ``name``, ``extra_args`` or"
        " ``extra_envs``, and a one-line ``reason``. You do not benchmark."
    ),
}


# Forbids global process cleanup that could kill the optimizer's serving /
# benchmark process. Shared by bash-enabled specialist and leaf prompts.
BASH_KILL_SAFETY_PREAMBLE = (
    "Do NOT run global process cleanup. Never run `ps aux | grep ... | xargs "
    "kill`, `pgrep -f ... | xargs kill`, or `killall` — these can kill the "
    "optimizer's serving / benchmark process. Only manage processes you "
    "started yourself, by their own PID."
)


# Per-domain focus templates: each injects a "Domain focus" block into
# Section 1; a missing key falls back to the generic body.


def _is_atom(inp: SpecialistPromptInputs) -> bool:
    """True when ``_focus_*`` blocks should use atom-flavoured hints
    (empty framework falls back to the canonical sglang/vllm block).

    Args:
        inp: The specialist prompt inputs.

    Returns:
        True when the framework is ``atom``.
    """
    return (inp.framework or "").strip().lower() == "atom"


def _focus_serving_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the serving specialist.

    Selects atom-flavoured or canonical sglang/vllm "what to read / winning
    techniques / pitfalls" hints based on the active framework.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs (used to pick the
            framework flavour).

    Returns:
        list[str]: Markdown lines for the serving specialist's focus block.
    """
    if _is_atom(inp):
        return [
            "You target **atom scheduler / cuda_graph / kv_cache** code.",
            "",
            "**What to read first**",
            "- `atom/entrypoints/openai_server.py` (HTTP request routing, " + "`/start_profile`, `/stop_profile`).",
            "- `atom/model_engine/engine_core.py` (engine main loop, " + "`start_profiler` call sites).",
            "- `atom/model_engine/llm_engine.py` (engine API surface).",
            "- `atom/model_engine/model_runner.py` (per-rank forward, " + "profiler hooks, cudagraph capture).",
            "- `atom/model_engine/arg_utils.py` (CLI flag inventory; "
            + "ground-truth for `--level`, `--enable_prefix_caching`, "
            + "`--cudagraph-capture-sizes`, `--kv_cache_dtype`, etc.).",
            "- `atom/config.py` (engine config shape).",
            "- KB anchor `framework.*` (cuda_graph / batching / kv_cache).",
            "",
            "**Winning techniques to consider**",
            "- `--level {2,3}` (atom's torch.compile / cudagraph bracket).",
            "- `--enable_prefix_caching` for shared-prefix workloads.",
            "- `--cudagraph-capture-sizes` bracketed around the live CONC.",
            "- `--kv_cache_dtype fp8` on FP8-shipped models (gate accuracy).",
            "- `--max-num-seqs` / `--max-num-batched-tokens` at concurrency "
            + "boundaries (same scheduler-side tuning as sglang/vllm).",
            "",
            "**Pitfalls (historical REVERTs / atom-specific)**",
            "- Atom is single-node only. Multi-node distributed proposals "
            + "are non-actionable — pivot to rank-local optimisations.",
            "- `--enforce-eager` is a debug fallback; almost never a perf win.",
            "- Cudagraph capture size lists that don't bracket the live "
            + "CONC trigger silent recapture on every batch boundary.",
        ]
    return [
        "You target **vLLM / SGLang scheduler / cuda_graph / kv_cache** code.",
        "",
        "**What to read first**",
        "- `vllm/v1/engine/` and `vllm/v1/worker/` (scheduler, model_runner).",
        "- `sglang/python/sglang/srt/scheduler/` and `sglang/python/sglang/srt/managers/`.",
        "- KB anchor `framework.*` (cuda_graph / batching / chunked_prefill / kv_cache).",
        "",
        "**Config levers (cheap, try first — these are env/flag changes)**",
        "- `--enable-chunked-prefill` + matched `--max-num-batched-tokens`.",
        "- `--enforce-eager=false` + cuda graph capture for stable batch sizes.",
        "- `--kv-cache-dtype fp8_e4m3` when the gap is HBM-bound (gate accuracy!).",
        "- `--max-num-seqs` tuning at concurrency boundaries.",
        "- AITER umbrella (`VLLM_ROCM_USE_AITER=1`) is **ALWAYS_ON** on MI300X;",
        "  `VLLM_ROCM_USE_AITER_RMSNORM=0` / `...PAGED_ATTN=1` are **NEVER_TOUCH**",
        "  (crash / dead var). Do NOT propose flags in the NEVER_TOUCH set.",
        "",
        "**Source-patch playbook (the high-ceiling work — author real code)**",
        "Config tuning has a low ceiling. When the gap persists after the cheap",
        "levers (or the orchestrator escalates you for a code patch), modify the",
        "framework **source** and write a unified diff into your worktree",
        "`patches/` dir (see the patch protocol section). Map the profile gap to",
        "the module:",
        "- **Scheduler / batch composition gap** (low batch occupancy, decode",
        "  starvation) → `scheduler.py` batch policy: prefill/decode interleave,",
        "  `max_num_batched_tokens` chunk split, waiting-queue admission order.",
        "- **KV-cache / block-manager gap** (HBM-bound, fragmentation) →",
        "  block_manager / paged-cache: block-size policy, eviction, prefix-cache",
        "  reuse. Keep `block_size >= 16`.",
        "- **CUDA-graph / capture gap** (host-bound, dispatch overhead) →",
        "  capture-size set, inductor graph partition, eager-fallback conditions.",
        "- **Chunked-prefill granularity** → split-size heuristic in the",
        "  scheduler, not just the flag.",
        "Keep patches small (target ≤5 files); preserve upstream call-order",
        "contracts (e.g. vLLM `scheduler.add_seq_group` ordering) or you will",
        "break chunked-prefill / spec-decode interactions.",
        "",
        "**Pitfalls (historical REVERTs)**",
        "- Raising `--max-num-seqs` past 512 on MI300X → OOM on 671B MoE models.",
        "- `cuda_graph` + dynamic batch sizes → silent recapture cost > savings.",
        "- Chunked prefill without `--max-num-batched-tokens` → tail latency",
        "  regressions invisible to throughput-only benches.",
        "- `torch.compile` on MLA + FP8 (DeepSeek-R1 NSA path) → incompatible;",
        "  disable compile on that path.",
    ]


def _focus_kernel_switch_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the kernel-switch specialist.

    Selects atom-flavoured or canonical sglang/vllm kernel (attention / MoE /
    GEMM) hints based on the active framework.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs (used to pick the
            framework flavour).

    Returns:
        list[str]: Markdown lines for the kernel-switch specialist's focus
        block.
    """
    if _is_atom(inp):
        return [
            "You target **aiter / atom kernels / triton** code (attention,",
            "MoE, GEMM, fused-attention paths).",
            "",
            "**What to read first**",
            "- `aiter/csrc/` and `aiter/aiter/ops/` (CK / hipBLASLt "
            + "wrappers — **shared with sglang and vllm**, so aiter "
            + "patches apply transparently across all three).",
            "- `atom/model_ops/` (atom-specific kernel call sites).",
            "- `atom/quantization/` (atom's FP8 / weight-quant paths).",
            "- `atom/models/` (built-in model implementations; cross-",
            "reference for which kernels each model touches).",
            "- KB anchor `kernel.*` (CDNA3 tiling / MoE / attention / GEMM).",
            "",
            "**Winning techniques to consider**",
            "- aiter env switches (`VLLM_ROCM_USE_AITER=1` umbrella + "
            + "per-op overrides) — the shared aiter surface means the "
            + "same env knobs that work on sglang/vllm carry over to atom.",
            "- Tile-size / occupancy tuning for short-OSL decode.",
            "- Fused-attention enable flags for prefill chunks.",
            "",
            "**Pitfalls**",
            "- Searching `sglang/python/sglang/srt/layers/attention/` on "
            + "an atom box: those paths are empty / absent. Use "
            + "`atom/model_ops/` + shared `aiter/` instead.",
            "- Mixing aiter overrides with `--enforce-eager` invalidates " + "atom's cudagraph captures silently.",
        ]
    return [
        "You target **aiter / SGLang kernels / triton** code (attention,",
        "MoE, GEMM, fused-attention paths).",
        "",
        "**What to read first**",
        "- `aiter/csrc/` and `aiter/aiter/ops/` (CK / hipBLASLt wrappers).",
        "- `sglang/python/sglang/srt/layers/attention/` (backend selection).",
        "- KB anchor `kernel.*` (CDNA3 tiling / MoE / attention / GEMM).",
        "",
        "**Winning techniques to consider**",
        "- Switch `--attention-backend` at the workload's prefill/decode mix.",
        "  IMPORTANT: valid CLI values differ by framework and image version —",
        "  do NOT invent names. Use values from `<framework> serve --help`.",
        "  Example sglang values: `ROCM_AITER_MLA`, `TRITON_MLA`,",
        "  `ROCM_AITER_TRITON_MLA`. Example vLLM values: `ROCM_ATTN`,",
        "  `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `FLASH_ATTN`.",
        "- `VLLM_ROCM_USE_AITER=1` umbrella + per-op overrides for MoE / RMSNorm.",
        "- Tile-size tuning for `M < 256` GEMMs (hipBLASLt vs Triton).",
        "- Fused-attention enable flags for prefill chunks.",
        "",
        "**Pitfalls**",
        "- Using a backend name that does not exist in the current image",
        "  (e.g. `ROCM_FLASH`) → immediate ValueError, wasted budget.",
        "- Forcing AITER MLA on workloads with short OSL — kernel selection",
        "  cost dominates the saving.",
        "- Mixing `--attention-backend` with `--enforce-eager=true` invalidates",
        "  cuda graphs silently.",
        "- Trying triton fp4 paths on CDNA3 without `AMDGCN_USE_BUFFER_OPS=1`.",
    ]


def _focus_comm_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the communication specialist.

    Selects atom-flavoured (single-node, intra-node collectives) or canonical
    multi-node RCCL/NCCL/QuickReduce hints based on the active framework.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs (used to pick the
            framework flavour).

    Returns:
        list[str]: Markdown lines for the communication specialist's focus
        block.
    """
    if _is_atom(inp):
        return [
            "You target **intra-node RCCL / NCCL / QuickReduce / " + "AllReduce** tuning on atom.",
            "",
            "**Atom is single-node only.** Multi-node tensor / data / "
            + "pipeline parallelism is NOT available on atom. Cross-node "
            + "collectives proposals are non-actionable — focus on "
            + "intra-node concerns (rank-local optimisations, "
            + "intra-node NCCL/RCCL config, allreduce algorithm choice "
            + "for the on-box TP group).",
            "",
            "**What to read first**",
            "- `atom/utils/distributed/utils.py` (single-node "
            + "`torch.distributed` helper — NOT a multi-node TP "
            + "orchestration layer).",
            "- `aiter/csrc/quick_reduce/` and RCCL plugin paths "
            + "(shared with sglang/vllm; intra-node only on atom).",
            "- KB anchor `communication.*` (allreduce / QuickReduce / " + "topology).",
            "",
            "**Winning techniques to consider (single-node)**",
            "- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` when " + "intra-node TP allreduce message size > 1MiB.",
            "- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` tuning for " + "the on-box XGMI topology.",
            "- `--enable-dp-attention` (MLA models) — DP-attention "
            + "shifts work onto a TP-free per-rank path, reducing the "
            + "allreduce footprint.",
            "",
            "**Pitfalls**",
            "- Proposing multi-node TP / PP topologies — atom rejects "
            + "them at startup. The Coordinator collapses `--nodes>1` to "
            + "single-node mode on atom (IR-8).",
            "- INT4 QuickReduce at TP=2 — overhead dominates the " + "bandwidth savings on small message sizes.",
        ]
    return [
        "You target **RCCL / NCCL / QuickReduce / AllReduce** code and tuning.",
        "",
        "**What to read first**",
        "- `vllm/distributed/`, `vllm/distributed/parallel_state.py`.",
        "- `aiter/csrc/quick_reduce/` and RCCL plugin paths.",
        "- KB anchor `communication.*` (allreduce / QuickReduce / topology).",
        "",
        "**Winning techniques to consider**",
        "- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` when message size > 1MiB.",
        "- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` tuning per topology.",
        "- TP allreduce vs PP collective trade-offs at high concurrency.",
        "",
        "**Pitfalls**",
        "- INT4 QuickReduce at TP=2 — overhead dominates the bandwidth savings.",
        "- Tuning NCCL env vars without confirming `rocm-smi --showtopo` shows",
        "  the expected XGMI / PCIe topology.",
    ]


def _focus_compiler_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the compiler specialist.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs (unused beyond protocol
            parity; this block is framework-agnostic).

    Returns:
        list[str]: Markdown lines covering torch.compile / inductor / triton /
        AMDGCN codegen hints.
    """
    return [
        "You target **torch.compile / inductor / triton / AMDGCN** codegen",
        "and register-pressure tuning.",
        "",
        "**What to read first**",
        "- `triton/python/triton/runtime/` and `triton/lib/Conversion/`.",
        "- `torch/_inductor/codegen/triton.py` and `torch/_inductor/scheduler.py`.",
        "- KB anchor `compiler.*` (inductor / triton / AMDGCN).",
        "",
        "**Winning techniques to consider**",
        "- `--compilation-config '{\"level\": 3, ...}'` with surgical level=2",
        "  fallback for kernels that don't quantise cleanly.",
        "- `torch._inductor.config.triton.unique_kernel_names` + per-kernel",
        "  autotune cache pinning.",
        "- VGPR-budget tuning via `num_warps` / `num_stages` in @triton.autotune.",
        "",
        "**Pitfalls**",
        "- Raising level=3 globally — some kernels recompile on every batch",
        "  size, wiping the gain.",
        "- VGPR > 256 spills to scratch on CDNA3; profile occupancy first.",
    ]


def _focus_system_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the system specialist.

    Covers KFD driver / ROCm runtime / memory / dispatch-overhead hints.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs for this dispatch.

    Returns:
        list[str]: Markdown lines for the system specialist's focus block.
    """
    return [
        "You target **KFD driver / ROCm runtime / memory / dispatch overhead**.",
        "",
        "**What to read first**",
        "- `/sys/class/kfd/kfd/` (read-only probes via Bash).",
        "- `rocm-smi --showmeminfo VRAM` / `rocm-smi --showtopo`.",
        "- KB anchor `systems.*` (KFD / dispatch / memory).",
        "",
        "**Winning techniques to consider**",
        "- `HSA_ENABLE_SDMA=0` when host↔device dispatch dominates.",
        "- `HIP_HIDDEN_FREE_MEM` to expose hidden VRAM headroom for large MoE.",
        "- `numactl --cpunodebind` pinning at high concurrency.",
        "",
        "**Pitfalls**",
        "- `HSA_ENABLE_SDMA=0` on small-message decode workloads → latency up.",
        "- Disabling `--gpu-memory-utilization` headroom past 0.95 → OOM on",
        "  prefill chunks for long-context workloads.",
    ]


def _focus_candidate_discovery_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the candidate-discovery specialist.

    Frames it as the owner of the whole upstream-candidate funnel: find, rank,
    and judge. It replaced a Coordinator loop that did those three steps with
    a fixed query, a scoring call and an audit call, so the block has to say
    that a candidate is only useful once it carries a verdict and a route.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs for this dispatch.

    Returns:
        list[str]: Markdown lines for the discovery specialist's focus block.
    """
    return [
        "You own the **upstream candidate funnel**: find what is worth landing,",
        "rank it, and judge it. This is a first-class lever alongside",
        "configuration search — not an occasional top-up.",
        "",
        "**What to do**",
        "1. **Find.** Use ``mcp__pr_monitor__*`` + ``WebSearch`` across the",
        "   allowlisted repos for work that addresses the live bottleneck.",
        "   Read the gap and the profile evidence first; a PR that does not",
        "   touch the hot path is not a candidate however good it looks.",
        "2. **Rank.** Order what you found against the current stack, the",
        "   already-tried ledger, and the KB priors you were given. Say why",
        "   the top one is first — the ordering is the deliverable, not a",
        "   by-product.",
        "3. **Judge each one.** Exactly one verdict per candidate:",
        "   - ``already_present`` — the change is already in the installed",
        "     version. Cite the evidence; a guess here silently skips a real win.",
        "   - ``not_applicable`` — wrong framework, wrong arch, or it cannot",
        "     apply to this tree.",
        "   - ``worth_a_bench`` — plus the route: apply the upstream diff",
        "     directly, or have a specialist author against it as inspiration.",
        "     Direct apply needs a git tree and a same-repo candidate; say so",
        "     when either is missing.",
        "",
        "**What to return**",
        "- ``proposal_set`` entries carrying (repo, number, title, diff_url,",
        "  head_sha, files touched, verdict, route, and the reason for each).",
        "  Orchestration reads these and proposes ``integrate_patch`` for the",
        "  ones it wants benched; you do not apply or benchmark anything.",
        "",
        "**Pitfalls**",
        "- Citing a PR without verifying its target framework matches the",
        "  current install.",
        "- An unevidenced ``already_present``: it is the one verdict that",
        "  discards a candidate without ever measuring it.",
        "- Returning candidates with no ordering and no verdicts. A bare list",
        "  puts the work back on the caller that dispatched you.",
    ]


def _recipe_sites_source_lines(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe-site research source; the built-in defaults when unset, nothing when disabled via the sentinel."""
    configured = tuple(s for s in inp.recipe_sites if s)
    if configured and all(s.strip().lower() in RECIPE_SITES_DISABLED_VALUES for s in configured):
        return []
    sites = configured or DEFAULT_RECIPE_SITES
    if not sites:
        return []
    lines = [
        "4. **Verified launch-recipe sites** — structured per",
        "   (model x hardware x quant x strategy) recipe pages carrying",
        "   validated serve flags, env vars, and benchmark numbers. Use",
        "   ``WebFetch`` on the page matching THIS model / GPU / precision",
        "   (fall back to ``WebSearch`` if the exact page 404s). Extract only",
        "   the serve flags, env vars, and reported throughput/accuracy;",
        "   emit them as ``proposal_set`` variants with the page URL in",
        "   ``source``. For a near-miss hardware/quant match, still surface it",
        "   but note the mismatch in ``accuracy_risk``. Sites:",
    ]
    lines.extend(f"   - {site}" for site in sites)
    lines.append("")
    return lines


def _focus_research_scout_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the focus section for the research-scout specialist prompt.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines highlighting already-proven priors and steering the
        scout toward net-new findings.
    """
    proven_lines: list[str] = []
    if inp.already_proven:
        proven_lines.append("**Already proven (warm-start recipe) — do NOT re-mine these; focus on net-new priors:**")
        for item in inp.already_proven[:12]:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            src = str(item.get("source") or "").strip()
            proven_lines.append(f"- {name}" + (f" (source={src})" if src else ""))
        proven_lines.append("")
    return [
        "You are the **research scout** — a read-only collector of",
        "*already-proven* priors. You do NOT benchmark, apply patches, or",
        "decide KEEP/REVERT. Your single deliverable is a prioritised list",
        "of source-backed findings and executable variants.",
        "",
        *proven_lines,
        "**Three research sources (cover all that are reachable)**",
        "1. **Reference launch scripts** — look under",
        "   ``$INFERENCEX_PATH/benchmarks/single_node/`` for scripts",
        "   matching this (model, GPU). Extract every validated env/flag",
        "   and the throughput it reached. ``$INFERENCEX_PATH`` may be",
        "   unset — skip this source silently if so.",
        "2. **Model architecture features** — read the model's",
        "   ``config.json`` (MTP ``num_nextn_predict_layers``, MoE expert",
        "   count / routing, attention type MLA/GQA, quantization support)",
        "   and infer optimizations those features unlock.",
        "3. **Cross-framework / NVIDIA research** — survey PRs, blogs, and",
        "   MLPerf results across frameworks and NVIDIA/TRT-LLM via",
        "   ``WebSearch`` / ``mcp__pr_monitor__*`` for proven wins. Avoid",
        "   re-listing PRs the FRAMEWORK_AGENT phase already covered (the",
        "   Coordinator dedups by PR id, but skip obvious repeats).",
        "",
        *_recipe_sites_source_lines(inp),
        "**Gap computation** — where you find a reference throughput, use",
        "the gap versus our current baseline only to prioritise your hints",
        "(a bigger gap means a higher-priority hint). Do NOT emit competitor",
        "numbers as a structured target: measured competitor baselines are",
        "sourced from InferenceX, never authored by this scout.",
        "",
        "**Output protocol** — use only the top-level ``specialist_done`` fields:",
        "- ``proposal_set``: executable variants using the standard explore",
        "  schema. Put inspected PRs and references in ``pr_evidence`` or",
        "  ``source_evidence`` on each proposal.",
        "- ``new_findings``: list of ``{what, expected_impact, accuracy_risk,",
        "  source, domain_tags[]}``. ``source`` is REQUIRED (PR link / blog /",
        "  MLPerf row / reference script path).",
        "- ``residual_questions``: unanswered questions for the next scout round.",
        "",
        "**Iron rule** — read-only. Never write a patch, never launch a",
        "benchmark, never recommend a phase transition. Turn proven priors",
        "into structured hints and stop.",
    ]


def _focus_static_recon_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the focus section for the static-recon specialist prompt.

    Steers a read-only sub-agent to grep the framework source tree for
    un-bridged capability switches (predicates that silently disable a fast
    path for the current GPU/precision), seeded with a curated checklist, and to
    emit structured bridge candidates rather than patches.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines describing the recon task, the seed checklist, and the
        ``recon`` output block schema.
    """
    checklist_lines: list[str] = []
    if inp.static_recon_checklist:
        checklist_lines = [
            "**Seed checklist (known un-bridged switches for this "
            + "(model, GPU, precision)) — verify each against the LIVE source; "
            + "do not assume it still applies:**",
            inp.static_recon_checklist,
            "",
        ]
    model_info_line = ""
    shared_expert_advisory: list[str] = []
    if inp.model_info:
        try:
            attn = str(inp.model_info.get("attention_type") or "").strip()
            is_moe = bool(inp.model_info.get("is_moe"))
            quant = str(inp.model_info.get("quantization") or "").strip()
            has_shared = bool(inp.model_info.get("has_shared_expert"))
            num_shared = inp.model_info.get("num_shared_experts")
            features = f"attention={attn or '?'} moe={is_moe}"
            if has_shared:
                n_str = str(int(num_shared)) if num_shared is not None else "?"
                features += f" shared_expert=True n_shared={n_str}"
            features += f" quant={quant or '?'}."
            model_info_line = f"Model features: {features}"
            if has_shared:
                shared_expert_advisory = [
                    "**Shared-expert fusion advisory**: this model has always-on shared "
                    + "experts. Confirm whether the shared expert still runs as a separate "
                    + "dense MLP per layer. If yes, investigate folding it into the routed "
                    + "grouped-GEMM path as an always-selected extra expert slot (code-path "
                    + "bridge, not just an env flag). Known caveat: expert parallelism (EP) "
                    + "is unsupported until the expert-map behaviour is explicitly handled.",
                    "",
                ]
        except Exception:  # noqa: BLE001 — advisory rendering only
            model_info_line = ""
            shared_expert_advisory = []
    return [
        "You are the **static-recon specialist** — a read-only reconnaissance",
        "agent. You do NOT benchmark, apply patches, build a worktree, or",
        "decide KEEP/REVERT. Your single deliverable is a prioritised list of",
        "**bridge candidates**: fast paths that *should* be enabled for this",
        f"GPU ({inp.gpu_type or '?'}) + precision ({inp.precision or '?'}) but",
        "are silently disabled in the LIVE framework source.",
        "",
        *([model_info_line, ""] if model_info_line else []),
        *shared_expert_advisory,
        *checklist_lines,
        "**How to hunt (read the LIVE source under the source roots / hint",
        "directories in Section 7):**",
        "1. grep for capability predicates — ``*_supported()`` /",
        "   ``*_enabled()`` / ``is_*()`` guards and feature gates in the",
        "   quantization, linear, fused_moe and attention dispatch paths.",
        "2. For each, determine whether it returns False (or routes to a slow",
        "   fallback) on THIS hardware/precision when a faster path exists",
        "   (e.g. a CUDA-only ``cutlass_*_supported()`` that is always False on",
        "   ROCm and so disqualifies an AITER kernel).",
        "3. Trace the consequence: which GEMM / kernel / backend the code then",
        "   falls back to, and why that is slower.",
        "4. Confirm the bridge is plausible (the faster path exists and only",
        "   the guard / scale-granularity / activation-config blocks it).",
        "",
        "**Output protocol** — emit ONE ``specialist_done`` carrying a",
        "``recon`` block with ``bridge_candidates``: a list of",
        "``{id, predicate_file, predicate_name, why_disabled_here, consequence,",
        "bridge_sketch, domain_hint, confidence}``. ``id`` is a short slug",
        "(reuse the seed checklist id when verifying one), ``predicate_file`` is",
        "the source path you read, ``why_disabled_here`` explains the False",
        "branch on this hardware, ``bridge_sketch`` is the proposed fix (a",
        "sketch — you do NOT write the patch), and ``domain_hint`` is the",
        "specialist that should author it (``freeform`` keeps the whole",
        "mandate). A candidate without ``predicate_file`` + ``why_disabled_here``",
        "is dropped.",
        "",
        "**Iron rule** — read-only. Never write a patch, never launch a",
        "benchmark, never recommend a phase transition. Turn verified source",
        "findings into structured bridge candidates and stop.",
    ]


def _focus_enablement_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Stable enablement-specialist identity blurb.

    The per-task mandate (failure context + the ladder book) is rendered
    separately into the user prompt by ``_section_enablement_playbook`` so this
    system-prompt block stays constant across dispatches (cacheable).

    Args:
        inp: Assembled prompt inputs for the current dispatch (unused; the
            identity is task-independent).

    Returns:
        Prompt lines for the enablement-specialist identity.
    """
    return [
        "You are the **enablement specialist** — an AUTHORING sub-agent for a",
        "(model, backend) combo that is non-runnable OR that boots but fails its",
        "accuracy eval. The gate is RUNNABILITY (the server boots and passes a",
        "minimal inference) or, for an eval-origin dispatch, the real model output",
        "meeting the accuracy floor — not throughput.",
        "",
        "Your deliverable is the smallest **runnable delta** that advances the",
        "boot (or the accuracy) — which may be a serve flag, an in-tree source patch, an",
        "attempt-scoped runtime, or a ``needs_targeted_build`` request. Do NOT",
        "stop at a token registration / two-line alias when the diagnosis says the",
        "architecture is genuinely new: advancing one boot step counts, and a",
        "compiled or from-source need should be requested, not faked. Follow the",
        "ENABLEMENT PLAYBOOK in the task context for the tiered methodology.",
    ]


def _focus_framework_rewrite_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the domain-focus block for framework-level source rewrites.

    Carries the rewrite-pattern taxonomy as a *prior* — the categories, the
    cache-key recipe, the correctness invariants and the switch-manifest
    contract — while leaving the landing points to be found from the measured
    evidence. That split is deliberate: a prior that named specific functions
    would only ever work on one model, whereas the pattern vocabulary transfers
    to any iterative pipeline.

    Args:
        inp: Assembled prompt inputs for the current dispatch (used only for the
            framework name; the taxonomy itself is workload-independent).

    Returns:
        Markdown lines for the framework-rewrite specialist's focus block.
    """
    framework = (inp.framework or "the framework").strip()
    return [
        "You are the **framework rewrite specialist** — an AUTHORING sub-agent",
        f"for **{framework}**, an iterative model pipeline: the same transformer",
        "stack runs once per block, per denoising step, per chunk. Nothing about",
        "request serving applies here (no scheduler, no continuous batching, no",
        "KV-cache admission policy). The wins are the redundant work the loop",
        "structure creates.",
        "",
        "**Where the wins are**",
        "A single step's cost is multiplied by (blocks x steps x chunks), so any",
        "work whose result does not change across that product is dead weight,",
        "and any host round-trip inside it stalls the whole pipeline. Two classes",
        "of cost dominate and neither one owns a GPU kernel, so neither appears in",
        "a kernel breakdown: **redundant recomputation** and **host stalls**.",
        "",
        "**Rewrite pattern taxonomy** (the categories the evidence is labelled with)",
        "- **(a) memoize a step- or block-invariant computation.** A pure function",
        "  called with arguments it already received. Cache the result.",
        "- **(b) hoist a loop-invariant computation out of the loop.** The value is",
        "  logically the same each iteration but rebuilt from scratch, so a cache",
        "  keyed on tensor identity would never hit. Compute once per outer",
        "  iteration and pass it in. **This is usually an enabler**: on its own it",
        "  measures flat, and its value is that it makes (a) start hitting.",
        "- **(c) eliminate a host round-trip or a device-to-host sync.** An object",
        "  collective agreeing on a shape the ranks could derive locally; an",
        "  `.item()` / `.tolist()` / `.cpu()` on the hot path.",
        "- **(d) fuse adjacent collectives, GEMMs or concatenations.** Several",
        "  same-shape payloads issued separately; pack them and issue one.",
        "- **(e) swap an operator implementation for a vendor kernel.** Read this",
        "  off the GPU kernel breakdown, not the host evidence.",
        "- **(f) keep a tensor resident on the device.** A table rebuilt on the",
        "  host and re-uploaded on every use.",
        "- **(g) drop no-op glue.** A dtype cast to the dtype the tensor already",
        "  has; an intermediate materialised only to be immediately consumed.",
        "",
        "**Cache-key recipe (get this wrong and you ship a correctness bug)**",
        "- Key on the COMPLETE argument identity: for a tensor,",
        "  `(data_ptr, shape, dtype, device, _version)`; plus every scalar that",
        "  changes the result. A key missing one input returns another input's",
        "  answer.",
        "- **Key on EVERY value you cache, not just the first one.** Caching a",
        "  `(k, v)` pair under a key derived from `k` alone returns the wrong `v`",
        "  the moment two calls share a `k` identity. A previous attempt shipped",
        "  exactly this and moved the output past the quality band.",
        "- **Pin the source tensors in the cache entry.** Under a caching",
        "  allocator a freed tensor's address is handed straight back to the next",
        "  allocation, so `data_ptr` alone will report a brand-new tensor as a hit.",
        "  Holding a reference to the keyed tensors prevents that.",
        "  Pinning means storing the SOURCE objects you keyed on. Storing the",
        "  computed result is not pinning: the sources are then unreferenced, free",
        "  to be deallocated, and their addresses recycled under a live key. A",
        "  previous attempt wrote `# Pin source tensors` above a line that stored",
        "  only the result — the comment is not the mechanism.",
        "- Check `_version` so an in-place mutation invalidates the entry.",
        "- Do NOT hash tensor *contents* to build a key: that forces a",
        "  device-to-host sync per call and costs more than it saves.",
        "- Bound the cache (small LRU) and size it for the calling pattern: under",
        "  classifier-free guidance the positive and negative branches alternate,",
        "  so a single-entry cache thrashes to a 0% hit rate.",
        "",
        "**Deliverable contract — every rewrite is a default-off switch**",
        "Each rewrite MUST be gated by its own environment switch that defaults",
        "OFF, so that with no switches set the code path is byte-for-byte the",
        "original. This is not a style preference; it is what makes each rewrite",
        "independently measurable and composable, and it is checked:",
        "- a parity leg runs with every switch unset and must reproduce the",
        "  baseline within its noise band, so a rewrite that changes behaviour",
        "  when disabled is rejected;",
        "- accepted switches become search levers, so the orchestrator measures",
        "  each one's own contribution and searches combinations rather than",
        "  taking your bundle as given.",
        "",
        "**The manifest is not optional and not documentation — it is what makes",
        "the guarantees above run.** With a gate the manifest does not declare,",
        "nothing is turned on for the measurement, no parity leg runs and no lever",
        "is registered: the patch is benched as an ordinary diff and whatever it",
        "does when 'off' is never checked. Integration now refuses a deliverable",
        "whose patch reads an `os.environ` switch the manifest omits, so an",
        "undeclared gate costs you the whole attempt.",
        "",
        "Alongside the patch, emit a `framework_switches` manifest — one entry per",
        "switch, each with: `switch` (the env var name), `category` (a taxonomy id",
        "above), `target` (file and symbol), `evidence` (which measured candidate",
        "it addresses), `depends_on` (switches that must also be on for this one",
        "to pay), and `enables` (switches that only pay once this one is on).",
        "",
        "**Declare `depends_on` / `enables` honestly — this is load-bearing.**",
        "An enabler measured alone shows no gain. If you do not declare the",
        "relationship, it is judged on its standalone number, rejected, and every",
        "rewrite that depended on it is silently devalued along with it. Declared,",
        "the whole bundle is benchmarked together and survives on its joint gain.",
        "",
        "**Always keep a fallback path.** Guard the fast path on the shapes and",
        "dtypes it actually requires and fall through to the original code",
        "otherwise, so an unexpected input degrades in speed and not in",
        "correctness.",
        "",
        "**Pitfalls**",
        "- Graph capture (HIP/CUDA graphs) conflicts with lazily populated",
        "  caches: the first call allocates inside the capture. Do not combine.",
        "- Caching across a chunk boundary needs the chunk identity in the key;",
        "  geometry alone repeats between chunks with different contents.",
        "- A switch name that collides with an upstream variable will be honoured",
        "  by upstream code too. Namespace yours.",
    ]


_DOMAIN_FOCUS_TEMPLATES: dict[str, "Callable[[SpecialistPromptInputs], list[str]]"] = {
    "serving_specialist": _focus_serving_specialist,
    "framework_rewrite_specialist": _focus_framework_rewrite_specialist,
    "kernel_switch_specialist": _focus_kernel_switch_specialist,
    "comm_specialist": _focus_comm_specialist,
    "compiler_specialist": _focus_compiler_specialist,
    "system_specialist": _focus_system_specialist,
    "candidate_discovery_specialist": _focus_candidate_discovery_specialist,
    "research_scout_specialist": _focus_research_scout_specialist,
    "static_recon_specialist": _focus_static_recon_specialist,
    "enablement_specialist": _focus_enablement_specialist,
}


@dataclass(frozen=True)
class SpecialistPromptInputs:
    """Typed inputs the Coordinator hands to the prompt builder."""

    # Identity
    task_id: str
    domain: SpecialistDomain
    max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS

    # ``tp`` defaults to 0 (sentinel for "unspecified"), not 1, so
    # comm_specialist doesn't veto its own TP proposals.
    gpu_type: str = ""
    allocated_gpu_ids: tuple[int, ...] = ()
    tp: int = 0
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""
    # Advisory competitor target gap block; direction hint only.
    target_gap_notes: str = ""
    # Already-proven warm-recipe optimizations the research scout should skip.
    already_proven: list[dict[str, str]] = field(default_factory=list)
    # Curated recipe-site URL templates the research scout may mine for
    # verified serve flags / envs; empty falls back to the built-in defaults.
    recipe_sites: tuple[str, ...] = ()
    # Advisory research-hint block; its presence suppresses cold-start fallback.
    research_hints: str = ""
    # Workload context mirrored from SharedState; renders in section 2.
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    # ``framework`` is the active server framework (``sglang`` / ``vllm`` /
    # ``atom``); empty falls back to the canonical sglang/vllm hint blocks.
    # ``framework_version`` is the precise install version (empty => no note).
    framework: str = ""
    framework_version: str = ""

    # Gap statement
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Optional structured KB context. Empty in the RecipeKB-first path.
    kb_subgraph: dict[str, Any] = field(default_factory=dict)

    # Roofline / TraceLens evidence from ``SharedState.last_trace_analyze``;
    # empty dict renders a placeholder.
    roofline_evidence: dict[str, Any] = field(default_factory=dict)

    # Recipe summary from the T0 warm-start recipe search
    # (``recipe_kb_t0._cascade_warm_start_search``)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 lessons — positive priors from prior KEEPs; rendered in the lessons section.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Generic sub_kind passthrough so ``_focus_*`` helpers can specialise.
    sub_kind: str = ""

    # Extra knowledge-domain tags; each contributes a focus block to Section 1.
    extra_focus_tags: tuple[str, ...] = ()

    # Local source navigation hint
    framework_source_roots: tuple[str, ...] = ()
    source_hint_directories: tuple[str, ...] = ()

    # Structured model architecture features mirrored from SharedState.model_info;
    # machine-parseable companion to ``arch_notes``. Empty dict => not warmed.
    model_info: dict[str, Any] = field(default_factory=dict)
    # Pre-rendered static-recon checklist block (Markdown); only populated for
    # the static_recon_specialist dispatch.
    static_recon_checklist: str = ""

    # Enablement dispatch evidence, folded into the §1b mandate. Both are empty
    # for every non-enablement domain, and the mandate degrades gracefully.
    enablement_source_context: str = ""
    enablement_candidate_refs: tuple[str, ...] = ()
    # Env / server-arg layers prior advanced rounds accepted; the bench for this
    # round launches with them, so the mandate has to name them.
    enablement_accepted_config: dict[str, Any] = field(default_factory=dict)

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""

    # Dispatch profile dials (see orchestrator.specialists.profile) that shape
    # single-domain / cross-domain / freeform / bench prompting.
    scope: str = "domain"
    mode: str = MODE_PATCH
    bench: bool = False
    lane: str = "gpu"
    # Free-form task description (only populated when scope == 'freeform').
    task_description: str = ""

    # Coordinator-injected note for a bounded auto-retry of a prior transient
    # (timeout / crash / stale-heartbeat) attempt; empty on the first attempt.
    auto_retry_reason: str = ""

    # Wall-clock budget (seconds) and dispatch start timestamp (ISO-8601 UTC)
    # so the specialist can self-throttle. 0 / "" => not supplied (the budget
    # section renders nothing).
    wall_budget_sec: float = 0.0
    started_at_iso: str = ""

    # Run-status snapshot for §0 MANDATE; 0/[] = absent (section renders gracefully).
    baseline_tput: float = 0.0
    current_tput: float = 0.0
    cumulative_gain_validated: float = 0.0
    keep_threshold_pct: float = 0.0
    applied_stack: list[dict[str, Any]] = field(default_factory=list)

    # Structured mandate payload; routes §0 boilerplate and per-dispatch context.
    task_kind: str = ""
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)
    pr_lead: dict[str, Any] = field(default_factory=dict)
    # "A" = emit_intent, "B" = file write, "" = render both (render-script path).
    exit_channel: str = ""


# Section 1 — Identity & autonomy
def _authors_patches(inp: SpecialistPromptInputs) -> bool:
    """Whether this dispatch may author patches; research mode gets no worktree."""
    return inp.mode == MODE_PATCH


def _section_identity(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 1 (identity & autonomy) of the specialist prompt.

    Appends the per-domain focus block from :data:`_DOMAIN_FOCUS_TEMPLATES`
    when one is registered for the active domain.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the identity section.
    """
    if _authors_patches(inp):
        capability_line = (
            "probe the host via Bash, **author source patches into your isolated"
            " worktree**, and use as many of your ``max_turns`` LLM turns as you need"
        )
        deliverable_line = "(Section 8) carrying ``proposal_set`` + ``patches_written``. The hard"
    else:
        capability_line = "probe the host via Bash, and use as many of your ``max_turns`` LLM turns as you need"
        deliverable_line = "(Section 8) carrying ``proposal_set``. The hard"
    if inp.allocated_gpu_ids:
        leaf_examples = "bench N candidates of one lever at once, or read several subsystems"
        leaf_devices = "inherit your VISIBLE_DEVICES (so they share your GPU and cannot oversubscribe), and "
    else:
        leaf_examples = "read several subsystems at once"
        leaf_devices = ""
    fanout_line = (
        f"Fan-out: to parallelize independent single-shot sub-tasks (e.g. {leaf_examples}), "
        'you MAY ``Task(subagent_type="hyperloom-leaf")``. Leaves are single-turn, '
        f"{leaf_devices}cannot fan out further. Use leaves for breadth; do multi-round "
        "depth (e.g. coordinate-descent autotune) yourself."
    )
    body: list[str] = [
        "## 1. IDENTITY & AUTONOMY",
        "",
        f"You are a fully autonomous **{inp.domain.key}** dispatched by the",
        f"Hyperloom Coordinator. Layer: {inp.domain.layer}.",
        f"KB anchor: {inp.domain.kb_anchor}.",
        "",
        f"Description: {inp.domain.description or '(generic)'}",
        "",
        "You operate **autonomously** inside your domain — no per-step approval",
        "is needed. You have full authority to read any code under the framework",
        "source roots (Section 7), search any public GitHub repo or NVIDIA PR,",
        capability_line,
        "to be thorough. Be creative. Investigate deeply. One-turn shortcuts",
        "are discouraged when a real bottleneck is on the table — but stop once",
        "rounds stop yielding new findings; the wall clock is not the only stop",
        "signal. Quality over quantity: **2 proposals is the norm, 4 the hard",
        "cap**. One real beats two padded; ``empty=true`` beats one padded.",
        "",
        "Division of labour: the Coordinator owns the serving GPU, runs the E2E",
        "benchmark, and decides KEEP/REVERT — you do not have to validate final",
        "throughput yourself. Your single deliverable is ONE final ``specialist_done``",
        deliverable_line,
        "capability boundary is fixed by Section 9 Iron Rules; everything inside",
        "it is yours.",
        "",
        fanout_line,
    ]
    rendered_focus_keys: set[str] = set()
    focus = _DOMAIN_FOCUS_TEMPLATES.get(inp.domain.key)
    if focus is not None:
        body.append("")
        body.append(f"### Domain focus — {inp.domain.key}")
        body.append("")
        body.extend(focus(inp))
        rendered_focus_keys.add(inp.domain.key)
    # Append each extra tag's focus block (multi-tag dispatch).
    for tag in inp.extra_focus_tags:
        tag_domain = domain_for_tag(tag)
        if tag_domain is None or tag_domain.key in rendered_focus_keys:
            continue
        tag_focus = _DOMAIN_FOCUS_TEMPLATES.get(tag_domain.key)
        if tag_focus is None:
            continue
        body.append("")
        body.append(f"### Domain focus — {tag_domain.key}")
        body.append("")
        body.extend(tag_focus(inp))
        rendered_focus_keys.add(tag_domain.key)
    if inp.scope == "domains":
        body.extend(_cross_domain_block(inp))
    elif inp.scope == "freeform":
        body.extend(_freeform_block(inp))
    if inp.allocated_gpu_ids:
        body.extend(_gpu_autonomy_block(inp))
    if inp.auto_retry_reason.strip():
        body.extend(_auto_retry_note_block(inp))
    return body


def _auto_retry_note_block(inp: SpecialistPromptInputs) -> list[str]:
    """Heads-up block when this dispatch is a bounded auto-retry of a prior
    transient (timeout / crash / stale-heartbeat) attempt. Advisory only —
    the mandate is unchanged; the note nudges the specialist to scope its work
    so it finishes within budget this time.

    Args:
        inp: The specialist prompt inputs (reads ``auto_retry_reason``).

    Returns:
        The rendered auto-retry notice lines.
    """
    reason = inp.auto_retry_reason.strip()
    return [
        "",
        "### Auto-retry notice",
        "",
        "Your previous attempt on this task did NOT finish cleanly — the "
        f"Coordinator is re-dispatching you. Reason: ``{reason}``.",
        "This is a transient infrastructure failure (a timeout, crash, or "
        + "silent hang), not a rejection of the approach. Scope your "
        + "investigation so you reach a single ``specialist_done`` within "
        + "``max_turns`` this time: prefer fewer, higher-confidence probes, "
        + "emit heartbeats, and avoid long-running shell that risks the same "
        + "timeout.",
    ]


def _gpu_autonomy_block(inp: SpecialistPromptInputs) -> list[str]:
    """On-GPU autonomy block appended for GPU specialists (those with a card
    allocation). Frames the broad capabilities the specialist has on its own
    leased cards and surfaces the *optional* ``rebench`` helper — none of it is
    a mandate; the Coordinator's ``integrate_patch`` E2E gate stays the single
    authoritative measure of truth.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        The rendered on-GPU autonomy lines.
    """
    cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
    return [
        "",
        "### On-GPU autonomy (your leased cards)",
        "",
        f"You exclusively own GPU card(s) [{cards}] for this task. On those "
        + "cards you are free to do whatever converges on a benched win:",
        "- For kernel/config autotune, search the installed framework/source "
        + "first for maintained benchmark/tuning entrypoints, config lookup "
        + "paths, and nearby config families; prefer those.",
        "- If the built-in path is missing or incomplete, write a small "
        + "source-derived harness around the framework primitive/config override "
        + "API. Use warmups, true-default/current/candidate baselines, "
        + "median/min-of-reps, and an accuracy guard.",
        "- Write and run arbitrary scripts — autotune harnesses, "
        + "microbenchmarks, profilers (rocprof / torch.profiler / your own "
        + "breakdown).",
        "- Start / restart a real server on your own cards and benchmark it " + "however you see fit.",
        "- Profile freely to get a fresh trace after a change — don't rely only "
        + "on the static roofline snapshot you were handed.",
        "- Tune the framework's config-file levers (e.g. MoE/GEMM/attention "
        + "Triton config JSONs) — a missing/untuned config is often the single "
        + "biggest lever.",
        "- Self-check accuracy (advisory ``max_abs_err`` / gsm8k) when you want "
        + "to — the Coordinator gate stays authoritative, so this is guidance, "
        + "not a requirement.",
        "",
        "Optional helper: a ``rebench`` convenience reuses the real Magpie "
        + "serving + benchmark path on your leased cards, so you can get numbers "
        + "directly comparable to the ``integrate_patch`` gate in one call:",
        "    python -m hyperloom.orchestrator.specialists.rebench \\",
        "        --config <magpie.yaml> --output ./scratch/rebench " + "[--extra-args '<server args>']",
        "  It prints a JSON result with ``output_throughput``. It is OPTIONAL "
        + "— you may instead write your own bench/autotune script. Throughput "
        + "does NOT have to come from rebench.",
    ]


def _freeform_block(inp: SpecialistPromptInputs) -> list[str]:
    """Free-form mandate appended when ``scope == 'freeform'``. The
    specialist is NOT bound
    to the domain catalogue — the Orchestration ``task_description`` is the
    whole mandate. The single deliverable is still ONE ``specialist_done``.

    Args:
        inp: The specialist prompt inputs (reads ``task_description``).

    Returns:
        The rendered free-form mandate lines.
    """
    desc = (inp.task_description or "").strip() or "(no task description provided)"
    if _authors_patches(inp):
        reach = "upstream PRs, host probing, source patches"
        deliverable = "``proposal_set`` + ``patches_written``"
    else:
        reach = "upstream PRs, host probing"
        deliverable = "``proposal_set``"
    return [
        "",
        "### Free-form mandate (scope = freeform)",
        "",
        "You are dispatched as a **free-form** specialist: you are NOT bound to "
        + "the domain catalogue above. The Orchestration mandate below is your "
        + f"whole task — investigate it wherever it leads (framework internals, {reach}).",
        "",
        "Mandate from Orchestration:",
        "",
        f"> {desc}",
        "",
        "Set ``scope='freeform'`` on each proposal. Your single deliverable is "
        + f"still ONE ``specialist_done`` carrying {deliverable}. Never "
        + "self-report numeric speedups — the Coordinator measures gain.",
    ]


def _cross_domain_block(inp: SpecialistPromptInputs) -> list[str]:
    """Cross-domain mandate appended when ``scope == 'domains'``. The
    single deliverable is still
    ONE ``specialist_done``; the difference is the patch may span every domain
    in scope and the Critic will hold it to the cross-domain rules.

    Args:
        inp: The specialist prompt inputs (reads ``extra_focus_tags`` /
            ``domain``).

    Returns:
        The rendered cross-domain mandate lines.
    """
    tags = ", ".join(inp.extra_focus_tags) if inp.extra_focus_tags else inp.domain.key
    return [
        "",
        "### Cross-domain mandate (scope = domains)",
        "",
        f"You are dispatched as a **cross-domain** specialist over: {tags}.",
        "You may author a single coherent patch that spans these domains "
        + "together when (and only when) the change must happen jointly — a "
        + "combination no single-domain specialist could surface from within "
        + "its own boundary.",
        "",
        "In your ``specialist_done`` you MUST justify the combination:",
        "- give an independent rationale for the change **within each domain** " + "in scope;",
        "- name the **coupling points** (why these changes must land together) "
        + "and at least one **side effect** of the combination;",
        "- show this is genuine cross-domain synthesis, not a concatenation of "
        + "two independent single-domain edits (that is an explore grid combo, "
        + "not a cross-domain patch).",
        "Set ``scope='domains'`` on the proposal so the Critic attaches the "
        + "cross-domain review rules. Never self-report numeric speedups — the "
        + "Coordinator measures gain.",
    ]


# Section 0 — Mandate (deliverable contract + run status)
def _section_mandate(inp: SpecialistPromptInputs) -> list[str]:
    """Render §0: deliverable contract, run status, and task-kind brief."""
    from hyperloom.inference_optimizer.framework_registry import format_primary_metric

    # Deliverable line based on scope × mode.
    scope = (inp.scope or "domain").lower()
    if scope == "freeform":
        anchor = inp.task_description.split("\n")[0].strip()[:120] if inp.task_description else ""
        deliverable = "freeform investigation — see task description"
    else:
        anchor = inp.gap_canonical_id or ""
        if not _authors_patches(inp):
            deliverable = "findings and up to 6 ranked config variants (read-only; no patch)"
        elif scope == "domains":
            deliverable = "a coupled patch spanning multiple domains + up to 6 ranked config variants"
        else:
            deliverable = "a source patch and/or up to 6 ranked config variants addressing the gap below"

    rows: list[str] = [
        "## 0. MANDATE",
        "",
        f"- deliverable: {deliverable}",
    ]
    if anchor:
        rows.append(f"- anchor: `{anchor}`")

    has_status = inp.baseline_tput > 0 or inp.keep_threshold_pct > 0
    if has_status:
        rows.append("")
        rows.append("Run status (read-only context; do NOT re-state these as your own measurements):")
        fw = inp.framework or None
        if inp.baseline_tput > 0:
            rows.append(f"- baseline: {format_primary_metric(fw, inp.baseline_tput)}")
        if inp.current_tput > 0:
            rows.append(f"- current best: {format_primary_metric(fw, inp.current_tput)}")
        if inp.cumulative_gain_validated != 0:
            rows.append(f"- validated cumulative gain: {inp.cumulative_gain_validated:+.2f}%")
        if inp.keep_threshold_pct > 0:
            rows.append(f"- KEEP threshold this cycle: {inp.keep_threshold_pct:.2f}%")
        if inp.applied_stack:
            stack_items = ", ".join(
                f"{e.get('variant_name', '?')} ({e.get('gain_pct', 0):+.2f}%)" for e in inp.applied_stack[:6]
            )
            rows.append(f"- applied stack: {stack_items}")

    rows.extend(
        [
            "",
            "Judged by: the Coordinator benches your proposals end-to-end against",
            "the sealed baseline and decides KEEP/REVERT; the accuracy gate runs",
            "alongside. You are not asked to prove the number.",
        ]
    )

    kind = (inp.task_kind or "").strip()
    brief = _TASK_KIND_BRIEFS.get(kind, "")
    if brief:
        rows.extend(["", brief])

    if inp.pr_lead:
        title = defang_prompt_structure(str(inp.pr_lead.get("title") or "").strip())
        url = str(inp.pr_lead.get("url") or "").strip()
        diff_url = str(inp.pr_lead.get("diff_url") or "").strip()
        rows.append("")
        if title:
            rows.append(f"PR lead: {title}")
        if url:
            rows.append(f"PR url: {url}")
        if diff_url:
            rows.append(f"Diff: {diff_url} (fetch with WebFetch)")

    if inp.prior_attempts:
        rows.extend(
            [
                "",
                "Already tried this session — avoid the same or equivalent change:",
            ]
        )
        for att in inp.prior_attempts[:20]:
            if not isinstance(att, dict):
                continue
            ref = str(att.get("ref") or "").strip()
            if not ref:
                continue
            status = str(att.get("status") or "?").strip()
            gain = att.get("gain_pct")
            gain_str = f" gain={float(gain):+.2f}%" if isinstance(gain, (int, float)) else ""
            why = str(att.get("why") or "").strip()
            why_str = f" — {why}" if why else ""
            rows.append(f"  - {ref} [{status}]{gain_str}{why_str}")

    return rows


# Section 2 — Hardware context
def _section_hardware(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 2 (hardware + workload context) of the prompt.

    Emits GPU type, TP, HBM, peak TFLOPs, and any populated workload
    fields (precision, concurrency, ISL/OSL, max_model_len, arch notes).

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the hardware-context section.
    """
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    if inp.allocated_gpu_ids:
        rows.append("- allocated specialist GPU ids: " + ", ".join(str(g) for g in inp.allocated_gpu_ids))
    if inp.tp > 0:
        rows.append(f"- TP: {inp.tp}")
    else:
        rows.append(f"- TP: {_NONE_PLACEHOLDER}")
    if inp.hbm_gb > 0:
        rows.append(f"- HBM per GPU: {inp.hbm_gb:.1f} GB")
    if inp.peak_tflops > 0:
        rows.append(f"- Peak TFLOPs (declared): {inp.peak_tflops:.1f}")
    # Workload context.
    workload_rows: list[str] = []
    if inp.precision:
        workload_rows.append(f"- precision: {inp.precision}")
    if inp.conc > 0:
        workload_rows.append(f"- concurrency: {inp.conc}")
    if inp.isl > 0:
        workload_rows.append(f"- ISL (input seq len): {inp.isl}")
    if inp.osl > 0:
        workload_rows.append(f"- OSL (output seq len): {inp.osl}")
    if inp.max_model_len > 0:
        workload_rows.append(f"- max_model_len: {inp.max_model_len}")
    if workload_rows:
        rows.append("")
        rows.append("Workload:")
        rows.extend(workload_rows)
    if inp.arch_notes:
        rows.append("")
        rows.append(f"Model architecture (advisory): {inp.arch_notes}")
    if inp.target_gap_notes:
        rows.append("")
        rows.append(inp.target_gap_notes)
    return rows


# Section 2a — Execution budget (wall-clock)
def _section_execution_budget(inp: SpecialistPromptInputs) -> list[str]:
    """Render the wall-clock budget block so the specialist can self-throttle.

    Renders the concrete WS1 budget (seconds + minutes) and the dispatch start
    timestamp. Returns ``[]`` when no budget was supplied (legacy turn-bounded
    path), so the section is omitted entirely rather than emitting a placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``wall_budget_sec`` /
            ``started_at_iso``).

    Returns:
        The rendered execution-budget section lines, or ``[]`` when no budget
        is set.
    """
    if inp.wall_budget_sec <= 0:
        return []
    mins = inp.wall_budget_sec / 60.0
    rows = [
        "## 2a. EXECUTION BUDGET (wall-clock)",
        "",
        f"- Hard wall-clock budget for this entire dispatch: **{inp.wall_budget_sec:.0f}s (~{mins:.0f} min)**.",
    ]
    if inp.started_at_iso:
        rows.append(f"- Dispatch started at: {inp.started_at_iso} (UTC).")
    rows.extend(
        [
            "- The Coordinator hard-kills your subprocess when this budget is "
            + "exhausted — turns are NOT the stop signal. Scope your work to "
            + "reach a deliverable conclusion inside the budget.",
            "- Self-throttle: check elapsed wall-clock with Bash "
            + "(``date -u +%s`` vs the start above), keep your "
            + "``specialist_done.partial.json`` checkpoint current, and write "
            + "the final ``specialist_done.json`` before the budget runs out so "
            + "your best work is never lost to a kill.",
        ]
    )
    return rows


# Section 3 — Gap statement
def _section_gap(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 3 (gap statement) of the specialist prompt.

    Emits the canonical gap id, layer, symptom, and most-recent evidence
    JSON, or a ``(none)`` placeholder when no gap is set.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the gap-statement section.
    """
    rows = ["## 3. GAP STATEMENT", ""]
    if not inp.gap_canonical_id:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append(f"- gap_canonical_id: `{inp.gap_canonical_id}`")
    if inp.gap_layer:
        rows.append(f"- layer: {inp.gap_layer}")
    if inp.gap_symptom:
        rows.append(f"- symptom: {inp.gap_symptom}")
    if inp.gap_evidence:
        rows.append("")
        rows.append("Most recent evidence:")
        rows.append("```json")
        rows.append(json.dumps(inp.gap_evidence, sort_keys=True, indent=2))
        rows.append("```")
    return rows


# Section 4 — optional KB context
def _is_cold_start(inp: SpecialistPromptInputs) -> bool:
    """Return True when every prior KB/PR/research source is empty, so the
    cold-start directive is injected in place of the KB block.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        True when every prior KB/PR/research source is empty.
    """
    return (
        not inp.kb_subgraph
        and not inp.warm_start_recipe
        and not inp.warm_start_lessons
        and not inp.warm_start_pitfalls
        and not inp.research_hints
    )


def _section_kb_subgraph(inp: SpecialistPromptInputs) -> list[str]:
    """Build the advisory KB-context section of the specialist prompt.

    Falls back to research hints when the structured KB subgraph is empty.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines rendering the KB subgraph (or hint-based fallback).
    """
    rows = ["## 4. KB CONTEXT (optional, advisory)", ""]
    cold = _is_cold_start(inp)
    if not inp.kb_subgraph:
        if inp.research_hints:
            rows.extend(
                [
                    "Structured KB context is empty for this (model, hardware, domain), but "
                    + "the research scout collected source-backed priors this "
                    + "session. Treat these as your advisory prior (co-equal with "
                    + "RecipeKB priors; the Critic still gates the final answer):",
                    "",
                    inp.research_hints,
                    "",
                    "Anchor proposals on these hints where they fit the gap " + "(Section 3) and hardware (Section 2).",
                ]
            )
            return rows
        if cold:
            # Cold-start directive: fall back to the Section 1 defaults, or exit empty with a rationale.
            rows.extend(
                [
                    "**COLD-START MODE — no priors available.**",
                    "",
                    "All prior sources for this gap are empty:",
                    "",
                    "- KB context: ``(none)`` — no RecipeKB warm-start facts or "
                    + "research hints were available for this "
                    + "(model, hardware, domain) tuple.",
                    "- Warm-start recipe: ``(none)`` (Section 5).",
                    "- Use ``mcp__pr_monitor__*`` tools (Section 6) to query PRs on demand.",
                    "",
                    "**Directive — a coin-flip proposal is worse than none.** "
                    + "Treat the *Winning techniques* + *Pitfalls* bullets in "
                    + "Section 1 as your fallback prior and take the **1–2 "
                    + "most conservative, well-attested defaults** that fit "
                    + "the hardware (Section 2) and the gap symptom "
                    + "(Section 3); flag each ``provenance: "
                    + "domain_focus_default`` and call it an unvalidated "
                    + "fallback in the proposal's ``reason``. If none clears "
                    + "that bar, emit ``empty=true`` and cite in ``summary`` "
                    + "which you considered and why each was rejected — a "
                    + "bare empty exit with no rationale reads as a tool "
                    + "failure. Do NOT add a ``confidence`` field: "
                    + "self-reported confidence / gain fields are stripped "
                    + "before review. Record in ``residual_questions`` what "
                    + "RecipeKB, research, or ``mcp__pr_monitor__*`` query a "
                    + "future round should pursue.",
                ]
            )
        else:
            rows.extend(
                [
                    _NONE_PLACEHOLDER,
                    "",
                    "(No structured KB context supplied. Use Sections 1, 3, 5, "
                    + "and 6 plus source inspection; record missing RecipeKB / "
                    + "research / PR questions in ``residual_questions`` so a "
                    + "future round can warm richer advisory context.)",
                ]
            )
        return rows
    rows.append("```json")
    rows.append(json.dumps(inp.kb_subgraph, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


def _vendor_substitution_candidates(hot_kernels: Any) -> list[dict[str, Any]]:
    """Select hot ``aten::`` ops worth re-dispatching to a different backend.

    An ``aten::`` name means the op still runs through PyTorch's own dispatch, so
    an alternative backend is available to the call site. A vendor kernel that is
    already in the trace under its own name has nothing left to swap.

    Args:
        hot_kernels: The ``hot_kernels_top15`` rows from the roofline evidence.

    Returns:
        Qualifying rows ordered by descending GPU share.
    """
    if not isinstance(hot_kernels, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in hot_kernels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name.startswith("aten::"):
            continue
        gpu_pct = entry.get("gpu_pct")
        if not isinstance(gpu_pct, (int, float)) or float(gpu_pct) < _VENDOR_SUBSTITUTION_MIN_GPU_PCT:
            continue
        rows.append(entry)
    rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
    return rows


def _vendor_substitution_directive(hot_kernels: Any) -> list[str]:
    """Render the backend-substitution directive for hot ATen ops (empty when none qualify)."""
    candidates = _vendor_substitution_candidates(hot_kernels)
    if not candidates:
        return []
    rows = [
        "**Backend substitution — these ops still dispatch through PyTorch:**",
        "",
        "| name | gpu_pct | category | call site |",
        "|---|---:|---|---|",
    ]
    for entry in candidates:
        gpu_pct = float(entry.get("gpu_pct") or 0.0)
        rows.append(
            f"| {entry.get('name')} | {gpu_pct:.2f}% | "
            f"{entry.get('kernel_category') or ''} | {entry.get('source_file') or ''} |"
        )
    rows.extend(
        [
            "",
            "You may not rewrite the body of a kernel PyTorch owns. You MAY change "
            "which kernel the **call site** dispatches to, and when the call site is "
            "in the tree you are optimizing that is an ordinary source rewrite — "
            "same class as any other switch you author, behind its own default-off "
            "environment switch, with the original path kept as the fallback.",
            "",
            "Section 7 lists the accelerator libraries installed in this container. "
            "Read them to find an entry point matching the op's semantics and "
            "tensor layout, and verify equivalence on the real shapes before "
            "proposing it. An op holding a large share of device time is worth more "
            "than anything you can win from the glue around it.",
        ]
    )
    return rows


def _section_roofline_evidence(inp: SpecialistPromptInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section from ``inp.roofline_evidence``;
    empty evidence renders a heading + ``(none)`` placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``roofline_evidence``).

    Returns:
        The rendered roofline-evidence section lines.
    """
    rows = ["## 4a. ROOFLINE EVIDENCE", ""]
    ev = inp.roofline_evidence or {}
    if not isinstance(ev, dict) or not ev:
        rows.append(
            "(none — no fresh roofline snapshot has been recorded yet. "
            "The Coordinator auto-enqueues `roofline` at the end of "
            "PRELUDE and again after every 10% watermark crossing; if "
            "you are seeing this, the snapshot is still in-flight.)"
        )
        return rows

    snap_id = ev.get("roofline_snapshot_id")
    if snap_id is not None:
        rows.append(f"**TraceLens snapshot #{snap_id}**")
        rows.append("")

    summary = ev.get("executive_summary") or {}
    if isinstance(summary, dict) and summary:
        rows.append("**Executive Summary:**")
        for label, key in (
            ("Compute %", "compute_pct"),
            ("Idle %", "idle_pct"),
            ("Exposed Comm %", "comm_pct"),
            ("Top bottleneck", "top_bottleneck"),
        ):
            val = summary.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)):
                rows.append(f"- {label}: {float(val):.1f}%")
            else:
                rows.append(f"- {label}: {val}")
        rows.append("")

    roofline = ev.get("kernel_roofline_top15") or []
    if isinstance(roofline, list) and roofline:
        rows.append(
            "**Kernel roofline "
            "(kernel_id | name | gpu_pct | bound | AI | eff_pct | "
            "compute_pct | bandwidth_pct | action):**"
        )
        rows.append("")
        rows.append("| kernel_id | name | gpu_pct | bound | AI | eff_pct | compute_pct | bandwidth_pct | action |")
        rows.append("|---|---|---:|---|---:|---:|---:|---:|---|")
        for k in roofline:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float)) else "—"
            bound = str(k.get("bound_type") or k.get("bottleneck") or "")
            ai = k.get("arithmetic_intensity")
            if ai is None:
                ai = k.get("flops_per_byte")
            ai_str = f"{float(ai):.3g}" if isinstance(ai, (int, float)) else "—"
            eff = k.get("efficiency_percent")
            eff_str = f"{float(eff):.2f}%" if isinstance(eff, (int, float)) else "—"
            comp = k.get("compute_utilization_pct")
            comp_str = f"{float(comp):.2f}%" if isinstance(comp, (int, float)) else "—"
            bw = k.get("bandwidth_utilization_pct")
            bw_str = f"{float(bw):.2f}%" if isinstance(bw, (int, float)) else "—"
            actions = k.get("recommended_actions") or []
            action = str(k.get("suggestion") or "")
            if not action and isinstance(actions, list) and actions:
                action = str(actions[0])
            rows.append(
                f"| `{kid}` | {name} | {gpu_pct_str} | {bound} | {ai_str} | "
                f"{eff_str} | {comp_str} | {bw_str} | {action} |"
            )
        rows.append("")

    hot = ev.get("hot_kernels_top15") or []
    if isinstance(hot, list) and hot:
        rows.append("**Top hot kernels (kernel_id | name | gpu_pct | bottleneck | source_file):**")
        rows.append("")
        rows.append("| kernel_id | name | gpu_pct | bottleneck | source_file |")
        rows.append("|---|---|---:|---|---|")
        for k in hot:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float)) else "—"
            bottleneck = str(k.get("bottleneck") or "")
            src = str(k.get("source_file") or "")
            rows.append(f"| `{kid}` | {name} | {gpu_pct_str} | {bottleneck} | {src} |")
        rows.append("")

    directive = _vendor_substitution_directive(hot)
    if directive:
        rows.extend(directive)
        rows.append("")

    analysis_path = str(ev.get("analysis_md_path") or "")
    if analysis_path:
        rows.append(f"**Full analysis.md path:** `{analysis_path}`")
        rows.append("")
        rows.append(
            "Use the `Read` tool on this path for the full TraceLens "
            "report (~10-20 KB). All section headings are stable: "
            "`## Executive Summary` / `## Top Operations` / "
            "`## Compute Kernel Optimizations` / "
            "`## Kernel Fusion Opportunities` / "
            "`## System-Level Optimizations` / `## Recommendations`."
        )
    return rows


# Section 5 — Recipe summary
def _section_recipe(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 5 (warm-start recipe summary) of the prompt.

    Dumps the T0 warm-start recipe as JSON, or a ``(none)`` placeholder
    when no warm-start recipe was supplied.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the recipe-summary section.
    """
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    if not inp.warm_start_recipe:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append("**find-recipe result:**")
    rows.append("```json")
    rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


# Section 5b — Related lessons (positive priors from prior KEEPs)
def _section_lessons(inp: SpecialistPromptInputs) -> list[str]:
    """Render KB ``kind=lesson`` points from prior KEEPs, compactly
    (statement + measured_impact).

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_lessons``).

    Returns:
        The rendered related-lessons section lines.
    """
    rows = ["## 5b. RELATED LESSONS (prior KEEPs on this model+hw)", ""]
    if not inp.warm_start_lessons:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_lessons:
        # External warm-start data may arrive as a plain string rather than a
        # dict "point"; render the bare statement to tolerate shape drift.
        if isinstance(point, str):
            statement = point.strip()
            if statement:
                rows.append(f"- **{statement}**")
            continue
        if not isinstance(point, dict):
            continue
        attrs = point.get("attrs") or {}
        statement = str(attrs.get("statement") or "").strip()
        if not statement:
            continue
        impact_str = _render_measured_impact(attrs.get("measured_impact"))
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        # validated_count is the strongest cross-session signal; fall back to source_session_id.
        vc = attrs.get("validated_count")
        if isinstance(vc, int) and vc > 1:
            meta_bits.append(f"validated={vc}")
        recent_ids = attrs.get("source_session_ids")
        if isinstance(recent_ids, list) and recent_ids:
            meta_bits.append(f"recent={recent_ids[-1]}")
        else:
            src_sid = str(attrs.get("source_session_id") or "").strip()
            if src_sid:
                meta_bits.append(f"src={src_sid}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        # Version-mismatch annotation; the LLM gets the final call.
        version_note = _format_version_note(inp, attrs)
        rows.append(f"- **{defang_prompt_structure(statement)}**{meta}{version_note}")
        if impact_str:
            rows.append(f"    impact: {impact_str}")
    if len(rows) == 2:  # only the header + blank line, all lessons filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


def _format_version_note(
    inp: SpecialistPromptInputs,
    lesson_attrs: dict[str, Any],
) -> str:
    """Render a ``[from sglang@X.Y, you're on A.B]`` annotation when
    the lesson's framework_version differs; empty when either side is
    unknown or they match.

    Args:
        inp: The specialist prompt inputs (reads ``framework`` /
            ``framework_version``).
        lesson_attrs: The lesson's attrs (reads ``framework_version``).

    Returns:
        The version-mismatch annotation, or "" when unknown or matching.
    """
    lesson_fv = str(lesson_attrs.get("framework_version") or "").strip()
    current_fv = (inp.framework_version or "").strip()
    if not lesson_fv or not current_fv:
        return ""
    if lesson_fv == current_fv:
        return ""
    framework_label = (inp.framework or "framework").strip() or "framework"
    return f" [from {framework_label}@{lesson_fv}, you're on {current_fv}]"


def _render_measured_impact(raw: Any) -> str:
    """Back-compat renderer for ``attrs.measured_impact`` (dict, legacy
    string, or other).

    Args:
        raw: The ``measured_impact`` value (dict, string, None, or other).

    Returns:
        A compact human-readable impact string ("" when ``raw`` is None).
    """
    if isinstance(raw, dict):
        parts: list[str] = []
        gain = raw.get("gain_pct")
        if isinstance(gain, (int, float)):
            parts.append(f"+{float(gain):.2f}%")
        tput = raw.get("throughput_after")
        if isinstance(tput, (int, float)):
            parts.append(f"tput={float(tput):.1f}")
        depth = raw.get("stack_depth_at_apply")
        if isinstance(depth, int):
            parts.append(f"depth={depth}")
        when = str(raw.get("measured_at") or "").strip()
        if when:
            parts.append(when[:10])  # keep yyyy-mm-dd for compactness
        return ", ".join(parts)
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


# Section 5c — Known pitfalls (anti-priors from prior REVERTs)
def _section_pitfalls(inp: SpecialistPromptInputs) -> list[str]:
    """Render KB ``kind=pitfall`` points from prior REVERTs (description +
    severity); framed as forbidden paths, not suggestions.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_pitfalls``).

    Returns:
        The rendered known-pitfalls section lines.
    """
    rows = ["## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)", ""]
    if not inp.warm_start_pitfalls:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_pitfalls:
        # Tolerate plain-string pitfalls alongside the structured dict "point".
        if isinstance(point, str):
            description = point.strip()
            if description:
                rows.append(f"- **{description}**")
            continue
        if not isinstance(point, dict):
            continue
        attrs = point.get("attrs") or {}
        description = str(attrs.get("description") or "").strip()
        if not description:
            continue
        severity = str(attrs.get("severity") or "").strip()
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if severity:
            meta_bits.append(f"severity={severity}")
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        vc = attrs.get("validated_count")
        if isinstance(vc, int) and vc > 1:
            meta_bits.append(f"observed={vc}")
        recent_ids = attrs.get("source_session_ids")
        if isinstance(recent_ids, list) and recent_ids:
            meta_bits.append(f"recent={recent_ids[-1]}")
        else:
            src_sid = str(attrs.get("source_session_id") or "").strip()
            if src_sid:
                meta_bits.append(f"src={src_sid}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        version_note = _format_version_note(inp, attrs)
        rows.append(f"- **{description}**{meta}{version_note}")
    if len(rows) == 2:  # only the header + blank line, all pitfalls filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


# Section 6 — PR query capability
def _section_pr_feed(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 6 (PR query capability) of the specialist prompt.

    When the PR Monitor MCP is available, describes the self-serve query tools
    and lists the global repo allowlist the specialist may query. When
    unavailable, outputs a placeholder.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the PR-query capability section.
    """
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES
    from hyperloom.orchestrator.specialists.domains import PR_QUERY_REPOS

    rows = ["## 6. PR MONITOR", ""]
    if not inp.pr_monitor_available:
        rows.append("(unavailable: pr_monitor disabled)")
        return rows
    _prefix = "mcp__pr_monitor__"
    tool_shorts = sorted(t[len(_prefix) :] for t in PR_MONITOR_TOOL_NAMES if t.startswith(_prefix))
    rows += [
        "Use ``mcp__pr_monitor__*`` tools to query PRs on demand:",
        " / ".join(f"``{t}``" for t in tool_shorts),
        "",
        "Repos you may query:",
    ]
    for repo in PR_QUERY_REPOS:
        rows.append(f"- {repo}")
    return rows


# Section 7 — Local source navigation hint
def _section_source_hint(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 7 (local source navigation hint) of the prompt.

    Lists the installed source roots and per-domain focus
    directories, or a ``(none)`` placeholder when neither is supplied.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the source-hint section.
    """
    rows = ["## 7. LOCAL SOURCE NAVIGATION HINT", ""]
    if not inp.framework_source_roots and not inp.source_hint_directories:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if inp.framework_source_roots:
        rows.append("Installed source roots (read-only):")
        for p in inp.framework_source_roots:
            rows.append(f"- {p}")
    if inp.source_hint_directories:
        rows.append("")
        rows.append("Focus directories for this domain:")
        for p in inp.source_hint_directories:
            rows.append(f"- {p}")
    rows.append("")
    rows.append(
        "These trees are read-only. Use Read / Grep / Glob to navigate. "
        "Do NOT attempt Edit / Write / git apply on these trees."
    )
    rows.append("")
    rows.append(
        "Use ``WebSearch`` to look up the latest upstream version of the local "
        "repo and compare the implementation you intend to modify against what "
        "is there now. Use ``WebFetch`` to read the relevant file or PR "
        "directly — before authoring a patch, confirm whether the upstream "
        "repo already contains the fix or optimization you are about to write."
    )
    return rows


# Section 8 — Output protocol
def _section_output_protocol(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 8 (output protocol) of the specialist prompt."""
    workspace = inp.workspace_path or "<workspace>"
    channel = (inp.exit_channel or "").upper().strip()
    authors_patches = _authors_patches(inp)

    exit_lines: list[str] = []
    if channel == "A" or channel == "":
        exit_lines.extend(
            [
                "**Exit — ``emit_intent`` tool:** call ``emit_intent`` exactly once",
                "with intent type ``specialist_done`` and the payload schema below.",
            ]
        )
    if channel == "B" or channel == "":
        if channel == "":
            exit_lines.append("")
        exit_lines.extend(
            [
                "**Exit — file write (subprocess runtime):** write the same payload to",
                f"``{workspace}/specialist_done.json`` as your **absolute last action**.",
                "The dispatcher polls for that file as the exit signal; stop after writing.",
            ]
        )

    if authors_patches:
        patch_fields = [
            "- ``patches_written`` (PR-A2) lists paths (relative to your",
            "  workspace or worktree) of any unified-diff patch files you",
            "  authored this round. Empty list = no patches; downstream",
            "  ``integrate_patch`` action skips when empty.",
            "- ``artifacts_written`` lists any non-diff tuned artifacts to install",
            "  (e.g. an autotuned config JSON) as objects ``{source, target, kind,",
            "  description}``: ``source`` is a path inside your worktree, ``target``",
            "  is the install path — PREFER a framework-relative path; an absolute",
            "  path is accepted only if it resolves inside an allowlisted framework",
            "  root. ``integrate_patch`` backs up the target, installs the artifact,",
            "  runs the same E2E gate, and restores the backup on REVERT. A non-diff",
            "  tuned artifact is a FULL result — set ``empty=false`` when",
            "  ``artifacts_written`` is non-empty.",
        ]
        no_output = "  AND no ``patches_written``/``artifacts_written``; in that case"
    else:
        patch_fields = []
        no_output = "  and no findings; in that case"
    empty_rule = [
        "- ``empty=true`` is legitimate ONLY when you have no actionable proposals",
        no_output,
        "  ``proposal_set=[]`` and you must put the reason in ``summary``.",
    ]

    return [
        "## 8. OUTPUT PROTOCOL",
        "",
        *exit_lines,
        "",
        "**Messages from the Orchestrator (check this as you work):** read",
        f"``{workspace}/inbox.json`` whenever you finish a step. It is a JSON",
        "list of ``{from, ts, body}`` entries, absent until the Orchestrator",
        "sends one. It is how the Orchestrator answers a question you raised",
        "or redirects you mid-run — if it tells you the mandate changed,",
        "follow it rather than finishing the original plan. Never write to",
        "this file.",
        "",
        "**Incremental checkpoint (do this throughout the run):** every time",
        "you reach a new finding or finish a candidate, rewrite your",
        "best-so-far payload to",
        f"``{workspace}/specialist_done.partial.json`` (write to",
        f"``{workspace}/specialist_done.partial.json.tmp`` first, then rename",
        "over the partial so a reader never sees a half-written file). This",
        "partial uses the **same payload schema** as the final file but does",
        "**NOT** end the run — keep working. There is a wall-clock budget; if",
        "you are stopped before finishing, whatever is in the partial is",
        "preserved as your result, so keep it current. The Orchestrator also",
        "reads each rewrite while you are still running — it is how you report",
        "direction and raise ``residual_questions`` early enough to get an",
        "answer back through ``inbox.json``. Write the final",
        "``specialist_done.json`` (which ends the run) only once, as your",
        "absolute last action.",
        "",
        "Payload schema (identical for both channels):",
        "",
        "```json",
        json.dumps(
            {
                "intent_type": "specialist_done",
                "payload": {
                    "gap_canonical_id": inp.gap_canonical_id or "<echo from dispatch>",
                    "domain": inp.domain.key,
                    "proposal_set": [
                        {
                            "name": "<unique-in-round>",
                            "extra_args": "--example-flag value",
                            "extra_envs": {"EXAMPLE_ENV": "1"},
                            "remove_args": ["--harmful-base-flag"],
                            "unset_envs": ["HARMFUL_BASE_ENV"],
                            "args_mode": "append",
                            "reason": "why this might help the gap",
                            "atomic": False,
                            "kb_evidence": [],
                            "pr_evidence": [],
                            "source_evidence": [],
                        }
                    ],
                    **({"patches_written": []} if authors_patches else {}),
                    "empty": False,
                    "summary": "≤ 500 char overview of what you tried this round",
                    "confidence": 0.6,
                    "new_findings": [],
                    "residual_questions": [],
                },
            },
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "Field contract:",
        "",
        (
            "- ``proposal_set`` items reuse the explore variant schema: "
            "``extra_args`` / ``extra_envs`` add or override knobs, "
            "``remove_args`` removes inherited server flags before appending, "
            "``unset_envs`` removes inherited env vars before applying "
            "``extra_envs``, and ``args_mode='replace'`` runs without "
            "inherited server args. Use removal fields when a user/base "
            "knob may be harmful; do not simulate deletion by adding an "
            "unrelated flag."
        ),
        (
            "- ``atomic`` (bool, default false): set ``true`` when this "
            "proposal's ``extra_args`` / ``extra_envs`` are a **coupled set "
            "that only works together** and MUST be benched as one variant "
            "(e.g. enabling MTP/speculative decoding REQUIRES a paired "
            "``--gpu-memory-utilization`` reduction so the draft model has "
            "headroom — split them and each half OOMs or shows no gain). "
            "Orchestration is instructed to dispatch an ``atomic`` proposal "
            "verbatim, without splitting, dropping, or re-deriving its flags. "
            "Put every co-required flag in THIS one entry; do not scatter a "
            "coupling across several proposals."
        ),
        (
            "- ``proposal_set``: **2 entries is the norm, 4 the hard cap.** You "
            "are a curator, not a brainstormer: rank by expected gain x "
            "confidence, drop anything contradicting ``kb_subgraph`` / "
            "``pr_evidence``, and stop at 2. A 3rd or 4th must beat the median "
            "of the first two. Padding is a failure, not thoroughness: each "
            "weak entry costs a Critic reject and a slot on the serial "
            "benchmark queue. One real proposal is a better round than two "
            "padded ones, and ``empty=true`` is better than one."
        ),
        (
            "- The Critic reviews each surviving variant against the KB "
            + "before benchmarking, so a marginal-quality proposal costs you "
            + "a reject (and a pitfall fact that will warn future sessions "
            + "off the same dead-end)."
        ),
        *patch_fields,
        *empty_rule,
        "- ``new_findings`` is a list of learned items. Research scouts must",
        "  emit source-backed ``{what, source, expected_impact, accuracy_risk,",
        "  domain_tags[]}`` records.",
        "- ``residual_questions`` carries to the next specialist round.",
        "",
        "**Heartbeat (Channel B only):** When running in subprocess mode,",
        f"write ``{workspace}/heartbeat.json`` periodically (≤5 min apart)",
        "via Bash so the dispatcher knows you are still alive. Format:",
        '``{"ts": "<iso8601>", "status": "running", "note": "<short>"}``.',
        "Going silent past 5 minutes kills your subprocess.",
        "",
        (
            f"Hard cap: at most **{inp.max_turns}** LLM turns. Silence past "
            "the cap = stale (robustness will synthesize an empty done)."
        ),
    ]


# Section 9 — Iron rules
def _section_iron_rules(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 9 (iron rules) of the specialist prompt."""
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    authors_patches = _authors_patches(inp)
    if inp.allocated_gpu_ids:
        cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
        gpu_rule = [
            f"1. You EXCLUSIVELY own GPU card(s) [{cards}] for this task. On",
            "   those cards do whatever you want: edit code, build, start/stop",
            "   your own servers, profile, autotune, install tuned artifacts,",
            "   and run real benchmark loops. The ONE thing you must NOT do:",
            "   touch the production serving process or its cards — co-residing",
            "   on them would corrupt both your measurement and production.",
            "   Manage only processes YOU started, by their own PID/PGID.",
        ]
    else:
        last = "   try and optionally author patches." if authors_patches else "   try."
        gpu_rule = [
            "1. You have no GPU allocation for this task, so do not run GPU",
            "   benchmarks or start servers. The ONE hard boundary that always",
            "   holds: never touch the production serving process / its cards /",
            "   port 8888. The Coordinator runs benchmarks; you propose what to",
            last,
        ]
    if authors_patches:
        integration_rule = [
            "2. **You MAY** produce changes for integration, but stage them ONLY",
            f"   inside your own worktree at ``{workspace}/``. Two output kinds:",
            "   - Unified-diff patches: ``git diff > patches/NNN_<slug>.patch``",
            "     from the worktree; list paths in ``patches_written``.",
            "   - Tuned non-diff artifacts (e.g. an autotuned config JSON): write",
            "     under the worktree and list in ``artifacts_written`` as",
            "     ``{source, target, kind, description}``.",
            "   **NEVER** ``git apply`` / ``git commit`` against the shared",
            "   ``framework_source_roots`` directly — ``integrate_patch`` is",
            "   the single integration point.",
        ]
    else:
        integration_rule = [
            "2. **Read-only dispatch:** you have no worktree and MUST NOT author",
            "   patches or edit ``framework_source_roots``. Report what you found",
            "   through ``specialist_done``; a patch-capable specialist authors any",
            "   source change you recommend.",
        ]
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.3)",
        "",
        *gpu_rule,
        *integration_rule,
        "3. Only ``specialist_done``, ``send_message``, and ``alert`` are",
        "   accepted intents; all others are dropped.",
        "4. You **MUST** finish within ``max_turns`` LLM turns and end with",
        "   exactly one ``specialist_done`` exit signal. Silence past the cap",
        "   synthesizes an empty done.",
        f"5. Use ``{workspace}/`` for ALL writes. The dispatcher exposes only",
        "   this directory + read-only access to ``framework_source_roots``",
        "   and ``SESSION_DIR``.",
        "6. On tool error or no useful action left, emit",
        "   ``specialist_done{empty=true, summary='<why>'}``.",
        f"7. {BASH_KILL_SAFETY_PREAMBLE}",
    ]


# Section 1b — Enablement playbook (per-task; enablement specialist only)
def _section_enablement_playbook(inp: SpecialistPromptInputs) -> list[str]:
    """Render the per-task enablement mandate + ladder book into the user prompt.

    Classifies the failure carried in ``gap_symptom`` / ``gap_evidence`` and
    renders the mandate's ``task_description`` (which embeds the ladder book) from
    ``framework_agent.enablement_ops.build_mandate``. Kept in the user prompt so
    the cached system prompt stays task-independent.

    The dispatch's own evidence — source lines near the offending site (plus the
    checkpoint weight inventory on a weight-init failure) and the ranked bridging
    refs — is folded in from ``enablement_*`` inputs. Without them the mandate
    renders its generic skeleton, so the agent is told to find a bridge while the
    candidates already discovered for it are withheld.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        list[str]: The enablement-playbook section lines.
    """
    from hyperloom.agents.framework.enablement import EnablementRequest
    from hyperloom.agents.framework.enablement_ops import build_mandate

    model = str((inp.gap_evidence or {}).get("model") or "").strip()
    req = EnablementRequest(
        framework=(inp.framework or "").strip().lower(),
        model=model or "(target model)",
        repo_url="",
        launch_log=inp.gap_symptom or "",
        gpu_type=(inp.gpu_type or "").strip().lower(),
    )
    mandate = build_mandate(
        req,
        candidate_refs=inp.enablement_candidate_refs,
        source_context=inp.enablement_source_context,
    )
    rows = ["## 1b. ENABLEMENT PLAYBOOK", ""]
    rows.extend(mandate.task_description.splitlines())
    acc_cfg = inp.enablement_accepted_config or {}
    acc_envs = {str(k): str(v) for k, v in (acc_cfg.get("extra_envs") or {}).items()}
    acc_args = str(acc_cfg.get("extra_server_args") or "").strip()
    if acc_envs or acc_args:
        rows.append("")
        rows.append("### Config already in effect")
        rows.append(
            "Prior rounds accepted these and your bench launches with them. Re-propose "
            "one only to override or remove it."
        )
        rows.extend(f"- `export {k}={v}`" for k, v in acc_envs.items())
        if acc_args:
            rows.append(f"- server args: `{acc_args}`")
    return rows


# Section 1a — PD-disaggregation
def _section_pd_disaggregation(inp: SpecialistPromptInputs) -> list[str]:
    """§1a — PD-disaggregation context (omitted unless pd_mode==disaggregated).

    Surfaces the prefill/decode split so the specialist targets each role's
    distinct bottleneck (prefill: compute / TTFT; decode: memory-bandwidth /
    TPOT) and the KV-transfer path, instead of treating the server as one pool.
    Reads the multi-node state directly; returns ``[]`` on the single-node /
    aggregated paths so the section is dropped.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs (unused; the
            PD topology is read from multi-node state).

    Returns:
        list[str]: The PD-disaggregation section lines, or ``[]`` when not
        disaggregated.
    """
    try:
        from hyperloom.orchestrator.actions.executors._multi_node_env import (
            pd_topology_from_state,
        )

        pd = pd_topology_from_state()
    except Exception:
        return []
    if not pd:
        return []
    tb = pd.get("transfer_backend") or "the KV transfer backend"
    return [
        "## 1a. PD-DISAGGREGATION (prefill/decode separated)",
        "",
        "This deployment runs **prefill/decode disaggregation**: prefill and "
        "decode execute on SEPARATE GPU nodes, exchanging KV cache via "
        f"`{tb}`. Optimize each role for its OWN bottleneck — do NOT treat the "
        "server as a single pool:",
        "",
        f"- **Prefill** ({pd.get('prefill_nodes')} node(s), tp={pd.get('prefill_tp')}, "
        f"ep={pd.get('prefill_ep')}): compute-bound; drives **TTFT**. Levers: "
        "chunked-prefill size, attention backend for long ISL, MoE dispatch, "
        "prefill token/batch budget.",
        f"- **Decode** ({pd.get('decode_nodes')} node(s), tp={pd.get('decode_tp')}, "
        f"ep={pd.get('decode_ep')}): memory-bandwidth-bound; drives **TPOT/ITL**. "
        "Levers: KV-cache / mem-fraction, max-running-requests, dp-attention, "
        "decode MoE a2a backend.",
        f"- **KV transfer** (`{tb}`): watch bootstrap / transfer stalls; RDMA/IB "
        "device selection affects decode start latency.",
        "- **Balance**: tune the prefill:decode node/TP ratio to the ISL:OSL "
        "shape — a saturated role caps end-to-end throughput.",
        "",
        "Per-role GPU telemetry is in the benchmark report's "
        "`gpu_monitor_by_role` (prefill vs decode util / power / VRAM); use it to "
        "confirm which role is the bottleneck BEFORE proposing changes.",
    ]


# Top-level assembler
def build_specialist_prompts(inp: SpecialistPromptInputs) -> tuple[str, str]:
    """Assemble the full specialist prompt from its section builders.

    The system prompt carries the immutable contract (identity, output
    protocol, iron rules) and the user prompt carries the per-task context
    (hardware, optional PD-disaggregation, execution budget, gap, KB,
    roofline, recipe, lessons, pitfalls, PR feed, source hint and
    orchestration notes). The split lets the LLM backend cache the system
    prompt across specialists.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        tuple[str, str]: The ``(system_prompt, user_prompt)`` pair.
    """

    system_sections = [
        _section_identity(inp),
        _section_output_protocol(inp),
        _section_iron_rules(inp),
    ]
    if inp.domain.key == "enablement_specialist":
        # Pre-baseline enablement: the perf context (roofline / recipe / lessons /
        # pitfalls / KB subgraph) is noise when the server cannot boot or the
        # baseline fails its accuracy eval. Carry only the failure, the tiered
        # playbook, and the tools to discover + navigate a fix.
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_enablement_playbook(inp),  # § 1b (mandate + ladder book)
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    else:
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (PD-disaggregation; omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_kb_subgraph(inp),
            _section_roofline_evidence(inp),
            _section_recipe(inp),
            _section_lessons(inp),
            _section_pitfalls(inp),
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    if inp.notes:
        user_sections.append(
            [
                "## 10. NOTES FROM ORCHESTRATION",
                "",
                defang_prompt_structure(inp.notes),
            ]
        )

    def _flatten(sections: list[list[str]]) -> str:
        """Join section line-lists into a single newline-separated string.

        Inserts one blank line between non-empty sections and appends a
        trailing newline.

        Args:
            sections (list[list[str]]): The per-section lists of lines.

        Returns:
            str: The flattened prompt text.
        """
        out: list[str] = []
        for sec in sections:
            if not sec:  # skip omitted sections (e.g. no execution budget)
                continue
            if out:
                out.append("")
            out.extend(sec)
        return "\n".join(out) + "\n"

    return _flatten(system_sections), _flatten(user_sections)


__all__ = [
    "SpecialistPromptInputs",
    "build_specialist_prompts",
]
