# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist sub-agent prompt assembler.

Returns ``(system_prompt, user_prompt)``: the system prompt carries the
immutable contract (identity / output protocol / iron rules) so the
backend can cache it across specialists; the user prompt carries per-task
context (hardware / gap / KB / recipe / PR / source hint). Each section is
independently nullable, rendering ``(none)``. Pure function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..specialists.domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SpecialistDomain,
    domain_for_tag,
)


_NONE_PLACEHOLDER = "(none)"


# Forbids global process cleanup that could kill the optimizer's serving /
# benchmark process. Shared by bash-enabled specialist and leaf prompts.
BASH_KILL_SAFETY_PREAMBLE = (
    "Do NOT run global process cleanup. Never run `ps aux | grep ... | xargs "
    "kill`, `pgrep -f ... | xargs kill`, or `killall` — these can kill the "
    "optimizer's serving / benchmark process. Only manage processes you "
    "started yourself, by their own PID."
)


# Soft cap on ``proposal_set`` size; re-exported so the prompt-side cap and the
# runner-side hard truncate stay aligned.
from hyperloom.orchestrator.policy.gate import (
    DEFAULT_SPECIALIST_MAX_PROPOSALS,
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


def _focus_cross_framework_rewrite_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for cross-framework feature porting.

    Static porting methodology only (cacheable in the system prompt). The
    per-task landing data (source diff, symbol-level landing points, target
    module current source) is injected separately via the Coordinator seed.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs (unused; the block is
            framework-agnostic methodology).

    Returns:
        list[str]: Markdown lines for the cross-framework rewrite focus block.
    """
    return [
        "You **PORT a feature across frameworks** (e.g. SGLang <-> vLLM). This",
        "is a REWRITE task, NOT a `git apply`.",
        "",
        "**Hard rules**",
        "- The upstream diff targets a DIFFERENT framework's repo layout / API.",
        "  It can NEVER be applied directly. Re-implement the EQUIVALENT feature",
        "  against the TARGET framework's live source in your worktree.",
        "- Land ONLY at the provided symbol-level landing points and their direct",
        "  dependencies. Do not refactor unrelated modules (blast-radius control).",
        "- Match the target framework's abstractions, naming and error handling;",
        "  translate SEMANTICS, not syntax.",
        "",
        "**Method**",
        "1. Read the source diff as INSPIRATION for the feature's intent.",
        "2. Read the target module's current source (in the seed) to learn its",
        "   API surface, data structures and call-order contracts.",
        "3. Re-implement the feature at the landing points using the target API.",
        "4. SELF-CHECK before finishing: the touched modules must import / compile",
        "   cleanly (`python -c 'import ...'` / `py_compile`). A patch that fails",
        "   self-check is NOT a deliverable — fix it or report blocked.",
        "",
        "**Deliverable**",
        "- A unified-diff source patch (`patches_written`) against the TARGET",
        "  framework source. A pure config-lever proposal is NOT sufficient for a",
        "  cross-framework port.",
        "- Echo `source_framework` / `target_framework` in your proposal and set",
        "  the `provenance` exactly as instructed in the task seed so the KB",
        "  ledger records the cross-framework outcome.",
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


def _focus_pr_intel_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the PR-intel specialist.

    Frames the specialist as a cross-repo PR researcher that surfaces
    upstream PRs / commits / issues rather than proposing config knobs.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs for this dispatch.

    Returns:
        list[str]: Markdown lines for the PR-intel specialist's focus block.
    """
    return [
        "You are a **cross-repo PR researcher**. Your role is NOT to propose",
        "configuration knobs — it is to surface PRs / commits / issues from",
        "(ROCm/aiter, sgl-project/sglang, ROCm/vllm, triton-lang/triton,",
        "ROCm/rccl) that other specialists should follow up on.",
        "",
        "**What to do**",
        "- Use ``mcp__pr_monitor__*`` + ``WebSearch`` to find recent PRs",
        "  related to the gap.",
        "- For each PR, extract: (repo, number, title, summary, files",
        "  touched, NVIDIA equivalent if any).",
        "- Surface as ``proposal_set`` entries where ``provenance`` = research",
        "  and ``pr_evidence`` is non-empty. Do NOT propose source patches",
        "  yourself — that's the kernel-switch / serving specialist's job once",
        "  they read your PR list.",
        "",
        "**Pitfalls**",
        "- Citing a PR without verifying its target framework matches the",
        "  current install.",
        "- Spending more than one round; PR intel is best dispatched once",
        "  per gap and used as input to other specialists.",
    ]


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
        "of research hints, each with an explicit source.",
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
        "**Gap computation** — where you find a reference throughput,",
        "compute the gap versus our current baseline and let the gap size",
        "drive each hint's priority.",
        "",
        "**Output protocol** — emit ONE ``specialist_done`` carrying a",
        "``research`` block:",
        "- ``hints``: list of ``{what, expected_impact, accuracy_risk,",
        "  source, domain_tags[]}``. ``source`` is REQUIRED (PR link / blog",
        "  / MLPerf row / reference script path); a hint without a source",
        "  is dropped.",
        "- optional ``competitor_target``: ``{gpu, model, framework,",
        "  precision, per_conc:[{conc, tput_per_gpu, tpot_ms,",
        "  interactivity, source}], notes}`` — every per-conc number MUST",
        "  carry its own ``source`` or it is discarded.",
        "- optional ``prs_fetched`` / ``pr_diffs_read`` / ``nvidia_refs``:",
        "  ids you actually inspected (feeds exploration-depth tracking).",
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
        *( [model_info_line, ""] if model_info_line else [] ),
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
        "EXPLORE specialist that should author it (``freeform`` keeps the whole",
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
    """Build the enablement-specialist focus by delegating to ``build_mandate``.

    Classifies the failure carried in ``gap_symptom`` / ``gap_evidence`` and
    renders the mandate's ``task_description`` verbatim from
    ``framework_agent.enablement_ops.build_mandate``.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines rendered from the enablement mandate.
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
    mandate = build_mandate(req)
    lines = [
        "You are the **enablement specialist** — an AUTHORING sub-agent whose",
        "single deliverable is a bridging patch that makes a currently",
        "non-runnable (model, backend) combo *boot and pass a minimal",
        "inference*. The gate is RUNNABILITY, not throughput.",
        "",
    ]
    lines.extend(mandate.task_description.splitlines())
    return lines


_DOMAIN_FOCUS_TEMPLATES: dict[str, "Callable[[SpecialistPromptInputs], list[str]]"] = {
    "serving_specialist": _focus_serving_specialist,
    "cross_framework_rewrite_specialist": _focus_cross_framework_rewrite_specialist,
    "kernel_switch_specialist": _focus_kernel_switch_specialist,
    "comm_specialist": _focus_comm_specialist,
    "compiler_specialist": _focus_compiler_specialist,
    "system_specialist": _focus_system_specialist,
    "pr_intel_specialist": _focus_pr_intel_specialist,
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
    # Soft cap on ``proposal_set`` size (rendered into Sections 1 + 8).
    max_proposals: int = DEFAULT_SPECIALIST_MAX_PROPOSALS

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

    # Recipe summary from T0 ``find-recipe``
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 lessons — positive priors from prior KEEPs; rendered in the lessons section.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # KG graph-recommended knobs (cross-recipe IMPROVES candidates via the
    # architecture family graph). Each entry: ``{knob, expected_gain, confidence, source}``.
    kg_recommended_knobs: list[dict[str, Any]] = field(default_factory=list)
    # KG graph-guided config knobs (journal ``KNOB_IMPROVES``) carrying runnable
    # ``args``/``envs``. Each entry: ``{knob, args, envs, name, expected_gain, confidence, source}``.
    kg_guided_knobs: list[dict[str, Any]] = field(default_factory=list)
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

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""

    # Dispatch profile dials (see orchestrator.specialist_profile) that shape
    # single-domain / cross-domain / freeform / bench prompting.
    scope: str = "domain"
    mode: str = "patch"
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


# Section 1 — Identity & autonomy
def _section_identity(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 1 (identity & autonomy) of the specialist prompt.

    Appends the per-domain focus block from :data:`_DOMAIN_FOCUS_TEMPLATES`
    when one is registered for the active domain.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the identity section.
    """
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
        "probe the host via Bash, **author source patches into your isolated",
        "worktree**, and use as many of your ``max_turns`` LLM turns as you need",
        "to be thorough. Be creative. Investigate deeply. One-turn shortcuts",
        "are discouraged when a real bottleneck is on the table. Quality is",
        "scored over quantity: cap your final ``proposal_set`` at the",
        f"**top-{inp.max_proposals}** ranked picks (see Section 8).",
        "",
        "Division of labour: the Coordinator owns the serving GPU, runs the E2E",
        "benchmark, and decides KEEP/REVERT — you do not have to validate final",
        "throughput yourself. Your single deliverable is ONE final ``specialist_done``",
        "(Section 8) carrying ``proposal_set`` + ``patches_written``. The hard",
        "capability boundary is fixed by Section 9 Iron Rules; everything inside",
        "it is yours.",
        "",
        "Fan-out: to parallelize independent single-shot sub-tasks (e.g. bench "
        + "N candidates of one lever at once, or read several subsystems), you "
        + "MAY ``Task(subagent_type=\"hyperloom-leaf\")``. Leaves are single-turn, "
        + "inherit your VISIBLE_DEVICES (so they share your GPU and cannot "
        + "oversubscribe), and cannot fan out further. Use leaves for breadth; do "
        + "multi-round depth (e.g. coordinate-descent autotune) yourself.",
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
        "- Start / restart a real server on your own cards (any port that is "
        + "NOT the production serving port 8888) and benchmark it however you "
        + "see fit.",
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
        + "serving + benchmark path on your leased cards + a non-8888 port, so "
        + "you can get numbers directly comparable to the ``integrate_patch`` "
        + "gate in one call:",
        "    python -m hyperloom.orchestrator.specialists.rebench \\",
        "        --config <magpie.yaml> --output ./scratch/rebench "
        + "[--extra-args '<server args>']",
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
    return [
        "",
        "### Free-form mandate (scope = freeform)",
        "",
        "You are dispatched as a **free-form** specialist: you are NOT bound to "
        + "the domain catalogue above. The Orchestration mandate below is your "
        + "whole task — investigate it wherever it leads (framework internals, "
        + "upstream PRs, host probing, source patches).",
        "",
        "Mandate from Orchestration:",
        "",
        f"> {desc}",
        "",
        "Set ``scope='freeform'`` on each proposal. Your single deliverable is "
        + "still ONE ``specialist_done`` carrying ``proposal_set`` + "
        + "``patches_written``. Never self-report numeric speedups — the "
        + "Coordinator measures gain.",
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
        f"- Hard wall-clock budget for this entire dispatch: "
        f"**{inp.wall_budget_sec:.0f}s (~{mins:.0f} min)**.",
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
    """Return True when every prior KB/PR/research source is empty, so a
    cold-start directive is injected instead of letting specialists return
    an empty proposal_set.

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
            # Cold-start directive: propose domain-focus defaults, not an empty set.
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
                    "**Directive — DO NOT return an empty proposal_set.** "
                    + "Treat the *Winning techniques* + *Pitfalls* in your "
                    + "**domain focus** block (Section 1) as your fallback "
                    + "prior. Pick the **1–2 most conservative, "
                    + "well-attested defaults** from those bullets that are "
                    + "compatible with the hardware (Section 2) and the "
                    + "gap symptom (Section 3); flag each as "
                    + "``confidence: low`` and ``provenance: "
                    + "domain_focus_default`` in the proposal. Use the "
                    + "``residual_questions`` field to record what RecipeKB, "
                    + "research, or ``mcp__pr_monitor__*`` query a future round should pursue.",
                    "",
                    "If the *Winning techniques* block is generic enough "
                    + "that no proposal is safer than a coin-flip, you may "
                    + "still emit ``empty=true`` — but you MUST cite which "
                    + "bullets you considered and why each was rejected "
                    + "(in ``summary``). A bare empty exit with no rationale "
                    + "will be treated as a tool failure by the Coordinator.",
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

    Dumps the ``find-recipe`` result as JSON, or a ``(none)`` placeholder
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
        rows.append(f"- **{statement}**{meta}{version_note}")
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


# Section 5d — KG graph-recommended knobs (advisory positive candidates)
def _section_kg_recommended(inp: SpecialistPromptInputs) -> list[str]:
    """Render cross-recipe ``IMPROVES`` candidates from the knowledge graph.

    These are advisory priors reached via the architecture family graph
    (``USES_ARCH`` / ``VARIANT_OF``) — knobs that improved a related
    architecture on the same hw+fw. The specialist prioritises but never
    blindly trusts them; the Critic still gates the final answer.

    Args:
        inp: The specialist prompt inputs (reads ``kg_recommended_knobs``).

    Returns:
        The rendered graph-recommended-knobs section lines.
    """
    rows = ["## 5d. GRAPH-RECOMMENDED KNOBS (cross-recipe IMPROVES — advisory, prioritise but verify)", ""]
    if not inp.kg_recommended_knobs:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for entry in inp.kg_recommended_knobs:
        if not isinstance(entry, dict):
            continue
        knob = str(entry.get("knob") or "").strip()
        if not knob:
            continue
        meta_bits: list[str] = []
        gain = entry.get("expected_gain")
        if isinstance(gain, (int, float)) and gain:
            meta_bits.append(f"gain={float(gain):+.1f}%")
        conf = entry.get("confidence")
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        rows.append(f"- **{knob}**{meta}")
    if len(rows) == 2:  # header + blank only; all entries filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


# Section 5e — KG graph-guided config knobs (runnable args/envs)
def _section_kg_guided_knobs(inp: SpecialistPromptInputs) -> list[str]:
    """Render journal-derived ``KNOB_IMPROVES`` candidates with runnable config.

    These come from prior sessions' ``optimization_journal`` config knobs that
    kept a positive gain on the same architecture+precision. Each carries the
    exact ``args``/``envs`` to apply, so the specialist can try them directly.
    Advisory: the specialist verifies and the Critic still gates the answer.

    Args:
        inp: The specialist prompt inputs (reads ``kg_guided_knobs``).

    Returns:
        The rendered graph-guided-knobs section lines.
    """
    rows = ["## 5e. GRAPH-GUIDED CONFIG KNOBS (journal KNOB_IMPROVES — runnable, prioritise but verify)", ""]
    if not inp.kg_guided_knobs:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for entry in inp.kg_guided_knobs:
        if not isinstance(entry, dict):
            continue
        args = str(entry.get("args") or "").strip()
        envs = entry.get("envs") if isinstance(entry.get("envs"), dict) else {}
        if not args and not envs:
            continue
        name = str(entry.get("name") or "").strip()
        meta_bits: list[str] = []
        gain = entry.get("expected_gain")
        if isinstance(gain, (int, float)) and gain:
            meta_bits.append(f"gain={float(gain):+.1f}%")
        ev = entry.get("evidence_count")
        if isinstance(ev, (int, float)) and ev:
            meta_bits.append(f"kept={int(ev)}x")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        label = name or args
        rows.append(f"- **{label}**{meta}")
        if args:
            rows.append(f"  - args: `{args}`")
        if envs:
            env_str = " ".join(f"{k}={v}" for k, v in envs.items())
            rows.append(f"  - envs: `{env_str}`")
    if len(rows) == 2:  # header + blank only; all entries filtered out
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
    from hyperloom.orchestrator.specialists.domains import PR_QUERY_REPOS

    rows = ["## 6. PR MONITOR", ""]
    if not inp.pr_monitor_available:
        rows.append("(unavailable: pr_monitor disabled)")
        return rows
    rows += [
        "Use ``mcp__pr_monitor__*`` tools to query PRs on demand:",
        "``pr_search`` / ``pr_list`` / ``pr_get`` / ``pr_files`` / ``pr_patches`` / ``pr_file_patch`` / ``pr_blob``",
        "",
        "Repos you may query:",
    ]
    for repo in PR_QUERY_REPOS:
        rows.append(f"- {repo}")
    return rows


# Section 7 — Local source navigation hint
def _section_source_hint(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 7 (local source navigation hint) of the prompt.

    Lists the read-only framework source roots and per-domain focus
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
        rows.append("Framework source roots (read-only):")
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
        "Do NOT attempt Edit / Write / git apply (PolicyGate R4)."
    )
    return rows


# Section 8 — Output protocol
def _section_output_protocol(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 8 (output protocol) of the specialist prompt.

    Describes the two equivalent ``specialist_done`` exit channels, the
    payload schema, field contract, heartbeat rules, and the turn cap.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs (source
            of workspace path, gap id, domain, max proposals, and turns).

    Returns:
        list[str]: Markdown lines for the output-protocol section.
    """
    workspace = inp.workspace_path or "<workspace>"
    return [
        "## 8. OUTPUT PROTOCOL",
        "",
        "Your run terminates by producing **exactly one** specialist_done",
        "record. The Hyperloom runner accepts either of two equivalent",
        "exit channels — use whichever your tool surface supports:",
        "",
        "**Channel A — ``emit_intent`` tool (in-process / SDK runtime):**",
        "Call the ``emit_intent`` tool exactly once with an intent of type",
        "``specialist_done`` and the payload schema below.",
        "",
        "**Channel B — file write (subprocess / production runtime,",
        "PR-A2 Arbor-into-Hyperloom):** When ``emit_intent`` is not in",
        "your tool list, write the same payload to",
        f"``{workspace}/specialist_done.json`` via the ``Write`` tool as",
        "your **absolute last action**. The Hyperloom dispatcher polls",
        "for that file and treats its appearance as the run's exit",
        "signal. After writing it, stop — do not call any further tools.",
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
        "preserved as your result, so keep it current. Write the final",
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
                            "reason": "why this might help the gap",
                            "atomic": False,
                            "kb_evidence": [],
                            "pr_evidence": [],
                            "source_evidence": [],
                        }
                    ],
                    "patches_written": [],
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
        "- ``proposal_set`` items reuse the explore variant schema.",
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
            f"- ``proposal_set`` MUST contain AT MOST **{inp.max_proposals}** "
            "entries. You are a curator, not a brainstormer: rank candidates "
            "by expected gain x your confidence, drop everything that "
            "contradicts ``kb_subgraph`` / ``pr_evidence`` already in "
            f"your prompt, and only emit the surviving top {inp.max_proposals}. "
            "Fewer is better than padding."
        ),
        (
            "- The Critic reviews each surviving variant against the KB "
            + "before benchmarking, so a marginal-quality proposal costs you "
            + "a reject (and a pitfall fact that will warn future sessions "
            + "off the same dead-end)."
        ),
        "- ``patches_written`` (PR-A2) lists paths (relative to your",
        "  workspace or worktree) of any unified-diff patch files you",
        "  authored this round. Empty list = no patches; downstream",
        "  ``integrate_patch`` action skips when empty.",
        "- ``artifacts_written`` lists any non-diff tuned artifacts to install",
        "  (e.g. an autotuned config JSON) as objects ``{source, target, kind,",
        "  description}``: ``source`` is a path inside your worktree, ``target``",
        "  is the install path — PREFER a framework-relative path (e.g.",
        "  ``configs/model_configs/foo.csv``); an absolute path is accepted only",
        "  if it resolves inside an allowlisted framework root. ``integrate_patch``",
        "  backs up the target, installs the artifact, runs the same E2E gate, and",
        "  restores the backup on REVERT. A non-diff tuned artifact is a FULL",
        "  result — set ``empty=false`` when ``artifacts_written`` is non-empty.",
        "- ``empty=true`` is legitimate ONLY when you have no actionable proposals",
        "  AND no ``patches_written``/``artifacts_written``; in that case",
        "  ``proposal_set=[]`` and you must put the reason in ``summary``.",
        "- ``new_findings`` is your free-form summary of anything you",
        "  learned this round — Coordinator funnels it into the KB",
        "  fact-write pipeline (lesson on KEEP, pitfall on REVERT).",
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
    """Render Section 9 (iron rules) of the specialist prompt.

    Emits the immutable capability boundary (full autonomy on the
    specialist's own leased cards with the single production-serving boundary,
    worktree-only patch/artifact staging, no KB writes, allowed intents, turn
    cap, and workspace confinement).

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs (source
            of the workspace path interpolated into the rules).

    Returns:
        list[str]: Markdown lines for the iron-rules section.
    """
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    if inp.allocated_gpu_ids:
        cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
        gpu_rule = [
            f"1. You EXCLUSIVELY own GPU card(s) [{cards}] for this task. On",
            "   those cards do whatever you want: edit code, build, start/stop",
            "   your own servers (on any port that is NOT 8888), profile,",
            "   autotune, install tuned artifacts, run real benchmark loops.",
            "   The ONE thing you must NOT do: touch the production serving",
            "   process, its cards, or port 8888 — co-residing on them would",
            "   corrupt both your measurement and production. Manage only",
            "   processes YOU started, by their own PID/PGID.",
        ]
    else:
        gpu_rule = [
            "1. You have no GPU allocation for this task, so do not run GPU",
            "   benchmarks or start servers. The ONE hard boundary that always",
            "   holds: never touch the production serving process / its cards /",
            "   port 8888. The Coordinator runs benchmarks; you propose what to",
            "   try and optionally author patches.",
        ]
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.2 / Inv-5.3)",
        "",
        *gpu_rule,
        "2. **You MAY** produce changes for integration, but stage them ONLY",
        f"   inside your own worktree at ``{workspace}/`` (a git checkout",
        "   branched off the framework HEAD just for this task). Two output",
        "   kinds are accepted by the orchestrator's ``integrate_patch`` gate:",
        "   - Unified-diff patches: ``git diff > patches/NNN_<slug>.patch``",
        "     from inside the worktree; list paths in ``patches_written``.",
        "   - Tuned non-diff artifacts (e.g. an autotuned config JSON): write",
        "     the file under your worktree and list it in ``artifacts_written``",
        "     as ``{source, target, kind, description}`` (``source`` relative to",
        "     the worktree, ``target`` the framework-relative install path).",
        "   You **MUST NEVER** ``git apply`` / ``git commit`` against or",
        "   otherwise mutate the main ``framework_source_roots`` directly —",
        "   the orchestrator's ``integrate_patch`` action is the single",
        "   integration point that applies your patches/artifacts with the",
        "   throughput + accuracy gate. (Starting/stopping YOUR OWN servers on",
        "   YOUR OWN leased cards per rule 1 is fine; the prohibition here is",
        "   only about mutating the shared framework tree directly.)",
        "3. **NEVER** call ``cortex-kb`` write endpoints (propose-point /",
        "   propose-edge / propose-lesson / propose-pitfall / update-recipe)",
        "   directly. The Coordinator owns KB writes (PolicyGate R4). KB",
        "   read context is pre-warmed into Section 4 of this prompt; the",
        "   specialist subprocess has no live KB connection.",
        "4. **NEVER** emit any intent other than ``specialist_done``,",
        "   ``send_message`` (heartbeat), or ``alert``. Any other intent",
        "   type triggers PolicyGate R3 ``specialist_done_source``.",
        "5. You **MUST** finish within ``max_turns`` LLM turns and end with",
        "   a single ``specialist_done`` exit signal (intent OR",
        "   ``specialist_done.json`` file write per Section 8). Sub-agent",
        "   silence past the cap is treated as stale (an empty",
        "   ``specialist_done`` is synthesized for you so the EXPLORE",
        "   round still progresses).",
        f"6. Use ``{workspace}/`` for ALL writes (patches, transcript notes,",
        "   heartbeat). Do not write anywhere else in the filesystem; the",
        "   dispatcher only exposes this directory + read-only access to",
        "   ``framework_source_roots`` and SESSION_DIR.",
        "7. If you hit a tool error or run out of useful actions, emit",
        "   ``specialist_done{empty=true, summary='<why>'}`` rather than",
        "   stalling.",
        f"8. {BASH_KILL_SAFETY_PREAMBLE}",
    ]


# Top-level assembler
def _section_pd_disaggregation(inp: SpecialistPromptInputs) -> list[str]:
    """§1a — PD-disaggregation context (omitted unless pd_mode==disaggregated).

    Surfaces the prefill/decode split so the specialist targets each role's
    distinct bottleneck (prefill: compute / TTFT; decode: memory-bandwidth /
    TPOT) and the KV-transfer path, instead of treating the server as one pool.
    Reads the multi-node state directly; returns ``[]`` on the single-node /
    colocated paths so the section is dropped.

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


def build_specialist_prompts(inp: SpecialistPromptInputs) -> tuple[str, str]:
    """Assemble the full specialist prompt from its section builders.

    The system prompt carries the immutable contract (identity, output
    protocol, iron rules) and the user prompt carries the per-task context
    (hardware, gap, KB, roofline, recipe, lessons, pitfalls, PR feed,
    source hint, optional session snapshot and orchestration notes). The
    split lets the LLM backend cache the system prompt across specialists.

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
    user_sections = [
        _section_hardware(inp),
        _section_pd_disaggregation(inp),  # § 1a (PD-disaggregation; omitted unless disaggregated)
        _section_execution_budget(inp),  # omitted when no budget
        _section_gap(inp),
        _section_kb_subgraph(inp),
        _section_roofline_evidence(inp),
        _section_recipe(inp),
        _section_lessons(inp),
        _section_pitfalls(inp),
        _section_kg_recommended(inp),  # KG graph-recommended knobs
        _section_kg_guided_knobs(inp),  # KG graph-guided runnable knobs
        _section_pr_feed(inp),
        _section_source_hint(inp),
    ]
    if inp.notes:
        user_sections.append(
            [
                "## 10. NOTES FROM ORCHESTRATION",
                "",
                inp.notes,
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
