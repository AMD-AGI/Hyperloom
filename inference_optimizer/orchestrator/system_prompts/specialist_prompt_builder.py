# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist sub-agent prompt assembler — v0.8 M5.

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

from ..specialist_domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SpecialistDomain,
    domain_for_tag,
    get_domain,
)


_NONE_PLACEHOLDER = "(none)"


# Soft cap on ``proposal_set`` size; re-exported from ``policy.py`` so the
# prompt-side cap and the runner-side hard truncate stay aligned.
from inference_optimizer.orchestrator.policy import (
    DEFAULT_SPECIALIST_MAX_PROPOSALS,
)


# Per-domain focus templates: each injects a "Domain focus" block into
# Section 1; a missing key falls back to the generic body.


def _is_atom(inp: SpecialistPromptInputs) -> bool:
    """True when ``_focus_*`` blocks should use atom-flavoured hints
    (empty framework falls back to the canonical sglang/vllm block)."""
    return (inp.framework or "").strip().lower() == "atom"


def _focus_serving_specialist(inp: SpecialistPromptInputs) -> list[str]:
    if _is_atom(inp):
        return [
            "You target **atom scheduler / cuda_graph / kv_cache** code.",
            "",
            "**What to read first**",
            "- `atom/entrypoints/openai_server.py` (HTTP request routing, "
            "`/start_profile`, `/stop_profile`).",
            "- `atom/model_engine/engine_core.py` (engine main loop, "
            "`start_profiler` call sites).",
            "- `atom/model_engine/llm_engine.py` (engine API surface).",
            "- `atom/model_engine/model_runner.py` (per-rank forward, "
            "profiler hooks, cudagraph capture).",
            "- `atom/model_engine/arg_utils.py` (CLI flag inventory; "
            "ground-truth for `--level`, `--enable_prefix_caching`, "
            "`--cudagraph-capture-sizes`, `--kv_cache_dtype`, etc.).",
            "- `atom/config.py` (engine config shape).",
            "- KB anchor `framework.*` (cuda_graph / batching / kv_cache).",
            "",
            "**Winning techniques to consider**",
            "- `--level {2,3}` (atom's torch.compile / cudagraph bracket).",
            "- `--enable_prefix_caching` for shared-prefix workloads.",
            "- `--cudagraph-capture-sizes` bracketed around the live CONC.",
            "- `--kv_cache_dtype fp8` on FP8-shipped models (gate accuracy).",
            "- `--max-num-seqs` / `--max-num-batched-tokens` at concurrency "
            "boundaries (same scheduler-side tuning as sglang/vllm).",
            "",
            "**Pitfalls (historical REVERTs / atom-specific)**",
            "- Atom is single-node only. Multi-node distributed proposals "
            "are non-actionable — pivot to rank-local optimisations.",
            "- `--enforce-eager` is a debug fallback; almost never a perf win.",
            "- Cudagraph capture size lists that don't bracket the live "
            "CONC trigger silent recapture on every batch boundary.",
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
    if _is_atom(inp):
        return [
            "You target **aiter / atom kernels / triton** code (attention,",
            "MoE, GEMM, fused-attention paths).",
            "",
            "**What to read first**",
            "- `aiter/csrc/` and `aiter/aiter/ops/` (CK / hipBLASLt "
            "wrappers — **shared with sglang and vllm**, so aiter "
            "patches apply transparently across all three).",
            "- `atom/model_ops/` (atom-specific kernel call sites).",
            "- `atom/quantization/` (atom's FP8 / weight-quant paths).",
            "- `atom/models/` (built-in model implementations; cross-",
            "reference for which kernels each model touches).",
            "- KB anchor `kernel.*` (CDNA3 tiling / MoE / attention / GEMM).",
            "",
            "**Winning techniques to consider**",
            "- aiter env switches (`VLLM_ROCM_USE_AITER=1` umbrella + "
            "per-op overrides) — the shared aiter surface means the "
            "same env knobs that work on sglang/vllm carry over to atom.",
            "- Tile-size / occupancy tuning for short-OSL decode.",
            "- Fused-attention enable flags for prefill chunks.",
            "",
            "**Pitfalls**",
            "- Searching `sglang/python/sglang/srt/layers/attention/` on "
            "an atom box: those paths are empty / absent. Use "
            "`atom/model_ops/` + shared `aiter/` instead.",
            "- Mixing aiter overrides with `--enforce-eager` invalidates "
            "atom's cudagraph captures silently.",
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
        "- Switch attention backend (`ROCM_AITER_MLA` ↔ `TRITON_MLA` ↔",
        "  `ROCM_AITER_TRITON_MLA`) at the workload's prefill/decode mix.",
        "- `VLLM_ROCM_USE_AITER=1` umbrella + per-op overrides for MoE / RMSNorm.",
        "- Tile-size tuning for `M < 256` GEMMs (hipBLASLt vs Triton).",
        "- Fused-attention enable flags for prefill chunks.",
        "",
        "**Pitfalls**",
        "- Forcing AITER MLA on workloads with short OSL — kernel selection",
        "  cost dominates the saving.",
        "- Mixing `--attention-backend` with `--enforce-eager=true` invalidates",
        "  cuda graphs silently.",
        "- Trying triton fp4 paths on CDNA3 without `AMDGCN_USE_BUFFER_OPS=1`.",
    ]


def _focus_comm_specialist(inp: SpecialistPromptInputs) -> list[str]:
    if _is_atom(inp):
        return [
            "You target **intra-node RCCL / NCCL / QuickReduce / "
            "AllReduce** tuning on atom.",
            "",
            "**Atom is single-node only.** Multi-node tensor / data / "
            "pipeline parallelism is NOT available on atom. Cross-node "
            "collectives proposals are non-actionable — focus on "
            "intra-node concerns (rank-local optimisations, "
            "intra-node NCCL/RCCL config, allreduce algorithm choice "
            "for the on-box TP group).",
            "",
            "**What to read first**",
            "- `atom/utils/distributed/utils.py` (single-node "
            "`torch.distributed` helper — NOT a multi-node TP "
            "orchestration layer).",
            "- `aiter/csrc/quick_reduce/` and RCCL plugin paths "
            "(shared with sglang/vllm; intra-node only on atom).",
            "- KB anchor `communication.*` (allreduce / QuickReduce / "
            "topology).",
            "",
            "**Winning techniques to consider (single-node)**",
            "- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` when "
            "intra-node TP allreduce message size > 1MiB.",
            "- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` tuning for "
            "the on-box XGMI topology.",
            "- `--enable-dp-attention` (MLA models) — DP-attention "
            "shifts work onto a TP-free per-rank path, reducing the "
            "allreduce footprint.",
            "",
            "**Pitfalls**",
            "- Proposing multi-node TP / PP topologies — atom rejects "
            "them at startup. The Coordinator collapses `--nodes>1` to "
            "single-node mode on atom (IR-8).",
            "- INT4 QuickReduce at TP=2 — overhead dominates the "
            "bandwidth savings on small message sizes.",
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
    proven_lines: list[str] = []
    if inp.already_proven:
        proven_lines.append(
            "**Already proven (warm-start recipe) — do NOT re-mine these; "
            "focus on net-new priors:**"
        )
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
        "   re-listing PRs the FRAMEWORK_PR phase already covered (the",
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


_DOMAIN_FOCUS_TEMPLATES: dict[
    str, "Callable[[SpecialistPromptInputs], list[str]]"
] = {
    "serving_specialist": _focus_serving_specialist,
    "kernel_switch_specialist":    _focus_kernel_switch_specialist,
    "comm_specialist":      _focus_comm_specialist,
    "compiler_specialist":  _focus_compiler_specialist,
    "system_specialist":    _focus_system_specialist,
    "pr_intel_specialist":  _focus_pr_intel_specialist,
    "research_scout_specialist": _focus_research_scout_specialist,
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

    # Hardware context. ``tp`` defaults to 0 (sentinel for "unspecified"),
    # NOT 1, so comm_specialist doesn't veto its own TP proposals.
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
    # Runtime fingerprint so the specialist can judge lesson applicability;
    # ``framework_version`` is the precise install version. Empty => no
    # version annotation.
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
    # T0 lessons — positive priors from prior KEEPs; rendered as § 5b.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # PR feed
    pr_feed: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Generic sub_kind passthrough so ``_focus_*`` helpers can specialise.
    sub_kind: str = ""

    # Extra knowledge-domain tags; each contributes a focus block to Section 1.
    extra_focus_tags: tuple[str, ...] = ()

    # Active server framework (``sglang`` / ``vllm`` / ``atom``); empty falls
    # back to the canonical sglang/vllm hint blocks. Switches "what to read
    # first" bullets to atom paths when ``framework == 'atom'``.
    framework: str = ""

    # Local source navigation hint
    framework_source_roots: tuple[str, ...] = ()
    source_hint_directories: tuple[str, ...] = ()

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""

    # Dispatch profile dials (see orchestrator.specialist_profile). Defaults
    # preserve the legacy single-domain patch-authoring behaviour; later phases
    # consume these to shape cross-domain / freeform / bench prompting.
    scope: str = "domain"
    mode: str = "patch"
    bench: bool = False
    lane: str = "gpu"
    # Free-form task description (only populated when scope == 'freeform').
    task_description: str = ""


# Section 1 — Identity & autonomy
def _section_identity(inp: SpecialistPromptInputs) -> list[str]:
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
        f"scored over quantity: cap your final ``proposal_set`` at the",
        f"**top-{inp.max_proposals}** ranked picks (see Section 8).",
        "",
        "Division of labour: the Coordinator owns the serving GPU, runs the E2E",
        "benchmark, and decides KEEP/REVERT — you do not have to validate final",
        "throughput yourself. Your single deliverable is ONE final ``specialist_done``",
        "(Section 8) carrying ``proposal_set`` + ``patches_written``. The hard",
        "capability boundary is fixed by Section 9 Iron Rules; everything inside",
        "it is yours.",
    ]
    # Per-domain expertise + focus blocks.
    rendered_focus_keys: set[str] = set()
    focus = _DOMAIN_FOCUS_TEMPLATES.get(inp.domain.key)
    if focus is not None:
        body.append("")
        body.append(f"### Domain focus — {inp.domain.key}")
        body.append("")
        body.extend(focus(inp))
        rendered_focus_keys.add(inp.domain.key)
    # Multi-tag dispatch: append each extra tag's focus block.
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
    return body


def _cross_domain_block(inp: SpecialistPromptInputs) -> list[str]:
    """Cross-domain mandate appended when ``scope == 'domains'`` (absorbed
    from the retired dynamic_action channel). The single deliverable is still
    ONE ``specialist_done``; the difference is the patch may span every domain
    in scope and the Critic will hold it to the cross-domain rules."""
    tags = ", ".join(inp.extra_focus_tags) if inp.extra_focus_tags else inp.domain.key
    return [
        "",
        "### Cross-domain mandate (scope = domains)",
        "",
        f"You are dispatched as a **cross-domain** specialist over: {tags}.",
        "You may author a single coherent patch that spans these domains "
        "together when (and only when) the change must happen jointly — a "
        "combination no single-domain specialist could surface from within "
        "its own boundary.",
        "",
        "In your ``specialist_done`` you MUST justify the combination:",
        "- give an independent rationale for the change **within each domain** "
        "in scope;",
        "- name the **coupling points** (why these changes must land together) "
        "and at least one **side effect** of the combination;",
        "- show this is genuine cross-domain synthesis, not a concatenation of "
        "two independent single-domain edits (that is an explore grid combo, "
        "not a cross-domain patch).",
        "Set ``scope='domains'`` on the proposal so the Critic attaches the "
        "cross-domain review rules. Never self-report numeric speedups — the "
        "Coordinator measures gain.",
    ]


# Section 2 — Hardware context
def _section_hardware(inp: SpecialistPromptInputs) -> list[str]:
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    if inp.allocated_gpu_ids:
        rows.append(
            "- allocated specialist GPU ids: "
            + ", ".join(str(g) for g in inp.allocated_gpu_ids)
        )
        rows.append(
            "- GPU specialist scope: short experiments / microbenchmarks only; "
            "do not launch a persistent serving server or Magpie benchmark loop."
        )
    if inp.tp > 0:
        rows.append(f"- TP: {inp.tp}")
    else:
        rows.append(f"- TP: {_NONE_PLACEHOLDER}")
    if inp.hbm_gb > 0:
        rows.append(f"- HBM per GPU: {inp.hbm_gb:.1f} GB")
    if inp.peak_tflops > 0:
        rows.append(f"- Peak TFLOPs (declared): {inp.peak_tflops:.1f}")
    # Workload context — concrete numbers so the specialist doesn't guess.
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


# Section 3 — Gap statement
def _section_gap(inp: SpecialistPromptInputs) -> list[str]:
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
    """Issue-J: all prior sources empty, so inject a cold-start directive
    instead of letting specialists return an empty proposal_set."""
    return (
        not inp.kb_subgraph
        and not inp.warm_start_recipe
        and not inp.warm_start_lessons
        and not inp.warm_start_pitfalls
        and not inp.pr_feed
        and not inp.research_hints
    )


def _section_kb_subgraph(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 4. KB CONTEXT (optional, advisory)", ""]
    cold = _is_cold_start(inp)
    if not inp.kb_subgraph:
        if inp.research_hints:
            # Research hints stand in as an advisory prior when KB is empty.
            rows.extend([
                "Structured KB context is empty for this (model, hardware, domain), but "
                "the research scout collected source-backed priors this "
                "session. Treat these as your advisory prior (co-equal with "
                "RecipeKB priors; the Critic still gates the final answer):",
                "",
                inp.research_hints,
                "",
                "Anchor proposals on these hints where they fit the gap "
                "(Section 3) and hardware (Section 2).",
            ])
            return rows
        if cold:
            # Cold-start directive so the specialist proposes domain-focus
            # defaults rather than an empty proposal_set.
            rows.extend([
                "**COLD-START MODE — no priors available.**",
                "",
                "All prior sources for this gap are empty:",
                "",
                "- KB context: ``(none)`` — no RecipeKB warm-start facts, "
                "research hints, or PR feed entries were available for this "
                "(model, hardware, domain) tuple.",
                "- Warm-start recipe: ``(none)`` (Section 5).",
                "- PR feed: ``(none)`` (Section 6).",
                "",
                "**Directive — DO NOT return an empty proposal_set.** "
                "Treat the *Winning techniques* + *Pitfalls* in your "
                "**domain focus** block (Section 1) as your fallback "
                "prior. Pick the **1–2 most conservative, "
                "well-attested defaults** from those bullets that are "
                "compatible with the hardware (Section 2) and the "
                "gap symptom (Section 3); flag each as "
                "``confidence: low`` and ``provenance: "
                "domain_focus_default`` in the proposal. Use the "
                "``residual_questions`` field to record what RecipeKB, "
                "research, or PR query a future round should pre-warm.",
                "",
                "If the *Winning techniques* block is generic enough "
                "that no proposal is safer than a coin-flip, you may "
                "still emit ``empty=true`` — but you MUST cite which "
                "bullets you considered and why each was rejected "
                "(in ``summary``). A bare empty exit with no rationale "
                "will be treated as a tool failure by the Coordinator.",
            ])
        else:
            rows.extend([
                _NONE_PLACEHOLDER,
                "",
                "(No structured KB context supplied. Use Sections 1, 3, 5, "
                "and 6 plus source inspection; record missing RecipeKB / "
                "research / PR questions in ``residual_questions`` so a "
                "future round can warm richer advisory context.)",
            ])
        return rows
    rows.append("```json")
    rows.append(json.dumps(inp.kb_subgraph, sort_keys=True, indent=2))
    rows.append("```")
    return rows


# Section 4a — Roofline / TraceLens evidence
def _section_roofline_evidence(inp: SpecialistPromptInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section from ``inp.roofline_evidence``;
    empty evidence renders a heading + ``(none)`` placeholder."""
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
            ("Compute %",        "compute_pct"),
            ("Idle %",           "idle_pct"),
            ("Exposed Comm %",   "comm_pct"),
            ("Top bottleneck",   "top_bottleneck"),
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
        rows.append(
            "| kernel_id | name | gpu_pct | bound | AI | eff_pct | "
            "compute_pct | bandwidth_pct | action |"
        )
        rows.append("|---|---|---:|---|---:|---:|---:|---:|---|")
        for k in roofline:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = (
                f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float))
                else "—"
            )
            bound = str(k.get("bound_type") or k.get("bottleneck") or "")
            ai = k.get("arithmetic_intensity")
            if ai is None:
                ai = k.get("flops_per_byte")
            ai_str = (
                f"{float(ai):.3g}" if isinstance(ai, (int, float)) else "—"
            )
            eff = k.get("efficiency_percent")
            eff_str = (
                f"{float(eff):.2f}%" if isinstance(eff, (int, float)) else "—"
            )
            comp = k.get("compute_utilization_pct")
            comp_str = (
                f"{float(comp):.2f}%" if isinstance(comp, (int, float)) else "—"
            )
            bw = k.get("bandwidth_utilization_pct")
            bw_str = (
                f"{float(bw):.2f}%" if isinstance(bw, (int, float)) else "—"
            )
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
            gpu_pct_str = (
                f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float))
                else "—"
            )
            bottleneck = str(k.get("bottleneck") or "")
            src = str(k.get("source_file") or "")
            rows.append(
                f"| `{kid}` | {name} | {gpu_pct_str} | {bottleneck} | {src} |"
            )
        rows.append("")

    analysis_path = str(ev.get("analysis_md_path") or "")
    if analysis_path:
        rows.append(
            f"**Full analysis.md path:** `{analysis_path}`"
        )
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
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    if not inp.warm_start_recipe:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append("**find-recipe result:**")
    rows.append("```json")
    rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, indent=2))
    rows.append("```")
    return rows


# Section 5b — Related lessons (positive priors from prior KEEPs)
def _section_lessons(inp: SpecialistPromptInputs) -> list[str]:
    """Render KB ``kind=lesson`` points from prior KEEPs, compactly
    (statement + measured_impact)."""
    rows = ["## 5b. RELATED LESSONS (prior KEEPs on this model+hw)", ""]
    if not inp.warm_start_lessons:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_lessons:
        attrs = (point or {}).get("attrs") or {}
        statement = str(attrs.get("statement") or "").strip()
        if not statement:
            continue
        impact_str = _render_measured_impact(attrs.get("measured_impact"))
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        # validated_count first (strongest cross-session signal); legacy
        # rows fall back to source_session_id.
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
    inp: SpecialistPromptInputs, lesson_attrs: dict[str, Any],
) -> str:
    """GAP 8 — render a ``[from sglang@X.Y, you're on A.B]`` annotation when
    the lesson's framework_version differs; empty when either side is
    unknown or they match."""
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
    string, or other)."""
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
    severity); framed as forbidden paths, not suggestions."""
    rows = ["## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)", ""]
    if not inp.warm_start_pitfalls:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_pitfalls:
        attrs = (point or {}).get("attrs") or {}
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


# Section 6 — PR feed
def _section_pr_feed(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 6. PR FEED", ""]
    if not inp.pr_monitor_available:
        rows.append("(empty: pr_monitor unavailable)")
        return rows
    if not inp.pr_feed:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for pr in inp.pr_feed:
        title = str(pr.get("title") or "").strip()
        url = str(pr.get("url") or "").strip()
        labels = pr.get("labels") or []
        labels_text = (
            " " + " ".join(f"[{l}]" for l in labels)
            if isinstance(labels, list) and labels else ""
        )
        rows.append(f"- {title} — <{url}>{labels_text}")
    return rows


# Section 7 — Local source navigation hint
def _section_source_hint(inp: SpecialistPromptInputs) -> list[str]:
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
        "Payload schema (identical for both channels):",
        "",
        "```json",
        json.dumps({
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
                        "kb_evidence": [],
                        "pr_evidence": [],
                        "source_evidence": []
                    }
                ],
                "patches_written": [],
                "empty": False,
                "summary": "≤ 500 char overview of what you tried this round",
                "confidence": 0.6,
                "new_findings": [],
                "residual_questions": []
            },
        }, sort_keys=True, indent=2),
        "```",
        "",
        "Field contract:",
        "",
        "- ``proposal_set`` items reuse the §3.4 explore variant schema.",
        (
            f"- ``proposal_set`` MUST contain AT MOST **{inp.max_proposals}** "
            "entries. You are a curator, not a brainstormer: rank candidates "
            "by expected gain x your confidence, drop everything that "
            "contradicts ``kb_subgraph`` / ``pr_feed`` evidence already in "
            f"your prompt, and only emit the surviving top {inp.max_proposals}. "
            "Fewer is better than padding."
        ),
        (
            "- The Critic reviews each surviving variant against the KB "
            "before benchmarking, so a marginal-quality proposal costs you "
            "a reject (and a pitfall fact that will warn future sessions "
            "off the same dead-end)."
        ),
        "- ``patches_written`` (PR-A2) lists paths (relative to your",
        "  workspace or worktree) of any unified-diff patch files you",
        "  authored this round. Empty list = no patches; downstream",
        "  ``integrate_patch`` action skips when empty.",
        "- ``empty=true`` is legitimate when you have no actionable proposals;",
        "  in that case ``proposal_set=[]`` and you must put the reason in",
        "  ``summary``.",
        "- ``new_findings`` is your free-form summary of anything you",
        "  learned this round — Coordinator funnels it into the KB",
        "  fact-write pipeline (lesson on KEEP, pitfall on REVERT).",
        "- ``residual_questions`` carries to the next specialist round.",
        "",
        "**Heartbeat (Channel B only):** When running in subprocess mode,",
        f"write ``{workspace}/heartbeat.json`` periodically (≤5 min apart)",
        "via Bash so the dispatcher knows you are still alive. Format:",
        "``{\"ts\": \"<iso8601>\", \"status\": \"running\", \"note\": \"<short>\"}``.",
        "Going silent past 5 minutes kills your subprocess.",
        "",
        (
            f"Hard cap: at most **{inp.max_turns}** LLM turns. Silence past "
            "the cap = stale (robustness will synthesize an empty done)."
        ),
    ]


# Section 9 — Iron rules
def _section_iron_rules(inp: SpecialistPromptInputs) -> list[str]:
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    if inp.allocated_gpu_ids:
        gpu_rule = [
            "1. You have an explicit GPU specialist allocation for this task.",
            "   You MAY run short GPU experiments or microbenchmarks on the",
            "   allocated visible devices only. You MUST NOT launch persistent",
            "   serving servers, run Magpie benchmark loops, restart vLLM/SGLang,",
            "   or control the production serving process.",
        ]
    else:
        gpu_rule = [
            "1. **NEVER** touch the serving GPU (no Magpie / no benchmark / no",
            "   server restart / no vllm or sglang process control). The",
            "   Coordinator runs benchmarks; you only propose what to try and",
            "   optionally author patches.",
        ]
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.2 / Inv-5.3)",
        "",
        *gpu_rule,
        "2. **You MAY** write source patches, but ONLY into your own",
        f"   worktree at ``{workspace}/`` (a git checkout branched off",
        "   the framework HEAD just for this task). Concretely:",
        "   - Edit files inside the worktree.",
        "   - ``git diff > patches/NNN_<slug>.patch`` from inside the",
        "     worktree to produce a unified-diff patch file.",
        "   - List patch paths in ``patches_written`` in your",
        "     ``specialist_done`` payload (relative to the worktree).",
        "   You **MUST NEVER** ``git apply``, ``git commit``, restart a",
        "   server, or otherwise mutate the main ``framework_source_roots``",
        "   directly — the orchestrator's ``integrate_patch`` action is",
        "   the single integration point that applies your patches with",
        "   the throughput + accuracy gate. (PR-A2, Arbor-into-Hyperloom:",
        "   Inv-5.1 updated.)",
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
    ]


# Top-level assembler
def build_specialist_prompts(inp: SpecialistPromptInputs) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for one specialist task."""

    system_sections = [
        _section_identity(inp),
        _section_output_protocol(inp),
        _section_iron_rules(inp),
    ]
    user_sections = [
        _section_hardware(inp),           # 0: § 1
        _section_gap(inp),                # 1: § 2-3
        _section_kb_subgraph(inp),        # 2: § 4
        _section_roofline_evidence(inp),  # 3: § 4a
        _section_recipe(inp),             # 4: § 5
        _section_lessons(inp),            # 5: § 5b
        _section_pitfalls(inp),           # 6: § 5c
        _section_pr_feed(inp),            # 7: § 6
        _section_source_hint(inp),        # 8: § 7
    ]
    if inp.notes:
        user_sections.append([
            "## 10. NOTES FROM ORCHESTRATION",
            "",
            inp.notes,
        ])

    def _flatten(sections: list[list[str]]) -> str:
        out: list[str] = []
        for sec in sections:
            if out:
                out.append("")
            out.extend(sec)
        return "\n".join(out) + "\n"

    return _flatten(system_sections), _flatten(user_sections)


def build_specialist_prompts_for_domain(
    *,
    task_id: str,
    domain_key: str,
    **kwargs: Any,
) -> tuple[str, str]:
    """Helper that resolves ``domain_key`` to a SpecialistDomain first."""
    domain = get_domain(domain_key)
    if domain is None:
        raise ValueError(
            f"unknown specialist domain={domain_key!r}; see specialist_domains"
        )
    inp = SpecialistPromptInputs(task_id=task_id, domain=domain, **kwargs)
    return build_specialist_prompts(inp)


__all__ = [
    "SpecialistPromptInputs",
    "build_specialist_prompts",
    "build_specialist_prompts_for_domain",
]
