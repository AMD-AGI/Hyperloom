"""Specialist sub-agent prompt assembler — v0.8 M5.

The Coordinator hands the SpecialistRunner a typed input bundle and
this module returns the fully-assembled 9-section prompt. The 9 sections
are fixed; each is independently nullable (renders as ``(none)``
placeholder) so the prompt structure stays stable regardless of which
KB / PR / source-tree slots happen to be populated for the current
specialist + gap.

Output is a tuple ``(system_prompt, user_prompt)`` where:

* ``system_prompt`` carries sections 1 (identity), 8 (output protocol),
  9 (iron rules) — the immutable specialist contract.
* ``user_prompt`` carries sections 2 (hardware), 3 (gap), 4 (KB), 5
  (recipe), 6 (PR), 7 (source hint) — the per-task context.

This split lets the LLM backend cache the system prompt across multiple
specialists in the same session (identity / iron rules don't change).

Pure function: no IO besides reading the assembled inputs, no env
access, no logging side effects. The output is snapshotted to
``runs/specialist/<task_id>/prompt.md`` by SpecialistRunner.
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


# Cap on how many entries a specialist may emit in its final
# ``proposal_set``. Re-exported from ``orchestrator/policy.py`` so the
# prompt-side soft cap (self-curation instruction in Section 8) and the
# SpecialistRunner-side hard truncate (write path) stay aligned. The
# Critic separately rejects any marginal-quality survivors against KB
# priors. Override per-task via ``SpecialistPromptInputs.max_proposals``
# (still clamped to this value by the runner).
from inference_optimizer.orchestrator.policy import (
    DEFAULT_SPECIALIST_MAX_PROPOSALS,
)


# ---------------------------------------------------------------------------
# PR-A6 (Arbor-into-Hyperloom) — per-domain focus templates
#
# Each entry produces the body that the prompt builder injects into
# Section 1 under "### Domain focus — <key>". The shape mirrors
# Arbor's ``agent expertise`` table (launcher/orchestrator.md):
# - "What you read" (which sub-trees of the KB / which framework
#   directories to grep first).
# - "Winning techniques" (concrete patterns the specialist should
#   sanity-check against the gap before proposing).
# - "Pitfalls" (anti-patterns that historically reverted on this
#   domain — sourced from KB_design lessons + Arbor's lessons table).
#
# When a domain key is missing from this map, ``_section_identity``
# falls back to the generic body (the legacy M5 default).
# ---------------------------------------------------------------------------


def _is_atom(inp: SpecialistPromptInputs) -> bool:
    """True when ``_focus_*`` blocks should use atom-flavoured hints.

    ``framework`` may be empty on legacy / test dispatches where the
    Coordinator did not plumb it; treat that as "use the canonical
    sglang/vllm hint block" so existing tests keep their semantics.
    """
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


def _focus_session_steward_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    return [
        "You are the **session steward** — an honest end-of-EXPLORE assessor.",
        "Your single job is to look at the session as a whole and recommend",
        "one of three exits: continue exploring, advance to kernel phase, or",
        "stop the session. You are NOT proposing knobs or patches.",
        "",
        "**What to read first** — the Coordinator inlines a panoramic",
        "state digest into **§ 5d. SESSION SNAPSHOT** below. Use that as",
        "your primary evidence; you may use ``Bash`` to ``cat",
        "$SESSION_DIR/state.json`` only if § 5d is empty (KB-degraded",
        "boot) or you need a field that isn't included. Key signals:",
        "- ``optimization_stack_len`` + ``gain_per_stack_entry_tail`` —",
        "  diminishing returns over the last 5 KEEPs.",
        "- ``rejected_counts`` — REVERT reasons aggregated from",
        "  ``explore_search.rejected``. A long tail of one kind",
        "  (``stack_unstable`` / ``gain_below_threshold``) is a signal.",
        "- ``specialist_empty_streak`` — per-domain empty-round counter.",
        "  Three consecutive ``empty=True`` rounds across the active",
        "  domains is a hard plateau signal.",
        "- ``gaps_count`` + ``gaps_top5_canonical_ids`` — open gaps the",
        "  Coordinator believes still exist. If non-empty and the",
        "  recommended specialist domain has not been exhausted,",
        "  ``continue_explore`` may be justified.",
        "- ``policy_denial_history_tail`` — when the LLM has been",
        "  thrashing against the same rule, this is evidence that further",
        "  exploration is unlikely to land KEEPs.",
        "- ``steward_continuation_used`` — IR-7 antiloop flag; if True",
        "  you've already burned your one continuation and must NOT",
        "  recommend ``continue_explore`` again.",
        "",
        "**Output protocol** (your single ``specialist_done`` payload must",
        "carry these extra fields beyond the standard schema):",
        "- ``recommendation`` ∈ ``{continue_explore, advance_to_kernel, stop_session}``",
        "  — REQUIRED. Anything else is coerced to ``stop_session``.",
        "- ``next_gap_canonical_id``: str (REQUIRED iff",
        "  ``recommendation='continue_explore'``). Must reference an entry",
        "  the Coordinator can plausibly act on; otherwise the",
        "  Coordinator falls back to ``advance_to_kernel``.",
        "- ``remaining_potential_pct_estimate``: float — your best estimate",
        "  of cumulative gain still reachable in EXPLORE. Used for the",
        "  final report's section 9.1 (remaining gaps).",
        "- ``rationale``: str (<= 2000 chars). One paragraph; the final",
        "  report quotes this verbatim, so write for an operator reader.",
        "",
        "**Antiloop** — you can be invoked at most twice per session. The",
        "Coordinator records ``steward_continuation_used=True`` after the",
        "first ``continue_explore`` you return; the second invocation MUST",
        "NOT recommend ``continue_explore`` again (the Coordinator coerces",
        "to ``advance_to_kernel`` if you do). Use the first continuation",
        "judiciously.",
        "",
        "**Iron-rule alignment**",
        "- IR-6: when ``=== Phase ===`` reports ``session_buffer_sec < 0``",
        "  the HARD force-exit gate is about to fire on the next tick;",
        "  ``stop_session`` is the only honest answer.",
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
    "session_steward_specialist": _focus_session_steward_specialist,
    "research_scout_specialist": _focus_research_scout_specialist,
}


@dataclass(frozen=True)
class SpecialistPromptInputs:
    """Typed inputs the Coordinator hands to the prompt builder."""

    # Identity
    task_id: str
    domain: SpecialistDomain
    max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS
    # Soft cap on ``proposal_set`` size — rendered into Sections 1 + 8
    # so the specialist self-curates to its top-K picks rather than
    # padding with marginal candidates.
    max_proposals: int = DEFAULT_SPECIALIST_MAX_PROPOSALS

    # Hardware context (§3.5 §6 part 2). ``tp`` defaults to 0
    # (sentinel for "unspecified"), NOT 1 — a silent default of 1
    # would make comm_specialist veto its own proposals on
    # tensor-parallel sessions where the Coordinator forgot to
    # plumb ``params['tp']`` from SharedState.
    gpu_type: str = ""
    allocated_gpu_ids: tuple[int, ...] = ()
    tp: int = 0
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""
    # Advisory competitor target gap block (mirrored from the
    # Coordinator). Direction hint only; never gates the specialist.
    target_gap_notes: str = ""
    # Already-proven warm-recipe optimizations (``{name, source}``) the
    # research scout should skip re-mining. Empty on cold-start.
    already_proven: list[dict[str, str]] = field(default_factory=list)
    # Compact advisory research-hint block (source-backed priors collected
    # this session). Rendered alongside the KB sub-graph as a co-equal
    # prior; its presence suppresses the cold-start fallback.
    research_hints: str = ""
    # Workload context (mirrored from SharedState by
    # Coordinator._warm_specialist_params; renders in section 2 so
    # the specialist sees the actual benchmark workload instead of
    # the dataclass default).
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    # GAP 5 / GAP 8 — runtime fingerprint surfaced into prompts so the
    # specialist can judge "is this lesson from an old framework still
    # applicable?". ``framework`` is the active backend (sglang / vllm);
    # ``framework_version`` is the precise install version (e.g. "0.5.11").
    # Both empty when SharedState doesn't carry them (legacy SDK
    # callers / pre-PR sessions); the prompt renderer treats absent
    # values as "no version annotation".
    framework: str = ""
    framework_version: str = ""

    # Gap statement (§3.5 §6 part 3)
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Cortex KB sub-graph (§3.5 §6 part 4)
    kb_subgraph: dict[str, Any] = field(default_factory=dict)

    # Roofline / TraceLens evidence (§3.5 §6 part 4a).
    # Filled by ``Coordinator._warm_specialist_params`` from
    # :attr:`SharedState.last_trace_analyze`. Expected keys:
    # ``analysis_md_path``, ``roofline_snapshot_id``,
    # ``executive_summary`` (compute/idle/comm/top_bottleneck percentages),
    # ``hot_kernels_top15`` (capped at top 8 by the warmer to bound
    # token cost). Empty dict → section renders empty / placeholder.
    roofline_evidence: dict[str, Any] = field(default_factory=dict)

    # Recipe summary from T0 ``find-recipe`` (§3.5 §6 part 5)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 ``lessons`` query result — positive priors from prior KEEPs
    # on (model, hardware), sorted by KB-side confidence. Rendered as
    # § 5b for the specialist (separate from § 5 recipe so the LLM can
    # reason about each independently).
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # IR-7 — session_steward_specialist panoramic state digest. Only
    # populated when the dispatcher is dispatching a session_steward
    # task (other specialists get an empty dict and the section is
    # skipped entirely). See ``Coordinator._build_session_snapshot``
    # for the field shape. Rendered as § 5d.
    session_snapshot: dict[str, Any] = field(default_factory=dict)

    # PR feed (§3.5 §6 part 6)
    pr_feed: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Generic sub_kind passthrough from the dispatch params. Still
    # routed into the prompt so per-domain ``_focus_*`` helpers can
    # specialise on it where useful; the legacy ``framework_pr_scout``
    # branch was retired with the FRAMEWORK_PR phase migration.
    sub_kind: str = ""

    # Additional knowledge-domain tags carried by a multi-tag dispatch.
    # Each tag contributes its per-domain focus block to Section 1 (the
    # primary ``domain`` block renders first). Empty for single-tag
    # dispatch.
    extra_focus_tags: tuple[str, ...] = ()

    # Active server framework name (``sglang`` / ``vllm`` / ``atom``).
    # Mirrored from ``SharedState.framework`` by
    # ``Coordinator._warm_specialist_params``. Empty string means the
    # Coordinator didn't plumb it (legacy / test path); per-domain
    # focus helpers must treat an empty string as "fall back to the
    # canonical sglang/vllm hint blocks". Used to switch the
    # "what to read first" bullets to atom-equivalent source paths
    # when ``framework == 'atom'``.
    framework: str = ""

    # Local source navigation hint (§3.5 §6 part 7)
    framework_source_roots: tuple[str, ...] = ()
    source_hint_directories: tuple[str, ...] = ()

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""


# ---------------------------------------------------------------------------
# Section 1 — Identity & autonomy
# ---------------------------------------------------------------------------
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
    # PR-A6 (Arbor-into-Hyperloom): per-domain expertise + focus
    # paragraph. Each domain template emphasises the surface area the
    # specialist should reason about + the typical winning techniques
    # (lifted from Arbor's orchestrator.md "agent expertise" table).
    rendered_focus_keys: set[str] = set()
    focus = _DOMAIN_FOCUS_TEMPLATES.get(inp.domain.key)
    if focus is not None:
        body.append("")
        body.append(f"### Domain focus — {inp.domain.key}")
        body.append("")
        body.extend(focus(inp))
        rendered_focus_keys.add(inp.domain.key)
    # Multi-tag dispatch: append the focus block of each additional
    # knowledge-domain tag (resolved to a representative specialist
    # key) so a combined-domain specialist sees every relevant surface.
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
    return body


# ---------------------------------------------------------------------------
# Section 2 — Hardware context
# ---------------------------------------------------------------------------
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
    # Workload context — surfacing concrete numbers prevents the
    # specialist from guessing (or assuming defaults) when reasoning
    # about whether a given knob is reachable for this run.
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


# ---------------------------------------------------------------------------
# Section 3 — Gap statement
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section 4 — Cortex KB sub-graph
# ---------------------------------------------------------------------------
def _is_cold_start(inp: SpecialistPromptInputs) -> bool:
    """Issue-J (Saturday May 2026): all three prior sources are empty.

    When the model is brand new to the KB (HTTP 4xx schema rejects on
    ``propose_point`` for the recipe canonical_id) AND PR Monitor has
    no domain-tagged PRs AND ``find-recipe`` returned no recipe, the
    specialist's ``## 4`` / ``## 5`` / ``## 6`` sections all render
    ``(none)``. Historically this caused specialists to return
    ``proposal_set=[]`` (no priors → no anchor → no candidates),
    which the orchestrator then read as "exhausted" and routed into
    ``no_more_leverage``. Detecting this condition lets us inject an
    explicit cold-start directive instead of relying on the model to
    self-recover.
    """
    return (
        not inp.kb_subgraph
        and not inp.warm_start_recipe
        and not inp.warm_start_lessons
        and not inp.warm_start_pitfalls
        and not inp.pr_feed
        and not inp.research_hints
    )


def _section_kb_subgraph(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 4. CORTEX KB SUB-GRAPH", ""]
    cold = _is_cold_start(inp)
    if not inp.kb_subgraph:
        if inp.research_hints:
            # Research hints stand in as the advisory prior when the KB
            # anchor is empty — co-equal with the KB sub-graph, never a
            # deterministic gate. This keeps the cold-start fallback from
            # firing whenever the scout produced fresh source-backed priors.
            rows.extend([
                "KB anchor is empty for this (model, hardware, domain), but "
                "the research scout collected source-backed priors this "
                "session. Treat these as your advisory prior (co-equal with "
                "the KB sub-graph; the Critic still gates the final answer):",
                "",
                inp.research_hints,
                "",
                "Anchor proposals on these hints where they fit the gap "
                "(Section 3) and hardware (Section 2).",
            ])
            return rows
        if cold:
            # Cold-start directive: replaces the bare "(none)" with a
            # specific instruction so the specialist proposes
            # canonical defaults from its domain focus block (Section
            # 1) rather than emitting an empty proposal_set. The
            # Critic still gates the final answer; this only ensures
            # the KB cold-start path doesn't degenerate to silence.
            rows.extend([
                "**COLD-START MODE — no priors available.**",
                "",
                "All three prior sources for this gap are empty:",
                "",
                "- KB sub-graph: ``(none)`` — Cortex anchor has no "
                "committed points for this (model, hardware, domain) "
                "tuple yet, OR the warmup hit a 4xx schema reject "
                "(common on first-time models).",
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
                "``residual_questions`` field to record what KB "
                "anchor / PR query a future round should pre-warm.",
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
                "(No KB sub-graph supplied. The Coordinator pre-warms this "
                "section via select_kb_for_domain before dispatch; an empty "
                "block means the anchor has no committed entries yet (cold "
                "start) or the warmup hit a soft failure. The specialist "
                "subprocess has no live KB connection — surface what you "
                "need in ``residual_questions`` so a future round can "
                "re-warm with a richer anchor.)",
            ])
        return rows
    rows.append("```json")
    rows.append(json.dumps(inp.kb_subgraph, sort_keys=True, indent=2))
    rows.append("```")
    return rows


# ---------------------------------------------------------------------------
# Section 4a — Roofline / TraceLens evidence
# ---------------------------------------------------------------------------
def _section_roofline_evidence(inp: SpecialistPromptInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section.

    Sourced from ``Coordinator._warm_specialist_params`` which mirrors
    :attr:`SharedState.last_trace_analyze`. Expected keys on
    ``inp.roofline_evidence``:

    * ``analysis_md_path``: absolute path to the TraceLens
      ``analysis.md`` (specialist Read tool can pull the full report on
      demand).
    * ``roofline_snapshot_id``: monotonic counter the orchestration
      prompt also surfaces.
    * ``executive_summary``: structured dict with
      ``compute_pct / idle_pct / comm_pct / top_bottleneck``
      (extracted via :func:`roofline_snapshot.extract_workload_summary`).
    * ``hot_kernels_top15``: list of hot-kernel dicts (top 8 already
      sliced by the warmer to bound token cost).
    * ``kernel_roofline_top15``: optional per-kernel roofline projection
      with AI / efficiency / utilization fields.

    Returns an empty section (just the heading + ``(none)`` placeholder)
    when ``roofline_evidence`` is empty so the specialist still sees the
    structural slot.
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


# ---------------------------------------------------------------------------
# Section 5 — Recipe summary
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section 5b — Related lessons (positive priors from prior KEEPs)
# ---------------------------------------------------------------------------
def _section_lessons(inp: SpecialistPromptInputs) -> list[str]:
    """Render KB ``kind=lesson`` points that previous KEEP decisions
    on this (model, hardware) wrote — the positive counterpart of
    the § 5 pitfalls block.

    Each lesson is shown compactly: the ``statement`` line (one
    actionable claim) + the ``measured_impact`` string (the numeric
    delta from the prior session), with ``applicable_models`` /
    ``applicable_hardware`` collapsed into a single header so the
    specialist can scan a dozen lessons at a glance instead of
    reading JSON dumps.
    """
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
        # Optional: confidence + source session hint + validator count
        # for "how transferable is this lesson?".
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        # GAP 4 — surface the validated_count first because "5 sessions
        # confirmed this" is the strongest cross-session signal. Fall
        # back to the singular ``source_session_id`` for legacy rows.
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
        # GAP 8 — version mismatch annotation. Surface this AFTER the
        # statement so the LLM sees ``- **X works on sglang** [from
        # sglang@0.4.5, you're on 0.5.11]`` and can decide if the
        # lesson still applies. Client-side ranking already downweighted
        # the lesson, but the LLM gets the final call.
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
    """GAP 8 — render a ``[from sglang@X.Y, you're on A.B]`` annotation
    when the lesson's framework_version differs from the current session.

    Returns an empty string when:

    * The lesson didn't carry a framework_version (legacy / pre-PR row).
    * The current session doesn't know its own framework_version
      (legacy SDK caller without manifest stack_fingerprint).
    * The versions match exactly.

    Format is intentionally compact (single bracket pair) so it
    doesn't dominate the bullet line; the meaningful action is "LLM
    still gets to decide".
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
    """Back-compat renderer for ``attrs.measured_impact``.

    Two shapes accepted:

    * Dict (GAP 3 — new shape): formatted as
      ``+12.3% (tput=678.0, depth=3, 2026-05-26)``.
    * String (legacy): returned verbatim.
    * Anything else (None / numbers): returned as ``str(raw)`` or "".
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


# ---------------------------------------------------------------------------
# Section 5d — Session snapshot (session_steward specialist only)
# ---------------------------------------------------------------------------
def _section_session_snapshot(inp: SpecialistPromptInputs) -> list[str]:
    """Inline the panoramic SharedState digest the session_steward
    specialist consumes to decide ``continue_explore`` /
    ``advance_to_kernel`` / ``stop_session``.

    Returns an empty list (section omitted entirely) when
    ``session_snapshot`` is empty, so non-steward specialists never
    see this section. The dispatcher populates this dict only for
    ``session_steward_specialist`` tasks (see
    :meth:`Coordinator._warm_specialist_params`).

    Renders as a single fenced JSON block — small enough that the LLM
    parses it in one pass, structured enough that the field-name
    references in the focus block resolve to concrete values.
    """
    snap = inp.session_snapshot or {}
    if not snap:
        return []
    rows = [
        "## 5d. SESSION SNAPSHOT (panoramic state for steward decision)",
        "",
        "```json",
        json.dumps(snap, sort_keys=True, indent=2),
        "```",
    ]
    return rows


# ---------------------------------------------------------------------------
# Section 5c — Known pitfalls (anti-priors from prior REVERTs)
# ---------------------------------------------------------------------------
def _section_pitfalls(inp: SpecialistPromptInputs) -> list[str]:
    """Render KB ``kind=pitfall`` points that previous REVERT / crash /
    OOM decisions on this (model, hardware) wrote — the negative
    counterpart of § 5b lessons.

    Same compact rendering as lessons: ``description`` (one actionable
    anti-pattern) + ``severity`` tag, with optional ``conf`` / ``src``
    metadata. The "do NOT repeat" framing is critical — the specialist
    must understand these are *forbidden* paths, not suggestions.

    Replaces the legacy ``raw`` JSON-dump rendering that the old
    ``traps(symptom=...)`` reader produced (which the LLM couldn't
    reliably parse).
    """
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
        # GAP 4 — repeat observations strengthen the "don't try this" signal.
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


# ---------------------------------------------------------------------------
# Section 6 — PR feed
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section 7 — Local source navigation hint
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section 8 — Output protocol
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Section 9 — Iron rules
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------
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
    # § 5d session snapshot — only the session_steward specialist
    # populates ``session_snapshot``; for everyone else this section
    # function returns ``[]`` and gets stripped by the flattener.
    # Insert between § 5c (pitfalls, index 6) and § 6 (PR feed, was
    # index 7). Done as conditional insert (rather than always in the
    # list) so the section number stays meaningful — non-steward
    # specialists don't see a "## 5d." header at all.
    snapshot_section = _section_session_snapshot(inp)
    if snapshot_section:
        user_sections.insert(7, snapshot_section)
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
