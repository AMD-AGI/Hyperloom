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


def _focus_serving_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **vLLM / SGLang scheduler / cuda_graph / kv_cache** code.",
        "",
        "**What to read first**",
        "- `vllm/v1/engine/` and `vllm/v1/worker/` (scheduler, model_runner).",
        "- `sglang/python/sglang/srt/scheduler/` and `sglang/python/sglang/srt/managers/`.",
        "- KB anchor `framework.*` (cuda_graph / batching / chunked_prefill / kv_cache).",
        "",
        "**Winning techniques to consider**",
        "- `--enable-chunked-prefill` + matched `--max-num-batched-tokens`.",
        "- `--enforce-eager=false` + cuda graph capture for stable batch sizes.",
        "- `--kv-cache-dtype fp8_e4m3` when the gap is HBM-bound (gate accuracy!).",
        "- `--max-num-seqs` tuning at concurrency boundaries.",
        "",
        "**Pitfalls (historical REVERTs)**",
        "- Raising `--max-num-seqs` past 512 on MI300X → OOM on 671B MoE models.",
        "- `cuda_graph` + dynamic batch sizes → silent recapture cost > savings.",
        "- Chunked prefill without `--max-num-batched-tokens` → tail latency",
        "  regressions invisible to throughput-only benches.",
    ]


def _focus_kernel_switch_specialist(inp: SpecialistPromptInputs) -> list[str]:
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
        "**What to read first** (no Bash needed — everything is in",
        "$SESSION_DIR/state.json which the Coordinator pre-warms below):",
        "- ``optimization_stack`` — what's been KEEP'd. Count entries +",
        "  inspect ``gain_per_stack_entry`` for diminishing returns.",
        "- ``explore_search.rejected`` — REVERT reasons grouped by",
        "  ``stack_unstable`` / ``gain_below_threshold``. A long tail of",
        "  one kind is a signal.",
        "- ``specialist_rounds`` — empty_streak counters per domain. Three",
        "  consecutive ``empty=True`` rounds across the active domains is a",
        "  hard plateau signal.",
        "- ``gaps[]`` — open gaps the Coordinator believes still exist.",
        "  If non-empty and the recommended specialist domain has not been",
        "  exhausted, ``continue_explore`` may be justified.",
        "- ``policy_denial_history`` (tail) — when the LLM has been",
        "  thrashing against the same rule, this is evidence that further",
        "  exploration is unlikely to land KEEPs.",
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
    tp: int = 0
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""
    # Workload context (mirrored from SharedState by
    # Coordinator._warm_specialist_params; renders in section 2 so
    # the specialist sees the actual benchmark workload instead of
    # the dataclass default).
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0

    # Gap statement (§3.5 §6 part 3)
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Cortex KB sub-graph (§3.5 §6 part 4)
    kb_subgraph: dict[str, Any] = field(default_factory=dict)

    # Roofline / TraceLens evidence (§3.5 §6 part 4a — post-N31).
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

    # PR feed (§3.5 §6 part 6)
    pr_feed: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # F2-3 framework_pr_scout sub_kind: Coordinator pre-fetched PR
    # candidates from ``fa candidates`` (metadata only — diff bodies
    # the specialist fetches itself via the curl template in the
    # rendered FRAMEWORK PR CANDIDATES section). Empty list → section
    # renders empty / placeholder. ``sub_kind`` mirrors the dispatch
    # ``params.sub_kind`` so the section can short-circuit when the
    # specialist is NOT a framework_pr_scout (avoid leaking PR feed
    # noise into other domains).
    sub_kind: str = ""
    pr_candidates: list[dict[str, Any]] = field(default_factory=list)

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
    focus = _DOMAIN_FOCUS_TEMPLATES.get(inp.domain.key)
    if focus is not None:
        body.append("")
        body.append(f"### Domain focus — {inp.domain.key}")
        body.append("")
        body.extend(focus(inp))
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
        rows.append(f"Architecture notes: {inp.arch_notes}")
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
        and not inp.warm_start_pitfalls
        and not inp.warm_start_lessons
        and not inp.pr_feed
    )


def _section_kb_subgraph(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 4. CORTEX KB SUB-GRAPH", ""]
    cold = _is_cold_start(inp)
    if not inp.kb_subgraph:
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
# Section 4a — Roofline / TraceLens evidence (post-N31)
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
    has_recipe = bool(inp.warm_start_recipe)
    has_pitfalls = bool(inp.warm_start_pitfalls)
    if not has_recipe and not has_pitfalls:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if has_recipe:
        rows.append("**find-recipe result:**")
        rows.append("```json")
        rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, indent=2))
        rows.append("```")
    if has_pitfalls:
        rows.append("")
        rows.append("**Known pitfalls (do NOT repeat):**")
        for p in inp.warm_start_pitfalls:
            rows.append(f"- {json.dumps(p, sort_keys=True)}")
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
        impact = str(attrs.get("measured_impact") or "").strip()
        # Optional: confidence + source session hint for "how
        # transferable is this lesson?"
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        src_sid = str(attrs.get("source_session_id") or "").strip()
        if src_sid:
            meta_bits.append(f"src={src_sid}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        rows.append(f"- **{statement}**{meta}")
        if impact:
            rows.append(f"    impact: {impact}")
    if len(rows) == 2:  # only the header + blank line, all lessons filtered out
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
# Section 6b — Framework PR candidates (F2-3 framework_pr_scout sub_kind)
# ---------------------------------------------------------------------------
def _section_framework_pr_candidates(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Render the FRAMEWORK PR CANDIDATES section for serving_specialist
    runs with ``sub_kind='framework_pr_scout'``.

    Coordinator pre-fetches the PR list via the ``fa candidates`` CLI
    (see :mod:`framework_agent_client`); the section instructs the
    specialist how to pull each PR's diff body via ``curl`` inside the
    subprocess sandbox (the diff itself is too large to inline here).

    Returns an empty list (section omitted) when the specialist is not
    a framework_pr_scout OR when no candidates landed (graceful
    degrade for offline / fa-binary-missing scenarios).
    """
    if (inp.sub_kind or "").strip() != "framework_pr_scout":
        return []
    candidates = inp.pr_candidates or []
    if not isinstance(candidates, list) or not candidates:
        return []
    rows = ["## 6b. FRAMEWORK PR CANDIDATES", ""]
    rows.append(
        f"Coordinator pre-fetched these **{len(candidates)}** PR "
        "candidates from `fa candidates`. Diff bodies are NOT included; "
        "fetch them yourself before authoring patches."
    )
    rows.append("")
    rows.append("| # | repo | pr_number | ref | title | summary | score | diff_url |")
    rows.append("|---|---|---:|---|---|---|---:|---|")
    for i, c in enumerate(candidates[:20], start=1):
        if not isinstance(c, dict):
            continue
        repo = str(c.get("repo") or "")
        pr_number = c.get("pr_number") or c.get("number") or ""
        ref = str(c.get("ref") or "")
        title = str(c.get("title") or "").replace("|", "/")
        summary = str(c.get("summary") or "").replace("|", "/")
        if len(summary) > 140:
            summary = summary[:137] + "..."
        score = c.get("score")
        score_str = (
            f"{float(score):.2f}" if isinstance(score, (int, float))
            else "—"
        )
        diff_url = str(c.get("diff_url") or c.get("source_url") or "")
        rows.append(
            f"| {i} | {repo} | {pr_number} | {ref} | {title} | "
            f"{summary} | {score_str} | {diff_url} |"
        )
    rows.append("")
    rows.append("### How to fetch a PR diff")
    rows.append("")
    rows.append(
        "```bash"
    )
    rows.append(
        "mkdir -p $WORKTREE/incoming"
    )
    rows.append(
        "curl -fsSL -o $WORKTREE/incoming/<pr_number>.diff '<diff_url>'"
    )
    rows.append(
        "git -C $FRAMEWORK_ROOT apply --check "
        "$WORKTREE/incoming/<pr_number>.diff"
    )
    rows.append(
        "# Then re-author into worktree/patches/NNN_<slug>.patch via Edit"
    )
    rows.append("```")
    rows.append("")
    rows.append(
        "**Iron rule:** do NOT commit a raw GitHub diff into "
        "`worktree/patches/` — always Edit your own `.patch` so "
        "`integrate_patch` can attribute provenance correctly. The "
        "incoming GitHub diff is reference material; the patches/ "
        "entry is your own work product."
    )
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
            "a reject (and a refuted KB edge that will follow you on "
            "future rounds)."
        ),
        "- ``patches_written`` (PR-A2) lists paths (relative to your",
        "  workspace or worktree) of any unified-diff patch files you",
        "  authored this round. Empty list = no patches; downstream",
        "  ``integrate_patch`` action skips when empty.",
        "- ``empty=true`` is legitimate when you have no actionable proposals;",
        "  in that case ``proposal_set=[]`` and you must put the reason in",
        "  ``summary``.",
        "- ``new_findings`` becomes HYPOTHESIZED KB edges at T4 commit even",
        "  when not KEEP'd this round — surface anything you learned.",
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
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.2 / Inv-5.3)",
        "",
        "1. **NEVER** touch the serving GPU (no Magpie / no benchmark / no",
        "   server restart / no vllm or sglang process control). The",
        "   Coordinator runs benchmarks; you only propose what to try and",
        "   optionally author patches.",
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
        _section_pr_feed(inp),            # 6: § 6
        _section_source_hint(inp),        # 7: § 7
    ]
    # F2-3: framework_pr_scout candidates ride right after PR feed so
    # the specialist sees PR Monitor warm cache + the fa-fetched
    # candidates in adjacent sections. Returns an empty list for any
    # other sub_kind / empty candidate list, so non-framework_pr_scout
    # specialists never see this section.
    fa_candidates_section = _section_framework_pr_candidates(inp)
    if fa_candidates_section:
        # Insert after the PR feed section (index 6 — the list above
        # shows PR feed at index 6 after § 5b was added).
        user_sections.insert(7, fa_candidates_section)
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
