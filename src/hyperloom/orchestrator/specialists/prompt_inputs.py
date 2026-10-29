# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed inputs the Coordinator hands to the specialist prompt builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domains import DEFAULT_SPECIALIST_MAX_TURNS, SpecialistDomain
from .profile import MODE_PATCH


@dataclass(frozen=True)
class SpecialistPromptInputs:
    """Typed inputs the Coordinator hands to the prompt builder."""

    # Identity
    domain: SpecialistDomain
    max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS

    # ``tp`` defaults to 0 (sentinel for "unspecified"), not 1, so
    # comm_specialist doesn't veto its own TP proposals.
    gpu_type: str = ""
    allocated_gpu_ids: tuple[int, ...] = ()
    tp: int = 0
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
    # ``"agentx"`` for agentic trace replay, else synthetic; selects the
    # workload and grading blocks. ``agentx_corpus_shape`` supplies their
    # numbers, mirrored from SharedState so the prompt describes the corpus the
    # session actually replayed.
    benchmark_mode: str = ""
    agentx_corpus_shape: dict[str, Any] = field(default_factory=dict)
    agentx_grading: dict[str, Any] = field(default_factory=dict)
    agentx_backend: str = ""

    # Gap statement
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Experience service ``prompt_block`` read for this dispatch; empty when no Experience was rendered.
    experience_kb_block: str = ""

    # Roofline / TraceLens evidence from ``SharedState.last_trace_analyze``;
    # empty dict renders a placeholder.
    roofline_evidence: dict[str, Any] = field(default_factory=dict)
    # Rendered runtime findings of the current best's server log; empty omits the section.
    runtime_findings: str = ""

    # Recipe summary from the T0 warm-start recipe search
    # (``recipe_kb_t0._cascade_warm_start_search``)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 lessons — positive priors from prior KEEPs; rendered in the lessons section.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Extra knowledge-domain tags; each contributes a focus block to Section 1.
    extra_focus_tags: tuple[str, ...] = ()

    # Local source navigation hint. ``worktree_base`` is the tree the worktree
    # stands for -- the checkout it was cut from, or the installed tree it holds
    # a snapshot of -- and is empty when the specialist has no worktree.
    session_framework_tree: str = ""
    framework_source_roots: tuple[str, ...] = ()
    worktree_base: str = ""
    source_hint_directories: tuple[str, ...] = ()
    # Framework package directory relative to the worktree; empty without a worktree.
    worktree_package_dir: str = ""

    # Target model identity warmed from SharedState.model_name / model_path; "" => not recorded.
    model_name: str = ""
    model_path: str = ""
    # Structured model architecture features mirrored from SharedState.model_info;
    # machine-parseable companion to ``arch_notes``. Empty dict => not warmed.
    model_info: dict[str, Any] = field(default_factory=dict)
    # Pre-rendered static-recon checklist block (Markdown); only populated for
    # the static_recon_specialist dispatch.
    static_recon_checklist: str = ""

    # Enablement dispatch evidence, folded into the §1b mandate. Empty for
    # every non-enablement domain; the mandate omits whichever is empty.
    enablement_source_context: str = ""
    enablement_candidate_refs: tuple[str, ...] = ()
    # The serialized FailureSignature the round was dispatched on; required for
    # the enablement domain, empty for every other.
    enablement_failure_signature: dict[str, Any] = field(default_factory=dict)
    # Env / server-arg layers prior advanced rounds accepted; the bench for this
    # round launches with them, so the mandate has to name them.
    enablement_accepted_config: dict[str, Any] = field(default_factory=dict)

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""

    # Dispatch profile dials (see orchestrator.specialists.profile) that shape
    # single-domain / cross-domain / freeform prompting.
    scope: str = "domain"
    mode: str = MODE_PATCH
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

    # Operator-defined specialist focus, rendered verbatim; empty when not configured.
    custom_specialist_prompt: str = ""


__all__ = [
    "SpecialistPromptInputs",
]
