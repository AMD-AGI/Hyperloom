# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared action-surface constants and the action catalogue.

Keep ownership, transport, and prompt-visibility classifications here so
PolicyGate, prompt rendering, and CLI wiring do not grow separate
action-name lists.

:data:`ACTION_CATALOGUE` models only the fields production code reads:

* ``requires_lanes`` / ``lease_ttl_sec`` -- dispatch gate and GPU lease TTL
* ``side_effects`` -- stamped onto the dispatched task
* ``pipeline_phase`` -- runs-workspace ownership plus prompt grouping
* ``verdict_class`` -- selects the Critic prompt rule set
* the rest -- rendered into the Orchestration prompt catalogue
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


# Actions owned by the Kernel role; requested via request{target_agent="kernel_agent"}.
KERNEL_AGENT_OWNED_ACTIONS: frozenset[str] = frozenset(
    {
        "kernel_opt",
        "integrate",
        "gemm_tuning",
    }
)


# Kernel-owned action name -> the request ``kind`` its handler is registered
# under in ``request_handlers.KERNEL_REQUEST_HANDLERS``. The two differ, so the
# prompt must advertise the kind.
KERNEL_ACTION_REQUEST_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "kernel_opt": "run_optimization",
        "gemm_tuning": "run_gemm_tuning",
        "integrate": "integrate",
    }
)

assert set(KERNEL_ACTION_REQUEST_KINDS) == KERNEL_AGENT_OWNED_ACTIONS


# Request-kind aliases that route to a kernel-owned handler. apply_patch is
# an alias of integrate (both dispatch to integrate_handler); PolicyGate
# resolves the alias to its canonical owned action so the phase-action gate
# applies identically.
KERNEL_REQUEST_KIND_ALIASES: dict[str, str] = {
    "apply_patch": "integrate",
}


# Request ``kind`` -> the kernel-owned action it gates as, derived from the two
# tables above so a new kind cannot fall out of sync with the catalogue.
# ``trace_analyze`` is absent by design: it owns no action and no phase, and
# mapping it onto one would deny it everywhere.
REQUEST_KIND_TO_OWNED_ACTION: Mapping[str, str] = MappingProxyType(
    {
        **{kind: action for action, kind in KERNEL_ACTION_REQUEST_KINDS.items()},
        **KERNEL_REQUEST_KIND_ALIASES,
    }
)


# Request kinds the Coordinator dispatches itself at KERNEL entry; PolicyGate
# rejects them from an LLM, which would bypass the lane's gate and accounting.
# Unlike ``COORDINATOR_INTERNAL_ACTIONS`` these are request kinds, not actions:
# they have no executor and no prompt entry.
COORDINATOR_OWNED_KERNEL_REQUEST_KINDS: frozenset[str] = frozenset(
    {
        "run_collective",
    }
)


# Coordinator-managed actions that agents should not directly propose.
INTERNAL_ONLY_ACTION_NAMES: frozenset[str] = frozenset(
    {
        "conc_sweep",
        "roofline",
        "profile",
        "replay_warm_recipe",
        # Off-loop compiled-component builds; dispatched by the Coordinator,
        # never by an LLM agent.  Not in ACTION_CATALOGUE to avoid pulling the
        # kind into PHASE_LLM_PROPOSABLE_ACTIONS or _RUNS_ACTIONS (which would
        # create a runs/ workspace and collide with the enablement/builds/
        # attempt-root contract).
        "targeted_build",
    }
)


COORDINATOR_INTERNAL_ACTIONS: frozenset[str] = INTERNAL_ONLY_ACTION_NAMES


# Robustness-only actions (driven via its action-ladder); Orchestration must
# ALERT instead. ``recover`` walks SIGTERM/SIGKILL against server owners.
ROBUSTNESS_DELEGATE_ONLY_ACTIONS: frozenset[str] = frozenset(
    {
        "recover",
    }
)


# Actions rendered in the Orchestration prompt for full kernel-enabled runs.
# Prompt visibility only; phase_state and PolicyGate decide legality per tick.
FULL_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis",
    "baseline",
    "roofline",
    "explore",
    "specialist",
    "integrate_patch",
    "kernel_opt",
    "integrate",
    "gemm_tuning",
    "report",
)


# Prompt-visible actions for --no-kernel runs. Kernel-owned request actions
# and analysis actions that only feed kernel optimization stay hidden.
NO_KERNEL_AGENT_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis",
    "baseline",
    "explore",
    "specialist",
    "integrate_patch",
    "report",
)


@dataclass(frozen=True)
class ActionMetadata:
    """One action's dispatch contract plus its prompt-catalogue copy."""

    name: str
    family: str
    pipeline_phase: str
    verdict_class: str
    expected_gain_pct: tuple[float, float]
    accuracy_risk: float
    crash_risk: float
    typical_runtime_min: float
    lease_ttl_sec: int
    side_effects: tuple[str, ...]
    description: str
    requires_lanes: tuple[str, ...] = ()


ACTION_CATALOGUE: Mapping[str, ActionMetadata] = MappingProxyType(
    {
        "baseline": ActionMetadata(
            name="baseline",
            family="prep",
            pipeline_phase="measure",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_min=5.0,
            lease_ttl_sec=4200,
            requires_lanes=("server_lifecycle", "benchmark_lane"),
            side_effects=("launches_server", "reads_server", "writes_results"),
            description="Launch a fresh server with NO accepted modifications, run Magpie benchmark, and set baseline_tput.",
        ),
        "conc_sweep": ActionMetadata(
            name="conc_sweep",
            family="shallow",
            pipeline_phase="explore",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_min=30.0,
            lease_ttl_sec=9000,
            requires_lanes=("server_lifecycle", "benchmark_lane"),
            side_effects=("launches_server", "writes_results"),
            description=(
                "Post-sweep concurrency comparison: benchmark baseline vs current_best across a CONC ladder. "
                "On by default; opt out via --no-enable-conc-sweep; bounded by --conc-sweep-total-budget-sec "
                "(default 2.5h)."
            ),
        ),
        "explore": ActionMetadata(
            name="explore",
            family="shallow",
            pipeline_phase="explore",
            verdict_class="exploration",
            expected_gain_pct=(2.0, 12.0),
            accuracy_risk=0.0,
            crash_risk=0.1,
            typical_runtime_min=12.0,
            lease_ttl_sec=7200,
            requires_lanes=("server_lifecycle", "benchmark_lane"),
            side_effects=("launches_server", "reads_server", "writes_results"),
            description=(
                "Apply a batch of N candidate variants serially; KEEP/REVERT each, stack onto optimization_stack. "
                "Each variant is benched on the stack (replaces backends/params/validate_stack)."
            ),
        ),
        "gemm_tuning": ActionMetadata(
            name="gemm_tuning",
            family="deep_kernel",
            pipeline_phase="deep",
            verdict_class="exploration",
            expected_gain_pct=(5.0, 20.0),
            accuracy_risk=0.1,
            crash_risk=0.1,
            typical_runtime_min=20.0,
            lease_ttl_sec=5400,
            requires_lanes=("server_lifecycle", "workspace_mutation", "benchmark_lane"),
            side_effects=("workspace_write", "server_restart", "writes_config"),
            description=(
                "GEMM dispatch tuning request. Current GEAK owns the default KERNEL phase and decides applicability "
                "internally. Private forge tuning is available only when the operator set exactly "
                "KERNEL_OPT_BACKEND_ORDER=forge."
            ),
        ),
        "integrate": ActionMetadata(
            name="integrate",
            family="deep_kernel",
            pipeline_phase="deep",
            verdict_class="promotion",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.15,
            crash_risk=0.15,
            typical_runtime_min=25.0,
            lease_ttl_sec=3600,
            requires_lanes=("server_lifecycle", "workspace_mutation", "benchmark_lane"),
            side_effects=("workspace_write", "server_restart", "patches_inductor_cache"),
            description=(
                "REQUEST kernel: apply a KEEP'd kernel patch, re-baseline E2E, and emit KEEP / REVERT / NEEDS_REVIEW."
            ),
        ),
        "integrate_patch": ActionMetadata(
            name="integrate_patch",
            family="shallow",
            pipeline_phase="explore",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 12.0),
            accuracy_risk=0.1,
            crash_risk=0.15,
            typical_runtime_min=10.0,
            lease_ttl_sec=3600,
            requires_lanes=("server_lifecycle", "workspace_mutation", "benchmark_lane"),
            side_effects=("workspace_write", "server_restart", "launches_server", "reads_server", "writes_results"),
            description=(
                "Apply specialist worktree patches to framework_source_roots, restart server, run throughput + "
                "accuracy gate, KEEP or REVERT. Deterministic executor for FRAMEWORK_AGENT; also serves "
                "the enablement launch-only build probe and framework-agent authoring lanes."
            ),
        ),
        "kernel_opt": ActionMetadata(
            name="kernel_opt",
            family="deep_kernel",
            pipeline_phase="deep",
            verdict_class="exploration",
            expected_gain_pct=(5.0, 25.0),
            accuracy_risk=0.1,
            crash_risk=0.2,
            typical_runtime_min=60.0,
            lease_ttl_sec=7200,
            requires_lanes=("server_lifecycle", "workspace_mutation", "benchmark_lane"),
            side_effects=("workspace_write", "server_restart", "launches_server"),
            description=(
                "REQUEST kernel: parallel-submit kernel optimization candidates for one reusable native kernel id "
                "picked from the latest profile."
            ),
        ),
        "profile": ActionMetadata(
            name="profile",
            family="analysis",
            pipeline_phase="analysis",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_min=3.0,
            lease_ttl_sec=2700,
            requires_lanes=("profile_lane",),
            side_effects=("reads_server", "writes_results"),
            description=(
                "Coordinator-internal: lightweight roofline alternative — torch_profiler trace only, no analysis.md. "
                "Enqueued when ``--no-enable-roofline``; LLM-proposed delegate is denied."
            ),
        ),
        "recover": ActionMetadata(
            name="recover",
            family="resilience",
            pipeline_phase="support",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.1,
            typical_runtime_min=5.0,
            lease_ttl_sec=1200,
            requires_lanes=("server_lifecycle", "workspace_mutation"),
            side_effects=("workspace_write", "server_restart", "reads_checkpoint"),
            description=(
                "Restore the workspace from the last good checkpoint and relaunch the server after a crash or REVERT."
            ),
        ),
        "replay_warm_recipe": ActionMetadata(
            name="replay_warm_recipe",
            family="prep",
            pipeline_phase="measure",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 25.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_min=5.0,
            lease_ttl_sec=4200,
            requires_lanes=("server_lifecycle", "benchmark_lane"),
            side_effects=("launches_server", "reads_server", "writes_results"),
            description=(
                "Coordinator-internal one-shot replay of T0 warm_start_recipe.best_config; reproducing "
                "≥ --warm-replay-min-reproduce-pct of the historical gain pushes the warm config onto "
                "optimization_stack."
            ),
        ),
        "report": ActionMetadata(
            name="report",
            family="shallow",
            pipeline_phase="finalize",
            verdict_class="archival",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.0,
            typical_runtime_min=2.0,
            lease_ttl_sec=300,
            side_effects=("writes_results",),
            description=(
                "Write final.md / final.json under reports/. Coordinator auto-flushes deterministic report at the "
                "deadline (invariant); LLM may propose earlier on stop_reason or low remaining."
            ),
        ),
        "roofline": ActionMetadata(
            name="roofline",
            family="analysis",
            pipeline_phase="analysis",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_min=10.0,
            lease_ttl_sec=2700,
            requires_lanes=("profile_lane",),
            side_effects=("reads_server", "writes_results"),
            description=(
                "Composite action: runs profile + trace_analyze atomically to produce a fresh TraceLens analysis.md "
                "snapshot. Required prerequisite for explore / kernel_opt."
            ),
        ),
        "session_breakdown": ActionMetadata(
            name="session_breakdown",
            family="shallow",
            pipeline_phase="finalize",
            verdict_class="archival",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.0,
            typical_runtime_min=0.2,
            lease_ttl_sec=90,
            side_effects=("writes_results",),
            description=(
                "Refresh $SESSION_DIR/session_breakdown.json for downstream dashboards (cheap, idempotent). "
                "End-of-session export already runs from cli.py finally; only dispatch mid-run for live consumers."
            ),
        ),
        "specialist": ActionMetadata(
            name="specialist",
            family="creative",
            pipeline_phase="explore",
            verdict_class="exploration",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.0,
            typical_runtime_min=6.0,
            lease_ttl_sec=1800,
            requires_lanes=("research_lane",),
            side_effects=("workspace_write",),
            description=(
                "Dispatch an LLM specialist on research_lane; reads KB / PR feed for knowledge-domain tags, may write "
                "worktree patches, emits one specialist_done intent."
            ),
        ),
        "target_analysis": ActionMetadata(
            name="target_analysis",
            family="prep",
            pipeline_phase="prep",
            verdict_class="archival",
            expected_gain_pct=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.0,
            typical_runtime_min=0.1,
            lease_ttl_sec=60,
            side_effects=("reads_model_files", "writes_state"),
            description=(
                "Runs first in PRELUDE, before baseline; always writes target_analysis/target_baseline.json. With "
                "--compare-against-gpu set, fetches InferenceX reference; otherwise writes a "
                "'no_target_gpu_configured' marker. Advisory only."
            ),
        ),
    }
)


__all__ = [
    "ACTION_CATALOGUE",
    "ActionMetadata",
    "COORDINATOR_INTERNAL_ACTIONS",
    "COORDINATOR_OWNED_KERNEL_REQUEST_KINDS",
    "FULL_ENABLED_ACTIONS",
    "INTERNAL_ONLY_ACTION_NAMES",
    "KERNEL_ACTION_REQUEST_KINDS",
    "KERNEL_AGENT_OWNED_ACTIONS",
    "KERNEL_REQUEST_KIND_ALIASES",
    "NO_KERNEL_AGENT_ENABLED_ACTIONS",
    "REQUEST_KIND_TO_OWNED_ACTION",
    "ROBUSTNESS_DELEGATE_ONLY_ACTIONS",
]
