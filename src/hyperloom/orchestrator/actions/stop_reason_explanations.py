# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the run report says about each terminal ``stop_reason``."""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.stop_reasons import (
    AGENTX_PREFLIGHT_STOP_REASON,
    BACKEND_UNHEALTHY_STOP_REASON,
    PATCH_RECOVERY_INCOMPLETE_STOP_REASON,
)
from ..measurement.baseline_budget import BASELINE_BUDGET_STOP_EXPLANATIONS


#: Prose for every stop a session can end on; the report reads it, a module that decides a stop adds to it.
STOP_REASON_EXPLANATIONS: dict[str, str] = {
    # Terminal reasons the coordinator loop sets directly.
    "target_reached": "Target reached: a requested --target-gain / --target-tput / --target-roofline was met.",
    "time_exhausted": (
        "Wall-clock budget (--max-hours) was exhausted, or too little of it remained for another macro-cycle; "
        "the best validated result was kept."
    ),
    "max_ticks": "The coordinator hit its max-ticks safety cap; the best validated result was kept.",
    "signal": "Stopped on an OS stop signal (e.g. SIGINT / SIGTERM).",
    "emergency": (
        "Emergency stop: recoverable crashes spiked past the safety threshold inside the crash "
        "window; the best validated result was preserved before exit."
    ),
    "coordinator_exception": "The coordinator loop raised an unhandled exception; the last validated result was preserved.",
    "custom": "A caller-supplied stop condition fired.",
    "unknown": "No specific stop reason was recorded (e.g. a terminal session was resumed); treat as unclassified.",
    "baseline_failed": "Baseline never produced a valid measurement; see the failure summary / server log.",
    "server_argv_invalid": (
        "The composed server argv was refused before any server was launched: the installed framework's parser "
        "rejected a flag, or Hyperloom could not split the args into argv tokens intact. Nothing in the framework "
        "source is at fault: correct the offending flag or value in --server-args / the recipe, then re-run."
    ),
    "environment_fault": (
        "The checkpoint path this run serves resolves to nothing on this host. No patch to the model or the "
        "framework source repairs that, so the run stopped instead of spending rounds authoring against the machine."
    ),
    # PRELUDE-phase early exits (before optimization begins).
    "prelude_baseline_failed": "PRELUDE baseline failed before optimization could start; see the baseline failure summary.",
    "time_exhausted_during_prelude": (
        "The session's wall-clock budget ran out during preparation, before optimization began. Whatever PRELUDE "
        "was doing when the clock reached zero — measuring the baseline, bringing up the framework agent, taking "
        "the roofline — is where the time went; the phase record shows which arms ran and what each cost."
    ),
    "prelude_budget_exhausted": (
        "Preparation used up everything the session holds outside the reserve kept for optimization, and no "
        "baseline was ever anchored. With a baseline this leg had already failed and no arm the machine can buy "
        "on its own, staying in PRELUDE would have spent the optimization reserve on waiting, so the run stopped "
        "with the clock still showing time. The phase record shows what preparation spent it on. A resumed "
        "session is granted one baseline attempt whatever the budget, so fix the fault in the failure summary "
        "before resuming, and raise the budget for the optimization that attempt is meant to open."
    ),
    "prelude_cold_anchor_low_budget": (
        "The baseline's hot pass was skipped because the clock could not cover it together with one variant to "
        "measure against it, so the only figure available is the cold pass's — depressed by the server boot, the "
        "first request's kernel compile and the graph capture. Optimizing against it would report every variant as "
        "an improvement over a baseline that was never the baseline, so the run stopped with the figure kept and "
        "marked. Resume with more budget to measure a comparable baseline."
    ),
    **BASELINE_BUDGET_STOP_EXPLANATIONS,
    # Recipe KB knowledge-plane bootstrap failures.
    "warm_replay_rollback_failed": (
        "Warm replay rollback could not restore every Recipe/Kernel mutation; "
        "the run stopped to avoid continuing from an uncertain code state."
    ),
    "active_inferencex_checkout_missing": (
        "The session-authoritative InferenceX checkout is missing during "
        "resume; the run stopped instead of falling back to a different tree."
    ),
    # Search / phase plateaus and completions.
    "sweep_done": "SWEEP finished the concurrency ladder.",
    "sweep_failed": "The concurrency sweep reached a failed terminal result.",
    "framework_agent_phase_done": "The framework-enablement agent completed its phase.",
    "global_converged": (
        "Cyclic phases converged: repeated macro-cycles stopped yielding new validated gain, or every steering "
        "direction is saturated."
    ),
    # Pre-flight gates (fail fast before booting a server).
    "model_context_window_too_small": "Preflight gate: the model's max context window cannot hold the requested ISL + OSL.",
    "unsupported_model_arch": "Preflight gate: the model architecture (e.g. multimodal / vision) is unsupported.",
    "model_config_incompatible": (
        "Preflight gate: config.json is corrupt or declares RoPE scaling without a max-position field, "
        "which would crash engine init."
    ),
    "baseline_arg_error": "Two or more baseline attempts failed argument or configuration validation, so further baseline retries were stopped.",
    "enablement_attempts_exhausted": "The enablement loop stopped after too many consecutive rounds bought no ground. A bring-up that is still clearing new boot failures is bounded by the run's wall clock instead.",
    "baseline_accuracy_failed": "The baseline produced no accuracy result even though the accuracy test was expected to run (broken eval or missing quality gate). The run stopped rather than optimize against an unvalidated baseline.",
    AGENTX_PREFLIGHT_STOP_REASON: (
        "The MLPerf agentic client failed its execution preflight. Check the named missing dependency "
        "and rerun setup before resuming; a framework patch cannot repair this environment gap."
    ),
    PATCH_RECOVERY_INCOMPLETE_STOP_REASON: (
        "A patch lifecycle owed the framework tree a revert and could not finish it, so the tree still "
        "holds patches nothing measured against. The run stopped instead of attributing later results to "
        "a baseline that is not on disk. The recovery keeps its checkpoint and retries the teardown on the "
        "next resume; if that retry also fails, reconcile the tree by hand against the recorded backup "
        "manifests before resuming."
    ),
    # Host-level terminals: something outside the model ended the run.
    "supervisor_coordinator_died": "The out-of-band supervisor found the coordinator's process gone; this record was written by the supervisor because there was no coordinator left to write one.",
    "supervisor_tick_stalled": "The out-of-band supervisor found the coordinator's tick not advancing inside its stall window and asked the session to end.",
    BACKEND_UNHEALTHY_STOP_REASON: (
        "A reactor agent stopped producing decisions, so the run stopped instead of spending the rest of its budget "
        "on it; the best validated result was kept. A no_intent_streak observation naming the agent means it kept "
        "answering without emitting an intent; for the orchestration agent, agents/orchestration/mcp_setup.json "
        "records whether Claude Code offered it the emit_intent tool. Without one, its LLM backend kept failing for "
        "an hour without one successful turn, and the backend_error observations name the agent and carry the error "
        "each call returned: repair the backend (credentials, endpoint, quota, or a prompt the model rejects). "
        "Resume once the cause is fixed."
    ),
}
