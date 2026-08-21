# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EnablementRound: per-round enablement state, nested in SharedState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnablementRound:
    """State scoped to a single enablement repair attempt."""

    # Eval-origin enablement carriers: set when the first baseline runs but its
    # accuracy eval fails, so the enablement pump/gate can reconstruct the trigger
    # and re-run the same eval contract. Empty for boot-origin enablement.
    origin: str = ""
    accuracy_floor: float = 0.0
    probe_config_path: str = ""
    eval_contract_fingerprint: str = ""
    baseline_eval_evidence: str = ""
    baseline_eval_kind: str = ""
    observed_accuracy: float = 0.0
    observed_task: str = ""
    observed_metric: str = ""
    pending: bool = False
    # Set on an eval-origin KEEP: the patch passed the gate but a genuine baseline
    # must revalidate accuracy before the run is considered enabled.
    validation_pending: bool = False
    # ``launch_log``: captured launch/traceback text when baseline cannot launch.
    # ``attempts``: number of dispatches (candidate rotation / idempotency).
    # ``succeeded``: terminal KEEP guard.
    launch_log: str = ""
    attempts: int = 0
    succeeded: bool = False
    # Identity of the currently-running authoring specialist; empty when no round
    # is in flight. In-flight status is derived from the task registry, not stored.
    inflight_task_id: str = ""
    # Task id of the most recently completed enablement specialist round.
    last_specialist_task_id: str = ""
    # Ordered, deduped patch paths from prior enablement rounds that made forward
    # progress; re-applied as a base before the next round's patch.
    kept_patches: list = field(default_factory=list)
    # Framework source tree the kept patches were applied against. Persisted so a
    # phase-synthesised round, which carries no framework_root, does not drop it.
    framework_root: str = ""
    # Ordered, deduped allowlisted env-setup shell commands prior rounds ran;
    # re-run idempotently by integrate_patch before applying patches and booting.
    setup_commands: list = field(default_factory=list)
    # Consecutive enablement rounds that neither became runnable nor advanced to
    # a new failure signature; at _ENABLEMENT_MAX_STALL the loop stops with
    # stop_reason enablement_stalled.
    stall_streak: int = 0
    # Launch-log hashes already recorded as needs_human_review; one record per log.
    human_review_logged: list = field(default_factory=list)
    # Path to the materialized config produced by the KEEP'd candidate bench.
    accepted_config_path: str = ""
    # Env/arg layers the KEEP'd bench ran with; replayed by the revalidation baseline.
    accepted_config: dict = field(default_factory=dict)
    # Task identity for the current revalidation baseline task.
    revalidation_task_id: str = ""
    # Monotonically increasing counter for fresh revalidation idempotency keys.
    revalidation_generation: int = 0
    # Attempt-scoped runtime acquisition state.
    stack_actions: list = field(default_factory=list)
    active_runtime: dict = field(default_factory=dict)
    attempt_runtimes: list = field(default_factory=list)
    kept_stack_action: dict = field(default_factory=dict)
    localization_manifest: list = field(default_factory=list)
    # Off-loop targeted-build state.
    build_manifest: list = field(default_factory=list)
    last_build_failure: dict = field(default_factory=dict)
    build_novelty: list = field(default_factory=list)
    candidate_refs: list = field(default_factory=list)
    # Patches written by the last round that were grounding-dropped (all missing
    # targets). Injected into the next round's prompt so the specialist knows
    # to switch to artifacts instead of repeating a doomed patch.
    last_grounding_drop_reason: list = field(default_factory=list)
    # Installed whole-file artifacts that have been KEPT or ADVANCED; replayed
    # at the start of each subsequent round (analogous to kept_patches).
    kept_artifacts: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnablementRound":
        """Construct from a raw mapping; unknown keys dropped, missing keys default."""
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)
