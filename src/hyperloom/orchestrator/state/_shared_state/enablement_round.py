# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EnablementRound: per-round enablement state, nested in SharedState."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class EnablementRound:
    """State scoped to a single enablement repair attempt."""

    # Eval-origin enablement carriers: set when the first baseline runs but its accuracy eval fails, so the enablement
    # pump/gate can reconstruct the trigger and re-run the same eval contract.
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
    # Set on an eval-origin KEEP: the patch passed the gate but a genuine baseline must revalidate accuracy before the
    # run is considered enabled.
    validation_pending: bool = False
    # ``launch_log``: captured launch/traceback text when baseline cannot launch.
    launch_log: str = ""
    launch_observation_path: str = ""
    succeeded: bool = False
    # Task id of the most recently completed enablement specialist round.
    last_specialist_task_id: str = ""
    # Authoritative per-round record: list of {"patches": [...], "artifacts": [...]} dicts, one entry per accepted
    # round in order. kept_patches and kept_artifacts are derived from this list and kept for downstream
    # compatibility.
    kept_rounds: list = field(default_factory=list)
    # Flat ordered deduped patch paths derived from kept_rounds; re-applied as a base before the next round's patch.
    kept_patches: list = field(default_factory=list)
    # Framework source tree the kept patches were applied against.
    framework_root: str = ""
    # Ordered, deduped allowlisted env-setup shell commands prior rounds ran; re-run idempotently by integrate_patch
    # before applying patches and booting.
    setup_commands: list = field(default_factory=list)
    # Digests of server argvs that have already spent their one drop-only
    # preflight repair. A distinct argv gets exactly one; the same argv failing
    # again is terminal.
    argv_repairs: list = field(default_factory=list)
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
    active_runtime: dict = field(default_factory=dict)
    attempt_runtimes: list = field(default_factory=list)
    kept_stack_action: dict = field(default_factory=dict)
    localization_manifest: list = field(default_factory=list)
    # Off-loop targeted-build state.
    build_manifest: list = field(default_factory=list)
    last_build_failure: dict = field(default_factory=dict)
    build_novelty: list = field(default_factory=list)
    candidate_refs: list = field(default_factory=list)
    # Why the last round's patches were all dropped for absent targets; injected into the next round's mandate so it
    # stops writing diffs that cannot apply.
    last_grounding_drop_reason: list = field(default_factory=list)
    # Serialized ApplyFeedback records from the last round's failed ``git apply``
    # (stderr, reject hunks, target source window), injected into the next
    # round's mandate so it re-grounds instead of resubmitting the same diff.
    last_apply_feedback: list = field(default_factory=list)
    # Whether the last round's kept patches targeted more than one source tree;
    # injected into the next mandate so the specialist splits them per round.
    patches_span_multiple_roots: bool = False
    # Flat ordered deduped artifact dicts derived from kept_rounds (last-wins per
    # target); used for the specialist mandate note and session-breakdown reporting.
    kept_artifacts: list = field(default_factory=list)
    # Append-only, one row per ATTEMPTED setup execution. Parallel to
    # setup_commands, which stays a deduped command list: a command that ran
    # twice, or ran and failed, has no representation there at all.
    setup_executions: list = field(default_factory=list)
    # One record per root that contributed a patch or artifact to the accepted
    # stack; patch steps and artifacts carry its id, so a round spanning several
    # trees stays representable where framework_root keeps only the last one.
    roots: list = field(default_factory=list)
    # Per-patch apply root, where the authoring stage recorded one.
    patch_roots: dict = field(default_factory=dict)
    # HEAD of the session framework root BEFORE the accepted round mutated it:
    # the tree the kept patches apply to. Read pre-mutation, never after a KEEP
    # commit, or the recorded sha would already contain the patches.
    base_sha: str = ""
    # {root: sha} read BEFORE the stack's first mutation of each root, and
    # never replaced. ``base_sha`` above is the one this round's KEEP reports;
    # this is the map that carries the FIRST mutating round's reading forward.
    # An ADVANCED round commits and stacks a patch while recording no per-root
    # identity at all, so without this the KEEP that finally reports a base
    # reads a HEAD that already contains every advanced round's patch -- and the
    # recipe still replays those patches on top of it.
    base_sha_by_root: dict = field(default_factory=dict)
    # Per-root snapshot manifests captured at the enablement KEEP.
    source_snapshots: list = field(default_factory=list)
    # {root_id: {rel: op}} the accepted stack declares, checked against what each
    # snapshot actually captured.
    accepted_stack_targets: dict = field(default_factory=dict)
    # {patch_path: {rel: op}} each kept patch declares, as its own diff headers
    # state it. The recipe emits one patch step per kept patch, and this is the
    # only record of what any one of them touches: without it the decision can
    # check that *some* targets were captured but not that *this step's* were,
    # which is how a recipe covering the final round alone reads as complete.
    patch_targets: dict = field(default_factory=dict)
    # Which branch produced accepted_config: a booted kept bench, or an advanced
    # round's proposal merge, which is by construction never booted.
    accepted_config_source: str = ""
    # Persisted projection of the graded launch evidence; raw env values and
    # host-internal paths are removed before the result reaches durable state.
    launch_evidence: dict = field(default_factory=dict)
    launch_argv_refused: bool = False
    # Version assertions observed AT the KEEP, after every mutation that reaches
    # the launched image.
    installed_versions_at_keep: dict = field(default_factory=dict)
    # Compiled extensions the linked build produced for the framework package
    # that the framework root does not carry. A build installs nothing itself:
    # its outputs travel only as artifacts a specialist declares one by one.
    build_extensions_not_carried: list = field(default_factory=list)
    # {interpreter_tag, distributions} of the accepted runtime.
    environment_closure: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnablementRound":
        """Construct from a raw mapping; unknown keys dropped, missing keys default."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)
