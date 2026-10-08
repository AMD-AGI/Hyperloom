# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EnablementRound: per-round enablement state, nested in SharedState."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

#: Identity and payload of the accepted stack, accumulated across the rounds
#: that contributed to it: a round that contributes none leaves the standing
#: records alone.
_KEEP_STACK_FIELDS = ("roots", "patch_roots", "base_sha", "source_snapshots")

#: Declared targets and observations of this KEEP replace the previous KEEP's
#: records, including when a target set is empty or a probe could not run.
_KEEP_OBSERVED_FIELDS = (
    "accepted_stack_targets",
    "patch_targets",
    "launch_evidence",
    "environment_closure",
    "installed_versions_at_keep",
)

#: Observed fields whose value is meaningful beyond truthiness, so the
#: truthy-or-empty coercion the others get would destroy them. ``None`` here
#: says the observation could not be made, which the sufficiency rules read as
#: a reason to refuse -- collapsing it into an empty mapping would read as a
#: clean scan and certify exactly what the observation exists to withhold.
_KEEP_TRISTATE_FIELDS = ("build_extensions_not_carried", "levers_without_readers")


def _first_writer_wins(bound: Mapping[str, str], incoming: Mapping[str, str]) -> dict[str, str]:
    """``bound`` plus each binding of ``incoming`` it lacks; a recorded binding is never replaced."""
    return {**bound, **{k: v for k, v in incoming.items() if k not in bound}}


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
    # Accepted env levers in the framework's own namespace that nothing in the
    # framework tree reads. A lever is accepted because a round advanced, not
    # because a reader was shown to exist.
    levers_without_readers: list = field(default_factory=list)
    # {interpreter_tag, distributions} of the accepted runtime.
    environment_closure: dict = field(default_factory=dict)

    def record_base_sha(self, root: str, sha: str) -> bool:
        """Record ``root``'s pre-mutation head, first writer wins; return whether this call recorded it."""
        before = self.base_sha_by_root
        self.base_sha_by_root = _first_writer_wins(before, {root: sha})
        return len(self.base_sha_by_root) > len(before)

    def record_patch_roots(self, roots: Mapping[str, str]) -> None:
        """Bind each patch to the tree it applied to, first writer wins.

        An ADVANCED round never reaches the KEEP capture that writes the durable
        ``patch_roots``, so a stack whose rounds used different trees would leave
        every advanced patch to be re-bound to the FINAL round's framework root.
        The round that applied a patch is the one that knows which tree it
        applied to.
        """
        self.patch_roots = _first_writer_wins(self.patch_roots, roots)

    def stack_setup_commands(self, commands: Iterable[str]) -> None:
        """Append the commands not yet on the stack, in the order they ran."""
        self.setup_commands = list(dict.fromkeys([*self.setup_commands, *commands]))

    def push_kept_round(self, *, task_id: str, patches: Iterable[str], artifacts: Iterable[dict]) -> None:
        """Append one accepted round and re-derive the flat projections from every round.

        Artifacts dedupe last-wins per target so a later round supersedes an
        earlier fix to the same file. The round's ``task_id`` is stored with it
        because the replay script sources each round's patches from the archive
        directory that id names; a row without one contributes nothing to the
        script, silently.
        """
        self.kept_rounds = [
            *self.kept_rounds,
            {"task_id": task_id, "patches": list(patches), "artifacts": [dict(a) for a in artifacts]},
        ]
        self.kept_patches = list(dict.fromkeys(p for rnd in self.kept_rounds for p in rnd["patches"]))
        self.kept_artifacts = list({a["target"]: a for rnd in self.kept_rounds for a in rnd["artifacts"]}.values())

    def record_keep(self, result: Mapping[str, Any]) -> None:
        """Persist a KEEP's per-root identity, payload and assertions.

        ``result`` is the integrate result, which reports each record as
        ``enablement_<field>``.
        """
        for name in _KEEP_STACK_FIELDS:
            value = result.get(f"enablement_{name}")
            if value:
                setattr(self, name, value)
        for name in _KEEP_OBSERVED_FIELDS:
            setattr(self, name, result.get(f"enablement_{name}") or {})
        for name in _KEEP_TRISTATE_FIELDS:
            key = f"enablement_{name}"
            if key in result:
                setattr(self, name, result[key])
        self.launch_argv_refused = bool(result.get("enablement_launch_argv_refused"))

    def inherited_base_shas(self) -> dict[str, str]:
        """Return the base sha an earlier accepted round already named, per root.

        Each KEEP is committed, so this round's pre-mutation HEAD already contains
        its predecessors: recording it would name a tree the recipe's own earlier
        patch steps have already been applied to, and a replay would apply them
        again on top of their own result. The first accepted round's reading is the
        one that names the tree the whole stack applies to.
        """
        # The roots records only exist from the first KEEP onward; the durable map
        # is written by every round that mutates a tree, advanced ones included, so
        # it is the one that survives an ADVANCED -> KEEP sequence.
        return _first_writer_wins(
            self.base_sha_by_root,
            {record["path"]: record["base_sha"] for record in self.roots if record["base_sha"]},
        )

    def accepted_stack(self, applied: Iterable[Any]) -> list[str]:
        """Every kept patch, then this round's ``applied``: the order the stack was built and is replayed in."""
        return [*self.kept_patches, *(str(p) for p in applied)]

    def accepted_patch_roots(
        self,
        *,
        done_payload: Mapping[str, Any] | None,
        applied: Iterable[Any],
        framework_root: str,
    ) -> dict[str, str]:
        """Map every patch in the accepted stack, plus this round's ``applied``, to the tree it applies against.

        The stack is cumulative and the recipe emits one patch step per entry in
        ``kept_patches``, so an entry an earlier round bound has to keep its root
        here: a step whose root resolves to no record is refused as
        ``root_unidentified``, and one silently re-pointed at this round's root
        would be captured against a tree that never held it.

        The ``done_payload`` contribution is admitted only for patches that ARE in
        the accepted stack, on the same rule ``_sole_patch_root`` applies to the
        selected set: a recorded entry for a patch this integration did not take
        cannot attest anything about the stack. Admitting it would add a root record
        and a set of declared targets for a tree nothing in the stack touched, and
        the capture would then be judged against files no round wrote.
        """
        accepted = self.accepted_stack(applied)
        in_stack = set(accepted)
        recorded = {
            str(k): str(v)
            for k, v in ((done_payload or {}).get("patch_roots") or {}).items()
            if str(k) and str(v) and str(k) in in_stack
        }
        roots = _first_writer_wins(self.patch_roots, recorded)
        if framework_root:
            # The same fallback the projection uses, so the captured set and the
            # replayed set cannot disagree about which tree a patch belongs to.
            roots = _first_writer_wins(roots, dict.fromkeys(accepted, framework_root))
        return roots

    def last_execution_seq(self) -> int:
        """Return the highest ``seq`` already in the durable setup ledger."""
        return max((row["seq"] for row in self.setup_executions), default=0)

    def append_setup_execution(self, row: dict[str, Any]) -> None:
        """Append one execution row to the append-only ledger."""
        self.setup_executions.append(row)

    def mark_setup_round(self, round_task_id: str, disposition: str, *, accepted: bool) -> None:
        """Record a round's outcome onto its ledger rows, its membership and its presence.

        A round with no task id of its own claims no rows: leaving them
        ``unreported`` states that no lane observed them, which no rule reads
        as verified.

        Presence is per occurrence and succession is a separate fact: an entry *is*
        an entry of the accepted round, while another occurrence of the same digest
        merely reproduces its effect. Keying both off the command string would let a
        failed occurrence be certified by a later successful one.

        Only one round terminates the lane, so an accepted round takes presence away
        from whichever earlier round held it: leaving it standing would let a stale
        row answer for a command the validated launch never ran.
        """
        if not round_task_id:
            return
        rows = [dict(row) for row in self.setup_executions]
        for row in rows:
            own = row["round_task_id"] == round_task_id
            if own:
                row["round_disposition"] = disposition
            if own or accepted:
                # Membership, not effect: a round whose every reached command failed
                # is still the round the lane accepted.
                row["at_accepted_round"] = own and accepted
                row["present_at_final_launch"] = own and accepted and row["outcome"] == "applied"
        if accepted:
            replayed = {row["cmd_digest"] for row in rows if row["present_at_final_launch"]}
            for row in rows:
                row["replayed_at_final_launch"] = row["cmd_digest"] in replayed
        self.setup_executions = rows

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnablementRound":
        """Construct from a raw mapping; unknown keys dropped, missing keys default."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)
