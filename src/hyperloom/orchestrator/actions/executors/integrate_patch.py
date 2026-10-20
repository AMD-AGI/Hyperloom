# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply specialist patches to live framework roots and KEEP or REVERT by benchmark."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config
import asyncio
import csv
import functools
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from collections.abc import Iterable
from typing import Any, NamedTuple

from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.session.session_paths import enablement_stacks_dir
from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
)
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.gpu_types import amd_gpu_dispatch_identity
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.inference_optimizer.framework_paths import (
    framework_apply_tree,
    resolve_framework_tree,
    resolve_kernel_search_roots,
    resolve_session_framework_root,
)
from ...specialists.patch_safety import (
    is_unified_diff,
    patch_escapes_tree,
    patch_targets_missing,
    resolve_patch_apply_root,
)
from ...state.shared_state import (
    inject_stack_base_params,
    resolve_anchor_with_drift,
    resolve_graded_comparison,
)
from hyperloom.orchestrator.lever import LEVER_UPSTREAM_PR
from hyperloom.common.failure_signature import CapabilityGap, FailureSignature
from hyperloom.common.github_urls import repo_slug
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_OUTPUT,
    VERDICT_KEEP,
    constraint_fields,
    holds_within_band,
    constraint_veto_reason,
    incomparability_reason,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
)
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.orchestrator.measurement.runtime_findings import (
    CORRECTNESS_FIX_MAX_DROP_PCT,
    KEEP_REASON_CORRECTNESS_FIX,
    correctness_fix_refusal,
    load_runtime_findings,
)
from ...bringup import load_boot_observation, observation_summary, verdict_of, write_boot_observation
from ...delivery import file_digest
from ...delivery.ledger import append_record, load_prepared_records, load_records, mark_prepared, restore_records
from ..stop_attribution import stopped_by_the_run_class
from ...policy.gate import INTEGRATE_PATCH_PERMISSIVE_VERDICTS
from ._accuracy_gate import (
    accuracy_keep_block,
    enablement_correctness,
    eval_probe_summary,
    framework_run_eval_envs,
    grade_accuracy,
    is_eval_origin,
    parse_eval_results,
    read_eval_probe,
)
from ._apply_feedback import ApplyFeedback, build_apply_feedback
from ._enablement_keep_capture import capture_keep_records
from ._framework_kb_record import KbLesson, write_framework_lesson
from ._git import _git_head_sha, _run_git_cp
from ._integrate_attempt import IntegrateAttempt
from ._integrate_stash import (
    _git_reset_clean,
    _git_restore_stash_if_needed,
    _git_restore_to_head,
    _git_stash_if_dirty,
)
from ._setup_replay import replay_setup_commands
from .source_tree_identity import new_recovery_record
from ._patch_source_pr import (
    DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    _candidate_slug,
    materialize_candidate_patches,
)
from ._nogit_patch import (
    _P_LEVELS,
    _apply_patch_no_git,
    _is_git_tree,
    _is_within,
)
from ...enablement.lane import bind_integrate_to_lane
from ...enablement.recipe.credentials import detect_credential_channels
from ._patch_snapshot import _git_commit_kept, _patch_touched_paths
from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
from ._grid_runner import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    GridVariant,
    SessionDirField,
    VariantResult,
    _num_gpus_for_config,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from . import _framework_switch_manifest as _switch_manifest
from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    dedupe_extra_server_args,
    merge_server_args,
    tokenize_server_args_preserving_json,
)
from ._grid_variant_filter import (
    apply_aiter_moe_pin_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    resolve_skip_spec,
)
from ._server_argv import ConfigUnreadable
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
)


log = logging.getLogger(__name__)


# Two-sided band, in percent of the pre-patch base, that a switch-off parity leg
# must land inside. The rewrite workloads this gates measure with a run-to-run
# spread well under 1%, so a band this wide clears noise by a comfortable margin
# while still catching a patch that is not actually inert when disabled.
DEFAULT_SWITCH_OFF_PARITY_BAND_PCT = 2.0


#: Where the patches this action applies came from. Every source lands through
#: the same apply / vet / bench / KEEP-REVERT pipeline below; they differ only
#: in how the diff is obtained and, for ``upstream_pr``, in which gate admitted
#: it (see :meth:`IntegratePatchExecutor._stage_resolve`).
PATCH_SOURCE_SPECIALIST = "specialist_authored"
PATCH_SOURCE_UPSTREAM_PR = "upstream_pr"
PATCH_SOURCES = (PATCH_SOURCE_SPECIALIST, PATCH_SOURCE_UPSTREAM_PR)

#: Verdicts that accept the candidate, so the attempt leaves it in the tree.
KEEP_STATUSES: frozenset[str] = frozenset({"kept", "advanced", "kept_inert"})

#: Set on every result the bench-and-gate stage returned, whatever its verdict:
#: that attempt reached the bench, and no result an earlier stage returned did.
RESULT_BENCHED = "benched"


def resolve_patch_source(params: Mapping[str, Any]) -> str:
    """Return the declared patch source, defaulting to the specialist lane.

    Args:
        params: Task params.

    Returns:
        One of :data:`PATCH_SOURCES`.
    """
    declared = str(params.get("patch_source") or "").strip().lower()
    return declared if declared in PATCH_SOURCES else PATCH_SOURCE_SPECIALIST


_LAUNCH_ONLY_MUTATION_FIELDS: tuple[str, ...] = (
    "patches",
    "artifacts",
    "config_changes",
    "enablement_setup_commands",
)


def _acquiring_round(params: dict[str, Any]) -> bool:
    """True for an enablement round, except a launch-only bench.

    A launch-only bench carries a pre-built ``runtime_override`` and may
    acquire nothing of its own.
    """
    return bool(params.get("enablement")) and not params.get("enablement_launch_only")


def _enablement_gap(params: dict[str, Any]) -> CapabilityGap | None:
    """The round's capability gap, when it calls for code the stack lacks.

    Read from the verdict the round was dispatched on; None for a round that
    acquires nothing and for a gap no code closes.
    """
    if not _acquiring_round(params):
        return None
    raw = params.get("enablement_failure_signature")
    if not isinstance(raw, dict) or not raw:
        return None
    gap = CapabilityGap.from_signature(FailureSignature.from_dict(raw))
    return gap if gap.requires_code_acquisition else None


def _first_ref_in_repo(refs: Iterable[Any], repo_url: str) -> str:
    """The best-ranked candidate ref that points at ``repo_url``.

    Discovery ranks bridge-repo PRs (aiter, HIP, ROCm) alongside the
    framework's own, and a diff is only applicable to the tree it was cut from.
    A bare ``"PR:1234"`` names no repo and is the framework's by construction.
    """
    try:
        want = repo_slug(repo_url)
    except ValueError:
        return ""
    for raw in refs:
        ref = str(raw).strip()
        if not ref:
            continue
        try:
            if repo_slug(ref) == want:
                return ref
        except ValueError:
            return ref
    return ""


def _established_enablement_config(params: dict[str, Any], shared_state: Any) -> tuple[str, dict[str, str]]:
    """What earlier enablement rounds established, for a round that restated nothing.

    ``enablement.lane._rearm_on_advanced`` accumulates every advance into
    ``state.enablement.accepted_config`` so a later kept round replays them --
    but the accumulation only ever reached the emitted recipe. A launch that
    does not read it back walks into a wall an earlier round already cleared:
    observed live, a build launch probe booted with an empty ``EXTRA_VLLM_ARGS``
    and died on ``AssertionError: DeepseekV4 only supports fp8 kv-cache format
    for now, got auto`` while ``accepted_config`` held ``--kv-cache-dtype fp8``.
    Every dispatcher that opens an enablement integrate_patch crosses this
    executor, so the inheritance lives here rather than in each emitter.

    Optimization rounds inherit nothing -- the rule is the enablement lane's.
    Returns ``("", {})`` when the round is not enablement or when nothing has
    been established yet.
    """
    if not bool(params.get("enablement")):
        return "", {}
    established = shared_state.enablement.accepted_config
    if not established:
        return "", {}
    args = str(established.get("extra_server_args") or "").strip()
    raw_envs = established.get("extra_envs")
    envs = {str(k): str(v) for k, v in raw_envs.items()} if isinstance(raw_envs, dict) else {}
    envs, dropped = filter_untrusted_env_mapping(envs, allow_predicate=is_allowed_variant_env_key)
    if dropped:
        log.warning(
            "integrate_patch: dropping unsafe inherited enablement env key(s): %s",
            ", ".join(sorted(dropped)),
        )
    return args, envs


def _merge_established_server_args(inherited_args: str, round_args: str) -> str:
    """Inherited first, this round last, deduped -- the shape the bridge uses."""
    if not inherited_args:
        return round_args
    if not round_args:
        return inherited_args
    merged = merge_server_args(inherited_args, round_args)
    if tokenize_server_args_preserving_json(merged) is not None:
        return dedupe_extra_server_args(merged)
    # The combined string carries a quoted value with embedded whitespace, which
    # the deduper cannot parse; it would hand back the concatenation with two
    # copies of every inherited flag, and a duplicate is what the server
    # hard-errors on. Provenance cannot decide this either: only the autosubmit
    # bridge pre-merges the inheritance, while this executor also serves rounds
    # opened by the build probe, the framework-config bridge, an authored
    # proposal and a re-queued row, any of which may carry args of its own. So
    # add back only the inherited flags this round does not already name,
    # decided on the inherited side, which comes from ``accepted_config`` and is
    # already deduped.
    try:
        round_tokens = shlex.split(round_args)
    except ValueError:
        # Unbalanced quoting: fall back to whitespace splitting, which can only
        # over-report option names, never miss one, so the comparison below
        # stays conservative about adding a flag back.
        round_tokens = round_args.split()
    # Exact option names, never a substring test: ``--foo`` is not satisfied by
    # ``--foo-bar``, and a name appearing inside a quoted value is not an option
    # at all once shlex has stripped the quotes.
    round_names = {token.split("=", 1)[0] for token in round_tokens if token.startswith("-")}
    try:
        # Quote-aware on this side too: ``accepted_config`` may itself hold a
        # value carrying whitespace, and splitting it on spaces would keep only
        # the first word of it and emit a malformed option.
        inherited_tokens = shlex.split(inherited_args)
    except ValueError:
        inherited_tokens = inherited_args.split()
    missing: list[str] = []
    index = 0
    while index < len(inherited_tokens):
        token = inherited_tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        name = token.split("=", 1)[0]
        takes_value = (
            "=" not in token and index + 1 < len(inherited_tokens) and not inherited_tokens[index + 1].startswith("-")
        )
        if name not in round_names:
            # shlex stripped the quoting when it parsed; restore a spelling that
            # survives the next parse rather than re-emitting a bare value.
            missing.append(shlex.quote(token))
            if takes_value:
                missing.append(shlex.quote(inherited_tokens[index + 1]))
        index += 2 if takes_value else 1
    if not missing:
        return round_args
    return " ".join([*missing, round_args]).strip()


def _done_field(done_payload: dict[str, Any] | None, key: str) -> Any:
    """Read ``key`` from a specialist done payload, top level or nested under ``payload``."""
    if not isinstance(done_payload, dict):
        return None
    if done_payload.get(key):
        return done_payload[key]
    inner = done_payload.get("payload")
    return inner.get(key) if isinstance(inner, dict) else None


def _parse_framework_switches(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the framework-rewrite switch manifest for this integration.

    Looks in the task params first (an explicit dispatch), then in the
    specialist's done payload (the normal authoring path, possibly nested under
    ``payload``).

    Args:
        params: The integrate_patch task params.
        done_payload: The originating specialist's done payload, if any.

    Returns:
        ``(switches, problems)`` from :func:`_switch_manifest.parse_manifest`;
        ``([], [])`` when no manifest was delivered.
    """
    raw = params.get(_switch_manifest.MANIFEST_KEY) or _done_field(done_payload, _switch_manifest.MANIFEST_KEY)
    if not raw:
        return [], []
    # Env the benchmark already defines is reserved: a "switch" colliding with it
    # would be toggled by unrelated configuration rather than by the lever.
    reserved: set[str] = set()
    for source in (params.get("base_extra_envs"), params.get("extra_envs")):
        if isinstance(source, dict):
            reserved.update(str(k).strip().upper() for k in source)
    return _switch_manifest.parse_manifest(raw, reserved_env=reserved)


_now_iso = functools.partial(now_iso, "auto")


def _candidate_mutation_roots(*, params: dict[str, Any], done_payload: dict[str, Any] | None) -> list[str]:
    """Return every tree this round could mutate, before it mutates any of them.

    Read from the payload alone -- the root the round will apply into, the roots
    the authoring stage bound per patch, and the root each artifact target
    resolves into -- because the resolvers that return them run after the setup
    commands have already installed into those same trees.

    A declared ``framework_source_root`` is resolved here too: it is the tree the
    patches land in whenever it differs from the session's own, and the round
    does not otherwise name it until the stash, by which point setup has run.
    """
    explicit = str(params.get("framework_source_root") or "").strip() or None
    # No patch input, so this returns the declared root when one resolves and
    # the session root otherwise -- the pair the apply itself chooses between.
    effective = _resolve_framework_root(explicit, patch_paths=[])
    roots: list[str] = [str(resolve_session_framework_root() or ""), str(effective or "")]
    roots.extend(str(v) for v in ((done_payload or {}).get("patch_roots") or {}).values())
    entries = params.get("artifacts")
    if not isinstance(entries, list) or not entries:
        entries = (done_payload or {}).get("artifacts_written") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resolved = _resolve_artifact_target(str(entry.get("target") or "").strip())
        if resolved is not None:
            roots.append(str(resolved[2]))
    return list(dict.fromkeys(r for r in roots if r))


def _note_pre_mutation_head(
    attempt: IntegrateAttempt,
    root: str | Path | None,
    *,
    enablement: bool = False,
    session_dir: Path | None = None,
) -> None:
    """Record ``root``'s HEAD once, before the first mutation of it.

    Never replaced: a later read would name a tree that has already changed, and
    every patch the recipe carries would replay onto its own result.

    For an enablement round the scope of "once" is the STACK, not the round. A
    KEEP is committed, and an ADVANCED round commits and stacks its patch while
    recording no per-root identity of its own, so by the next round HEAD already
    contains every earlier round's work. Reading it again would name a tree the
    recipe's own earlier patch steps have already been applied to, and a
    consumer replaying that would apply them a second time. The durable map is
    therefore consulted before git and written first-writer-wins -- which also
    carries the reading across a resume, where the per-round map does not
    survive at all.

    Args:
        attempt: The integration owning the per-round map and shared-state reference.
        root: The tree about to be mutated.
        enablement: Whether this round belongs to the enablement stack. An
            ordinary patch round must NOT seed the enablement base: the head
            before an unrelated patch is not the tree the enablement stack
            applies to.
    """
    heads: dict[str, str] = attempt.base_sha_by_root
    key = str(root or "")
    if not key:
        return
    durable = attempt.shared_state.enablement.base_sha_by_root if enablement else {}
    if key not in heads:
        heads[key] = durable.get(key) or _git_head_sha(Path(key))
    if enablement and heads[key]:
        _persist_base_sha_for_root(attempt, key, heads[key], session_dir=session_dir)


def _persist_base_sha_for_root(
    attempt: IntegrateAttempt, root: str, sha: str, *, session_dir: Path | None = None
) -> None:
    """Record this root's pre-mutation head on the durable stack, once.

    Saved here rather than left for the rearm: the reading is only correct
    BEFORE the mutation, and the mutation is the next thing that happens. A
    round that commits its patch and then dies would otherwise resume with the
    entry absent and HEAD already moved, which is exactly the state this map
    exists to prevent.
    """
    shared_state = attempt.shared_state
    if not shared_state.enablement.record_base_sha(root, sha) or session_dir is None:
        return
    try:
        shared_state.save(session_dir)
    except (OSError, AttributeError):
        # The value is on the in-memory state either way, and the rearm saves
        # again; a failed write must not stop the round.
        log.debug("integrate_patch: save after base-sha record failed", exc_info=True)


def resolved_explicit_root(explicit: str) -> Path | None:
    """Resolve a declared framework root, or ``None`` when it is not a directory.

    Args:
        explicit: The declared ``framework_source_root``.

    Returns:
        The resolved directory, or ``None`` when it is unreadable or absent.
    """
    try:
        resolved = Path(explicit).resolve()
    except (OSError, RuntimeError):
        log.warning(
            "integrate_patch: framework_source_root override %r could not be resolved",
            explicit,
        )
        return None
    if not resolved.is_dir():
        log.warning(
            "integrate_patch: framework_source_root override %r does not exist",
            explicit,
        )
        return None
    return resolved


def _read_patch_texts(patch_paths: list[Path] | None) -> list[str]:
    """Return the diff text of every patch that could be read.

    Args:
        patch_paths: Patch files to read.

    Returns:
        The texts that were readable; a shorter list than ``patch_paths`` means
        some patch is missing or unreadable.
    """
    texts: list[str] = []
    for patch in patch_paths or []:
        try:
            texts.append(patch.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


def _sole_patch_root(
    done_payload: dict[str, Any] | None,
    patch_paths: list[Path],
    *,
    specialist_workspace: Path,
) -> str | None:
    """Return the one apply root recorded for every patch, or ``None``.

    Only metadata covering the selected patch set can replace target
    resolution. Unselected patches cannot supply or contradict its root.
    Localization patches outside the specialist workspace deliberately use
    full-set content resolution: the specialist cannot attest their paths.

    Args:
        done_payload: The originating specialist's done payload, if any.
        patch_paths: The complete patch set selected for this integration.
        specialist_workspace: Workspace used to resolve recorded patch paths.

    Returns:
        The sole recorded root, or ``None``.
    """
    raw = (done_payload or {}).get("patch_roots")
    if not isinstance(raw, dict) or not patch_paths:
        return None
    try:
        selected = {patch.resolve() for patch in patch_paths}
    except (OSError, RuntimeError, ValueError):
        return None
    covered: set[Path] = set()
    roots: set[str] = set()
    for recorded_patch, root in raw.items():
        if not isinstance(recorded_patch, str) or not recorded_patch.strip():
            continue
        try:
            resolved = _resolve_patch_paths(
                specialist_workspace=specialist_workspace,
                explicit_patches=[recorded_patch],
                done_payload=None,
            )
        except (OSError, RuntimeError, ValueError):
            return None
        for patch in resolved:
            if patch not in selected:
                continue
            if not isinstance(root, str) or not root.strip():
                return None
            covered.add(patch)
            try:
                root_path = Path(root).expanduser()
                if not root_path.is_absolute():
                    root_path = specialist_workspace / root_path
                roots.add(str(root_path.resolve()))
            except (OSError, RuntimeError, ValueError):
                return None
    return roots.pop() if covered == selected and len(roots) == 1 else None


def _edited_framework_tree() -> Path | None:
    """The tree the session's framework is edited at: the session's root, else its checkout or pip install root."""
    session_root = resolve_session_framework_root()
    if session_root and Path(session_root).is_dir():
        return Path(session_root)
    framework = os.environ.get("FRAMEWORK", "")
    if not framework:
        return None
    tree = framework_apply_tree(resolve_framework_tree(framework))
    return tree.root if tree is not None else None


def _resolve_framework_root(
    explicit: str | None,
    patch_paths: list[Path] | None = None,
    patch_texts: list[str] | None = None,
    recorded_root: str | None = None,
) -> Path | None:
    """Pick one unambiguous framework root under the shared Patch rules.

    A ``recorded_root`` — carried from the authoring stage through
    ``done_payload["patch_roots"]`` — is authoritative and skips probing
    entirely.

    Without a recorded root, the decision falls through to
    :func:`~...specialists.patch_safety.resolve_patch_apply_root`. Without any
    patches to place, the declared root wins, then the session's root, then
    the root the session's framework tree is edited at: its checkout, or the
    install root of a pip-installed package.

    Args:
        explicit: Declared framework root.
        patch_paths: Patch files to place; unreadable ones are skipped.
        patch_texts: Patch diffs already in memory, placed alongside
            ``patch_paths``.
        recorded_root: The apply root recorded at authoring time, if any.

    Returns:
        The resolved root, or ``None`` when the patches name no single tree.
    """
    if recorded_root:
        return resolved_explicit_root(recorded_root)

    explicit_path: Path | None = None
    if explicit:
        explicit_path = resolved_explicit_root(explicit)
        if explicit_path is None:
            return None

    texts = [str(text) for text in (patch_texts or []) if str(text).strip()]
    texts.extend(_read_patch_texts(patch_paths))
    has_patch_input = bool(patch_paths or patch_texts)
    if has_patch_input:
        session_root = resolve_session_framework_root()
        # The search roots do not necessarily hold it: they discover the
        # unprefixed env var, while the session root also answers to
        # <FRAMEWORK>_REPO_PATH and <FRAMEWORK>_DIR. Leaving it out turns the
        # tree under optimisation into a non-candidate, and default_root cannot
        # stand in -- that is consulted only for a create-only set, which has no
        # pre-image to match.
        roots = [Path(root) for root in resolve_kernel_search_roots()]
        candidates = [Path(session_root), *roots] if session_root else roots
        resolution = resolve_patch_apply_root(
            texts,
            explicit_root=explicit_path,
            candidate_roots=tuple(candidates),
            default_root=Path(session_root) if session_root else None,
        )
        if resolution.root is None:
            log.warning(
                "integrate_patch: Patch root resolution rejected: %s%s",
                resolution.reason,
                (f" matches={[str(root) for root in resolution.matches]!r}" if resolution.matches else ""),
            )
        return resolution.root

    return explicit_path if explicit_path is not None else _edited_framework_tree()


def _run_git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    p_level: int,
    check_only: bool,
) -> tuple[bool, str]:
    """Single ``git apply`` invocation at an explicit strip level.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        p_level: The ``-p<N>`` strip level.
        check_only: Whether to pass ``--check`` (dry run, no mutation).

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True on a zero return code.
    """
    args = ["-C", str(framework_root), "apply", f"-p{p_level}"]
    if check_only:
        args.append("--check")
    args.append(str(patch_path))
    cp = _run_git_cp(args, timeout=120.0)
    if cp is None:
        return False, "git apply spawn failed"
    return cp.returncode == 0, cp.stderr.strip()


def _derive_lane(params: dict[str, Any]) -> str:
    """Derive the retry lane name from integrate_patch params.

    Returns:
        ``"enablement"``, ``"perf_framework"``, or ``"perf_explore"``.
    """
    if params.get("enablement"):
        return "enablement"
    if params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"):
        return "perf_framework"
    return "perf_explore"


def _accuracy_delta_pct(measured: Any, baseline: Any) -> float | None:
    """Percent accuracy change of ``measured`` against ``baseline``.

    Returns ``None`` when either side is missing or the baseline is not
    positive, so callers fall back to their existing value.
    """
    try:
        m = float(measured)
        b = float(baseline)
    except (TypeError, ValueError):
        return None
    if b <= 0.0:
        return None
    return (m - b) / b * 100.0


def _measured_against(params: Mapping[str, Any], *, base_tput: float) -> dict[str, Any]:
    """The configuration this patch was measured on top of.

    Only the executor can answer this. The task's own params hold the stack as
    of dispatch, and a task queued before another KEEP landed is rebound onto
    the live stack top before it runs; ``base_tput`` is the drift-resolved
    anchor it was actually graded against. Carried on the result for the reason
    the configuration arm carries it -- what the session serves once this
    returns is no longer what the patch was judged against, so a reader that
    reconstructs the stack from the current config gets the wrong one.
    """
    return {
        "throughput": base_tput or None,
        "accuracy": float(params.get("accuracy_baseline") or 0.0) or None,
        "extra_server_args": str(params.get("base_extra_args") or "").strip(),
        "extra_envs": dict(params.get("base_extra_envs") or {}),
        "remove_args": to_str_list(params.get("base_remove_args")),
        "unset_envs": to_str_list(params.get("base_unset_envs")),
        "args_mode": str(params.get("base_args_mode") or "append"),
    }


def _preflight_missing_targets(
    framework_root: Path,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Return per-patch records for patches whose modify/delete targets are
    absent from ``framework_root`` at every ``-p`` strip level.

    A hallucinated-layout patch (e.g. modifying a CUDA-only file on a ROCm
    build) can never apply; flagging it here yields an actionable advisory
    instead of an opaque ``git_apply_failed`` after a wasted apply attempt.
    Patches supplied directly via ``params.patches`` bypass the
    authoring-time ``specialists.patch_safety`` vetting gate
    (:func:`vet_patches`), so they are checked here.

    Args:
        framework_root: The git checkout the patches target.
        patch_paths: The patch files to preflight.

    Returns:
        A list of per-patch records (``patch`` + ``missing_targets``) for
        patches whose targets are absent at every strip level.
    """
    records: list[dict[str, Any]] = []
    for patch in patch_paths:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        missing = patch_targets_missing(text, framework_root)
        if missing:
            records.append({"patch": str(patch), "missing_targets": missing})
    return records


def _detect_p_level(framework_root: Path, patch_path: Path) -> int | None:
    """Return the first ``-p`` level whose ``--check`` applies cleanly.

    Args:
        framework_root: The git checkout to test against.
        patch_path: The patch file to probe.

    Returns:
        The first ``-p<N>`` level that applies cleanly, or ``None`` when none
        do.
    """
    for lvl in _P_LEVELS:
        ok, _ = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=True)
        if ok:
            return lvl
    return None


def _git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    check_only: bool = False,
) -> tuple[bool, str]:
    """Run ``git apply -p<auto> [--check] <patch>`` inside
    ``framework_root``, auto-detecting the strip level. Returns
    ``(ok, stderr)``.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        check_only: Whether to only check (dry run) rather than mutate.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the apply (or check)
        succeeds.
    """
    lvl = _detect_p_level(framework_root, patch_path)
    if lvl is None:
        # Surface a representative error at the git-native default level.
        return _run_git_apply(framework_root, patch_path, p_level=1, check_only=check_only)
    if check_only:
        return True, ""
    return _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=False)


def _git_reverse_applies_cleanly(framework_root: Path, patch_path: Path) -> bool:
    """True when ``patch_path`` is already fully applied in ``framework_root``.

    The git-channel twin of
    :func:`._nogit_patch._reverse_applies_cleanly`. ``git apply -R --check``
    succeeds only when every hunk's *post*-state is already present, i.e. the
    tree already equals what a forward apply would produce. Read-only:
    ``--check`` never mutates the tree.

    Args:
        framework_root: The git checkout the patch targets.
        patch_path: The patch file to probe.

    Returns:
        ``True`` when some strip level reverse-checks cleanly.
    """
    from ._nogit_patch import _P_LEVELS

    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is not None and cp.returncode == 0:
            return True
    return False


def _git_apply_collect_feedback(
    framework_root: Path,
    patch_path: Path,
) -> "tuple[bool, str, ApplyFeedback | None]":
    """Like :func:`_git_apply` but also returns an :class:`ApplyFeedback` on failure.

    On success returns ``(True, "", None)``.  On failure returns
    ``(False, stderr, ApplyFeedback)`` where *ApplyFeedback* carries the
    stderr of every tried ``-p`` level and a source-context snippet.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.

    Returns:
        ``(ok, err, feedback)`` — feedback is ``None`` on success.
    """
    from ._nogit_patch import _P_LEVELS

    # Collect per-level check stderr for the feedback record.
    tried_levels: list[int] = []
    level_stderrs: list[str] = []
    for lvl in _P_LEVELS:
        ok_check, stderr_check = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=True)
        tried_levels.append(lvl)
        if stderr_check:
            level_stderrs.append(f"-p{lvl}: {stderr_check}")
        if ok_check:
            # Level works; now apply for real.
            ok_apply, stderr_apply = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=False)
            if ok_apply:
                return True, "", None
            feedback = build_apply_feedback(
                patch_path,
                channel="git",
                tried_levels=tried_levels,
                stderr=stderr_apply,
                framework_root=framework_root,
            )
            return False, stderr_apply, feedback

    # A subset of a patch already applied is a no-op that ``--check`` still
    # rejects; a clean reverse check proves every hunk is in the tree.
    if _git_reverse_applies_cleanly(framework_root, patch_path):
        log.info(
            "integrate_patch: %s is already fully applied (clean git apply -R --check); treating as a no-op",
            patch_path.name,
        )
        return True, "", None
    all_stderrs = "\n".join(level_stderrs)
    feedback = build_apply_feedback(
        patch_path,
        channel="git",
        tried_levels=tried_levels,
        stderr=all_stderrs,
        framework_root=framework_root,
    )
    return False, all_stderrs, feedback


def _git_apply_reverse(
    framework_root: Path,
    patch_path: Path,
) -> tuple[bool, str]:
    """Reverse-apply ``patch_path`` (``git apply -R -p<auto>``) as the REVERT
    path; caller falls back to ``git checkout`` on failure. Auto-detects the
    same strip level the forward apply used via ``-R --check``.

    Args:
        framework_root: The git checkout to reverse-apply into.
        patch_path: The patch file to reverse-apply.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the reverse apply
        succeeds.
    """
    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is None:
            return False, "git apply -R spawn failed"
        if cp.returncode != 0:
            continue
        cp2 = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", str(patch_path)],
            timeout=120.0,
        )
        if cp2 is None:
            return False, "git apply -R spawn failed"
        if cp2.returncode == 0:
            return True, ""
        return False, cp2.stderr.strip()
    return False, f"git apply -R: no matching -p level for {patch_path}"


class _RecoveryEntry(NamedTuple):
    """A pending integration's recovery record, or why there is nothing to act on."""

    record: dict[str, Any] | None
    refusal: str


class _PatchUndo(NamedTuple):
    """How an attempt's patches come back out: a git base to reset to, or the preimages taken instead."""

    base: str
    records: list[dict[str, Any]]


def _recovery_to_discharge(pending: dict[str, Any]) -> _RecoveryEntry:
    """The complete recovery record of a pending integration, or why it carries no obligation to discharge."""
    recovery = pending.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("version") != 1:
        mutated = pending.get("patches") or pending.get("artifacts") or pending.get("attempt_venv_root")
        return _RecoveryEntry(None, "pending integration has no complete recovery evidence" if mutated else "")
    phase = recovery.get("phase")
    if phase in ("restored", "accepted"):
        return _RecoveryEntry(None, "")
    if phase not in ("ready", "applied", "files_restored"):
        return _RecoveryEntry(
            None,
            "the operator's uncommitted work was merged back on top of the ungraded candidate and "
            "the stash was dropped, so nothing on disk separates the two; taking the candidate back "
            "out would destroy that work, and this refuses rather than guess"
            if phase == "applied_with_restored_stash"
            else f"integration interrupted before stash identity was durable: {phase!r}",
        )
    root = recovery.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        return _RecoveryEntry(None, "attempt-owned recovery root missing")
    return _RecoveryEntry(recovery, "")


def _recorded_stash_ref(framework_root: Path | None, stash_oid: str) -> str:
    """The stash entry holding the operator's pre-candidate work, found by the oid recorded for it."""
    if not stash_oid:
        return ""
    if framework_root is None:
        raise ValueError("stash recovery root missing")
    cp = _run_git_cp(["-C", str(framework_root), "stash", "list", "--format=%H %gd"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        raise OSError("cannot read stash identities")
    refs = [line.split(" ", 1)[1] for line in cp.stdout.splitlines() if line.startswith(stash_oid + " ")]
    if len(refs) != 1:
        raise ValueError("recorded user stash is absent or ambiguous; refusing to guess")
    return refs[0]


def _prepared_artifact_records(
    pending: dict[str, Any],
    recovery: dict[str, Any],
    workspace: Path,
) -> list[dict[str, Any]]:
    """The artifact preimages this attempt committed, proven to cover its whole mutation plan.

    The artifact phase commits its whole plan's preimages and then records a
    witness before it installs anything, so the witness tells "never started"
    (nothing to undo) apart from "started, evidence now gone" (refuse loudly).
    """
    if not recovery.get("artifacts_prepared"):
        return []
    records = load_prepared_records(workspace / "artifact_backups")
    expected = {str(Path(row["target"]).resolve()) for row in pending.get("artifacts", [])}
    recorded = {str(Path(row["target"]).resolve()) for row in records}
    if not expected <= recorded:
        raise ValueError("artifact recovery ledger does not cover the complete mutation plan")
    return records


def _patch_undo_plan(
    pending: dict[str, Any],
    recovery: dict[str, Any],
    *,
    framework_root: Path | None,
    workspace: Path,
) -> _PatchUndo:
    """How this attempt's patches come back out of the tree they were applied to.

    A git attempt takes no backups, so the recorded base is its only way back and
    a HEAD that has moved since means the tree is no longer the one it mutated. A
    non-git attempt writes a preimage record before every mutation it makes, so an
    absent ledger there is proof that no patch reached the tree -- the first patch
    failing its dry-run leaves exactly that state, and refusing would stop the
    session over an untouched tree.
    """
    if not pending.get("patches"):
        return _PatchUndo("", [])
    base = str(recovery.get("git_head") or "")
    if not base and not (framework_root is not None and _is_git_tree(framework_root)):
        return _PatchUndo("", load_prepared_records(workspace / "patch_backups"))
    if not base:
        raise ValueError("git attempt recorded no base to restore to")
    if framework_root is None or _git_head_sha(framework_root) != base:
        raise ValueError("framework HEAD differs from the recorded attempt base")
    return _PatchUndo(base, [])


def _take_candidate_out(
    pending: dict[str, Any],
    recovery: dict[str, Any],
    *,
    framework_root: Path | None,
    workspace: Path,
) -> dict[str, Any]:
    """Undo this attempt's artifact installs and patch applies, reporting what came back out."""
    summary: dict[str, Any] = {"reversed": [], "artifacts_reverted": [], "failed": []}
    try:
        artifact_records = _prepared_artifact_records(pending, recovery, workspace)
        undo = _patch_undo_plan(pending, recovery, framework_root=framework_root, workspace=workspace)
        # Both ledgers are read and the artifacts are undone before the git sweep,
        # which is irreversible: it has to run against a tree that already holds
        # every preimage, and an artifact target git will not touch -- ignored, or
        # outside this root -- has no second chance.
        _, errors = restore_records(artifact_records)
        if errors:
            summary["failed"].extend(errors)
            return summary
        summary["artifacts_reverted"] = [row.get("rel_target") or row["target"] for row in artifact_records]
        if not pending.get("patches"):
            return summary
        if undo.base:
            # Everything still untracked here was created by this attempt: the
            # pre-candidate auto-stash took the operator's untracked files with
            # ``push -u``, and ``clean`` without ``-x`` leaves ignored paths (an
            # ignored artifact target included) where they are. Only this
            # attempt's own recovery data has to survive the sweep.
            exclusions = (
                ["/" + workspace.relative_to(framework_root).as_posix()]
                if framework_root is not None and workspace.is_relative_to(framework_root)
                else []
            )
            ok, error = _git_reset_clean(framework_root, undo.base, exclude=exclusions)
            if not ok:
                raise OSError(error)
        else:
            _, errors = restore_records(undo.records)
            if errors:
                summary["failed"].extend(errors)
                return summary
        summary["reversed"] = list(reversed(pending["patches"]))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        summary["failed"].append(str(exc))
    return summary


def restore_pending_integrate(pending: dict[str, Any], *, keep: bool = False) -> dict[str, Any]:
    """Discharge one recorded integration's file and stash obligations, or retain them."""
    summary: dict[str, Any] = {"reversed": [], "artifacts_reverted": [], "failed": []}
    entry = _recovery_to_discharge(pending)
    if entry.record is None:
        if entry.refusal:
            summary["failed"].append(entry.refusal)
        return summary
    recovery = entry.record
    root_text = str(pending.get("framework_source_root") or "")
    framework_root = Path(root_text) if root_text else None
    try:
        stash_ref = _recorded_stash_ref(framework_root, str(recovery.get("stash_oid") or ""))
        if not keep and recovery.get("phase") != "files_restored":
            summary = _take_candidate_out(
                pending, recovery, framework_root=framework_root, workspace=Path(recovery["root"])
            )
            if summary["failed"]:
                return summary
            recovery["phase"] = "files_restored"
        if stash_ref:
            note = _git_restore_stash_if_needed(framework_root, "stashed", stash_ref)
            if note:
                raise OSError(note)
        recovery["phase"] = "accepted" if keep else "restored"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        summary["failed"].append(str(exc))
    return summary


def _resolve_patch_paths(
    *,
    specialist_workspace: Path,
    explicit_patches: list[str] | None,
    done_payload: dict[str, Any] | None,
) -> list[Path]:
    """Resolve the list of patch files to apply.

    Order: ``params.patches`` → ``specialist_done.patches_written`` →
    filesystem scan of ``specialist_workspace/{worktree/,}patches/``.
    Entries normalised to absolute Paths; missing ones logged + dropped.

    Security: a resolved patch path must live inside the specialist workspace
    (or its worktree); an absolute path pointing outside the sandbox is dropped.
    Both sides are ``resolve()``-d first so a symlinked workspace still matches.

    Args:
        specialist_workspace: The specialist task workspace to resolve
            relative paths / scan for patches.
        explicit_patches: Explicit patch paths from params, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        The resolved, existing patch files as absolute Paths.
    """
    candidates: list[str] = []
    if explicit_patches:
        candidates.extend(str(p) for p in explicit_patches)
    elif done_payload and isinstance(done_payload.get("patches_written"), list):
        candidates.extend(str(p) for p in done_payload["patches_written"] if p)
    else:
        for base in (
            specialist_workspace / "worktree" / "patches",
            specialist_workspace / "patches",
        ):
            if base.is_dir():
                for p in sorted(base.glob("*.patch")):
                    candidates.append(str(p))
                for p in sorted(base.glob("*.diff")):
                    candidates.append(str(p))

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]

    out: list[Path] = []
    for c in candidates:
        p = Path(c)
        # Resolve relative paths against the specialist workspace + worktree.
        if not p.is_absolute():
            for base in (
                specialist_workspace / "worktree",
                specialist_workspace,
            ):
                cand = base / c
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            log.warning(
                "integrate_patch: patch %r not found (specialist_workspace=%s)",
                c,
                specialist_workspace,
            )
            continue
        resolved = p.resolve()
        if not any(_is_within(resolved, root) for root in allowed_roots):
            log.warning(
                "integrate_patch: patch %r resolves outside the specialist workspace (%s); dropping for safety",
                c,
                specialist_workspace,
            )
            continue
        out.append(resolved)
    return out


@dataclass
class _ArtifactSpec:
    """One resolved non-diff tuned artifact to install at integration.

    Attributes:
        source: Absolute path to the artifact file inside the specialist
            workspace / worktree (sandbox-validated).
        target: Absolute install path inside a framework root (no escape).
        rel_target: The framework-relative target, normalized to the matched
            root via ``_resolve_artifact_target`` (an author's absolute target
            is converted to this relative form). Used for reporting AND as the
            framework-relative key for the durable KEEP source snapshot.
        root: The root ``rel_target`` is relative to. The KEEP
            source snapshot is keyed on one root, so an artifact installed into
            a different tree than the patches must be recognisable as such.
        kind: Free-form artifact kind label (e.g. ``config_json``).
        description: Free-form human description.
    """

    source: Path
    target: Path
    rel_target: str
    root: Path
    kind: str = ""
    description: str = ""


def _artifact_candidates(root: Path, rel: str) -> list[Path]:
    """The joins a root admits for a framework-relative artifact target.

    A pip-installed root is the package directory itself, so a target that
    repeats the package name (``vllm/model_executor/...``) belongs under the
    root's parent; a checkout holds the package one level down and joins
    directly. Both are offered and the caller picks by which parent exists,
    the same rule ``_resolve_focus_dir`` applies to prompt focus directories.

    Args:
        root: A framework search root.
        rel: The framework-relative target.

    Returns:
        Candidate absolute paths, direct join first.
    """
    candidates = [(root / rel).resolve()]
    if (root / "__init__.py").is_file():
        candidates.append((root.parent / rel).resolve())
    return candidates


def _resolve_artifact_target(rel_target: str) -> tuple[Path, str, Path] | None:
    """Resolve an artifact target (framework-relative, or absolute) to a path.

    A relative target picks the framework root whose tree already contains the
    target's parent directory; else the first existing root. On a pip-installed
    root both joins in :func:`_artifact_candidates` are tried, so a ``vllm/...``
    config lands beside the package rather than under a doubled ``vllm/vllm/``.
    Either way the resolved path must stay within the chosen root, so a ``..``
    cannot walk out of the tree it names.

    Args:
        rel_target: The install path authored by the specialist (framework-
            relative, or absolute).

    Returns:
        A ``(absolute_target, framework_relative_target, root)`` tuple, or
        ``None`` when nothing resolves safely. Callers MUST persist the relative
        target AND the root so the durable KEEP source snapshot captures the
        installed file even when the author used an absolute path.
    """
    rel = (rel_target or "").strip()
    if not rel or ".." in Path(rel).parts:
        return None
    roots = [Path(r).resolve() for r in resolve_kernel_search_roots()]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    if Path(rel).is_absolute():
        cand = Path(rel).resolve()
        for root in roots:
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix(), root
        return None
    # Prefer a root whose tree already holds the target's parent dir.
    for root in roots:
        for cand in _artifact_candidates(root, rel):
            if _is_within(cand, root) and cand.parent.is_dir():
                return cand, cand.relative_to(root).as_posix(), root
    for root in roots:
        for cand in _artifact_candidates(root, rel):
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix(), root
    return None


def _declared_artifacts(
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> list[Any]:
    """The artifact entries this attempt installs: the params override, else what the specialist wrote."""
    if explicit_artifacts:
        return list(explicit_artifacts)
    written = (done_payload or {}).get("artifacts_written")
    return list(written) if isinstance(written, list) else []


def _resolve_artifact_specs(
    *,
    specialist_workspace: Path,
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> tuple[list[_ArtifactSpec], list[dict[str, str]]]:
    """Resolve non-diff tuned artifacts to install.

    Order: ``params.artifacts`` → ``specialist_done.artifacts_written``. Each
    entry is ``{source, target, kind, description}``: ``source`` is resolved
    inside the specialist workspace/worktree (sandbox) and ``target`` is
    resolved inside a framework root. Malformed / out-of-sandbox entries are
    dropped and reported.

    Args:
        specialist_workspace: The specialist task workspace.
        explicit_artifacts: ``params.artifacts`` override list, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        A ``(specs, errors)`` tuple: resolved specs, plus per-entry error
        records (``{artifact, error}``) for entries that could not be resolved.
    """
    raw = _declared_artifacts(explicit_artifacts, done_payload)
    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]
    specs: list[_ArtifactSpec] = []
    errors: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            errors.append({"artifact": str(entry), "error": "not_a_mapping"})
            continue
        src_rel = str(entry.get("source") or "").strip()
        tgt_rel = str(entry.get("target") or "").strip()
        if not src_rel or not tgt_rel:
            errors.append({"artifact": json.dumps(entry), "error": "missing_source_or_target"})
            continue
        # Resolve source inside the workspace sandbox.
        src = Path(src_rel)
        if not src.is_absolute():
            for base in (specialist_workspace / "worktree", specialist_workspace):
                cand = base / src_rel
                if cand.exists():
                    src = cand
                    break
        if not src.exists() or not src.resolve().is_file():
            errors.append({"artifact": src_rel, "error": "source_not_found"})
            continue
        src_resolved = src.resolve()
        if not any(_is_within(src_resolved, root) for root in allowed_roots):
            errors.append({"artifact": src_rel, "error": "source_outside_workspace"})
            continue
        resolved = _resolve_artifact_target(tgt_rel)
        if resolved is None:
            errors.append({"artifact": tgt_rel, "error": "target_unresolved_or_escapes_root"})
            continue
        target, rel_norm, root = resolved
        specs.append(
            _ArtifactSpec(
                source=src_resolved,
                target=target,
                rel_target=rel_norm,
                root=root,
                kind=str(entry.get("kind") or "").strip(),
                description=str(entry.get("description") or "").strip(),
            )
        )
    return specs, errors


def _is_aiter_gemm_model_config(spec: _ArtifactSpec) -> bool:
    """Return whether an artifact is an AITER runtime GEMM model-config CSV."""
    if spec.source.suffix.lower() != ".csv":
        return False
    kind = spec.kind.strip().lower()
    target = spec.target.as_posix().lower()
    filename = spec.target.name.lower()
    is_aiter_model_config = "/aiter/configs/model_configs/" in target
    return is_aiter_model_config and (kind == "model_config" or "_tuned_gemm" in filename)


def _rows_not_runtime_ready(rows: list[dict[str, Any]]) -> int:
    """How many tuned rows name no kernel the runtime can dispatch.

    A placeholder name or a non-positive runtime is what an untuned seed row
    looks like: the entry exists, but AITER falls back to its own heuristic.
    """
    unusable = 0
    for row in rows:
        kernel_name = str(row.get("kernelName") or "").strip().lower()
        try:
            runtime_us = float(str(row.get("us") or "0").strip())
        except ValueError:
            runtime_us = 0.0
        if not math.isfinite(runtime_us) or runtime_us <= 0.0 or "placeholder" in kernel_name:
            unusable += 1
    return unusable


def _validate_aiter_gemm_artifacts(
    specs: list[_ArtifactSpec],
    *,
    gpu_type: str | None,
) -> list[dict[str, str]]:
    """Reject AITER GEMM CSVs that cannot dispatch on the target GPU.

    A model-config seed containing rows for another architecture, or placeholder
    rows that still require an offline tuning step, has no runtime effect. Such
    an artifact must not enter E2E promotion because benchmark variance could
    otherwise attribute an unrelated gain to it.
    """
    identity = amd_gpu_dispatch_identity(gpu_type)
    if identity is None:
        return []
    expected_gfx, expected_cu_num = identity
    errors: list[dict[str, str]] = []
    required_columns = {"gfx", "cu_num", "us", "kernelName"}

    for spec in specs:
        if not _is_aiter_gemm_model_config(spec):
            continue
        base_error = {
            "artifact": str(spec.source),
            "expected_gfx": expected_gfx,
            "expected_cu_num": str(expected_cu_num),
        }
        try:
            with spec.source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                if not required_columns.issubset(fieldnames):
                    errors.append(
                        {
                            **base_error,
                            "error": "invalid_aiter_gemm_schema",
                        }
                    )
                    continue
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            errors.append({**base_error, "error": "invalid_aiter_gemm_csv"})
            continue

        def _cu_num_matches(row: dict[str, Any]) -> bool:
            raw = str(row.get("cu_num") or "").strip()
            if not raw:
                return False
            try:
                return int(float(raw)) == expected_cu_num
            except ValueError:
                return False

        target_rows = [
            row for row in rows if str(row.get("gfx") or "").strip().lower() == expected_gfx and _cu_num_matches(row)
        ]
        if not target_rows:
            errors.append({**base_error, "error": "no_target_gpu_rows"})
            continue

        invalid_target_rows = _rows_not_runtime_ready(target_rows)
        if invalid_target_rows:
            errors.append(
                {
                    **base_error,
                    "error": "target_gpu_rows_not_runtime_ready",
                    "invalid_rows": str(invalid_target_rows),
                    "target_rows": str(len(target_rows)),
                }
            )
    return errors


def _read_done_payload(workspace: Path) -> dict[str, Any] | None:
    """Read and parse ``specialist_done.json`` from a workspace.

    Args:
        workspace (Path): The specialist task workspace directory.

    Returns:
        dict[str, Any] | None: The parsed payload, or ``None`` when the
        file is absent or cannot be parsed.
    """
    done = workspace / "specialist_done.json"
    if not done.exists():
        return None
    try:
        return json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "integrate_patch: failed to parse %s: %r",
            done,
            exc,
        )
        return None


_FRAMEWORK_KB_PROVENANCE_PREFIX = "specialist:serving:framework"


def _stamp_framework_kb_provenance(
    done_payload: dict[str, Any] | None,
    *,
    params: dict[str, Any],
    shared_state: Any,
) -> None:
    """Ensure a FRAMEWORK-dispatched deliverable carries KB-writeback provenance.

    Stamps the ``specialist:serving:framework...`` provenance prefix (that
    :meth:`IntegratePatchExecutor._find_framework_proposal` requires) from the
    dispatch context, so same-framework deliverables reach ``lessons.jsonl``.

    Mutates ``done_payload["proposal_set"][0]`` in place; no-ops when this
    action was not dispatched from FRAMEWORK authoring, or when a proposal
    already carries a matching provenance.

    Args:
        done_payload: The specialist's parsed ``specialist_done.json`` (may
            be ``None``/malformed; no-ops in that case).
        params: The ``integrate_patch`` action's dispatch params (carries
            ``framework_agent_authoring`` / ``framework_agent_candidate_id``
            when this run came from FRAMEWORK_AGENT).
        shared_state: The run's ``SharedState`` (best-effort ``framework``
            read for the provenance suffix).
    """
    if not isinstance(done_payload, dict):
        return
    if not params.get("framework_agent_authoring"):
        return
    pr_url = str(params.get("framework_agent_candidate_id") or "").strip()
    if not pr_url:
        return
    proposals = done_payload.get("proposal_set")
    if not isinstance(proposals, list) or not proposals or not isinstance(proposals[0], dict):
        # No proposal_set entry to stamp; synthesize a minimal anchor.
        done_payload["proposal_set"] = [{}]
        proposals = done_payload["proposal_set"]
    target = proposals[0]
    existing = str(target.get("provenance") or "")
    if existing.startswith(_FRAMEWORK_KB_PROVENANCE_PREFIX):
        return  # cross-framework (or already-stamped) path already complies
    framework = str(getattr(shared_state, "framework", "") or "").strip().lower()
    target["provenance"] = (
        f"{_FRAMEWORK_KB_PROVENANCE_PREFIX}:{framework}" if framework else _FRAMEWORK_KB_PROVENANCE_PREFIX
    )
    target.setdefault("fa_pr_url", pr_url)
    target.setdefault("framework", framework)


_FIX_ACCURACY_UNAVAILABLE = "accuracy result unavailable"


def _fix_axis(graded: Any) -> str:
    return "interactivity p50" if graded.graded_on_intvty else "throughput"


def _intvty_guard_refusals(shared_state: Any, bench_result: dict[str, Any]) -> list[str]:
    """The interactivity KEEP guards a correctness fix still owes; only the p50 gain is relaxed."""
    ref_perf, _ = resolve_grading_anchor_perf(shared_state)
    cand_perf = perf_snapshot_from_mapping(bench_result)
    _, noise_pct = resolved_grading(shared_state)
    band = f"{noise_pct:.1f}% noise band" if noise_pct is not None else "noise band"
    refusals = [
        f"{key} fell outside the {band}"
        for key in (GRADED_INTVTY, GRADED_OUTPUT)
        if not holds_within_band(cand_perf, ref_perf, key, noise_pct=noise_pct)
    ]
    incomparable = incomparability_reason(cand_perf, ref_perf)
    if incomparable:
        refusals.append(f"rounds are not comparable ({incomparable})")
    return refusals


def _correctness_fix_refusals(
    resolves_finding: str,
    *,
    shared_state: Any,
    bench_result: dict[str, Any],
    graded: Any,
    delta_pct: float | None,
) -> list[str]:
    """Performance and finding reasons a declared correctness fix may not KEEP below the graded-axis threshold."""
    refusals: list[str] = []
    finding_refusal = correctness_fix_refusal(
        load_runtime_findings(getattr(shared_state, "current_best_measurement", None) or {}),
        load_runtime_findings(bench_result),
        resolves_finding,
    )
    if finding_refusal:
        refusals.append(finding_refusal)
    # The resolver only vetoes a KEEP verdict, and this route runs after a REVERT, so the budgets are read here.
    veto = constraint_veto_reason(bench_result, shared_state)
    if not graded.comparable:
        refusals.append(f"performance comparison unavailable: {graded.degrade_reason}")
    elif veto:
        refusals.append(veto)
    elif delta_pct is None:
        refusals.append(f"no measurable {_fix_axis(graded)}")
    elif delta_pct < -CORRECTNESS_FIX_MAX_DROP_PCT:
        refusals.append(
            f"{_fix_axis(graded)} delta {delta_pct:+.2f}% exceeds the "
            f"{CORRECTNESS_FIX_MAX_DROP_PCT:.1f}% correctness-fix allowance"
        )
    if graded.comparable and graded.graded_on_intvty:
        refusals.extend(_intvty_guard_refusals(shared_state, bench_result))
    return refusals


def _fix_accuracy_refusals(accuracy_pass: bool | None) -> list[str]:
    """The accuracy reason a declared correctness fix may not KEEP; empty when its eval passed."""
    if accuracy_pass is None:
        return [_FIX_ACCURACY_UNAVAILABLE]
    if accuracy_pass is not True:
        return ["accuracy eval failed"]
    return []


def _enforce_critic_gate(
    shared_state: Any,
    subject: str,
) -> "dict[str, Any] | None":
    """Enforce a permissive Critic verdict on ``subject`` before any side
    effect; returns a ``rejected_by_critic`` dict on failure, else ``None``
    when no SharedState is available or the verdict is permissive."""
    if shared_state is None:
        return None
    try:
        recorded = shared_state.get_specialist_patch_verdict(subject)
    except AttributeError:
        recorded = ""
    if (recorded or "").lower() not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
        _detail = f"verdict {recorded!r}" if recorded else "no Critic verdict on record"
        # No side effect has occurred yet; reject cleanly (nothing to revert).
        return {
            "status": "rejected_by_critic",
            "specialist_task_id": subject,
            "patches_applied": [],
            "patches_reverted": [],
            "reason": (
                f"integrate_patch requires a permissive Critic verdict "
                f"(approve/advise) for {subject!r}; {_detail}. Refusing to run."
            ),
        }
    return None


def _integration_refusal(params: dict[str, Any], shared_state: Any) -> dict[str, Any] | None:
    """What stops an integration from starting at all, before any side effect, if anything does."""
    from ._multi_node_env import is_multi_node

    if is_multi_node():
        return {
            "status": "skipped",
            "skipped_reason": "multi_node_unsupported",
            "specialist_task_id": str(params.get("specialist_task_id") or "").strip(),
            "patches_applied": [],
            "patches_reverted": [],
            "reason": (
                "specialist integrate_patch is not supported in "
                "multi-node mode (no git-diff pod fan-out); skipped "
                "without applying any patch. Other actions "
                "(baseline/profile/explore/conc_sweep/roofline) continue "
                "normally. Use the kernel-agent integrate path (which "
                "fans out via `multi_node apply-patch`) or run single-node."
            ),
        }
    pending = getattr(shared_state, "pending_integrate", None)
    if (
        isinstance(pending, dict)
        and pending
        and (pending.get("recovery") or {}).get("phase") not in ("restored", "accepted")
    ):
        return {
            "status": "failed",
            "error_class": "integrate_restore_incomplete",
            "error": "previous integration still requires recovery",
        }
    return None


def _uncommitted_revert(attempt: IntegrateAttempt, failure: str, what: str, **evidence: Any) -> dict[str, Any]:
    """Revert accepted work that could not be committed, since the next revert to HEAD would erase it unseen."""
    log.error("integrate_patch: %s could not be committed (%s); reverting", what, failure)
    return {
        "status": "reverted",
        "error_class": "keep_commit_failed",
        "error": failure,
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
        "artifacts_reverted": [],
        "reason": f"{what} could not be committed: {failure}",
        "workspace": str(attempt.output_root),
        **evidence,
    }


def _bringup_evidence(before: Any, after: Any) -> dict[str, Any]:
    """Both boots of an enablement round as the verdict reports them.

    These are the persisted observations read back by path; neither side
    re-classifies a log. A half that could not be loaded is named on the verdict
    rather than re-read.
    """
    return {
        "before_observation": (observation_summary(before.observation) if before.observation is not None else None),
        "before_observation_path": before.path,
        "before_observation_degraded": before.degraded,
        "after_observation": (observation_summary(after.observation) if after.observation is not None else None),
        "after_observation_path": after.path,
        "after_observation_degraded": after.degraded,
    }


def _kept_runtime_identity(stack_action: Any, provision_result: Any) -> dict[str, Any]:
    """What a KEEP'd attempt runtime must carry so every later bench re-activates the stack it graded.

    The action names a branch, a tag or an unpinned spec list; the resolved
    identity beside it is what makes the acquisition a rebuild path rather than a
    source that moves. A round that acquired no runtime records nothing.
    """
    if stack_action is None or provision_result is None or not getattr(provision_result, "ok", False):
        return {}
    return {
        "enablement_kept_stack_action": {
            **stack_action.to_state(),
            "resolved_ref": str(getattr(provision_result, "resolved_ref", "") or ""),
            "resolved_packages": {
                str(name): dict(spec)
                for name, spec in (getattr(provision_result, "resolved_packages", {}) or {}).items()
            },
            # The clone and the install inherit the whole process environment, so
            # a runtime acquired over an authenticated remote replays no better
            # than an install that was.
            "credential_channels": detect_credential_channels(os.environ),
        },
        "enablement_active_runtime": provision_result.runtime.to_state(),
        "installed_versions": dict(getattr(provision_result, "installed_versions", {}) or {}),
    }


def _advanced_round(
    attempt: IntegrateAttempt,
    bench_result: dict[str, Any],
    *,
    after: Any,
    run_reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """The verdict of a round that did not serve but drove the boot to a deeper wall."""
    framework_root = str(attempt.framework_root or "")
    wall = after.observation.stage_failed if after.observation is not None else None
    verdict = verdict_of(after.observation) if after.observation is not None else None
    signature = verdict.signature if verdict is not None else None
    return {
        "status": "advanced",
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [str(patch) for patch in attempt.applied],
        # An ADVANCED round stacks its patch and never reaches the KEEP capture,
        # so without this the tree its patch applied to is never recorded and a
        # later KEEP binds it to that round's framework root instead. The capture
        # proves a patch against the tree it is bound to, so a mis-binding refuses
        # the whole recipe -- for a legitimate multi-root stack a false refusal
        # rather than a false pass.
        "enablement_patch_roots": attempt.shared_state.enablement.accepted_patch_roots(
            done_payload=attempt.done_payload,
            applied=attempt.applied,
            framework_root=framework_root,
        ),
        "patches_reverted": [],
        "artifacts_applied": attempt.applied_artifacts,
        "extra_envs_applied": attempt.extra_envs_applied,
        "extra_server_args_applied": attempt.extra_server_args_applied,
        "framework_root": framework_root,
        "output_throughput": bench_result.get("output_throughput"),
        "enablement": True,
        "advanced": True,
        "runnable": False,
        "correctness_verified": False,
        "reason": (
            f"enablement progressed: {run_reason}; boot advanced "
            f"to a new gap ({wall.name if wall is not None else 'no wall recorded'})"
        ),
        "setup_commands_applied": attempt.setup_commands_applied,
        "after_signature": signature.to_dict() if signature is not None else {},
        "enablement_launch_log": str(bench_result.get("error") or ""),
        # The wall this round advanced to, for the next round's before half.
        "enablement_observation_path": after.path,
        "bench_result": bench_result,
        "workspace": str(attempt.output_root),
        **evidence,
    }


def _inert_keep_reason(
    delta_pct: float | None,
    keep_threshold_pct: float,
    patch_count: int,
    switch_manifest: list[dict[str, Any]],
    enablers: list[str],
) -> str:
    """Why a bundle was kept inert rather than kept or reverted."""
    bits = [
        f"bundle throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%"
        if delta_pct is not None
        else "bundle throughput not measurable",
        f"code kept inert ({patch_count} patch(es), all switches default-off) and "
        f"{len(switch_manifest)} lever(s) registered for per-lever exploration",
    ]
    if enablers:
        bits.append(
            f"{len(enablers)} declared enabler(s) ({', '.join(enablers)}) cannot pay standalone "
            f"and are only measurable inside their bundle"
        )
    return "; ".join(bits)


class _BenchSetup(NamedTuple):
    """The materialized base configuration one bench leg launches from."""

    config_path: Path
    model_path: str
    gpu_type: str
    benchmark_script: Any
    result_dir: Any


class _VariantOverlay(NamedTuple):
    """One bench leg's own layer over the base configuration."""

    extra_server_args: str
    extra_envs: dict[str, str]
    unset_envs: list[str]
    name_suffix: str


class _AdmittedGrid(NamedTuple):
    """The variant the origin-independent guards let through, or the rejection that stopped it."""

    grid: list[GridVariant]
    rejection: dict[str, Any] | None


def _materialized_bench_config(bench_leg: _PatchBench, default_config_path: Path | None) -> _BenchSetup:
    """Materialize the base configuration this leg's variant launches under."""
    params = bench_leg.params
    config_path = Path(params.get("config_path") or default_config_path or default_baseline_config())
    if not config_path.exists():
        raise RuntimeError(f"integrate_patch bench: config not found at {config_path}")
    model_path = resolve_session_model_path(
        params=params,
        state_model_path=bench_leg.state_model_path,
        for_serving=True,
    )
    gpu_type = str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
    benchmark_script = sanitize_script_name(params.get("benchmark_script"))
    return _BenchSetup(
        config_path=prepare_grid_benchmark_config(
            config_path,
            bench_leg.output_root,
            model_path=model_path or None,
            gpu_type=gpu_type or None,
            benchmark_script=benchmark_script,
            extra_envs=framework_run_eval_envs(params),
            remove_args=params.get("base_remove_args"),
            unset_envs=params.get("base_unset_envs"),
            args_mode=str(params.get("base_args_mode") or "append"),
            out_name="integrate_patch.with_envs.yaml",
        ),
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
        result_dir=sanitize_result_dir(params.get("result_dir")),
    )


def _patch_variant(params: dict[str, Any], specialist_task_id: str, overlay: _VariantOverlay) -> GridVariant:
    """The single variant one integrate_patch bench leg measures.

    It carries this round's own levers; run_grid composes the stack under them.
    """
    envs = dict(overlay.extra_envs)
    # Eval-origin enablement needs a raw accuracy for its runnable gate, so
    # RUN_EVAL=true must survive any variant overlay.
    if bool(params.get("enablement")) and is_eval_origin(params):
        envs["RUN_EVAL"] = "true"
    variant = GridVariant(
        name=f"integrate-patch-{specialist_task_id[:8]}{overlay.name_suffix}",
        extra_server_args=overlay.extra_server_args,
        extra_envs=envs,
        unset_envs=list(overlay.unset_envs),
        note=f"integrate_patch:{specialist_task_id}{overlay.name_suffix}",
    )
    variant.overlay_pythonpath = params.get("overlay_pythonpath", "")
    variant.source_patches = list(params.get("source_patches", []))
    runtime_override = params.get("runtime_override")
    if isinstance(runtime_override, dict) and runtime_override:
        # Preserve list/dict values; apply_runtime_override expects them.
        variant.runtime_override = dict(runtime_override)
    return variant


def _admitted_grid(variant: GridVariant, params: dict[str, Any]) -> _AdmittedGrid:
    """Hold an authored variant to the explore grid's origin-independent guards.

    An authored variant is no more exempt from a known-bad multi-node lever, an
    operator's env pin or the operator's skip list than a proposed one. The
    compatibility filter is deliberately absent -- it shells out to build the
    framework's arg parser, and a single authored variant must not pay for that
    probe.
    """
    grid, mn_dropped = apply_multi_node_invalid_variants([variant])
    grid, pin_dropped = apply_aiter_moe_pin_filter(grid)
    grid, skip_dropped = apply_user_skip_list(grid, skip_spec=resolve_skip_spec(params))
    if grid:
        return _AdmittedGrid(grid, None)
    return _AdmittedGrid(grid, (mn_dropped + pin_dropped + skip_dropped)[0])


async def _bench_grid_results(
    setup: _BenchSetup,
    bench_leg: _PatchBench,
    grid: list[GridVariant],
) -> list[VariantResult]:
    """Run this leg's grid while holding the whole-machine serving lease.

    Ray-managed GPU execution holds a serving lease (num_gpus=TP + serving_slot)
    for the whole run_grid so the patch benchmark serializes against other
    serving instead of colliding with a concurrently-running GPU specialist
    server on the same card (the observed ``reverted_smoke_fail`` root cause).
    ``None`` keeps the local path (multi-node / RAY_EXEC off / pytest default).
    """
    from ._ray_serving import maybe_serving_lease

    params = bench_leg.params
    serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(setup.config_path))
    try:
        return await run_grid(
            base_yaml_path=setup.config_path,
            base_extra_args=str(params.get("base_extra_args") or "").strip(),
            grid=grid,
            output_root=bench_leg.output_root,
            magpie_python=params.get("magpie_python") or None,
            keep_going_on_failure=False,
            model_path=setup.model_path or None,
            gpu_type=setup.gpu_type or None,
            benchmark_script=setup.benchmark_script,
            result_dir=setup.result_dir,
            base_args_mode=str(params.get("base_args_mode") or "append"),
            base_extra_envs=dict(params.get("base_extra_envs") or {}),
            base_remove_args=to_str_list(params.get("base_remove_args")),
            base_unset_envs=to_str_list(params.get("base_unset_envs")),
            serving_lease=serving_lease,
            session_deadline_sec=bench_leg.session_deadline_sec,
            variant_expected_sec=bench_leg.variant_expected_sec,
        )
    finally:
        if serving_lease is not None:
            serving_lease.close()


def _bench_record(
    result: VariantResult,
    setup: _BenchSetup,
    variant: GridVariant,
    params: dict[str, Any],
) -> dict[str, Any]:
    """The measurement one variant produced, in the shape every gate and collector reads."""
    base_envs = dict(params.get("base_extra_envs") or {})
    base_remove = to_str_list(params.get("base_remove_args"))
    args_mode = str(params.get("base_args_mode") or "append")
    effective_unset = list(dict.fromkeys(to_str_list(params.get("base_unset_envs")) + list(variant.unset_envs)))
    return {
        **result.to_dict(),
        "status": result.status,
        "output_throughput": result.output_throughput,
        # ``VariantResult`` names these ``ttft_mean_ms`` / ``tpot_mean_ms``;
        # the emitted keys stay ``ttft_ms`` / ``itl_ms`` for the collectors.
        "ttft_ms": result.ttft_mean_ms,
        "itl_ms": result.tpot_mean_ms,
        # The session's constraints fail closed: a lane that drops these refuses every KEEP.
        **constraint_fields(result),
        # Benchmark dir; ``grade_accuracy`` locates accuracy artifacts here.
        "workspace": result.workspace or "",
        "error": result.error or "",
        "error_class": result.error_class,
        "nonfatal_warnings": list(result.nonfatal_warnings),
        # Launch evidence is the immutable proof of the server that produced this
        # measurement. It must survive the patch result and current-best
        # promotion so GEAK can verify the handoff.
        "launch_evidence": dict(result.launch_evidence),
        "launch_evidence_path": result.launch_evidence_path or "",
        "server_log_path": result.server_log_path or "",
        # Materialized config used for this bench; needed by revalidation.
        "materialized_config": str(setup.config_path),
        # The composed stack, not the variant alone: revalidation replays this
        # and would otherwise boot without the base layer. RUN_EVAL is dropped --
        # the replay owns its own eval contract.
        "effective_config": {
            "extra_envs": {
                key: value
                for key, value in {**base_envs, **variant.extra_envs}.items()
                if key != "RUN_EVAL" and key not in effective_unset
            },
            "extra_server_args": compose_server_args(
                inherited_args="",
                base_extra_args=str(params.get("base_extra_args") or "").strip(),
                variant_extra_args=variant.extra_server_args,
                remove_args=base_remove,
                args_mode=args_mode,
            ),
            "remove_args": list(base_remove),
            "unset_envs": list(effective_unset),
            "args_mode": args_mode,
        },
    }


def _eval_evidence(
    bench: dict[str, Any],
    params: dict[str, Any],
    *,
    eval_search_root: str,
    benchmark_mode: str,
) -> dict[str, Any]:
    """What the eval run says about this measurement: the KEEP verdict, the raw score and the probe."""
    framework = params.get("framework") or os.environ.get("FRAMEWORK") or None
    served = bench.get("status") == "succeeded"
    results = parse_eval_results(eval_search_root, framework=framework, benchmark_mode=benchmark_mode) if served else {}
    accuracy = results.get("accuracy")
    # Raw accuracy for the KB record; ``accuracy_pass`` only carries a verdict.
    measured = float(accuracy) if isinstance(accuracy, (int, float)) else None
    # The enablement branch applies a floor of its own, so it needs the raw score.
    enablement = bool(params.get("enablement"))
    return {
        "accuracy_pass": (
            grade_accuracy(
                eval_search_root,
                params.get("accuracy_baseline"),
                framework=framework,
                benchmark_mode=benchmark_mode,
            )
            if served
            else None
        ),
        "accuracy": measured,
        "enablement_accuracy": measured if enablement else None,
        "enablement_accuracy_task": str(results.get("task") or "") if enablement else "",
        "enablement_accuracy_metric": str(results.get("metric") or "") if enablement else "",
        # Guarded: an empty root would send the recursive scan over the cwd.
        "eval_probe": read_eval_probe(eval_search_root) if eval_search_root else None,
    }


class _EnvOverrides(NamedTuple):
    """The variant env layer a proposal asks for, and the keys refused as unsafe."""

    envs: dict[str, str]
    dropped: list[str]


class _PatchRejection(NamedTuple):
    """Why an apply-stage verdict refused the deliverable, and what the author can do about it."""

    error_class: str
    error: Any
    advisory: str = ""
    feedback: tuple[ApplyFeedback, ...] = ()


class _PatchApplyOutcome(NamedTuple):
    """What an apply run left behind: the patch that did not land, with feedback for re-authoring."""

    errors: list[dict[str, str]]
    feedback: tuple[ApplyFeedback, ...]


class _ApplyRoot(NamedTuple):
    """The tree this round's patches apply to, or the verdict that stopped it being found."""

    framework_root: Path | None
    refusal: dict[str, Any] | None


@dataclass(frozen=True)
class _Deliverable:
    """What a specialist handed over for this round: its patches, config, artifacts and rewrite switches."""

    patch_paths: list[Path]
    config_changes: dict[str, str]
    extra_server_args: str
    extra_envs: dict[str, str]
    dropped_env_overrides: list[str]
    switch_manifest: list[dict[str, Any]]
    artifact_specs: list[_ArtifactSpec]
    artifact_resolve_errors: list[str]
    artifact_runtime_errors: list[str]


def _proposal_config_changes(params: dict[str, Any], done_payload: dict[str, Any] | None) -> dict[str, str]:
    """The env changes this round delivers, as the task carries them or as the specialist wrote them down."""
    config_changes = dict(params.get("config_changes") or {})
    if config_changes or not done_payload:
        return config_changes
    recorded = done_payload.get("config_changes")
    if not isinstance(recorded, dict):
        return config_changes
    return {str(key): str(value) for key, value in recorded.items()}


def _proposal_env_overrides(config_changes: dict[str, str], params: dict[str, Any]) -> _EnvOverrides:
    """The env layer this proposal asks for, with every key outside the variant allowlist refused."""
    requested = dict(config_changes)
    raw_extra_envs = params.get("extra_envs")
    if isinstance(raw_extra_envs, dict):
        requested.update({str(key): str(value) for key, value in raw_extra_envs.items()})
    envs, unsafe = filter_untrusted_env_mapping(requested, allow_predicate=is_allowed_variant_env_key)
    dropped = sorted(unsafe)
    if dropped:
        log.warning(
            "integrate_patch: dropping unsafe env override keys: %s",
            ", ".join(dropped),
        )
    return _EnvOverrides(envs, dropped)


def _rewrite_switches(
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """The framework-rewrite switches this round benches with on.

    Every rewrite in such a patch sits behind a switch that defaults OFF, so the
    applied patch is inert and benching it as-is would measure the baseline. The
    manifest also carries the dependency edges the gate needs to decide between a
    throughput KEEP and an inert one.
    """
    manifest, problems = _parse_framework_switches(params=params, done_payload=done_payload)
    if manifest and not patch_paths:
        # A manifest without a patch describes switches that gate code which was
        # never delivered. Setting them would be a no-op, and registering them as
        # levers would leave the ledger pointing at absent code.
        problems.append(
            f"discarded {len(manifest)} switch(es): the deliverable carries no patch, "
            f"so there is no rewrite for them to gate"
        )
        manifest = []
    if manifest:
        log.info(
            "integrate_patch: benching with %d framework rewrite switch(es) on\n%s",
            len(manifest),
            _switch_manifest.summarize(manifest, problems),
        )
    elif problems:
        log.warning(
            "integrate_patch: framework switch manifest unusable\n%s",
            _switch_manifest.summarize(manifest, problems),
        )
    return manifest


def _target_gpu_type(params: dict[str, Any], shared_state: Any) -> str:
    """The GPU architecture this round's tuned artifacts have to carry measured rows for."""
    return str(
        getattr(shared_state, "gpu_type", "") or params.get("gpu_type") or params.get("target_platform") or ""
    ).strip()


def _resolve_deliverable(
    specialist_workspace: Path,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
    shared_state: Any,
) -> _Deliverable:
    """Resolve everything this round delivers, from the task and from the specialist's own output."""
    explicit_patches = params.get("patches") or None
    patch_paths = _resolve_patch_paths(
        specialist_workspace=specialist_workspace,
        explicit_patches=(list(explicit_patches) if isinstance(explicit_patches, list) else None),
        done_payload=done_payload,
    )
    config_changes = _proposal_config_changes(params, done_payload)
    overrides = _proposal_env_overrides(config_changes, params)
    switch_manifest = _rewrite_switches(params, done_payload, patch_paths)
    if switch_manifest:
        overrides.envs.update(_switch_manifest.switch_env(switch_manifest))
    explicit_artifacts = params.get("artifacts")
    artifact_specs, artifact_resolve_errors = _resolve_artifact_specs(
        specialist_workspace=specialist_workspace,
        explicit_artifacts=(list(explicit_artifacts) if isinstance(explicit_artifacts, list) else None),
        done_payload=done_payload,
    )
    raw_args = params.get("extra_server_args")
    return _Deliverable(
        patch_paths=patch_paths,
        config_changes=config_changes,
        extra_server_args=(raw_args if isinstance(raw_args, str) and raw_args.strip() else ""),
        extra_envs=overrides.envs,
        dropped_env_overrides=overrides.dropped,
        switch_manifest=switch_manifest,
        artifact_specs=artifact_specs,
        artifact_resolve_errors=artifact_resolve_errors,
        artifact_runtime_errors=_validate_aiter_gemm_artifacts(
            artifact_specs,
            gpu_type=_target_gpu_type(params, shared_state),
        ),
    )


def _leaves_the_stack_untouched(attempt: IntegrateAttempt, deliverable: _Deliverable) -> bool:
    """Whether this round would leave the tree, the launch configuration and the runtime as it found them.

    A runtime this round acquired changes what the next boot executes; a
    re-provisioned kept one is the stack the last KEEP was already graded on.
    """
    return not (
        deliverable.patch_paths
        or deliverable.extra_server_args
        or deliverable.extra_envs
        or deliverable.artifact_specs
        or attempt.setup_commands_applied
        or (bool(attempt.attempt_venv_root) and not attempt.shared_state.enablement.kept_stack_action)
    )


def _no_patches_result(
    attempt: IntegrateAttempt,
    deliverable: _Deliverable,
    params: dict[str, Any],
) -> dict[str, Any]:
    """The verdict of a round whose deliverable would change nothing about the stack."""
    result: dict[str, Any] = {
        "status": "no_patches",
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
        "artifacts_applied": [],
        "artifact_errors": deliverable.artifact_resolve_errors,
        "reason": (
            "neither patches, config_changes, installable artifacts, nor "
            "allowlisted setup commands were supplied / discoverable for "
            "this specialist task"
        ),
        "setup_commands_applied": attempt.setup_commands_applied,
    }
    if params.get("enablement"):
        result["enablement"] = True
    # Forward the ungrounded-patch details so framework.py can surface them in
    # the next round's mandate. The field lives on done_payload (written by
    # runner.py) and must be forwarded here because this is the concrete dict
    # framework.py reads via maybe_rearm_enablement.
    ungrounded = (attempt.done_payload or {}).get("patches_ungrounded")
    if isinstance(ungrounded, list) and ungrounded:
        result["patches_ungrounded"] = ungrounded
    return result


def _apply_failed(attempt: IntegrateAttempt, error_class: str, error: Any) -> dict[str, Any]:
    """The shape every apply-stage rejection shares: nothing landed, so nothing was taken back out."""
    return {
        "status": "apply_failed",
        "error_class": error_class,
        "error": error,
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
    }


def _reauthorable_apply_failure(
    attempt: IntegrateAttempt,
    params: dict[str, Any],
    patch_paths: list[Path],
    rejection: _PatchRejection,
) -> dict[str, Any]:
    """Reject the delivered patches with everything the specialist needs to author them again."""
    result = _apply_failed(attempt, rejection.error_class, rejection.error)
    result["lane"] = _derive_lane(params)
    result["retry_feedback"] = [item.to_dict() for item in rejection.feedback]
    result["prior_patches"] = [str(patch) for patch in patch_paths]
    if rejection.advisory:
        result["advisory"] = rejection.advisory
    if params.get("enablement"):
        result["enablement"] = True
    return result


def _unresolved_root_rejection(explicit_root: str | None, patch_paths: list[Path]) -> _PatchRejection:
    """Why these patches could not be bound to a framework tree, in terms their author can act on."""
    if not explicit_root:
        return _PatchRejection(
            "no_framework_agent_root",
            "no framework_source_root resolved; cannot apply "
            "patches. Configure $INFERENCEX_PATH or pass "
            "params.framework_source_root.",
        )
    declared_root = resolved_explicit_root(explicit_root)
    if declared_root is None:
        return _PatchRejection(
            "framework_root_unresolved",
            f"framework_source_root {explicit_root!r} is not a readable directory",
        )
    if not _read_patch_texts(patch_paths):
        return _PatchRejection(
            "patch_unreadable",
            "no patch file could be read; verify paths and permissions",
        )
    missing_records = _preflight_missing_targets(declared_root, patch_paths)
    if missing_records:
        return _PatchRejection(
            "patch_target_missing",
            missing_records,
            "patch target file(s) absent from framework_source_root "
            f"{explicit_root}; author patches only against files "
            "that exist in the installed framework tree.",
        )
    return _PatchRejection(
        "patch_root_ambiguous",
        f"framework_source_root {explicit_root!r} could not be unambiguously matched to the patch targets",
    )


def _pending_sentinel(
    attempt: IntegrateAttempt,
    deliverable: _Deliverable,
    *,
    framework_root: Path | None,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """The record of what this attempt is about to change, written before it changes anything."""
    return {
        "specialist_task_id": attempt.specialist_task_id,
        "task_id": attempt.task_id,
        "patches": [str(patch) for patch in deliverable.patch_paths],
        "artifacts": [
            {"target": str(spec.target), "rel_target": spec.rel_target} for spec in deliverable.artifact_specs
        ],
        "config_changes": dict(deliverable.config_changes),
        "extra_server_args": deliverable.extra_server_args,
        "extra_envs": dict(deliverable.extra_envs),
        "framework_source_root": str(framework_root or ""),
        "workspace": str(attempt.output_root),
        "attempt_venv_root": attempt.attempt_venv_root,
        "recovery": recovery,
        "ts": _now_iso(),
    }


def _unvetted_patch_refusal(patch: Path) -> str:
    """Why this patch may not be let near a live tree, if it may not.

    ``vet_patches`` runs at authoring time inside the specialist runner, so a
    patch from anywhere else -- ``params.patches``, and every remotely-fetched
    ``upstream_pr`` diff -- arrives unvetted.
    """
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unreadable: {exc!r}"
    if not is_unified_diff(text):
        return "not a unified diff"
    escaping = patch_escapes_tree(text)
    return f"path escapes tree: {escaping!r}" if escaping is not None else ""


def _apply_patch_series(
    attempt: IntegrateAttempt,
    patch_paths: list[Path],
    *,
    git_head: str,
    backup_root: Path,
) -> _PatchApplyOutcome:
    """Apply each delivered patch to the attempt's tree, stopping at the first one that does not land.

    ``attempt.applied`` is appended to in place, so a cancellation mid-series
    still leaves the undo able to see what reached the tree.
    """
    framework_root = attempt.framework_root
    errors: list[dict[str, str]] = []
    feedback: list[ApplyFeedback] = []
    nogit_backups: list[dict[str, Any]] = []
    for patch in patch_paths:
        unvetted = _unvetted_patch_refusal(patch)
        if unvetted:
            errors.append({"patch": str(patch), "stderr": unvetted})
            break
        if git_head:
            ok, err, fb = _git_apply_collect_feedback(framework_root, patch)
        else:
            ok, err, taken, fb = _apply_patch_no_git(framework_root, patch, backup_root, seq_offset=len(nogit_backups))
            nogit_backups.extend(taken)
        if not ok:
            errors.append({"patch": str(patch), "stderr": err})
            if fb is not None:
                feedback.append(fb)
            break
        attempt.applied.append(patch)
    return _PatchApplyOutcome(errors, tuple(feedback))


class _FixClaim(NamedTuple):
    """A patch's claim to resolve a recorded finding, and what stands in the way of accepting it."""

    finding: str
    refusals: list[str]


class _SnapshotScope(NamedTuple):
    """Which paths a KEEP's snapshot covers, how each was touched, and what landed outside the tree."""

    rel_paths: list[str]
    declared_ops: dict[str, str]
    artifacts_outside_root: int


@dataclass(frozen=True)
class _AccuracyGate:
    """What the accuracy evidence allows: whether it blocks a KEEP, and how far the score moved."""

    verdict: bool | None
    blocks_keep: bool
    reason: str
    delta_pct: float | None


@dataclass(frozen=True)
class _KeepTerms:
    """Why a patch was kept: the axis it was graded on, the finding it fixes, and the levers it brings."""

    graded: Any
    correctness_fix: bool
    resolves_finding: str
    switch_manifest: list[dict[str, Any]]
    parity: dict[str, Any]


@dataclass
class _SourceLayer:
    """The committed source files behind one KEEP, as the optimization stack records them."""

    snapshot_dir: str = ""
    manifest_path: str = ""
    target_files: list[str] = field(default_factory=list)
    base_sha: str = ""
    complete: bool = False
    import_root: str = ""
    realized_patch: str = ""
    artifacts_outside_root: int = 0

    def record_capture(self, snap: Any, framework_root: Path, rel_paths: list[str], *, committed: bool) -> None:
        """Record what the snapshot captured, and the realized diff where a commit made one readable."""
        if not snap:
            return
        self.snapshot_dir = str(snap.get("snapshot_dir") or "")
        if self.snapshot_dir:
            from ...source_snapshot import MANIFEST_NAME

            self.manifest_path = str(Path(self.snapshot_dir) / MANIFEST_NAME)
        self.target_files = [
            str(item.get("rel") or "")
            for item in (snap.get("files") or [])
            if isinstance(item, dict) and item.get("rel")
        ]
        self.complete = bool(snap.get("complete"))
        # Only harvest when a real commit landed. Without a new commit
        # ``HEAD^..HEAD`` is the previous KEEP; harvesting it would publish that
        # diff as this KEEP's realized change. Leaving it empty falls back to the
        # delivered patch (realized=False), which is the honest record.
        if committed:
            from ._patch_snapshot import harvest_realized_diff

            self.realized_patch = harvest_realized_diff(
                framework_root, rel_paths, Path(self.snapshot_dir) / "realized.patch"
            )

    def evidence(self) -> dict[str, Any]:
        """How a KEEP verdict reports what was captured behind it."""
        return {
            "source_snapshot": self.snapshot_dir,
            "source_manifest": self.manifest_path,
            "source_snapshot_complete": self.complete,
            "source_import_root": self.import_root,
            "source_realized_patch": self.realized_patch,
            "source_artifacts_outside_root": self.artifacts_outside_root,
            "target_files": self.target_files,
            "base_sha": self.base_sha,
        }


def _snapshot_scope(framework_root: Path, applied: list[Path], artifacts: list[dict[str, Any]]) -> _SnapshotScope:
    """The files a KEEP changed under ``framework_root``, and how many of its artifacts landed outside it.

    An artifact installed into a sibling tree is not addressable by a rel path
    under this root, so it belongs to no snapshot here. The count travels so a
    KEEP whose gain lives outside the tree reads as a known gap rather than as a
    clean capture.
    """
    from ._patch_snapshot import _patch_touched_paths_split

    upserted, deleted = _patch_touched_paths_split(framework_root, applied)
    declared_ops = dict.fromkeys(upserted, "upsert")
    declared_ops.update(dict.fromkeys(deleted, "delete"))
    here = framework_root.resolve()
    rooted = [
        (str(entry["rel_target"]), Path(str(entry.get("root") or framework_root)).resolve())
        for entry in (artifacts or [])
        if isinstance(entry, dict) and entry.get("rel_target")
    ]
    return _SnapshotScope(
        rel_paths=upserted + deleted + [rel for rel, root in rooted if root == here],
        declared_ops=declared_ops,
        artifacts_outside_root=sum(1 for _rel, root in rooted if root != here),
    )


def _graded_delta_pct(graded: Any, new_tput: Any, *, base_tput: float) -> float | None:
    """The promotable gain of this measurement on the axis the session grades.

    Both sides come from one resolver; on the output axis the drift-resolved
    ``base_tput`` is the reference. A REVERT has no promotable delta; the reason
    travels to the ledger so the agent can tell it from a failed run.
    """
    if not graded.comparable:
        log.info("integrate_patch: performance comparison unavailable (%s)", graded.degrade_reason)
    if not graded.graded_on_intvty:
        return gain_pct(new_tput, base_tput)
    if graded.verdict == VERDICT_KEEP:
        return gain_pct(graded.candidate, graded.reference)
    log.info(
        "integrate_patch: %s intvty %.1f->%.1f tput %.1f->%.1f",
        graded.verdict,
        graded.reference,
        graded.candidate,
        graded.tput_reference,
        graded.tput_candidate,
    )
    return None


def _accuracy_keep_gate(
    params: dict[str, Any],
    extra: dict[str, Any],
    gate_evidence: dict[str, Any],
    *,
    specialist_task_id: str,
) -> _AccuracyGate:
    """Whether this round's accuracy evidence lets a KEEP through, and how far the score moved."""
    accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
    fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
    required = bool(params.get("require_accuracy_for_keep", fw_authored))
    shared_state = extra.get("shared_state") or extra.get("state")
    baseline = params.get("accuracy_baseline")
    if required and not baseline and shared_state is not None:
        baseline = getattr(shared_state, "baseline_accuracy", None)
    blocks, reason, degraded = accuracy_keep_block(
        accuracy_pass,
        required=required,
        baseline_accuracy=baseline,
    )
    if degraded:
        log.warning(
            "integrate_patch: accuracy gate required but no baseline accuracy; "
            "KEEP allowed on throughput only (task=%s)",
            specialist_task_id,
        )
    return _AccuracyGate(
        verdict=accuracy_pass,
        blocks_keep=blocks,
        reason=reason,
        delta_pct=_accuracy_delta_pct(
            gate_evidence.get("accuracy"),
            baseline or getattr(shared_state, "baseline_accuracy", None),
        ),
    )


def _clears_keep_gate(graded: Any, delta_pct: float | None, threshold_pct: float, accuracy: _AccuracyGate) -> bool:
    """Whether the measurement earns a KEEP on its own terms."""
    return (
        graded.comparable
        and delta_pct is not None
        and delta_pct >= threshold_pct
        and not graded.veto_reason
        and not accuracy.blocks_keep
    )


def _gate_failure_reasons(
    graded: Any,
    measured: _PatchMeasurement,
    *,
    accuracy: _AccuracyGate,
    fix: _FixClaim,
    eval_probe: Any,
) -> list[str]:
    """Everything the KEEP gate held against this measurement, in the order it was weighed."""
    reasons: list[str] = []
    if not graded.comparable:
        reasons.append(f"performance comparison unavailable: {graded.degrade_reason}")
    elif graded.graded_on_intvty and measured.delta_pct is None:
        reasons.append(
            graded.veto_reason
            or f"interactivity p50 {graded.reference:.1f}->{graded.candidate:.1f} did not clear the interactivity KEEP gate"
        )
    elif measured.delta_pct is None:
        reasons.append("no measurable throughput")
    elif measured.delta_pct < measured.keep_threshold_pct:
        reasons.append(
            f"throughput delta {measured.delta_pct:+.2f}% < keep_threshold {measured.keep_threshold_pct:.2f}%"
        )
    elif graded.veto_reason:
        reasons.append(graded.veto_reason)
    if accuracy.blocks_keep and accuracy.reason:
        reasons.append(accuracy.reason)
    if fix.refusals:
        reasons.append(f"correctness fix {fix.finding} refused: {'; '.join(fix.refusals)}")
    probe_reason = eval_probe_summary(eval_probe)
    if probe_reason:
        reasons.append(probe_reason)
    return reasons


def _revert_status(measured: _PatchMeasurement, accuracy: _AccuracyGate, fix: _FixClaim) -> str:
    """Whether the patch was rejected on its merits or only because its accuracy could not be read."""
    tput_ok = measured.delta_pct is not None and measured.delta_pct >= measured.keep_threshold_pct
    unreadable = accuracy.blocks_keep and accuracy.verdict is None and tput_ok
    return "accuracy_unavailable_reject" if unreadable or fix.refusals == [_FIX_ACCURACY_UNAVAILABLE] else "reverted"


def _perf_revert(
    attempt: IntegrateAttempt,
    measured: _PatchMeasurement,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    """The verdict of a patch that was benched and then taken back out."""
    return {
        "status": status,
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
        "artifacts_reverted": [],
        **measured.envelope(),
        "reason": reason,
        "workspace": str(attempt.output_root),
    }


def _kept_patch(
    attempt: IntegrateAttempt,
    measured: _PatchMeasurement,
    *,
    params: dict[str, Any],
    source: _SourceLayer,
    terms: _KeepTerms,
) -> dict[str, Any]:
    """The verdict of a patch that cleared the gate and stays in the tree."""
    return {
        "status": "kept",
        "specialist_task_id": attempt.specialist_task_id,
        # Proposal ownership must survive delegated-result persistence
        # so resume replay cannot replace it with the then-current phase.
        "source_phase": str(params.get("source_phase") or ""),
        "domain": str(params.get("domain") or params.get("source_domain") or ""),
        "provenance": str(params.get("provenance") or ""),
        "gap_canonical_id": str(params.get("gap_canonical_id") or ""),
        "gap_layer": str(params.get("gap_layer") or ""),
        "framework_agent_authoring": bool(params.get("framework_agent_authoring")),
        "patches_applied": [str(patch) for patch in attempt.applied],
        "patches_reverted": [],
        "artifacts_applied": attempt.applied_artifacts,
        "extra_server_args_applied": attempt.extra_server_args_applied,
        "extra_envs_applied": attempt.extra_envs_applied,
        **measured.envelope(),
        "reason": (
            f"correctness fix {terms.resolves_finding} verified at "
            f"{_fix_axis(terms.graded)} delta {measured.delta_pct:+.2f}%"
            if terms.correctness_fix
            else f"throughput delta {measured.delta_pct:+.2f}% >= {measured.keep_threshold_pct:.2f}%"
        ),
        "keep_reason": KEEP_REASON_CORRECTNESS_FIX if terms.correctness_fix else "",
        "resolves_finding": terms.resolves_finding if terms.correctness_fix else "",
        "workspace": str(attempt.output_root),
        **source.evidence(),
        "framework_root": str(attempt.framework_root or ""),
        # The bundle cleared the gate, so its switches join the running
        # configuration and are registered as levers that are already on.
        # Attribution from here is leave-one-out.
        "framework_levers": terms.switch_manifest,
        "framework_lever_outcome": ("default_on" if terms.switch_manifest else ""),
        "switch_off_parity": terms.parity,
    }


@dataclass(frozen=True)
class _PatchBench:
    """What every bench leg of one integration shares: its params, workspace, specialist and session bounds."""

    params: dict[str, Any]
    output_root: Path
    specialist_task_id: str
    state_model_path: str = ""
    session_deadline_sec: float | None = None
    variant_expected_sec: float | None = None
    benchmark_mode: str = ""

    @classmethod
    def for_attempt(cls, attempt: IntegrateAttempt, params: dict[str, Any]) -> _PatchBench:
        """The legs' shared inputs, bounded by the session wall-clock as it stands now."""
        shared_state = attempt.shared_state
        session_deadline_sec, variant_expected_sec = session_grid_bounds(shared_state)
        return cls(
            params=params,
            output_root=attempt.output_root,
            specialist_task_id=attempt.specialist_task_id,
            state_model_path=str(getattr(shared_state, "model_path", "") or ""),
            session_deadline_sec=session_deadline_sec,
            variant_expected_sec=variant_expected_sec,
            benchmark_mode=str(getattr(shared_state, "benchmark_mode", "") or ""),
        )


@dataclass(frozen=True)
class _PatchMeasurement:
    """The benched candidate measured against its base, as a verdict reports it."""

    bench_result: dict[str, Any]
    new_tput: Any
    delta_pct: float | None
    base_tput: float
    keep_threshold_pct: float
    accuracy_pass: bool | None
    acc_delta_pct: float | None
    cfg_fingerprint: str
    measured_against: dict[str, Any]

    def lesson(self, outcome: str) -> KbLesson:
        """The KB record of this measurement under ``outcome``."""
        return KbLesson(outcome, float(self.delta_pct or 0.0), self.acc_delta_pct, self.cfg_fingerprint)

    def envelope(self) -> dict[str, Any]:
        """How every perf verdict reports the measurement behind it."""
        return {
            "output_throughput": self.new_tput,
            "delta_pct": self.delta_pct,
            "accuracy_pass": self.accuracy_pass,
            "base_tput": self.base_tput,
            "measured_against": self.measured_against,
            "keep_threshold_pct": self.keep_threshold_pct,
            "bench_result": self.bench_result,
        }


class IntegratePatchExecutor:
    """ActionRunner for the ``integrate_patch`` action (PR-A4)."""

    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the integrate-patch executor.

        Args:
            session_dir (Path | str | None): Session output directory;
                auto-resolved when ``None``.
            default_config_path (Path | str | None): Fallback benchmark
                config path, if any.
            keep_threshold_pct (float): Minimum gain to KEEP a patch.
                Defaults to :data:`DEFAULT_KEEP_THRESHOLD_PCT`.
        """
        self.session_dir = session_dir
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Apply a specialist's patches/config changes and benchmark them."""
        attempt = IntegrateAttempt(task_id=ctx.task.task_id)
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}

        early = await self._stage_resolve(attempt, params, extra)
        if early is not None:
            return early

        # Provision an attempt-scoped runtime AFTER the Critic gate (in
        # _stage_resolve) and BEFORE any patch apply / setup replay.
        provision_early = await self._stage_provision_attempt_runtime(attempt, params, attempt.specialist_task_id)
        if provision_early is not None:
            return provision_early

        await self._stage_localize_source(attempt, params)

        # The apply and the gate both mutate the framework tree behind the
        # operator's auto-stash, and both cross awaits while it is on the stack --
        # the apply stage writes a KB record on each of its failure verdicts. So
        # the guard spans both: whichever stage was running, the candidate is
        # taken back out and the stash handed back, and the stop is re-raised
        # rather than graded. The attempt records mutation as it happens, even
        # when a stage cannot return because it was cancelled.
        try:
            result = await self._stage_apply(attempt, params, extra)
            if result is None:
                result = {**await self._stage_gate(attempt, params, extra), RESULT_BENCHED: True}
        except BaseException:
            try:
                self._finish_attempt(attempt, {"status": "cancelled"})
            except (OSError, AttributeError) as exc:
                log.error("integrate_patch: could not persist cancelled attempt recovery: %s", exc)
            raise
        return self._finish_attempt(attempt, result)

    def _finish_attempt(self, attempt: IntegrateAttempt, result: dict[str, Any]) -> dict[str, Any]:
        """Discharge the attempt's restore obligations exactly once before returning."""
        applied_only = result.get("status") == "applied_no_bench"
        accepted = result.get("status") in KEEP_STATUSES
        summary = (
            restore_pending_integrate(attempt.pending, keep=accepted or applied_only)
            if attempt.pending
            else {"failed": [], "artifacts_reverted": []}
        )
        if applied_only and attempt.pending and not summary["failed"]:
            recovery = attempt.pending["recovery"]
            recovery["phase"] = "applied"
            if recovery.get("stash_oid"):
                recovery["phase"] = "applied_with_restored_stash"
        state = attempt.shared_state
        if state is not None:
            state.save(self.session_dir)
        if summary["failed"]:
            if state is not None and hasattr(state, "set_stop_reason"):
                # Not environment_fault: the host is fine, the tree this attempt
                # patched is not, and the report for environment_fault tells the
                # operator to look at the install instead.
                state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
                state.save(self.session_dir)
            return {
                **result,
                "status": "failed",
                "error_class": "integrate_restore_incomplete",
                "recovery_errors": summary["failed"],
            }
        if not accepted and not applied_only:
            result["patches_reverted"] = [str(patch) for patch in attempt.applied]
            if attempt.applied_artifacts:
                result["artifacts_reverted"] = summary["artifacts_reverted"]
            root = attempt.attempt_venv_root
            if root:
                self._gc_attempt_dir(Path(root).parent)
        return result

    # ---------------------------------------------------------------------------
    # Stage helpers (called sequentially by __call__)
    # ---------------------------------------------------------------------------

    def _stage_resolve_launch_only(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        shared_state: Any,
    ) -> dict[str, Any] | None:
        """Give a launch-only round its own scratch workspace, or refuse a round that would mutate.

        Launch-only benches a pre-built runtime: there is no specialist to look up
        and nothing to apply, so any mutation field on the task contradicts the
        mode rather than extending it.
        """
        mutation_fields = [key for key in _LAUNCH_ONLY_MUTATION_FIELDS if params.get(key)]
        if mutation_fields:
            return {
                "status": "failed",
                "error_class": "launch_only_mutation_forbidden",
                "error": (
                    "enablement_launch_only accepts a pre-built runtime_override "
                    f"but no mutation fields; received {mutation_fields}"
                ),
                "patches_applied": [],
                "patches_reverted": [],
            }
        scratch = runs_dir(self.session_dir, "integrate_patch", attempt.task_id)
        scratch.mkdir(parents=True, exist_ok=True)
        attempt.specialist_task_id = attempt.task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = scratch
        return None

    async def _stage_resolve(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Guards (multi-node, task-id, workspace, Critic) + param normalisation.

        Returns an early-exit result dict on failure, or None to continue.
        Resolved inputs belong to this attempt, not to the runner context.
        """
        shared_state = extra.get("shared_state") or extra.get("state")
        refusal = _integration_refusal(params, shared_state)
        if refusal is not None:
            return refusal
        if shared_state is not None:
            bind_integrate_to_lane(params, shared_state, session_dir=Path(self.session_dir))
        if shared_state is not None and not params.get("accuracy_baseline"):
            _base_acc = getattr(shared_state, "baseline_accuracy", 0.0)
            if isinstance(_base_acc, (int, float)) and _base_acc > 0:
                params["accuracy_baseline"] = float(_base_acc)

        # Launch-only mode: pure bench of a pre-built runtime (no specialist, no Critic).
        if params.get("enablement_launch_only"):
            return self._stage_resolve_launch_only(attempt, params, shared_state)

        # Upstream-PR mode has no specialist to look up and no specialist
        # verdict to enforce: the Critic verdict on this proposal is the gate.
        if resolve_patch_source(params) == PATCH_SOURCE_UPSTREAM_PR:
            return self._stage_resolve_upstream_pr(attempt, params, shared_state)

        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "integrate_patch requires params.specialist_task_id "
                    "(the completed specialist whose worktree carries "
                    "the patches to integrate)"
                ),
            }
        # Rebind from live current_best so a task queued before an Explore KEEP
        # still measures against the real stack top.
        if shared_state is not None:
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)
        # Specialist workspace conventionally at runs/specialist/<id>/.
        specialist_workspace = runs_dir(self.session_dir, "specialist", specialist_task_id)
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (f"specialist workspace not found at {specialist_workspace}"),
                "specialist_task_id": specialist_task_id,
            }

        # Critic-verdict gate — enforced BEFORE any side effect (setup replay,
        # stash, patch/artifact apply, pod fan-out). Paths that bypass PolicyGate
        # (notably a queued/resume-dispatched task) are not re-validated there, so
        # a forged coordinator.db row with no genuine Critic verdict must be
        # rejected here, all-or-nothing, before it can install packages or mutate
        # the live framework tree. specialist_patch_verdicts is a Coordinator-only
        # field an LLM/forged row cannot write, and a legitimate
        # integrate_patch always has its verdict persisted before the queued task
        # is created (see intent_router._handle_single_verdict), so a genuine
        # task is unaffected. No-op when SharedState is absent.
        critic_reject = _enforce_critic_gate(shared_state, specialist_task_id)
        if critic_reject is not None:
            return critic_reject

        done_payload = _read_done_payload(specialist_workspace)
        _stamp_framework_kb_provenance(done_payload, params=params, shared_state=shared_state)

        attempt.specialist_task_id = specialist_task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = specialist_workspace
        attempt.done_payload = done_payload
        return None

    def _stage_resolve_upstream_pr(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        shared_state: Any,
    ) -> dict[str, Any] | None:
        """Resolve an upstream-PR candidate into patches on the task's own scratch.

        Sets ``params['patches']`` so the shared apply stage below sees the same
        explicit-paths input a specialist lane produces, and points the
        workspace at this task's scratch dir since there is no specialist one.

        Args:
            attempt: The integration attempt receiving the resolved inputs.
            params: Task params, mutated with the resolved patch paths.
            shared_state: SharedState, or ``None``.

        Returns:
            A terminal result when the candidate has no permissive Critic
            verdict or could not be materialised, else ``None``.
        """
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": "patch_source=upstream_pr requires params.candidate (the discovered PR row)",
                "patches_applied": [],
                "patches_reverted": [],
            }
        if shared_state is not None:
            # Same rebind the specialist lane does: a task queued before a KEEP
            # landed must still measure against the real stack top.
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)

        # A queued or resume-dispatched task is not re-validated by PolicyGate,
        # and this lane fetches a remote diff into the live framework tree.
        critic_reject = _enforce_critic_gate(
            shared_state,
            str(params.get("framework_agent_candidate_id") or "").strip(),
        )
        if critic_reject is not None:
            return critic_reject

        task_id = attempt.task_id
        scratch = runs_dir(self.session_dir, "integrate_patch", task_id)
        scratch.mkdir(parents=True, exist_ok=True)
        framework_root = _resolve_framework_root(str(params.get("framework_source_root") or "").strip() or None)
        if framework_root is None:
            return {
                "status": "failed",
                "error_class": "framework_root_unresolved",
                "error": "cannot resolve a framework source root to apply the candidate to",
                "patches_applied": [],
                "patches_reverted": [],
            }

        materialized = materialize_candidate_patches(
            candidate=candidate,
            params=params,
            framework_root=framework_root,
            dest=scratch / f"{_candidate_slug(candidate)}.patch",
            diff_fetch_timeout_sec=DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
        )
        if materialized.failure is not None:
            return {
                **materialized.failure,
                "candidate": candidate,
                "patches_applied": [],
                "patches_reverted": [],
                "patch_source_mode": materialized.mode,
                "workspace": str(scratch),
            }
        params["patches"] = [str(p) for p in materialized.patches]
        params["patch_source_mode"] = materialized.mode

        attempt.specialist_task_id = task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = scratch
        return None

    async def _stage_provision_attempt_runtime(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Provision the attempt-scoped runtime this round boots on (Rung 3).

        The runtime the last kept round promoted is re-provisioned whatever this round asks for, since the rest
        of the stack was proven on it. With none kept, the framework adapter builds the runtime the specialist
        requested (``needs_runtime`` in its done payload); no-op when it requested none. Runs a disk preflight,
        delegates provision+probe to the framework adapter (off the event loop; an in-flight pip install is not
        killed if the await is cancelled), and on success stores the resolved runtime on the attempt for the gate
        to activate via the YAML-layer ``runtime_override``. Returns an early-exit ``reverted`` dict on any
        provision failure (no patch side effects yet), or ``None`` to continue.
        """
        if not _acquiring_round(params):
            return None

        from ...enablement.runtime.adapters import get_adapter
        from ...enablement.runtime.stack_actions import EnablementStackAction, RuntimeRequest

        request = RuntimeRequest.from_payload(_done_field(attempt.done_payload, "needs_runtime"))
        kept = attempt.shared_state.enablement.kept_stack_action
        if kept:
            action = EnablementStackAction.from_state(kept)
            if request is not None:
                log.info("integrate_patch: needs_runtime ignored; the kept %s runtime stays", action.framework)
        else:
            if request is None:
                return None
            framework = str(getattr(attempt.shared_state, "framework", "") or "").strip().lower()
            action = get_adapter(framework).build_stack_action(request)
            if action is None:
                return None

        attempt_dir = (
            enablement_stacks_dir(self.session_dir)
            / (action.framework or "unknown")
            / (specialist_task_id or "attempt")
        )

        from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

        try:
            disk_preflight(attempt_dir.parent, n_candidates=1)
        except DiskPreflightError as exc:
            return {
                "status": "reverted",
                "error_class": "disk_preflight_failed",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision aborted: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — preflight is best-effort advisory
            log.warning("integrate_patch: disk preflight raised (%r); continuing", exc)

        adapter = get_adapter(action.framework)

        def _provision_and_probe():
            provisioned = adapter.provision(action, attempt_dir)
            if provisioned.ok and not adapter.probe(provisioned, action):
                from ...enablement.runtime.stack_actions import ProvisionResult as _PR

                return _PR(ok=False, log_path=provisioned.log_path, error="adapter probe failed after provision")
            return provisioned

        try:
            result = await asyncio.to_thread(_provision_and_probe)
        except Exception as exc:
            log.exception("integrate_patch: attempt-runtime provision raised")
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision raised: {exc!r}",
            }

        if not result.ok:
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_failed",
                "error": result.error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision failed: {result.error}",
                "provision_result": result.to_state(),
            }

        # Record the attempt venv root on the action so KEEP can persist it and
        # resume/GC can find it.
        action = EnablementStackAction.from_state({**action.to_state(), "attempt_venv_root": result.runtime.venv_root})
        attempt.provision_result = result
        attempt.stack_action = action
        log.info(
            "integrate_patch: attempt runtime provisioned for %s (venv=%s, versions=%s)",
            action.framework,
            result.runtime.venv_root,
            result.installed_versions,
        )
        return None

    @staticmethod
    def _gc_attempt_dir(attempt_dir: Path) -> None:
        """Remove a half/failed attempt-runtime dir (best-effort)."""
        try:
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir, ignore_errors=True)
        except Exception:
            log.debug("integrate_patch: attempt-dir GC failed for %s", attempt_dir, exc_info=True)

    async def _stage_localize_source(self, attempt: IntegrateAttempt, params: dict[str, Any]) -> None:
        """Fetch a localization diff and record it on ``attempt`` for ``_stage_apply`` (Rung 4).

        Does nothing when the round's gap needs no code acquisition, no
        bridging candidate in the framework's own repo was discovered, or the
        adapter cannot localize. A change that cannot be localized (compiled,
        oversize, outside the package, or not fetched) is skipped and the
        round runs on its own patches.
        """
        gap = _enablement_gap(params)
        if gap is None:
            return

        from hyperloom.inference_optimizer import framework_registry

        from ...enablement.runtime.adapters import get_adapter
        from ...enablement.runtime.localization import build_localization_diff

        refs = attempt.shared_state.enablement.candidate_refs
        if not refs:
            return
        framework = str(getattr(attempt.shared_state, "framework", "") or "").strip().lower()
        repo_url = framework_registry.repo_url(framework)
        if not repo_url:
            return
        ref = _first_ref_in_repo(refs, repo_url)
        if not ref:
            return

        adapter = get_adapter(framework)
        action = adapter.build_localization_action(gap, candidate_ref=ref, repo_url=repo_url)
        if action is None:
            return

        from hyperloom.agents.framework.sources import github as _gh

        diff_text, touched_paths, verdict = await asyncio.to_thread(
            build_localization_diff,
            action,
            fetch_pr_patches=lambda slug, num: _gh.pr_patches(slug, num),
            fetch_raw_file=lambda slug, ref, path: _gh.fetch_raw_file(slug, ref, path),
            scope=adapter.package_path,
        )
        if not verdict.is_localizable:
            log.info("integrate_patch: localization of %s skipped: %s", ref, verdict.reason)
            return

        loc_dir = runs_dir(self.session_dir, "integrate_patch", attempt.task_id) / "localization"
        loc_dir.mkdir(parents=True, exist_ok=True)
        gap_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.gap_id or "localization")
        patch_path = loc_dir / f"{gap_slug}.patch"
        patch_path.write_text(diff_text, encoding="utf-8")

        attempt.localization_patches = [patch_path]
        attempt.localization_action = action
        attempt.localization_touched = list(touched_paths)
        log.info(
            "integrate_patch: localization staged %s (%d file(s), kind=%s)",
            patch_path,
            len(touched_paths),
            action.kind,
        )

    @staticmethod
    def _admit_localization(attempt: IntegrateAttempt, framework_root: Path, patch_paths: list[Path]) -> list[Path]:
        """Put the localization ahead of the round's patches when it applies to the stashed tree, else drop it.

        The recorded patch set is what recovery sweeps, so it follows.
        """
        loc = attempt.localization_patches
        if not loc:
            return patch_paths
        if any(_detect_p_level(framework_root, p) is None for p in loc):
            log.info("integrate_patch: localization %s does not apply to %s; skipped", loc, framework_root)
            attempt.drop_localization()
            return patch_paths
        patch_paths = [*loc, *patch_paths]
        attempt.pending["patches"] = [str(p) for p in patch_paths]
        return patch_paths

    async def _record_closure_baseline(self, attempt: IntegrateAttempt, params: dict[str, Any]) -> None:
        """Observe the environment before the enablement's first setup command, through the KEEP's interpreter.

        The KEEP closure alone cannot say which versions the enablement changed;
        the setup script checks the difference between the two.
        """
        from ...enablement.recipe.keep_probe import KeepEvidence, graded_framework, probe_keep_environment

        closure, _assertions = await asyncio.to_thread(
            probe_keep_environment,
            params,
            KeepEvidence(
                build_manifest=(),
                specialist_task_id=attempt.specialist_task_id,
                provision_result=attempt.provision_result,
            ),
            framework=graded_framework(params, ""),
        )
        if not attempt.shared_state.enablement.record_closure_baseline(closure):
            return
        try:
            attempt.shared_state.save(self.session_dir)
        except (OSError, AttributeError):
            # Held in memory either way, and the rearm saves again.
            log.debug("integrate_patch: save after closure baseline failed", exc_info=True)

    async def _stage_apply(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Setup replay, patch/artifact apply, pending_integrate sentinel.

        Returns an early-exit result dict on failure/no-patches/apply_only,
        or None to continue to bench+gate. Records mutations on the attempt.
        """
        shared_state = attempt.shared_state
        # Read back what earlier rounds established before either publish point
        # below decides what this round launches with.
        inherited_args, inherited_envs = _established_enablement_config(params, shared_state)
        setup_refusal = await self._record_pre_mutation_state(attempt, params)
        if setup_refusal is not None:
            return setup_refusal

        specialist_workspace = attempt.specialist_workspace
        assert specialist_workspace is not None
        deliverable = _resolve_deliverable(specialist_workspace, params, attempt.done_payload, shared_state)
        attempt.switch_manifest = deliverable.switch_manifest
        if deliverable.artifact_runtime_errors:
            refused = await self._reject_artifacts(
                attempt,
                params,
                extra,
                _PatchRejection("artifact_not_runtime_ready", deliverable.artifact_runtime_errors),
            )
            refused["artifact_errors"] = deliverable.artifact_runtime_errors
            refused["reason"] = (
                "AITER GEMM model-config artifacts must contain measured, "
                "non-placeholder rows for the target GPU architecture and CU count"
            )
            return refused

        if _leaves_the_stack_untouched(attempt, deliverable):
            if params.get("enablement_launch_only"):
                # Launch-only mode: skip the no-patches early-return and fall through to bench.
                output_root = runs_dir(self.session_dir, "integrate_patch", attempt.specialist_task_id)
                output_root.mkdir(parents=True, exist_ok=True)
                attempt.output_root = output_root
                attempt.extra_envs_applied = dict(inherited_envs)
                attempt.extra_server_args_applied = inherited_args
                return None
            return _no_patches_result(attempt, deliverable, params)

        root = await self._resolve_apply_root(attempt, params, deliverable, extra)
        if root.refusal is not None:
            return root.refusal
        stash_refusal = self._begin_mutation(attempt, deliverable, params, extra, framework_root=root.framework_root)
        if stash_refusal is not None:
            return stash_refusal

        recovery = attempt.pending["recovery"]
        recovery_root = Path(recovery["root"])
        patch_paths = self._admit_localization(attempt, root.framework_root, deliverable.patch_paths)
        recovery["phase"] = "ready"
        if shared_state is not None:
            shared_state.save(self.session_dir)

        # An artifact preimage has to describe the tree as this attempt found
        # it, so it is taken before the patches run: a patch that creates the
        # very file an artifact then overwrites would otherwise be recorded as
        # pre-existing, and the recovery would leave it behind instead of
        # deleting it. Nothing is installed here, so a failure leaves the tree
        # exactly as the stash left it.
        if deliverable.artifact_specs:
            backup_errors = self._backup_artifacts(
                deliverable.artifact_specs,
                backup_root=recovery_root / "artifact_backups",
            )
            if backup_errors:
                return await self._reject_artifacts(
                    attempt,
                    params,
                    extra,
                    _PatchRejection("artifact_backup_failed", deliverable.artifact_resolve_errors + backup_errors),
                )
            recovery["artifacts_prepared"] = True
            if shared_state is not None:
                shared_state.save(self.session_dir)

        outcome = _apply_patch_series(
            attempt,
            patch_paths,
            git_head=str(recovery.get("git_head") or ""),
            backup_root=recovery_root / "patch_backups",
        )
        if outcome.errors:
            await self._maybe_write_framework_kb_record(
                KbLesson("rejected_apply_fail", 0.0), done_payload=attempt.done_payload, params=params, extra=extra
            )
            rejected = _reauthorable_apply_failure(
                attempt,
                params,
                patch_paths,
                _PatchRejection("git_apply_failed", outcome.errors, feedback=outcome.feedback),
            )
            rejected["workspace"] = str(attempt.output_root)
            return rejected

        if deliverable.artifact_specs:
            applied_artifacts, artifact_apply_errors = self._apply_artifacts(
                deliverable.artifact_specs,
                backup_root=recovery_root / "artifact_backups",
            )
            attempt.applied_artifacts = applied_artifacts
            if artifact_apply_errors:
                return await self._reject_artifacts(
                    attempt,
                    params,
                    extra,
                    _PatchRejection(
                        "artifact_install_failed", deliverable.artifact_resolve_errors + artifact_apply_errors
                    ),
                )

        # Inherited first, this round last: inheriting is not pinning.
        attempt.extra_server_args_applied = _merge_established_server_args(
            inherited_args, deliverable.extra_server_args
        )
        attempt.extra_envs_applied = {**inherited_envs, **deliverable.extra_envs}
        attempt.dropped_env_overrides = deliverable.dropped_env_overrides
        if params.get("apply_only"):
            return {
                "status": "applied_no_bench",
                "specialist_task_id": attempt.specialist_task_id,
                "patches_applied": [str(patch) for patch in attempt.applied],
                "patches_reverted": [],
                "artifacts_applied": attempt.applied_artifacts,
                "extra_server_args_applied": attempt.extra_server_args_applied,
                "extra_envs_applied": attempt.extra_envs_applied,
                "dropped_env_overrides": deliverable.dropped_env_overrides,
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(attempt.output_root),
            }
        return None

    async def _record_pre_mutation_state(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record every tree's HEAD, then replay this round's setup commands.

        The HEADs are recorded first because the setup commands are the round's
        first mutation: an install writes into the same trees the patches and
        artifacts land in. Returns the replay's refusal, or ``None`` to continue.
        """
        is_enablement = bool(params.get("enablement"))
        for candidate in _candidate_mutation_roots(params=params, done_payload=attempt.done_payload):
            _note_pre_mutation_head(attempt, candidate, enablement=is_enablement, session_dir=self.session_dir)
        if not is_enablement:
            return None
        if not attempt.shared_state.enablement.environment_closure_baseline:
            await self._record_closure_baseline(attempt, params)
        return await replay_setup_commands(attempt, params=params, session_dir=self.session_dir)

    async def _resolve_apply_root(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        deliverable: _Deliverable,
        extra: dict[str, Any],
    ) -> _ApplyRoot:
        """Bind this round's patches to the tree they will be applied to.

        An acquired runtime is what the boot resolves, so the shared framework
        tree is not what the server imports: this round's patches land in the
        tree it runs, and the root the specialist authored against does not
        describe that tree.
        """
        patch_paths = deliverable.patch_paths
        runtime_source_root = attempt.runtime_source_root
        explicit_root = runtime_source_root or str(params.get("framework_source_root") or "").strip() or None
        framework_root = (
            resolved_explicit_root(runtime_source_root)
            if runtime_source_root
            # Off the loop: settling a tie between matching trees runs git once per patch.
            else await asyncio.to_thread(
                _resolve_framework_root,
                explicit_root,
                patch_paths=patch_paths,
                recorded_root=_sole_patch_root(
                    attempt.done_payload, patch_paths, specialist_workspace=attempt.specialist_workspace
                ),
            )
        )
        if not patch_paths:
            return _ApplyRoot(framework_root, None)
        if framework_root is None:
            unresolved = _reauthorable_apply_failure(
                attempt, params, patch_paths, _unresolved_root_rejection(explicit_root, patch_paths)
            )
            # A set spanning two trees is exactly what fails root resolution here,
            # so this is the result that has to carry the reason onward.
            if (attempt.done_payload or {}).get("patches_span_multiple_roots"):
                unresolved["patches_span_multiple_roots"] = True
            return _ApplyRoot(None, unresolved)
        missing_records = _preflight_missing_targets(framework_root, patch_paths)
        if not missing_records:
            return _ApplyRoot(framework_root, None)
        await self._maybe_write_framework_kb_record(
            KbLesson("rejected_apply_fail", 0.0), done_payload=attempt.done_payload, params=params, extra=extra
        )
        return _ApplyRoot(
            None,
            _reauthorable_apply_failure(
                attempt,
                params,
                patch_paths,
                _PatchRejection(
                    "patch_target_missing",
                    missing_records,
                    "patch target file(s) absent from framework_source_root "
                    f"{framework_root}; author patches only against files that "
                    "exist in the installed framework tree (inspect it with "
                    "Glob/Grep before writing the diff).",
                ),
            ),
        )

    def _begin_mutation(
        self,
        attempt: IntegrateAttempt,
        deliverable: _Deliverable,
        params: dict[str, Any],
        extra: dict[str, Any],
        *,
        framework_root: Path | None,
    ) -> dict[str, Any] | None:
        """Open this attempt's workspace and recovery evidence, then bank the operator's work.

        The sentinel and the recovery record are written before any framework
        tree mutation; the Coordinator clears the sentinel after promoting the
        final result. Returns a refusal when the operator's uncommitted work
        could not be stashed, and ``None`` once the tree is safe to mutate.
        """
        shared_state = attempt.shared_state
        task_runs = runs_dir(self.session_dir, "integrate_patch", attempt.task_id)
        output_root = Path(params.get("output_dir") or extra.get("workspace") or task_runs)
        output_root.mkdir(parents=True, exist_ok=True)
        attempt.output_root = output_root
        git_head = ""
        if framework_root is not None and _is_git_tree(framework_root):
            git_head = _git_head_sha(framework_root)
            if deliverable.patch_paths and not git_head:
                # Patches on a git tree keep no per-file backups, so the recorded HEAD is their only way back.
                raise OSError(f"cannot read the HEAD of {framework_root} to restore to")
        task_runs.mkdir(parents=True, exist_ok=True)
        recovery_root = Path(tempfile.mkdtemp(prefix="recovery-", dir=task_runs)).resolve()
        recovery = new_recovery_record(framework_root, git_head=git_head, recovery_root=recovery_root)
        attempt.pending = _pending_sentinel(attempt, deliverable, framework_root=framework_root, recovery=recovery)
        if shared_state is not None:
            shared_state.pending_integrate = attempt.pending
            try:
                shared_state.save(self.session_dir)
            except OSError:
                log.exception("integrate_patch: failed to persist pending_integrate sentinel")

        # Normally already recorded before the setup commands; a root that
        # resolved only here is still recorded before the stash and the apply.
        _note_pre_mutation_head(
            attempt, framework_root, enablement=bool(params.get("enablement")), session_dir=self.session_dir
        )
        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            recovery["phase"] = "restored"
            log.error(
                "integrate_patch: cannot stash user changes in %s: %s; aborting to avoid data loss",
                framework_root,
                stash_note,
            )
            return _apply_failed(
                attempt,
                "stash_failed",
                f"refusing to proceed: user changes could not be stashed ({stash_note})",
            )

        # The stash is on the stack and the tree is about to be mutated, so
        # ``__call__``'s undo has to be able to see both before anything writes.
        attempt.framework_root = framework_root
        if stash_state == "stashed":
            cp = _run_git_cp(["-C", str(framework_root), "rev-parse", stash_note], timeout=30.0)
            if cp is None or cp.returncode != 0:
                raise OSError("cannot persist the exact pre-candidate stash identity")
            recovery["stash_oid"] = cp.stdout.strip()
        return None

    async def _reject_artifacts(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
        rejection: _PatchRejection,
    ) -> dict[str, Any]:
        """Refuse the round because its tuned artifacts are not installable as delivered."""
        await self._maybe_write_framework_kb_record(
            KbLesson("rejected_apply_fail", 0.0), done_payload=attempt.done_payload, params=params, extra=extra
        )
        result = _apply_failed(attempt, rejection.error_class, rejection.error)
        result["artifacts_applied"] = []
        if attempt.output_root is not None:
            result["workspace"] = str(attempt.output_root)
        return result

    async def _stage_gate(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Bench + enablement/perf KEEP/REVERT gate.

        Runs _bench_patch, applies the appropriate gate, and returns the
        final integration result. Never returns None.
        """
        specialist_task_id = attempt.specialist_task_id
        output_root = attempt.output_root
        assert output_root is not None
        params = attempt.benchmark_params(params)

        # Bound both bench legs by the session wall-clock, as the sweep and explore
        # arms already are: the declared cap answers "how long before this counts
        # as hung", not "how much budget is left", so without this a patch benched
        # near the end of a run could outlive the run itself. Resolved once here
        # and reused by the parity leg -- the deadline is an absolute monotonic
        # timestamp, so it stays correct as the gate progresses.
        try:
            bench_result, gate_evidence = await self._bench_patch(
                _PatchBench.for_attempt(attempt, params),
                extra_server_args_applied=attempt.extra_server_args_applied,
                extra_envs_applied=attempt.extra_envs_applied,
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "reverted",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "reason": str(exc),
                "workspace": str(output_root),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            }

        if params.get("enablement"):
            verdict = await self._gate_enablement(
                attempt=attempt,
                params=params,
                extra=extra,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
            )
        else:
            verdict = await self._gate_perf(
                attempt=attempt,
                params=params,
                extra=extra,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
            )
        if attempt.dropped_env_overrides:
            verdict["dropped_env_overrides"] = attempt.dropped_env_overrides
        return verdict

    async def _gate_enablement(
        self,
        *,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Enablement gate: runnability + minimal-correctness.

        A round is accepted when the boot runs or reaches a deeper wall, and its
        writes then stay in the tree for the next round to build on; ``kept`` and
        ``advanced`` differ only in whether the lane is finished. Anything short
        of that reverts to the state the previous round left.

        The verdict depends on the trigger origin, because the two origins have
        different evidence available:

        * ``accuracy >= floor`` -> ``correctness_ok=True`` (KEEP, verified).
          eval-origin additionally requires the score to carry a task + metric;
          without them it did not come from a real eval and fails closed.
        * present but below floor / non-positive / non-finite ->
          ``correctness_ok=False`` (REVERT, garbage output).
        * ``accuracy is None`` -> eval-origin fails closed
          (``correctness_ok=False``): the trigger *was* an accuracy failure, so a
          candidate that produces no score has not shown it fixed anything.
          boot-origin stays ``None`` (KEEP but provisional) — it only ever
          claimed to make the model boot, and eval-less runs must not be blocked.

        On KEEP the benched env/arg layers are reported as
        ``enablement_effective_config``, captured from the variant that ran: the
        materialized YAML holds only the base layer, so the revalidation baseline
        needs them to re-run the graded config.
        When an attempt runtime was provisioned, the stack action is recorded in
        the result (``enablement_kept_stack_action``) so it survives rearm. On
        REVERT / non-KEEP, the attempt runtime dir is GC'd.

        Every verdict carries ``framework_root`` (the source tree patches were applied
        against, needed to replay them on a fresh machine).
        """
        specialist_task_id = attempt.specialist_task_id
        done_payload = attempt.done_payload
        output_root = attempt.output_root
        assert output_root is not None
        framework_root = attempt.framework_root
        applied = attempt.applied
        applied_artifacts = attempt.applied_artifacts
        extra_envs_applied = attempt.extra_envs_applied
        extra_server_args_applied = attempt.extra_server_args_applied
        stack_action = attempt.stack_action
        provision_result = attempt.provision_result

        from hyperloom.common.failure_signature import runnable_decision

        from ...bringup import round_advanced
        from .benchmark_result import is_valid_measurement

        new_tput = bench_result.get("output_throughput")

        correctness_ok, eval_provenance = enablement_correctness(params, gate_evidence)

        after_loaded = load_boot_observation(bench_result.get("boot_observation_path"))
        before_loaded = load_boot_observation(params.get("enablement_before_observation_path"))
        evidence = {**_bringup_evidence(before_loaded, after_loaded), **eval_provenance}

        # A measurement exists only where the client completed requests, so it
        # witnesses the serving instead of inferring it from a log marker.
        served = is_valid_measurement(bench_result)

        runs, run_reason = runnable_decision(served=served, correctness_ok=correctness_ok)
        advanced = not runs and not served and round_advanced(before_loaded.observation, after_loaded.observation)
        if not runs and not advanced:
            await self._maybe_write_framework_kb_record(
                KbLesson("reverted_smoke_fail", 0.0), done_payload=done_payload, params=params, extra=extra
            )
            return {
                "status": "reverted",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "framework_root": str(framework_root or ""),
                "output_throughput": new_tput,
                "enablement": True,
                "runnable": False,
                "correctness_verified": correctness_ok is True,
                "reason": f"enablement not runnable: {run_reason}",
                "setup_commands_applied": attempt.setup_commands_applied,
                "bench_result": bench_result,
                "workspace": str(output_root),
                **evidence,
            }

        # Accepted: the boot either runs or reached a deeper wall. Both keep the
        # work in the tree and differ only in whether the lane is finished.
        commit_failure, _ = self._commit_accepted_work(
            attempt, f"hyperloom enablement {'advanced' if advanced else 'kept'} {specialist_task_id}"
        )
        if commit_failure:
            return _uncommitted_revert(
                attempt,
                commit_failure,
                "enablement progress",
                enablement=True,
                framework_root=str(framework_root or ""),
            )

        if advanced:
            await self._maybe_write_framework_kb_record(
                KbLesson("integrated", 0.0), done_payload=done_payload, params=params, extra=extra
            )
            return _advanced_round(attempt, bench_result, after=after_loaded, run_reason=run_reason, evidence=evidence)

        provisional = correctness_ok is None
        reason = f"enablement runnable: {run_reason}"
        if provisional:
            reason += " (provisional: served but eval produced no accuracy; correctness not verified)"
        await self._maybe_write_framework_kb_record(
            KbLesson("integrated", 0.0), done_payload=done_payload, params=params, extra=extra
        )
        kept_result: dict[str, Any] = {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "artifacts_applied": applied_artifacts,
            "extra_server_args_applied": extra_server_args_applied,
            "extra_envs_applied": extra_envs_applied,
            "framework_root": str(framework_root or ""),
            "output_throughput": new_tput,
            "enablement": True,
            "runnable": True,
            "correctness_verified": correctness_ok is True,
            "provisional": provisional,
            "reason": reason,
            "setup_commands_applied": attempt.setup_commands_applied,
            "bench_result": bench_result,
            "workspace": str(output_root),
            # Base YAML only; the env/arg layers live in enablement_effective_config.
            "enablement_accepted_config_path": str(bench_result.get("materialized_config") or ""),
            # Captured from the variant this leg launched, so a revalidation
            # replays the graded configuration rather than a re-derived one.
            "enablement_effective_config": dict(bench_result.get("effective_config") or {}),
            **evidence,
        }
        kept_result.update(_kept_runtime_identity(stack_action, provision_result))
        # Snapshot a manifest of the localized closure that survives rearm, so
        # the closure is recorded and not re-fetched.
        manifest = await asyncio.to_thread(
            self._finalize_localization_keep,
            attempt,
            framework_root=framework_root,
            specialist_task_id=specialist_task_id,
        )
        if manifest:
            kept_result["enablement_localization_manifest"] = manifest
        try:
            kept_result.update(self._enablement_keep_records(attempt, params=params, bench_result=bench_result))
        except (OSError, subprocess.SubprocessError, ConfigUnreadable):
            # Every field this fills is one the decision refuses the replay for
            # when absent, so a capture that cannot read the tree or the graded
            # config, spawn the probe, or see the durable stack leaves the
            # recipe insufficient rather than failing the round.
            log.exception("integrate_patch: enablement KEEP record capture failed")
        return kept_result

    def _enablement_keep_records(
        self,
        attempt: IntegrateAttempt,
        *,
        params: dict[str, Any],
        bench_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture the per-root identity, payload and assertions of this KEEP."""
        return capture_keep_records(
            attempt,
            params=params,
            bench_result=bench_result,
            session_dir=self.session_dir,
        )

    @staticmethod
    def _commit_accepted_work(attempt: IntegrateAttempt, message: str) -> tuple[str, bool]:
        """Commit this round's patch and artifact writes so they survive a later revert.

        A non-git root needs no commit: nothing there resets the tree between
        rounds, so the writes are already durable.

        Returns:
            ``(failure_note, head_advanced)``. ``failure_note`` is empty on
            success. ``head_advanced`` is ``True`` only when a real commit
            landed, which is what makes ``HEAD^..HEAD`` this round's diff.
        """
        framework_root = attempt.framework_root
        if framework_root is None or not _is_git_tree(framework_root):
            return "", False
        root_resolved = framework_root.resolve()
        try:
            touched = _patch_touched_paths(framework_root, attempt.applied)
            # An artifact installed into a sibling root is not addressable by a
            # rel path under this one, so it belongs to no commit here.
            artifact_rels = [
                str(a["rel_target"])
                for a in attempt.applied_artifacts
                if a.get("rel_target") and Path(str(a["root"])).resolve() == root_resolved
            ]
            ok, note = _git_commit_kept(framework_root, message, list(dict.fromkeys(touched + artifact_rels)))
        except (OSError, RuntimeError) as exc:
            return f"commit raised: {exc!r}", False
        if not ok:
            return note or "git commit failed", False
        return "", note == ""

    def _finalize_localization_keep(
        self,
        attempt: IntegrateAttempt,
        *,
        framework_root: Path | None,
        specialist_task_id: str,
    ) -> dict[str, Any]:
        """Record a localization manifest via :func:`snapshot_source_layer`.

        Blocking (filesystem copies); call via ``asyncio.to_thread``. Returns the
        manifest dict, empty when no localization ran.
        """
        touched = attempt.localization_touched
        if not touched or framework_root is None:
            return {}
        action = attempt.localization_action
        try:
            from ...source_snapshot import snapshot_source_layer

            base_sha = _git_head_sha(framework_root)
            dest = self.session_dir / "optimization_stack" / "localization" / (specialist_task_id or "keep")
            snap = snapshot_source_layer(
                framework_root=framework_root,
                base_sha=base_sha,
                rel_paths=touched,
                dest_dir=dest,
                provenance="localization",
                extra={
                    "specialist_task_id": specialist_task_id,
                    "kind": str(getattr(action, "kind", "") or ""),
                    "repo_url": str(getattr(action, "repo_url", "") or ""),
                    "pr_number": int(getattr(action, "pr_number", 0) or 0),
                },
            )
            return dict(snap) if snap else {}
        except Exception:
            log.exception("integrate_patch: localization snapshot failed")
            return {}

    async def _gate_perf(
        self,
        *,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Throughput KEEP / REVERT decision, or no verdict when the run stopped it."""
        specialist_task_id = attempt.specialist_task_id
        shared_state = attempt.shared_state
        done_payload = attempt.done_payload
        assert attempt.output_root is not None
        # Grade against the current live anchor, not a stale task snapshot.
        base_tput, anchor_drifted = resolve_anchor_with_drift(
            float(params.get("base_tput") or 0.0),
            shared_state,
        )
        if anchor_drifted:
            log.warning(
                "integrate_patch: anchor drift; grading against live anchor %.1f",
                base_tput,
            )

        keep_threshold_pct = float(params.get("keep_threshold_pct", self.keep_threshold_pct))

        stopped = stopped_by_the_run_class(bench_result.get("error_class"))
        if stopped is not None:
            return {
                "status": "failed",
                "error_class": stopped.error_class,
                "error": stopped.interrupted,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "bench_result": bench_result,
                "workspace": str(attempt.output_root),
            }

        graded = resolve_graded_comparison(shared_state, bench_result, keep_threshold_pct=keep_threshold_pct)
        delta_pct = _graded_delta_pct(graded, bench_result.get("output_throughput"), base_tput=base_tput)
        accuracy = _accuracy_keep_gate(params, extra, gate_evidence, specialist_task_id=specialist_task_id)
        gate_pass = _clears_keep_gate(graded, delta_pct, keep_threshold_pct, accuracy)
        fix = _FixClaim(str(_done_field(done_payload, "resolves_finding") or "").strip(), [])
        correctness_fix = False
        if not gate_pass and fix.finding:
            fix_delta_pct = gain_pct(graded.candidate, graded.reference) if graded.graded_on_intvty else delta_pct
            fix = _FixClaim(
                fix.finding,
                _correctness_fix_refusals(
                    fix.finding,
                    shared_state=shared_state,
                    bench_result=bench_result,
                    graded=graded,
                    delta_pct=fix_delta_pct,
                )
                + _fix_accuracy_refusals(accuracy.verdict),
            )
            if not fix.refusals:
                gate_pass = correctness_fix = True
                delta_pct = fix_delta_pct
        measured = _PatchMeasurement(
            bench_result=bench_result,
            new_tput=bench_result.get("output_throughput"),
            delta_pct=delta_pct,
            base_tput=base_tput,
            keep_threshold_pct=keep_threshold_pct,
            accuracy_pass=accuracy.verdict,
            acc_delta_pct=accuracy.delta_pct,
            cfg_fingerprint=canonical_fingerprint(params.get("extra_server_args"), params.get("extra_envs")),
            measured_against=_measured_against(params, base_tput=base_tput),
        )

        switch_manifest: list[dict[str, Any]] = list(attempt.switch_manifest)

        # Switch-off parity. Run before either KEEP verdict, since both of them
        # leave the patch on disk and therefore both depend on it being inert when
        # disabled.
        #
        # It runs on a quality regression too, which looks wasteful and is not: the
        # switches are benched together, so a moved output localises to the bundle
        # rather than to a switch. On a live session a four-switch bundle reached
        # +65.5% and was reverted whole on the gate, discarding three switches that
        # were never implicated along with the one that was. Default-off code costs
        # nothing to keep and explore can bisect it per lever -- but only if the tree
        # is genuinely unchanged with every switch unset, which is exactly what this
        # leg measures. An unswitched patch has no "off" state to fall back to, so
        # it still reverts without spending the leg.
        parity: dict[str, Any] = {"ran": False, "ok": True, "reason": ""}
        if switch_manifest:
            # The parity leg is an additional full bench, so it needs the same
            # session bound the first bench got. Resolved here rather than
            # threaded from the caller because the deadline is an absolute
            # monotonic timestamp: the budget the first bench spent is already
            # reflected in it. Only this leg reads it, so it is resolved only
            # when the leg runs.
            parity = await self._switch_off_parity(
                _PatchBench.for_attempt(attempt, params), switch_manifest=switch_manifest, base_tput=base_tput
            )
            if not parity.get("ok"):
                return await self._revert_on_broken_parity(attempt, measured, parity, params=params, extra=extra)

        if not gate_pass:
            # Two-tier verdict for a framework-rewrite patch. Every rewrite in it
            # is behind a switch that defaults OFF, so keeping the code with the
            # switches unset changes nothing at runtime -- which makes reverting
            # it the more expensive choice. The bundle failed as a bundle, but a
            # bundle usually mixes rewrites that pay with one that does not, and
            # some of them are enablers that cannot pay until measured together
            # with what they unlock. So keep the code inert, register the switches
            # as levers, and let the explore phase find the subset that wins.
            #
            # A quality regression does not condemn the bundle either, for the same
            # reason: the switches are benched together, so a moved output says
            # "at least one of these is wrong", not "all of them are". The bundle
            # stays inert and explore bisects it per lever -- the verdict carries
            # ``quality_unverified`` so nothing downstream mistakes it for a clean
            # keep. What still condemns it is failing parity, handled above: code
            # that is not inert when disabled would skew every later measurement,
            # and that check now runs on this path too.
            if switch_manifest and attempt.applied:
                return await self._keep_inert_switches(attempt, measured, params=params, extra=extra, parity=parity)
            await self._maybe_write_framework_kb_record(
                measured.lesson("reverted_smoke_fail"), done_payload=done_payload, params=params, extra=extra
            )
            reasons = _gate_failure_reasons(
                graded, measured, accuracy=accuracy, fix=fix, eval_probe=gate_evidence.get("eval_probe")
            )
            return _perf_revert(
                attempt,
                measured,
                status=_revert_status(measured, accuracy, fix),
                reason="; ".join(reasons) or "gate failed",
            )

        await self._maybe_write_framework_kb_record(
            measured.lesson("integrated"), done_payload=done_payload, params=params, extra=extra
        )
        commit_failure, keep_committed = self._commit_accepted_work(
            attempt, f"hyperloom KEEP {specialist_task_id} ({delta_pct:+.2f}%)"
        )
        if commit_failure:
            return _uncommitted_revert(
                attempt,
                commit_failure,
                "KEEP",
                output_throughput=measured.new_tput,
                delta_pct=delta_pct,
                measured_against=measured.measured_against,
                bench_result=bench_result,
            )
        return _kept_patch(
            attempt,
            measured,
            params=params,
            source=self._capture_source_layer(attempt, params, committed=keep_committed),
            terms=_KeepTerms(
                graded=graded,
                correctness_fix=correctness_fix,
                resolves_finding=fix.finding,
                switch_manifest=switch_manifest,
                parity=parity,
            ),
        )

    async def _revert_on_broken_parity(
        self,
        attempt: IntegrateAttempt,
        measured: _PatchMeasurement,
        parity: dict[str, Any],
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Take a rewrite patch back out because it did not prove inert with its switches unset.

        An unmeasurable parity leg reverts under its own verdict: the patch was
        never shown to break, so neither the log line nor the KB lesson may say
        that it did.
        """
        from ...knowledge import kb_writeback as _kb

        inconclusive = bool(parity.get("inconclusive"))
        kb_outcome = (
            _kb.OUTCOME_REVERTED_PARITY_INCONCLUSIVE if inconclusive else _kb.OUTCOME_REVERTED_SWITCH_OFF_PARITY
        )
        log.warning(
            "integrate_patch: switch-off parity %s task=%s: %s",
            "INCONCLUSIVE" if inconclusive else "FAILED",
            attempt.specialist_task_id,
            parity.get("reason"),
        )
        await self._maybe_write_framework_kb_record(
            measured.lesson(kb_outcome), done_payload=attempt.done_payload, params=params, extra=extra
        )
        result = _perf_revert(
            attempt,
            measured,
            status="reverted",
            reason=str(parity.get("reason") or "switch-off parity failed"),
        )
        result["error_class"] = "switch_off_parity_inconclusive" if inconclusive else "switch_off_parity_failed"
        result["switch_off_parity"] = parity
        return result

    def _capture_source_layer(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        *,
        committed: bool,
    ) -> _SourceLayer:
        """Snapshot the source files this KEEP leaves in the tree, so it can be replayed elsewhere."""
        layer = _SourceLayer()
        framework_root = attempt.framework_root
        if framework_root is None:
            return layer
        try:
            from ...enablement.runtime.adapters import get_adapter
            from ...source_snapshot import snapshot_source_layer

            layer.base_sha = _git_head_sha(framework_root)
            scope = _snapshot_scope(framework_root, attempt.applied, attempt.applied_artifacts)
            layer.artifacts_outside_root = scope.artifacts_outside_root
            layer.import_root = get_adapter(str(params.get("framework") or "")).source_import_root(str(framework_root))
            layer.record_capture(
                snapshot_source_layer(
                    framework_root=framework_root,
                    base_sha=layer.base_sha,
                    rel_paths=scope.rel_paths,
                    dest_dir=(
                        self.session_dir
                        / "optimization_stack"
                        / "src"
                        / (attempt.specialist_task_id or str(attempt.task_id or "keep"))
                    ),
                    provenance="integrate_patch",
                    extra={"specialist_task_id": attempt.specialist_task_id},
                    declared_ops=scope.declared_ops,
                    import_root=layer.import_root,
                ),
                framework_root,
                scope.rel_paths,
                committed=committed,
            )
        except Exception:
            log.exception("integrate_patch: source-layer snapshot failed")
        return layer

    async def _switch_off_parity(
        self,
        bench: _PatchBench,
        *,
        switch_manifest: list[dict[str, Any]],
        base_tput: float,
    ) -> dict[str, Any]:
        """Verify the patch is genuinely inert with every rewrite switch unset.

        The whole lever mechanism rests on one invariant: with no switch set, the
        patched tree behaves exactly like the original. That invariant is what
        makes it safe to keep unprofitable rewrite code on disk, what makes a
        per-lever measurement mean anything, and what keeps the baseline
        comparable across a session that has accumulated several rewrite patches.
        It is also the invariant an LLM is most likely to break by accident — by
        reading the switch once at import, inverting a default, or restructuring
        code outside the guard — and nothing else in the pipeline would notice: a
        switches-on bench that improves throughput looks like a success whether or
        not the switches-off path still works.

        So it is measured, not assumed. One extra leg with the switches removed
        must land inside a noise band around the pre-patch base.

        Args:
            bench: The task params, workspace, originating specialist and the
                session bounds the parity bench runs under.
            switch_manifest: Parsed switch manifest.
            base_tput: Pre-patch throughput to compare against.

        Returns:
            ``{"ran", "ok", "tput", "delta_pct", "band_pct", "accuracy_pass",
            "reason"}``. ``ran`` is False when the check was skipped: disabled
            passes and says so; no usable base to compare against is
            inconclusive and reverts.
        """
        params = bench.params
        band_pct = float(params.get("switch_off_parity_band_pct", DEFAULT_SWITCH_OFF_PARITY_BAND_PCT))
        if not bool(params.get("enable_switch_off_parity", True)):
            return {"ran": False, "ok": True, "reason": "switch-off parity check disabled"}
        if base_tput <= 0:
            return {
                "ran": False,
                "ok": False,
                "inconclusive": True,
                "reason": "switch-off parity could not be measured: no positive base throughput to compare against",
            }
        switch_names = [entry["switch"] for entry in switch_manifest]
        try:
            parity_bench, parity_evidence = await self._bench_patch(
                bench,
                extra_server_args_applied="",
                extra_envs_applied={},
                unset_envs=switch_names,
                variant_suffix="-parity",
            )
        except Exception as exc:  # noqa: BLE001 — a failed probe must not read as a pass
            return {
                "ran": True,
                "ok": False,
                "reason": f"switch-off parity leg raised: {exc!r}",
            }
        parity_tput = parity_bench.get("output_throughput")
        accuracy_pass = parity_evidence.get("accuracy_pass")
        if not isinstance(parity_tput, (int, float)) or parity_tput <= 0:
            # No measurement is not evidence of a behavioural change: the patch is
            # still reverted, but the verdict must not claim the invariant was broken.
            return {
                "ran": True,
                "ok": False,
                "inconclusive": True,
                "tput": parity_tput,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity could not be measured: the parity leg returned "
                    "no throughput, so the switches-unset invariant was never tested. "
                    "Reverting because an unverified rewrite must not stay on disk, not "
                    "because the patch was shown to be non-inert"
                ),
            }
        delta_pct = (float(parity_tput) - base_tput) / base_tput * 100.0
        if abs(delta_pct) > band_pct:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    f"switch-off parity leg moved throughput {delta_pct:+.2f}% "
                    f"(band +/-{band_pct:.2f}%): the patch changes behaviour with "
                    f"every switch unset, so it is not a default-off rewrite"
                ),
            }
        if accuracy_pass is False:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity leg failed its correctness gate: the patch "
                    "changes output with every switch unset"
                ),
            }
        return {
            "ran": True,
            "ok": True,
            "tput": float(parity_tput),
            "delta_pct": delta_pct,
            "band_pct": band_pct,
            "accuracy_pass": accuracy_pass,
            "reason": f"switch-off parity within +/-{band_pct:.2f}% ({delta_pct:+.2f}%)",
        }

    async def _keep_inert_switches(
        self,
        attempt: IntegrateAttempt,
        measured: _PatchMeasurement,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        parity: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep a correct-but-unprofitable rewrite patch dormant and register its levers.

        The bundle passed correctness but not the throughput threshold. Because
        every rewrite is behind a switch that defaults OFF, the applied code is
        inert: leaving it in place costs nothing at runtime, while reverting it
        would discard the rewrites that do pay along with the one that does not,
        and would discard any enabler whose whole purpose is to make another
        rewrite profitable rather than to be profitable itself.

        So the code stays, committed like any KEEP so the next attempt's revert to
        HEAD keeps it, and the switches are registered as search levers, with
        ``extra_envs_applied`` deliberately empty so nothing enters the running
        configuration. The explore phase then turns them on one dependency-closed
        bundle at a time.

        Args:
            attempt: The attempt whose applied patches, artifacts and switch
                manifest are kept.
            measured: The bench with the switches on against the comparison
                base, the throughput threshold it did not meet, its accuracy
                verdict, and the accuracy delta and config fingerprint for the
                KB record.
            params: The task params.
            extra: The runner's extra context.
            parity: The switch-off parity verdict, the inert KEEP's evidence of being inert.

        Returns:
            The ``kept_inert`` result envelope.
        """
        delta_pct = measured.delta_pct
        switch_manifest = attempt.switch_manifest
        commit_failure, _ = self._commit_accepted_work(attempt, f"hyperloom KEEP_INERT {attempt.specialist_task_id}")
        if commit_failure:
            return _uncommitted_revert(
                attempt,
                commit_failure,
                "KEEP_INERT",
                output_throughput=measured.new_tput,
                delta_pct=delta_pct,
                bench_result=measured.bench_result,
            )
        enablers = [entry["switch"] for entry in switch_manifest if entry.get("enabler")]
        await self._maybe_write_framework_kb_record(
            measured.lesson("kept_inert_levers_registered"),
            done_payload=attempt.done_payload,
            params=params,
            extra=extra,
        )
        log.info(
            "integrate_patch: KEEP_INERT task=%s delta=%s threshold=%.2f%% levers=%d enablers=%d",
            attempt.specialist_task_id,
            f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a",
            measured.keep_threshold_pct,
            len(switch_manifest),
            len(enablers),
        )
        return {
            "status": "kept_inert",
            # True when the bundle moved the output with every switch on. The
            # code is still kept, because the switches are benched together and
            # that verdict does not say which one is at fault — explore bisects
            # per lever from here. The flag exists so nothing downstream reads
            # this as a clean keep.
            "quality_unverified": measured.accuracy_pass is False,
            "specialist_task_id": attempt.specialist_task_id,
            "patches_applied": [str(p) for p in attempt.applied],
            "patches_reverted": [],
            "artifacts_applied": attempt.applied_artifacts,
            # Empty on purpose: the code is present but dormant, so nothing
            # may enter current_best. The levers below are how it gets turned
            # on, one measured bundle at a time.
            "extra_server_args_applied": "",
            "extra_envs_applied": {},
            **measured.envelope(),
            "reason": _inert_keep_reason(
                delta_pct, measured.keep_threshold_pct, len(attempt.applied), switch_manifest, enablers
            ),
            "workspace": str(attempt.output_root),
            "framework_root": str(attempt.framework_root or ""),
            "framework_levers": switch_manifest,
            "framework_lever_outcome": "registered_off",
            "switch_off_parity": parity,
        }

    # Helpers
    @staticmethod
    def _find_framework_proposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.
            params: Task params; the upstream-PR lane's PR identity lives on
                ``params['candidate']`` rather than in a specialist proposal.

        Returns:
            The matching framework proposal dict, or ``None`` when absent.
        """
        if not isinstance(done_payload, dict):
            return None
        proposal_set = done_payload.get("proposal_set") or []
        if not isinstance(proposal_set, list):
            return None
        for proposal in proposal_set:
            if not isinstance(proposal, dict):
                continue
            provenance = str(proposal.get("provenance") or "")
            if provenance.startswith("specialist:serving:framework"):
                return proposal
        return None

    @staticmethod
    def _upstream_pr_kb_proposal(params: Mapping[str, Any]) -> dict[str, Any] | None:
        """Present an upstream-PR candidate in the shape the KB writer reads.

        The KB writer reads the PR identity from ``fa_pr_url`` / ``fa_pr_sha``.
        A candidate row carries it directly, so map rather than relay it through
        a specialist's output.

        Args:
            params: Task params, read for ``candidate``.

        Returns:
            A proposal-shaped mapping, or ``None`` when this is not an
            upstream-PR task or the candidate carries no dedup key.
        """
        if resolve_patch_source(params) != PATCH_SOURCE_UPSTREAM_PR:
            return None
        candidate = params.get("candidate")
        if not isinstance(candidate, dict):
            return None
        pr_url = str(candidate.get("pr_url") or candidate.get("url") or "").strip()
        pr_sha = str(candidate.get("head_sha") or "").strip()
        if not pr_url and not pr_sha:
            return None
        return {
            "fa_pr_url": pr_url,
            "fa_pr_sha": pr_sha,
            "framework": candidate.get("framework") or "",
            "gap_canonical_id": candidate.get("gap_canonical_id") or "",
            "gap_keywords": candidate.get("gap_keywords") or [],
            "changed_files": candidate.get("changed_files") or [],
            "applicability": candidate.get("applicability") or "",
            # Names the lever, matching ``lever_kind``. No ledger reader filters
            # on provenance; it is audit metadata.
            "provenance": LEVER_UPSTREAM_PR,
            "source_framework": candidate.get("source_framework") or "",
            "target_framework": candidate.get("target_framework") or "",
        }

    async def _maybe_write_framework_kb_record(
        self,
        lesson: KbLesson,
        *,
        done_payload: dict[str, Any] | None,
        params: Mapping[str, Any] | None = None,
        extra: dict[str, Any],
    ) -> None:
        """Record this attempt's outcome against the patch's upstream PR identity.

        A patch with no such identity leaves no lesson: it is a one-off edit the
        ledger has no key to dedup future discovery against.
        """
        proposal = self._find_framework_proposal(done_payload)
        if proposal is None:
            # The upstream-PR lane carries the PR identity on the candidate
            # rather than in a specialist's ``fa_*`` markers.
            proposal = self._upstream_pr_kb_proposal(params or {})
            if proposal is None:
                return
        await write_framework_lesson(
            lesson,
            proposal,
            done_payload=done_payload,
            shared_state=extra.get("shared_state") or extra.get("state"),
        )

    def _backup_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> list[dict[str, str]]:
        """Commit every artifact target's preimage without touching the tree.

        Runs before the attempt's first mutation, so each record describes the
        tree as the attempt found it. The checkpoint that closes the batch is
        what :func:`restore_pending_integrate` later reads: until it lands, no
        artifact may be installed.

        Args:
            specs: The resolved artifact specs whose targets will be installed.
            backup_root: Directory under which clobbered targets are saved.

        Returns:
            Per-artifact error records; empty when the whole plan is committed.
        """
        errors: list[dict[str, str]] = []
        backup_root.mkdir(parents=True, exist_ok=True)
        for idx, spec in enumerate(specs):
            try:
                existed = spec.target.exists()
                backup_path: str | None = None
                mode = spec.target.stat().st_mode & 0o7777 if existed else None
                if existed:
                    backup_path = str(backup_root / f"{idx:03d}_{spec.target.name}.bak")
                    shutil.copy2(spec.target, backup_path)
                record = {
                    "target": str(spec.target),
                    "rel_target": spec.rel_target,
                    "root": str(spec.root),
                    "kind": spec.kind,
                    "existed": existed,
                    "backup": backup_path,
                    "source": str(spec.source),
                    "pre_image_sha256": file_digest(Path(backup_path)) if backup_path else "",
                    "mode": mode,
                }
                if existed and not record["pre_image_sha256"]:
                    raise OSError(f"artifact backup unreadable: {spec.target}")
                if not append_record(backup_root, record):
                    raise OSError(f"artifact backup ledger write failed: {spec.target}")
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
        if not errors and not mark_prepared(backup_root):
            errors.append({"artifact": "", "error": "artifact backup checkpoint could not be persisted"})
        return errors

    def _apply_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Install non-diff tuned artifacts over their committed preimages.

        Args:
            specs: The resolved artifact specs to install.
            backup_root: Directory :meth:`_backup_artifacts` committed under.

        Returns:
            A ``(applied, errors)`` tuple: the committed record of each artifact
            now installed, in order, and per-artifact error records.
        """
        committed = {str(row["target"]): row for row in load_records(backup_root)}
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for spec in specs:
            record = committed.get(str(spec.target))
            if record is None:
                errors.append({"artifact": spec.rel_target, "error": "no committed preimage for this target"})
                continue
            try:
                spec.target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(spec.source, spec.target)
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
                continue
            applied.append(record)
        return applied, errors

    async def _bench_patch(
        self,
        bench_leg: _PatchBench,
        *,
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        unset_envs: "list[str] | None" = None,
        variant_suffix: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy gate.

        Args:
            bench_leg: The task params (config / model / bench knobs), the
                per-task workspace root, the originating specialist task id
                (names the variant), ``SharedState.model_path`` (the last rung
                of the model-path precedence) and the session bounds. The
                caller owns the session context, so it resolves the bounds:
                ``session_deadline_sec`` is the monotonic-clock session budget
                deadline, or ``None`` when unbounded, and
                ``variant_expected_sec`` the expected bench runtime used to
                decide whether the remaining budget can fit this bench at all.
            extra_server_args_applied: Server CLI arguments for the variant.
            extra_envs_applied: Environment overrides layered onto the variant.
            unset_envs: Extra env names to remove for this leg, on top of the
                task's ``base_unset_envs``. Used by the switch-off parity leg,
                which has to guarantee the rewrite switches are absent even when
                an earlier KEEP put them into the base configuration.
            variant_suffix: Appended to the variant name so a second leg does not
                collide with the first one's grid slot.

        Returns:
            A ``(bench_result_dict, gate_evidence)`` tuple where
            ``bench_result_dict`` carries ``effective_config`` (the env/arg layers
            the variant launched with, for a faithful replay), and
            ``gate_evidence`` carries ``accuracy_pass`` (True / False / None)
            and ``eval_probe`` (the generation-pathology record, or ``None``).
        """
        setup = _materialized_bench_config(bench_leg, self.default_config_path)
        variant = _patch_variant(
            bench_leg.params,
            bench_leg.specialist_task_id,
            _VariantOverlay(
                extra_server_args=extra_server_args_applied,
                extra_envs=dict(extra_envs_applied),
                unset_envs=to_str_list(unset_envs),
                name_suffix=variant_suffix,
            ),
        )
        admitted = _admitted_grid(variant, bench_leg.params)
        if admitted.rejection is not None:
            return (
                {
                    "status": "skipped",
                    "error_class": "filtered_before_bench",
                    "error": str(admitted.rejection.get("reason") or admitted.rejection.get("source") or ""),
                    "workspace": "",
                    "materialized_config": str(setup.config_path),
                },
                {"accuracy_pass": None, "eval_probe": None},
            )
        results = await _bench_grid_results(setup, bench_leg, admitted.grid)
        bench: dict[str, Any] = _bench_record(results[0], setup, variant, bench_leg.params) if results else {}
        # Classified here so the gate compares two observations produced by the
        # same reader.
        bench["boot_observation_path"] = self._record_bench_bringup(bench, bench_leg.output_root)
        # lm-eval writes to ``$EVAL_RESULT_DIR`` under the grid slot, not inside
        # the ``benchmark_*`` workspace. Grade from the slot so the recursive
        # search finds eval output while honoring an explicit ``result_dir``
        # override the same way the grid subprocess does.
        eval_search_root = setup.result_dir or (str(Path(bench["workspace"]).parent) if bench.get("workspace") else "")
        return bench, _eval_evidence(
            bench,
            bench_leg.params,
            eval_search_root=eval_search_root,
            benchmark_mode=bench_leg.benchmark_mode,
        )

    def _record_bench_bringup(self, bench: dict[str, Any], output_root: Path) -> str:
        """Classify the server this bench ran against and persist the observation.

        Args:
            bench: The variant result, naming the server log it produced.
            output_root: The task workspace, used to key the artifact when the
                variant reports no slot of its own.

        Returns:
            str: The observation artifact's path, empty when there was no server
            log to classify or no session to write under.
        """
        from ...bringup import observe_bringup
        from .baseline import open_bringup_attempt, read_bringup_log, server_child_elapsed_sec

        server_log = str(bench.get("server_log_path") or "").strip()
        if not server_log or self.session_dir is None:
            return ""
        slot = Path(bench["workspace"]) if bench.get("workspace") else output_root
        read = read_bringup_log(Path(server_log))
        verdict = observe_bringup(
            server_log=read.text,
            server_elapsed_sec=server_child_elapsed_sec(read.text),
            wrapper_stderr=str(bench.get("error") or ""),
            session_dir=Path(self.session_dir),
        )
        return write_boot_observation(
            verdict.observation,
            session_dir=Path(self.session_dir),
            output_dir=slot,
            attempt=open_bringup_attempt(slot),
        )


__all__ = [
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "IntegratePatchExecutor",
    "KEEP_STATUSES",
    "RESULT_BENCHED",
    "_detect_p_level",
    "_git_apply",
    "_git_apply_reverse",
    "_git_reset_clean",
    "_git_restore_to_head",
    "_run_git_apply",
    "_resolve_framework_root",
    "_resolve_patch_paths",
    "_read_done_payload",
]
