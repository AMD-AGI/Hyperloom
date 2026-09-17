"""Build the enablement section the recorder writes at author time.

The Session Breakdown's read side -- which rebuilt every section by walking
state and disk -- was retired when the breakdown moved to recording at author
time (#1455). The enablement section still has to be assembled from the
enablement's own durable state, so the assembly lives here, on the recording
side that consumes it, instead of reaching back into a module upstream has
stood down.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.timeutil import iso_z, now_iso
from hyperloom.orchestrator.enablement.recipe.attempts import build_attempt_summary

from ..collectors._common import _to_int
from ..session_package import deliverable

log = logging.getLogger(__name__)

#: ``recipe.steps`` kind for a patch, matched without importing the recipe package.
PATCH_KIND_NAME = "patch"

# --------------------------------------------------------------------------

_ENABLEMENT_LOG_EXCERPT_CHARS = 2000


#: Distinguishes "this state never recorded the field" from a recorded ``None``.
_ABSENT = object()



_RECIPE_STATE_FIELDS: tuple[str, ...] = (
    "active_runtime",
    "accepted_config",
    "accepted_config_source",
    "accepted_stack_targets",
    "base_sha",
    "build_extensions_not_carried",
    "build_manifest",
    "levers_without_readers",
    "environment_closure",
    "framework_root",
    "installed_versions_at_keep",
    "kept_artifacts",
    "kept_patches",
    # ``select_linked_build`` falls back from ``last_specialist_task_id`` to the
    # kept rounds, precisely because that marker is one-shot and is normally
    # already consumed by the time a build is linked. Omitted here, the fallback
    # exists at runtime and can never fire in the recorded recipe: a build
    # reachable only through a round's durable identity disappears from the
    # projection, which is the case the fallback was added for.
    "kept_rounds",
    "kept_stack_action",
    "last_specialist_task_id",
    "launch_argv_refused",
    "launch_evidence",
    "patch_roots",
    "patch_targets",
    "roots",
    "setup_commands",
    "setup_executions",
    "source_snapshots",
)


#: Round-lifecycle counters the durable round ledger owns since the bring-up
#: round rework. Read for a pre-rework state document, never synthesised.
_LEGACY_ROUND_COUNTERS: tuple[str, ...] = ("attempts", "stall_streak")


#: Reasons that specifically deny a closed dependency set. The status cannot read
#: "verified" while one stands: pinned components are not a pinned environment,
#: and a closure captured over the Python layer alone does not cover a build or
#: an installer outside it.
_CLOSURE_DENYING_CODES: frozenset[str] = frozenset(
    {
        "build_inputs_incomplete",
        "build_attempt_unjoined",
        "environment_closure_absent",
        "closure_scope_incomplete",
        "setup_occurrences_unknown",
        "setup_ledger_truncated",
    }
)


def _eg(state: dict, name: str, default: Any = None) -> Any:
    """Read an enablement round field from a v4 nested or v3 flat state dict."""
    nested = state.get("enablement")
    if isinstance(nested, dict):
        return nested.get(name, default)
    return state.get(f"enablement_{name}", default)


def _rel(path: Path | None, session_dir: Path) -> str | None:
    """Express ``path`` relative to ``session_dir`` as a POSIX string.

    Returns ``None`` for ``None``, and falls back to ``str(path)`` when the
    path is not under the session.
    """
    if path is None:
        return None
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _as_int(value: Any, *, default: int = 0) -> int:
    """Coerce a state counter to int, falling back to ``default``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stack_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    """Project a stack-action dict onto the landed-stack summary."""
    return {
        "kind": str(action.get("kind") or ""),
        "framework": str(action.get("framework") or ""),
        "capability": str(action.get("capability") or ""),
        "acquisition_method": str(action.get("acquisition_method") or ""),
        "repo_url": str(action.get("repo_url") or ""),
        "ref": str(action.get("ref") or ""),
        "index_url": str(action.get("index_url") or ""),
        "reason": str(action.get("reason") or ""),
    }


def _runtime_summary(runtime: dict[str, Any], *, promoted: bool) -> dict[str, Any]:
    """Project a FrameworkRuntime-shaped dict onto an attempt-runtime row."""
    versions = runtime.get("installed_versions")
    return {
        "venv_root": str(runtime.get("venv_root") or ""),
        "bin_path": str(runtime.get("bin_path") or ""),
        "python_path": str(runtime.get("python_path") or ""),
        "installed_versions": {str(k): str(v) for k, v in versions.items()} if isinstance(versions, dict) else {},
        "promoted": bool(promoted),
    }



def _closure_status(decision: dict[str, Any], enablement: dict[str, Any]) -> str:
    """Return whether the recipe pins the dependency set, not merely components.

    Args:
        decision: The ``replay_sufficiency`` verdict this section carries.
        enablement: The projected round state, read for its execution ledger.
    """
    # The ledger is the only record of which installer families ran, and the
    # only durable one that records a failed execution, so an empty one leaves
    # the closure's scope unobserved rather than clean.
    if not (enablement.get("setup_executions") or []):
        return "unverified"
    denied = {str(r.get("code")) for r in decision.get("reasons") or []} & _CLOSURE_DENYING_CODES
    return "unverified" if denied else "verified"


def _recipe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Read the enablement fields the recipe contract is projected from."""
    return {name: _eg(state, name) for name in _RECIPE_STATE_FIELDS}


def _delivered_payloads(
    out: dict[str, Any],
    steps: list[dict[str, Any]],
    session_dir: Path,
) -> set[tuple[str, str]] | None:
    """What the session bundle actually hands a consumer of this recipe.

    The recipe names the bytes behind its manifests and digests, and the
    packager says which of them arrive as the recipe describes them; neither
    side restates the other's rules. ``None`` when the recipe references
    nothing, which is the one case with nothing to deliver.
    """
    from hyperloom.orchestrator.enablement.recipe.sufficiency import referenced_payloads

    referenced = referenced_payloads(out, steps)
    if not referenced:
        return None
    return deliverable(session_dir, referenced)


def _collect_recipe(
    out: dict[str, Any],
    state: dict[str, Any],
    *,
    session_dir: Path,
) -> None:
    """Emit the ordered replay contract and the verdict over it.

    ``recipe_steps`` is emitted only when non-empty, so a session that
    contributed nothing emits no key at all. ``replay_sufficiency`` is emitted
    unconditionally beside it: its own absence is the one absence that carries
    meaning, and a consumer must read it as insufficient.
    """
    from hyperloom.orchestrator.enablement.recipe import build_recipe_steps, evaluate_replay_sufficiency
    from hyperloom.orchestrator.enablement.recipe.projections import (
        project_accepted_config,
        project_launch_evidence,
        project_roots,
        project_runtime_provenance,
        project_source_snapshots,
    )

    enablement = _recipe_state(state)
    steps = build_recipe_steps(enablement, attempt_summary=build_attempt_summary)
    if steps:
        # Normalized on the way out, never on the way in: ``patch_targets`` and
        # ``patch_roots`` are keyed by the raw path, so rewriting the input makes
        # every lookup miss and the whole recipe reports ``patch_targets_unknown``
        # -- a portability fix that silently costs the steps their targets. The
        # published value is the only part a consumer reads.
        out["recipe_steps"] = [_portable_step(st, session_dir) for st in steps]
    accepted_config = project_accepted_config(enablement.get("accepted_config"))
    if accepted_config:
        archived = str(_eg(state, "accepted_config_path", "") or "")
        config_path = archived or str(_eg(state, "probe_config_path", "") or "")
        if config_path:
            accepted_config["config_path"] = _rel(Path(config_path), session_dir) or config_path
        out["accepted_config"] = accepted_config
    evidence, argv_refused = project_launch_evidence(enablement.get("launch_evidence"))
    out["accepted_config_source"] = str(enablement.get("accepted_config_source") or "") or None
    out["launch_evidence"] = evidence
    for key, value in (
        ("roots", project_roots(enablement.get("roots"))),
        ("source_snapshots", project_source_snapshots(enablement.get("source_snapshots"))),
        ("accepted_stack_targets", enablement.get("accepted_stack_targets") or {}),
        ("base_sha", str(enablement.get("base_sha") or "") or None),
        ("runtime_provenance", project_runtime_provenance(enablement)),
        ("environment_closure", enablement.get("environment_closure") or None),
        ("installed_versions_at_keep", enablement.get("installed_versions_at_keep") or None),
    ):
        if value:
            out[key] = value
    for _tri in ("build_extensions_not_carried", "levers_without_readers"):
        _val = _eg(state, _tri, _ABSENT)
        if _val is not _ABSENT:
            out[_tri] = _val
    _carry = _eg(state, "build_extensions_not_carried", _ABSENT)
    if _carry is not _ABSENT:
        # Assigned outside the loop above, which drops anything falsy: this
        # observation is a tri-state where ``None`` (the build's outputs could
        # not be read) and ``[]`` (they were all carried) mean opposite things,
        # and dropping either would read as the safe one. Read through a
        # sentinel default so a session that predates the observation stays
        # absent instead of arriving as an unreadable build.
        out["build_extensions_not_carried"] = _carry
    decision = evaluate_replay_sufficiency(
        enablement,
        steps=steps,
        section=out,
        delivered_payloads=_delivered_payloads(out, steps, session_dir),
        launch_argv_refused=argv_refused or bool(enablement.get("launch_argv_refused")),
    )
    out["replay_sufficiency"] = decision
    out["dependency_closure_status"] = _closure_status(decision, enablement)


def _lane_dispatched(state: dict[str, Any]) -> bool:
    """Whether the lane ever opened a round, over both state generations.

    A round no longer parks its task id in the enablement state, so the
    post-rework evidence that one ran is the specialist it settled onto, the
    per-round records it kept, and the setup rows a round stamped its own id
    onto. ``inflight_task_id`` / ``attempts`` are read for a document written
    before the rework, not as the primary signal -- reading only those would
    report every current session as never dispatched.

    ``launch_observation_path`` is deliberately NOT read here. All three of its
    writers (``writeback._record_enablement_eval_trigger`` and the two boot
    failure paths) set it from the *trigger* observation -- the failed launch or
    failed eval that gives the lane something to author against -- and they run
    before any round is dispatched. Reading it would report ``dispatched: true``
    for a session whose lane never opened a round, which is a false positive in
    the one direction this section must not fail.
    """
    return bool(
        _eg(state, "last_specialist_task_id")
        or _eg(state, "kept_rounds")
        or any(
            str(row.get("round_task_id") or "")
            for row in (_eg(state, "setup_executions") or [])
            if isinstance(row, dict)
        )
        or _eg(state, "inflight_task_id")
        or _as_int(_eg(state, "attempts")) > 0
    )


def _enablement_lane_status(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the lane's own status keys, or ``None`` when nothing is emitted.

    The section exists when the lane did something or was explicitly turned off;
    with ``all`` the default, "armed but never needed" is the case that stays
    hidden.
    """
    origin = str(_eg(state, "origin", "") or "")
    # eval_kind is NOT cleared on success, so it can identify an eval-origin
    # enablement even after the run succeeds and origin is reset to "".
    eval_kind = str(_eg(state, "baseline_eval_kind", "") or "")
    # A state document carrying no mode is read as the SharedState default,
    # which is the value the lane actually ran under.
    mode = str(state.get("enablement_mode") or "all").strip().lower() or "all"
    dispatched = _lane_dispatched(state)
    have_eval = origin == "eval" or bool(eval_kind)
    engaged = bool(
        dispatched
        or have_eval
        or _eg(state, "kept_patches")
        or _eg(state, "setup_executions")
        or _eg(state, "human_review_logged")
    )
    provisioned = any(
        _eg(state, name) for name in ("active_runtime", "attempt_runtimes", "build_manifest", "last_build_failure")
    )
    if not (engaged or mode == "off" or provisioned):
        return None
    out: dict[str, Any] = {
        "mode": mode,
        "engaged": engaged,
        "origin": "eval" if have_eval else "boot",
        "dispatched": dispatched,
        "succeeded": bool(_eg(state, "succeeded")),
        "pending": bool(_eg(state, "pending")),
        "validation_pending": bool(_eg(state, "validation_pending")),
    }
    for name in _LEGACY_ROUND_COUNTERS:
        raw = _eg(state, name)
        if raw is not None:
            out[name] = _as_int(raw)
    return out


def _collect_round_identity(out: dict[str, Any], state: dict[str, Any]) -> None:
    """Emit the task identities and the trigger log of the current round."""
    for key, value in (
        ("inflight_task_id", str(_eg(state, "inflight_task_id", "") or "")),
        ("last_specialist_task_id", str(_eg(state, "last_specialist_task_id", "") or "")),
        ("revalidation_generation", _as_int(_eg(state, "revalidation_generation"))),
        ("revalidation_task_id", str(_eg(state, "revalidation_task_id", "") or "")),
    ):
        if value:
            out[key] = value
    # The boot-origin trigger evidence: without it a launch-failure round shows
    # no reason for having run at all.
    launch_log = str(_eg(state, "launch_log", "") or "")
    if launch_log:
        out["launch_log_excerpt"] = launch_log[-_ENABLEMENT_LOG_EXCERPT_CHARS:]



def _portable_patch_ref(raw: str, session_dir: Path) -> str:
    """A patch reference that names no directory on the authoring host."""
    rel = _rel(Path(raw), session_dir)
    if not rel:
        return Path(raw).name
    return Path(rel).name if Path(rel).is_absolute() else rel



def _portable_step(step: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """A recipe step with its host paths reduced to what a consumer can act on.

    ``root`` is the tree a patch was resolved against, and it is absolute by
    construction. Nothing reads it -- the rules join on ``root_id``, and
    ``project_roots`` drops the same path for the same reason -- so shipping it
    only tells the consumer about a directory layout that is not theirs.
    """
    if step.get("kind") != PATCH_KIND_NAME:
        return step
    out = dict(step)
    if out.get("path"):
        out["path"] = _portable_patch_ref(str(out["path"]), session_dir)
    out.pop("root", None)
    return out


def _collect_landed_stack(out: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> None:
    """Emit what the lane landed: patches, artifacts, stack action and setup."""
    from hyperloom.orchestrator.enablement.recipe.steps import root_ids_by_path

    kept_patches_raw = _eg(state, "kept_patches")
    if isinstance(kept_patches_raw, list) and kept_patches_raw:
        # Same portability rule as ``kept_rounds`` below, and for the same
        # reason: ``_rel`` falls back to ``str(path)``, so the ``or`` that used
        # to stand here never fired and a patch outside the session travelled as
        # an authoring-host absolute path. Left alone it would also put the same
        # patch in the recipe twice under two different names -- one of them
        # naming a directory the consumer does not have.
        out["kept_patches"] = [_portable_patch_ref(str(p), session_dir) for p in kept_patches_raw]
    # Exported beside the patches, not only fed to the projection: a build linked
    # through a round's durable identity is reachable only from here once the
    # one-shot ``last_specialist_task_id`` has been consumed, which is the normal
    # state by the time a build lands.
    kept_rounds_raw = _eg(state, "kept_rounds")
    if isinstance(kept_rounds_raw, list) and kept_rounds_raw:
        out["kept_rounds"] = [
            {
                # The linkage the fallback joins on, and nothing host-local with
                # it. ``_push_kept_round`` stores authoring-workspace patch paths
                # and raw artifact dicts carrying source and target; copied
                # verbatim they would put absolute paths from this machine into
                # a recipe meant to be replayed on another -- the same reason
                # ``kept_patches`` is relativized and ``kept_artifacts`` reduced
                # to its normalized fields a few lines below.
                "task_id": str(r.get("task_id") or ""),
                # Session-relative where it can be, the bare name otherwise --
                # never the authoring absolute path. ``_push_kept_round`` notes
                # that the replay sources each round's patches from the archive
                # by name, so the name is the whole linkage and the directory it
                # sat in on this machine is not part of it. Checked explicitly
                # rather than through ``_rel``'s falsy branch: ``_rel`` falls
                # back to ``str(path)``, so ``or`` never fires and the absolute
                # path would travel exactly as if nothing had been done.
                "patches": [_portable_patch_ref(str(p), session_dir) for p in (r.get("patches") or [])],
                # ``rel_target`` when the producer recorded one; otherwise the
                # target's own name. Falling back to ``target`` verbatim put an
                # install path from this machine into the recipe, which is the
                # same leak this projection exists to close.
                "artifacts": [
                    str(a.get("rel_target") or "") or Path(str(a.get("target") or "")).name
                    for a in (r.get("artifacts") or [])
                    if isinstance(a, dict) and (a.get("rel_target") or a.get("target"))
                ],
            }
            for r in kept_rounds_raw
            if isinstance(r, dict)
        ]
    kept_artifacts_raw = _eg(state, "kept_artifacts")
    framework_root = str(_eg(state, "framework_root", "") or "")
    if isinstance(kept_artifacts_raw, list) and kept_artifacts_raw:
        root_ids = root_ids_by_path({"roots": _eg(state, "roots")})
        out["kept_artifacts"] = [
            {
                # ``rel_target`` and ``root_id`` are the portable pair, and the
                # only two the rules read. ``target`` is an install path on this
                # machine: shipped verbatim it named a directory the consumer
                # does not have, for a field nothing consults.
                "rel_target": str(a.get("rel_target") or ""),
                "kind": str(a.get("kind") or ""),
                "root_id": root_ids.get(str(a.get("root") or "") or framework_root) or None,
            }
            for a in kept_artifacts_raw
            if isinstance(a, dict) and a.get("target")
        ]
    if framework_root:
        out["framework_root"] = framework_root
    kept_stack_action_raw = _eg(state, "kept_stack_action")
    if isinstance(kept_stack_action_raw, dict) and kept_stack_action_raw:
        out["kept_stack_action"] = _stack_action_summary(kept_stack_action_raw)
    for key, name, project in (
        ("candidate_refs", "candidate_refs", str),
        ("setup_commands", "setup_commands", str),
        ("localization_manifest", "localization_manifest", str),
        ("build_novelty", "build_novelty", str),
    ):
        raw = _eg(state, name)
        if isinstance(raw, list) and raw:
            out[key] = [project(v) for v in raw]
    human_review = _eg(state, "human_review_logged")
    if isinstance(human_review, list) and human_review:
        out["human_review_count"] = len(human_review)
    accepted_cfg = str(_eg(state, "accepted_config_path", "") or "")
    if accepted_cfg:
        out["accepted_config_path"] = _rel(Path(accepted_cfg), session_dir) or accepted_cfg
    setting_script_path = session_dir / "reports" / "enablement" / "enablement_setting.sh"
    # is_file(), not exists(): a directory at that path is not a script a
    # consumer can source, and emitting it would name a replay input that
    # cannot be replayed.
    if setting_script_path.is_file():
        out["setting_script"] = str(
            _rel(setting_script_path, session_dir) or "reports/enablement/enablement_setting.sh"
        )


def _collect_eval_trigger(out: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> None:
    """Emit the eval-origin trigger the round was opened against."""
    out["trigger_kind"] = str(_eg(state, "baseline_eval_kind", "") or "")
    out["observed_accuracy"] = float(_eg(state, "observed_accuracy", 0.0) or 0.0)
    out["accuracy_floor"] = float(_eg(state, "accuracy_floor", 0.0) or 0.0)
    out["observed_task"] = str(_eg(state, "observed_task", "") or "")
    out["observed_metric"] = str(_eg(state, "observed_metric", "") or "")
    out["eval_contract_fingerprint"] = str(_eg(state, "eval_contract_fingerprint", "") or "")
    probe_cfg = str(_eg(state, "probe_config_path", "") or "")
    if probe_cfg:
        out["probe_config_path"] = _rel(Path(probe_cfg), session_dir) or probe_cfg
    evidence = str(_eg(state, "baseline_eval_evidence", "") or "")
    if evidence:
        out["trigger_evidence_excerpt"] = evidence[-_ENABLEMENT_LOG_EXCERPT_CHARS:]


def _collect_runtimes_and_builds(out: dict[str, Any], state: dict[str, Any]) -> None:
    """Emit the runtimes the lane provisioned and the targeted builds it ran."""
    active_runtime_raw = _eg(state, "active_runtime")
    have_active = isinstance(active_runtime_raw, dict) and bool(active_runtime_raw)
    active_root = str(active_runtime_raw.get("venv_root") or "") if have_active else ""
    if have_active:
        out["active_runtime"] = _runtime_summary(active_runtime_raw, promoted=True)
    attempt_runtimes_raw = _eg(state, "attempt_runtimes")
    if isinstance(attempt_runtimes_raw, list) and attempt_runtimes_raw:
        out["attempt_runtimes"] = [
            _runtime_summary(r, promoted=(str(r.get("venv_root") or "") == active_root))
            for r in attempt_runtimes_raw
            if isinstance(r, dict)
        ]
    # The failure classification moved onto the recorder's attempt row with the
    # round rework; a pre-rework state document still carries it here.
    failure_kind = str(_eg(state, "failure_kind", "") or "")
    if failure_kind:
        out["failure_kind"] = failure_kind
    build_manifest_raw = _eg(state, "build_manifest")
    if isinstance(build_manifest_raw, list) and build_manifest_raw:
        build_attempts = [
            build_attempt_summary(e) for e in build_manifest_raw if isinstance(e, dict) and e.get("ok") is not None
        ]
        if build_attempts:
            out["build_attempts"] = build_attempts
            out["build_attempt_count"] = len(build_attempts)
    last_build_failure_raw = _eg(state, "last_build_failure")
    if isinstance(last_build_failure_raw, dict) and last_build_failure_raw:
        out["last_build_failure"] = {
            "failure_class": str(last_build_failure_raw.get("failure_class") or ""),
            "failure_summary": str(last_build_failure_raw.get("failure_summary") or ""),
        }


def collect_enablement(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the enablement replay contract from the durable round state.

    The lane's own account of what it did is recorded at author time; what is
    projected here is the part no author-time site can state -- the ordered
    ``recipe_steps`` a consumer would replay and the ``replay_sufficiency``
    verdict over them, judged against the evidence the session actually
    captured. The surrounding lane and landed-stack keys are kept because the
    verdict is computed over the emitted section, not over raw state.
    """
    out = _enablement_lane_status(state)
    if out is None:
        return {}
    _collect_round_identity(out, state)
    _collect_landed_stack(out, state, session_dir=session_dir)
    if out["origin"] == "eval":
        _collect_eval_trigger(out, state, session_dir=session_dir)
    _collect_runtimes_and_builds(out, state)
    _collect_recipe(out, state, session_dir=session_dir)
    return out
