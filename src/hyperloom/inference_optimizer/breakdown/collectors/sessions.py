# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.timeutil import iso_z, now_iso

from ._common import _to_int
from ..session_package import deliverable


log = logging.getLogger(__name__)


def _detect_image_for_session(manifest: dict[str, Any]) -> str | None:
    """Resolve the container image for ``collect_session``."""
    manifest_image = manifest.get("image") if isinstance(manifest, dict) else None
    if isinstance(manifest_image, str) and manifest_image.strip():
        return manifest_image.strip()
    for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    for marker in ("/etc/podinfo/image", "/etc/hyperloom-image"):
        try:
            p = Path(marker)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    return txt
        except OSError:
            continue
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            for line in cgroup.read_text(encoding="utf-8", errors="replace").splitlines():
                if "docker" not in line and "containerd" not in line:
                    continue
                m = re.search(r"([0-9a-f]{12,64})", line)
                if m:
                    return f"unknown@{m.group(1)[:12]}"
    except OSError as exc:
        # /proc/1/cgroup may be unreadable; fall through to None.
        log.debug("cgroup-based image detection failed: %r", exc)
    return None


def _leg_start_ts(state: dict[str, Any], start_ts: str) -> str:
    """When the session's current run leg began."""
    resumed_ts = str(state.get("resumed_ts") or "")
    dated = [(to_unix(ts), ts) for ts in (start_ts, resumed_ts)]
    parseable = [(at, ts) for at, ts in dated if at is not None]
    if not parseable:
        return start_ts
    return max(parseable)[1]


def _close_phase_stop_reason(state: dict[str, Any], *, leg_start_ts: str) -> tuple[str, str]:
    """Recover terminal reason/time from the current leg's CLOSE transition (next-best when ``state.stop_reason`` wasn't mirrored)."""
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return "", ""
    leg_start = to_unix(leg_start_ts)
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        if str(row.get("to_phase") or "").strip().upper() != "CLOSE":
            continue
        reason = str(row.get("reason") or row.get("stop_reason") or row.get("exit_reason") or "").strip()
        ts = str(row.get("ts") or row.get("entered_ts") or "").strip()
        closed_at = to_unix(ts)
        if leg_start is not None and closed_at is not None and closed_at < leg_start:
            continue
        return reason, ts
    return "", ""


def _first_recorded_end(*candidates: Any) -> str:
    """The first candidate that reads as a timestamp, canonicalised to ``...Z``."""
    for value in candidates:
        if to_unix(value) is not None:
            return iso_z(value)
    return ""


def _session_has_ended(stop_reason: Any) -> bool:
    """Whether a stop reason marks the session as no longer running."""
    return bool(str(stop_reason or "").strip())


def _measured_duration_seconds(start_ts: Any, ended_at_utc: Any, stop_reason: Any) -> int | None:
    """Seconds the session ran, or ``None`` when no window can be established."""
    start = to_unix(start_ts)
    if start is None:
        return None
    end = to_unix(ended_at_utc)
    if end is None and not _session_has_ended(stop_reason):
        end = datetime.now(timezone.utc).timestamp()
    if end is None or end <= start:
        return None
    return int(round(end - start))


def session_elapsed_minutes(session_section: dict[str, Any]) -> float:
    """Wall-clock minutes of the leg described by a resolved ``session`` section."""
    duration_s = _measured_duration_seconds(
        session_section.get("start_ts") or session_section.get("created_at_utc"),
        session_section.get("ended_at_utc"),
        session_section.get("stop_reason"),
    )
    return round(duration_s / 60.0, 2) if duration_s is not None else 0.0


def _should_use_close_stop_reason(stop_reason: str, close_stop_reason: str) -> bool:
    """Decide whether the CLOSE-phase stop reason should override the session's."""
    if not close_stop_reason:
        return False
    if not stop_reason:
        return True
    return stop_reason == "time_exhausted" and close_stop_reason != "time_exhausted"


# Session metadata
def _collect_recovery(state: dict[str, Any]) -> dict[str, Any]:
    """Project SharedState's crash / interruption / resume signals."""
    crash_count = _to_int(state.get("crash_count")) or 0
    crash_ts_iso: list[str] = []
    raw_ts = state.get("crash_timestamps")
    if isinstance(raw_ts, list):
        for t in raw_ts:
            try:
                crash_ts_iso.append(datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat())
            except (TypeError, ValueError, OSError, OverflowError):
                continue

    last_exc: dict[str, Any] | None = None
    lte = state.get("last_tick_exception")
    if isinstance(lte, dict) and lte:
        # Drop the large traceback; keep the compact postmortem header.
        last_exc = {
            "tick": lte.get("tick"),
            "ts": lte.get("ts"),
            "stage": lte.get("stage"),
            "agent": lte.get("agent"),
            "type": lte.get("type"),
            "message": (str(lte.get("message") or "")[:500] or None),
        }

    resume_pending = bool(state.get("resume_pending_revalidation"))
    degraded = bool(state.get("degraded_mode"))
    recovered = bool(crash_count > 0 or crash_ts_iso or resume_pending or last_exc)
    return {
        "recovered": recovered,
        "crash_count": crash_count,
        "crash_timestamps": crash_ts_iso,
        "degraded_mode": degraded,
        "resume_pending_revalidation": resume_pending,
        "last_tick_exception": last_exc,
    }


def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the session-identification + lifecycle section."""
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state, leg_start_ts=_leg_start_ts(state, start_ts))
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if _session_has_ended(stop_reason):
        # ``stop_ts`` is stamped once, when the reason is written, so a re-export of a finished session keeps
        # reporting the same end.
        ended_at_utc = _first_recorded_end(state.get("stop_ts"), close_ts) or now_iso(timespec="seconds")
    image = _detect_image_for_session(manifest)
    if image is None:
        warnings.append("image: not configured (set HYPERLOOM_IMAGE env var)")
    section = {
        "session_id": str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id": manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id": manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc": manifest.get("created_at_utc") or start_ts,
        "start_ts": start_ts,
        "ended_at_utc": ended_at_utc,
        "stop_reason": stop_reason,
        "max_minutes": int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes": 0.0,
        "host": str(manifest.get("host") or ""),
        "image": image,
        "code_revision": str(manifest.get("code_revision") or ""),
        "pid": int(manifest.get("pid") or 0),
        "session_dir": str(session_dir),
        # USER_DATA_PATH root (the operator-chosen workspace base).
        "user_data_path": str(
            manifest.get("user_data_path") or state.get("user_data_path") or os.environ.get("USER_DATA_PATH") or ""
        ),
        "tick_count": int(state.get("tick") or 0),
        # Crash / interruption / resume history.
        "recovery": _collect_recovery(state),
    }
    section["elapsed_minutes"] = session_elapsed_minutes(section)
    return section


# Workload
def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the workload-description section."""
    wl = manifest.get("workload") or {}
    return {
        "framework_name": str(state.get("framework") or manifest.get("framework") or ""),
        "framework_version": str(manifest.get("framework_version") or ""),
        "model_name": str(state.get("model_name") or manifest.get("model_name") or ""),
        "model_path": str(state.get("model_path") or manifest.get("model_path") or ""),
        "model_class": str(state.get("model_class") or ""),
        "gpu_type": str(state.get("gpu_type") or manifest.get("gpu_type") or ""),
        "tp": _to_int(manifest.get("tp")),
        "conc": _to_int(wl.get("conc")),
        "isl": _to_int(wl.get("isl")),
        "osl": _to_int(wl.get("osl")),
        "max_model_len": _to_int(wl.get("max_model_len")),
        "precision": str(wl.get("precision") or ""),
        "objective": dict(manifest.get("objective") or {"kind": "time_only", "value": None}),
    }


# Model basics
def collect_model_info(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the ``model_info`` section (state.model_info passthrough)."""
    info = state.get("model_info")
    return dict(info) if isinstance(info, dict) else {}


# --------------------------------------------------------------------------
# Enablement: the replay contract and the verdict over it.
#
# The rest of the enablement read side was retired in favour of author-time
# recording: the lane writes what it did through ``recorder/enablement_event``
# at the moment it does it. What survives here is the projection of the
# durable ``EnablementRound`` state onto the ordered ``recipe_steps`` array and
# the ``replay_sufficiency`` decision over it -- the fail-closed judgement of
# whether a consumer outside this session receives enough to replay the
# accepted stack.
#
# That judgement is a statement about the finished *stack*, not about what one
# round did, so it is the one enablement fact no per-round ``record_*`` can
# make. It is computed here and published by
# ``recorder/enablement_event.finish`` at the lane's terminal, which is the
# first moment the accepted stack is complete; this module stays its
# implementation so there is exactly one projection and one verdict rather than
# a read-side and an author-side that can disagree. It must not silently
# disappear either way: an absent ``replay_sufficiency`` is read as
# insufficient by contract, so dropping the producer would report every session
# as uncertifiable.
#
# The round lifecycle counters this used to report -- ``attempts``,
# ``inflight_task_id``, ``stall_streak`` -- are no longer fields of the
# enablement state; the durable round ledger owns them and the recorder
# reports them. They are still read here because a state document written
# before that rework still carries them, but they are only emitted when the
# document actually has them. Synthesising a ``0`` would report every session
# after the rework as a lane that never dispatched.
# --------------------------------------------------------------------------

_ENABLEMENT_LOG_EXCERPT_CHARS = 2000


#: Distinguishes "this state never recorded the field" from a recorded ``None``.
_ABSENT = object()


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


def _build_attempt_summary(manifest_entry: dict[str, Any]) -> dict[str, Any]:
    """Project a ``BuildResult.to_state()`` entry onto a build-attempt summary.

    Injected into :func:`build_recipe_steps`, which reads ``component``,
    ``ref``, ``gpu_arch`` and ``max_jobs`` off the result. Renaming one of
    those keys does not raise -- it silently empties the build step and the
    verdict then reports ``build_inputs_incomplete``.
    """
    action = manifest_entry.get("action") or {}
    installed = manifest_entry.get("installed_versions") or {}
    probes = manifest_entry.get("build_probes") or []
    return {
        "component": str(action.get("component") or manifest_entry.get("component") or ""),
        "ref": str(
            installed.get("aiter_ref")
            or installed.get("vllm_ref")
            or installed.get("sgl_kernel_ref")
            or action.get("ref")
            or ""
        ),
        "gpu_arch": str(installed.get("arch") or action.get("gpu_arch") or ""),
        "max_jobs": int(action.get("max_jobs") or 0),
        "ok": bool(manifest_entry.get("ok")),
        "failure_class": str(manifest_entry.get("failure_class") or "ok"),
        "failure_summary": str(manifest_entry.get("failure_summary") or ""),
        "installed_versions": {str(k): str(v) for k, v in installed.items()} if isinstance(installed, dict) else {},
        "build_probes": [str(p) for p in probes[:8]],
        "build_log_path": str(manifest_entry.get("build_log_path") or ""),
        "attempt_root": str(manifest_entry.get("attempt_root") or ""),
    }


#: ``EnablementRound`` fields the recipe projection and its decision read.
_RECIPE_STATE_FIELDS: tuple[str, ...] = (
    "active_runtime",
    "accepted_config",
    "accepted_config_source",
    "accepted_stack_targets",
    "base_sha",
    "build_extensions_not_carried",
    "build_manifest",
    "environment_closure",
    "framework_root",
    "installed_versions_at_keep",
    "kept_artifacts",
    "kept_patches",
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
    steps = build_recipe_steps(enablement, attempt_summary=_build_attempt_summary)
    if steps:
        out["recipe_steps"] = steps
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


def _collect_landed_stack(out: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> None:
    """Emit what the lane landed: patches, artifacts, stack action and setup."""
    from hyperloom.orchestrator.enablement.recipe.steps import root_ids_by_path

    kept_patches_raw = _eg(state, "kept_patches")
    if isinstance(kept_patches_raw, list) and kept_patches_raw:
        out["kept_patches"] = [_rel(Path(str(p)), session_dir) or str(p) for p in kept_patches_raw]
    kept_artifacts_raw = _eg(state, "kept_artifacts")
    framework_root = str(_eg(state, "framework_root", "") or "")
    if isinstance(kept_artifacts_raw, list) and kept_artifacts_raw:
        root_ids = root_ids_by_path({"roots": _eg(state, "roots")})
        out["kept_artifacts"] = [
            {
                "target": str(a.get("target") or ""),
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
            _build_attempt_summary(e) for e in build_manifest_raw if isinstance(e, dict) and e.get("ok") is not None
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
