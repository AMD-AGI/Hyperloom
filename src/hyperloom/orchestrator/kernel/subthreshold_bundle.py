# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel wins the AgentX P50 threshold alone refused, deferred to be measured together once."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args

from ..actions.executors._patch_snapshot import _git_commit_kept, _patch_touched_paths
from ..actions.stop_attribution import StoppedByTheRun
from .controller_patch_integration import (
    KeepSession,
    _committed_source_layer,
    _git_output,
    _head_commit,
    _revert_note,
)
from .controller_publication import ControllerPatchPublication
from .patch_conflict_merge import LandingContext, apply_patch_resolving_conflicts

log = logging.getLogger(__name__)

_BUNDLE_SOURCE = "kernel_bundle"
_ENV_KIND = "env"
_PATCH_KIND = "patch"


def _subthreshold_win(result: Mapping[str, Any]) -> bool:
    """A REVERT whose only failed gate was the P50 gain threshold, on a positive gain."""
    return (
        str(result.get("decision") or "").upper() == "REVERT"
        and float(result.get("gain_pct") or 0.0) > 0
        and result.get("refused_by") == "p50_gain_below_threshold"
        and not result.get("veto_reason")
    )


def defer_env_win(
    state: Any, verdict: Mapping[str, Any], *, tuner: str, envs: dict[str, str], server_args: str
) -> None:
    """Keep a GEMM tuning candidate's envs for the bundle when its verdict is a sub-threshold win."""
    if not _subthreshold_win(verdict):
        return
    state.kernel_bundle_pending.append(
        {
            "kind": _ENV_KIND,
            "id": f"gemm_tune_{tuner}",
            "envs": dict(envs),
            "server_args": server_args,
            "gain_pct": float(verdict["gain_pct"]),
            "macro_cycle": int(state.macro_cycle or 0),
        }
    )


def defer_patch_win(
    session: KeepSession, publication: ControllerPatchPublication, validation: Mapping[str, Any]
) -> None:
    """Keep a Controller patch for the bundle when its validation is a sub-threshold win."""
    if not _subthreshold_win(validation):
        return
    state = session.shared_state
    state.kernel_bundle_pending.append(
        {
            "kind": _PATCH_KIND,
            "id": publication.operator_id,
            "repo_root": str(publication.repo_root),
            "patch_path": str(publication.patch_path),
            "kernel_path": publication.kernel_path,
            "operator_name": publication.operator_name,
            "gain_pct": float(validation["gain_pct"]),
            "macro_cycle": int(state.macro_cycle or 0),
        }
    )
    state.save(session.session_dir)


async def _apply_member(member: Mapping[str, Any], repo_root: str) -> str:
    """Apply one patch member into the bundle's repository; why it was left out, or ``""`` once applied."""
    # One stack entry carries one source layer, so every patch member lands in one repository.
    if member["repo_root"] != repo_root:
        return f"lands in {member['repo_root']}, not in the bundle's repository {repo_root}"
    repo, patch_path = Path(repo_root), Path(member["patch_path"])
    touched = _patch_touched_paths(repo, [patch_path])
    # A failed apply restores its paths to HEAD, which would drop an earlier member's edit on them.
    if _git_output(repo, "status", "--porcelain", "--untracked-files=no", "--", *touched):
        return "uncommitted changes on the paths it modifies"
    merge = await apply_patch_resolving_conflicts(
        repo,
        patch_path,
        LandingContext(operator_id=member["id"], intent=f"{member['operator_name']} in {member['kernel_path']}"),
    )
    return "" if merge.applied else merge.error + merge.note()


def _bundle_envs(bundle: list[dict[str, Any]]) -> dict[str, str]:
    return {key: value for member in bundle if member["kind"] == _ENV_KIND for key, value in member["envs"].items()}


def _bundle_payload(state: Any, kernel_id: str, bundle: list[dict[str, Any]]) -> dict[str, Any]:
    server_args = " ".join(member["server_args"] for member in bundle if member["kind"] == _ENV_KIND)
    return {
        "task_id": kernel_id,
        "kernel_id": kernel_id,
        "source": _BUNDLE_SOURCE,
        "base_tput": float(state.current_best.get("tput") or 0.0),
        "model_path": resolve_session_model_path(state_model_path=str(state.model_path or "")),
        "extra_server_args": dedupe_extra_server_args(server_args),
        "extra_envs": _bundle_envs(bundle),
        "keep_threshold_pct": 1.0,
        "mode": "env_only",
    }


async def _keep_bundle(session: KeepSession, kernel_id: str, bundle: list[dict[str, Any]], verdict: dict) -> str:
    """Commit the patch members as one KEEP and record the bundle as one stack entry; why it did not land, or ``""``."""
    patches = [member for member in bundle if member["kind"] == _PATCH_KIND]
    layer: dict[str, Any] = {}
    if patches:
        repo = Path(patches[0]["repo_root"])
        head_before = _head_commit(repo)
        touched = _patch_touched_paths(repo, [Path(member["patch_path"]) for member in patches])
        message = f"hyperloom: keep sub-threshold kernel bundle {kernel_id}: {', '.join(m['id'] for m in patches)}"
        committed, note = _git_commit_kept(repo, message, touched)
        keep_commit = _head_commit(repo)
        if not committed or not keep_commit or keep_commit == head_before:
            return note or "git commit did not advance HEAD"
        framework = str(session.shared_state.framework or "")
        layer = {
            **_committed_source_layer(repo, keep_commit, kernel_id, session.session_dir, framework),
            "scope": "source_patch",
            "base_sha": head_before,
            "keep_commit": keep_commit,
        }
    record = {
        **verdict,
        "kernel_id": kernel_id,
        "stack_kernel_ids": [member["id"] for member in bundle],
        "source": _BUNDLE_SOURCE,
        "backend": "forge",
        "engine": _BUNDLE_SOURCE,
        **layer,
    }
    try:
        await session.record_keep(record)
        session.shared_state.save(session.session_dir)
    except Exception as error:  # isolation boundary: the committed bundle's KEEP recording
        log.warning("kernel bundle %s committed but recording it failed: %s", kernel_id, error, exc_info=True)
    return ""


def _take_cycle_members(session: KeepSession) -> list[dict[str, Any]]:
    """Consume the pending bundle and return the current macro cycle's members."""
    state = session.shared_state
    cycle = int(state.macro_cycle or 0)
    members = [entry for entry in state.kernel_bundle_pending if entry["macro_cycle"] == cycle]
    state.kernel_bundle_pending = []
    state.save(session.session_dir)
    return members


async def _apply_patch_members(members: list[dict[str, Any]], kernel_id: str, bundle: list[dict[str, Any]]) -> None:
    """Apply the patch members in their stored order, appending each one that landed to ``bundle``."""
    repo_root = next((member["repo_root"] for member in members if member["kind"] == _PATCH_KIND), "")
    for member in members:
        if member["kind"] != _PATCH_KIND:
            continue
        if dropped := await _apply_member(member, repo_root):
            log.warning("%s: dropping %s: %s", kernel_id, member["id"], dropped)
        else:
            bundle.append(member)


async def _revert_bundle(bundle: list[dict[str, Any]], envs: dict[str, str], jit_backup_dir: Path) -> str:
    """Revert every applied patch member and invalidate the env members' serving modules; the revert notes."""
    from ..actions.executors._aiter_jit import drop_serving_so_for_envs

    notes = "".join(
        _revert_note(Path(member["repo_root"]), Path(member["patch_path"]))
        for member in reversed(bundle)
        if member["kind"] == _PATCH_KIND
    )
    if envs:
        await asyncio.to_thread(drop_serving_so_for_envs, envs, backup_dir=jit_backup_dir)
    return notes


async def validate_subthreshold_bundle(
    session: KeepSession,
    *,
    measure: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    round_stop: Callable[[], StoppedByTheRun | None],
) -> None:
    """Measure this macro cycle's sub-threshold wins together in one E2E round.

    Args:
        session: The session the KEEP is recorded into; its pending bundle is consumed either way.
        measure: Runs one in-phase integrate round for a payload.
        round_stop: Why the phase cannot fit another measured round, or ``None`` when it can.
    """
    from ..actions.executors._aiter_jit import prepare_serving_so_for_csvs

    state = session.shared_state
    kernel_id = f"kernel_bundle_c{int(state.macro_cycle or 0)}"
    members = _take_cycle_members(session)
    stop = round_stop() if len(members) >= 2 else None
    if len(members) < 2 or stop is not None:
        log.info("%s: not measured (%s)", kernel_id, stop.never_started if stop else f"{len(members)} member(s)")
        return
    envs = _bundle_envs(members)
    jit_backup_dir = session.session_dir / "runs" / "aiter_jit_backup"
    bundle = [member for member in members if member["kind"] == _ENV_KIND]
    verdict: dict[str, Any] = {}
    try:
        await _apply_patch_members(members, kernel_id, bundle)
        if len(bundle) >= 2:
            if envs:
                await asyncio.to_thread(prepare_serving_so_for_csvs, envs, backup_dir=jit_backup_dir)
            verdict = await measure(_bundle_payload(state, kernel_id, bundle))
    except Exception as error:  # isolation boundary: the bundle's apply and E2E round, settled as a REVERT
        log.warning("%s: bundle validation raised: %s", kernel_id, error, exc_info=True)
    kept = str(verdict.get("status") or "").lower() == "ok" and str(verdict.get("decision")).upper() == "KEEP"
    not_landed = await _keep_bundle(session, kernel_id, bundle, verdict) if kept else "E2E did not KEEP"
    if not_landed:
        not_landed += await _revert_bundle(bundle, envs, jit_backup_dir)
    log.info(
        "%s: %s members=%s gain=%.2f%% %s",
        kernel_id,
        "REVERT" if not_landed else "KEEP",
        [member["id"] for member in bundle],
        float(verdict.get("gain_pct") or 0.0),
        not_landed,
    )
