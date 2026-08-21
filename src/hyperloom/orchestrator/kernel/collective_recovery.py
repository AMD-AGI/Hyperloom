# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resume a collective integration that was interrupted mid-patch.

The collective lane edits an editable-install repository in place, so a crash
between ``apply`` and ``finalize`` leaves the repository holding a patch that no
session owns. This module reads the apply checkpoint and its backup manifest and
decides whether the interrupted integration can be resumed, must be reverted, or
needs an operator. Everything here is a pure function of the checkpoint on disk
plus the recorded campaign, so the phase only sequences the outcome.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple


#: Manifest states an interrupted apply can still be resumed from.
_RESUMABLE_MANIFEST_STATES = frozenset({"applied"})
#: Manifest states that are terminal but not resumable, so they get reverted.
_REVERTIBLE_MANIFEST_STATES = frozenset({"applied", "failed", "prepared", "reverted_partial"})
_FINALIZED_MANIFEST_STATES = frozenset({"finalized", "finalized_partial"})
_REVERTED_MANIFEST_STATES = frozenset({"reverted", "reverted_partial"})


class IntegrationInputs(NamedTuple):
    """Validated fields a collective integration needs from its campaign."""

    patch: str
    target_file: str
    kernel_repo: str
    integration_id: str
    extra_envs: dict[str, str]


class RecoveredApply(NamedTuple):
    """Outcome of inspecting an interrupted apply.

    ``preapplied`` is a still-usable apply result the integrate handler can
    adopt instead of re-applying. ``integ`` short-circuits the integration with
    a terminal result. ``uncertain`` marks states an operator must confirm, and
    suppresses the ``complete`` integration status.
    """

    preapplied: dict[str, Any] | None
    integ: dict[str, Any] | None
    uncertain: bool


def patch_lifecycle_complete(result: Any) -> bool:
    """Return whether patch cleanup reached a terminal state."""
    return isinstance(result, dict) and result.get("status") in {"ok", "partial"}


def load_apply_checkpoint(
    checkpoint: Path,
    backup_root: Path,
) -> tuple[dict[str, Any], str]:
    """Load a trusted collective apply checkpoint and manifest state.

    The manifest path is required to resolve under ``backup_root`` so a
    tampered or stale checkpoint cannot point the revert at an unrelated tree.
    """
    recovered = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(recovered, dict):
        raise ValueError("Collective apply checkpoint must be a mapping")
    manifest_path = Path(str(recovered.get("manifest_path") or ""))
    if not manifest_path.is_file():
        raise ValueError("Collective apply manifest does not exist")
    manifest_path.resolve().relative_to(backup_root.resolve())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Collective apply manifest must be a mapping")
    return (
        {**recovered, "manifest_path": str(manifest_path)},
        str(manifest.get("status") or ""),
    )


def validate_integration_inputs(result: dict, state: Any) -> IntegrationInputs:
    """Validate the campaign and the session state an integration will mutate.

    Raises rather than degrading: every field here is required to revert
    cleanly, so continuing past a malformed one risks leaving the repository
    patched with no way back.
    """
    if not isinstance(result, dict):
        raise TypeError("Collective integration input must be a mapping")
    raw = {
        "patch": result.get("patch"),
        "target_file": result.get("source_file") or result.get("target_file"),
        "kernel_repo": result.get("kernel_repo"),
        "integration_id": result.get("integration_id"),
    }
    for field, value in raw.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Collective integration {field} must be a string")
    integration_id = (raw["integration_id"] or "").strip()
    if not integration_id:
        raise ValueError("Collective integration is missing integration_id")
    if not isinstance(state.optimization_stack, list):
        raise ValueError("optimization_stack must be a list")
    if not isinstance(state.gain_per_stack_entry, list):
        raise ValueError("gain_per_stack_entry must be a list")
    if not isinstance(state.current_best, dict):
        raise ValueError("current_best must be a mapping")
    extra_envs: dict[str, str] = {}
    raw_envs = state.current_best.get("extra_envs")
    if raw_envs is not None:
        if not isinstance(raw_envs, Mapping):
            raise ValueError("current_best.extra_envs must be a mapping")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw_envs.items()):
            raise ValueError("current_best.extra_envs must contain strings")
        extra_envs = dict(raw_envs)
    return IntegrationInputs(
        patch=(raw["patch"] or "").strip(),
        target_file=(raw["target_file"] or "").strip(),
        kernel_repo=(raw["kernel_repo"] or "").strip(),
        integration_id=integration_id,
        extra_envs=extra_envs,
    )


def _needs_review(error_class: str, error: str, patch: str, target: str) -> dict[str, Any]:
    """Build a terminal result that parks the integration for an operator."""
    return {
        "status": "failed",
        "decision": "NEEDS_REVIEW",
        "error_class": error_class,
        "error": error,
        "patch_path": patch,
        "target_file": target,
    }


def _read_recovery_source(
    checkpoint: Path,
    backup_root: Path,
    *,
    patch: str,
    target_file: str,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    """Return ``(recovered_apply, manifest_status, terminal_result)``.

    Prefers the checkpoint the applier writes; falls back to a lone manifest
    left in the backup tree when the process died before the checkpoint landed.
    """
    if checkpoint.is_file():
        try:
            recovered, status = load_apply_checkpoint(checkpoint, backup_root)
        except Exception as exc:  # noqa: BLE001 - any read fault parks the lane
            return (
                None,
                "",
                _needs_review(
                    "collective_apply_checkpoint_invalid",
                    repr(exc),
                    patch,
                    target_file,
                ),
            )
        return recovered, status, None
    if not backup_root.is_dir():
        return None, "", None
    manifests = sorted(backup_root.glob("**/manifest.json"))
    if not manifests:
        return None, "", None
    if len(manifests) > 1:
        return (
            None,
            "",
            _needs_review(
                "collective_apply_manifest_ambiguous",
                f"Collective recovery found multiple apply manifests under {backup_root}",
                patch,
                target_file,
            ),
        )
    try:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Collective apply manifest must be a mapping")
    except Exception as exc:  # noqa: BLE001 - any read fault parks the lane
        return (
            None,
            "",
            _needs_review(
                "collective_apply_checkpoint_invalid",
                repr(exc),
                patch,
                target_file,
            ),
        )
    # The manifest's own status is the only evidence here; stamping an
    # apply_result "ok" over it would report an outcome nothing measured.
    return (
        {**manifest, "manifest_path": str(manifests[0])},
        str(manifest.get("status") or ""),
        None,
    )


async def recover_apply_state(
    result: dict,
    *,
    checkpoint: Path,
    backup_root: Path,
    patch: str,
    target_file: str,
) -> RecoveredApply:
    """Decide how an interrupted collective apply should continue.

    ``result['integration_recovery_action']`` records what the previous session
    still owed, so a resumed run replays that step rather than re-deriving it
    from the manifest alone.
    """
    from .request_handlers import (
        _maybe_finalize_kernel_patch,
        _maybe_revert_kernel_patch,
    )

    recovered, manifest_status, terminal = _read_recovery_source(
        checkpoint,
        backup_root,
        patch=patch,
        target_file=target_file,
    )
    if terminal is not None:
        return RecoveredApply(None, terminal, True)
    if recovered is None:
        return RecoveredApply(None, None, False)

    recovery_action = str(result.get("integration_recovery_action") or "").strip()

    if recovery_action == "revert":
        if manifest_status in _REVERTED_MANIFEST_STATES:
            revert_result = {
                "status": "ok" if manifest_status == "reverted" else "partial",
                "reason": "manifest already reverted",
            }
        else:
            revert_result = await asyncio.to_thread(
                _maybe_revert_kernel_patch,
                recovered,
            )
        return RecoveredApply(
            None,
            {
                "status": "failed",
                "decision": "REVERT",
                "error_class": "collective_recovery_revert",
                "error": "Collective integration resumed pending revert",
                "patch_path": patch,
                "target_file": target_file,
                "apply_result": recovered,
                "revert_result": revert_result,
            },
            False,
        )

    resumable_finalize = (
        recovery_action == "finalize"
        and str(result.get("integration_decision") or "").upper() == "KEEP"
        and manifest_status in (_RESUMABLE_MANIFEST_STATES | _FINALIZED_MANIFEST_STATES)
    )
    if resumable_finalize:
        if manifest_status in _FINALIZED_MANIFEST_STATES:
            finalize_result = {
                "status": "ok" if manifest_status == "finalized" else "partial",
                "reason": "manifest already finalized",
            }
        else:
            finalize_result = await asyncio.to_thread(
                _maybe_finalize_kernel_patch,
                recovered,
            )
        finalize_complete = patch_lifecycle_complete(finalize_result)
        return RecoveredApply(
            None,
            {
                "status": str(result.get("integration_result_status") or "ok"),
                "decision": "KEEP",
                "gain_pct": result.get("integration_gain_pct"),
                "base_tput": result.get("integration_base_tput"),
                "new_tput": result.get("integration_new_tput"),
                "workspace": result.get("integration_workspace"),
                "report_path": result.get("integration_report_path"),
                "patch_path": patch,
                "target_file": target_file,
                "apply_result": recovered,
                "finalize_result": finalize_result,
                "integration_status": ("complete" if finalize_complete else "recovery_required"),
                "integration_recovery_action": ("" if finalize_complete else "finalize"),
            },
            False,
        )

    # Two state machines meet here: a checkpoint carries the applier's own
    # apply_result ("ok"), a lone manifest carries only the manifest state.
    if manifest_status in _RESUMABLE_MANIFEST_STATES and str(recovered.get("status") or "") in {"ok", manifest_status}:
        return RecoveredApply(recovered, None, False)

    if manifest_status == "reverted":
        return RecoveredApply(
            None,
            {
                "status": "failed",
                "decision": "REVERT",
                "error_class": "collective_apply_already_reverted",
                "error": "Collective apply manifest was already reverted",
                "patch_path": patch,
                "target_file": target_file,
                "apply_result": recovered,
                "revert_result": {
                    "status": "ok",
                    "reason": "manifest already reverted",
                },
            },
            False,
        )

    if manifest_status in _REVERTIBLE_MANIFEST_STATES:
        revert_result = await asyncio.to_thread(
            _maybe_revert_kernel_patch,
            recovered,
        )
        return RecoveredApply(
            None,
            {
                "status": "failed",
                "decision": "REVERT",
                "error_class": "collective_apply_not_resumable",
                "error": (f"Collective apply manifest is not resumable: {manifest_status or 'unknown'}"),
                "patch_path": patch,
                "target_file": target_file,
                "apply_result": recovered,
                "revert_result": revert_result,
            },
            False,
        )

    integ = _needs_review(
        "collective_apply_not_resumable",
        (f"Collective apply manifest has unsupported state: {manifest_status or 'unknown'}"),
        patch,
        target_file,
    )
    integ["apply_result"] = recovered
    return RecoveredApply(None, integ, True)


__all__ = [
    "IntegrationInputs",
    "RecoveredApply",
    "load_apply_checkpoint",
    "patch_lifecycle_complete",
    "recover_apply_state",
    "validate_integration_inputs",
]
