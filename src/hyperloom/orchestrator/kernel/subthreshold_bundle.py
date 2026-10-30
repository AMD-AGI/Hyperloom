# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel wins the AgentX P50 threshold alone refused, deferred to be measured together once."""

from __future__ import annotations

from typing import Any, Mapping

from .controller_patch_integration import KeepSession
from .controller_publication import ControllerPatchPublication

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
