# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The result documents one integration attempt returns, and the values they are built from."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.common.timeutil import now_iso

from . import _framework_switch_manifest as _switch_manifest
from ._apply_feedback import ApplyFeedback
from ._enablement_config import _parse_framework_switches
from ._grid_runner import session_grid_bounds
from ._integrate_attempt import IntegrateAttempt
from ._integrate_refusals import _FIX_ACCURACY_UNAVAILABLE, _derive_lane, _preflight_missing_targets
from ._patch_artifacts import _ArtifactSpec, _resolve_artifact_specs, _validate_aiter_gemm_artifacts
from ._patch_sources import _read_patch_texts, _resolve_patch_paths, resolved_explicit_root

log = logging.getLogger(__name__)


_now_iso = functools.partial(now_iso, "auto")


@dataclass(frozen=True)
class _KbLesson:
    """The outcome and measured deltas one framework KB record stores."""

    outcome: str
    tps_delta_pct: float
    accuracy_delta_pct: float | None = None
    config_fingerprint: str = ""


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

    def lesson(self, outcome: str) -> _KbLesson:
        """The KB record of this measurement under ``outcome``."""
        return _KbLesson(outcome, float(self.delta_pct or 0.0), self.acc_delta_pct, self.cfg_fingerprint)

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

    @classmethod
    def resolve(cls, attempt: IntegrateAttempt, params: dict[str, Any]) -> "_Deliverable":
        """Resolve everything this round delivers, from the task and from the specialist's own output."""
        specialist_workspace = attempt.specialist_workspace
        assert specialist_workspace is not None
        done_payload = attempt.done_payload
        explicit_patches = params.get("patches") or None
        patch_paths = _resolve_patch_paths(
            specialist_workspace=specialist_workspace,
            explicit_patches=(list(explicit_patches) if isinstance(explicit_patches, list) else None),
            done_payload=done_payload,
        )
        config_changes = dict(params.get("config_changes") or {})
        recorded = (done_payload or {}).get("config_changes")
        if not config_changes and isinstance(recorded, dict):
            config_changes = {str(key): str(value) for key, value in recorded.items()}
        raw_envs = params.get("extra_envs")
        extra_envs, unsafe = filter_untrusted_env_mapping(
            {**config_changes, **(raw_envs if isinstance(raw_envs, dict) else {})},
            allow_predicate=is_allowed_variant_env_key,
        )
        dropped = sorted(unsafe)
        if dropped:
            log.warning("integrate_patch: dropping unsafe env override keys: %s", ", ".join(dropped))
        switch_manifest = _parse_framework_switches(params=params, done_payload=done_payload, patch_paths=patch_paths)
        extra_envs.update(_switch_manifest.switch_env(switch_manifest))
        explicit_artifacts = params.get("artifacts")
        artifact_specs, artifact_resolve_errors = _resolve_artifact_specs(
            specialist_workspace=specialist_workspace,
            explicit_artifacts=(list(explicit_artifacts) if isinstance(explicit_artifacts, list) else None),
            done_payload=done_payload,
        )
        raw_args = params.get("extra_server_args")
        gpu_type = str(
            getattr(attempt.shared_state, "gpu_type", "")
            or params.get("gpu_type")
            or params.get("target_platform")
            or ""
        ).strip()
        return cls(
            patch_paths=patch_paths,
            config_changes=config_changes,
            extra_server_args=(raw_args if isinstance(raw_args, str) and raw_args.strip() else ""),
            extra_envs=extra_envs,
            dropped_env_overrides=dropped,
            switch_manifest=switch_manifest,
            artifact_specs=artifact_specs,
            artifact_resolve_errors=artifact_resolve_errors,
            artifact_runtime_errors=_validate_aiter_gemm_artifacts(artifact_specs, gpu_type=gpu_type),
        )


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


def _apply_failed(attempt: IntegrateAttempt, error_class: str, error: Any, **fields: Any) -> dict[str, Any]:
    """The shape every apply-stage rejection shares: nothing landed, so nothing was taken back out."""
    return {
        "status": "apply_failed",
        "error_class": error_class,
        "error": error,
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
        **fields,
    }


def _reauthor_fields(
    params: dict[str, Any],
    patch_paths: list[Path],
    *,
    advisory: str = "",
    feedback: tuple[ApplyFeedback, ...] = (),
) -> dict[str, Any]:
    """Everything the specialist needs to author the refused patches again."""
    fields: dict[str, Any] = {
        "lane": _derive_lane(params),
        "retry_feedback": [item.to_dict() for item in feedback],
        "prior_patches": [str(patch) for patch in patch_paths],
    }
    if advisory:
        fields["advisory"] = advisory
    if params.get("enablement"):
        fields["enablement"] = True
    return fields


def _unresolved_root_rejection(explicit_root: str | None, patch_paths: list[Path]) -> dict[str, Any]:
    """Why these patches could not be bound to a framework tree, in terms their author can act on."""
    if not explicit_root:
        return {
            "error_class": "no_framework_agent_root",
            "error": "no framework_source_root resolved; cannot apply "
            "patches. Configure $INFERENCEX_PATH or pass "
            "params.framework_source_root.",
        }
    declared_root = resolved_explicit_root(explicit_root)
    if declared_root is None:
        return {
            "error_class": "framework_root_unresolved",
            "error": f"framework_source_root {explicit_root!r} is not a readable directory",
        }
    if not _read_patch_texts(patch_paths):
        return {
            "error_class": "patch_unreadable",
            "error": "no patch file could be read; verify paths and permissions",
        }
    missing_records = _preflight_missing_targets(declared_root, patch_paths)
    if missing_records:
        return {
            "error_class": "patch_target_missing",
            "error": missing_records,
            "advisory": "patch target file(s) absent from framework_source_root "
            f"{explicit_root}; author patches only against files "
            "that exist in the installed framework tree.",
        }
    return {
        "error_class": "patch_root_ambiguous",
        "error": f"framework_source_root {explicit_root!r} could not be unambiguously matched to the patch targets",
    }


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


def _revert_status(
    measured: _PatchMeasurement,
    *,
    accuracy_blocks: bool,
    accuracy_verdict: bool | None,
    fix_refusals: list[str],
) -> str:
    """Whether the patch was rejected on its merits or only because its accuracy could not be read."""
    tput_ok = measured.delta_pct is not None and measured.delta_pct >= measured.keep_threshold_pct
    unreadable = accuracy_blocks and accuracy_verdict is None and tput_ok
    return "accuracy_unavailable_reject" if unreadable or fix_refusals == [_FIX_ACCURACY_UNAVAILABLE] else "reverted"


def _perf_revert(
    attempt: IntegrateAttempt,
    measured: _PatchMeasurement,
    *,
    status: str,
    reason: str,
    **fields: Any,
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
        **fields,
    }


def _kept_patch(
    attempt: IntegrateAttempt,
    measured: _PatchMeasurement,
    *,
    params: dict[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    """The verdict of a patch that cleared the gate and stays in the tree."""
    return {
        "status": "kept",
        "specialist_task_id": attempt.specialist_task_id,
        # Proposal ownership must survive delegated-result persistence
        # so resume replay cannot replace it with the then-current phase.
        "source_phase": str(params.get("source_phase") or ""),
        "domain": str(params.get("domain") or ""),
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
        "workspace": str(attempt.output_root),
        "framework_root": str(attempt.framework_root or ""),
        **fields,
    }
