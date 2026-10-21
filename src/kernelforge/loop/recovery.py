# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable caller-facing recovery artifacts for forge-loop."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.tools.durable_io import atomic_write_text
from kernelforge.loop.reporting import (
    BestResultPublisher,
    KeepBundle,
    KeepIdentity,
    KeepScore,
    is_best_manifest,
)
from kernelforge.evaluation.scoring import warm_start_improvement_flags


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Durably publish one JSON snapshot via temp-file replacement."""
    atomic_write_text(path, json.dumps(payload))


def load_published_best(workspace_dir: str) -> dict | None:
    """Return the published best result, or ``None`` when the campaign has no authoritative one.

    ``best_result.json`` is the commit point: every consumer of a campaign's outcome reads it
    through here so that one verdict on what counts as published governs them all.
    """
    path = Path(workspace_dir) / "forge_experiments" / "best_result.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not is_best_manifest(payload) or payload.get("correctness_passed") is not True:
        return None
    return payload


def _validated_warm_start_result(
    workspace_dir: str,
    *,
    commit_hash: str,
    baseline_ms: float,
    best_ms: float,
    mean_case_speedup: float,
) -> dict | None:
    """Return the published warm-start commit point when it is authoritative."""
    payload = load_published_best(workspace_dir)
    if payload is None or payload.get("commit_hash") != commit_hash or int(payload.get("iteration", -1)) != 0:
        return None
    try:
        published_baseline = float(payload.get("baseline_wall_ms"))
        published_best = float(payload.get("best_wall_ms"))
        published_mean_case_speedup = float(payload.get("mean_case_speedup"))
    except (TypeError, ValueError):
        return None
    if (
        published_baseline != float(baseline_ms)
        or published_best != float(best_ms)
        or published_mean_case_speedup != float(mean_case_speedup)
        or published_mean_case_speedup <= 1.0
    ):
        return None
    return payload


def rollback_unpublished_warm_start(
    workspace_dir: str,
    *,
    base_commit: str,
    result_json: str | None,
) -> None:
    """Restore pristine HEAD when warm-start has no authoritative commit point."""
    git("reset", "--hard", base_commit, cwd=workspace_dir)
    head = git("rev-parse", "HEAD", cwd=workspace_dir).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=workspace_dir).stdout.strip()
    if head != base_commit or dirty:
        raise RuntimeError("warm-start rollback did not restore pristine workspace")

    root = Path(workspace_dir) / "forge_experiments"
    shutil.rmtree(root / "best" / "iter_000", ignore_errors=True)
    for path in (
        root / "best" / "manifest.json",
        root / "best_result.json",
        root / "optimization_report.md",
    ):
        path.unlink(missing_ok=True)
    if result_json:
        Path(result_json).unlink(missing_ok=True)


@dataclass(frozen=True)
class CallerRecovery:
    """How the external caller finds a killed run again: its experiment record, KB experience id and result file."""

    caller_experiment_id: str
    experience_id: str
    result_json: str | None


@dataclass(frozen=True)
class _WarmStartMeasurements:
    score: KeepScore
    case_times: dict[str, float]
    unscored_cases: list[str]


def _warm_start_measurements(warm: dict) -> _WarmStartMeasurements:
    baseline_ms = warm.get("pristine_ms")
    best_ms = warm.get("keep_baseline_ms")
    mean_case_speedup = warm.get("mean_case_speedup")
    if (
        not isinstance(baseline_ms, (int, float))
        or not isinstance(best_ms, (int, float))
        or not isinstance(mean_case_speedup, (int, float))
        or baseline_ms <= 0
        or best_ms <= 0
        or mean_case_speedup <= 1.0
    ):
        raise ValueError("validated warm-start is missing improving measurements")
    baseline_ms = float(baseline_ms)
    best_ms = float(best_ms)
    mean_case_speedup = float(mean_case_speedup)
    return _WarmStartMeasurements(
        score=KeepScore(
            baseline_wall_ms=baseline_ms,
            best_wall_ms=best_ms,
            mean_case_speedup=mean_case_speedup,
            search_start_mean_case_speedup=mean_case_speedup,
            search_start_ms=best_ms,
        ),
        case_times=dict(warm.get("case_times") or {}),
        unscored_cases=list(warm.get("unscored_cases") or []),
    )


def _warm_start_identity(workspace_dir: str, base_commit: str, warm: dict, external_id: str) -> KeepIdentity:
    """KeepIdentity for HEAD when it differs from ``base_commit``."""
    head = git("rev-parse", "HEAD", cwd=workspace_dir).stdout.strip()
    if not head or head == base_commit:
        raise ValueError("validated warm-start has no committed patch")
    solution = str(warm.get("solution_slug") or "warm-start")
    return KeepIdentity(
        campaign_id=f"warm-start:{solution}",
        session_index=0,
        experiment_id=external_id,
        iteration=0,
        commit_hash=head,
        plan=f"apply prior solution {solution}",
    )


def _warm_start_bundle(
    workspace_dir: str, base_commit: str, keep: KeepIdentity, measurements: _WarmStartMeasurements
) -> KeepBundle:
    score = measurements.score
    patch = git("diff", base_commit, keep.commit_hash, "--", ".", cwd=workspace_dir).stdout
    changed_files = [
        line.strip()
        for line in git(
            "diff", "--name-only", base_commit, keep.commit_hash, "--", ".", cwd=workspace_dir
        ).stdout.splitlines()
        if line.strip()
    ]
    if not patch.strip() or not changed_files:
        raise ValueError("validated warm-start produced no publishable diff")
    return KeepBundle(
        validation_text=("validated KB warm-start passed canonical correctness"),
        benchmark={
            "median_ms": score.best_wall_ms,
            "mean_case_speedup": score.mean_case_speedup,
            "case_times": measurements.case_times,
            "unscored_cases": measurements.unscored_cases,
            "warm_start": True,
        },
        changed_files=changed_files,
        patch=patch,
    )


def _persist_warm_recovery(caller: CallerRecovery, tracker, checkpoint: dict, result: dict) -> dict:
    """Write checkpoint and result-json sidecars; update persistence_* on ``result``."""
    errors = result["persistence_errors"]
    if caller.caller_experiment_id:
        try:
            tracker.set_checkpoint(caller.caller_experiment_id, checkpoint)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"checkpoint: {error}")
            result["persistence_degraded"] = True
    if caller.result_json:
        try:
            atomic_write_json(caller.result_json, result)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"result-json: {error}")
            result["persistence_degraded"] = True
    return result


def publish_warm_start_recovery(
    workspace_dir: str,
    *,
    base_commit: str,
    warm: dict,
    caller: CallerRecovery,
    tracker,
) -> dict | None:
    """Publish a validated warm-start as a kill-recoverable local best."""
    if warm.get("applied") is not True:
        return None
    measurements = _warm_start_measurements(warm)
    score = measurements.score
    external_id = caller.caller_experiment_id or caller.experience_id or "warm-start"
    keep = _warm_start_identity(workspace_dir, base_commit, warm, external_id)
    bundle = _warm_start_bundle(workspace_dir, base_commit, keep, measurements)
    errors: list[str] = []
    try:
        manifest = BestResultPublisher(workspace_dir).publish(keep, score, bundle)
    except Exception as error:
        # A derived view can fail after best_result.json is already durable.
        manifest = _validated_warm_start_result(
            workspace_dir,
            commit_hash=keep.commit_hash,
            baseline_ms=score.baseline_wall_ms,
            best_ms=score.best_wall_ms,
            mean_case_speedup=score.mean_case_speedup,
        )
        if manifest is None:
            raise
        errors.append(f"derived-best-view: {error}")
    # The manifest publish above withholds improvement when aggregate wall times contradict the case score.
    improvement = warm_start_improvement_flags(
        pristine_ms=score.baseline_wall_ms, best_ms=score.best_wall_ms, mean_case_speedup=score.mean_case_speedup
    )
    scores = {
        "baseline_ms": score.baseline_wall_ms,
        "best_ms": score.best_wall_ms,
        "mean_case_speedup": score.mean_case_speedup,
        "search_start_mean_case_speedup": score.mean_case_speedup,
    }
    root = Path(workspace_dir) / "forge_experiments"
    checkpoint = {
        "state": "best_committed",
        "decision": "WARM_START",
        "experiment_id": external_id,
        "base_commit": base_commit,
        "best_commit": keep.commit_hash,
        "best_iteration": 0,
        **scores,
        **improvement,
        "validation_passed": True,
        "validation_summary": "validated KB warm-start",
        "snr_db": None,
        "case_times": measurements.case_times,
        "unscored_cases": measurements.unscored_cases,
    }
    result = {
        **scores,
        "case_times": measurements.case_times,
        "unscored_cases": measurements.unscored_cases,
        **improvement,
        "experiment_id": caller.caller_experiment_id or None,
        "campaign_id": manifest["campaign_id"],
        "session_index": 0,
        "segment_index": 0,
        "next_iteration": 1,
        "best_iteration": 0,
        "best_commit": keep.commit_hash,
        "best_manifest": str(root / "best" / "manifest.json"),
        "optimization_report": str(root / "optimization_report.md"),
        "optimization_history": str(root / "optimization_history.md"),
        "persistence_degraded": bool(errors),
        "persistence_errors": errors,
        "iteration_count": 0,
        "kb_experience": None,
        "warm_start": True,
    }
    return _persist_warm_recovery(caller, tracker, checkpoint, result)
