# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable caller-facing recovery artifacts for forge-loop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.durable_io import atomic_write_text
from kernelforge.loop.reporting import (
    MANIFEST_SCHEMA_VERSION,
    BestResultPublisher,
)
from kernelforge.loop.scoring import warm_start_improvement_flags


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Durably publish one JSON snapshot via temp-file replacement."""
    atomic_write_text(path, json.dumps(payload))


def _validated_warm_start_result(
    workspace_dir: str,
    *,
    commit_hash: str,
    baseline_ms: float,
    best_ms: float,
    mean_case_speedup: float,
) -> dict | None:
    """Return the published warm-start commit point when it is authoritative."""
    path = Path(workspace_dir) / "forge_experiments" / "best_result.json"
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("correctness_passed") is not True
        or payload.get("commit_hash") != commit_hash
        or int(payload.get("iteration", -1)) != 0
    ):
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


def publish_warm_start_recovery(
    *,
    workspace_dir: str,
    base_commit: str,
    warm: dict,
    caller_experiment_id: str,
    experience_id: str,
    tracker,
    result_json: str | None,
) -> dict | None:
    """Publish a validated warm-start as a kill-recoverable local best."""
    if warm.get("applied") is not True:
        return None
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
    case_times = dict(warm.get("case_times") or {})
    unscored_cases = list(warm.get("unscored_cases") or [])

    head = git("rev-parse", "HEAD", cwd=workspace_dir).stdout.strip()
    if not head or head == base_commit:
        raise ValueError("validated warm-start has no committed patch")
    patch = git("diff", base_commit, head, "--", ".", cwd=workspace_dir).stdout
    changed_files = [
        line.strip()
        for line in git("diff", "--name-only", base_commit, head, "--", ".", cwd=workspace_dir).stdout.splitlines()
        if line.strip()
    ]
    if not patch.strip() or not changed_files:
        raise ValueError("validated warm-start produced no publishable diff")

    root = Path(workspace_dir) / "forge_experiments"
    solution = str(warm.get("solution_slug") or "warm-start")
    external_id = caller_experiment_id or experience_id or "warm-start"
    publisher = BestResultPublisher(workspace_dir)
    publish_kwargs = {
        "campaign_id": f"warm-start:{solution}",
        "session_index": 0,
        "experiment_id": external_id,
        "iteration": 0,
        "commit_hash": head,
        "plan": f"apply prior solution {solution}",
        "baseline_wall_ms": float(baseline_ms),
        "search_start_ms": float(best_ms),
        "best_wall_ms": float(best_ms),
        "mean_case_speedup": float(mean_case_speedup),
        "search_start_mean_case_speedup": float(mean_case_speedup),
        "snr_db": None,
        "validation_text": ("validated KB warm-start passed canonical correctness"),
        "benchmark": {
            "median_ms": float(best_ms),
            "mean_case_speedup": float(mean_case_speedup),
            "case_times": case_times,
            "unscored_cases": unscored_cases,
            "warm_start": True,
        },
        "changed_files": changed_files,
        "patch": patch,
    }
    publication_errors: list[str] = []
    try:
        manifest = publisher.publish(**publish_kwargs)
    except Exception as error:
        # A derived view can fail after best_result.json is already durable. In
        # that case the external recovery contract is satisfied and the run may
        # continue; otherwise the caller must rollback to the pristine base.
        manifest = _validated_warm_start_result(
            workspace_dir,
            commit_hash=head,
            baseline_ms=float(baseline_ms),
            best_ms=float(best_ms),
            mean_case_speedup=float(mean_case_speedup),
        )
        if manifest is None:
            raise
        publication_errors.append(f"derived-best-view: {error}")
    # The manifest publish() just wrote withholds the improvement badge when the
    # aggregate wall times contradict the score. The checkpoint and the caller's
    # result are written from the same adoption and used to assert an
    # improvement outright, so a reader's conclusion depended on which of the
    # three artifacts it happened to open.
    improvement = warm_start_improvement_flags(
        pristine_ms=float(baseline_ms),
        best_ms=float(best_ms),
        mean_case_speedup=float(mean_case_speedup),
    )
    checkpoint = {
        "schema_version": 1,
        "state": "best_committed",
        "decision": "WARM_START",
        "experiment_id": external_id,
        "base_commit": base_commit,
        "best_commit": head,
        "best_iteration": 0,
        "baseline_ms": float(baseline_ms),
        "best_ms": float(best_ms),
        "mean_case_speedup": float(mean_case_speedup),
        "search_start_mean_case_speedup": float(mean_case_speedup),
        **improvement,
        "validation_passed": True,
        "validation_summary": "validated KB warm-start",
        "snr_db": None,
        "case_times": case_times,
        "unscored_cases": unscored_cases,
    }
    result = {
        "baseline_ms": float(baseline_ms),
        "best_ms": float(best_ms),
        "mean_case_speedup": float(mean_case_speedup),
        "search_start_mean_case_speedup": float(mean_case_speedup),
        "case_times": case_times,
        "unscored_cases": unscored_cases,
        **improvement,
        "experiment_id": caller_experiment_id or None,
        "campaign_id": manifest["campaign_id"],
        "session_index": 0,
        "segment_index": 0,
        "next_iteration": 1,
        "best_iteration": 0,
        "best_commit": head,
        "best_manifest": str(root / "best" / "manifest.json"),
        "optimization_report": str(root / "optimization_report.md"),
        "optimization_history": str(root / "optimization_history.md"),
        "persistence_degraded": False,
        "persistence_errors": [],
        "iteration_count": 0,
        "kb_experience": None,
        "warm_start": True,
    }
    persistence_errors: list[str] = list(publication_errors)
    if caller_experiment_id:
        try:
            tracker.set_checkpoint(caller_experiment_id, checkpoint)
        except Exception as error:
            persistence_errors.append(f"checkpoint: {error}")
    if persistence_errors:
        result["persistence_degraded"] = True
        result["persistence_errors"] = list(persistence_errors)
    if result_json:
        try:
            atomic_write_json(result_json, result)
        except Exception as error:
            persistence_errors.append(f"result-json: {error}")
    if persistence_errors:
        result["persistence_degraded"] = True
        result["persistence_errors"] = persistence_errors
    return result
