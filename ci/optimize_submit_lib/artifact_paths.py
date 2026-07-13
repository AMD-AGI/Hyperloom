from __future__ import annotations

import json
from pathlib import Path

from .records import SubmissionRecord


DEFAULT_ARTIFACT_PATTERNS = (
    "optimization_report",
    "ci_metrics.json",
    "session_breakdown.json",
    "baseline_summary.json",
    "sweep_results.csv",
    "sweep_results.txt",
    "kernel_candidates.json",
    "kernel_results.json",
    "run_context.env",
    "gpu_timeline.csv",
    "ci_summary.json",
    "ci_report.md",
)


def _is_wanted_artifact(path: str, all_artifacts: bool) -> bool:
    if all_artifacts:
        return True
    p = path.lower()
    return any(pat in p for pat in DEFAULT_ARTIFACT_PATTERNS)


def _safe_local_path(artifacts_dir: Path, task_id: str, remote_path: str) -> Path:
    rel = remote_path.lstrip("/").replace("\\", "/")
    parts = [seg for seg in rel.split("/") if seg and seg != ".." and seg != "."]
    return artifacts_dir / task_id / Path(*parts) if parts else artifacts_dir / task_id / "artifact.bin"


def _record_artifact_source(
    rec: SubmissionRecord,
    local_path: Path,
    source_type: str,
    *,
    remote_path: str | None = None,
    source_path: str | None = None,
    session_dir: str | None = None,
) -> None:
    entry = {
        "source_type": source_type,
        "local_path": str(local_path).replace("\\", "/"),
        "file_name": local_path.name,
    }
    if remote_path:
        entry["remote_path"] = remote_path
    if source_path:
        entry["source_path"] = source_path
    if session_dir:
        entry["session_dir"] = session_dir
    rec.artifact_sources.append(entry)


def _write_artifact_sources(task_dir: Path, rec: SubmissionRecord) -> None:
    if not rec.artifact_sources:
        return
    payload = {
        "task_id": rec.task_id,
        "model": rec.model,
        "claw_session_id": rec.claw_session_id,
        "artifact_dir": str(task_dir).replace("\\", "/"),
        "files": rec.artifact_sources,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "artifact_sources.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
