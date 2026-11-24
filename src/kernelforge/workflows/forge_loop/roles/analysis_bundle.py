# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The validated Analysis bundle, its outcome, and the checks that read a bundle or checkpoint from disk."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.io import DURABLE, JsonFormat, atomic_write_json
from kernelforge.workflows.forge_loop.roles.contracts import (
    CaseEvidence,
    EvidenceRef,
    EvidenceContext,
)
from kernelforge.workflows.forge_loop.roles.analysis_prompts import LANE_PARTITION_FILE
from kernelforge.workflows.forge_loop.roles.analysis_session import _sha256


# Artifacts every analysis bundle carries at its root, whatever its status.
_REQUIRED_BUNDLE_FILES = (
    "request.json",
    "workflow.json",
    "workflow_events.jsonl",
    "artifact_catalog.json",
    "source_map.md",
    "case_inventory.json",
    "progress.json",
    "commands.jsonl",
)


@dataclass(frozen=True)
class AnalysisCase:
    """Map one canonical case ID to its bundle directory."""

    case_id: str
    directory: str
    latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "directory": self.directory,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class AnalysisBundle:
    """Validated, immutable analysis artifact for one canonical commit."""

    analysis_commit: str
    root: Path
    manifest: dict[str, Any]
    cases: tuple[CaseEvidence, ...]
    outcome: AnalysisOutcome | None = None

    def apply(self, context: EvidenceContext) -> EvidenceContext:
        """Return an orchestration context backed by this bundle."""
        source_map = self.root / "source_map.md"
        report = self.root / "report.md"
        catalog = self.root / "artifact_catalog.json"
        evidence_by_path = {reference.path: reference for reference in context.evidence_refs}
        evidence_by_path[str(self.root)] = EvidenceRef(
            kind="analysis_bundle",
            path=str(self.root),
            summary=(f"Validated commit-bound Analysis bundle with {self.manifest.get('status')} status."),
        )
        evidence_by_path[str(report)] = EvidenceRef(
            kind="analysis_summary",
            path=str(report),
            summary="Cross-case profiling and potential summary.",
        )
        catalog_payload = json.loads(catalog.read_text())
        for artifact in catalog_payload["artifacts"]:
            path = str(artifact["path"])
            evidence_by_path[path] = EvidenceRef(
                kind=str(artifact["kind"]),
                path=path,
                summary=(
                    f"{artifact['description']} "
                    f"Status: {artifact['status']}. "
                    "Available information: "
                    f"{', '.join(artifact['available_information'])}."
                ),
            )
        return EvidenceContext(
            analysis_commit=context.analysis_commit,
            workspace=context.workspace,
            gpu_target=context.gpu_target,
            objective=context.objective,
            program_context=context.program_context,
            source_map_path=str(source_map),
            editable_sources=context.editable_sources,
            cases=self.cases,
            knowledge_index=context.knowledge_index,
            evidence_refs=tuple(evidence_by_path.values()),
            canonical_commit=(context.canonical_commit or context.analysis_commit),
            evidence_commit=self.analysis_commit,
            evidence_stale=(self.analysis_commit != (context.canonical_commit or context.analysis_commit)),
            evidence_status=(context.evidence_status or str(self.manifest.get("status") or "published").lower()),
            evidence_mean_case_speedup=(context.evidence_mean_case_speedup),
            current_mean_case_speedup=context.current_mean_case_speedup,
            cumulative_diff_path=context.cumulative_diff_path,
            cumulative_diff_error=context.cumulative_diff_error,
        )


_LANE_HEADING = re.compile(r"^#{1,6}\s*lane\s+\d+\b.*$", re.IGNORECASE | re.MULTILINE)


def read_lane_focuses(bundle_root: Path) -> list[str]:
    """The focuses a published Analysis divided its evidence into, in lane order; empty when it divided nothing.

    A focus is the text under one ``## Lane N`` heading of the bundle's partition file. The loop reads nothing else
    of it: what a focus says is between the Analysis Agent and the session that works on it.
    """
    try:
        text = (bundle_root / LANE_PARTITION_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    headings = list(_LANE_HEADING.finditer(text))
    sections = [
        text[heading.end() : following.start() if following is not None else len(text)].strip()
        for heading, following in zip(headings, [*headings[1:], None])
    ]
    return [section for section in sections if section]


class AnalysisBundleError(RuntimeError):
    """Report an invalid or unsafe Analysis Agent result."""


@dataclass(frozen=True)
class AnalysisOutcome:
    """Structured Analysis attempt result for campaign events and orchestration."""

    analysis_commit: str
    requested_tier: str
    available_tier: str
    attempt: int
    checkpoint_level: str
    artifact_path: str = ""
    failure_type: str | None = None
    upgrade_exhausted: bool = False
    parent_reuse_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_commit": self.analysis_commit,
            "requested_tier": self.requested_tier,
            "available_tier": self.available_tier,
            "attempt": self.attempt,
            "checkpoint_level": self.checkpoint_level,
            "artifact_path": self.artifact_path,
            "failure_type": self.failure_type,
            "upgrade_exhausted": self.upgrade_exhausted,
            "parent_reuse_commit": self.parent_reuse_commit,
        }


def _parse_request_payload(
    payload: Any,
    *,
    analysis_commit: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnalysisBundleError("analysis request must be an object")
    if payload.get("analysis_commit") != analysis_commit:
        raise AnalysisBundleError("analysis request commit is invalid")
    if not isinstance(payload.get("analysis_profiling_enabled"), bool):
        raise AnalysisBundleError("analysis request profiling flag is invalid")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AnalysisBundleError("analysis request cases are invalid")
    for case in cases:
        if not isinstance(case, dict) or not case.get("case_id"):
            raise AnalysisBundleError("analysis request case entry is invalid")
    return payload


def _parse_workflow_payload(
    payload: Any,
    *,
    analysis_commit: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnalysisBundleError("analysis workflow must be an object")
    if payload.get("analysis_commit") != analysis_commit:
        raise AnalysisBundleError("analysis workflow commit is invalid")
    session = payload.get("session")
    if not isinstance(session, dict):
        raise AnalysisBundleError("analysis workflow session is invalid")
    attempts = session.get("attempts", 0)
    if not isinstance(attempts, int) or attempts < 0:
        raise AnalysisBundleError("analysis workflow attempts are invalid")
    return payload


def _parse_catalog_payload(
    payload: Any,
    *,
    analysis_commit: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnalysisBundleError("analysis artifact catalog must be an object")
    if payload.get("analysis_commit") != analysis_commit:
        raise AnalysisBundleError("analysis catalog commit is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(artifact, dict) for artifact in artifacts):
        raise AnalysisBundleError("analysis catalog artifacts are invalid")
    return payload


def _case_directory(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._-")
    readable = readable or "case"
    digest = hashlib.sha256(case_id.encode()).hexdigest()[:8]
    return f"{readable}-{digest}"


def _analysis_cases(context: EvidenceContext) -> tuple[AnalysisCase, ...]:
    return tuple(
        AnalysisCase(case_id=case.case_id, directory=_case_directory(case.case_id), latency_ms=case.latency_ms)
        for case in context.cases
    )


def _source_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class _BundleIdentity:
    """What a bundle must answer for: the orchestration context, its cases, and the driver and source digests."""

    context: EvidenceContext
    cases: tuple[AnalysisCase, ...]
    driver_digest: str
    source_digest: str


@dataclass(frozen=True)
class _BundleReading:
    """A bundle under validation: its root, the request it answers and the cases its manifest completed or skipped."""

    root: Path
    request_payload: dict
    completed: frozenset[str]
    skipped: frozenset[str]

    @property
    def profiling_enabled(self) -> bool:
        return bool(self.request_payload["analysis_profiling_enabled"])


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_valid_profile_evidence(
    work_root: Path,
    case: AnalysisCase,
) -> bool:
    profile_root = work_root / "cases" / case.directory / "profile"
    profile_files = (
        [
            path
            for path in profile_root.rglob("*")
            if path.is_file() and path.stat().st_size > 0 and path.name not in {"error.log", "stderr.log"}
        ]
        if profile_root.is_dir()
        else []
    )
    if not profile_files:
        return False

    metrics_path = work_root / "cases" / case.directory / "normalized_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metrics, dict) or not any(value not in (None, "", [], {}) for value in metrics.values()):
        return False

    framework_commands_path = work_root / "framework_commands.jsonl"
    provenance_payload = {
        "case_id": case.case_id,
        "framework_owned": True,
        "validation": "artifact_digest",
        "artifacts": {str(path.relative_to(profile_root)): _sha256(path) for path in profile_files},
        "normalized_metrics_sha256": _sha256(metrics_path),
    }
    existing_rows: list[dict[str, Any]] = []
    if framework_commands_path.is_file():
        try:
            existing_rows = [
                json.loads(line) for line in framework_commands_path.read_text().splitlines() if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            existing_rows = []
    if not any(
        isinstance(row, dict) and row.get("case_id") == case.case_id and row.get("framework_owned") is True
        for row in existing_rows
    ):
        with framework_commands_path.open("a") as stream:
            stream.write(json.dumps(provenance_payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    atomic_write_json(
        profile_root.parent / "profile_provenance.json",
        {
            "case_id": case.case_id,
            "framework_owned": True,
            "artifacts": provenance_payload["artifacts"],
            "normalized_metrics_sha256": provenance_payload["normalized_metrics_sha256"],
        },
        fmt=JsonFormat(trailing_newline=True),
        options=DURABLE,
    )
    return True


def _validate_bundle(root: Path, identity: _BundleIdentity) -> AnalysisBundle:
    context = identity.context
    manifest, statuses = _validated_manifest(root, identity)
    request_payload = _validated_session_metadata(
        root,
        context,
        manifest,
        completed=statuses["completed"],
    )
    reading = _BundleReading(root, request_payload, frozenset(statuses["completed"]), frozenset(statuses["skipped"]))
    normalized_cases = [
        _case_evidence(case, original, reading) for case, original in zip(identity.cases, context.cases)
    ]
    return AnalysisBundle(
        analysis_commit=context.analysis_commit,
        root=root,
        manifest=manifest,
        cases=tuple(normalized_cases),
    )


def _validated_manifest(root: Path, identity: _BundleIdentity) -> tuple[dict, dict[str, list]]:
    """The bundle's manifest and its per-case status lists, or the reason it is not one."""
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise AnalysisBundleError(f"invalid manifest.json: {error}") from error
    if not isinstance(manifest, dict):
        raise AnalysisBundleError("manifest.json must be an object")
    if manifest.get("analysis_commit") != identity.context.analysis_commit:
        raise AnalysisBundleError("analysis manifest commit does not match")
    if manifest.get("driver_digest") != identity.driver_digest:
        raise AnalysisBundleError("analysis manifest driver digest does not match")
    if manifest.get("source_digest") != identity.source_digest:
        raise AnalysisBundleError("analysis manifest source digest does not match")
    if manifest.get("status") not in {"READY", "PARTIAL", "FAILED"}:
        raise AnalysisBundleError("analysis manifest status is invalid")
    expected = [case.case_id for case in identity.cases]
    if manifest.get("expected_case_ids") != expected:
        raise AnalysisBundleError("analysis manifest case inventory does not match")
    completed = manifest.get("completed_case_ids")
    failed = manifest.get("failed_case_ids")
    skipped = manifest.get("skipped_case_ids") or []
    if not isinstance(completed, list) or not isinstance(failed, list) or not isinstance(skipped, list):
        raise AnalysisBundleError("analysis manifest case statuses are invalid")
    if sorted(completed + failed + skipped) != sorted(expected):
        raise AnalysisBundleError("analysis manifest does not account for every case")
    return manifest, {"completed": completed, "failed": failed, "skipped": skipped}


def _validated_session_metadata(
    root: Path,
    context: EvidenceContext,
    manifest: dict,
    *,
    completed: list,
) -> dict:
    """The bundle's request payload, once its session metadata and root artifacts check out."""
    try:
        request_payload = _parse_request_payload(
            json.loads((root / "request.json").read_text()),
            analysis_commit=context.analysis_commit,
        )
        _parse_workflow_payload(
            json.loads((root / "workflow.json").read_text()),
            analysis_commit=context.analysis_commit,
        )
        _parse_catalog_payload(
            json.loads((root / "artifact_catalog.json").read_text()),
            analysis_commit=context.analysis_commit,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisBundleError(f"invalid Analysis session metadata: {error}") from error
    except AnalysisBundleError as error:
        raise AnalysisBundleError(f"malformed analysis checkpoint: {error}") from error
    if not bool(request_payload["analysis_profiling_enabled"]):
        if completed:
            raise AnalysisBundleError("static-only analysis cannot report profiled COMPLETE cases")
        if manifest.get("status") != "PARTIAL":
            raise AnalysisBundleError("static-only analysis manifest status must be PARTIAL")
    missing_root = [name for name in _REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
    if missing_root:
        raise AnalysisBundleError("analysis bundle missing root artifacts: " + ", ".join(missing_root))
    if not _nonempty_file(root / "report.md"):
        raise AnalysisBundleError("analysis bundle has no report.md")
    return request_payload


def _case_evidence(case: AnalysisCase, original, reading: _BundleReading) -> CaseEvidence:
    """One case's evidence, checked against the artifacts its manifest status promises."""
    case_root = reading.root / "cases" / case.directory
    if not case_root.is_dir():
        raise AnalysisBundleError(f"missing case directory for {case.case_id}")
    try:
        case_payload = json.loads((case_root / "case.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisBundleError(f"invalid case.json for {case.case_id}: {error}") from error
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise AnalysisBundleError(f"case.json identity mismatch for {case.case_id}")
    if case.case_id in reading.completed:
        return _completed_case_evidence(case, original, case_root, reading)
    if case.case_id in reading.skipped:
        return _skipped_case_evidence(
            original,
            case_root=case_root,
            profiling_enabled=reading.profiling_enabled,
        )
    if not _nonempty_file(case_root / "failure.md"):
        raise AnalysisBundleError(f"failed case {case.case_id} has no failure record")
    return CaseEvidence(
        case_id=original.case_id,
        shape=original.shape,
        dtype=original.dtype,
        latency_ms=original.latency_ms,
        bottleneck=original.bottleneck,
        profile_summary_path=original.profile_summary_path,
        flags=tuple(dict.fromkeys([*original.flags, "analysis_case_failed"])),
    )


def _completed_case_evidence(case: AnalysisCase, original, case_root: Path, reading: _BundleReading) -> CaseEvidence:
    """A COMPLETE case's evidence, which must carry the analysis the status claims."""
    analysis_path = case_root / "analysis.md"
    if not _nonempty_file(analysis_path):
        raise AnalysisBundleError(f"completed case {case.case_id} has no analysis.md")
    if reading.profiling_enabled and not _has_valid_profile_evidence(reading.root, case):
        raise AnalysisBundleError(f"completed case {case.case_id} has no validated profile evidence")
    bottleneck = _case_bottleneck(case_root / "bottleneck.json")
    profile_flag = (
        "analysis_profile_incremental"
        if reading.request_payload["analysis_trigger"] == "post_keep_incremental"
        else "analysis_profiled"
    )
    return CaseEvidence(
        case_id=case.case_id,
        shape=original.shape,
        dtype=original.dtype,
        latency_ms=original.latency_ms,
        bottleneck=str(bottleneck.get("classification") or original.bottleneck),
        profile_summary_path=str(analysis_path),
        flags=tuple(dict.fromkeys([*(str(item) for item in (bottleneck.get("flags") or [])), profile_flag])),
    )


def _skipped_case_evidence(original, *, case_root: Path, profiling_enabled: bool) -> CaseEvidence:
    """A SKIPPED case's evidence, flagged by how far the session got with it."""
    analysis_path = case_root / "analysis.md"
    has_analysis = _nonempty_file(analysis_path)
    if not profiling_enabled:
        reason = "analysis_static_only"
    else:
        reason = "analysis_interpretation_only" if has_analysis else "analysis_profile_skipped"
    return CaseEvidence(
        case_id=original.case_id,
        shape=original.shape,
        dtype=original.dtype,
        latency_ms=original.latency_ms,
        bottleneck=original.bottleneck,
        profile_summary_path=(str(analysis_path) if has_analysis else original.profile_summary_path),
        flags=tuple(dict.fromkeys([*original.flags, reason])),
    )


def _checkpoint_artifact_refs(artifacts: list, *, work_root: Path) -> dict[str, EvidenceRef]:
    """The well-formed catalog entries of an unpublished checkpoint that stay inside its work root, by path."""
    refs: dict[str, EvidenceRef] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path") or "")
        information = artifact.get("available_information")
        if (
            not path
            or str(artifact.get("status")) not in {"AVAILABLE", "COMPLETE", "FAILED", "SKIPPED"}
            or not isinstance(information, list)
            or not Path(path).resolve().is_relative_to(work_root.resolve())
        ):
            continue
        refs[path] = EvidenceRef(
            kind=str(artifact.get("kind") or "analysis_artifact"),
            path=path,
            summary=(
                f"{artifact.get('description') or 'Analysis artifact'} "
                f"Available information: "
                f"{', '.join(str(item) for item in information)}."
            ),
        )
    return refs


def _checkpoint_case_evidence(original: CaseEvidence, *, case_root: Path, profiling_enabled: bool) -> CaseEvidence:
    """A case's evidence from an unpublished checkpoint, flagged by how far the session got with it."""
    bottleneck_path = case_root / "bottleneck.json"
    analysis_path = case_root / "analysis.md"
    normalized_path = case_root / "normalized_metrics.json"
    profile_root = case_root / "profile"
    bottleneck = original.bottleneck
    summary_path = original.profile_summary_path
    flags = list(original.flags)
    if not profiling_enabled:
        flags.append("analysis_static_only")
    if bottleneck_path.is_file() and analysis_path.is_file():
        case_bottleneck = _case_bottleneck(bottleneck_path)
        bottleneck = str(case_bottleneck.get("classification") or bottleneck)
        summary_path = str(analysis_path.resolve())
        flags.append(
            "analysis_checkpoint_profile_interpretation"
            if (normalized_path.is_file() and profile_root.is_dir() and any(profile_root.iterdir()))
            else "analysis_checkpoint_interpretation_only"
        )
    elif normalized_path.is_file():
        summary_path = str(normalized_path.resolve())
        flags.append("analysis_checkpoint_normalized_only")
    elif profile_root.is_dir() and any(profile_root.iterdir()):
        summary_path = str(profile_root.resolve())
        flags.append("analysis_checkpoint_raw_profile_only")
    return CaseEvidence(
        case_id=original.case_id,
        shape=original.shape,
        dtype=original.dtype,
        latency_ms=original.latency_ms,
        bottleneck=bottleneck,
        profile_summary_path=summary_path,
        flags=tuple(dict.fromkeys(flags)),
    )


def _case_bottleneck(bottleneck_path: Path) -> dict:
    """The case's recorded bottleneck, or an empty one when it was not written or is unreadable."""
    if not bottleneck_path.is_file():
        return {}
    try:
        loaded = json.loads(bottleneck_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _published_generation_root(commit_root: Path) -> Path | None:
    """Resolve the active immutable generation for one analysis commit."""
    pointer_path = commit_root / "published.json"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text())
        except (OSError, json.JSONDecodeError):
            pointer = {}
        if isinstance(pointer, dict):
            generation_name = str(pointer.get("generation_root") or "")
            if generation_name:
                candidate = (commit_root / generation_name).resolve()
                if candidate.is_dir():
                    return candidate
    generations = sorted(commit_root.glob("generation-*"))
    if generations:
        return generations[-1]
    return None


def _tier_label(*, profiling_enabled: bool, profiled: bool) -> str:
    if profiling_enabled and profiled:
        return "profiled"
    if profiling_enabled:
        return "partial"
    return "static"
