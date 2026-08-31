"""Commit-bound profiling and analysis bundle produced by one Agent session."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kernelforge.llm.git import git
from kernelforge.agent_backends import (
    AgentBackend,
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.session_resume import (
    EXHAUSTED_END_REASON,
    run_session_with_api_resume,
)
from kernelforge.config import Config
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    EvidenceRef,
    OrchestrationContext,
)
from kernelforge.orchestrator.analysis_session import (
    AnalysisAttemptLimitError,
    AnalysisSessionJournal,
    MAX_ANALYSIS_SESSION_ATTEMPTS,
    SESSION_SCHEMA_VERSION,
)
from kernelforge.durable_io import atomic_write_text
from kernelforge.resources import assert_sandbox_grant


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_SESSION_STEP_ID = "analysis_session"
PROFILING_METHODOLOGY_FILES = (
    "measure_rocpc_workflow.md",
    "measure_triage.md",
    "measure_roofline.md",
    "measure_protocol.md",
)

log = logging.getLogger(__name__)
_WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
_BASH_WRITE_MARKERS = (
    " >",
    ">>",
    "sed -i",
    "perl -i",
    " tee ",
    " rm ",
    " mv ",
    " cp ",
)
_ROOT_FIND_RE = re.compile(r"""(?:^|[;&|]\s*)find\s+["']?/["']?(?:\s|$)""")


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
class IncrementalAnalysisInput:
    """Describe a KEEP-derived commit relative to its analyzed parent."""

    parent_commit: str
    parent_bundle: Path
    commit_diff: str
    changed_source_files: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisBundle:
    """Validated, immutable analysis artifact for one canonical commit."""

    analysis_commit: str
    root: Path
    manifest: dict[str, Any]
    cases: tuple[CaseEvidence, ...]
    outcome: AnalysisOutcome | None = None

    def apply(self, context: OrchestrationContext) -> OrchestrationContext:
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
        return OrchestrationContext(
            analysis_commit=context.analysis_commit,
            workspace=context.workspace,
            gpu_target=context.gpu_target,
            objective=context.objective,
            program_context=context.program_context,
            source_map_path=str(source_map),
            editable_sources=context.editable_sources,
            cases=self.cases,
            knowledge_index=context.knowledge_index,
            supervisor_guidance=context.supervisor_guidance,
            search_mode=context.search_mode,
            search_reason_codes=context.search_reason_codes,
            search_objective=context.search_objective,
            search_mode_residence_remaining=(context.search_mode_residence_remaining),
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


class AnalysisBundleError(RuntimeError):
    """Report an invalid or unsafe Analysis Agent result."""


class AnalysisConfigurationError(AnalysisBundleError):
    """Report a missing packaged resource or other non-retryable setup error."""


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
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisBundleError("analysis request schema_version is invalid")
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
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise AnalysisBundleError("analysis workflow schema_version is invalid")
    if payload.get("analysis_commit") not in {analysis_commit, None, ""}:
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
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisBundleError("analysis catalog schema_version is invalid")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


_GLOBAL_ARTIFACTS = {
    "request.json": (
        "analysis_request",
        "Immutable commit, input digests, objective, and expected cases.",
        ["analysis commit", "driver/source provenance", "expected case IDs"],
    ),
    "incremental_diff.patch": (
        "keep_diff",
        "Source changes between the previous analyzed commit and this KEEP.",
        ["changed source", "incremental analysis scope"],
    ),
    "workflow.json": (
        "analysis_workflow",
        "Durable status, attempts, outputs, and errors for the Analysis session.",
        ["session status", "attempt history", "resume point"],
    ),
    "workflow_events.jsonl": (
        "analysis_workflow_events",
        "Append-only timeline of Analysis session transitions.",
        ["session timing", "attempt history", "failure chronology"],
    ),
    "source_map.md": (
        "source_map",
        "Target call chain, source ownership, and performance-relevant regions.",
        ["call graph", "target regions", "editable source map"],
    ),
    "report.md": (
        "analysis_summary",
        "Primary Markdown Analysis report for downstream agents.",
        ["case findings", "evidence interpretation", "optimization directions"],
    ),
    "case_inventory.json": (
        "case_inventory",
        "Expected cases and their COMPLETE/FAILED coverage status.",
        ["case IDs", "shape coverage", "missing or failed cases"],
    ),
    "progress.json": (
        "analysis_progress",
        "Framework-owned phase and per-case durability checkpoint.",
        ["completed phases", "completed cases", "resume point"],
    ),
    "commands.jsonl": (
        "analysis_commands",
        "Executed commands with timeout, exit status, signal, and output paths.",
        ["command provenance", "profiler failures", "artifact locations"],
    ),
    "manifest.json": (
        "analysis_manifest",
        "Final READY/PARTIAL/FAILED bundle status and case accounting.",
        ["bundle status", "completed cases", "failed cases"],
    ),
}

_CASE_ARTIFACTS = {
    "case.json": (
        "case_definition",
        "Case identity, shape, dtype, and baseline/current latency context.",
        ["shape", "dtype", "latency context"],
    ),
    "profile": (
        "raw_profile",
        "Raw per-case profiler output and preserved workload files.",
        ["raw counters", "kernel traces", "profiler logs"],
    ),
    "normalized_metrics.json": (
        "normalized_metrics",
        "Validated target-kernel metrics normalized from raw profiler artifacts.",
        ["utilization", "occupancy", "memory and compute metrics"],
    ),
    "bottleneck.json": (
        "case_bottleneck",
        "Structured bottleneck classification, confidence, and evidence links.",
        ["primary bottleneck", "confidence", "flags"],
    ),
    "analysis.md": (
        "case_analysis",
        "Per-case explanation separating measured facts from interpretation.",
        ["profile interpretation", "limiting mechanisms"],
    ),
    "directions.md": (
        "case_directions",
        "Markdown optimization directions for this case.",
        ["strategy families", "candidate variants", "risks"],
    ),
    "failure.md": (
        "case_failure",
        "Markdown record of why this case could not be completed.",
        ["failed step", "error reason", "degradation cause"],
    ),
}


class _AnalysisProtection:
    """Keep every workspace input immutable while staging remains writable."""

    def __init__(
        self,
        *,
        workspace: Path,
        staging_root: Path,
        protected_paths: tuple[Path, ...],
        deadline_monotonic: float,
    ) -> None:
        self.workspace = workspace.resolve()
        self.staging_root = staging_root.resolve()
        self.protected_paths = tuple(path.resolve() for path in protected_paths)
        self.deadline_monotonic = deadline_monotonic
        self._snapshots = {path: path.read_bytes() for path in self.protected_paths if path.is_file()}
        self._protected_basenames = {path.name for path in self.protected_paths}

    def hooks(self) -> AgentHooks:
        return AgentHooks(
            pre_tool_use=[
                AgentHook(
                    matcher="Edit|Write|MultiEdit|NotebookEdit",
                    callback=self._on_pre_write,
                ),
                AgentHook(
                    matcher="Bash",
                    callback=self._on_pre_bash,
                ),
            ]
        )

    async def _on_pre_write(self, input_data, _tool_use_id, _context) -> dict:
        if input_data.get("tool_name") not in _WRITE_TOOLS:
            return {}
        tool_input = input_data.get("tool_input") or {}
        raw_path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path") or ""
        if self._inside_staging(raw_path):
            return {}
        return self._deny("Analysis output may only be written inside the supplied staging directory.")

    async def _on_pre_bash(self, input_data, _tool_use_id, _context) -> dict:
        if input_data.get("tool_name") != "Bash":
            return {}
        command = str((input_data.get("tool_input") or {}).get("command") or "")
        if _ROOT_FIND_RE.search(command):
            return self._deny(
                "Unbounded root filesystem searches are forbidden. Use the exact "
                "staging and knowledge paths supplied in the Analysis request."
            )
        lowered = f" {command.lower()} "
        has_write_intent = any(marker in lowered for marker in _BASH_WRITE_MARKERS)
        names_protected = any(basename.lower() in lowered for basename in self._protected_basenames)
        if has_write_intent and names_protected:
            return self._deny("This command may modify immutable source, driver, or test inputs.")
        remaining = max(1, int(self.deadline_monotonic - time.monotonic()))
        try:
            tokens = shlex.split(command)
        except ValueError:
            return self._deny("Analysis Bash command could not be parsed safely.")
        if not tokens or Path(tokens[0]).name != "timeout":
            return self._deny(
                "Every Analysis Bash command must be bounded by the shared "
                "session deadline. Prefix it exactly with "
                f"`timeout --signal=TERM --kill-after=5s {remaining}s ...`."
            )
        if "--signal=TERM" not in tokens or "--kill-after=5s" not in tokens:
            return self._deny(
                "Analysis Bash commands must use both `--signal=TERM` and "
                "`--kill-after=5s` so child process groups terminate reliably."
            )
        duration = self._timeout_duration_sec(tokens)
        if duration is None:
            return self._deny("Analysis Bash timeout duration is missing or invalid.")
        if duration > remaining:
            return self._deny(
                f"Analysis Bash timeout {duration:.1f}s exceeds the shared session remaining time {remaining}s."
            )
        return {}

    @staticmethod
    def _timeout_duration_sec(tokens: list[str]) -> float | None:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", token)
            if not match:
                return None
            value = float(match.group(1))
            multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]
            duration = value * multiplier
            return duration if duration > 0 else None
        return None

    def _inside_staging(self, raw_path: str) -> bool:
        if not raw_path:
            return False
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.staging_root / path
        try:
            path.resolve().relative_to(self.staging_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _deny(reason: str) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    def restore_and_report_changes(self) -> list[str]:
        changed = []
        for path, expected in self._snapshots.items():
            current = path.read_bytes() if path.is_file() else None
            if current == expected:
                continue
            changed.append(str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
        return changed


class AnalysisAgentService:
    """Run one long-lived Analysis Agent and publish its validated bundle."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        config: Config,
        timeout_sec: int,
        max_turns: int,
        profiling_enabled: bool = True,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.backend = backend
        self.config = config
        self.timeout_sec = timeout_sec
        self.max_turns = max_turns
        self.profiling_enabled = bool(profiling_enabled)

    def _stage_reference_profiling_script(self, work_root: Path) -> Path:
        profiling_source = Path(self.config.local_knowledge_dir).resolve() / "common_methodology" / "profiling"
        source = profiling_source / "rocpc_profile.py"
        if not source.is_file():
            raise AnalysisConfigurationError(f"packaged Analysis profiling script is missing: {source}")
        target = work_root / "tools" / "rocpc_profile.py"
        payload = source.read_bytes()
        if not target.is_file() or target.read_bytes() != payload:
            target.write_bytes(payload)
        target.chmod(source.stat().st_mode & 0o777)
        methodology_root = work_root / "tools" / "profiling"
        methodology_root.mkdir(parents=True, exist_ok=True)
        missing_methodology = []
        for name in PROFILING_METHODOLOGY_FILES:
            methodology_source = profiling_source / name
            if not methodology_source.is_file():
                missing_methodology.append(str(methodology_source))
                continue
            methodology_target = methodology_root / name
            methodology_payload = methodology_source.read_bytes()
            if not methodology_target.is_file() or methodology_target.read_bytes() != methodology_payload:
                methodology_target.write_bytes(methodology_payload)
        if missing_methodology:
            log.warning(
                "optional Analysis profiling methodology is missing: %s",
                ", ".join(missing_methodology),
            )
        return target.resolve()

    @staticmethod
    def _initialize_framework_artifacts(
        work_root: Path,
        context: OrchestrationContext,
        cases: tuple[AnalysisCase, ...],
    ) -> None:
        for case in cases:
            case_root = work_root / "cases" / case.directory
            case_root.mkdir(parents=True, exist_ok=True)
            if not (case_root / "case.json").is_file():
                _atomic_write_json(case_root / "case.json", case.to_dict())
        inventory_path = work_root / "case_inventory.json"
        if not inventory_path.is_file():
            _atomic_write_json(
                inventory_path,
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "analysis_commit": context.analysis_commit,
                    "cases": [case.to_dict() for case in cases],
                },
            )
        progress_path = work_root / "progress.json"
        if not progress_path.is_file():
            _atomic_write_json(
                progress_path,
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "analysis_commit": context.analysis_commit,
                    "status": "RUNNING",
                    "cases": [{"case_id": case.case_id, "status": "PENDING"} for case in cases],
                },
            )

    @staticmethod
    def _nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
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
            "schema_version": ANALYSIS_SCHEMA_VERSION,
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
        _atomic_write_json(
            profile_root.parent / "profile_provenance.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "case_id": case.case_id,
                "framework_owned": True,
                "artifacts": provenance_payload["artifacts"],
                "normalized_metrics_sha256": provenance_payload["normalized_metrics_sha256"],
            },
        )
        return True

    @classmethod
    def _finalize_framework_artifacts(
        cls,
        work_root: Path,
        context: OrchestrationContext,
        cases: tuple[AnalysisCase, ...],
        *,
        driver_digest: str,
        source_digest: str,
        profiling_enabled: bool,
    ) -> None:
        completed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        case_states = []
        for case in cases:
            case_root = work_root / "cases" / case.directory
            _atomic_write_json(case_root / "case.json", case.to_dict())
            analysis_path = case_root / "analysis.md"
            profile_root = case_root / "profile"
            has_analysis = cls._nonempty_file(analysis_path)
            has_profile = cls._has_valid_profile_evidence(
                work_root,
                case,
            )
            has_failure = cls._nonempty_file(case_root / "failure.md")
            if has_analysis and (has_profile or not profiling_enabled):
                if profiling_enabled:
                    completed.append(case.case_id)
                    state = "COMPLETE"
                else:
                    skipped.append(case.case_id)
                    state = "SKIPPED"
            elif has_failure:
                failed.append(case.case_id)
                state = "FAILED"
            else:
                skipped.append(case.case_id)
                state = "SKIPPED"
            case_states.append(
                {
                    **case.to_dict(),
                    "status": state,
                    "analysis_path": (str(analysis_path.relative_to(work_root)) if has_analysis else ""),
                    "profile_path": (str(profile_root.relative_to(work_root)) if has_profile else ""),
                }
            )

        status = "READY" if len(completed) == len(cases) else "PARTIAL"
        _atomic_write_json(
            work_root / "case_inventory.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_commit": context.analysis_commit,
                "cases": case_states,
            },
        )
        _atomic_write_json(
            work_root / "progress.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_commit": context.analysis_commit,
                "status": status,
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "status": case["status"],
                    }
                    for case in case_states
                ],
            },
        )
        _atomic_write_json(
            work_root / "manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_commit": context.analysis_commit,
                "driver_digest": driver_digest,
                "source_digest": source_digest,
                "status": status,
                "expected_case_ids": [case.case_id for case in cases],
                "completed_case_ids": completed,
                "failed_case_ids": failed,
                "skipped_case_ids": skipped,
                "report": "report.md",
            },
        )

    async def ensure_bundle(
        self,
        context: OrchestrationContext,
        *,
        kernel_file: str,
        driver_script: str,
        source_files: list[str],
        usage=None,
        deadline_unix: float | None = None,
        incremental: IncrementalAnalysisInput | None = None,
    ) -> AnalysisBundle:
        """Return a current bundle, running the Analysis Agent when absent."""
        workspace = Path(context.workspace).resolve()
        analysis_root = workspace / "forge_experiments" / "analysis"
        final_root = analysis_root / context.analysis_commit
        cases = tuple(
            AnalysisCase(
                case_id=case.case_id,
                directory=_case_directory(case.case_id),
                latency_ms=case.latency_ms,
            )
            for case in context.cases
        )
        resolved_sources = tuple(
            sorted(
                {
                    Path(kernel_file).resolve(),
                    *(Path(path).resolve() for path in source_files),
                }
            )
        )
        driver_path = Path(driver_script).resolve()
        driver_digest = _sha256(driver_path)
        source_digest = _source_digest(resolved_sources)
        work_root = analysis_root / "work" / context.analysis_commit
        retry_published_bundle = False
        requested_tier = "profiled" if self.profiling_enabled else "static"
        parent_reuse_commit = incremental.parent_commit if incremental is not None else ""
        published_root = self._published_generation_root(final_root) if final_root.is_dir() else None
        if published_root is not None:
            try:
                request_payload = _parse_request_payload(
                    json.loads((published_root / "request.json").read_text()),
                    analysis_commit=context.analysis_commit,
                )
                workflow_payload = _parse_workflow_payload(
                    json.loads((published_root / "workflow.json").read_text()),
                    analysis_commit=context.analysis_commit,
                )
            except (OSError, json.JSONDecodeError) as error:
                raise AnalysisBundleError(f"published analysis checkpoint is malformed: {error}") from error
            cached = self._validate_bundle(
                published_root,
                context,
                cases,
                driver_digest=driver_digest,
                source_digest=source_digest,
            )
            attempts = int(workflow_payload["session"].get("attempts", 0))
            cached_profiled = request_payload["analysis_profiling_enabled"] is True
            tier_satisfied = cached_profiled or not self.profiling_enabled
            status_satisfied = cached.manifest["status"] == "READY" if self.profiling_enabled else True
            available_tier = self._tier_label(
                profiling_enabled=self.profiling_enabled,
                profiled=cached_profiled and bool(cached.manifest.get("completed_case_ids")),
            )
            if tier_satisfied and (status_satisfied or attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS):
                upgrade_exhausted = not status_satisfied and attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS
                return AnalysisBundle(
                    analysis_commit=cached.analysis_commit,
                    root=cached.root,
                    manifest={
                        **cached.manifest,
                        **({"upgrade_exhausted": True} if upgrade_exhausted else {}),
                    },
                    cases=cached.cases,
                    outcome=self._build_outcome(
                        analysis_commit=context.analysis_commit,
                        requested_tier=requested_tier,
                        available_tier=available_tier,
                        attempt=attempts,
                        checkpoint_level="published",
                        artifact_path=str(cached.root),
                        failure_type=("upgrade_exhausted" if upgrade_exhausted else None),
                        upgrade_exhausted=upgrade_exhausted,
                        parent_reuse_commit=parent_reuse_commit,
                    ),
                )
            if attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
                return AnalysisBundle(
                    analysis_commit=cached.analysis_commit,
                    root=cached.root,
                    manifest={
                        **cached.manifest,
                        "upgrade_exhausted": True,
                    },
                    cases=cached.cases,
                    outcome=self._build_outcome(
                        analysis_commit=context.analysis_commit,
                        requested_tier=requested_tier,
                        available_tier=available_tier,
                        attempt=attempts,
                        checkpoint_level="published",
                        artifact_path=str(cached.root),
                        failure_type="upgrade_exhausted",
                        upgrade_exhausted=True,
                        parent_reuse_commit=parent_reuse_commit,
                    ),
                )
            if work_root.is_dir():
                try:
                    work_flow = _parse_workflow_payload(
                        json.loads((work_root / "workflow.json").read_text()),
                        analysis_commit=context.analysis_commit,
                    )
                    work_attempts = int(work_flow.get("session", {}).get("attempts", 0))
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                    AnalysisBundleError,
                ):
                    work_attempts = MAX_ANALYSIS_SESSION_ATTEMPTS
                if work_attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
                    return AnalysisBundle(
                        analysis_commit=cached.analysis_commit,
                        root=cached.root,
                        manifest={
                            **cached.manifest,
                            "upgrade_exhausted": True,
                        },
                        cases=cached.cases,
                        outcome=self._build_outcome(
                            analysis_commit=context.analysis_commit,
                            requested_tier=requested_tier,
                            available_tier=available_tier,
                            attempt=work_attempts,
                            checkpoint_level="published",
                            artifact_path=str(cached.root),
                            failure_type="upgrade_exhausted",
                            upgrade_exhausted=True,
                            parent_reuse_commit=parent_reuse_commit,
                        ),
                    )
            if not work_root.exists():
                work_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(published_root, work_root)
            retry_published_bundle = True

        session_deadline_unix = min(
            time.time() + self.timeout_sec,
            deadline_unix if deadline_unix is not None else float("inf"),
        )
        session_timeout_sec = session_deadline_unix - time.time()
        if session_timeout_sec <= 1:
            raise AnalysisBundleError("Analysis deadline exhausted before session start")

        (work_root / "tools").mkdir(parents=True, exist_ok=True)
        reference_script = self._stage_reference_profiling_script(work_root)
        self._initialize_framework_artifacts(work_root, context, cases)

        incremental_diff_path = work_root / "incremental_diff.patch"
        if incremental is not None:
            parent_commit_root = (analysis_root / incremental.parent_commit).resolve()
            parent_root = self._published_generation_root(parent_commit_root)
            expected_parent = incremental.parent_bundle.resolve()
            if parent_root is None or parent_root != expected_parent or not parent_root.is_dir():
                # The current canonical can still be analyzed from scratch.
                # A missing or superseded parent must not permanently block
                # this commit's two durable Analysis session attempts.
                incremental = None
                parent_reuse_commit = ""
        if incremental is not None:
            incremental_diff_path.write_text(incremental.commit_diff)
        else:
            incremental_diff_path.unlink(missing_ok=True)

        protected_paths = self._protected_paths(
            workspace=workspace,
            kernel_file=kernel_file,
            driver_script=driver_script,
            source_files=source_files,
        )
        protection = _AnalysisProtection(
            workspace=workspace,
            staging_root=work_root,
            protected_paths=protected_paths,
            deadline_monotonic=(time.monotonic() + session_timeout_sec),
        )
        request_path = work_root / "request.json"
        request_payload = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_commit": context.analysis_commit,
            "workspace": str(workspace),
            "kernel_file": str(Path(kernel_file).resolve()),
            "driver_script": str(driver_path),
            "driver_digest": driver_digest,
            "source_files": [str(path) for path in resolved_sources],
            "source_digest": source_digest,
            "cases": [case.to_dict() for case in cases],
            "objective": context.objective,
            "analysis_profiling_enabled": self.profiling_enabled,
            "reference_profiling_script": str(reference_script),
            "reference_profiling_script_sha256": _sha256(reference_script),
            "analysis_trigger": ("post_keep_incremental" if incremental is not None else "canonical_baseline"),
            "previous_analysis_commit": (incremental.parent_commit if incremental is not None else ""),
            "previous_analysis_bundle": (str(incremental.parent_bundle.resolve()) if incremental is not None else ""),
            "incremental_diff_path": (str(incremental_diff_path.resolve()) if incremental is not None else ""),
            "incremental_diff_sha256": (_sha256(incremental_diff_path) if incremental is not None else ""),
            "changed_source_files": (list(incremental.changed_source_files) if incremental is not None else []),
        }
        if request_path.is_file():
            try:
                durable_request = json.loads(request_path.read_text())
                immutable_keys = {
                    "schema_version",
                    "analysis_commit",
                    "workspace",
                    "kernel_file",
                    "driver_script",
                    "driver_digest",
                    "source_files",
                    "source_digest",
                    "cases",
                    "objective",
                }
                if any(durable_request.get(key) != request_payload.get(key) for key in immutable_keys):
                    raise AnalysisBundleError("durable analysis request does not match current inputs")
                if retry_published_bundle:
                    durable_request["analysis_profiling_enabled"] = self.profiling_enabled
                    request_payload = durable_request
                    _atomic_write_json(request_path, request_payload)
                else:
                    request_payload = durable_request
            except json.JSONDecodeError as error:
                raise AnalysisBundleError(f"durable analysis request is invalid: {error}") from error
        else:
            _atomic_write_json(request_path, request_payload)
        (work_root / "commands.jsonl").touch(exist_ok=True)
        try:
            workflow = AnalysisSessionJournal(
                work_root,
                analysis_commit=context.analysis_commit,
                driver_digest=driver_digest,
                source_digest=source_digest,
            )
            if retry_published_bundle:
                workflow.reopen()
            await self._run_analysis_session(
                workflow=workflow,
                context=context,
                work_root=work_root,
                request_path=request_path,
                kernel_file=Path(kernel_file).resolve(),
                driver_script=driver_path,
                source_files=resolved_sources,
                reference_script=reference_script,
                protection=protection,
                usage=usage,
                timeout_sec=session_timeout_sec,
                force_refresh=retry_published_bundle,
            )
        except asyncio.CancelledError:
            protection.restore_and_report_changes()
            raise
        except AnalysisAttemptLimitError:
            protection.restore_and_report_changes()
            raise
        except Exception as error:
            changed = protection.restore_and_report_changes()
            raise AnalysisBundleError(
                "Analysis workflow failed: "
                f"{type(error).__name__}: {error}; "
                f"checkpoint={work_root}; restored inputs={changed}"
            ) from error

        changed = protection.restore_and_report_changes()
        if changed:
            raise AnalysisBundleError("Analysis Agent modified immutable inputs; restored: " + ", ".join(changed))
        self._validate_bundle(
            work_root,
            context,
            cases,
            driver_digest=driver_digest,
            source_digest=source_digest,
        )
        manifest = json.loads((work_root / "manifest.json").read_text())
        workflow.finalize(str(manifest["status"]))
        generation_root = self._publish_generation(work_root, final_root)
        self._write_artifact_catalog(
            generation_root,
            workflow,
            cases,
            artifact_root=generation_root,
        )
        validated = self._validate_bundle(
            generation_root,
            context,
            cases,
            driver_digest=driver_digest,
            source_digest=source_digest,
        )
        workflow_payload = _parse_workflow_payload(
            json.loads((generation_root / "workflow.json").read_text()),
            analysis_commit=context.analysis_commit,
        )
        attempts = int(workflow_payload["session"].get("attempts", 0))
        request_payload = _parse_request_payload(
            json.loads((generation_root / "request.json").read_text()),
            analysis_commit=context.analysis_commit,
        )
        available_tier = self._tier_label(
            profiling_enabled=self.profiling_enabled,
            profiled=request_payload["analysis_profiling_enabled"]
            and bool(validated.manifest.get("completed_case_ids")),
        )
        return AnalysisBundle(
            analysis_commit=validated.analysis_commit,
            root=validated.root,
            manifest=validated.manifest,
            cases=validated.cases,
            outcome=self._build_outcome(
                analysis_commit=context.analysis_commit,
                requested_tier="profiled" if self.profiling_enabled else "static",
                available_tier=available_tier,
                attempt=attempts,
                checkpoint_level="published",
                artifact_path=str(validated.root),
                parent_reuse_commit=parent_reuse_commit,
            ),
        )

    @staticmethod
    def _session_outputs(
        work_root: Path,
        cases: tuple[AnalysisCase, ...],
    ) -> tuple[Path, ...]:
        outputs = [
            work_root / "request.json",
            work_root / "report.md",
            work_root / "source_map.md",
            work_root / "case_inventory.json",
            work_root / "progress.json",
            work_root / "commands.jsonl",
            work_root / "manifest.json",
            *(work_root / "cases" / case.directory for case in cases),
        ]
        incremental_diff = work_root / "incremental_diff.patch"
        if incremental_diff.is_file():
            outputs.append(incremental_diff)
        return tuple(outputs)

    @staticmethod
    def _write_artifact_catalog(
        work_root: Path,
        workflow: AnalysisSessionJournal,
        cases: tuple[AnalysisCase, ...],
        *,
        artifact_root: Path | None = None,
    ) -> Path:
        """Publish a compact map of every currently usable Analysis artifact."""
        target_root = (artifact_root or work_root).resolve()
        session_complete = workflow.status == "COMPLETE"
        manifest = {}
        manifest_path = work_root / "manifest.json"
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text())
                if isinstance(loaded, dict):
                    manifest = loaded
            except (OSError, json.JSONDecodeError):
                manifest = {}
        completed = set(manifest.get("completed_case_ids") or [])
        failed = set(manifest.get("failed_case_ids") or [])
        skipped = set(manifest.get("skipped_case_ids") or [])

        def output_path(path: Path) -> str:
            relative = path.resolve().relative_to(work_root.resolve())
            return str(target_root / relative)

        def case_status(case_id: str) -> str:
            if not session_complete:
                return "AVAILABLE"
            if case_id in completed:
                return "COMPLETE"
            if case_id in failed:
                return "FAILED"
            if case_id in skipped:
                return "SKIPPED"
            return "AVAILABLE"

        artifacts = []
        for name, (kind, description, information) in _GLOBAL_ARTIFACTS.items():
            path = work_root / name
            if not path.exists():
                continue
            artifacts.append(
                {
                    "kind": kind,
                    "scope": "global",
                    "case_id": None,
                    "status": ("COMPLETE" if session_complete else "AVAILABLE"),
                    "path": output_path(path),
                    "description": description,
                    "available_information": information,
                }
            )
        for case in cases:
            case_root = work_root / "cases" / case.directory
            for name, (
                kind,
                description,
                information,
            ) in _CASE_ARTIFACTS.items():
                path = case_root / name
                if not path.exists():
                    continue
                if path.is_dir() and not any(path.iterdir()):
                    continue
                artifacts.append(
                    {
                        "kind": kind,
                        "scope": "case",
                        "case_id": case.case_id,
                        "status": case_status(case.case_id),
                        "path": output_path(path),
                        "description": description,
                        "available_information": information,
                    }
                )
        catalog_path = work_root / "artifact_catalog.json"
        artifacts.append(
            {
                "kind": "analysis_artifact_catalog",
                "scope": "global",
                "case_id": None,
                "status": ("COMPLETE" if session_complete else "AVAILABLE"),
                "path": output_path(catalog_path),
                "description": (
                    "Index of Analysis artifact paths, contents, status, and "
                    "available information for downstream agents."
                ),
                "available_information": [
                    "artifact discovery",
                    "partial checkpoint status",
                    "per-file semantics",
                ],
            }
        )
        _atomic_write_json(
            catalog_path,
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_commit": workflow.analysis_commit,
                "workflow_status": workflow.state["status"],
                "analysis_session_status": workflow.status,
                "artifacts": artifacts,
            },
        )
        return catalog_path

    def apply_checkpoint(
        self,
        context: OrchestrationContext,
    ) -> OrchestrationContext:
        """Expose validated partial Analysis outputs to downstream agents."""
        work_root = (
            Path(context.workspace).resolve() / "forge_experiments" / "analysis" / "work" / context.analysis_commit
        )
        workflow_path = work_root / "workflow.json"
        catalog_path = work_root / "artifact_catalog.json"
        if not workflow_path.is_file() or not catalog_path.is_file():
            return context
        try:
            workflow = json.loads(workflow_path.read_text())
            catalog = json.loads(catalog_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return context
        if not isinstance(workflow, dict) or not isinstance(catalog, dict):
            return context
        if (
            workflow.get("schema_version") != SESSION_SCHEMA_VERSION
            or workflow.get("analysis_commit") != context.analysis_commit
            or catalog.get("schema_version") != ANALYSIS_SCHEMA_VERSION
            or catalog.get("analysis_commit") != context.analysis_commit
        ):
            return context
        try:
            request_payload = json.loads((work_root / "request.json").read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return context
        if (
            not isinstance(request_payload, dict)
            or request_payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION
            or not isinstance(
                request_payload.get("analysis_profiling_enabled"),
                bool,
            )
        ):
            return context
        static_only = request_payload["analysis_profiling_enabled"] is False

        evidence_by_path = {reference.path: reference for reference in context.evidence_refs}
        artifacts = catalog.get("artifacts")
        if not isinstance(artifacts, list):
            return context
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = str(artifact.get("path") or "")
            status = str(artifact.get("status") or "")
            information = artifact.get("available_information")
            if (
                not path
                or status
                not in {
                    "AVAILABLE",
                    "COMPLETE",
                    "FAILED",
                    "SKIPPED",
                }
                or not isinstance(information, list)
            ):
                continue
            try:
                Path(path).resolve().relative_to(work_root.resolve())
            except ValueError:
                continue
            evidence_by_path[path] = EvidenceRef(
                kind=str(artifact.get("kind") or "analysis_artifact"),
                path=path,
                summary=(
                    f"{artifact.get('description') or 'Analysis artifact'} "
                    f"Available information: "
                    f"{', '.join(str(item) for item in information)}."
                ),
            )

        normalized_cases = []
        for original in context.cases:
            case_root = work_root / "cases" / _case_directory(original.case_id)
            bottleneck_path = case_root / "bottleneck.json"
            analysis_path = case_root / "analysis.md"
            normalized_path = case_root / "normalized_metrics.json"
            profile_root = case_root / "profile"
            bottleneck = original.bottleneck
            summary_path = original.profile_summary_path
            flags = list(original.flags)
            if static_only:
                flags.append("analysis_static_only")
            if bottleneck_path.is_file() and analysis_path.is_file():
                try:
                    payload = json.loads(bottleneck_path.read_text())
                    bottleneck = str(payload.get("classification") or bottleneck)
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    bottleneck = original.bottleneck
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
            normalized_cases.append(
                CaseEvidence(
                    case_id=original.case_id,
                    shape=original.shape,
                    dtype=original.dtype,
                    latency_ms=original.latency_ms,
                    bottleneck=bottleneck,
                    profile_summary_path=summary_path,
                    flags=tuple(dict.fromkeys(flags)),
                )
            )
        source_map = work_root / "source_map.md"
        return OrchestrationContext(
            analysis_commit=context.analysis_commit,
            workspace=context.workspace,
            gpu_target=context.gpu_target,
            objective=context.objective,
            program_context=context.program_context,
            source_map_path=(str(source_map.resolve()) if source_map.is_file() else context.source_map_path),
            editable_sources=context.editable_sources,
            cases=tuple(normalized_cases),
            knowledge_index=context.knowledge_index,
            supervisor_guidance=context.supervisor_guidance,
            search_mode=context.search_mode,
            search_reason_codes=context.search_reason_codes,
            search_objective=context.search_objective,
            search_mode_residence_remaining=(context.search_mode_residence_remaining),
            evidence_refs=tuple(evidence_by_path.values()),
            canonical_commit=(context.canonical_commit or context.analysis_commit),
            evidence_commit=(context.evidence_commit or context.analysis_commit),
            evidence_stale=context.evidence_stale,
            evidence_status=(context.evidence_status or "partial_checkpoint"),
            evidence_mean_case_speedup=(context.evidence_mean_case_speedup),
            current_mean_case_speedup=context.current_mean_case_speedup,
            cumulative_diff_path=context.cumulative_diff_path,
            cumulative_diff_error=context.cumulative_diff_error,
        )

    def apply_published_evidence(
        self,
        context: OrchestrationContext,
        *,
        evidence_commit: str,
    ) -> OrchestrationContext:
        """Restore one published bundle as stale-safe planning evidence.

        Validation cross-checks the bundle's immutable request and manifest
        digests rather than comparing stale evidence with the current canonical
        source digest. Current case latencies remain authoritative while
        bottlenecks and per-case profile paths come from the evidence commit.
        """
        commit = str(evidence_commit or "").strip()
        if not commit:
            return context
        workspace = Path(context.workspace).resolve()
        commit_root = workspace / "forge_experiments" / "analysis" / commit
        root = self._published_generation_root(commit_root)
        if root is None:
            return context
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            request_payload = _parse_request_payload(
                json.loads((root / "request.json").read_text()),
                analysis_commit=commit,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            AnalysisBundleError,
        ):
            return context
        if not isinstance(manifest, dict):
            return context
        driver_digest = str(request_payload.get("driver_digest") or "")
        source_digest = str(request_payload.get("source_digest") or "")
        if manifest.get("driver_digest") != driver_digest or manifest.get("source_digest") != source_digest:
            return context

        cases = tuple(
            AnalysisCase(
                case_id=case.case_id,
                directory=_case_directory(case.case_id),
                latency_ms=case.latency_ms,
            )
            for case in context.cases
        )
        validation_context = replace(
            context,
            analysis_commit=commit,
            canonical_commit=(context.canonical_commit or context.analysis_commit),
        )
        try:
            bundle = self._validate_bundle(
                root,
                validation_context,
                cases,
                driver_digest=driver_digest,
                source_digest=source_digest,
            )
        except AnalysisBundleError:
            return context

        applied = bundle.apply(context)
        canonical_commit = context.canonical_commit or context.analysis_commit
        stale = commit != canonical_commit
        refs = {reference.path: reference for reference in applied.evidence_refs}
        if stale and applied.source_map_path:
            source_map = Path(applied.source_map_path).resolve()
            if source_map.is_file():
                refs[str(source_map)] = EvidenceRef(
                    kind="analysis_source_map",
                    path=str(source_map),
                    summary=(f"Source map produced with stale Analysis evidence at commit {commit}."),
                )
        normalized_cases = tuple(
            replace(
                case,
                flags=tuple(
                    dict.fromkeys(
                        [
                            *case.flags,
                            *(["analysis_evidence_stale"] if stale else []),
                        ]
                    )
                ),
            )
            for case in applied.cases
        )
        return replace(
            applied,
            source_map_path=(context.source_map_path if stale else applied.source_map_path),
            cases=normalized_cases,
            evidence_refs=tuple(refs.values()),
            canonical_commit=canonical_commit,
            evidence_commit=commit,
            evidence_stale=stale,
            evidence_status=context.evidence_status,
            evidence_mean_case_speedup=(context.evidence_mean_case_speedup),
            current_mean_case_speedup=context.current_mean_case_speedup,
            cumulative_diff_path=context.cumulative_diff_path,
            cumulative_diff_error=context.cumulative_diff_error,
        )

    async def _run_analysis_session(
        self,
        *,
        workflow: AnalysisSessionJournal,
        context: OrchestrationContext,
        work_root: Path,
        request_path: Path,
        kernel_file: Path,
        driver_script: Path,
        source_files: tuple[Path, ...],
        reference_script: Path,
        protection: _AnalysisProtection,
        usage,
        timeout_sec: float,
        force_refresh: bool,
    ) -> None:
        cases = tuple(
            AnalysisCase(
                case_id=case.case_id,
                directory=_case_directory(case.case_id),
                latency_ms=case.latency_ms,
            )
            for case in context.cases
        )
        outputs = self._session_outputs(work_root, cases)
        request = json.loads(request_path.read_text())

        def sync_checkpoint() -> None:
            self._write_artifact_catalog(work_root, workflow, cases)

        def verify_session_bundle() -> None:
            self._validate_bundle(
                work_root,
                context,
                cases,
                driver_digest=str(request["driver_digest"]),
                source_digest=str(request["source_digest"]),
            )

        sync_checkpoint()
        if not force_refresh:
            try:
                verify_session_bundle()
            except AnalysisBundleError:
                pass
            else:
                workflow.complete(outputs)
                sync_checkpoint()
                return

        workflow.begin()
        sync_checkpoint()
        try:
            system_prompt, user_prompt = self._prompts(
                context=context,
                staging_root=work_root,
                request_path=request_path,
                kernel_file=kernel_file,
                driver_script=driver_script,
                source_files=source_files,
                reference_script=reference_script,
                cases=cases,
                timeout_sec=timeout_sec,
            )
            hooks = protection.hooks()
            run_result = await asyncio.wait_for(
                run_session_with_api_resume(
                    self.backend,
                    AgentRunSpec(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        cwd=str(work_root),
                        writable=True,
                        timeout_sec=timeout_sec,
                        reasoning_effort="high",
                        additional_directories=[
                            context.workspace,
                            str(assert_sandbox_grant(self.config.local_knowledge_dir, what="local_knowledge_dir")),
                        ],
                        allow_untracked=True,
                        tool_policy=AgentToolPolicy(
                            read=True,
                            search=True,
                            write=True,
                            shell=True,
                            max_turns=self.max_turns,
                        ),
                        hooks=hooks,
                    ),
                    usage=usage,
                    deadline_sec=timeout_sec,
                ),
                timeout=watchdog_timeout_sec(timeout_sec),
            )
            if run_result.end_reason == EXHAUSTED_END_REASON:
                raise AnalysisBundleError(
                    "Analysis Agent API resume budget exhausted: "
                    + (run_result.stderr_tail or run_result.text or run_result.end_reason)[:1800]
                )
        except asyncio.CancelledError:
            workflow.fail("CancelledError: Analysis session cancelled")
            sync_checkpoint()
            raise
        except Exception as error:
            workflow.fail(f"{type(error).__name__}: {error}")
            sync_checkpoint()
            raise
        changed = protection.restore_and_report_changes()
        if changed:
            error = "Analysis Agent modified immutable inputs: " + ", ".join(changed)
            workflow.fail(error)
            sync_checkpoint()
            raise AnalysisBundleError("Analysis Agent modified immutable inputs; restored: " + ", ".join(changed))
        self._finalize_framework_artifacts(
            work_root,
            context,
            cases,
            driver_digest=str(request["driver_digest"]),
            source_digest=str(request["source_digest"]),
            profiling_enabled=bool(request["analysis_profiling_enabled"]),
        )
        try:
            verify_session_bundle()
        except Exception as error:
            workflow.fail(f"{type(error).__name__}: {error}")
            sync_checkpoint()
            raise
        workflow.complete(outputs)
        sync_checkpoint()

    def _prompts(
        self,
        *,
        context: OrchestrationContext,
        staging_root: Path,
        request_path: Path,
        kernel_file: Path,
        driver_script: Path,
        source_files: tuple[Path, ...],
        reference_script: Path,
        cases: tuple[AnalysisCase, ...],
        timeout_sec: float,
    ) -> tuple[str, str]:
        request = json.loads(request_path.read_text())
        knowledge_root = Path(self.config.local_knowledge_dir).resolve()
        profiling_root = staging_root / "tools" / "profiling"
        methodology_candidates = tuple(profiling_root / name for name in PROFILING_METHODOLOGY_FILES)
        methodology = tuple(path.resolve() for path in methodology_candidates if path.is_file())
        missing_methodology = tuple(path.name for path in methodology_candidates if not path.is_file())
        if self.profiling_enabled:
            system_prompt = """\
You are the single Analysis Agent for one immutable canonical kernel. Complete
the entire source, case, profiling, interpretation, potential, direction, and
summary workflow in this one session. Do not stop after planning or after one
case.

Do not optimize or edit the kernel, driver, tests, harness, or scoring inputs.
Correctness, accuracy, benchmark, KEEP, and REVERT remain controlled by the
outer loop. Write analysis artifacts only inside the supplied staging directory.
You may read and execute the canonical source and driver. Any temporary adapter,
script, cache, log, profiler output, or other generated file must live below the
staging directory.

Markdown-first output contract:
- Write one authoritative report.md with the cross-case findings, evidence
  interpretation, remaining headroom, and prioritized optimization directions.
- For each requested case, write cases/<directory>/analysis.md, or failure.md
  when the case cannot be analyzed. Preserve raw profiler output below that
  case's profile/ directory.
- Optional JSON or additional Markdown may be written when it improves the
  analysis, but it is never required for publication. Do not duplicate the same
  conclusion across several formats merely to satisfy a schema.
- request.json, case_inventory.json, progress.json, manifest.json,
  workflow.json, workflow_events.jsonl, and artifact_catalog.json are owned by
  the framework. Read them as inputs but do not edit them.
- Append profiler commands and outcomes to commands.jsonl when practical.
- Resume useful Markdown and raw evidence already present in staging instead of
  regenerating it.

The profiling reference has already been copied into the staging tools
directory. Inspect that exact file; do not search the filesystem for another
copy. If it does not fit this driver, create a separate adapter in staging/tools.
You may use other
profilers, compiler IR, ISA, register, occupancy, cache, or roofline tools when
useful. Preserve raw evidence and record every command in commands.jsonl.

Profiler safety contract:
- Never invoke rocprofv3 --pmc with an ad-hoc or oversized counter list.
- Prefer the supplied absolute rocpc_profile.py reference.
- If rocprof-compute is unavailable, its dependency preflight fails, or its
  collection cannot isolate the target kernel, fall back to rocprofv3
  kernel-trace plus small hardware-compatible PMC groups. Record why the
  fallback was selected.
- If raw PMC is necessary, run one hardware-compatible counter group per
  rocprofv3 process and use a separate output directory for every pass.
- Every profiler command must have a finite timeout.
- On error 38, SIGABRT, timeout, non-zero exit, or missing output, stop that
  counter group immediately and never retry the same group.
- Write failure.md for the affected case and continue with other cases.
- Record one JSON row per profiler command in commands.jsonl with the exact
  case_id, command, timeout, exit_code or returncode, success, output directory,
  signal, and failure reason. A case is considered profiled only when it also
  has non-empty normalized_metrics.json and successful command provenance.

Case grouping contract:
- You are NOT required to profile every test shape independently.
- You own the semantic grouping decision. Use source structure, dispatch
  behavior, shapes, measurements, and domain judgment to decide whether one
  representative profile can support the same bottleneck and optimization
  conclusion for multiple cases. The outer code does not hard-code grouping
  categories.
- Signals such as dispatch path, dtype/layout, algorithmic regime, size regime,
  and expected bottleneck are advisory evidence, not mandatory equality rules.
- Choose the case that is most representative and information-rich. It may
  appear anywhere in request order. Profile it before drawing conclusions for
  member cases.
- Reuse only conclusions that you believe transfer. Every member still needs its
  own case artifacts that identify the representative and reuse rationale.
- report.md must explain the grouping decision, representative cases,
  transferable evidence, and concrete rationale.

Finish the useful analysis in this session. Prefer a complete report, but return
PARTIAL evidence rather than spending turns repairing optional output formats.
"""
        else:
            system_prompt = """\
You are the single Analysis Agent for one immutable canonical kernel. Complete
the entire static source, case, potential, direction, and summary workflow in
this one session.

Do not optimize or edit the kernel, driver, tests, harness, or scoring inputs.
Correctness, accuracy, benchmark, KEEP, and REVERT remain controlled by the
outer loop. Write analysis artifacts only inside the supplied staging directory.
You may read and execute canonical files but may not modify them. Any generated
file must live below the staging directory.

Analysis profiling is disabled by campaign budget policy. Do not invoke a
profiler, collect hardware counters, adapt the driver, or present inferred
behavior as measured evidence. Use source, driver, case definitions, historical
canonical evidence, and domain reasoning only. Clearly label static inference.

Inspect existing artifacts first. Write report.md plus one analysis.md or
failure.md per case. Profiling is disabled, so clearly label every conclusion as
static inference. The framework owns control JSON and will publish the result as
PARTIAL without requiring you to repair optional formats.
"""
        if request["analysis_trigger"] == "post_keep_incremental":
            system_prompt += f"""

This is a cumulative re-analysis after one or more solutions were validated and
KEPT since the last Analysis refresh.
The previous analyzed commit is {request["previous_analysis_commit"]}.
Read the cumulative canonical diff at {request["incremental_diff_path"]} and the
previous published bundle at {request["previous_analysis_bundle"]} before doing
work. The diff may span multiple accepted KEEP commits.

Update analysis incrementally:
- Identify which source regions, dispatch paths, and cases changed across the
  accepted KEEP sequence.
- Re-profile only when the change invalidates previous measurements or when new
  evidence is necessary to assess the changed mechanism.
- Reuse unaffected parent analysis or profile artifacts by copying only the
  needed files into this staging directory. Clearly label reused measurements
  with their parent commit; never present inherited evidence as newly measured.
- Refresh report.md, source_map.md, and affected case analysis files so they
  describe the new canonical solution. A complete re-profile is not required.
"""
        system_prompt += (
            "\nThe exact profiling reference for this session is:\n"
            f"    {reference_script}\n"
            "It already exists inside staging. Use it directly and never search "
            "the root filesystem for another copy.\n"
        )
        user_payload = {
            "analysis_commit": context.analysis_commit,
            "analysis_mode": ("PROFILED" if self.profiling_enabled else "STATIC_ONLY"),
            "workspace": context.workspace,
            "kernel_file": str(kernel_file),
            "driver_script": str(driver_script),
            "source_files": [str(path) for path in source_files],
            "request_file": str(request_path),
            "analysis_staging_dir": str(staging_root),
            "workflow_file": str(staging_root / "workflow.json"),
            "workflow_events": str(staging_root / "workflow_events.jsonl"),
            "artifact_catalog": str(staging_root / "artifact_catalog.json"),
            "analysis_session": {
                "session_id": ANALYSIS_SESSION_STEP_ID,
                "agent_outputs": [
                    "report.md",
                    "source_map.md",
                    "cases/<directory>/analysis.md or failure.md",
                    "cases/<directory>/profile/ when profiling succeeds",
                    "optional Markdown or JSON supporting evidence",
                ],
                "instructions": (
                    "Produce useful Markdown analysis and raw evidence in this "
                    "single session. The framework generates and validates the "
                    "control manifest; do not spend turns repairing optional "
                    "serialization formats."
                ),
            },
            "analysis_trigger": request["analysis_trigger"],
            "analysis_session_timeout_sec": timeout_sec,
            "previous_analysis_commit": request["previous_analysis_commit"],
            "previous_analysis_bundle": request["previous_analysis_bundle"],
            "incremental_diff_path": request["incremental_diff_path"],
            "changed_source_files": request["changed_source_files"],
            "knowledge_root": str(knowledge_root),
            "knowledge_index": context.knowledge_index,
            "reference_profiling_script": str(reference_script),
            "profiling_methodology": [str(path) for path in methodology],
            "profiling_methodology_missing": list(missing_methodology),
            "cases": [
                {
                    **evidence.to_dict(),
                    "directory": case.directory,
                }
                for evidence, case in zip(context.cases, cases)
            ],
            "step_rules": [
                "Complete the full analysis_session in this single Agent run.",
                "Read existing durable outputs instead of regenerating completed work.",
                "Write each case's analysis.md immediately after its evidence is ready.",
                "Use the exact reference_profiling_script path; do not search for it.",
                (
                    "Begin every Bash command with `timeout --signal=TERM "
                    "--kill-after=5s <duration>` and keep duration within the "
                    "analysis_session_timeout_sec remaining budget."
                ),
                (
                    "Never treat pooled multi-case counters as a valid per-case profile."
                    if self.profiling_enabled
                    else "Do not run profiling or claim static inference as measured evidence."
                ),
                (
                    "Preserve raw profiler artifacts and separate measured facts from interpretation."
                    if self.profiling_enabled
                    else "Separate source facts, historical evidence, and static inference."
                ),
                "If a case cannot complete, write failure.md with a concrete reason.",
            ],
            "markdown_contract": {
                "global_report": "report.md",
                "source_map": "source_map.md",
                "per_case": "cases/<directory>/analysis.md or failure.md",
                "raw_profile": "cases/<directory>/profile/",
                "optional_directions": "cases/<directory>/directions.md",
            },
            "framework_owned_files": [
                "request.json",
                "case_inventory.json",
                "progress.json",
                "manifest.json",
                "workflow.json",
                "workflow_events.jsonl",
                "artifact_catalog.json",
                "framework_commands.jsonl",
                "published.json",
            ],
            "publication_policy": {
                "format_errors_block_publication": False,
                "missing_case_reports_publish_as": "PARTIAL",
                "optional_json": "catalogued when present, never required",
            },
        }
        return system_prompt, json.dumps(user_payload, indent=2, sort_keys=True)

    @staticmethod
    def _protected_paths(
        *,
        workspace: Path,
        kernel_file: str,
        driver_script: str,
        source_files: list[str],
    ) -> tuple[Path, ...]:
        paths = {
            Path(kernel_file).resolve(),
            Path(driver_script).resolve(),
            *(Path(path).resolve() for path in source_files),
        }
        result = git("ls-files", "-z", cwd=workspace, check=False, text=False)
        if result.returncode == 0:
            for raw in result.stdout.split(b"\0"):
                if raw:
                    paths.add((workspace / os.fsdecode(raw)).resolve())
        return tuple(sorted(paths))

    @staticmethod
    def _validate_bundle(
        root: Path,
        context: OrchestrationContext,
        cases: tuple[AnalysisCase, ...],
        *,
        driver_digest: str,
        source_digest: str,
    ) -> AnalysisBundle:
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as error:
            raise AnalysisBundleError(f"invalid manifest.json: {error}") from error
        if not isinstance(manifest, dict):
            raise AnalysisBundleError("manifest.json must be an object")
        if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            raise AnalysisBundleError("unsupported analysis manifest schema")
        if manifest.get("analysis_commit") != context.analysis_commit:
            raise AnalysisBundleError("analysis manifest commit does not match")
        if manifest.get("driver_digest") != driver_digest:
            raise AnalysisBundleError("analysis manifest driver digest does not match")
        if manifest.get("source_digest") != source_digest:
            raise AnalysisBundleError("analysis manifest source digest does not match")
        if manifest.get("status") not in {"READY", "PARTIAL", "FAILED"}:
            raise AnalysisBundleError("analysis manifest status is invalid")
        expected = [case.case_id for case in cases]
        if manifest.get("expected_case_ids") != expected:
            raise AnalysisBundleError("analysis manifest case inventory does not match")
        completed = manifest.get("completed_case_ids")
        failed = manifest.get("failed_case_ids")
        skipped = manifest.get("skipped_case_ids") or []
        if not isinstance(completed, list) or not isinstance(failed, list) or not isinstance(skipped, list):
            raise AnalysisBundleError("analysis manifest case statuses are invalid")
        if sorted(completed + failed + skipped) != sorted(expected):
            raise AnalysisBundleError("analysis manifest does not account for every case")

        request_path = root / "request.json"
        try:
            request_payload = _parse_request_payload(
                json.loads(request_path.read_text()),
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
        profiling_enabled = bool(request_payload["analysis_profiling_enabled"])
        if not profiling_enabled and completed:
            raise AnalysisBundleError("static-only analysis cannot report profiled COMPLETE cases")
        if not profiling_enabled and manifest.get("status") != "PARTIAL":
            raise AnalysisBundleError("static-only analysis manifest status must be PARTIAL")
        required_root_files = (
            "request.json",
            "workflow.json",
            "workflow_events.jsonl",
            "artifact_catalog.json",
            "source_map.md",
            "case_inventory.json",
            "progress.json",
            "commands.jsonl",
        )
        missing_root = [name for name in required_root_files if not (root / name).is_file()]
        if missing_root:
            raise AnalysisBundleError("analysis bundle missing root artifacts: " + ", ".join(missing_root))
        if not AnalysisAgentService._nonempty_file(root / "report.md"):
            raise AnalysisBundleError("analysis bundle has no report.md")

        normalized_cases = []
        completed_set = set(completed)
        skipped_set = set(skipped)
        for case, original in zip(cases, context.cases):
            case_root = root / "cases" / case.directory
            if not case_root.is_dir():
                raise AnalysisBundleError(f"missing case directory for {case.case_id}")
            try:
                case_payload = json.loads((case_root / "case.json").read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise AnalysisBundleError(f"invalid case.json for {case.case_id}: {error}") from error
            if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
                raise AnalysisBundleError(f"case.json identity mismatch for {case.case_id}")
            if case.case_id in completed_set:
                analysis_path = case_root / "analysis.md"
                if not AnalysisAgentService._nonempty_file(analysis_path):
                    raise AnalysisBundleError(f"completed case {case.case_id} has no analysis.md")
                if profiling_enabled and not AnalysisAgentService._has_valid_profile_evidence(
                    root,
                    case,
                ):
                    raise AnalysisBundleError(f"completed case {case.case_id} has no validated profile evidence")
                bottleneck = {}
                bottleneck_path = case_root / "bottleneck.json"
                if bottleneck_path.is_file():
                    try:
                        loaded = json.loads(bottleneck_path.read_text())
                        if isinstance(loaded, dict):
                            bottleneck = loaded
                    except (OSError, json.JSONDecodeError):
                        bottleneck = {}
                profile_flag = "analysis_profiled"
                if request_payload["analysis_trigger"] == "post_keep_incremental":
                    profile_flag = "analysis_profile_incremental"
                normalized_cases.append(
                    CaseEvidence(
                        case_id=case.case_id,
                        shape=original.shape,
                        dtype=original.dtype,
                        latency_ms=original.latency_ms,
                        bottleneck=str(bottleneck.get("classification") or original.bottleneck),
                        profile_summary_path=str(analysis_path),
                        flags=tuple(
                            dict.fromkeys(
                                [
                                    *(str(item) for item in (bottleneck.get("flags") or [])),
                                    profile_flag,
                                ]
                            )
                        ),
                    )
                )
            elif case.case_id in skipped_set:
                analysis_path = case_root / "analysis.md"
                has_analysis = AnalysisAgentService._nonempty_file(analysis_path)
                normalized_cases.append(
                    CaseEvidence(
                        case_id=original.case_id,
                        shape=original.shape,
                        dtype=original.dtype,
                        latency_ms=original.latency_ms,
                        bottleneck=original.bottleneck,
                        profile_summary_path=(str(analysis_path) if has_analysis else original.profile_summary_path),
                        flags=tuple(
                            dict.fromkeys(
                                [
                                    *original.flags,
                                    (
                                        "analysis_static_only"
                                        if not profiling_enabled
                                        else (
                                            "analysis_interpretation_only"
                                            if has_analysis
                                            else "analysis_profile_skipped"
                                        )
                                    ),
                                ]
                            )
                        ),
                    )
                )
            else:
                if not (AnalysisAgentService._nonempty_file(case_root / "failure.md")):
                    raise AnalysisBundleError(f"failed case {case.case_id} has no failure record")
                normalized_cases.append(
                    CaseEvidence(
                        case_id=original.case_id,
                        shape=original.shape,
                        dtype=original.dtype,
                        latency_ms=original.latency_ms,
                        bottleneck=original.bottleneck,
                        profile_summary_path=(original.profile_summary_path),
                        flags=tuple(dict.fromkeys([*original.flags, "analysis_case_failed"])),
                    )
                )
        return AnalysisBundle(
            analysis_commit=context.analysis_commit,
            root=root,
            manifest=manifest,
            cases=tuple(normalized_cases),
        )

    @staticmethod
    def _published_generation_root(commit_root: Path) -> Path | None:
        """Resolve the active immutable generation for one analysis commit."""
        pointer_path = commit_root / "published.json"
        if pointer_path.is_file():
            try:
                pointer = json.loads(pointer_path.read_text())
            except (OSError, json.JSONDecodeError):
                pointer = {}
            if isinstance(pointer, dict):
                generation_name = str(pointer.get("generation_root") or pointer.get("artifact_root") or "")
                if generation_name:
                    candidate = (commit_root / generation_name).resolve()
                    if candidate.is_dir():
                        return candidate
        generations = sorted(commit_root.glob("generation-*"))
        if generations:
            return generations[-1]
        if (commit_root / "manifest.json").is_file():
            return commit_root
        return None

    @staticmethod
    def _next_generation_root(commit_root: Path) -> Path:
        generation = 1
        while True:
            candidate = commit_root / f"generation-{generation:03d}"
            if not candidate.exists():
                return candidate
            generation += 1

    @staticmethod
    def _publish_generation(staging_root: Path, commit_root: Path) -> Path:
        """Publish one immutable analysis generation without moving prior bundles."""
        commit_root.mkdir(parents=True, exist_ok=True)
        generation_root = AnalysisAgentService._next_generation_root(commit_root)
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        directory_fd = os.open(str(staging_root), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary = Path(
            tempfile.mkdtemp(
                dir=str(commit_root),
                prefix=f".{generation_root.name}.",
            )
        )
        try:
            shutil.copytree(staging_root, temporary, dirs_exist_ok=True)
            AnalysisAgentService._fsync_tree(temporary)
            os.replace(temporary, generation_root)
            parent_fd = os.open(str(commit_root), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        _atomic_write_json(
            commit_root / "published.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "generation_root": generation_root.name,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return generation_root

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        directory_fd = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _tier_label(*, profiling_enabled: bool, profiled: bool) -> str:
        if profiling_enabled and profiled:
            return "profiled"
        if profiling_enabled:
            return "partial"
        return "static"

    def _build_outcome(
        self,
        *,
        analysis_commit: str,
        requested_tier: str,
        available_tier: str,
        attempt: int,
        checkpoint_level: str,
        artifact_path: str = "",
        failure_type: str | None = None,
        upgrade_exhausted: bool = False,
        parent_reuse_commit: str = "",
    ) -> AnalysisOutcome:
        return AnalysisOutcome(
            analysis_commit=analysis_commit,
            requested_tier=requested_tier,
            available_tier=available_tier,
            attempt=attempt,
            checkpoint_level=checkpoint_level,
            artifact_path=artifact_path,
            failure_type=failure_type,
            upgrade_exhausted=upgrade_exhausted,
            parent_reuse_commit=parent_reuse_commit,
        )


def make_analysis_agent_service(
    *,
    config: Config,
    usage=None,
    timeout_sec: int | None = None,
    profiling_enabled: bool = True,
) -> AnalysisAgentService:
    """Build the Analysis Agent through the configured provider."""
    runtime = config.agent_runtime()
    backend = create_registered_backend(
        runtime,
        probe_cwd=config.workspace,
        usage=usage,
    )
    return AnalysisAgentService(
        backend=backend,
        config=config,
        timeout_sec=(timeout_sec if timeout_sec is not None else backend.runtime.timeout_sec),
        max_turns=config.max_turns,
        profiling_enabled=profiling_enabled,
    )
