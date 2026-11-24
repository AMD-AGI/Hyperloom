"""Commit-bound profiling and analysis bundle produced by one Agent session."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.utils.git import git_bytes
from kernelforge.agent_backends import (
    AgentBackend,
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON, ResumePolicy, run_session_with_api_resume
from kernelforge.config import Config
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.workflows.forge_loop.roles.contracts import (
    EvidenceRef,
    OrchestrationContext,
)
from kernelforge.workflows.forge_loop.roles.analysis_bundle import (
    AnalysisBundle,
    AnalysisBundleError,
    AnalysisCase,
    AnalysisOutcome,
    _analysis_cases,
    _BundleIdentity,
    _case_directory,
    _checkpoint_artifact_refs,
    _checkpoint_case_evidence,
    _has_valid_profile_evidence,
    _nonempty_file,
    _parse_request_payload,
    _parse_workflow_payload,
    _published_generation_root,
    _source_digest,
    _tier_label,
    _validate_bundle,
)
from kernelforge.workflows.forge_loop.roles.analysis_catalog import _write_artifact_catalog
from kernelforge.workflows.forge_loop.roles.analysis_protection import AnalysisProtection
from kernelforge.workflows.forge_loop.roles.analysis_prompts import (
    ANALYSIS_SESSION_BRIEF,
    FRAMEWORK_OWNED_FILES,
    INCREMENTAL_REANALYSIS_PROMPT,
    MARKDOWN_CONTRACT,
    PROFILED_SYSTEM_PROMPT,
    PUBLICATION_POLICY,
    STATIC_SYSTEM_PROMPT,
)
from kernelforge.workflows.forge_loop.roles.analysis_session import (
    AnalysisAttemptLimitError,
    AnalysisSessionJournal,
    MAX_ANALYSIS_SESSION_ATTEMPTS,
    _sha256,
)
from hyperloom_common.io import DURABLE, JsonFormat, atomic_write_json, fsync_directory, fsync_tree
from kernelforge.config.resources import assert_sandbox_grant


ANALYSIS_SESSION_STEP_ID = "analysis_session"
PROFILING_METHODOLOGY_FILES = (
    "measure_rocpc_workflow.md",
    "measure_triage.md",
    "measure_roofline.md",
    "measure_protocol.md",
)

_INCREMENTAL_DIFF_FILE = "incremental_diff.patch"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncrementalAnalysisInput:
    """Describe a KEEP-derived commit relative to its analyzed parent."""

    parent_commit: str
    parent_bundle: Path
    commit_diff: str
    changed_source_files: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisSources:
    """The canonical kernel, its driver and the source files one analysis reads and keeps immutable."""

    kernel_file: str
    driver_script: str
    source_files: list[str]


class AnalysisConfigurationError(AnalysisBundleError):
    """Report a missing packaged resource or other non-retryable setup error."""


@dataclass(frozen=True)
class _AnalysisSession:
    """One session: the context, its staging root and request, the resolved inputs, profiling reference and budget."""

    context: OrchestrationContext
    work_root: Path
    request_path: Path
    kernel_file: Path
    driver_script: Path
    source_files: tuple[Path, ...]
    reference_script: Path
    timeout_sec: float


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
                atomic_write_json(
                    case_root / "case.json", case.to_dict(), fmt=JsonFormat(trailing_newline=True), options=DURABLE
                )
        inventory_path = work_root / "case_inventory.json"
        if not inventory_path.is_file():
            atomic_write_json(
                inventory_path,
                {
                    "analysis_commit": context.analysis_commit,
                    "cases": [case.to_dict() for case in cases],
                },
                fmt=JsonFormat(trailing_newline=True),
                options=DURABLE,
            )
        progress_path = work_root / "progress.json"
        if not progress_path.is_file():
            atomic_write_json(
                progress_path,
                {
                    "analysis_commit": context.analysis_commit,
                    "status": "RUNNING",
                    "cases": [{"case_id": case.case_id, "status": "PENDING"} for case in cases],
                },
                fmt=JsonFormat(trailing_newline=True),
                options=DURABLE,
            )

    @classmethod
    def _finalize_framework_artifacts(
        cls, work_root: Path, identity: _BundleIdentity, *, profiling_enabled: bool
    ) -> None:
        context = identity.context
        cases = identity.cases
        completed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        case_states = []
        for case in cases:
            case_root = work_root / "cases" / case.directory
            atomic_write_json(
                case_root / "case.json", case.to_dict(), fmt=JsonFormat(trailing_newline=True), options=DURABLE
            )
            analysis_path = case_root / "analysis.md"
            profile_root = case_root / "profile"
            has_analysis = _nonempty_file(analysis_path)
            has_profile = _has_valid_profile_evidence(
                work_root,
                case,
            )
            has_failure = _nonempty_file(case_root / "failure.md")
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
        cls._write_case_progress(work_root, context.analysis_commit, status, case_states)
        atomic_write_json(
            work_root / "manifest.json",
            {
                "analysis_commit": context.analysis_commit,
                "driver_digest": identity.driver_digest,
                "source_digest": identity.source_digest,
                "status": status,
                "expected_case_ids": [case.case_id for case in cases],
                "completed_case_ids": completed,
                "failed_case_ids": failed,
                "skipped_case_ids": skipped,
                "report": "report.md",
            },
            fmt=JsonFormat(trailing_newline=True),
            options=DURABLE,
        )

    @staticmethod
    def _write_case_progress(work_root: Path, analysis_commit: str, status: str, case_states: list[dict]) -> None:
        """Write the case inventory and the per-case progress the session finalized."""
        atomic_write_json(
            work_root / "case_inventory.json",
            {
                "analysis_commit": analysis_commit,
                "cases": case_states,
            },
            fmt=JsonFormat(trailing_newline=True),
            options=DURABLE,
        )
        atomic_write_json(
            work_root / "progress.json",
            {
                "analysis_commit": analysis_commit,
                "status": status,
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "status": case["status"],
                    }
                    for case in case_states
                ],
            },
            fmt=JsonFormat(trailing_newline=True),
            options=DURABLE,
        )

    @staticmethod
    def _exhausted_upgrade_bundle(cached: AnalysisBundle, published: AnalysisOutcome) -> AnalysisBundle:
        """The cached bundle, marked as one no further session will upgrade."""
        return AnalysisBundle(
            analysis_commit=cached.analysis_commit,
            root=cached.root,
            manifest={**cached.manifest, "upgrade_exhausted": True},
            cases=cached.cases,
            outcome=replace(published, failure_type="upgrade_exhausted", upgrade_exhausted=True),
        )

    @staticmethod
    def _work_root_attempts(work_root: Path, analysis_commit: str) -> int:
        """Attempts already spent in the work root, or the limit when it cannot be read.

        An unreadable journal counts as exhausted: another session would resume
        from a checkpoint nothing can say anything about.
        """
        try:
            work_flow = _parse_workflow_payload(
                json.loads((work_root / "workflow.json").read_text()),
                analysis_commit=analysis_commit,
            )
            return int(work_flow.get("session", {}).get("attempts", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError, AnalysisBundleError):
            return MAX_ANALYSIS_SESSION_ATTEMPTS

    def _bundle_from_published_checkpoint(
        self, published_root: Path, work_root: Path, identity: _BundleIdentity, parent_reuse_commit: str
    ) -> AnalysisBundle | None:
        """The bundle a published checkpoint already answers with, or None to run a session.

        Returning None also seeds ``work_root`` from the checkpoint, so the
        retry resumes from it rather than from nothing.
        """
        context = identity.context
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
        cached = _validate_bundle(published_root, identity)
        attempts = int(workflow_payload["session"].get("attempts", 0))
        cached_profiled = request_payload["analysis_profiling_enabled"] is True
        tier_satisfied = cached_profiled or not self.profiling_enabled
        status_satisfied = cached.manifest["status"] == "READY" if self.profiling_enabled else True
        available_tier = _tier_label(
            profiling_enabled=self.profiling_enabled,
            profiled=cached_profiled and bool(cached.manifest.get("completed_case_ids")),
        )
        published = AnalysisOutcome(
            analysis_commit=context.analysis_commit,
            requested_tier="profiled" if self.profiling_enabled else "static",
            available_tier=available_tier,
            attempt=attempts,
            checkpoint_level="published",
            artifact_path=str(cached.root),
            parent_reuse_commit=parent_reuse_commit,
        )
        if tier_satisfied and (status_satisfied or attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS):
            if not status_satisfied and attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
                return self._exhausted_upgrade_bundle(cached, published)
            return AnalysisBundle(
                analysis_commit=cached.analysis_commit,
                root=cached.root,
                manifest=dict(cached.manifest),
                cases=cached.cases,
                outcome=published,
            )
        if attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
            return self._exhausted_upgrade_bundle(cached, published)
        if work_root.is_dir():
            work_attempts = self._work_root_attempts(work_root, context.analysis_commit)
            if work_attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
                return self._exhausted_upgrade_bundle(cached, replace(published, attempt=work_attempts))
        if not work_root.exists():
            work_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(published_root, work_root)
        return None

    async def ensure_bundle(
        self,
        context: OrchestrationContext,
        *,
        sources: AnalysisSources,
        usage=None,
        deadline_unix: float | None = None,
        incremental: IncrementalAnalysisInput | None = None,
    ) -> AnalysisBundle:
        """Return a current bundle, running the Analysis Agent when absent."""
        workspace = Path(context.workspace).resolve()
        analysis_root = workspace / "forge_experiments" / "analysis"
        final_root = analysis_root / context.analysis_commit
        work_root = analysis_root / "work" / context.analysis_commit
        cases = _analysis_cases(context)
        kernel_path = Path(sources.kernel_file).resolve()
        resolved_sources = tuple(sorted({kernel_path, *(Path(path).resolve() for path in sources.source_files)}))
        driver_path = Path(sources.driver_script).resolve()
        identity = _BundleIdentity(context, cases, _sha256(driver_path), _source_digest(resolved_sources))
        parent_reuse_commit = incremental.parent_commit if incremental is not None else ""
        published_root = _published_generation_root(final_root)
        retry_published_bundle = published_root is not None
        if published_root is not None:
            reused = self._bundle_from_published_checkpoint(published_root, work_root, identity, parent_reuse_commit)
            if reused is not None:
                return reused
        session_deadline_unix = min(
            time.time() + self.timeout_sec,
            deadline_unix if deadline_unix is not None else float("inf"),
        )
        session_timeout_sec = session_deadline_unix - time.time()
        if session_timeout_sec <= 1:
            raise AnalysisBundleError("Analysis deadline exhausted before session start")
        (work_root / "tools").mkdir(parents=True, exist_ok=True)
        session = _AnalysisSession(
            context,
            work_root,
            work_root / "request.json",
            kernel_path,
            driver_path,
            resolved_sources,
            self._stage_reference_profiling_script(work_root),
            session_timeout_sec,
        )
        self._initialize_framework_artifacts(work_root, context, cases)
        incremental = self._stage_incremental_diff(analysis_root, work_root, incremental)
        protection = AnalysisProtection(
            workspace=workspace,
            staging_root=work_root,
            protected_paths=self._protected_paths(workspace, sources),
            deadline_monotonic=(time.monotonic() + session_timeout_sec),
        )
        self._stage_request(session, identity, incremental, retry_published_bundle=retry_published_bundle)
        (work_root / "commands.jsonl").touch(exist_ok=True)
        workflow = await self._run_protected_session(
            session, identity, protection, usage=usage, retry_published_bundle=retry_published_bundle
        )
        changed = protection.restore_and_report_changes()
        if changed:
            raise AnalysisBundleError("Analysis Agent modified immutable inputs; restored: " + ", ".join(changed))
        return self._publish_session_bundle(work_root, final_root, identity, workflow, incremental)

    async def _run_protected_session(
        self,
        session: _AnalysisSession,
        identity: _BundleIdentity,
        protection: AnalysisProtection,
        *,
        usage,
        retry_published_bundle: bool,
    ) -> AnalysisSessionJournal:
        """Run the journaled session, restoring the protected inputs before any failure leaves."""
        work_root, context = session.work_root, identity.context
        try:
            workflow = AnalysisSessionJournal(
                work_root,
                analysis_commit=context.analysis_commit,
                driver_digest=identity.driver_digest,
                source_digest=identity.source_digest,
            )
            if retry_published_bundle:
                workflow.reopen()
            await self._run_analysis_session(
                session, workflow=workflow, protection=protection, usage=usage, force_refresh=retry_published_bundle
            )
        except (asyncio.CancelledError, AnalysisAttemptLimitError):
            protection.restore_and_report_changes()
            raise
        except Exception as error:
            changed = protection.restore_and_report_changes()
            raise AnalysisBundleError(
                f"Analysis workflow failed: {type(error).__name__}: {error}; "
                f"checkpoint={work_root}; restored inputs={changed}"
            ) from error
        return workflow

    def _stage_incremental_diff(
        self, analysis_root: Path, work_root: Path, incremental: IncrementalAnalysisInput | None
    ) -> IncrementalAnalysisInput | None:
        """Stage the diff an incremental session reads; None once its parent bundle is no longer published."""
        incremental_diff_path = work_root / _INCREMENTAL_DIFF_FILE
        if incremental is not None:
            parent_commit_root = (analysis_root / incremental.parent_commit).resolve()
            parent_root = _published_generation_root(parent_commit_root)
            expected_parent = incremental.parent_bundle.resolve()
            if parent_root is None or parent_root != expected_parent or not parent_root.is_dir():
                # The current canonical can still be analyzed from scratch.
                incremental = None
        if incremental is not None:
            incremental_diff_path.write_text(incremental.commit_diff)
        else:
            incremental_diff_path.unlink(missing_ok=True)
        return incremental

    def _stage_request(
        self,
        session: _AnalysisSession,
        identity: _BundleIdentity,
        incremental: IncrementalAnalysisInput | None,
        *,
        retry_published_bundle: bool,
    ) -> None:
        """Write the session's request, or check that a resumed session's durable one still answers the same inputs."""
        incremental_diff_path = session.work_root / _INCREMENTAL_DIFF_FILE
        request_payload = {
            "analysis_commit": session.context.analysis_commit,
            "workspace": str(Path(session.context.workspace).resolve()),
            "kernel_file": str(session.kernel_file),
            "driver_script": str(session.driver_script),
            "driver_digest": identity.driver_digest,
            "source_files": [str(path) for path in session.source_files],
            "source_digest": identity.source_digest,
            "cases": [case.to_dict() for case in identity.cases],
            "objective": session.context.objective,
            "analysis_profiling_enabled": self.profiling_enabled,
            "reference_profiling_script": str(session.reference_script),
            "reference_profiling_script_sha256": _sha256(session.reference_script),
            "analysis_trigger": ("post_keep_incremental" if incremental is not None else "canonical_baseline"),
            "previous_analysis_commit": (incremental.parent_commit if incremental is not None else ""),
            "previous_analysis_bundle": (str(incremental.parent_bundle.resolve()) if incremental is not None else ""),
            "incremental_diff_path": (str(incremental_diff_path.resolve()) if incremental is not None else ""),
            "incremental_diff_sha256": (_sha256(incremental_diff_path) if incremental is not None else ""),
            "changed_source_files": (list(incremental.changed_source_files) if incremental is not None else []),
        }
        if not session.request_path.is_file():
            atomic_write_json(
                session.request_path, request_payload, fmt=JsonFormat(trailing_newline=True), options=DURABLE
            )
            return
        try:
            durable_request = json.loads(session.request_path.read_text())
        except json.JSONDecodeError as error:
            raise AnalysisBundleError(f"durable analysis request is invalid: {error}") from error
        immutable_keys = {
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
        if any(durable_request.get(key) != request_payload[key] for key in immutable_keys):
            raise AnalysisBundleError("durable analysis request does not match current inputs")
        if retry_published_bundle:
            durable_request["analysis_profiling_enabled"] = self.profiling_enabled
            atomic_write_json(
                session.request_path, durable_request, fmt=JsonFormat(trailing_newline=True), options=DURABLE
            )

    def _publish_session_bundle(
        self,
        work_root: Path,
        final_root: Path,
        identity: _BundleIdentity,
        workflow: AnalysisSessionJournal,
        incremental: IncrementalAnalysisInput | None,
    ) -> AnalysisBundle:
        """Publish a finished session's bundle as the commit's next generation, with the outcome it reached."""
        staged = _validate_bundle(work_root, identity)
        workflow.finalize(str(staged.manifest["status"]))
        generation_root = self._publish_generation(work_root, final_root)
        _write_artifact_catalog(generation_root, workflow, identity.cases, artifact_root=generation_root)
        validated = _validate_bundle(generation_root, identity)
        return replace(
            validated,
            outcome=AnalysisOutcome(
                analysis_commit=identity.context.analysis_commit,
                requested_tier="profiled" if self.profiling_enabled else "static",
                available_tier=_tier_label(
                    profiling_enabled=self.profiling_enabled,
                    profiled=bool(validated.manifest.get("completed_case_ids")),
                ),
                attempt=workflow.attempts,
                checkpoint_level="published",
                artifact_path=str(validated.root),
                parent_reuse_commit=incremental.parent_commit if incremental is not None else "",
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
        incremental_diff = work_root / _INCREMENTAL_DIFF_FILE
        if incremental_diff.is_file():
            outputs.append(incremental_diff)
        return tuple(outputs)

    def apply_checkpoint(self, context: OrchestrationContext) -> OrchestrationContext:
        """Expose validated partial Analysis outputs to downstream agents."""
        work_root = (
            Path(context.workspace).resolve() / "forge_experiments" / "analysis" / "work" / context.analysis_commit
        )
        try:
            workflow = json.loads((work_root / "workflow.json").read_text())
            catalog = json.loads((work_root / "artifact_catalog.json").read_text())
            request_payload = json.loads((work_root / "request.json").read_text())
        except (OSError, ValueError):
            return context
        if (
            not isinstance(workflow, dict)
            or not isinstance(catalog, dict)
            or not isinstance(request_payload, dict)
            or workflow.get("analysis_commit") != context.analysis_commit
            or catalog.get("analysis_commit") != context.analysis_commit
            or not isinstance(request_payload.get("analysis_profiling_enabled"), bool)
            or not isinstance(catalog.get("artifacts"), list)
        ):
            return context
        evidence_by_path = {reference.path: reference for reference in context.evidence_refs}
        evidence_by_path.update(_checkpoint_artifact_refs(catalog["artifacts"], work_root=work_root))
        normalized_cases = tuple(
            _checkpoint_case_evidence(
                original,
                case_root=work_root / "cases" / _case_directory(original.case_id),
                profiling_enabled=request_payload["analysis_profiling_enabled"],
            )
            for original in context.cases
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
            cases=normalized_cases,
            knowledge_index=context.knowledge_index,
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

    def apply_published_evidence(self, context: OrchestrationContext, *, evidence_commit: str) -> OrchestrationContext:
        """Restore one published bundle as stale-safe planning evidence."""
        commit_root = Path(context.workspace).resolve() / "forge_experiments" / "analysis" / evidence_commit
        root = _published_generation_root(commit_root)
        if root is None:
            return context
        try:
            request_payload = _parse_request_payload(
                json.loads((root / "request.json").read_text()), analysis_commit=evidence_commit
            )
        except (OSError, ValueError, AnalysisBundleError):
            return context
        identity = _BundleIdentity(
            replace(context, analysis_commit=evidence_commit),
            _analysis_cases(context),
            str(request_payload.get("driver_digest") or ""),
            str(request_payload.get("source_digest") or ""),
        )
        try:
            bundle = _validate_bundle(root, identity)
        except AnalysisBundleError:
            return context
        applied = bundle.apply(context)
        if applied.evidence_stale:
            source_map = Path(applied.source_map_path).resolve()
            refs = {reference.path: reference for reference in applied.evidence_refs}
            refs[str(source_map)] = EvidenceRef(
                kind="analysis_source_map",
                path=str(source_map),
                summary=f"Source map produced with stale Analysis evidence at commit {evidence_commit}.",
            )
            applied = replace(
                applied,
                source_map_path=context.source_map_path,
                cases=tuple(
                    replace(case, flags=tuple(dict.fromkeys([*case.flags, "analysis_evidence_stale"])))
                    for case in applied.cases
                ),
                evidence_refs=tuple(refs.values()),
            )
        return replace(applied, evidence_status=context.evidence_status)

    def _session_spec(
        self, session: _AnalysisSession, cases: tuple[AnalysisCase, ...], protection: AnalysisProtection
    ) -> AgentRunSpec:
        """The one Analysis session: it writes only into its staging root and reads the workspace and knowledge."""
        system_prompt, user_prompt = self._prompts(session, cases)
        return AgentRunSpec(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            role="analysis",
            cwd=str(session.work_root),
            writable=True,
            timeout_sec=session.timeout_sec,
            additional_directories=[
                session.context.workspace,
                str(assert_sandbox_grant(self.config.local_knowledge_dir, what="local_knowledge_dir")),
            ],
            allow_untracked=True,
            tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True, max_turns=self.max_turns),
            hooks=protection.hooks(),
            protected_path_rules=FORGE_PROTECTED_PATHS,
        )

    async def _run_analysis_session(
        self,
        session: _AnalysisSession,
        *,
        workflow: AnalysisSessionJournal,
        protection: AnalysisProtection,
        usage,
        force_refresh: bool,
    ) -> None:
        context = session.context
        work_root = session.work_root
        timeout_sec = session.timeout_sec
        cases = _analysis_cases(context)
        outputs = self._session_outputs(work_root, cases)
        request = json.loads(session.request_path.read_text())
        identity = _BundleIdentity(context, cases, str(request["driver_digest"]), str(request["source_digest"]))

        def sync_checkpoint() -> None:
            _write_artifact_catalog(work_root, workflow, cases)

        def verify_session_bundle() -> None:
            _validate_bundle(work_root, identity)

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
            run_result = await asyncio.wait_for(
                run_session_with_api_resume(
                    self.backend,
                    self._session_spec(session, cases, protection),
                    usage=usage,
                    policy=ResumePolicy(deadline_sec=timeout_sec),
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
            work_root, identity, profiling_enabled=bool(request["analysis_profiling_enabled"])
        )
        try:
            verify_session_bundle()
        except Exception as error:
            workflow.fail(f"{type(error).__name__}: {error}")
            sync_checkpoint()
            raise
        workflow.complete(outputs)
        sync_checkpoint()

    def _prompts(self, session: _AnalysisSession, cases: tuple[AnalysisCase, ...]) -> tuple[str, str]:
        context = session.context
        staging_root = session.work_root
        reference_script = session.reference_script
        request = json.loads(session.request_path.read_text())
        knowledge_root = Path(self.config.local_knowledge_dir).resolve()
        profiling_root = staging_root / "tools" / "profiling"
        methodology_candidates = tuple(profiling_root / name for name in PROFILING_METHODOLOGY_FILES)
        methodology = tuple(path.resolve() for path in methodology_candidates if path.is_file())
        missing_methodology = tuple(path.name for path in methodology_candidates if not path.is_file())
        system_prompt = self._system_prompt(request, reference_script)
        user_payload = {
            "analysis_commit": context.analysis_commit,
            "analysis_mode": ("PROFILED" if self.profiling_enabled else "STATIC_ONLY"),
            "workspace": context.workspace,
            "kernel_file": str(session.kernel_file),
            "driver_script": str(session.driver_script),
            "source_files": [str(path) for path in session.source_files],
            "request_file": str(session.request_path),
            "analysis_staging_dir": str(staging_root),
            "workflow_file": str(staging_root / "workflow.json"),
            "workflow_events": str(staging_root / "workflow_events.jsonl"),
            "artifact_catalog": str(staging_root / "artifact_catalog.json"),
            "analysis_session": {"session_id": ANALYSIS_SESSION_STEP_ID, **ANALYSIS_SESSION_BRIEF},
            "analysis_trigger": request["analysis_trigger"],
            "analysis_session_timeout_sec": session.timeout_sec,
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
            "step_rules": self._step_rules(),
            "markdown_contract": MARKDOWN_CONTRACT,
            "framework_owned_files": FRAMEWORK_OWNED_FILES,
            "publication_policy": PUBLICATION_POLICY,
        }
        return system_prompt, json.dumps(user_payload, indent=2, sort_keys=True)

    def _system_prompt(self, request: dict, reference_script: Path) -> str:
        """The Analysis Agent's instructions for this tier, trigger and profiling reference."""
        if self.profiling_enabled:
            system_prompt = PROFILED_SYSTEM_PROMPT
        else:
            system_prompt = STATIC_SYSTEM_PROMPT
        if request["analysis_trigger"] == "post_keep_incremental":
            system_prompt += INCREMENTAL_REANALYSIS_PROMPT.format(
                previous_analysis_commit=request["previous_analysis_commit"],
                incremental_diff_path=request["incremental_diff_path"],
                previous_analysis_bundle=request["previous_analysis_bundle"],
            )
        system_prompt += (
            "\nThe exact profiling reference for this session is:\n"
            f"    {reference_script}\n"
            "It already exists inside staging. Use it directly and never search "
            "the root filesystem for another copy.\n"
        )
        return system_prompt

    def _step_rules(self) -> list:
        """The rules every step of the session follows, for this profiling tier."""
        return [
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
        ]

    @staticmethod
    def _protected_paths(workspace: Path, sources: AnalysisSources) -> tuple[Path, ...]:
        paths = {
            Path(sources.kernel_file).resolve(),
            Path(sources.driver_script).resolve(),
            *(Path(path).resolve() for path in sources.source_files),
        }
        result = git_bytes("ls-files", "-z", cwd=workspace, check=False)
        if result.returncode == 0:
            for raw in result.stdout.split(b"\0"):
                if raw:
                    paths.add((workspace / os.fsdecode(raw)).resolve())
        return tuple(sorted(paths))

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
        fsync_tree(staging_root)
        temporary = Path(
            tempfile.mkdtemp(
                dir=str(commit_root),
                prefix=f".{generation_root.name}.",
            )
        )
        try:
            shutil.copytree(staging_root, temporary, dirs_exist_ok=True)
            fsync_tree(temporary)
            os.replace(temporary, generation_root)
            fsync_directory(commit_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        atomic_write_json(
            commit_root / "published.json",
            {
                "generation_root": generation_root.name,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            fmt=JsonFormat(trailing_newline=True),
            options=DURABLE,
        )
        return generation_root


def make_analysis_agent_service(
    *,
    config: Config,
    usage=None,
    timeout_sec: int | None = None,
    profiling_enabled: bool = True,
) -> AnalysisAgentService:
    """Build the Analysis Agent through the configured provider."""
    runtime = agent_runtime(config)
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
