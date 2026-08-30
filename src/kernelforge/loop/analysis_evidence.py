# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build, restore, and render commit-bound Analysis evidence."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.kernel_backends.constants import resolve_language_dirs
from kernelforge.orchestrator.contracts import EvidenceRef
from kernelforge.orchestrator.supervisor import latest_supervisor_ruling_path
from kernelforge.durable_io import atomic_write_text


log = logging.getLogger(__name__)

ANALYSIS_DIFF_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class AnalysisDiffResult:
    """One materialized cumulative diff or an explicit degradation reason."""

    path: str = ""
    error: str = ""


class AnalysisEvidenceMixin:
    """Own Analysis artifact paths, diffs, resume, and prompt rendering."""

    def _analysis_cumulative_diff(
        self,
        *,
        evidence_commit: str,
        canonical_commit: str,
    ) -> AnalysisDiffResult:
        """Persist the cumulative code delta from active evidence to canonical."""
        if (
            not evidence_commit
            or not canonical_commit
            or evidence_commit == canonical_commit
            or not self._looks_like_git_commit(evidence_commit)
            or not self._looks_like_git_commit(canonical_commit)
        ):
            return AnalysisDiffResult()
        cache_key = (evidence_commit, canonical_commit)
        cached = self._analysis_diff_results.get(cache_key)
        if cached is not None:
            return cached
        root = Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "analysis" / "deltas"
        path = root / f"{evidence_commit}_to_{canonical_commit}.patch"
        if path.is_file():
            result = AnalysisDiffResult(path=str(path.resolve()))
            self._analysis_diff_results[cache_key] = result
            return result
        try:
            completed = git(
                "diff",
                "--no-ext-diff",
                evidence_commit,
                canonical_commit,
                cwd=self.ic.workspace_dir,
                check=False,
                timeout=ANALYSIS_DIFF_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as error:
            message = f"cumulative Analysis diff timed out for {evidence_commit}..{canonical_commit}: {error}"
            result = AnalysisDiffResult(error=message)
            self._analysis_diff_results[cache_key] = result
            self.persistence_degraded = True
            self.persistence_errors.append(message)
            self.persistence_errors = self.persistence_errors[-10:]
            log.warning(message)
            return result
        if completed.returncode != 0:
            message = (
                "could not build cumulative Analysis diff for "
                f"{evidence_commit}..{canonical_commit}: "
                f"{completed.stderr.strip()}"
            )
            result = AnalysisDiffResult(error=message)
            self._analysis_diff_results[cache_key] = result
            self.persistence_degraded = True
            self.persistence_errors.append(message)
            self.persistence_errors = self.persistence_errors[-10:]
            log.warning(message)
            return result
        try:
            atomic_write_text(path, completed.stdout)
        except OSError as error:
            message = f"could not persist cumulative Analysis diff for {evidence_commit}..{canonical_commit}: {error}"
            self.persistence_degraded = True
            self.persistence_errors.append(message)
            self.persistence_errors = self.persistence_errors[-10:]
            result = AnalysisDiffResult(error=message)
            self._analysis_diff_results[cache_key] = result
            log.warning(message)
            return result
        result = AnalysisDiffResult(path=str(path.resolve()))
        self._analysis_diff_results[cache_key] = result
        return result

    def _canonical_commit(self) -> str:
        """The tree state everything planned this round is attributed to.

        Named separately from the context it is built into because a caller can
        need the commit alone: a round of lane plans records the tree it
        describes, and the process that picks those plans back up has to compare
        against it without paying for a whole planning context.
        """
        head_lines = self._git("rev-parse", "HEAD").strip().splitlines()
        return (
            self.run_state.best.commit_hash
            or self.run_state.head_commit
            or (head_lines[0] if head_lines else "")
            or self.ic.campaign_base_commit
            or "uncommitted"
        )

    def _build_orchestration_context(self):
        """Build one immutable planning context from current loop evidence."""
        from kernelforge.orchestrator.contracts import (
            CaseEvidence,
            EvidenceRef,
            OrchestrationContext,
        )

        workspace = Path(self.ic.workspace_dir).resolve()
        canonical_commit = self._canonical_commit()
        analysis_state = self.run_state.analysis
        evidence_commit = analysis_state.evidence_commit
        cumulative_diff = self._analysis_cumulative_diff(
            evidence_commit=evidence_commit,
            canonical_commit=canonical_commit,
        )
        cumulative_diff_path = cumulative_diff.path
        source_map_path = Path(self.ic.kernel_file).resolve()

        scored_case_ids = [
            case_id for case_id in sorted(self._baseline_case_times) if case_id not in self._unscored_cases
        ]
        if not scored_case_ids:
            scored_case_ids = sorted(self._baseline_case_times)
        cases = tuple(
            CaseEvidence(
                case_id=case_id,
                latency_ms=(self._best_case_times.get(case_id) or self._baseline_case_times.get(case_id)),
            )
            for case_id in scored_case_ids
        )

        evidence_refs = []
        if source_map_path.is_file():
            evidence_refs.append(
                EvidenceRef(
                    kind="source_map",
                    path=str(source_map_path),
                    summary="Current source map or anchor source.",
                )
            )
        artifact_candidates = (
            workspace / "forge_experiments" / "candidates" / "index.jsonl",
            workspace / "forge_experiments" / "run_state.json",
        )
        for path in artifact_candidates:
            if path.is_file():
                evidence_refs.append(
                    EvidenceRef(
                        kind=("candidate_archive" if path.name == "index.jsonl" else "run_state"),
                        path=str(path.resolve()),
                        summary=f"Current {path.stem.replace('_', ' ')} evidence.",
                    )
                )
        if cumulative_diff_path:
            evidence_refs.append(
                EvidenceRef(
                    kind="analysis_cumulative_diff",
                    path=cumulative_diff_path,
                    summary=(
                        "Cumulative canonical source diff from the Analysis "
                        f"evidence commit {evidence_commit} to "
                        f"{canonical_commit}."
                    ),
                )
            )
        if getattr(self, "lessons", None) is not None:
            lesson_iterations = self.lessons.existing_iterations()
            if lesson_iterations:
                latest_lesson = self.lessons.path(lesson_iterations[-1])
                evidence_refs.extend(
                    (
                        EvidenceRef(
                            kind="lesson_directory",
                            path=str(self.lessons.root.resolve()),
                            summary=(
                                "Free-form Implementer session records for all "
                                "completed iterations; historical evidence only."
                            ),
                        ),
                        EvidenceRef(
                            kind="latest_lesson",
                            path=str(latest_lesson.resolve()),
                            summary=(
                                f"Latest free-form Implementer session record from iteration {lesson_iterations[-1]}."
                            ),
                        ),
                    )
                )
        if self._supervisor_ruling:
            supervisor_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
            if supervisor_path.is_file():
                evidence_refs.append(
                    EvidenceRef(
                        kind="supervisor_guidance",
                        path=str(supervisor_path.resolve()),
                        summary=(
                            "Latest free-form Supervisor Ruling. It overrides "
                            "subjective conclusions in historical lesson records "
                            "but not objective measurements."
                        ),
                    )
                )
        knowledge_index = ""
        local_knowledge_root = getattr(self.config, "local_knowledge_dir", None)
        if local_knowledge_root:
            try:
                from kernelforge.knowledge import build_forge_knowledge

                stored = (self.ic.kernel_backend or "").strip()
                backend = stored or self.ic.backend or ""
                root = Path(local_knowledge_root)
                language = resolve_language_dirs(backend, root)
                include_aiter = backend == "aiter" or any(
                    "aiter" in Path(source).parts for source in self._target_source_files()
                )
                knowledge_index = build_forge_knowledge(
                    root,
                    language=language,
                    include_aiter=include_aiter,
                )
            except Exception:
                log.debug("failed to build orchestration knowledge index", exc_info=True)

        return OrchestrationContext(
            analysis_commit=canonical_commit,
            workspace=str(workspace),
            gpu_target=self.config.gpu_target,
            objective=(
                "Preserve correctness and complete benchmark case coverage while "
                "maximizing equal-weight mean incumbent-to-candidate case speedup."
            ),
            program_context=(self.ic.program_md or f"Optimize the kernel at {self.ic.kernel_file}."),
            source_map_path=str(source_map_path),
            # The declared source set, verbatim and in campaign order, so the
            # planner is told what it may edit instead of inferring it from the
            # one path in program_context.
            editable_sources=tuple(self._target_source_files()),
            cases=cases,
            knowledge_index=knowledge_index,
            supervisor_guidance=self._supervisor_ruling,
            last_critic_verdict=self._last_critic_verdict,
            last_critic_review=self._last_critic_review,
            search_mode=self.run_state.search_mode,
            search_reason_codes=tuple(self.run_state.search_reason_codes),
            search_objective=self.run_state.search_objective,
            search_mode_residence_remaining=(self.run_state.search_mode_residence_remaining),
            evidence_refs=tuple(evidence_refs),
            canonical_commit=canonical_commit,
            evidence_commit=evidence_commit,
            evidence_stale=bool(evidence_commit and evidence_commit != canonical_commit),
            evidence_status=analysis_state.evidence_status,
            evidence_mean_case_speedup=(analysis_state.evidence_mean_case_speedup),
            current_mean_case_speedup=self.best_mean_case_speedup,
            cumulative_diff_path=cumulative_diff_path,
            cumulative_diff_error=cumulative_diff.error,
        )

    @staticmethod
    def _looks_like_git_commit(value: str) -> bool:
        candidate = str(value or "").strip().lower()
        return bool(re.fullmatch(r"[0-9a-f]{4,40}", candidate))

    def _published_analysis_bundle_root(self, commit: str) -> Path | None:
        """Return the published analysis generation directory for one commit."""
        from kernelforge.orchestrator.analysis import AnalysisAgentService

        if not self._looks_like_git_commit(commit):
            return None
        workspace = Path(self.ic.workspace_dir).resolve()
        commit_root = workspace / "forge_experiments" / "analysis" / commit
        return AnalysisAgentService._published_generation_root(commit_root)

    def _nearest_published_analysis_commit(self, start_commit: str) -> str:
        """Walk Git ancestors until a published analysis bundle is found."""
        commit = str(start_commit or "").strip()
        seen: set[str] = set()
        while commit and commit not in seen and self._looks_like_git_commit(commit):
            seen.add(commit)
            if self._published_analysis_bundle_root(commit) is not None:
                return commit
            result = git(
                "rev-parse",
                f"{commit}^",
                cwd=self.ic.workspace_dir,
                check=False,
            )
            if result.returncode != 0:
                break
            parent = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if not self._looks_like_git_commit(parent):
                break
            commit = parent
        return ""

    def _restore_published_analysis_commit(self) -> None:
        """Rehydrate the last successfully published analysis commit."""
        if getattr(self, "state_store", None) is None:
            return
        recorded = self.run_state.analysis.evidence_commit
        if self._looks_like_git_commit(recorded) and self._published_analysis_bundle_root(recorded) is not None:
            self._last_published_analysis_commit = recorded
            return
        for event in reversed(list(self.state_store.read_events())):
            if event.get("type") != "analysis_result":
                continue
            if event.get("status") != "published":
                continue
            commit = str(event.get("analysis_commit") or "").strip()
            if self._looks_like_git_commit(commit) and self._published_analysis_bundle_root(commit) is not None:
                self._last_published_analysis_commit = commit
                self.run_state.analysis.evidence_commit = commit
                self.run_state.analysis.evidence_status = str(event.get("available_tier") or "published")
                current_best_commit = self.run_state.best.commit_hash or self.run_state.head_commit
                if commit == current_best_commit:
                    self.run_state.analysis.evidence_mean_case_speedup = self.run_state.best.mean_case_speedup or 1.0
                return
        head_lines = self._git("rev-parse", "HEAD").splitlines()
        if head_lines and self._looks_like_git_commit(head_lines[0]):
            self._last_published_analysis_commit = self._nearest_published_analysis_commit(head_lines[0])
            if self._last_published_analysis_commit:
                self.run_state.analysis.evidence_commit = self._last_published_analysis_commit

    def _incremental_analysis_input(
        self,
        *,
        current_commit: str,
        previous_commit: str,
    ):
        """Build post-KEEP Analysis context from the nearest analyzed parent."""
        from kernelforge.orchestrator.analysis import (
            IncrementalAnalysisInput,
        )

        parent_commit = previous_commit
        if (
            not parent_commit
            or parent_commit == current_commit
            or self._published_analysis_bundle_root(parent_commit) is None
        ):
            parent_commit = self._nearest_published_analysis_commit(current_commit)
        parent_bundle_root = self._published_analysis_bundle_root(parent_commit)
        if not parent_commit or parent_bundle_root is None:
            return None

        changed_files = tuple(
            line
            for line in self._git(
                "diff",
                "--name-only",
                parent_commit,
                current_commit,
            ).splitlines()
            if line
        )
        commit_diff = self._git(
            "diff",
            "--no-ext-diff",
            parent_commit,
            current_commit,
        )
        return IncrementalAnalysisInput(
            parent_commit=parent_commit,
            parent_bundle=parent_bundle_root.resolve(),
            commit_diff=commit_diff,
            changed_source_files=changed_files,
        )

    def _apply_last_analysis_evidence(self, context):
        """Attach the last published Analysis paths to the current canonical."""
        state = self.run_state.analysis
        evidence_commit = state.evidence_commit
        stale = bool(evidence_commit and evidence_commit != context.analysis_commit)

        previous = self._active_analysis_context
        previous_evidence_commit = getattr(previous, "evidence_commit", "") if previous is not None else ""
        if previous is not None and previous_evidence_commit == evidence_commit:
            current_cases = {case.case_id: case for case in context.cases}
            cases = []
            for old_case in previous.cases:
                current = current_cases.get(old_case.case_id)
                if current is None:
                    continue
                flags = list(old_case.flags)
                if stale:
                    flags.append("analysis_evidence_stale")
                cases.append(
                    replace(
                        old_case,
                        latency_ms=current.latency_ms,
                        flags=tuple(dict.fromkeys(flags)),
                    )
                )
            refs = {reference.path: reference for reference in context.evidence_refs}
            superseded_kinds = {
                "analysis_cumulative_diff",
                "candidate_archive",
                "latest_lesson",
                "lesson_directory",
                "run_state",
                "supervisor_guidance",
            }
            refs.update(
                {
                    reference.path: reference
                    for reference in previous.evidence_refs
                    if reference.kind not in superseded_kinds
                }
            )
            if stale and previous.source_map_path:
                source_map = Path(previous.source_map_path).resolve()
                if source_map.is_file():
                    refs[str(source_map)] = EvidenceRef(
                        kind="analysis_source_map",
                        path=str(source_map),
                        summary=(f"Source map produced with the stale Analysis evidence at commit {evidence_commit}."),
                    )
            return replace(
                context,
                source_map_path=(context.source_map_path if stale else previous.source_map_path),
                cases=tuple(cases) if cases else context.cases,
                evidence_refs=tuple(refs.values()),
                evidence_commit=evidence_commit,
                evidence_stale=stale,
                evidence_status=state.evidence_status,
                evidence_mean_case_speedup=(state.evidence_mean_case_speedup),
            )

        root = self._published_analysis_bundle_root(evidence_commit)
        if root is None:
            return replace(
                context,
                evidence_commit=evidence_commit,
                evidence_stale=stale,
                evidence_status=state.evidence_status,
                evidence_mean_case_speedup=(state.evidence_mean_case_speedup),
            )

        refs = {reference.path: reference for reference in context.evidence_refs}
        root = root.resolve()
        refs[str(root)] = EvidenceRef(
            kind="analysis_bundle",
            path=str(root),
            summary=(
                "Last published Analysis bundle. "
                f"Measured commit: {evidence_commit}. "
                f"Current canonical: {context.analysis_commit}."
            ),
        )
        for name, kind, summary in (
            (
                "artifact_catalog.json",
                "analysis_artifact_catalog",
                "Artifact map for the last published Analysis bundle.",
            ),
            (
                "report.md",
                "analysis_summary",
                "Cross-case report from the last published Analysis bundle.",
            ),
            (
                "workflow.json",
                "analysis_workflow",
                "Workflow state for the last published Analysis bundle.",
            ),
        ):
            path = (root / name).resolve()
            if path.is_file():
                refs[str(path)] = EvidenceRef(
                    kind=kind,
                    path=str(path),
                    summary=summary,
                )
        catalog_path = root / "artifact_catalog.json"
        if catalog_path.is_file():
            try:
                catalog = json.loads(catalog_path.read_text())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                catalog = {}
            for artifact in catalog.get("artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                path = Path(str(artifact.get("path") or "")).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                if not path.exists():
                    continue
                refs[str(path)] = EvidenceRef(
                    kind=str(artifact.get("kind") or "analysis_artifact"),
                    path=str(path),
                    summary=(
                        f"{artifact.get('description') or 'Analysis artifact'} from evidence commit {evidence_commit}."
                    ),
                )

        source_map = (root / "source_map.md").resolve()
        if source_map.is_file():
            refs[str(source_map)] = EvidenceRef(
                kind="analysis_source_map",
                path=str(source_map),
                summary=(f"Source map produced with the Analysis evidence at commit {evidence_commit}."),
            )
        stale_cases = tuple(
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
            for case in context.cases
        )
        return replace(
            context,
            source_map_path=(str(source_map) if source_map.is_file() and not stale else context.source_map_path),
            cases=stale_cases,
            evidence_refs=tuple(refs.values()),
            evidence_commit=evidence_commit,
            evidence_stale=stale,
            evidence_status=state.evidence_status,
            evidence_mean_case_speedup=(state.evidence_mean_case_speedup),
        )

    def _build_supervisor_evidence_context(self, iteration: int) -> str:
        """Serialize current profiling and search evidence for stall review."""
        context = self._build_orchestration_context()
        if self._analysis_bundle is not None and self._analysis_bundle.analysis_commit == context.analysis_commit:
            context = self._analysis_bundle.apply(context)
        elif (
            self._active_analysis_context is not None
            and self._active_analysis_context.analysis_commit == context.analysis_commit
        ):
            context = self._active_analysis_context
        orchestration_root = Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "orchestration"
        latest_lesson_path = ""
        if getattr(self, "lessons", None) is not None:
            lesson_iterations = self.lessons.existing_iterations()
            if lesson_iterations:
                latest_lesson_path = str(self.lessons.path(lesson_iterations[-1]).resolve())
        payload = {
            "iteration": iteration,
            "persistence_budget": max(
                1,
                self.ic.supervise_cooldown if self.ic.supervise_cooldown > 0 else self.ic.supervise_after,
            ),
            "latest_optimization_plan": (self._latest_optimization_plan_path or None),
            "orchestration_context": context.to_prompt_dict(),
            "artifact_paths": {
                "orchestration_root": str(orchestration_root),
                "candidate_archive": str(Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "candidates"),
                "lessons": str(Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "lessons"),
                "latest_lesson": latest_lesson_path,
                "analysis_bundle": next(
                    (reference.path for reference in context.evidence_refs if reference.kind == "analysis_bundle"),
                    "",
                ),
                "analysis_artifact_catalog": next(
                    (
                        reference.path
                        for reference in context.evidence_refs
                        if reference.kind == "analysis_artifact_catalog"
                    ),
                    "",
                ),
                "analysis_cumulative_diff": (context.cumulative_diff_path),
                "analysis_cumulative_diff_error": (context.cumulative_diff_error),
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _render_analysis_evidence_for_implementer(self) -> str:
        """Render a bounded map to complete or partial Analysis artifacts."""
        context = self._active_analysis_context
        if context is None:
            return ""
        catalog = next(
            (reference for reference in context.evidence_refs if reference.kind == "analysis_artifact_catalog"),
            None,
        )
        analysis_refs = [
            reference
            for reference in context.evidence_refs
            if reference.kind
            in {
                "analysis_bundle",
                "analysis_cumulative_diff",
                "analysis_summary",
                "analysis_source_map",
                "analysis_workflow",
            }
        ]
        if (
            catalog is None
            and not analysis_refs
            and not any(case.profile_summary_path or case.bottleneck for case in context.cases)
        ):
            return ""
        lines = [
            "## Analysis Evidence (authoritative paths; read on demand)",
            (f"Canonical commit: {context.canonical_commit or context.analysis_commit}"),
            (f"Evidence commit: {context.evidence_commit or context.analysis_commit}"),
            f"Evidence status: {context.evidence_status or 'current'}",
            f"Evidence stale: {'yes' if context.evidence_stale else 'no'}",
            f"Source map: {context.source_map_path}",
        ]
        if context.evidence_stale:
            if context.cumulative_diff_error:
                lines.append(
                    "The profiling evidence predates the current canonical and "
                    "the cumulative diff is unavailable. Treat inherited "
                    "measurements as historical evidence only, inspect the "
                    "current source directly, and never present them as current."
                )
            else:
                lines.append(
                    "The profiling evidence predates the current canonical. Use "
                    "the current timings and cumulative diff to interpret it; "
                    "never present inherited measurements as current."
                )
        if context.cumulative_diff_path:
            lines.append(f"Cumulative diff since evidence: {context.cumulative_diff_path}")
        elif context.cumulative_diff_error:
            lines.append(f"Cumulative diff unavailable: {context.cumulative_diff_error}")
        if catalog is not None:
            lines.extend(
                (
                    f"Artifact catalog: {catalog.path}",
                    (
                        "The catalog states what every file contains, which "
                        "information it exposes, and whether the artifact is "
                        "COMPLETE, AVAILABLE, SKIPPED, or FAILED."
                    ),
                )
            )
        lines.extend(f"{reference.kind}: {reference.path}" for reference in analysis_refs)
        if any("analysis_static_only" in case.flags for case in context.cases):
            lines.append(
                "STATIC_ONLY: no hardware profiling evidence is available. "
                "Treat bottleneck and potential claims as inference, not "
                "measured profiler facts."
            )
        lines.append("Per-case evidence:")
        for case in context.cases:
            parts = [f"- {case.case_id}"]
            if case.latency_ms is not None:
                parts.append(f"latency={case.latency_ms:.6f} ms")
            if case.bottleneck:
                parts.append(f"bottleneck={case.bottleneck}")
            if case.profile_summary_path:
                parts.append(f"evidence={case.profile_summary_path}")
            if case.flags:
                parts.append("flags=" + ",".join(case.flags))
            lines.append(" | ".join(parts))
        return "\n".join(lines)
