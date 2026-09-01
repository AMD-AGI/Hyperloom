"""Tests for the commit-bound Analysis Agent bundle."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends import AgentRunResult
from kernelforge.orchestrator.analysis import (
    AnalysisAgentService,
    AnalysisBundleError,
    AnalysisConfigurationError,
    IncrementalAnalysisInput,
    _AnalysisProtection,
)
from kernelforge.orchestrator.analysis_session import (
    AnalysisAttemptLimitError,
    AnalysisSessionJournal,
)
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    OrchestrationContext,
)


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    harness = workspace / "test_harness.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver.write_text("print('driver')\n")
    harness.write_text("def test_kernel():\n    pass\n")
    subprocess.run(
        ["git", "init", "-q", "-b", "analysis-test"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "analysis@test"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Analysis Test"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"],
        cwd=workspace,
        check=True,
    )
    return workspace, kernel, driver


def _context(workspace: Path) -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize test kernel.",
        source_map_path=str(workspace / "kernel.py"),
        editable_sources=(
            str(workspace / "kernel.py"),
            str(workspace / "configs" / "tuned_shapes.csv"),
        ),
        cases=(
            CaseEvidence(case_id="case-a", latency_ms=1.0),
            CaseEvidence(case_id="case-b", latency_ms=2.0),
        ),
        knowledge_index="Knowledge index",
    )


class _BundleBackend:
    def __init__(self, *, modify_kernel: bool = False) -> None:
        self.modify_kernel = modify_kernel
        self.calls = 0
        self.specs = []
        self.initial_progress = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        self.specs.append(spec)
        payload = json.loads(spec.user_prompt)
        root = Path(payload["analysis_staging_dir"])
        request = json.loads(Path(payload["request_file"]).read_text())
        self.initial_progress.append(
            json.loads((root / "progress.json").read_text()) if (root / "progress.json").is_file() else None
        )
        if self.modify_kernel:
            Path(payload["kernel_file"]).write_text("def kernel():\n    return 2\n")

        case_ids = [item["case_id"] for item in request["cases"]]
        if not (root / "source_map.md").is_file():
            (root / "source_map.md").write_text("# Source map\n")
        (root / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "COMPLETE",
                    "completed_case_ids": case_ids,
                }
            )
        )
        (root / "commands.jsonl").write_text("")
        (root / "report.md").write_text("# Analysis Report\n")
        (root / "case_inventory.json").write_text(json.dumps({"case_ids": case_ids}))
        for item in request["cases"]:
            case_root = root / "cases" / item["directory"]
            profile_root = case_root / "profile"
            profile_root.mkdir(parents=True, exist_ok=True)
            (case_root / "case.json").write_text(json.dumps(item))
            (case_root / "normalized_metrics.json").write_text(json.dumps({"metrics": {"occupancy": 0.75}}))
            with (root / "commands.jsonl").open("a") as commands:
                commands.write(
                    json.dumps(
                        {
                            "case_id": item["case_id"],
                            "command": "rocprofv3 --kernel-trace",
                            "exit_code": 0,
                            "success": True,
                        }
                    )
                    + "\n"
                )
            (case_root / "bottleneck.json").write_text(
                json.dumps(
                    {
                        "classification": ("MEMORY" if item["case_id"] == "case-a" else "COMPUTE"),
                        "flags": [],
                    }
                )
            )
            (case_root / "analysis.md").write_text("# Analysis\n")
            (case_root / "directions.md").write_text("# Directions\n")
            (profile_root / "raw.txt").write_text("raw profile\n")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis_commit": payload["analysis_commit"],
                    "driver_digest": request["driver_digest"],
                    "source_digest": request["source_digest"],
                    "status": "READY",
                    "expected_case_ids": case_ids,
                    "completed_case_ids": case_ids,
                    "failed_case_ids": [],
                }
            )
        )
        return AgentRunResult(text="analysis complete")


class _ResumableAnalysisBackend(_BundleBackend):
    capabilities = SimpleNamespace(resumable=True)

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.resume_calls = 0
        self.resume_session_ids = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.start_calls += 1
        self.specs.append(spec)
        return AgentRunResult(
            text="[session interrupted]",
            subtype="error",
            end_reason="sdk_error",
            session_id="analysis-session-1",
            stderr_tail="stream interrupted",
        )

    async def resume(
        self,
        spec,
        session_id,
        prompt,
        usage=None,
    ) -> AgentRunResult:
        self.resume_calls += 1
        self.resume_session_ids.append(session_id)
        assert "SAME session" in prompt
        return await super().run(spec, usage=usage)


class _FailAfterSourceBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        payload = json.loads(spec.user_prompt)
        root = Path(payload["analysis_staging_dir"])
        (root / "source_map.md").write_text("# Durable source map\n")
        (root / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "SOURCE_DISCOVERY_COMPLETE",
                    "completed_case_ids": [],
                }
            )
        )
        raise RuntimeError("simulated later step failure")


class _SourceThenFailBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        payload = json.loads(spec.user_prompt)
        root = Path(payload["analysis_staging_dir"])
        (root / "source_map.md").write_text("# Validated partial source map\n")
        raise RuntimeError("simulated failure after durable source step")


class _StaticAnalysisBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.step_kinds = []
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        self.specs.append(spec)
        payload = json.loads(spec.user_prompt)
        root = Path(payload["analysis_staging_dir"])
        request = json.loads(Path(payload["request_file"]).read_text())
        step = payload["analysis_session"]["session_id"]
        self.step_kinds.append(step)
        cases = request["cases"]
        case_ids = [item["case_id"] for item in cases]

        if step != "analysis_session":
            raise AssertionError(f"unexpected static analysis step: {step}")
        (root / "source_map.md").write_text("# Static source map\n")
        (root / "case_inventory.json").write_text(
            json.dumps(
                {
                    "cases": cases,
                    "skipped_case_ids": case_ids,
                    "skip_reason": "analysis profiling disabled",
                }
            )
        )
        (root / "progress.json").write_text(json.dumps({"phase": "COMPLETE", "completed_case_ids": []}))
        (root / "commands.jsonl").write_text(json.dumps({"decision": "profiling_disabled"}) + "\n")
        for item in cases:
            case_root = root / "cases" / item["directory"]
            case_root.mkdir(parents=True, exist_ok=True)
            (case_root / "case.json").write_text(json.dumps(item))
        (root / "report.md").write_text("# Static Analysis Report\n")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis_commit": payload["analysis_commit"],
                    "driver_digest": request["driver_digest"],
                    "source_digest": request["source_digest"],
                    "status": "PARTIAL",
                    "expected_case_ids": case_ids,
                    "completed_case_ids": [],
                    "failed_case_ids": [],
                    "skipped_case_ids": case_ids,
                }
            )
        )
        return AgentRunResult(text=f"{step} complete")


class _MarkdownOnlyBackend:
    def __init__(self, *, complete_cases: int | None = None) -> None:
        self.calls = 0
        self.complete_cases = complete_cases
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        self.specs.append(spec)
        payload = json.loads(spec.user_prompt)
        root = Path(payload["analysis_staging_dir"])
        cases = payload["cases"]
        limit = len(cases) if self.complete_cases is None else self.complete_cases
        (root / "report.md").write_text("# Analysis Report\n\nMarkdown-first findings.\n")
        (root / "source_map.md").write_text("# Source Map\n")
        for item in cases[:limit]:
            case_root = root / "cases" / item["directory"]
            (case_root / "profile").mkdir(parents=True, exist_ok=True)
            (case_root / "profile" / "raw.txt").write_text("raw\n")
            (case_root / "analysis.md").write_text(f"# {item['case_id']}\n\nMeasured analysis.\n")
        return AgentRunResult(text="markdown analysis complete")


def _service(tmp_path, backend, *, profiling_enabled=True):
    knowledge = tmp_path / "knowledge"
    profiling = knowledge / "common_methodology" / "profiling"
    profiling.mkdir(
        parents=True,
        exist_ok=True,
    )
    (profiling / "rocpc_profile.py").write_text("#!/usr/bin/env python3\nprint('reference')\n")
    for name in (
        "measure_rocpc_workflow.md",
        "measure_triage.md",
        "measure_roofline.md",
        "measure_protocol.md",
    ):
        (profiling / name).write_text(f"# {name}\n")
    return AnalysisAgentService(
        backend=backend,
        config=SimpleNamespace(local_knowledge_dir=knowledge),
        timeout_sec=10,
        max_turns=10,
        profiling_enabled=profiling_enabled,
    )


async def test_analysis_protection_only_allows_artifact_writes(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    staging = workspace / "forge_experiments" / "analysis" / "work" / "abc"
    staging.mkdir(parents=True)
    protection = _AnalysisProtection(
        workspace=workspace,
        staging_root=staging,
        protected_paths=(kernel, driver),
        deadline_monotonic=time.monotonic() + 60,
    )

    allowed = await protection._on_pre_write(
        {
            "tool_name": "Write",
            "tool_input": {"file_path": str(staging / "summary.json")},
        },
        None,
        None,
    )
    denied = await protection._on_pre_write(
        {
            "tool_name": "Write",
            "tool_input": {"file_path": str(kernel)},
        },
        None,
        None,
    )

    assert allowed == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    root_search = await protection._on_pre_bash(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "ls /opt/rocpc_profile.py; find / -name rocpc_profile.py"},
        },
        None,
        None,
    )
    assert root_search["hookSpecificOutput"]["permissionDecision"] == "deny"
    unbounded = await protection._on_pre_bash(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "python driver.py"},
        },
        None,
        None,
    )
    assert unbounded["hookSpecificOutput"]["permissionDecision"] == "deny"
    bounded = await protection._on_pre_bash(
        {
            "tool_name": "Bash",
            "tool_input": {"command": ("timeout --signal=TERM --kill-after=5s 30s python driver.py")},
        },
        None,
        None,
    )
    assert bounded == {}
    over_budget = await protection._on_pre_bash(
        {
            "tool_name": "Bash",
            "tool_input": {"command": ("timeout --signal=TERM --kill-after=5s 120s python driver.py")},
        },
        None,
        None,
    )
    assert over_budget["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_analysis_attempt_limit_persists_across_resume(tmp_path) -> None:
    root = tmp_path / "analysis-work"
    journal = AnalysisSessionJournal(
        root,
        analysis_commit="abc123",
        driver_digest="driver",
        source_digest="source",
    )
    journal.begin()
    journal.fail("first failure")
    journal.begin()
    journal.fail("second failure")

    resumed = AnalysisSessionJournal(
        root,
        analysis_commit="abc123",
        driver_digest="driver",
        source_digest="source",
    )

    assert resumed.attempts == 2
    with pytest.raises(
        AnalysisAttemptLimitError,
        match="2/2",
    ):
        resumed.begin()


async def test_analysis_agent_publishes_multi_case_bundle(tmp_path) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _BundleBackend()
    service = _service(tmp_path, backend)
    context = _context(workspace)

    bundle = await service.ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert backend.calls == 1
    assert bundle.root.parent.name == "abc123"
    assert bundle.root.name == "generation-001"
    assert [case.case_id for case in bundle.cases] == ["case-a", "case-b"]
    assert [case.bottleneck for case in bundle.cases] == ["MEMORY", "COMPUTE"]
    assert all("analysis_profiled" in case.flags for case in bundle.cases)
    assert all(".staging" not in case.profile_summary_path for case in bundle.cases)
    assert backend.specs[0].cwd.endswith("analysis/work/abc123")
    assert backend.specs[0].tool_policy.shell is True
    assert backend.specs[0].writable is True
    assert backend.specs[0].reasoning_effort == "high"
    assert backend.specs[0].hooks.stop == []
    assert "Profiler safety contract" in backend.specs[0].system_prompt
    prompt_payload = json.loads(backend.specs[0].user_prompt)
    assert prompt_payload["analysis_session"]["session_id"] == "analysis_session"
    reference_script = Path(prompt_payload["reference_profiling_script"])
    assert reference_script.name == "rocpc_profile.py"
    assert (bundle.root / "tools" / "rocpc_profile.py").is_file()
    assert str(reference_script) in backend.specs[0].system_prompt
    methodology = [Path(path) for path in prompt_payload["profiling_methodology"]]
    assert len(methodology) == 4
    assert all(path.is_absolute() and path.is_file() for path in methodology)
    staging_root = Path(prompt_payload["analysis_staging_dir"])
    assert all(path.is_relative_to(staging_root) for path in methodology)
    prompt = backend.specs[0].system_prompt.lower()
    assert "single analysis agent" in prompt
    assert "entire source, case, profiling" in prompt
    assert "markdown-first output contract" in prompt
    assert "current_step" not in json.loads(backend.specs[0].user_prompt)
    catalog = json.loads((bundle.root / "artifact_catalog.json").read_text())
    assert all(Path(artifact["path"]).is_relative_to(bundle.root) for artifact in catalog["artifacts"])
    assert {artifact["status"] for artifact in catalog["artifacts"]} == {"COMPLETE"}
    assert not {
        "profiling_plan",
        "case_status",
        "case_benchmark",
        "case_potential",
    } & {artifact["kind"] for artifact in catalog["artifacts"]}
    assert not {
        "profiling_plan.json",
        "status.json",
        "benchmark.json",
        "potential.json",
        "directions.json",
    } & {Path(artifact["path"]).name for artifact in catalog["artifacts"]}
    applied = bundle.apply(context)
    # The bundle rebuilds the context field by field; the editable set is a
    # property of the campaign, not of the analysis, so it must survive intact.
    assert applied.editable_sources == context.editable_sources
    evidence_paths = {reference.path for reference in applied.evidence_refs}
    directions_path = next(
        artifact["path"]
        for artifact in catalog["artifacts"]
        if artifact["kind"] == "case_directions" and artifact["case_id"] == "case-a"
    )
    assert directions_path in evidence_paths
    marked = bundle.apply(replace(context, evidence_status="profiled"))
    assert marked.evidence_status == "profiled"
    resumed_context = replace(
        context,
        analysis_commit="def456",
        canonical_commit="def456",
        evidence_commit="abc123",
        evidence_stale=True,
        evidence_status="profiled",
        cases=tuple(replace(case, latency_ms=(case.latency_ms or 0) + 0.5) for case in context.cases),
    )
    restored = service.apply_published_evidence(
        resumed_context,
        evidence_commit="abc123",
    )
    assert [case.bottleneck for case in restored.cases] == [
        "MEMORY",
        "COMPUTE",
    ]
    assert [case.latency_ms for case in restored.cases] == [1.5, 2.5]
    assert all(case.profile_summary_path for case in restored.cases)
    assert all("analysis_evidence_stale" in case.flags for case in restored.cases)
    assert restored.evidence_commit == "abc123"
    assert restored.evidence_stale is True

    cached = await service.ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    assert cached.root == bundle.root
    assert backend.calls == 1


async def test_published_evidence_cross_checks_request_and_manifest_digests(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    service = _service(tmp_path, _BundleBackend())
    context = _context(workspace)
    bundle = await service.ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    manifest_path = bundle.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_digest"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))
    current = replace(
        context,
        analysis_commit="def456",
        canonical_commit="def456",
        evidence_commit="abc123",
        evidence_stale=True,
    )

    restored = service.apply_published_evidence(
        current,
        evidence_commit="abc123",
    )

    assert restored == current
    assert all(not case.profile_summary_path for case in restored.cases)


async def test_analysis_reports_missing_optional_profiling_methodology(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _BundleBackend()
    service = _service(tmp_path, backend)
    missing = tmp_path / "knowledge" / "common_methodology" / "profiling" / "measure_roofline.md"
    missing.unlink()

    await service.ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert backend.calls == 1
    payload = json.loads(backend.specs[0].user_prompt)
    assert payload["profiling_methodology_missing"] == ["measure_roofline.md"]
    assert all(Path(path).is_file() for path in payload["profiling_methodology"])


async def test_analysis_requires_packaged_profiling_script(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _BundleBackend()
    service = _service(tmp_path, backend)
    script = tmp_path / "knowledge" / "common_methodology" / "profiling" / "rocpc_profile.py"
    script.unlink()

    with pytest.raises(
        AnalysisConfigurationError,
        match="packaged Analysis profiling script is missing",
    ):
        await service.ensure_bundle(
            _context(workspace),
            kernel_file=str(kernel),
            driver_script=str(driver),
            source_files=[str(kernel)],
        )

    assert backend.calls == 0


async def test_post_keep_analysis_is_incremental_and_deadline_bounded(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _BundleBackend()
    service = _service(tmp_path, backend)
    parent_context = _context(workspace)
    parent = await service.ensure_bundle(
        parent_context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    kernel.write_text("def kernel():\n    return 2\n")
    child_context = replace(
        parent_context,
        analysis_commit="def456",
    )

    child = await service.ensure_bundle(
        child_context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
        deadline_unix=time.time() + 5,
        incremental=IncrementalAnalysisInput(
            parent_commit=parent_context.analysis_commit,
            parent_bundle=parent.root,
            commit_diff="diff --git a/kernel.py b/kernel.py\n",
            changed_source_files=("kernel.py",),
        ),
    )

    spec = backend.specs[1]
    payload = json.loads(spec.user_prompt)
    assert 0 < spec.timeout_sec <= 5
    assert payload["analysis_trigger"] == "post_keep_incremental"
    assert payload["previous_analysis_commit"] == "abc123"
    assert Path(payload["incremental_diff_path"]).name == ("incremental_diff.patch")
    assert (child.root / "incremental_diff.patch").is_file()
    assert "cumulative re-analysis after one or more solutions" in (spec.system_prompt)
    assert "diff may span multiple accepted KEEP commits" in (spec.system_prompt)
    assert all("analysis_profile_incremental" in case.flags for case in child.cases)


async def test_invalid_incremental_parent_falls_back_to_full_analysis(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _BundleBackend()
    service = _service(tmp_path, backend)
    context = replace(_context(workspace), analysis_commit="def456")

    bundle = await service.ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
        incremental=IncrementalAnalysisInput(
            parent_commit="abc123",
            parent_bundle=workspace / "missing-parent",
            commit_diff="diff --git a/kernel.py b/kernel.py\n",
            changed_source_files=("kernel.py",),
        ),
    )

    payload = json.loads(backend.specs[0].user_prompt)
    assert payload["analysis_trigger"] == "canonical_baseline"
    assert payload["previous_analysis_commit"] == ""
    assert payload["previous_analysis_bundle"] == ""
    assert payload["incremental_diff_path"] == ""
    assert not (bundle.root / "incremental_diff.patch").exists()
    assert bundle.outcome is not None
    assert bundle.outcome.parent_reuse_commit == ""


async def test_unvalidated_profile_files_publish_partial(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _MarkdownOnlyBackend()

    bundle = await _service(tmp_path, backend).ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert bundle.manifest["status"] == "PARTIAL"
    assert bundle.manifest["completed_case_ids"] == []
    assert bundle.manifest["skipped_case_ids"] == ["case-a", "case-b"]
    assert (bundle.root / "report.md").is_file()
    assert not (bundle.root / "summary.json").exists()
    assert all("analysis_profiled" not in case.flags for case in bundle.cases)


async def test_incomplete_markdown_analysis_publishes_partial(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _MarkdownOnlyBackend(complete_cases=1)
    service = _service(tmp_path, backend)

    bundle = await service.ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert bundle.manifest["status"] == "PARTIAL"
    assert bundle.manifest["completed_case_ids"] == []
    assert bundle.manifest["skipped_case_ids"] == ["case-a", "case-b"]
    retried = await service.ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    cached = await service.ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    assert retried.manifest["status"] == "PARTIAL"
    assert cached.root == retried.root
    assert cached.manifest["upgrade_exhausted"] is True
    assert cached.outcome is not None
    assert cached.outcome.upgrade_exhausted is True
    assert backend.calls == 2
    assert "analysis_profile_skipped" in bundle.cases[1].flags


async def test_analysis_agent_resumes_same_session_after_api_failure(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FORGE_AGENT_API_BASE_DELAY_SEC", "0")
    monkeypatch.setenv("FORGE_AGENT_API_MAX_DELAY_SEC", "0")
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _ResumableAnalysisBackend()

    bundle = await _service(tmp_path, backend).ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert bundle.manifest["status"] == "READY"
    assert backend.start_calls == 1
    assert backend.resume_calls == 1
    assert backend.resume_session_ids == ["analysis-session-1"]
    assert backend.calls == 1


async def test_static_analysis_uses_one_session_and_marks_cases_skipped(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _StaticAnalysisBackend()
    service = _service(tmp_path, backend, profiling_enabled=False)

    bundle = await service.ensure_bundle(
        _context(workspace),
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert bundle.manifest["status"] == "PARTIAL"
    assert bundle.manifest["skipped_case_ids"] == ["case-a", "case-b"]
    assert all("analysis_static_only" in case.flags for case in bundle.cases)
    assert backend.calls == 1
    assert backend.step_kinds == ["analysis_session"]
    assert "collect hardware counters" in (backend.specs[0].system_prompt.lower())
    workflow = json.loads((bundle.root / "workflow.json").read_text())
    assert workflow["session"]["status"] == "COMPLETE"
    catalog = json.loads((bundle.root / "artifact_catalog.json").read_text())
    assert all(artifact["kind"] != "profiling_plan" for artifact in catalog["artifacts"])
    assert catalog["analysis_session_status"] == "COMPLETE"
    assert {artifact["status"] for artifact in catalog["artifacts"] if artifact["scope"] == "case"} == {"SKIPPED"}


async def test_profiled_service_upgrades_cached_static_bundle(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    context = _context(workspace)
    static_backend = _StaticAnalysisBackend()
    static_bundle = await _service(
        tmp_path,
        static_backend,
        profiling_enabled=False,
    ).ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    assert static_bundle.manifest["status"] == "PARTIAL"

    profiled_backend = _BundleBackend()
    profiled_bundle = await _service(
        tmp_path,
        profiled_backend,
        profiling_enabled=True,
    ).ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert profiled_backend.calls == 1
    assert profiled_bundle.manifest["status"] == "READY"
    assert all("analysis_profiled" in case.flags for case in profiled_bundle.cases)


async def test_analysis_agent_restores_modified_kernel_and_rejects_bundle(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    original = kernel.read_bytes()
    service = _service(tmp_path, _BundleBackend(modify_kernel=True))

    with pytest.raises(AnalysisBundleError, match="modified immutable inputs"):
        await service.ensure_bundle(
            _context(workspace),
            kernel_file=str(kernel),
            driver_script=str(driver),
            source_files=[str(kernel)],
        )

    assert kernel.read_bytes() == original
    assert not (workspace / "forge_experiments" / "analysis" / "abc123").exists()


async def test_analysis_session_preserves_and_resumes_completed_outputs(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    failing = _FailAfterSourceBackend()
    context = _context(workspace)

    with pytest.raises(AnalysisBundleError, match="checkpoint="):
        await _service(tmp_path, failing).ensure_bundle(
            context,
            kernel_file=str(kernel),
            driver_script=str(driver),
            source_files=[str(kernel)],
        )

    work_root = workspace / "forge_experiments" / "analysis" / "work" / "abc123"
    assert (work_root / "source_map.md").read_text() == ("# Durable source map\n")
    failed_workflow = json.loads((work_root / "workflow.json").read_text())
    assert failed_workflow["session"]["status"] == "FAILED"

    recovered_backend = _BundleBackend()
    bundle = await _service(tmp_path, recovered_backend).ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    assert bundle.root.parent.name == "abc123"
    assert bundle.root.name.startswith("generation-")
    assert recovered_backend.calls == 1
    assert recovered_backend.initial_progress == [
        {
            "phase": "SOURCE_DISCOVERY_COMPLETE",
            "completed_case_ids": [],
        }
    ]
    assert (bundle.root / "source_map.md").read_text() == ("# Durable source map\n")
    workflow = json.loads((bundle.root / "workflow.json").read_text())
    assert workflow["status"] == "READY"
    assert workflow["session"]["status"] == "COMPLETE"
    assert (bundle.root / "cases").is_dir()
    assert all((case_root / "analysis.md").is_file() for case_root in (bundle.root / "cases").iterdir())


async def test_partial_checkpoint_is_exposed_as_orchestration_evidence(
    tmp_path,
) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    backend = _SourceThenFailBackend()
    service = _service(tmp_path, backend)
    context = _context(workspace)

    with pytest.raises(AnalysisBundleError):
        await service.ensure_bundle(
            context,
            kernel_file=str(kernel),
            driver_script=str(driver),
            source_files=[str(kernel)],
        )

    checkpoint = service.apply_checkpoint(context)
    evidence = {item.kind: item for item in checkpoint.evidence_refs}

    assert checkpoint.source_map_path.endswith("source_map.md")
    assert Path(checkpoint.source_map_path).read_text() == ("# Validated partial source map\n")
    # A partial checkpoint narrows the evidence, never the edit surface.
    assert checkpoint.editable_sources == context.editable_sources
    assert "analysis_artifact_catalog" in evidence
    catalog = json.loads(Path(evidence["analysis_artifact_catalog"].path).read_text())
    source_entry = next(item for item in catalog["artifacts"] if item["kind"] == "source_map")
    assert source_entry["status"] == "AVAILABLE"
    assert "call graph" in source_entry["available_information"]
