# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A scripted Analysis backend and the context and service around it, shared by the Analysis and recovery tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.roles.analysis import AnalysisAgentService
from kernelforge.workflows.forge_loop.roles.contracts import CaseEvidence, OrchestrationContext


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
