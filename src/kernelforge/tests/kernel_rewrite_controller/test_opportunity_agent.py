# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernelforge.kernel_rewrite_controller.opportunity_agent as opportunity_agent_module
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRuntimeConfig,
)
from kernelforge.kernel_rewrite_controller import ControllerLayout, read_handoff
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_TIMED_OUT,
    OpportunityAnalysisAgent,
    _StagingProtection,
    run_opportunity_analysis,
)
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "controller-test",
    "GIT_AUTHOR_EMAIL": "controller-test@local",
    "GIT_COMMITTER_NAME": "controller-test",
    "GIT_COMMITTER_EMAIL": "controller-test@local",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _handoff(tmp_path: Path):
    root = tmp_path / "handoff"
    root.mkdir()
    (root / "workload.md").write_text("# Workload\n", encoding="utf-8")
    (root / "serving-context.md").write_text("# Serving Context\n", encoding="utf-8")
    (root / "trace-evidence.md").write_text("# Trace Evidence\n", encoding="utf-8")
    return read_handoff(root)


def _write_staged_task(staging_root: Path, repo: Path) -> str:
    identity = {
        "producer": "forge-loop",
        "kernel_name": "kernel",
        "framework": "standalone",
        "framework_version": "unknown",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity))
    task = staging_root / "draft"
    task.mkdir(parents=True)
    (task / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (task / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "base_commit": "",
                "repo_root": str(repo),
                "kernel_path": "kernel.py",
                "operator_name": "kernel",
                "driver_path": "driver.py",
                "source_files": ["kernel.py"],
                "target_functions": ["kernel"],
                "shape_cases": [],
                "priority": 0,
                "reason": "hot operator",
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    return operator_id


class _Backend:
    name = "fake"
    runtime = AgentRuntimeConfig(provider="fake", model="fake")
    capabilities = AgentCapabilities(writable=True, stop_hooks=True)

    def __init__(self, callback, *, error: Exception | None = None, sleep: float = 0.0):
        self.callback = callback
        self.error = error
        self.sleep = sleep
        self.spec = None

    async def run(self, spec, usage=None):
        self.spec = spec
        self.callback(Path(spec.cwd))
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.error is not None:
            raise self.error
        return AgentRunResult(text="done")


def test_agent_publishes_complete_tasks_and_pins_repo_head(tmp_path: Path) -> None:
    repo, base_commit = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    backend = _Backend(lambda staging: _write_staged_task(staging, repo))
    agent = OpportunityAnalysisAgent(backend=backend, timeout_sec=10, max_turns=20)

    result = asyncio.run(agent.run(handoff=_handoff(tmp_path), layout=layout))

    assert result.status == ANALYSIS_STATUS_COMPLETED
    assert result.published_task_count == 1
    task_dirs = [path for path in layout.tasks_root.iterdir() if not path.name.startswith(".")]
    assert len(task_dirs) == 1
    payload = json.loads((task_dirs[0] / "task.json").read_text(encoding="utf-8"))
    assert payload["base_commit"] == base_commit
    assert payload["driver_path"] == "driver.py"
    assert backend.spec.tool_policy.shell is False
    assert backend.spec.hooks is not None


def test_backend_model_probe_receives_an_existing_agent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    backend = _Backend(lambda _staging: None)
    runtime = AgentRuntimeConfig(provider="fake", model="fake")
    config = SimpleNamespace(
        max_turns=20,
        agent_runtime=lambda: runtime,
    )

    monkeypatch.setattr(
        opportunity_agent_module.Config,
        "from_env",
        lambda **_kwargs: config,
    )
    monkeypatch.setattr(
        opportunity_agent_module,
        "with_writable_sandbox",
        lambda selected: selected,
    )

    def _create_backend(selected_runtime, *, probe_cwd):
        assert selected_runtime is runtime
        assert Path(probe_cwd).is_dir()
        return backend

    monkeypatch.setattr(
        opportunity_agent_module,
        "create_registered_backend",
        _create_backend,
    )

    result = run_opportunity_analysis(
        handoff=_handoff(tmp_path),
        layout=layout,
        controller_deadline_unix=time.time() + 60,
    )

    assert result.status == ANALYSIS_STATUS_COMPLETED
    assert (layout.agent_root / "analysis-result.json").is_file()


def test_agent_failure_still_publishes_a_complete_task(tmp_path: Path) -> None:
    repo, _base_commit = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    backend = _Backend(
        lambda staging: _write_staged_task(staging, repo),
        error=RuntimeError("agent crashed"),
    )
    agent = OpportunityAnalysisAgent(backend=backend, timeout_sec=10, max_turns=20)

    result = asyncio.run(agent.run(handoff=_handoff(tmp_path), layout=layout))

    assert result.status == ANALYSIS_STATUS_FAILED
    assert "agent crashed" in result.reason
    assert result.published_task_count == 1


def test_agent_timeout_keeps_tasks_written_before_cancellation(tmp_path: Path) -> None:
    repo, _base_commit = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    backend = _Backend(
        lambda staging: _write_staged_task(staging, repo),
        sleep=60,
    )
    agent = OpportunityAnalysisAgent(backend=backend, timeout_sec=1, max_turns=20)

    result = asyncio.run(agent.run(handoff=_handoff(tmp_path), layout=layout))

    assert result.status == ANALYSIS_STATUS_TIMED_OUT
    assert result.published_task_count == 1


def test_incomplete_staging_directory_is_not_published(tmp_path: Path) -> None:
    layout = ControllerLayout(tmp_path / "output")

    def _incomplete(staging: Path) -> None:
        task = staging / "draft"
        task.mkdir(parents=True)
        (task / "task.json").write_text("{}", encoding="utf-8")

    agent = OpportunityAnalysisAgent(
        backend=_Backend(_incomplete),
        timeout_sec=10,
        max_turns=20,
    )

    result = asyncio.run(agent.run(handoff=_handoff(tmp_path), layout=layout))

    assert result.published_task_count == 0
    assert not list(layout.tasks_root.glob("*")) if layout.tasks_root.exists() else True


def test_analysis_requires_a_hook_capable_provider() -> None:
    backend = _Backend(lambda _staging: None)
    backend.capabilities = AgentCapabilities(writable=True, stop_hooks=False)

    with pytest.raises(ValueError, match="requires a provider with tool hooks"):
        OpportunityAnalysisAgent(backend=backend, timeout_sec=10, max_turns=20)


def test_write_hook_allows_staging_and_denies_other_paths(tmp_path: Path) -> None:
    protection = _StagingProtection(tmp_path / "staging")
    allowed = asyncio.run(
        protection._on_pre_write(
            {"tool_input": {"file_path": "draft/task.json"}},
            "",
            None,
        )
    )
    denied = asyncio.run(
        protection._on_pre_write(
            {"tool_input": {"file_path": str(tmp_path / "source.py")}},
            "",
            None,
        )
    )

    assert allowed == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
