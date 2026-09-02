# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRuntimeConfig,
)
from kernelforge.kernel_rewrite_controller import controller, dispatcher
from kernelforge.kernel_rewrite_controller.forge_runner import ForgeLoopOutcome
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
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


def _source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo


def _handoff(tmp_path: Path) -> Path:
    root = tmp_path / "handoff"
    root.mkdir()
    (root / "workload.md").write_text("# Workload\n", encoding="utf-8")
    (root / "serving-context.md").write_text("# Serving Context\n", encoding="utf-8")
    (root / "trace-evidence.md").write_text("# Trace Evidence\n", encoding="utf-8")
    return root


class _TaskAgentBackend:
    name = "fake"
    runtime = AgentRuntimeConfig(provider="fake", model="fake")
    capabilities = AgentCapabilities(writable=True, stop_hooks=True)

    def __init__(self, repo: Path, *, fail_after_write: bool = False, malformed: bool = False):
        self.repo = repo
        self.fail_after_write = fail_after_write
        self.malformed = malformed

    async def run(self, spec, usage=None):
        draft = Path(spec.cwd) / "draft"
        draft.mkdir()
        (draft / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
        if self.malformed:
            (draft / "task.json").write_text("{}", encoding="utf-8")
        else:
            identity = {
                "producer": "forge-loop",
                "kernel_name": "kernel",
                "framework": "standalone",
                "framework_version": "unknown",
                "backend": "triton",
                "gpu": "mi355x",
            }
            (draft / "task.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "identity": identity,
                        "base_commit": "",
                        "repo_root": str(self.repo),
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
        if self.fail_after_write:
            raise RuntimeError("analysis failed after task publication")
        return AgentRunResult(text="done")


def _successful_forge(invocation, *, on_checkpoint=None):
    kernel = Path(invocation.command[invocation.command.index("--kernel") + 1])
    kernel.write_text("VALUE = 2\n", encoding="utf-8")
    _git(invocation.workspace, "add", ".")
    _git(invocation.workspace, "commit", "-m", "optimize kernel")
    best_commit = _git(invocation.workspace, "rev-parse", "HEAD")
    payload = {
        "improved": True,
        "best_commit": best_commit,
        "mean_case_speedup": 1.2,
    }
    invocation.result_json.write_text(json.dumps(payload), encoding="utf-8")
    if on_checkpoint is not None:
        on_checkpoint()
    return ForgeLoopOutcome(
        returncode=0,
        stdout="",
        stderr="",
        result=payload,
        timed_out=False,
        command=invocation.command,
    )


def _wire_fake_analysis(monkeypatch, backend) -> None:
    monkeypatch.setattr(
        controller,
        "run_opportunity_analysis",
        lambda **kwargs: run_opportunity_analysis(**kwargs, backend=backend),
    )


def test_controller_full_path_publishes_a_shared_base_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _source_repo(tmp_path)
    backend = _TaskAgentBackend(repo)
    _wire_fake_analysis(monkeypatch, backend)
    monkeypatch.setattr(dispatcher, "run_forge_loop", _successful_forge)

    state = controller.run_controller(
        handoff_dir=_handoff(tmp_path),
        budget_minutes=120,
        output_dir=tmp_path / "output",
    )

    identity = KernelRecipeIdentity(
        producer="forge-loop",
        kernel_name="kernel",
        framework="standalone",
        framework_version="unknown",
        backend="triton",
        gpu="mi355x",
    )
    operator_id = kernel_recipe_canonical_id(identity)
    patch_dir = tmp_path / "output" / "result" / "patches" / operator_id
    assert state.status == "completed"
    assert state.analysis_status == "completed"
    assert state.task_count == 1
    assert state.patch_count == 1
    assert patch_dir.is_dir()
    assert "VALUE = 2" in (patch_dir / "change.patch").read_text(encoding="utf-8")
    assert operator_id in (tmp_path / "output" / "result" / "summary.md").read_text(encoding="utf-8")


def test_invalid_agent_task_becomes_no_result_without_starting_forge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _wire_fake_analysis(monkeypatch, _TaskAgentBackend(_source_repo(tmp_path), malformed=True))
    monkeypatch.setattr(
        dispatcher,
        "run_forge_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("forge-loop must not start")),
    )

    state = controller.run_controller(
        handoff_dir=_handoff(tmp_path),
        budget_minutes=120,
        output_dir=tmp_path / "output",
    )

    assert state.status == "no_result"
    assert state.analysis_rejected_task_count == 1
    assert state.task_count == 0
    assert state.patch_count == 0


def test_agent_failure_after_task_write_still_dispatches_and_returns_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _source_repo(tmp_path)
    _wire_fake_analysis(
        monkeypatch,
        _TaskAgentBackend(repo, fail_after_write=True),
    )
    monkeypatch.setattr(dispatcher, "run_forge_loop", _successful_forge)

    state = controller.run_controller(
        handoff_dir=_handoff(tmp_path),
        budget_minutes=120,
        output_dir=tmp_path / "output",
    )

    assert state.status == "partial"
    assert state.analysis_status == "failed"
    assert state.task_count == 1
    assert state.patch_count == 1
