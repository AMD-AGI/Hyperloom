# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a prepared Ave optimization through validation and patch export."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError
import json
import logging
import os
from pathlib import Path
import shlex
import statistics
import sys
import time

from hyperloom_common.kernel_backend import AVE_BACKEND
from ave.measurement import DRIVER_TIMEOUT_SEC, driver_environment, measure
from ave.task import DeadlineExceeded, OptimizationRun, PromptStep, scope_prompts
from ave.workspace import COMMIT_IDENTITY, StepWorkspace, checkout, snapshot
from ave.preparation import PreparedKernel
from hyperloom_common.io import atomic_write_json
from hyperloom_common.git_safety import app_git
from ave.claude import query_result
from hyperloom_common.llm.llm_config import DEFAULT_CLAUDE_MODEL

MAX_SESSION_SEC = 3600
MIN_SESSION_SEC = 30
STEP_ATTEMPTS = 3
FAILED = "failed"
TIMED_OUT = "timed_out"
GIT_DIFF = ("diff", "--no-ext-diff", "--no-textconv")
log = logging.getLogger(__name__)


def publish_candidate(run: OptimizationRun, result: dict) -> dict:
    """Export the verified revision in Hyperloom's existing integrate format."""
    revision = result["final_commit"]
    changed = app_git(
        run.workspace, *GIT_DIFF, "--name-only", run.base_commit, revision, check=True
    ).stdout.splitlines()
    if not changed:
        return {}
    if not set(changed) <= set(run.source_files) or not result["correctness_passed"]:
        raise ValueError("Ave checkpoint does not describe a verified source-scoped candidate")
    output = run.result_json.parent
    snapshot = output / "snapshot"
    for source in changed:
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(app_git(run.workspace, "show", f"{revision}:{source}", check=True).stdout)
    patch = output / "candidate.patch"
    patch.write_text(
        app_git(run.workspace, *GIT_DIFF, "--binary", run.base_commit, revision, "--", *changed, check=True).stdout
    )
    kernel_id = f"ave:{run.operator_name}:{output.name}"
    params = {
        "kernel_id": kernel_id,
        "integration_id": kernel_id,
        "patch_path": str(patch),
        "snapshot_dir": str(snapshot),
        "target_file": str(run.repo_root / run.kernel_path),
        "repo": str(run.repo_root),
        "source": AVE_BACKEND,
        "backend": AVE_BACKEND,
    }
    atomic_write_json(output / "integrate.json", params)
    return params


async def optimize(
    kernel: PreparedKernel,
    *,
    steps: tuple[PromptStep, ...],
    output: Path,
    budget_sec: float,
    run_worker: Callable[[OptimizationRun], Awaitable[int]],
) -> dict:
    """Optimize a prepared workload kernel and export the verified prefix."""
    if budget_sec <= 0:
        return {"status": "skipped", "reason": "time budget exhausted"}
    deadline = time.time() + budget_sec
    steps = scope_prompts(steps, kernel.source_files)
    base_commit = snapshot(kernel.repo_root)
    driver = output / "driver.py"
    driver.write_text(kernel.driver_source, encoding="utf-8")
    with checkout(kernel.repo_root, base_commit, output / "workspace") as workspace:
        run = OptimizationRun(
            repo_root=kernel.repo_root,
            base_commit=base_commit,
            kernel_path=kernel.kernel_path,
            source_files=kernel.source_files,
            operator_name=kernel.operator_name,
            steps=steps,
            reason=kernel.reason,
            workspace=workspace,
            driver=driver,
            result_json=output / "result.json",
            deadline_unix=deadline,
        )
        try:
            returncode = await run_worker(run)
        except (asyncio.CancelledError, CancelledError):
            if run.result_json.exists():
                result = json.loads(run.result_json.read_text())
                result.update(status="cancelled", reason="Ave optimization was cancelled")
                atomic_write_json(run.result_json, result)
                publish_candidate(run, result)
            raise
        if not run.result_json.exists():
            return {"status": "failed", "error": "Ave exited without a checkpoint", "returncode": returncode}
        result = json.loads(run.result_json.read_text())
        if returncode == -9:
            result.update(status=TIMED_OUT, reason="Ave subprocess exceeded its time budget")
        elif result["status"] == "running":
            result.update(status=FAILED, reason=f"Ave subprocess exited before completion (rc={returncode})")
        atomic_write_json(run.result_json, result)
        response = {"result_json": str(run.result_json), "returncode": returncode, "optimization": result}
        return {**response, "status": result["status"], "integrate_params": publish_candidate(run, result)}


class AveLoop:
    """Keep the verified prefix of a caller-supplied sequence of transformations."""

    def __init__(self, run: OptimizationRun):
        self.run = run
        self.model = os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.result: dict = {
            "operator_id": run.operator_name,
            "base_commit": run.base_commit,
            "final_commit": run.base_commit,
            "correctness_passed": False,
            "completed_steps": [],
            "requested_steps": [step.id for step in run.steps],
            "attempts": [],
            "mean_case_speedup": 1.0,
        }

    def checkpoint(self, status: str, reason: str = "") -> dict:
        self.result.update(status=status, reason=reason, agent_model=self.model)
        atomic_write_json(self.run.result_json, self.result)
        return self.result

    async def run_step_agent(self, step: PromptStep, directory: Path, timeout_sec: float) -> None:
        """Let Claude edit and test the current step inside one bounded session."""
        from claude_agent_sdk import ClaudeAgentOptions

        command = ["timeout", str(min(DRIVER_TIMEOUT_SEC, int(timeout_sec))), sys.executable, str(self.run.driver)]
        context = {
            "operator": self.run.operator_name,
            "reason": self.run.reason,
            "step_id": step.id,
            "prompt": step.prompt,
            "editable_files": step.files,
            "driver": str(self.run.driver),
            "correctness_command": shlex.join(command),
            "benchmark_command": shlex.join([*command, "--bench-mode", "--warmup", "3", "--iters", "20"]),
            "artifact_directory": str(directory),
            "time_budget_sec": timeout_sec,
            "baseline_case_ms": self.result["baseline_case_ms"],
            "current_case_ms": self.result["final_case_ms"],
            "completed_steps": self.result["completed_steps"],
            "recent_attempts": self.result["attempts"][-STEP_ATTEMPTS:],
        }
        options = ClaudeAgentOptions(
            model=self.model,
            cwd=str(self.run.workspace),
            add_dirs=[str(self.run.result_json.parent)],
            tools=["Read", "Glob", "Grep", "Edit", "Write", "Bash"],
            permission_mode="acceptEdits",
            env=driver_environment(self.run.workspace),
        )
        (directory / "context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
        await asyncio.wait_for(
            query_result(context, options, operation="optimize_step", transcript=directory / "claude.jsonl"),
            timeout=timeout_sec,
        )

    def evaluate(self, step: PromptStep, directory: Path, workspace: StepWorkspace) -> dict:
        changed = workspace.validate(step.files)
        measured = measure(self.run, directory)
        scores = measured.scores(self.result["baseline_case_ms"])
        if changed:
            app_git(self.run.workspace, "add", "--", *step.files, check=True)
            app_git(
                self.run.workspace,
                *COMMIT_IDENTITY,
                "commit",
                "-m",
                f"Apply Ave optimization step: {step.id}",
                check=True,
            )
        self.result.update(
            final_commit=app_git(self.run.workspace, "rev-parse", "HEAD", check=True).stdout.strip(),
            final_case_ms=measured.case_ms,
            mean_case_speedup=statistics.fmean(scores),
        )
        self.result["completed_steps"].append(step.id)
        return {"status": "passed", "case_ms": measured.case_ms, "commit": self.result["final_commit"]}

    async def execute_step(self, step: PromptStep, index: int, reserve: float) -> bool:
        for attempt in range(1, STEP_ATTEMPTS + 1):
            remaining = self.run.deadline_unix - time.time() - reserve
            if remaining < MIN_SESSION_SEC:
                raise DeadlineExceeded("Ave deadline reached before the next step attempt")
            directory = self.run.logs / f"{index:04d}-{attempt}"
            directory.mkdir()
            workspace = StepWorkspace(self.run.workspace, self.result["final_commit"], self.run.driver)
            timeout_sec = min(MAX_SESSION_SEC, remaining)
            try:
                await self.run_step_agent(step, directory, timeout_sec)
                record = self.evaluate(step, directory, workspace)
            except (DeadlineExceeded, CancelledError):
                raise
            except asyncio.TimeoutError:
                record = {"status": FAILED, "reason": f"Claude session timed out after {timeout_sec:g}s"}
            except Exception as error:  # isolation boundary: one optimization attempt
                log.warning("Ave step %s attempt %s failed", step.id, attempt, exc_info=True)
                record = {"status": FAILED, "reason": str(error)}
            finally:
                workspace.restore(self.result["final_commit"])
            record.update(step_id=step.id, attempt=attempt)
            self.result["attempts"].append(record)
            atomic_write_json(directory / "result.json", record)
            self.checkpoint("running")
            if record["status"] == "passed":
                return True
        return False

    async def optimize(self) -> dict:
        self.run.logs.mkdir(parents=True, exist_ok=True)
        try:
            started = time.monotonic()
            baseline = measure(self.run, self.run.logs / "baseline")
            reserve = max(MIN_SESSION_SEC, (time.monotonic() - started) * 1.5)
            self.result.update(
                baseline_case_ms=baseline.case_ms,
                final_case_ms=baseline.case_ms,
                correctness_passed=True,
            )
            self.checkpoint("running")
            for index, step in enumerate(self.run.steps, start=1):
                if not await self.execute_step(step, index, reserve):
                    return self.checkpoint(FAILED, f"step {step.id} failed after {STEP_ATTEMPTS} attempts")
        except DeadlineExceeded as error:
            return self.checkpoint(TIMED_OUT, str(error))
        except (asyncio.CancelledError, CancelledError) as error:
            self.checkpoint("cancelled", str(error))
            raise
        except Exception as error:
            self.checkpoint(FAILED, str(error))
            raise
        return self.checkpoint("completed")
