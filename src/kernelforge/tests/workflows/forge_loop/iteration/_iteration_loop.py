# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders and fakes the iteration-loop test modules share."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationConfig, IterationLoop, IterationResult
from kernelforge.workflows.forge_loop.campaign.experiment import ExperimentTracker
from kernelforge.workflows.forge_loop.iteration import attempt_verdict
from kernelforge.tests.conftest import patch_iteration_name, silence_jit_rebuild


# The stand-in for "budget is not what this test is about". It has to clear the
# round admission guard, which prices a whole round -- planning, a session worth
# starting, the canonical measurement and the finalize reserve -- so a value near
# the reserve itself would silently turn these tests into admission tests.
_AMPLE_BUDGET_SEC = 12 * 3600.0


def _initialize_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver.write_text("pass\n")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "KernelForge Tests"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    silence_jit_rebuild(monkeypatch)
    return workspace, kernel, driver


def _make_loop(
    tmp_path,
    monkeypatch,
    *,
    supervise_after=5,
    session_count=1,
    resume=False,
    baseline_case_times=None,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)

    config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        baseline_wall_ms=1.0,
        baseline_case_times=({"case": 1.0} if baseline_case_times is None else dict(baseline_case_times)),
        max_time_hours=1.0,
        git_branch="test-loop",
        workspace_dir=str(workspace),
        supervise_after=supervise_after,
        supervise_cooldown=0,
    )
    tracker = ExperimentTracker(tmp_path / "experiments")
    loop = IterationLoop(
        config,
        tracker,
        config=SimpleNamespace(gpu_target="gfx942"),
        resume=resume,
    )
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(loop.results) < session_count else 0.0,
    )
    return loop, workspace


async def _unused_supervisor(**_kwargs):
    return ""


async def _no_change_agent(_kernel_path, _history, session_sink):
    session_sink["plan"] = "inspect only"
    return "No source change was needed."


def _reduction_loop(tmp_path, monkeypatch, workspace_state=None):
    """A loop wired only for reducing outcomes into durable control state."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = workspace_state if workspace_state is not None else RunState()
    loop.best_wall_ms = 1.0
    loop.best_mean_case_speedup = 1.0
    return loop, workspace


def _lane_agent_factory(edits):
    """An agent that writes the edit its lane was assigned, in that lane's copy."""

    def factory(lane_dir, _serialized_driver):
        async def agent(_kernel_file, prompt, session_sink=None):
            path = Path(lane_dir) / "kernel.py"
            for plan, (old, new) in edits.items():
                if plan in prompt:
                    path.write_text(path.read_text().replace(old, new))
                    return

        return agent

    return factory


def _counting_plan(loop, workspace, rounds, *, plans=("a", "b"), unavailable=False):
    """Stand in for planning, recording every round an iteration is charged for."""

    async def _plan(*, iteration, lanes=1, **_kwargs):
        rounds.append(lanes)
        if unavailable:
            return None, "OrchestrationInfrastructureError: backend unreachable"
        loop._last_lane_plans = list(plans)
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(f"# Optimization plan\n{plans[0]}\n")
        return plan_path, ""

    return _plan


def _diff_of(workspace: Path, edit) -> str:
    """A patch for one edit, taken back off the tree once it is captured."""
    restore = {path: path.read_text() for path in workspace.rglob("*.py") if ".git" not in path.parts}
    edit()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "-M", "--", "."],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subprocess.run(
        ["git", "restore", "--source=HEAD", "--staged", "--worktree", "--", "."],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    for path, text in restore.items():
        path.write_text(text)
    return diff


def _measurement_loop(monkeypatch, benchmark_result, workspace_dir="."):
    report = SimpleNamespace(
        all_passed=True,
        results=[
            SimpleNamespace(stage=5, stage_name="full", passed=True, status_label="PASS", snr_db=40.0),
        ],
        summary=lambda: "PASS",
    )
    benchmark_calls = []

    async def fake_validation(**_kwargs):
        return report

    async def fake_benchmark(_run, **kwargs):
        benchmark_calls.append(kwargs)
        return dict(benchmark_result)

    async def fake_registers(**_kwargs):
        return {"success": False}

    monkeypatch.setattr(attempt_verdict, "run_validation_pipeline", fake_validation)
    patch_iteration_name(monkeypatch, "measure_wallclock", fake_benchmark)
    monkeypatch.setattr(attempt_verdict, "check_registers", fake_registers)
    silence_jit_rebuild(monkeypatch)

    loop = object.__new__(IterationLoop)
    loop.ic = SimpleNamespace(
        build_command=None,
        driver_script="driver.py",
        snr_threshold=30.0,
        validate_stage_timeout_sec=300,
        bench_timeout_sec=300,
        bench_repeat=1,
        nproc_per_node=1,
        build_dir=None,
        baseline_wall_ms=5.0,
        kernel_backend="",
        kernel_file="kernel.py",
        source_files=[],
        target_functions=[],
        workspace_dir=str(workspace_dir),
    )
    loop.best_wall_ms = 5.0
    loop.best_mean_case_speedup = 1.0
    loop.experiment = None
    loop._baseline_case_times = {"small": 1.0, "large": 1.0}
    loop._best_case_times = {"small": 1.0, "large": 1.0}
    loop._case_times = {}
    loop.config = SimpleNamespace(gpu_target="gfx942")
    # No round is open around these iterations, so the measurement they run is
    # charged to nothing -- which is also what a drain iteration does.
    loop._round_started_at = None
    loop._round_measurement_sec = 0.0
    return loop, benchmark_calls


# ── lesson recording ──────────────────────────────────────────────────────────


def _lesson_result(**overrides) -> IterationResult:
    base = dict(
        iteration=3,
        duration_sec=1.0,
        validation_passed=True,
        validation_summary="all stages passed",
        wall_ms=1.5,
        snr_db=41.0,
        kept=False,
        session_end_reason="turn_cap",
    )
    base.update(overrides)
    return IterationResult(**base)


# ── end-to-end: agent session -> lesson document -> next prompt ───────────────


class _StubStage:
    def __init__(self, stage, name, snr_db):
        self.stage, self.stage_name, self.snr_db = stage, name, snr_db
        self.passed = True
        self.status_label = "PASS"


class _StubReport:
    all_passed = True
    failed_stage = 0
    failed_output = ""
    results = [_StubStage(5, "snr", 55.0)]

    def summary(self):
        return "all stages passed"


def _stub_measurement(monkeypatch, walls):
    """Stub build/validate/bench so the loop runs without a GPU."""

    async def fake_validation(**_kwargs):
        return _StubReport()

    async def fake_bench(_run, **_kwargs):
        wall_ms = walls.pop(0) if walls else 1.0
        return {
            "success": True,
            "median_ms": wall_ms,
            "case_times": {"case": wall_ms},
            "unscored_cases": [],
            "measurements": [
                {
                    "success": True,
                    "case_times": {"case": wall_ms},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "stub",
        }

    async def fake_registers(**_kwargs):
        return {"success": False}

    monkeypatch.setattr(attempt_verdict, "run_validation_pipeline", fake_validation)
    patch_iteration_name(monkeypatch, "measure_wallclock", fake_bench)
    monkeypatch.setattr(attempt_verdict, "check_registers", fake_registers)
