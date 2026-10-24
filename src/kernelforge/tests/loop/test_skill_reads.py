# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which files of an external skill each round's implementer opened: recorded with the round, and shown to the next."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.knowledge.local_wiki.skills import resolve_external_skill
from kernelforge.loop import runner as runner_module
from kernelforge.loop.run_state import LoopStateStore, RunState, make_event
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult
from kernelforge.tracker import ExperimentTracker


@pytest.fixture
def skill(tmp_path: Path):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: flydsl-kernel-optimization\n---\n# FlyDSL\n", encoding="utf-8")
    return resolve_external_skill(root)


def _loop(tmp_path: Path, monkeypatch, skills) -> IterationLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel():\n    return 1\n")
    (workspace / "driver.py").write_text("pass\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "initial"], cwd=workspace, check=True
    )
    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)
    loop = IterationLoop(
        IterationConfig(
            kernel_file=str(workspace / "kernel.py"),
            driver_script=str(workspace / "driver.py"),
            baseline_wall_ms=1.0,
            baseline_case_times={"case": 1.0},
            max_time_hours=1.0,
            git_branch="test-skill-reads",
            workspace_dir=str(workspace),
        ),
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(external_skills=skills),
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.start_time = time.time()
    return loop


def test_a_round_records_the_skill_files_its_session_opened(tmp_path: Path, monkeypatch, skill) -> None:
    loop = _loop(tmp_path, monkeypatch, (skill,))
    sink = {"tool_calls": [("Read", {"file_path": f"{skill.root}/gemm/small_m.md"}), ("Bash", {"command": "ls"})]}
    result = IterationResult(1, 0.0, True, "", skill_reads=loop._skill_reads(sink))

    event = loop._iteration_result_event(result, plan="split K")

    assert event["skill_reads"] == ["flydsl-kernel-optimization/gemm/small_m.md"]


def test_a_round_that_opened_no_skill_file_records_an_empty_list(tmp_path: Path, monkeypatch, skill) -> None:
    loop = _loop(tmp_path, monkeypatch, (skill,))
    result = IterationResult(1, 0.0, True, "", skill_reads=loop._skill_reads({}))
    assert loop._iteration_result_event(result, plan="p")["skill_reads"] == []


def test_the_next_round_is_told_what_the_recent_rounds_opened(tmp_path: Path, monkeypatch, skill) -> None:
    loop = _loop(tmp_path, monkeypatch, (skill,))
    for iteration, reads in enumerate([[], ["flydsl-kernel-optimization/SKILL.md"]], start=1):
        loop.state_store.append_event(make_event("iteration_result", iteration, decision="KEEP", skill_reads=reads))

    block = loop._render_recent_skill_reads()

    assert block.splitlines() == [
        "## Skill files the recent rounds opened",
        "- iteration 1: no skill file opened",
        "- iteration 2: flydsl-kernel-optimization/SKILL.md",
    ]


def test_a_campaign_without_skills_adds_no_block(tmp_path: Path, monkeypatch) -> None:
    loop = _loop(tmp_path, monkeypatch, ())
    loop.state_store.append_event(make_event("iteration_result", 1, decision="KEEP"))
    assert loop._render_recent_skill_reads() == ""
    assert loop._skill_reads({"tool_calls": [("Read", {"file_path": "/x.md"})]}) == []
