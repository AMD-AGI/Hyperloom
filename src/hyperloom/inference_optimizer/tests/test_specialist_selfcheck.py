# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the CPU patch specialist ``selfcheck`` helper and its prompt block."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.specialists import selfcheck


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    tree = tmp_path / "workspace" / "worktree"
    (tree / "python" / "sglang").mkdir(parents=True)
    (tree / "python" / "pyproject.toml").write_text("[project]\nname = 'sglang'\n", encoding="utf-8")
    (tree / "python" / "sglang" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    return tree


@pytest.fixture
def commands(monkeypatch) -> SimpleNamespace:
    """Record every command, faking the venv, constraints, git diff and a configurable pip result."""
    recorder = SimpleNamespace(calls=[], pip_fails=False)
    monkeypatch.setattr(selfcheck, "create_venv", lambda venv: (venv / "bin").mkdir(parents=True))
    monkeypatch.setattr(selfcheck, "write_rocm_torch_constraints", lambda python, path: Path(path).write_text(""))

    def _run(argv, **_kwargs):
        recorder.calls.append(list(argv))
        stdout = "python/sglang/mod.py\n" if argv[:2] == ["git", "diff"] else ""
        return subprocess.CompletedProcess(argv, int("pip" in argv and recorder.pip_fails), stdout=stdout, stderr="")

    monkeypatch.setattr(selfcheck.subprocess, "run", _run)
    return recorder


def test_project_dir_is_the_nearest_packaging_root(worktree: Path):
    assert selfcheck.project_dir(worktree, "python/sglang") == worktree / "python"
    with pytest.raises(FileNotFoundError):
        selfcheck.project_dir(worktree / "python" / "sglang", ".")


def test_selfcheck_installs_imports_compiles_and_tests_in_a_venv_outside_the_worktree(worktree, commands):
    result = selfcheck.run_selfcheck(
        worktree=worktree, package_dir="python/sglang", package="sglang", pytest_targets=["tests/t.py"]
    )

    assert result["ok"] is True
    assert [step["step"] for step in result["steps"]] == ["pip_install", "import", "py_compile", "pytest"]
    pip = next(c for c in commands.calls if "pip" in c)
    assert Path(pip[0]).parent.parent == worktree.parent / "selfcheck_venv"
    assert {"--no-deps", "--no-build-isolation"} <= set(pip)
    assert pip[pip.index("-e") + 1] == str(worktree / "python")
    assert ["-c", "import sglang"] == next(c for c in commands.calls if "-c" in c)[1:]
    assert next(c for c in commands.calls if "py_compile" in c)[-1] == "python/sglang/mod.py"


def test_selfcheck_stops_when_the_install_fails(worktree, commands):
    commands.pip_fails = True
    result = selfcheck.run_selfcheck(
        worktree=worktree, package_dir="python/sglang", package="sglang", pytest_targets=[]
    )
    assert result["ok"] is False
    assert [step["step"] for step in result["steps"]] == ["pip_install"]


def test_cli_prints_the_json_result(worktree, commands, capsys):
    rc = selfcheck.main(["--worktree", str(worktree), "--package-dir", "python/sglang", "--package", "sglang"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_prompt_offers_selfcheck_only_to_cpu_patch_specialists_with_a_worktree():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, _cpu_selfcheck_block
    from hyperloom.orchestrator.specialists.domains import get_domain

    base = dict(task_id="t", domain=get_domain("serving_specialist"), max_turns=4, framework="xdit")
    block = "\n".join(
        _cpu_selfcheck_block(
            SpecialistPromptInputs(**base, workspace_path="/ws/worktree", worktree_package_dir="xfuser")
        )
    )
    assert "--worktree /ws/worktree --package-dir xfuser --package xfuser" in block
    assert not _cpu_selfcheck_block(SpecialistPromptInputs(**base, workspace_path="/ws"))
    assert not _cpu_selfcheck_block(
        SpecialistPromptInputs(**base, worktree_package_dir="xfuser", allocated_gpu_ids=(0,))
    )
    assert not _cpu_selfcheck_block(SpecialistPromptInputs(**base, worktree_package_dir="xfuser", mode="research"))


def test_prompt_offers_no_selfcheck_for_a_framework_without_a_package():
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, _cpu_selfcheck_block
    from hyperloom.orchestrator.specialists.domains import get_domain

    inputs = SpecialistPromptInputs(
        task_id="t",
        domain=get_domain("serving_specialist"),
        max_turns=4,
        framework="custom",
        workspace_path="/ws/worktree",
        worktree_package_dir="hyvideo",
    )
    assert not _cpu_selfcheck_block(inputs)
