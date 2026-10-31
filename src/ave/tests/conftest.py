# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepared CPU fixtures for the Ave optimization contract."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from ave.task import OptimizationRun, read_prompts, scope_prompts
from hyperloom_common.git_safety import repo_git


@pytest.fixture
def prepared_run(tmp_path: Path) -> OptimizationRun:
    repo = tmp_path / "kernel-repo"
    repo.mkdir()
    repo_git(repo, "init")
    repo_git(repo, "config", "user.name", "Ave test")
    repo_git(repo, "config", "user.email", "ave-test@localhost")
    (repo / "kernel.py").write_text("VALUE = 7\nLATENCY = 2.0\n")
    (repo / "LANGUAGE.md").write_text("The kernel must preserve VALUE = 7.\n")
    repo_git(repo, "add", "kernel.py", "LANGUAGE.md")
    repo_git(repo, "commit", "-m", "baseline")
    workspace = tmp_path / "workspace"
    repo_git(repo, "worktree", "add", "--detach", str(workspace), "HEAD")
    task_dir = tmp_path / "prepared"
    task_dir.mkdir()
    driver = task_dir / "driver.py"
    driver.write_text(
        "import sys\nimport kernel\n"
        "if '--bench-mode' in sys.argv:\n"
        "    print(f'case_ms: decode {kernel.LATENCY}')\n"
        "    print(f'case_ms: prefill {kernel.LATENCY * 2}')\n"
        "    print(f'mean_ms: {kernel.LATENCY * 1.5}')\n"
        "else:\n"
        "    assert kernel.VALUE == 7, 'reference mismatch'\n"
        "    print('allclose: True')\n"
    )
    (task_dir / "prompts.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "id": "prepare_layout",
                        "prompt": "Prepare the layout for register tiling.",
                    },
                    {
                        "id": "register_tiling",
                        "prompt": "Apply register tiling to the prepared layout.",
                    },
                ]
            }
        )
    )
    return OptimizationRun(
        repo_root=repo,
        base_commit=repo_git(repo, "rev-parse", "HEAD").stdout.strip(),
        kernel_path="kernel.py",
        source_files=("kernel.py",),
        operator_name="kernel",
        steps=scope_prompts(read_prompts(task_dir / "prompts.json"), ("kernel.py",)),
        workspace=workspace,
        driver=driver,
        result_json=tmp_path / "result.json",
        deadline_unix=time.time() + 120,
    )
