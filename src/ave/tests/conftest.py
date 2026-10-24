# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepared CPU fixtures for the Ave optimization contract."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from ave.task import OptimizationRun, read_prompts, scope_prompts
from ave.workspace import git


@pytest.fixture
def prepared_run(tmp_path: Path) -> OptimizationRun:
    repo = tmp_path / "kernel-repo"
    repo.mkdir()
    git("init", cwd=repo)
    git("config", "user.name", "Ave test", cwd=repo)
    git("config", "user.email", "ave-test@localhost", cwd=repo)
    (repo / "kernel.py").write_text("VALUE = 7\nLATENCY = 2.0\n")
    (repo / "LANGUAGE.md").write_text("The kernel must preserve VALUE = 7.\n")
    git("add", "kernel.py", "LANGUAGE.md", cwd=repo)
    git("commit", "-m", "baseline", cwd=repo)
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
        base_commit=git("rev-parse", "HEAD", cwd=repo).stdout.strip(),
        kernel_path="kernel.py",
        source_files=("kernel.py",),
        operator_name="kernel",
        steps=scope_prompts(read_prompts(task_dir / "prompts.json"), ("kernel.py",)),
        workspace=repo,
        driver=driver,
        result_json=tmp_path / "result.json",
        deadline_unix=time.time() + 120,
    )
