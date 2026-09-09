# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.kernel_rewrite_controller.paths import operator_directory_name

BASE_COMMIT = "a" * 40

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "controller-test",
    "GIT_AUTHOR_EMAIL": "controller-test@local",
    "GIT_COMMITTER_NAME": "controller-test",
    "GIT_COMMITTER_EMAIL": "controller-test@local",
}


def _git(repo: Path, *args: str) -> str:
    """Run one git command in ``repo`` under a fixed identity and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def identity_payload() -> dict[str, str]:
    return {
        "producer": "forge-loop",
        "kernel_name": "fused_moe",
        "framework": "sglang",
        "framework_version": "0.5.0",
        "backend": "triton",
        "gpu": "mi355x",
    }


@pytest.fixture
def operator_id(identity_payload: dict[str, str]) -> str:
    return kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity_payload))


@pytest.fixture
def task_payload(identity_payload: dict[str, str], tmp_path: Path) -> dict:
    repo_root = tmp_path / "contract-repo"
    repo_root.mkdir()
    return {
        "schema_version": 1,
        "identity": identity_payload,
        "base_commit": BASE_COMMIT,
        "repo_root": str(repo_root),
        "kernel_path": "sglang/kernels/fused_moe.py",
        "operator_name": "backend::Fused.MoE-Kernel",
        "driver_path": "driver.py",
        "source_files": ["sglang/kernels/fused_moe.py"],
        "target_functions": ["fused_moe"],
        "shape_cases": [{"name": "decode", "args": [[1, 4096]]}],
        "priority": 1,
        "reason": "High cumulative GPU time",
        "evidence": [{"path": "/trace/analysis.md", "kind": "tracelens"}],
    }


@pytest.fixture
def task_dir(tmp_path: Path, operator_id: str, task_payload: dict) -> Path:
    root = tmp_path / "output" / "controller" / "tasks" / operator_directory_name(operator_id)
    root.mkdir(parents=True)
    (root / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (root / "task.json").write_text(
        json.dumps(task_payload, indent=2),
        encoding="utf-8",
    )
    return root
