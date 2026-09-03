# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.task_publisher import publish_staged_task

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "publisher-test",
    "GIT_AUTHOR_EMAIL": "publisher-test@local",
    "GIT_COMMITTER_NAME": "publisher-test",
    "GIT_COMMITTER_EMAIL": "publisher-test@local",
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


def _staged(layout: ControllerLayout, repo: Path, name: str = "draft") -> Path:
    staged = layout.agent_staging_root / name
    staged.mkdir(parents=True)
    (staged / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (staged / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "producer": "forge-loop",
                    "kernel_name": "kernel",
                    "framework": "standalone",
                    "framework_version": "unknown",
                    "backend": "triton",
                    "gpu": "mi355x",
                },
                "base_commit": "",
                "repo_root": str(repo),
                "kernel_path": "kernel.py",
                "operator_name": "kernel",
                "driver_path": "ignored.py",
                "source_files": ["kernel.py"],
                "target_functions": ["kernel"],
                "shape_cases": [],
                "priority": 0,
                "reason": "offline replay",
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    return staged


def test_publish_pins_live_head_and_moves_complete_task_atomically(tmp_path: Path) -> None:
    repo, head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert not staged.exists()
    payload = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert payload["base_commit"] == head
    assert payload["driver_path"] == "driver.py"


def test_publish_normalizes_harmless_agent_identity_variations(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["identity"].update(
        {
            "producer": " FORGE-LOOP ",
            "kernel_name": " Kernel ",
            "framework": " SGLang ",
            "framework_version": " 0.5.17+ROCM ",
            "backend": " TRITON ",
            "gpu": " MI355X ",
        }
    )
    task_json.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    published = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert published["identity"] == {
        "producer": "forge-loop",
        "kernel_name": "kernel",
        "framework": "sglang",
        "framework_version": "0.5.17+rocm",
        "backend": "triton",
        "gpu": "mi355x",
    }


def test_publish_rejects_a_repo_path_below_git_toplevel(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, nested)

    result = publish_staged_task(layout, staged)

    assert result.published is False
    assert "Git top-level" in result.reason
    assert staged.is_dir()


def test_publish_rejects_duplicate_operator_without_deleting_new_draft(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    first = _staged(layout, repo, "first")
    duplicate = _staged(layout, repo, "duplicate")
    assert publish_staged_task(layout, first).published is True

    result = publish_staged_task(layout, duplicate)

    assert result.published is False
    assert result.reason == "operator task is already published"
    assert duplicate.is_dir()


def test_publish_rejects_a_symlinked_staging_directory(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    real = _staged(layout, repo, "real")
    link = layout.agent_staging_root / "linked"
    link.symlink_to(real, target_is_directory=True)

    result = publish_staged_task(layout, link)

    assert result.published is False
    assert result.reason == "staged task is not a safe directory"
    assert real.is_dir()
