# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared plain-function test helpers for orchestrator tests.

These are plain functions (not pytest fixtures) shared across multiple test files
under ``src/hyperloom/orchestrator/``.  They were previously imported via
``from .conftest import X``, which worked only because ``inference_optimizer/tests/``
has ``__init__.py``.  Now that test files live in multiple packages, they need a
real importable module.

Files that still live in ``inference_optimizer/tests/`` import from here as:
    from hyperloom.orchestrator.tests._helpers import init_git_repo, ...
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def init_git_repo(
    path: Path,
    *,
    seed_file: str = "src.py",
    seed_text: str = "def f():\n    return 1\n",
) -> None:
    """Initialise a minimal git repo with one commit under ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        env=env,
    )
    (path / seed_file).write_text(seed_text, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        env=env,
    )


def git_commit_all(path: Path, message: str) -> None:
    """Stage everything under ``path`` and commit with a fixed non-interactive identity."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", message],
        check=True,
        capture_output=True,
        env=env,
    )


def patch_integrate_patch_roots(monkeypatch: Any, tmp_path: Path) -> None:
    """Register common tmp_path framework repos as integrate_patch search roots."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip
    from hyperloom.orchestrator.framework import paths as fp

    real = fp.resolve_kernel_search_roots

    def _merged() -> tuple[str, ...]:
        merged: list[str] = []
        for name in ("fw", "repo", "framework"):
            cand = tmp_path / name
            if cand.is_dir():
                merged.append(str(cand.resolve()))
        for root in real():
            if root not in merged:
                merged.append(root)
        return tuple(merged)

    monkeypatch.setattr(ip, "resolve_kernel_search_roots", _merged)


def variant_result(**overrides: Any) -> Any:
    """A real ``VariantResult`` with plausible defaults."""
    from hyperloom.orchestrator.actions.executors._grid_base import VariantResult

    fields: dict[str, Any] = {
        "name": "v1",
        "extra_server_args": "",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1000.0,
        "ttft_mean_ms": 10.0,
        "tpot_mean_ms": 2.0,
        "error": "",
        "nonfatal_warnings": [],
    }
    fields.update(overrides)
    return VariantResult(**fields)
