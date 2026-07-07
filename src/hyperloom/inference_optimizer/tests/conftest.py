# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pytest hooks and shared helpers for the inference_optimizer tests package."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.paths import make_session_dir


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """N17 isolation: drop the in-process session-dir pin between tests and point MULTI_NODE_STATE_FILE at a missing sentinel so tests run single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SESSION_LAYOUT", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


@pytest.fixture(autouse=True)
def _clear_kernel_request_handler_caches():
    """Clear ``lru_cache`` state on env-bound helpers between tests."""
    from hyperloom.orchestrator.kernel import request_handlers as krh

    krh._default_geak_budget_minutes.cache_clear()
    krh._default_kernel_batch_parallel.cache_clear()


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout."""
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    repo = Path(__file__).resolve().parents[4]
    kernel_agent = repo / "kernel-agent"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir."""
    from hyperloom.inference_optimizer.session_paths import target_baseline_json

    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "no_target_gpu_configured",
                "warning": "compare_against_gpu is empty",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    """A fresh session dir under an isolated ``USER_DATA_PATH``, seeded with the
    ``no_target_gpu_configured`` target-analysis marker.

    Shared across the test package; individual modules may still shadow it with
    a local fixture when they need different seeding.
    """
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    seed_target_analysis_marker(sd)
    return sd


def init_git_repo(
    path: Path,
    *,
    seed_file: str = "src.py",
    seed_text: str = "def f():\n    return 1\n",
) -> None:
    """Initialise a minimal git repo with one commit under ``path``.

    Seeds a single tracked file and commits it so ``git worktree add`` and
    patch application have a base commit to branch from. A fixed non-interactive
    author identity is used (tests do not assert on it).
    """
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
    """Stage everything under ``path`` and commit with a fixed non-interactive identity.

    Mirrors the author identity used by :func:`init_git_repo` so tests do not rely
    on a globally configured git ``user.name``/``user.email`` (absent in CI).
    """
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
