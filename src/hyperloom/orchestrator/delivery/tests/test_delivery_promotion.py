# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round whose delivery does not reproduce in the served tree does not promote."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task

_PATCH = "diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@t.invalid",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@t.invalid",
    }
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True, env=env)


def _write_round(session: Path, task_id: str) -> None:
    workspace = session / "runs" / "specialist" / task_id
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    (workspace / "worktree" / "patches" / "001.patch").write_text(_PATCH, encoding="utf-8")
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "deliverable": {
                    "tree_id": "fw",
                    "targets": ["src.py"],
                    "patches": ["patches/001.patch"],
                    "artifacts": [{"source": "tuned.json", "target": "tuned.json"}],
                    "envs": {},
                    "server_args": "",
                    "setup_commands": [],
                },
                "proposal_set": [],
            }
        ),
        encoding="utf-8",
    )


def _ctx(params: dict[str, Any]) -> RunnerContext:
    task = Task(
        task_id="t",
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key="t",
        requires_lanes=[],
    )
    return RunnerContext(task=task, lease=None, extra={})


async def _run_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reproduces: bool) -> dict[str, Any]:
    """Drive one integrate_patch round, optionally breaking the delivery in transit."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(tmp_path))
    session = tmp_path / "s"
    session.mkdir()
    repo = tmp_path / "fw"
    _init_repo(repo)
    _write_round(session, "spec")

    validated = tmp_path / "tuned.json"
    validated.write_text('{"tuned": true}\n', encoding="utf-8")
    target = repo / "tuned.json"

    monkeypatch.setattr(
        ip,
        "_resolve_artifact_specs",
        lambda *a, **k: (
            [
                ip._ArtifactSpec(
                    source=validated,
                    target=target,
                    rel_target="tuned.json",
                    root=repo,
                    kind="config_json",
                )
            ],
            [],
        ),
    )

    real_copy = shutil.copy2

    def _copy(src, dst, *args, **kwargs):
        out = real_copy(src, dst, *args, **kwargs)
        if not reproduces and Path(dst) == target:
            # Whatever the cause -- a truncated transport, a hook that rewrites
            # the file on arrival -- the served tree now holds bytes no round
            # validated.
            Path(dst).write_text('{"tuned": false}\n', encoding="utf-8")
        return out

    monkeypatch.setattr(ip.shutil, "copy2", _copy)

    executor = IntegratePatchExecutor(session_dir=session)
    result = await executor(_ctx({"specialist_task_id": "spec", "framework_source_root": str(repo)}))
    result["_frozen"] = executor._frozen_delivery
    result["_target_exists"] = target.exists()
    return result


async def test_a_delivery_that_does_not_reproduce_in_the_served_tree_does_not_promote(tmp_path, monkeypatch):
    result = await _run_round(tmp_path, monkeypatch, reproduces=False)

    frozen = result["_frozen"]
    assert frozen is not None, "the round was applied without freezing what it was validated against"
    assert frozen.artifacts[0].source_sha256, "no frozen digest to check the install against"

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "artifact_install_failed"
    assert any("not what was validated" in str(err) for err in result["error"])
    # Nothing was banked, and the tree does not keep the file that failed.
    assert result["artifacts_applied"] == []
    assert not result["_target_exists"]


async def test_the_same_round_gets_past_the_install_when_the_delivery_reproduces(tmp_path, monkeypatch):
    # Without this the check above proves only that the round failed, not that
    # the digest comparison is what failed it.
    result = await _run_round(tmp_path, monkeypatch, reproduces=True)
    assert result.get("error_class") != "artifact_install_failed"
