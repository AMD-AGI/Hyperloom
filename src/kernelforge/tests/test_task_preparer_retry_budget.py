"""A retry that cannot plausibly finish must not be started.

Measured over 25 recorded prep attempts: successful ones ran 350-896s, and every
retry that began with less than that (150s, 298s, 300s, 325s) burned its entire
budget without writing a byte, then reported "FAILED after 2 attempt(s)" — which
reads like the agent tried twice and failed, not like the second try never had a
chance. A first attempt still always runs, however little time is left.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kernelforge.loop import task_preparer


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    driver = workspace / "driver.py"
    driver.write_text("ORIGINAL\n", encoding="utf-8")
    return workspace, driver


def _patch_git(monkeypatch):
    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _w: None)
    monkeypatch.setattr(task_preparer, "_git_head", lambda _w: "base-head")
    monkeypatch.setattr(task_preparer, "_git_untracked", lambda _w: set())
    monkeypatch.setattr(task_preparer, "_git_diff_patch", lambda *_a: "")
    monkeypatch.setattr(task_preparer, "_git_changed_since", lambda *_a: [])
    monkeypatch.setattr(task_preparer, "_git", lambda _w, *_a: (0, ""))


def _prepare(workspace, driver, tmp_path, *, deadline_sec):
    return asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=str(tmp_path / "experiments")),
            workspace_dir=str(workspace),
            kernel=str(workspace / "kernel.py"),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(workspace / "kernel.py")],
            preflight=task_preparer.PreflightResult(ok=False, correctness_ok=False, bench_ok=False, reasons=["nope"]),
            deadline_sec=deadline_sec,
        )
    )


def _count_attempts(monkeypatch, *, burn_sec):
    """Patch the agent to consume `burn_sec` of the wall per attempt."""
    calls = {"n": 0}
    clock = {"t": 0.0}
    real_monotonic = task_preparer.time.monotonic

    def fake_monotonic():
        return real_monotonic() + clock["t"]

    async def agent(**_kwargs):
        calls["n"] += 1
        clock["t"] += burn_sec
        raise asyncio.TimeoutError

    async def preflight(*_a, **_k):
        return task_preparer.PreflightResult(ok=False, correctness_ok=False, bench_ok=False, reasons=["still bad"])

    monkeypatch.setattr(task_preparer.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", preflight)
    return calls


def test_starved_retry_is_not_started(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    # 1100s wall, first attempt eats 900 -> 200s left, below the 350s floor.
    calls = _count_attempts(monkeypatch, burn_sec=900)

    result = _prepare(workspace, driver, tmp_path, deadline_sec=1100)

    assert calls["n"] == 1
    assert result.attempts == 1
    assert "below the" in result.message
    assert "minimum retry budget" in result.message
    assert "raise the per-kernel deadline" in result.message


def test_retry_still_runs_when_the_budget_is_sufficient(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    # 1900s wall, 600s per attempt -> two retries clear the floor.
    calls = _count_attempts(monkeypatch, burn_sec=600)

    result = _prepare(workspace, driver, tmp_path, deadline_sec=1900)

    assert calls["n"] == 3
    assert result.attempts == 3
    assert "minimum retry budget" not in result.message


def test_first_attempt_always_runs_however_short_the_wall(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    # 200s wall is below the retry floor, but a first try is still worth it.
    calls = _count_attempts(monkeypatch, burn_sec=200)

    result = _prepare(workspace, driver, tmp_path, deadline_sec=200)

    assert calls["n"] == 1
    assert result.attempts == 1


def test_floor_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_PREPARE_MIN_RETRY", "50")
    monkeypatch.setattr(
        task_preparer,
        "PREPARE_MIN_RETRY_SEC",
        int(__import__("os").environ["FORGE_PREPARE_MIN_RETRY"]),
    )
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    calls = _count_attempts(monkeypatch, burn_sec=900)

    _prepare(workspace, driver, tmp_path, deadline_sec=1100)

    # 200s left now clears the lowered floor, so the retry runs.
    assert calls["n"] == 2
