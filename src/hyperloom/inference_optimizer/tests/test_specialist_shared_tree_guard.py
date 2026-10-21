# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch specialist that writes the shared framework tree is caught, reverted and discarded."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.orchestrator.actions.executors import integrate_patch as integrate_module
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.specialists import tree_snapshot as ts
from hyperloom.orchestrator.specialists.runner import (
    SpecialistFailureType,
    SpecialistRunner,
    classify_specialist_failure,
)
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessResult,
    head_commit,
)
from hyperloom.orchestrator.specialists.tree_snapshot import (
    TreeRestore,
    baseline_snapshot,
    held_snapshot,
    refresh_snapshots,
    restore_tree,
    snapshot_tree,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task

_DONE: dict[str, Any] = {
    "gap_canonical_id": "gap.test.example",
    "domain": "serving_specialist",
    "proposal_set": [{"name": "valuable", "extra_args": "--fast", "extra_envs": {}, "reason": "real gain"}],
    "patches_written": [],
    "summary": "a result worth keeping",
    "confidence": 0.9,
}


def _installed_package(root: Path) -> Path:
    """A pip-installed package with source, a shared object and a bytecode cache, and no git."""
    package = root / "vllm"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "envs.py").write_text("VLLM_USE_X = 0\n", encoding="utf-8")
    (package / "models" / "qwen3.py").write_text("BLOCK = 64\r\nKEEP = 1\n", encoding="utf-8")
    (package / "_C.abi3.so").write_bytes(b"\x7fELF shared object")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "envs.cpython-312.pyc").write_bytes(b"\x00bytecode")
    return package


def _contents(tree: Path) -> dict[str, bytes]:
    return {p.relative_to(tree).as_posix(): p.read_bytes() for p in sorted(tree.rglob("*")) if p.is_file()}


def _trespass(package: Path) -> None:
    """Modify, delete and add files the way a specialist editing the installed tree would."""
    (package / "envs.py").write_text("VLLM_USE_X = 1\n", encoding="utf-8")
    (package / "models" / "qwen3.py").unlink()
    (package / "models" / "fast_path.py").write_text("FAST = True\n", encoding="utf-8")
    (package / "new_pkg" / "inner").mkdir(parents=True)
    (package / "new_pkg" / "inner" / "mod.py").write_text("x = 1\n", encoding="utf-8")


def _build_output(package: Path) -> None:
    """What importing and building the package leaves behind; none of it is source."""
    (package / "models" / "__pycache__").mkdir()
    (package / "models" / "__pycache__" / "qwen3.cpython-312.pyc").write_bytes(b"\x00bc")
    (package / "fresh.pyo").write_bytes(b"\x00bc")
    (package / "_C.abi3.so").write_bytes(b"\x7fELF rebuilt")
    (package / "_moe.so.1").write_bytes(b"\x7fELF versioned")


# -- tree_snapshot ------------------------------------------------------------
def test_restore_puts_a_trespassed_tree_back_byte_for_byte(tmp_path: Path) -> None:
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    before = _contents(package)
    repo, err = snapshot_tree(session, package)
    assert err == "" and repo is not None

    _trespass(package)
    restore = restore_tree(session, package)

    assert restore.error == ""
    assert set(restore.changed) == {"envs.py", "models/qwen3.py", "models/fast_path.py", "new_pkg/inner/mod.py"}
    assert _contents(package) == before
    assert not (package / "new_pkg").exists()
    assert restore_tree(session, package) == TreeRestore()


def test_build_output_is_not_a_trespass(tmp_path: Path) -> None:
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    snapshot_tree(session, package)

    _build_output(package)
    after = _contents(package)

    assert restore_tree(session, package) == TreeRestore()
    assert _contents(package) == after


def test_no_snapshot_is_held_before_the_first_one(tmp_path: Path) -> None:
    package = _installed_package(tmp_path / "t")
    assert held_snapshot(tmp_path / "s", package) is None
    assert restore_tree(tmp_path / "s", package) == TreeRestore()


def test_a_snapshot_recorded_before_its_tree_was_named_still_refreshes(tmp_path: Path) -> None:
    """A session that took its snapshot before the tree was recorded in it gets the record on its next dispatch."""
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    repo, _ = snapshot_tree(session, package)
    subprocess.run(["git", "--git-dir", str(repo), "config", "--unset", "hyperloom.tree"], check=True)
    assert refresh_snapshots(session) != []

    baseline_snapshot(session, package, ())
    (package / "envs.py").write_text("VLLM_USE_X = 4\n", encoding="utf-8")

    assert refresh_snapshots(session) == []
    assert restore_tree(session, package) == TreeRestore()


def test_a_checkout_is_guarded_as_the_operator_left_it(tmp_path: Path) -> None:
    """The baseline is the working tree, uncommitted operator work included, never its HEAD; .git is not part of it."""
    checkout = tmp_path / "c"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "src.py").write_text("A = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed"],
        check=True,
    )
    (checkout / "src.py").write_text("A = 2  # operator work\n", encoding="utf-8")
    (checkout / "notes.txt").write_text("operator notes\n", encoding="utf-8")
    session = tmp_path / "s"
    snapshot_tree(session, checkout)
    before = {k: v for k, v in _contents(checkout).items() if not k.startswith(".git/")}
    head = head_commit(checkout)

    (checkout / "src.py").write_text("A = 3\n", encoding="utf-8")
    (checkout / "added.py").write_text("B = 1\n", encoding="utf-8")
    restore = restore_tree(session, checkout)

    assert set(restore.changed) == {"src.py", "added.py"}
    assert {k: v for k, v in _contents(checkout).items() if not k.startswith(".git/")} == before
    assert head_commit(checkout) == head


def test_a_session_inside_the_tree_is_not_part_of_its_baseline(tmp_path: Path) -> None:
    package = _installed_package(tmp_path / "t")
    session = package / "sessions" / "s1"
    snapshot_tree(session, package)

    (session / "runs").mkdir(parents=True)
    (session / "runs" / "log.txt").write_text("session output\n", encoding="utf-8")

    assert restore_tree(session, package) == TreeRestore()
    assert (session / "runs" / "log.txt").is_file()


# -- runner -------------------------------------------------------------------
def _guarded_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    during_run: Callable[[Path], Any],
    other_roots: tuple[str, ...] = (),
) -> tuple[SpecialistRunner, RunnerContext, Path, Path]:
    """A runner whose specialist calls ``during_run`` on the shared package before it reports ``_DONE``.

    ``other_roots`` are the Section 7 roots listed beside the package.
    """
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    session.mkdir()
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(framework_source_roots=(str(package),)),
        session_dir=session,
    )

    async def _run(spawn: Any) -> SpecialistSubprocessResult:
        outcome = during_run(package)
        if inspect.isawaitable(outcome):
            await outcome
        return SpecialistSubprocessResult(done_payload=json.loads(json.dumps(_DONE)), exit_code=0)

    monkeypatch.setattr(runner.subprocess_dispatcher, "run", _run)
    task = Task(
        task_id="t-guard",
        kind="specialist",
        state="queued",
        params={
            "domain": "serving_specialist",
            "framework": "vllm",
            "session_framework_tree": str(package),
            "framework_source_roots": [str(package), *other_roots],
            "gap_canonical_id": "gap.test.example",
            "max_turns": 2,
        },
        idempotency_key="t-guard",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": SharedState.load_or_init(session)})
    return runner, ctx, package, session


def _workspace_done(session: Path) -> dict[str, Any]:
    return json.loads((session / "runs" / "specialist" / "t-guard" / "specialist_done.json").read_text("utf-8"))


async def test_a_specialist_that_writes_the_shared_tree_is_reverted_and_discarded(tmp_path, monkeypatch, caplog):
    before = _contents(_installed_package(tmp_path / "ref"))
    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, _trespass)

    with caplog.at_level(logging.ERROR, logger=ts.log.name):
        result = await runner.run(ctx)

    assert _contents(package) == before
    assert result.status == "shared_tree_violation"
    assert result.error_class == "shared_tree_violation"
    assert "envs.py" in result.error and "new_pkg/inner/mod.py" in result.error
    assert result.specialist_done["proposal_set"] == []
    assert _workspace_done(session)["proposal_set"] == []
    assert any("t-guard" in r.getMessage() and "models/fast_path.py" in r.getMessage() for r in caplog.records)
    assert not ctx.extra["shared_state"].stop_reason
    assert classify_specialist_failure(result.status, result.error) == (
        SpecialistFailureType.SHARED_TREE_VIOLATION,
        False,
    )


async def test_a_specialist_that_leaves_the_shared_tree_alone_keeps_its_result(tmp_path, monkeypatch):
    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, _build_output)

    result = await runner.run(ctx)

    assert result.status == "succeeded" and result.error_class == ""
    assert [p["name"] for p in result.specialist_done["proposal_set"]] == ["valuable"]
    assert (package / "_moe.so.1").is_file()


def _kernel_keep(package: Path) -> None:
    """What a kernel-lane KEEP between two specialists writes into the shared tree."""
    (package / "envs.py").write_text("VLLM_USE_X = 9  # kernel KEEP\n", encoding="utf-8")
    (package / "kernels").mkdir()
    (package / "kernels" / "tuned_moe.py").write_text("TILE = 128\n", encoding="utf-8")


@pytest.mark.parametrize("trespasses", [False, True])
async def test_a_legitimate_write_between_two_runs_is_not_the_next_runs(tmp_path, monkeypatch, trespasses):
    """The next dispatch takes the tree as it is, so its check keeps the KEEP and reverts only its own writes."""
    second_run = _trespass if trespasses else _build_output
    runs = iter([lambda _p: None, second_run])
    runner, ctx, package, _session = _guarded_run(tmp_path, monkeypatch, lambda p: next(runs)(p))
    assert (await runner.run(ctx)).status == "succeeded"

    _kernel_keep(package)
    after_keep = _contents(package)
    ctx.task.task_id = "t-guard-next"
    result = await runner.run(ctx)

    assert (package / "kernels" / "tuned_moe.py").read_text(encoding="utf-8") == "TILE = 128\n"
    if trespasses:
        assert _contents(package) == after_keep
        assert result.status == "shared_tree_violation"
        assert "envs.py" in result.error and "kernels/" not in result.error
    else:
        assert result.status == "succeeded" and result.error_class == ""
        assert [p["name"] for p in result.specialist_done["proposal_set"]] == ["valuable"]
    assert not ctx.extra["shared_state"].stop_reason


async def test_a_snapshot_that_cannot_follow_the_tree_is_dropped_not_held_against_it(tmp_path, monkeypatch, caplog):
    """A failed refresh at dispatch is logged, stops nothing, and leaves the run unchecked rather than wrongly judged."""
    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, lambda _p: None)
    await runner.run(ctx)
    _kernel_keep(package)
    after_keep = _contents(package)
    monkeypatch.setattr(ts, "_commit", lambda *_a: "git commit rc=128: No space left on device")

    ctx.task.task_id = "t-guard-next"
    with caplog.at_level(logging.WARNING, logger=ts.log.name):
        result = await runner.run(ctx)

    assert held_snapshot(session, package) is None
    assert result.status == "succeeded" and _contents(package) == after_keep
    assert any("No space left on device" in r.getMessage() for r in caplog.records)
    assert not ctx.extra["shared_state"].stop_reason
    monkeypatch.undo()
    assert baseline_snapshot(session, package, ())[1] == ""
    assert restore_tree(session, package) == TreeRestore() and _contents(package) == after_keep


async def test_the_tree_is_not_judged_while_an_integrate_is_writing_it(tmp_path, monkeypatch):
    def _integrate_starts_then_specialist_writes(package: Path) -> None:
        state = SharedState.load_or_init(session)
        state.pending_integrate = {"task_id": "t-integrate", "patches": []}
        state.save(session)
        (package / "envs.py").write_text("VLLM_USE_X = 5\n", encoding="utf-8")

    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, _integrate_starts_then_specialist_writes)

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    assert (package / "envs.py").read_text(encoding="utf-8") == "VLLM_USE_X = 5\n"


async def test_a_tree_that_cannot_be_put_back_stops_the_session(tmp_path, monkeypatch):
    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, _trespass)
    monkeypatch.setattr(ts, "restore_tree", lambda *_a: TreeRestore(changed=("envs.py",), error="git reset rc=1"))

    result = await runner.run(ctx)

    assert result.status == "shared_tree_violation"
    assert result.specialist_done["proposal_set"] == []
    assert ctx.extra["shared_state"].stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    assert SharedState.load_or_init(session).stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON


async def test_a_killed_specialist_still_has_its_trespass_reverted(tmp_path, monkeypatch):
    before = _contents(_installed_package(tmp_path / "ref"))

    def _trespass_then_get_killed(package: Path) -> None:
        _trespass(package)
        raise asyncio.CancelledError("session stop")

    runner, ctx, package, _session = _guarded_run(tmp_path, monkeypatch, _trespass_then_get_killed)

    with pytest.raises(asyncio.CancelledError):
        await runner.run(ctx)
    assert _contents(package) == before


def _resume(ctx: RunnerContext, session: Path, task_id: str) -> None:
    """Resume the session the way an operator does: the stop is lifted, the next task dispatched."""
    state = SharedState.load_or_init(session)
    state.set_stop_reason("")
    state.save(session)
    ctx.extra["shared_state"] = state
    ctx.task.task_id = task_id


def _unrestorable(*_a: Any) -> TreeRestore:
    return TreeRestore(changed=("envs.py",), error="git reset rc=1: index.lock exists")


async def test_a_resumed_session_puts_the_tree_back_before_it_takes_a_new_baseline(tmp_path, monkeypatch):
    """The contamination a failed restore left behind is not absorbed into the next run's baseline."""
    before = _contents(_installed_package(tmp_path / "ref"))
    writes = iter([_trespass])
    runner, ctx, package, session = _guarded_run(tmp_path, monkeypatch, lambda p: next(writes, lambda _p: None)(p))
    with monkeypatch.context() as patched:
        patched.setattr(ts, "restore_tree", _unrestorable)
        assert (await runner.run(ctx)).status == "shared_tree_violation"
    assert ctx.extra["shared_state"].stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    _resume(ctx, session, "t-guard-resumed")
    result = await runner.run(ctx)

    assert result.status == "succeeded" and result.error_class == ""
    assert _contents(package) == before
    assert not ctx.extra["shared_state"].stop_reason
    _trespass(package)
    assert restore_tree(session, package).changed


async def test_a_resumed_session_whose_tree_still_cannot_be_put_back_stops_before_the_run(tmp_path, monkeypatch):
    runs: list[Path] = []
    runner, ctx, _package, session = _guarded_run(tmp_path, monkeypatch, lambda p: runs.append(p) or _trespass(p))
    monkeypatch.setattr(ts, "restore_tree", _unrestorable)
    await runner.run(ctx)

    _resume(ctx, session, "t-guard-resumed")
    result = await runner.run(ctx)

    assert len(runs) == 1
    assert result.status == "shared_tree_violation" and result.specialist_done["proposal_set"] == []
    assert "index.lock" in result.error
    assert SharedState.load_or_init(session).stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON


def _fail_to_read(target: str) -> Callable[..., Any]:
    def _raise(*_a: Any) -> Any:
        if target == "_drift":
            raise OSError("git status rc=128: fatal: index file corrupt")
        raise subprocess.TimeoutExpired("git rev-parse", 600)

    return _raise


@pytest.mark.parametrize("target", ["_drift", "held_snapshot"])
async def test_a_tree_that_cannot_be_read_keeps_the_result(tmp_path, monkeypatch, caplog, target):
    """Not being able to check a run is not evidence against it: logged, the result kept, the session going on."""
    runner, ctx, _package, _session = _guarded_run(
        tmp_path, monkeypatch, lambda _p: monkeypatch.setattr(ts, target, _fail_to_read(target))
    )

    with caplog.at_level(logging.ERROR, logger=ts.log.name):
        result = await runner.run(ctx)

    assert result.status == "succeeded" and result.error_class == ""
    assert [p["name"] for p in result.specialist_done["proposal_set"]] == ["valuable"]
    assert not ctx.extra["shared_state"].stop_reason
    assert any(r.levelno == logging.ERROR and "t-guard" in r.getMessage() for r in caplog.records)


async def test_a_stop_that_cannot_be_saved_does_not_mask_the_discard(tmp_path, monkeypatch):
    def _trespass_then_lose_the_disk(package: Path) -> None:
        _trespass(package)

        def _save(*_a: Any, **_k: Any) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr(ctx.extra["shared_state"], "save", _save)

    runner, ctx, _package, _session = _guarded_run(tmp_path, monkeypatch, _trespass_then_lose_the_disk)
    monkeypatch.setattr(ts, "restore_tree", _unrestorable)

    result = await runner.run(ctx)

    assert result.status == "shared_tree_violation"
    assert ctx.extra["shared_state"].stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON


# -- dependency checkouts -----------------------------------------------------
def _git(path: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


def _dependency_checkout(path: Path) -> Path:
    """A shared dependency such as aiter: a checkout with two commits, sitting at the second."""
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "csrc").mkdir()
    for n in (1, 2):
        (path / "csrc" / "op.py").write_text(f"V = {n}\n", encoding="utf-8")
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", f"c{n}")
    return path


async def test_a_specialist_that_moves_a_dependency_checkout_stops_the_session(tmp_path, monkeypatch):
    """A checked-out dependency breaks every later launch: the run is discarded and the session held until it is back."""
    dep = _dependency_checkout(tmp_path / "aiter")
    head, older = _git(dep, "rev-parse", "HEAD"), _git(dep, "rev-parse", "HEAD~1")
    runs: list[str] = []

    def _check_out_an_older_dependency(_package: Path) -> None:
        runs.append(ctx.task.task_id)
        if len(runs) == 1:
            _git(dep, "checkout", "-q", older)

    roots = (str(dep), str(dep / "csrc"))
    runner, ctx, _package, session = _guarded_run(tmp_path, monkeypatch, _check_out_an_older_dependency, roots)

    result = await runner.run(ctx)

    assert result.status == "shared_tree_violation" and result.specialist_done["proposal_set"] == []
    assert str(dep.resolve()) in result.error and older[:12] in result.error
    assert result.error.count(head[:12]) == 1
    assert ctx.extra["shared_state"].stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    assert _git(dep, "rev-parse", "HEAD") == older

    _resume(ctx, session, "t-guard-resumed")
    held = await runner.run(ctx)
    assert held.status == "shared_tree_violation" and runs == ["t-guard"]
    assert ctx.extra["shared_state"].stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON

    _git(dep, "checkout", "-q", head)
    _resume(ctx, session, "t-guard-fixed")
    fixed = await runner.run(ctx)
    assert fixed.status == "succeeded" and runs == ["t-guard", "t-guard-fixed"]
    assert not ctx.extra["shared_state"].stop_reason


async def test_a_dependency_checkout_is_not_judged_while_an_integrate_is_in_flight(tmp_path, monkeypatch):
    dep = _dependency_checkout(tmp_path / "aiter")

    def _integrate_starts_then_dependency_moves(_package: Path) -> None:
        state = SharedState.load_or_init(session)
        state.pending_integrate = {"task_id": "t-integrate", "patches": []}
        state.save(session)
        _git(dep, "checkout", "-q", "HEAD~1")

    runner, ctx, _package, session = _guarded_run(
        tmp_path, monkeypatch, _integrate_starts_then_dependency_moves, (str(dep),)
    )

    assert (await runner.run(ctx)).status == "succeeded"
    assert not ctx.extra["shared_state"].stop_reason


# -- integrate_patch ----------------------------------------------------------
_SETTLED_PENDING: dict[str, Any] = {"task_id": "t-int", "recovery": {"version": 1, "phase": "accepted"}}


async def _settle_integrate(
    session: Path,
    monkeypatch: pytest.MonkeyPatch,
    verdict: dict[str, Any] | BaseException,
    *,
    pending: dict[str, Any] = _SETTLED_PENDING,
    early: dict[str, Any] | None = None,
    write: Callable[[], Any] = lambda: None,
) -> dict[str, Any]:
    """Drive integrate_patch through an apply that records ``pending`` and runs ``write``, then to ``verdict``.

    ``verdict`` is the gate's, except that an ``applied_no_bench`` or ``apply_failed`` status ends the attempt at
    the apply stage, and an exception is raised there.
    """
    executor = IntegratePatchExecutor(session_dir=session)

    async def _resolve(*_a: Any) -> dict[str, Any] | None:
        return early

    async def _nothing(*_a: Any) -> None:
        return None

    async def _apply(attempt: IntegrateAttempt, *_a: Any) -> dict[str, Any] | None:
        attempt.pending = json.loads(json.dumps(pending))
        write()
        if isinstance(verdict, BaseException):
            raise verdict
        return dict(verdict) if verdict["status"] in ("applied_no_bench", "apply_failed") else None

    async def _gate(*_a: Any) -> dict[str, Any]:
        assert isinstance(verdict, dict)
        return dict(verdict)

    with monkeypatch.context() as patched:
        patched.setattr(executor, "_stage_resolve", _resolve)
        patched.setattr(executor, "_stage_provision_attempt_runtime", _nothing)
        patched.setattr(executor, "_stage_localize_source", _nothing)
        patched.setattr(executor, "_stage_apply", _apply)
        patched.setattr(executor, "_stage_gate", _gate)
        return await executor(SimpleNamespace(task=SimpleNamespace(task_id="t-int", params={}), extra={}))


def _seven(package: Path) -> Callable[[], Any]:
    return lambda: (package / "envs.py").write_text("VLLM_USE_X = 7\n", encoding="utf-8")


@pytest.mark.parametrize("status", ["kept", "kept_inert", "reverted"])
async def test_a_settled_integrate_moves_the_snapshot_to_the_tree(tmp_path, monkeypatch, status):
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    snapshot_tree(session, package)

    result = await _settle_integrate(session, monkeypatch, {"status": status}, write=_seven(package))

    assert result["status"] == status
    assert restore_tree(session, package) == TreeRestore()
    assert (package / "envs.py").read_text(encoding="utf-8") == "VLLM_USE_X = 7\n"


@pytest.mark.parametrize("case", ["early_exit", "apply_failed", "applied_only", "cancelled", "restore_failed"])
async def test_an_integrate_without_a_verdict_leaves_the_snapshot_alone(tmp_path, monkeypatch, case):
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    repo, _ = snapshot_tree(session, package)
    first = head_commit(repo)
    _seven(package)()
    settle: dict[str, Any] = {
        "early_exit": {"verdict": {"status": "kept"}, "early": {"status": "skipped"}},
        "apply_failed": {"verdict": {"status": "apply_failed"}},
        "applied_only": {"verdict": {"status": "applied_no_bench"}},
        "cancelled": {"verdict": asyncio.CancelledError("session stop")},
        "restore_failed": {"verdict": {"status": "reverted"}, "pending": {"patches": ["p.patch"]}},
    }[case]

    if case == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await _settle_integrate(session, monkeypatch, **settle)
    else:
        result = await _settle_integrate(session, monkeypatch, **settle)
        assert (result.get("error_class") == "integrate_restore_incomplete") == (case == "restore_failed")

    assert head_commit(repo) == first


async def test_a_settled_integrate_moves_the_snapshot_off_the_event_loop(tmp_path, monkeypatch):
    threads: list[threading.Thread] = []
    monkeypatch.setattr(integrate_module, "refresh_snapshots", lambda _s: threads.append(threading.current_thread()))

    await _settle_integrate(tmp_path / "s", monkeypatch, {"status": "kept"})

    assert len(threads) == 1 and threads[0] is not threading.main_thread()


async def test_a_settled_integrate_whose_snapshot_cannot_follow_drops_it(tmp_path, monkeypatch):
    package = _installed_package(tmp_path / "t")
    session = tmp_path / "s"
    snapshot_tree(session, package)
    _kernel_keep(package)
    monkeypatch.setattr(ts, "_commit", lambda *_a: "git add rc=128: index.lock exists")

    result = await _settle_integrate(session, monkeypatch, {"status": "kept"})

    assert result["status"] == "kept"
    assert held_snapshot(session, package) is None
    assert restore_tree(session, package) == TreeRestore()
    assert (package / "kernels" / "tuned_moe.py").is_file()


async def test_an_integrate_kept_into_a_dependency_checkout_is_not_the_running_specialists(tmp_path, monkeypatch):
    """integrate_patch commits a KEEP into whichever checkout the patch names, a dependency included."""
    dep = _dependency_checkout(tmp_path / "aiter")

    def _keep_into_the_dependency() -> None:
        (dep / "csrc" / "op.py").write_text("V = 3\n", encoding="utf-8")
        _git(dep, "commit", "-q", "-am", "hyperloom keep")

    async def _integrate_settles_during_the_run(_package: Path) -> None:
        await _settle_integrate(session, monkeypatch, {"status": "kept"}, write=_keep_into_the_dependency)

    runner, ctx, _package, session = _guarded_run(tmp_path, monkeypatch, _integrate_settles_during_the_run, (str(dep),))

    result = await runner.run(ctx)

    assert result.status == "succeeded" and result.error_class == ""
    assert not ctx.extra["shared_state"].stop_reason
