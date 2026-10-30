# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which persistence and git failures the loop degrades on, and which it lets propagate."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.utils.git import GitError
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.search.supervisor import SupervisionMonitor
from kernelforge.workflows.forge_loop.iteration import campaign_runtime
from kernelforge.workflows.forge_loop.iteration import locked_run
from kernelforge.tests.conftest import patch_iteration_name
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _lesson_result,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def test_a_snapshot_write_failure_records_degradation_and_continues(
    tmp_path,
    monkeypatch,
    capsys,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.pr_kb_snapshot = {"entries": {"ROCm/aiter#1@sha:1": {}}}
    loop.ic.pr_kb_event = {"position": "A", "reason": "ok"}

    def fail_snapshot(_workspace_dir, _snapshot):
        """Simulate a local sidecar write failure after the guard."""
        raise OSError("disk full")

    monkeypatch.setattr(
        "kernelforge.knowledge.pr_knowledge.references.commit_snapshot",
        fail_snapshot,
    )

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    events = [event for event in LoopStateStore(str(workspace)).read_events() if event["type"] == "pr_refs_refreshed"]
    assert len(events) == 1
    assert events[0]["reason"] == "ok"
    assert events[0]["degraded_reason"] == "local_failure"
    assert "warning: snapshot not persisted" in capsys.readouterr().out


def test_a_pr_event_write_failure_cannot_abort_the_campaign(
    tmp_path,
    monkeypatch,
    capsys,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.pr_kb_event = {"position": "A", "reason": "ok"}
    append_event = LoopStateStore.append_event

    def fail_pr_event(self, event):
        """Fail only the optional PR event and retain normal loop persistence."""
        if event.get("type") == "pr_refs_refreshed":
            raise OSError("events unavailable")
        append_event(self, event)

    monkeypatch.setattr(LoopStateStore, "append_event", fail_pr_event)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    assert "warning: event not recorded" in capsys.readouterr().out


def _raise_os_error(message):
    def _raise(*_args, **_kwargs):
        raise OSError(message)

    return _raise


def test_close_round_failure_propagates(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop._open_round(1, lanes=1)
    loop._round_planning_sec = 10.0
    monkeypatch.setattr(loop.state_store, "append_event", _raise_os_error("round event unavailable"))

    with pytest.raises(OSError, match="^round event unavailable$"):
        loop._close_round()

    assert loop._round_started_at is None
    assert loop._round_planning_sec == 0.0


def test_baseline_case_times_save_failure_propagates(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = SimpleNamespace(save=_raise_os_error("baseline save unavailable"))
    loop.run_state = RunState()
    loop._baseline_case_times = {}

    with pytest.raises(OSError, match="^baseline save unavailable$"):
        loop._set_baseline_case_times({"case": 2.0})

    assert loop._baseline_case_times == {"case": 2.0}
    assert loop.persistence_degraded is False
    assert loop.persistence_errors == []


def test_scoring_state_save_failure_propagates(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = SimpleNamespace(save=_raise_os_error("scoring save unavailable"))
    loop.run_state = RunState()

    with pytest.raises(OSError, match="^scoring save unavailable$"):
        loop._persist_scoring_state()

    assert loop.persistence_degraded is False
    assert loop.persistence_errors == []


def test_search_policy_persistence_failure_propagates(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    monkeypatch.setattr(loop.state_store, "append_event", _raise_os_error("policy event unavailable"))

    with pytest.raises(OSError, match="^policy event unavailable$"):
        loop._update_search_policy(1)


def _supervisor_loop(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.monitor = SupervisionMonitor()
    monkeypatch.setattr(loop, "_build_supervisor_evidence_context", lambda _iteration: "{}")

    async def supervisor(**_kwargs):
        return "Change the route."

    return loop, supervisor


def test_supervisor_ruling_event_failure_propagates(tmp_path, monkeypatch):
    loop, supervisor = _supervisor_loop(tmp_path, monkeypatch)
    real_append = loop.state_store.append_event

    def append_event(event):
        if event["type"] == "supervisor_ruling":
            raise OSError("ruling event unavailable")
        real_append(event)

    monkeypatch.setattr(loop.state_store, "append_event", append_event)

    with pytest.raises(OSError, match="^ruling event unavailable$"):
        asyncio.run(
            loop._run_supervisor_review(
                supervisor,
                iteration=3,
                digest="digest",
                supervisor_due=True,
                supervisor_reason="stalled",
            )
        )


def test_supervisor_intervention_save_failure_propagates(tmp_path, monkeypatch):
    loop, supervisor = _supervisor_loop(tmp_path, monkeypatch)
    real_save = loop.state_store.save
    saves = []

    def save(state):
        saves.append(1)
        if len(saves) > 1:
            raise OSError("intervention save unavailable")
        real_save(state)

    monkeypatch.setattr(loop.state_store, "save", save)

    with pytest.raises(OSError, match="^intervention save unavailable$"):
        asyncio.run(
            loop._run_supervisor_review(
                supervisor,
                iteration=3,
                digest="digest",
                supervisor_due=True,
                supervisor_reason="stalled",
            )
        )


def test_seed_and_hydrate_failure_propagates_on_resume(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    monkeypatch.setattr(loop, "_restore_scoring_state", _raise_os_error("scoring restore unavailable"))

    with pytest.raises(OSError, match="^scoring restore unavailable$"):
        loop._seed_and_hydrate_run_state()


_GIT_FAILURES = [
    GitError(128, ["git", "diff"], "", "fatal: bad object"),
    subprocess.TimeoutExpired(["git", "diff"], 300.0),
    OSError("git is not installed"),
]
_DISCARD_FAILURES = [RuntimeError("git clean failed: Permission denied"), *_GIT_FAILURES]


def _failure_id(failure: BaseException) -> str:
    return type(failure).__name__


def _raiser(failure: BaseException):
    def fail(*_args, **_kwargs):
        raise failure

    return fail


def _kept_result() -> IterationResult:
    return _lesson_result(iteration=1, kept=True, commit_hash="c0ffee", mean_case_speedup=1.1)


def _bare_loop(tmp_path, monkeypatch, *, running: bool = False):
    """A loop with the durable state ``run`` would have opened, before any iteration."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.best_publisher = campaign_runtime.BestResultPublisher(str(workspace))
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = loop.state_store.load()
    loop.experiment = None
    if running:
        locked_run.start_session(loop.run_state)
    return loop


def _site_full_diff(tmp_path, monkeypatch, failure):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    patch_iteration_name(monkeypatch, "git", _raiser(failure))
    return loop._full_diff("c0ffee")


def _site_publish_best(tmp_path, monkeypatch, failure):
    loop = _bare_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_publication_changed_files", lambda _commit: [])
    monkeypatch.setattr(loop, "_publication_patch", lambda _commit: "")
    monkeypatch.setattr(loop.best_publisher, "publish", _raiser(failure))
    published = loop._publish_best_result(_kept_result(), plan="p", best_before=None)
    return loop, published


def _site_archive_rebuild(tmp_path, monkeypatch, failure):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop._recovered_pending_keep = ({"plan": "recovered"}, _kept_result())
    steps: list[str] = []
    monkeypatch.setattr(loop, "_archive_pending_keep", _raiser(failure))
    monkeypatch.setattr(loop, "_publish_best_result", lambda *_a, **_k: steps.append("publish"))
    monkeypatch.setattr(loop, "_clear_pending_keep", lambda: steps.append("clear"))
    monkeypatch.setattr(loop, "_publish_optimization_history", lambda: steps.append("history"))
    monkeypatch.setattr(loop, "_checkpoint_llm_usage", lambda: steps.append("usage"))
    asyncio.run(loop._finish_recovered_pending_keep())
    return loop, steps


def _site_terminal_stamp(tmp_path, monkeypatch, failure):
    loop = _bare_loop(tmp_path, monkeypatch, running=True)
    steps: list[str] = []
    monkeypatch.setattr(loop, "_git", _raiser(failure))
    monkeypatch.setattr(loop, "_checkpoint_llm_usage", lambda: steps.append("usage"))
    loop._persist_terminal_state()
    return steps


def _fail_first_discard(loop, monkeypatch, failure: BaseException) -> None:
    """Make the session-recovery discard fail once; the verdict's own discard runs for real."""
    pending = [failure]
    real_discard = loop._git_discard_worktree

    def discard():
        if pending:
            raise pending.pop()
        real_discard()

    monkeypatch.setattr(loop, "_git_discard_worktree", discard)


def _run_verdict(tmp_path, monkeypatch, failure, **sink_fields):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    _fail_first_discard(loop, monkeypatch, failure)

    async def agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 99\n")
        session_sink.update(plan="candidate", end_reason="candidate_submitted", **sink_fields)
        return "candidate"

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent, supervisor_fn=_unused_supervisor)))
    return loop


def _site_integrity_verdict(tmp_path, monkeypatch, failure):
    return _run_verdict(
        tmp_path,
        monkeypatch,
        failure,
        integrity_violation=True,
        integrity_reason="modified oracle",
        integrity_restore=lambda: None,
    )


def _site_contention_verdict(tmp_path, monkeypatch, failure):
    return _run_verdict(tmp_path, monkeypatch, failure, workspace_contention="pid(s) [4321] hold a device node")


def _site_keep_commit(tmp_path, monkeypatch, failure):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "keep that cannot be committed"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "verified improvement"

    async def successful_iteration(iteration, plan=""):
        return replace(_kept_result(), iteration=iteration, commit_hash="", duration_sec=0.01)

    monkeypatch.setattr(loop, "run_one_iteration", successful_iteration)
    monkeypatch.setattr(loop, "_git_commit", _raiser(failure))
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=editing_agent, supervisor_fn=_unused_supervisor)))
    return loop, workspace


@pytest.mark.parametrize(
    "site",
    [
        _site_full_diff,
        _site_publish_best,
        _site_archive_rebuild,
        _site_terminal_stamp,
        _site_integrity_verdict,
        _site_contention_verdict,
        _site_keep_commit,
    ],
    ids=lambda site: site.__name__,
)
def test_an_unexpected_error_at_a_narrowed_handler_propagates(tmp_path, monkeypatch, site):
    with pytest.raises(TypeError, match="^seam drifted$"):
        site(tmp_path, monkeypatch, TypeError("seam drifted"))


@pytest.mark.parametrize("failure", _GIT_FAILURES, ids=_failure_id)
def test_a_failed_commit_diff_archives_an_empty_change_diff(tmp_path, monkeypatch, failure):
    assert _site_full_diff(tmp_path, monkeypatch, failure) == ""


@pytest.mark.parametrize(
    "failure",
    [OSError("disk full"), ValueError("best manifest is ahead of iteration 1: 2"), *_GIT_FAILURES[:2]],
    ids=_failure_id,
)
def test_a_failed_best_publication_degrades_persistence_instead_of_raising(tmp_path, monkeypatch, failure):
    loop, published = _site_publish_best(tmp_path, monkeypatch, failure)

    assert published is False
    assert (loop.persistence_degraded, loop.persistence_errors) == (True, [f"publish best iteration 1: {failure}"])


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("candidate archive conflicts with pending KEEP iteration 1"),
        RuntimeError("failed to recover candidate archive for iteration 1"),
        OSError("archive index not writable"),
    ],
    ids=_failure_id,
)
def test_a_failed_archive_rebuild_still_publishes_the_recovered_best(tmp_path, monkeypatch, failure):
    loop, steps = _site_archive_rebuild(tmp_path, monkeypatch, failure)

    assert (loop.persistence_degraded, loop.persistence_errors) == (
        True,
        [f"rebuild candidate archive iteration 1: {failure}"],
    )
    assert steps == ["publish", "clear", "history", "usage"]


@pytest.mark.parametrize("failure", _GIT_FAILURES[1:], ids=_failure_id)
def test_a_failed_terminal_stamp_does_not_skip_the_usage_checkpoint(tmp_path, monkeypatch, failure):
    assert _site_terminal_stamp(tmp_path, monkeypatch, failure) == ["usage"]


@pytest.mark.parametrize("failure", _DISCARD_FAILURES, ids=_failure_id)
@pytest.mark.parametrize(
    ("site", "summary"),
    [
        (
            _site_integrity_verdict,
            "REVERT (protected integrity violation): canonical correctness and benchmark were skipped before "
            "executing the measurement surface. modified oracle Restoration errors: tracked candidate restore failed: ",
        ),
        (
            _site_contention_verdict,
            "REVERT (workspace contention): canonical correctness and benchmark were skipped because the session's "
            "workspace could not be cleared. pid(s) [4321] hold a device node Candidate restore failed: ",
        ),
    ],
    ids=["integrity", "contention"],
)
def test_a_failed_restore_is_reported_on_the_verdict(tmp_path, monkeypatch, site, summary, failure):
    loop = site(tmp_path, monkeypatch, failure)

    assert loop.results[0].validation_summary == f"{summary}{type(failure).__name__}: {failure}"


@pytest.mark.parametrize(
    "failure",
    [
        OSError("pending journal not writable"),
        ValueError("verified KEEP has no candidate diff"),
        RuntimeError("git commit did not advance HEAD"),
        GitError(1, ["git", "commit"], "", "hook rejected commit"),
    ],
    ids=_failure_id,
)
def test_a_keep_that_cannot_be_committed_is_reverted_as_a_crash(tmp_path, monkeypatch, failure):
    loop, workspace = _site_keep_commit(tmp_path, monkeypatch, failure)

    result = loop.results[0]
    assert (result.kept, result.validation_passed, result.crashed) == (False, False, True)
    assert result.validation_summary == f"COMMIT FAILED: {failure}"
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    assert not (workspace / "forge_experiments" / "pending_keep.json").exists()
