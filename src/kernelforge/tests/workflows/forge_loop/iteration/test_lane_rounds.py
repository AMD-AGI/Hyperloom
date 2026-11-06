# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fan-out rounds: what each lane is handed, what a round pays for only once, and which lane candidates reach the tree."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.iteration import fanout
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.iteration import campaign_runtime
from kernelforge.workflows.forge_loop.iteration import lane_runtime
from kernelforge.workflows.forge_loop.iteration.fanout import LaneResult
from kernelforge.tests.conftest import patch_iteration_name
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _counting_plan,
    _diff_of,
    _lane_agent_factory,
    _make_loop,
    _no_change_agent,
    _reduction_loop,
    _stub_measurement,
    _unused_supervisor,
)


def _recording_lane_factory(seen):
    """A lane agent that records everything the round handed it."""

    def factory(lane_dir, serialized_driver):
        async def agent(kernel_file, prompt, session_sink=None):
            seen.append(
                {
                    "lane_dir": lane_dir,
                    "serialized_driver": serialized_driver,
                    "kernel_file": kernel_file,
                    "prompt": prompt,
                }
            )
            path = Path(lane_dir) / "kernel.py"
            path.write_text(path.read_text().replace("return 1", "return 2"))

        return agent

    return factory


async def test_a_lane_is_told_to_run_the_serialized_driver(tmp_path, monkeypatch):
    """A lane that ran the driver itself would time against its own siblings.

    The lock lives in the wrapper, so the round has to hand it to the factory --
    which installs it as the command the lane's own instructions name -- and the
    plan has to explain the wait it causes, which those instructions cannot know
    about. The plan the archive records stays the direction the lane was
    assigned, without either wrapped around it.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    seen: list[dict] = []

    await loop._fill_lane_queue(
        agent_factory=_recording_lane_factory(seen),
        lane_plans=["tune the prefill epilogue"],
    )

    assert len(seen) == 1
    wrapper = str(Path(seen[0]["lane_dir"]) / fanout.SERIALIZED_DRIVER_NAME)
    assert seen[0]["serialized_driver"] == wrapper
    assert wrapper in seen[0]["prompt"]
    assert "not a hang" in seen[0]["prompt"]
    assert "tune the prefill epilogue" in seen[0]["prompt"]
    assert [item.plan for item in loop._lane_queue] == ["tune the prefill epilogue"]


def test_a_lane_records_its_lesson_while_its_copy_still_exists(tmp_path, monkeypatch):
    """The record turn resumes the real session inside the round.

    Moving it to after ``run_lanes`` returns deletes the lane copy first, and
    the guard then fails the resume before the session can answer.
    """
    from kernelforge.workflows.forge_loop.bootstrap import lanes as cli_lanes
    from kernelforge.config import Config
    from kernelforge.workflows.forge_loop.roles import agent as agent_module
    from kernelforge.tests.agent_backends._claude_fakes import _FakeOptions, _backend

    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2)
    loop.ic = replace(loop.ic, lanes=2)
    _stub_measurement(monkeypatch, [2.0, 2.0])

    def create_backend(_runtime, **_kwargs):
        async def query(prompt, options):
            cwd = Path(options.kwargs["cwd"])
            lane = cwd.name
            if options.kwargs.get("resume"):
                text = f"lane {lane} widened the loads"
            else:
                kernel = cwd / "kernel.py"
                kernel.write_text(kernel.read_text().replace("return 1", "return 2"))
                text = "PLAN: widen the loads\nSUBMIT_CANDIDATE"
            session_id = f"sess-{lane}"
            yield SimpleNamespace(content=[], subtype="init", session_id=session_id)
            yield SimpleNamespace(
                content=[SimpleNamespace(text=text)],
                total_cost_usd=0.0,
                subtype="success",
                num_turns=2,
                session_id=session_id,
            )

        backend = _backend([], {})
        backend._query = query
        backend._options_type = _FakeOptions
        return backend

    monkeypatch.setattr(agent_module, "create_registered_backend", create_backend)
    config = Config(workspace=str(workspace), agent_backend="claude", agent_model="claude-test", agent_precheck=False)
    factory = cli_lanes._make_lane_agent_factory(
        make_agent=agent_module.make_agent_fn,
        config=config,
        binding=cli_lanes.LaneBinding(
            workspace_dir=str(workspace),
            driver=str(workspace / "driver.py"),
            source_files=[str(workspace / "kernel.py")],
        ),
        setup=agent_module.ImplementerSetup(
            agent_module.ImplementerTask(program_md="Optimize the kernel."),
            session=agent_module.SessionOptions(agent_backend="claude"),
        ),
    )

    async def _plan(*, iteration, lanes=1, **_kwargs):
        loop._last_lane_plans = ["tune prefill", "tune decode"]
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("# Optimization plan\ntune prefill\n")
        return plan_path, ""

    monkeypatch.setattr(loop, "_run_orchestration", _plan)
    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=factory,
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    lessons = workspace / "forge_experiments" / "lessons"
    first = (lessons / "iter_001.md").read_text()
    second = (lessons / "iter_002.md").read_text()
    assert first.startswith("lane 1 widened the loads")
    assert second.startswith("lane 2 widened the loads")
    assert "(no agent summary)" not in first + second


async def test_a_fan_out_round_queues_one_candidate_per_lane(tmp_path, monkeypatch):
    """Lanes are spent one per iteration, so each is measured on its own."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(12)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "wide"], cwd=workspace, check=True, capture_output=True)

    await loop._fill_lane_queue(
        agent_factory=_lane_agent_factory(
            {
                "tune prefill": ("line_0 = 0", "line_0 = 100"),
                "tune decode": ("line_11 = 11", "line_11 = 111"),
            }
        ),
        lane_plans=["tune prefill", "tune decode"],
    )

    assert [item.plan for item in loop._lane_queue] == ["tune prefill", "tune decode"]

    first = loop._take_lane_candidate()

    assert first is not None and first.plan == "tune prefill"
    assert "line_0 = 100" in kernel.read_text()
    assert len(loop._lane_queue) == 1


def _published_plan(workspace):
    """Stand in for the planning chain with one durable plan on disk."""

    async def _plan(**_kwargs):
        plan_path = workspace / "forge_experiments" / "orchestration" / "iter_001" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("# Optimization plan\nVectorize loads.\n")
        return plan_path, ""

    return _plan


def test_a_single_lane_runs_the_ordinary_session(tmp_path, monkeypatch):
    """--lanes 1 must behave exactly as it did before lanes existed."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    fanned_out: list[int] = []

    async def _fan_out(*, iteration, **_kwargs):
        fanned_out.append(iteration)

    monkeypatch.setattr(loop, "_fan_out_round", _fan_out)
    monkeypatch.setattr(loop, "_run_orchestration", _published_plan(workspace))

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert loop.ic.lanes == 1
    assert fanned_out == []
    assert decisions == ["NO_CHANGES"]


def test_more_than_one_lane_fans_the_round_out(tmp_path, monkeypatch):
    """Guards the single-lane assertion above from passing for the wrong reason."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    fanned_out: list[int] = []

    async def _fan_out(*, iteration, **_kwargs):
        fanned_out.append(iteration)

    monkeypatch.setattr(loop, "_fan_out_round", _fan_out)
    monkeypatch.setattr(loop, "_run_orchestration", _published_plan(workspace))

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    assert fanned_out == [1]


def _fan_out_iteration(tmp_path, monkeypatch, rounds, **plan_kwargs):
    """One real fan-out iteration, run for what its fallback path costs.

    Every way a fan-out round ends with an empty queue hands the iteration to
    the ordinary single-session path, which plans for itself. Planning is
    dispatch plus every specialist plus synthesis -- the most expensive thing an
    iteration buys -- so what these tests read off ``rounds`` is how many times
    one iteration bought it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    monkeypatch.setattr(loop, "_run_orchestration", _counting_plan(loop, workspace, rounds, **plan_kwargs))

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )
    return [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]


def test_a_planning_outage_is_not_paid_for_twice(tmp_path, monkeypatch):
    """Retrying an outage inside the iteration it stopped cannot clear it.

    The backend that just refused the fan-out round is the backend the fallback
    path would ask, moments later, for the same round -- so the iteration pays
    twice and trips the orchestration circuit breaker twice for one outage.
    """
    rounds: list[int] = []

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds, unavailable=True)

    assert rounds == [2]
    assert decisions == ["ORCHESTRATION_ERROR"]


def test_a_round_that_could_only_plan_one_lane_spends_that_plan(tmp_path, monkeypatch):
    """One plan is not too few to run; it is exactly what one session needs."""
    rounds: list[int] = []

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds, plans=("a",))

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def test_lanes_that_produced_nothing_do_not_buy_the_round_again(tmp_path, monkeypatch):
    """This round has already paid for planning and for every lane session."""
    rounds: list[int] = []

    async def _no_candidates(**_kwargs):
        return []

    monkeypatch.setattr(lane_runtime, "run_lanes", _no_candidates)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def test_candidates_refused_at_intake_do_not_buy_the_round_again(tmp_path, monkeypatch, capsys):
    """The one path where every lane produced something and none of it counts.

    A candidate refused at the boundary has already cost its own session, and
    the round it came from has already cost the planning. Both are spent before
    the refusal is known, so charging the same iteration for a second round
    would answer a candidate that must not be measured by buying another one.
    """
    rounds: list[int] = []
    tampered = "--- a/driver.py\n+++ b/driver.py\n@@ -1 +1 @@\n-pass\n+print('tampered')\n"

    async def _rejected_candidates(**_kwargs):
        return [
            LaneResult(lane_id="1", plan="a", diff=tampered),
            LaneResult(lane_id="2", plan="b", diff=tampered),
        ]

    monkeypatch.setattr(lane_runtime, "run_lanes", _rejected_candidates)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]
    # Pin the path: both candidates were refused, not merely unmeasurable for
    # some other reason that would reach the same iteration count.
    output = capsys.readouterr().out
    assert output.count("candidate rejected") == 2
    assert "driver.py" in output


def test_a_lane_infrastructure_failure_does_not_buy_the_round_again(tmp_path, monkeypatch):
    """The plans survive a workspace that could not be copied; only lanes fail."""
    rounds: list[int] = []

    async def _no_room(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(lane_runtime, "run_lanes", _no_room)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def _lane_plans_available(loop, plans):
    """Stand in for planning, which is not what these tests are about."""

    async def _plan(**_kwargs):
        loop._last_lane_plans = list(plans)
        return "forge_experiments/plan.md", ""

    return _plan


async def test_a_lane_infrastructure_failure_falls_back_to_one_session(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A single iteration's failure must never kill a multi-hour run."""
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    async def _no_room(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(loop, "_run_orchestration", _lane_plans_available(loop, ["a", "b"]))
    monkeypatch.setattr(lane_runtime, "run_lanes", _no_room)

    await loop._fan_out_round(
        iteration=3,
        orchestration_service=None,
        agent_factory=None,
    )

    assert loop._lane_queue == []
    assert "No space left on device" in capsys.readouterr().out


async def test_a_programming_error_in_a_fan_out_round_is_not_swallowed(
    tmp_path,
    monkeypatch,
):
    """Falling back on a lane bug would hide it behind a slower iteration."""
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    async def _wrong_call(**_kwargs):
        raise TypeError("session() got an unexpected keyword argument")

    monkeypatch.setattr(loop, "_run_orchestration", _lane_plans_available(loop, ["a", "b"]))
    monkeypatch.setattr(lane_runtime, "run_lanes", _wrong_call)

    with pytest.raises(TypeError):
        await loop._fan_out_round(
            iteration=3,
            orchestration_service=None,
            agent_factory=None,
        )


async def test_lane_copies_are_made_beside_the_workspace(tmp_path, monkeypatch):
    """/tmp is usually a smaller filesystem than the one the workspace lives on."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    seen: dict = {}

    async def _capture(*, workspace_dir, lanes, session, parent_dir, driver):
        seen["workspace_dir"] = workspace_dir
        seen["parent_dir"] = parent_dir
        seen["driver"] = driver
        return []

    monkeypatch.setattr(lane_runtime, "run_lanes", _capture)

    await loop._fill_lane_queue(agent_factory=None, lane_plans=["a", "b"])

    assert seen["workspace_dir"] == str(workspace)
    assert seen["parent_dir"] == str(Path(workspace).resolve().parent)
    assert seen["driver"] == "driver.py"


async def test_a_lane_that_wrote_nothing_is_never_queued(tmp_path, monkeypatch):
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)

    def factory(_lane_dir, _serialized_driver):
        async def agent(_kernel_file, _prompt, session_sink=None):
            return None

        return agent

    await loop._fill_lane_queue(agent_factory=factory, lane_plans=["a", "b"])

    assert loop._lane_queue == []
    assert loop._take_lane_candidate() is None


def test_a_lane_candidate_that_edits_the_driver_never_reaches_the_tree(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The driver is the measurement boundary; a lane session has no gate at all.

    Lanes run with the in-session gate off, so no protected-path hook is
    installed, and a lane's diff carries every tracked modification it made.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"
    canonical = driver.read_text()
    diff = _diff_of(
        workspace,
        lambda: driver.write_text("import time\nprint('bypass')\n"),
    )
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="1", plan="tamper", diff=diff)]

    assert loop._take_lane_candidate() is None
    assert loop._lane_queue == []
    assert driver.read_text() == canonical
    output = capsys.readouterr().out
    assert "rejected" in output and "driver.py" in output


def test_a_lane_candidate_that_renames_the_driver_away_is_rejected(
    tmp_path,
    monkeypatch,
):
    """Moving the driver aside leaves only the new path in git's numstat."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"

    def _rename() -> None:
        subprocess.run(
            ["git", "mv", "driver.py", "harmless.py"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )

    diff = _diff_of(workspace, _rename)
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="1", plan="move it aside", diff=diff)]

    assert loop._take_lane_candidate() is None
    assert driver.is_file()
    assert not (workspace / "harmless.py").exists()


def test_a_lane_candidate_is_rejected_when_the_driver_stops_being_canonical(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Defence in depth: a bypass the protected-path rule missed still cannot
    reach a measurement, and the tree is returned to canonical before the next
    candidate inherits it."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"
    loop.ic = replace(
        loop.ic,
        canonical_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
    )
    diff = _diff_of(workspace, lambda: driver.write_text("print('bypass')\n"))
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="1", plan="tamper", diff=diff)]
    patch_iteration_name(
        monkeypatch, "FORGE_PROTECTED_PATHS", SimpleNamespace(is_protected=lambda *_args, **_kwargs: False)
    )

    assert loop._take_lane_candidate() is None
    assert loop._validate_driver_integrity(loop.run_state)
    assert "driver integrity check failed" in capsys.readouterr().out


def test_a_tainted_workspace_driver_stops_the_lane_path(tmp_path, monkeypatch):
    """A driver that is not canonical once the candidate is gone measures nothing."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, canonical_driver_sha256=hashlib.sha256(b"other").hexdigest())
    kernel = workspace / "kernel.py"
    diff = _diff_of(workspace, lambda: kernel.write_text("def kernel():\n    return 2\n"))
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="1", plan="tune it", diff=diff)]

    with pytest.raises(ValueError, match="driver integrity"):
        loop._take_lane_candidate()


async def test_a_queued_candidate_that_no_longer_applies_is_dropped(
    tmp_path,
    monkeypatch,
):
    """Its tree moved underneath it; re-deriving it is the Implementer's job."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._lane_queue = [
        campaign_runtime.LaneResult(
            lane_id="1",
            plan="stale",
            diff="--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-absent\n+edited\n",
        )
    ]

    assert loop._take_lane_candidate() is None
    assert loop._lane_queue == []


def test_a_queued_candidate_survives_a_keep_that_only_moved_its_context(
    tmp_path,
    monkeypatch,
):
    """A sibling's KEEP must not discard a session over a textual near-miss.

    A round's lanes are partitioned so that no two edit the same code, but a
    hunk is located by the lines around it, so a KEEP three lines away moves
    the context out from under a candidate that changed nothing it touched.
    The diff names the blobs it was written against and the lane copies share
    the canonical object store, so it is merged against them rather than being
    dropped for a mismatch that is not a disagreement.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    a = 1\n    b = 2\n    c = 3\n    y = 4\n    d = 5\n    e = 6\n    return y\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    candidate = _diff_of(
        workspace,
        lambda: kernel.write_text(kernel.read_text().replace("y = 4", "y = 44")),
    )
    kernel.write_text(kernel.read_text().replace("a = 1", "a = 11"))
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "the sibling lane's KEEP"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="2", plan="tune y", diff=candidate)]

    taken = loop._take_lane_candidate()

    assert taken is not None and taken.lane_id == "2"
    assert "y = 44" in kernel.read_text()
    assert "a = 11" in kernel.read_text()


def test_a_queued_candidate_that_edits_the_same_lines_is_still_dropped(
    tmp_path,
    monkeypatch,
):
    """Merging against the recorded blobs must not become a way to guess.

    Two edits to the same line are a disagreement, not a moved context, and
    resolving one would measure a tree no plan describes.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    y = 4\n    return y\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "one line to fight over"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    candidate = _diff_of(
        workspace,
        lambda: kernel.write_text(kernel.read_text().replace("y = 4", "y = 44")),
    )
    kernel.write_text(kernel.read_text().replace("y = 4", "y = 55"))
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "a KEEP on the same line"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    loop._lane_queue = [campaign_runtime.LaneResult(lane_id="2", plan="tune y", diff=candidate)]

    assert loop._take_lane_candidate() is None
    assert kernel.read_text() == "def kernel():\n    y = 55\n    return y\n"
