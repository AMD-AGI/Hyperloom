# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tree campaign end to end on a real repository, with stand-in agents and a stubbed GPU."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop import candidate_measurement as measurement_module
from kernelforge.loop import runner as runner_module
from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    ProfileAvailability,
    ProviderAvailability,
)
from kernelforge.loop.runner import IterationConfig, LoopAgents
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool
from kernelforge.loop.tree.model import AttemptOutcome, NodeSource, Round, Slot, SlotType
from kernelforge.loop.tree.refs import pinned_nodes
from kernelforge.loop.tree.search_loop import TreeSearchLoop
from kernelforge.loop.tree.store import SearchTreeStore
from kernelforge.orchestrator.slot_selector import SelectorAnswer
from kernelforge.tracker import ExperimentTracker

_VALUE = re.compile(r"set VALUE = (\d+)")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("VALUE = 10\n")
    (workspace / "driver.py").write_text("pass\n")
    (workspace / ".gitignore").write_text("forge_experiments/\n")
    _git(workspace, "init", "-q", "-b", "opt")
    _git(workspace, "config", "user.email", "t@t")
    _git(workspace, "config", "user.name", "t")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-qm", "initial")
    return workspace


def _value_at(driver_script: str) -> int:
    """The kernel constant of the checkout whose driver is being run."""
    text = (Path(driver_script).parent / "kernel.py").read_text()
    return int(re.search(r"VALUE = (\d+)", text).group(1))


def _stub_device(monkeypatch, measured_values: list[int]):
    """Time a checkout by its kernel constant: VALUE 10 runs the one case in 1.0 ms."""

    async def bench(run, **_kwargs):
        wall_ms = _value_at(run.driver_script) / 10.0
        measured_values.append(_value_at(run.driver_script))
        runs = [
            {"success": True, "case_times": {"case": wall_ms * factor}, "unscored_cases": []}
            for factor in (1.0, 1.001, 0.999)
        ]
        return {"success": True, "median_ms": wall_ms, "case_times": {"case": wall_ms}, "measurements": runs}

    async def validation(*, driver_script, **_kwargs):
        return SimpleNamespace(
            all_passed=True,
            results=[SimpleNamespace(stage=5, stage_name="full", passed=True, status_label="PASS", snr_db=50.0)],
            summary=lambda: "PASS",
            failed_stage=None,
        )

    async def registers(**_kwargs):
        return {"success": False}

    for module in (runner_module, measurement_module):
        monkeypatch.setattr(module, "measure_wallclock", bench)
        monkeypatch.setattr(module, "force_jit_rebuild", lambda _files: None)
    monkeypatch.setattr(measurement_module, "run_validation_pipeline", validation)
    monkeypatch.setattr(measurement_module, "check_registers", registers)


class _Planning:
    """Two BRANCH plans at the root; one DEEPEN plan per direction later."""

    async def run(self, context, *, usage=None, lanes=1):
        # Planning reads the start's own code: its checkout must exist and be on the start commit.
        workspace = Path(context.workspace)
        assert workspace.is_dir() and _git(workspace, "rev-parse", "HEAD") == context.analysis_commit
        assert Path(context.source_map_path).is_relative_to(workspace)
        brief = context.program_context
        if "BRANCH" in brief:
            plans = (
                "## Direction\n- name: fast table\n- hypothesis: smaller value\n- expected ceiling: 2x\n"
                "- fails if: no gain\n\nset VALUE = 5",
                "## Direction\n- name: structural route\n- hypothesis: pays later\n- expected ceiling: 3x\n"
                "- fails if: never faster\n\nset VALUE = 12",
            )
        elif "DEEPEN direction d002" in brief:
            plans = ("set VALUE = 11",)
        else:
            plans = ("set VALUE = 4",)
        return SimpleNamespace(
            optimization_plans=plans[:lanes], structured_output_diagnostics={"lanes": {"planned": lanes}}
        )


def _lane_factory(_lane_dir, _serialized_driver):
    async def session(kernel_path, prompt):
        # A PORT plan names no constant: carrying the published route's tuning onto this structure lands at 3.
        value = "3" if "# Port plan" in prompt else _VALUE.search(prompt).group(1)
        Path(kernel_path).write_text(f"VALUE = {value}\n")
        return "edited the constant"

    return session


class _Selector:
    """Round 1 branches twice at the root; round 2 deepens both directions from their best nodes."""

    def __init__(self):
        self.loop = None
        self.facts: list[str] = []

    async def select(self, *, workspace, facts, lanes, usage=None):
        self.facts.append(facts)
        tree = self.loop._tree
        if len(tree.nodes) == 1:
            slots = [{"type": "BRANCH", "start": tree.root.commit, "reason": f"open route {i}"} for i in (1, 2)]
        elif len(tree.nodes) == 5:
            slots = [
                {
                    "type": "PORT",
                    "start": tree.direction_best("d002").commit,
                    "source": tree.published_best,
                    "gains": "the constant tuning",
                    "reason": "the structural route lacks the tuning",
                }
            ]
        else:
            slots = [
                {"type": "DEEPEN", "start": tree.direction_best(d).commit, "direction": d, "reason": f"continue {d}"}
                for d in ("d001", "d002")
            ]
        return SelectorAnswer(text=json.dumps({"slots": slots}), payload={"slots": slots})


class _SlowerPlanning(_Planning):
    """Every route it plans is slower than the root."""

    async def run(self, context, *, usage=None, lanes=1):
        plans = ("set VALUE = 12", "set VALUE = 15")
        return SimpleNamespace(
            optimization_plans=plans[:lanes], structured_output_diagnostics={"lanes": {"planned": lanes}}
        )


def _relay_of(agents: LoopAgents) -> LoopAgents:
    """The same agents served as the one profile of a relay."""
    profile = AgentProfile("claude_opus", "claude", "claude-opus-5")
    availability = AgentAvailabilitySnapshot(
        checked_at="2026-10-28T00:00:00Z",
        session_index=1,
        providers={"claude": ProviderAvailability(available=True)},
        profiles={
            "claude_opus": ProfileAvailability(
                provider="claude", model="claude-opus-5", provider_available=True, model_available=True, available=True
            )
        },
    )
    bundle = SearchActorBundle(
        profile=profile,
        agent_fn=lambda *_args, **_kwargs: None,
        agent_factory=agents.agent_factory,
        orchestration_factory=agents.orchestration_factory,
        slot_selector=agents.slot_selector,
    )
    pool = SearchActorPool(
        config=AgentRelayConfig(primary_profile_id="claude_opus", profiles=(profile,)),
        availability=availability,
        bundles={"claude_opus": bundle},
    )
    return replace(agents, search_actors=pool)


def _run_campaign(
    tmp_path,
    monkeypatch,
    *,
    rounds: int,
    workspace: Path | None = None,
    resume: bool = False,
    planning=_Planning,
    relay: bool = False,
):
    workspace = workspace or _workspace(tmp_path)
    measured: list[int] = []
    _stub_device(monkeypatch, measured)
    config = IterationConfig(
        kernel_file=str(workspace / "kernel.py"),
        driver_script=str(workspace / "driver.py"),
        baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        max_time_hours=1.0,
        git_branch="opt",
        workspace_dir=str(workspace),
        lanes=2,
    )
    loop = TreeSearchLoop(
        config,
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(gpu_target="gfx942"),
        resume=resume,
    )
    monkeypatch.setattr(loop, "_time_remaining", lambda: 12 * 3600.0 if len(loop.results) < rounds else 0.0)
    selector = _Selector()
    selector.loop = loop
    agents = LoopAgents(
        agent_factory=_lane_factory,
        orchestration_factory=planning,
        slot_selector=selector,
    )
    asyncio.run(loop.run(_relay_of(agents) if relay else agents))
    return loop, workspace, selector, measured


def test_a_round_keeps_a_slower_route_and_publishes_the_faster_one(tmp_path, monkeypatch):
    loop, workspace, _selector, _measured = _run_campaign(tmp_path, monkeypatch, rounds=1)
    tree = SearchTreeStore(workspace / "forge_experiments").load()

    assert [round(node.score, 3) for node in tree.nodes] == [1.0, 2.0, 0.833]
    assert [item.name for item in tree.directions] == ["fast table", "structural route"]
    fast, slow = tree.nodes[1], tree.nodes[2]
    assert tree.published_best == fast.commit
    assert (workspace / "kernel.py").read_text() == "VALUE = 5\n"
    assert _git(workspace, "rev-parse", "HEAD^{tree}") == _git(workspace, "rev-parse", f"{fast.commit}^{{tree}}")
    assert loop.run_state.best.commit_hash == tree.published_head == _git(workspace, "rev-parse", "HEAD")
    assert {node.commit for node in tree.nodes} <= pinned_nodes(workspace)
    assert _git(workspace, "rev-parse", "opt-d002") == slow.commit
    assert [attempt.outcome for attempt in tree.attempts] == [AttemptOutcome.NODE, AttemptOutcome.NODE]
    assert (workspace / "forge_experiments" / "tree_selection" / "iter_001.md").is_file()


def test_the_next_round_deepens_both_routes_from_their_own_code(tmp_path, monkeypatch):
    loop, workspace, selector, measured = _run_campaign(tmp_path, monkeypatch, rounds=2)
    tree = SearchTreeStore(workspace / "forge_experiments").load()
    by_value = {round(node.score, 3): node for node in tree.nodes}

    assert sorted(by_value) == [0.833, 0.909, 1.0, 2.0, 2.5]
    assert by_value[0.909].parent == by_value[0.833].commit and by_value[0.909].direction == "d002"
    assert by_value[2.5].parent == by_value[2.0].commit
    assert tree.published_best == by_value[2.5].commit
    assert (workspace / "kernel.py").read_text() == "VALUE = 4\n"
    # Round 2 measured 11 in the slow route's checkout and 4 in the fast one's, then 4 again on the campaign branch.
    assert measured[-3:] == [4, 11, 4] or measured[-3:] == [11, 4, 4]
    assert "structural route" in selector.facts[1]
    assert len([event for event in loop.state_store.recent_events(50) if event["type"] == "iteration_result"]) == 2


def test_a_port_carries_gains_onto_another_route_and_can_publish_it(tmp_path, monkeypatch):
    loop, workspace, _selector, _measured = _run_campaign(tmp_path, monkeypatch, rounds=3)
    tree = SearchTreeStore(workspace / "forge_experiments").load()
    ported = tree.nodes[-1]

    assert ported.source is NodeSource.PORT and ported.slot_type is SlotType.PORT
    assert ported.direction == "d002" and tree.node(ported.parent).direction == "d002"
    assert tree.published_best == ported.commit
    plan = next((workspace / "forge_experiments" / "tree_rounds" / "iter_003").glob("start_*/slot_01.md")).read_text()
    assert "# Port plan" in plan and "the constant tuning" in plan
    assert (workspace / "kernel.py").read_text() == "VALUE = 3\n"


def test_a_resume_records_the_round_a_killed_session_left_open_and_carries_on(tmp_path, monkeypatch):
    _loop, workspace, _selector, _measured = _run_campaign(tmp_path, monkeypatch, rounds=1)
    store = SearchTreeStore(workspace / "forge_experiments")
    tree = store.load()
    open_round = Round(iteration=2, slots=(Slot(1, SlotType.BRANCH, tree.published_best, "interrupted branch"),))
    tree.open_round(open_round)
    store.save(tree)

    _run_campaign(tmp_path, monkeypatch, rounds=1, workspace=workspace, resume=True)
    resumed = store.load()

    # The killed iteration's number is the one the resumed session runs next, as in the linear loop.
    second = [attempt for attempt in resumed.attempts if attempt.iteration == 2]
    assert [item.reason for item in second if item.outcome is AttemptOutcome.ERROR] == [
        "interrupted branch [the session ended before this round finished]"
    ]
    assert [item.outcome for item in second if item.outcome is not AttemptOutcome.ERROR] == [AttemptOutcome.NODE] * 2
    assert resumed.round is None
    assert len(resumed.nodes) == 5


def test_a_round_that_publishes_nothing_counts_against_the_profile_that_ran_it(tmp_path, monkeypatch):
    loop, workspace, _selector, _measured = _run_campaign(
        tmp_path, monkeypatch, rounds=1, planning=_SlowerPlanning, relay=True
    )
    handoff = json.loads((workspace / "forge_experiments" / "handoffs" / "iter_001.json").read_text())

    assert SearchTreeStore(workspace / "forge_experiments").load().published_best == loop._tree.root.commit
    assert loop.run_state.agent_relay.active_no_improvement_streak == 1
    assert (handoff["agent_profile_id"], handoff["agent_model"]) == ("claude_opus", "claude-opus-5")


def test_a_workspace_with_untracked_sources_is_refused(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    (workspace / "helper.py").write_text("VALUE = 1\n")

    with pytest.raises(ValueError, match="helper.py"):
        _run_campaign(tmp_path, monkeypatch, rounds=1, workspace=workspace)
