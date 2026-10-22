"""End-to-end contract of a tree-strategy campaign: seats, branches, the global best and what survives a kill."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop import runner as runner_module
from kernelforge.loop.run_state import BestRecord, LoopStateStore, RunState, make_event
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult, LoopAgents, LoopCallbacks
from kernelforge.loop.search_tree import SearchStrategy, SearchTreeStore, TreeSeat
from kernelforge.orchestrator.start_selector import StartSelection
from kernelforge.tracker import ExperimentTracker

_AMPLE_BUDGET_SEC = 12 * 3600.0


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel():\n    return 0\n")
    (workspace / "driver.py").write_text("pass\n")
    (workspace / ".gitignore").write_text("forge_experiments/\n")
    _git(workspace, "init", "-b", "main")
    _git(workspace, "config", "user.name", "KernelForge Tests")
    _git(workspace, "config", "user.email", "tests@example.com")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "initial")
    return workspace


def _loop(tmp_path, monkeypatch, workspace, *, iterations, resume=False):
    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)
    config = IterationConfig(
        kernel_file=str(workspace / "kernel.py"),
        driver_script=str(workspace / "driver.py"),
        baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        max_time_hours=1.0,
        git_branch="opt",
        workspace_dir=str(workspace),
        lanes=1,
        merge_stacking=False,
        search_strategy=SearchStrategy.TREE,
    )
    loop = IterationLoop(
        config, ExperimentTracker(tmp_path / "experiments"), config=SimpleNamespace(gpu_target="gfx942")
    )
    loop.resume = resume
    monkeypatch.setattr(loop, "_time_remaining", lambda: _AMPLE_BUDGET_SEC if len(loop.results) < iterations else 0.0)
    return loop


class _Script:
    """One scripted campaign: where each iteration starts, and what it measures."""

    def __init__(self, loop, steps):
        self.loop = loop
        self.steps = steps
        self.seen = []

    def _step(self):
        return self.steps[len(self.loop.results)]

    async def select(self, *, workspace, tree_view, resolve, usage=None):
        start = self._step()["start"](self.loop._search_tree)
        if start is None:
            return StartSelection(failure="the answer named no START_COMMIT")
        return StartSelection(commit=resolve(start), reason=f"build on {start[:8]}")

    async def agent(self, kernel_path, _history, session_sink):
        session_sink["plan"] = f"variant {len(self.loop.results) + 1}"
        Path(kernel_path).write_text(f"def kernel():\n    return {len(self.loop.results) + 1}\n")
        return "candidate ready"

    async def measure(self, iteration, plan="", **_kwargs):
        score = self._step()["score"]
        self.seen.append(
            {
                "incumbent": self.loop.best_mean_case_speedup,
                "branch": _git(Path(self.loop.ic.workspace_dir), "branch", "--show-current"),
                "scope": self.loop._history_scope(),
            }
        )
        kept = score > self.loop.best_mean_case_speedup
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=1.0 / score,
            mean_case_speedup=score,
            snr_db=40.0,
            kept=kept,
            bench_detail={"case_times": {"case": 1.0 / score}},
        )


def _root(tree):
    return tree.root.commit


def _node_of(iteration):
    return lambda tree: next(node.commit for node in tree.nodes if node.iteration == iteration)


def _no_answer(_tree):
    return None


_STEPS = [
    {"start": _root, "score": 1.2},  # KEEP: A on opt, global best
    {"start": _root, "score": 1.1},  # KEEP: B on opt-b001, not the best
    {"start": _node_of(2), "score": 1.15},  # KEEP: C on opt-b001, beats B but not A
    {"start": _no_answer, "score": 1.14},  # fallback continues opt-b001 from C; loses to C
]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    loop = _loop(tmp_path, monkeypatch, workspace, iterations=len(_STEPS))
    script = _Script(loop, _STEPS)
    monkeypatch.setattr(loop, "run_one_iteration", script.measure)
    committed = []
    asyncio.run(
        loop.run(
            LoopAgents(agent_fn=script.agent, start_selector=script),
            callbacks=LoopCallbacks(on_best_committed=committed.append),
        )
    )
    return SimpleNamespace(loop=loop, script=script, workspace=workspace, committed=committed)


def test_each_candidate_is_judged_against_the_node_it_started_from(campaign):
    seen = campaign.script.seen

    assert [step["incumbent"] for step in seen] == [1.0, 1.0, 1.1, 1.15]
    assert [step["branch"] for step in seen] == ["opt", "opt-b001", "opt-b001", "opt-b001"]
    assert [result.kept for result in campaign.loop.results] == [True, True, True, False]


def test_the_tree_records_every_node_branch_and_attempt(campaign):
    tree = SearchTreeStore(str(campaign.workspace)).load()
    by_iteration = {node.iteration: node for node in tree.nodes}

    assert sorted(by_iteration) == [0, 1, 2, 3]
    assert by_iteration[2].parent == by_iteration[0].commit
    assert by_iteration[3].parent == by_iteration[2].commit and by_iteration[3].depth == 2
    assert {branch.name: branch.head for branch in tree.branches} == {
        "opt": by_iteration[1].commit,
        "opt-b001": by_iteration[3].commit,
    }
    assert [attempt.decision for attempt in tree.attempts] == ["KEEP", "KEEP", "KEEP", "REVERT_PERF"]
    assert tree.attempts[3].start_commit == by_iteration[3].commit
    assert tree.attempts[3].selector_fallback == "the answer named no START_COMMIT"
    assert tree.seat is None
    assert _git(campaign.workspace, "rev-parse", "opt-b001") == by_iteration[3].commit


def test_a_node_commit_message_carries_its_start_plan_and_measurement(campaign):
    tree = SearchTreeStore(str(campaign.workspace)).load()
    node = next(node for node in tree.nodes if node.iteration == 3)
    message = _git(campaign.workspace, "show", "-s", "--format=%B", node.commit)

    assert message.splitlines()[0] == "iter-3: candidate ready"
    assert f"start: {node.parent}" in message
    assert "plan: variant 3" in message
    assert "mean case speedup: 1.15" in message
    assert "case ms: case=" in message and "snr_db: 40.0" in message


def test_only_a_new_global_best_is_published(campaign):
    tree = SearchTreeStore(str(campaign.workspace)).load()
    best = tree.global_best()
    manifest = json.loads((campaign.workspace / "forge_experiments" / "best" / "manifest.json").read_text())

    assert best.iteration == 1
    assert (manifest["iteration"], manifest["commit_hash"]) == (1, best.commit)
    assert [result.iteration for result in campaign.committed] == [1]


def test_the_campaign_ends_resting_on_the_global_best(campaign):
    loop, workspace = campaign.loop, campaign.workspace
    best = SearchTreeStore(str(workspace)).load().global_best()

    assert _git(workspace, "branch", "--show-current") == "opt"
    assert _git(workspace, "rev-parse", "HEAD") == best.commit
    assert loop.run_state.best.commit_hash == best.commit
    assert loop.best_mean_case_speedup == pytest.approx(1.2)
    assert LoopStateStore(str(workspace)).load().best.commit_hash == best.commit


def test_history_is_scoped_to_the_line_the_iteration_builds_on(campaign):
    scopes = [step["scope"] for step in campaign.script.seen]

    # From root: only attempts made from the root itself. From B: B's own creation; A's KEEP is another branch.
    assert scopes == [frozenset(), frozenset({1}), frozenset({2}), frozenset({2, 3})]


def test_a_kill_between_saving_a_seat_and_checking_out_its_branch_resumes_on_that_seat(campaign, tmp_path, monkeypatch):
    """The tree is saved first, so the resumed session completes the checkout the dead one never made."""
    workspace = campaign.workspace
    store = SearchTreeStore(str(workspace))
    tree = store.load()
    root = tree.root.commit
    branch = tree.allocate_branch(root, 5)
    tree.open_seat(TreeSeat(5, root, branch, "", ""))
    store.save(tree)

    resumed = _loop(tmp_path, monkeypatch, workspace, iterations=0, resume=True)
    state = resumed.validate_resume_preflight()

    assert _git(workspace, "branch", "--show-current") == "opt-b002"
    assert _git(workspace, "rev-parse", "HEAD") == root
    assert state.best.commit_hash == ""
    assert state.head_commit == root


def test_a_seat_whose_iteration_never_finished_is_dropped_on_resume(campaign, tmp_path, monkeypatch):
    workspace = campaign.workspace
    store = SearchTreeStore(str(workspace))
    tree = store.load()
    tree.open_seat(TreeSeat(5, tree.global_best().commit, "opt", "", ""))
    store.save(tree)

    resumed = _loop(tmp_path, monkeypatch, workspace, iterations=0, resume=True)
    resumed._initialize_campaign_runtime(None)
    resumed.state_store = LoopStateStore(str(workspace))
    resumed.run_state = resumed.state_store.load()
    resumed._begin_search_tree()

    reloaded = store.load()
    assert reloaded.seat is None
    assert len(reloaded.attempts) == len(_STEPS)


def test_a_keep_the_tree_never_recorded_is_added_and_published_on_resume(campaign, tmp_path, monkeypatch):
    """A kill after the KEEP checkpoint and before the tree write leaves the node only in git and the run state."""
    workspace = campaign.workspace
    store = SearchTreeStore(str(workspace))
    tree = store.load()
    start = tree.branch("opt-b001").head
    tree.open_seat(TreeSeat(5, start, "opt-b001", "", ""))
    store.save(tree)
    _git(workspace, "checkout", "opt-b001")
    (workspace / "kernel.py").write_text("def kernel():\n    return 5\n")
    _git(workspace, "commit", "-am", "iter-5: faster")
    commit = _git(workspace, "rev-parse", "HEAD")
    state_store = LoopStateStore(str(workspace))
    state = state_store.load()
    state.best = BestRecord(
        iteration=5, wall_ms=0.75, mean_case_speedup=1.3, commit_hash=commit, plan="faster", source="iteration"
    )
    state.best_case_times = {"case": 0.75}
    state_store.save(state)
    state_store.append_event(
        make_event("iteration_result", 5, decision="KEEP", commit_hash=commit, mean_case_speedup=1.3)
    )

    resumed = _loop(tmp_path, monkeypatch, workspace, iterations=0, resume=True)
    resumed._initialize_campaign_runtime(None)
    resumed.state_store = state_store
    resumed.run_state = state_store.load()
    resumed._best_case_times = dict(resumed.run_state.best_case_times)
    resumed._baseline_case_times = {"case": 1.0}
    resumed._begin_search_tree()

    reloaded = store.load()
    assert reloaded.node(commit).parent == start
    assert reloaded.branch("opt-b001").head == commit
    assert reloaded.global_best().commit == commit
    manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    assert (manifest["iteration"], manifest["commit_hash"]) == (5, commit)


def test_a_fresh_tree_is_rooted_at_the_incumbent(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    loop = _loop(tmp_path, monkeypatch, workspace, iterations=0)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.best_mean_case_speedup = 1.0
    loop.best_wall_ms = 1.0
    loop._best_case_times = {"case": 1.0}

    loop._begin_search_tree()

    tree = SearchTreeStore(str(workspace)).load()
    assert tree.root.commit == _git(workspace, "rev-parse", "HEAD")
    assert (tree.root.source, tree.root.mean_case_speedup, tree.current_branch) == ("baseline", 1.0, "opt")
