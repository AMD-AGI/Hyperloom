"""End-to-end contract of the seqany search policy inside the forge loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

import kernelforge.loop.runner as runner_module
import kernelforge.orchestrator.agent as agent_module
from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.loop.archive import CandidateArchive
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult, _decision_label
from kernelforge.loop.search_policy import SearchPolicy
from kernelforge.tests.test_loop_runner import _faster_bench, _git_workspace, _measurement_loop
from kernelforge.tracker import ExperimentTracker

_AMPLE_BUDGET_SEC = 12 * 3600.0

# Scores against the pristine anchor, whose single case takes 1.0 ms.
_SLOWER = 0.9
_FASTER = 1.25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "accepted", "label"),
    [(SearchPolicy.SEQANY, True, "ACCEPT"), (SearchPolicy.SEQUENTIAL, False, "REVERT_PERF")],
)
async def test_a_valid_candidate_that_misses_the_bar_is_accepted_only_under_seqany(
    monkeypatch, policy, accepted, label
):
    loop, _calls = _measurement_loop(monkeypatch, _faster_bench(candidate_ms=2.0))
    loop.ic.search_policy = policy

    result = await loop.run_one_iteration(1)

    assert result.validation_passed is True
    assert result.kept is False
    assert result.accepted is accepted
    assert _decision_label(result) == label


@pytest.mark.asyncio
async def test_seqany_does_not_accept_a_candidate_whose_benchmark_failed(monkeypatch):
    loop, _calls = _measurement_loop(monkeypatch, {**_faster_bench(candidate_ms=2.0), "success": False})
    loop.ic.search_policy = SearchPolicy.SEQANY

    result = await loop.run_one_iteration(1)

    assert (result.kept, result.accepted) == (False, False)
    assert _decision_label(result) == "REVERT_PERF"


@pytest.mark.asyncio
async def test_seqany_runs_the_assembly_suite_before_accepting_a_slower_candidate(tmp_path, monkeypatch):
    """Every committed candidate passes the backend's correctness suite, not only a new best."""
    from kernelforge.tests.test_numerical_contract import _task, evidence

    _task(tmp_path, evidence())
    tmp_path.joinpath("driver.py").write_text("raise AssertionError('normalized max err 0.02468 too high')\n")
    loop, _calls = _measurement_loop(
        monkeypatch, _faster_bench(candidate_ms=2.0), workspace_dir=_git_workspace(tmp_path)
    )
    loop.ic.kernel_backend = "assembly"
    loop.ic.pristine_baseline_wall_ms = 1.0
    loop.ic.search_policy = SearchPolicy.SEQANY

    result = await loop.run_one_iteration(1)

    assert result.validation_passed is False
    assert result.validation_outcome == "canonical_correctness_failure"
    assert result.accepted is False


def _workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver.write_text("pass\n")
    for command in (
        ["git", "init", "-b", "campaign-test"],
        ["git", "config", "user.name", "KernelForge Tests"],
        ["git", "config", "user.email", "tests@example.com"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)
    return workspace, kernel, driver


def _head(workspace) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _seqany_loop(workspace, kernel, driver, *, iterations, resume=False, loop_config=None):
    config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        max_time_hours=1.0,
        git_branch="campaign-test",
        workspace_dir=str(workspace),
        search_policy=SearchPolicy.SEQANY,
        lanes=1,
    )
    loop = IterationLoop(
        config,
        ExperimentTracker(workspace / "forge_experiments"),
        config=loop_config if loop_config is not None else object(),
        resume=resume,
    )
    loop._time_remaining = lambda: _AMPLE_BUDGET_SEC if len(loop.results) < iterations else 0.0
    return loop


def _scripted(outcomes: dict[int, str]):
    """A measured iteration whose verdict is scripted per iteration: KEEP, ACCEPT or REVERT_VALIDATION."""

    async def run_one_iteration(self, iteration, plan="", **_kwargs):
        outcome = outcomes[iteration]
        if outcome == "REVERT_VALIDATION":
            return IterationResult(
                iteration=iteration, duration_sec=0.01, validation_passed=False, validation_summary="FAIL"
            )
        score = _FASTER if outcome == "KEEP" else _SLOWER
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.0 / score,
            mean_case_speedup=score,
            snr_db=40.0,
            kept=outcome == "KEEP",
            accepted=outcome == "ACCEPT",
            bench_detail={"median_ms": 1.0 / score, "case_times": {"case": 1.0 / score}},
        )

    return run_one_iteration


async def _editing_agent(kernel_path, _history, session_sink):
    path = runner_module.Path(kernel_path)
    edits = path.read_text().count("# edit") + 1
    session_sink["plan"] = f"edit {edits}"
    path.write_text(path.read_text() + f"# edit {edits}\n")
    return f"edit {edits}"


def test_seqany_builds_on_an_accepted_candidate_and_publishes_only_the_best(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    base = _head(workspace)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT", 2: "KEEP", 3: "ACCEPT"}))

    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=3).run(agent_fn=_editing_agent))

    state = LoopStateStore(str(workspace)).load()
    first, second, third = state.candidates
    assert [record.decision for record in state.candidates] == ["ACCEPT", "KEEP", "ACCEPT"]
    assert (first.parent_iteration, first.parent_commit) == (0, base)
    assert (second.parent_iteration, second.parent_commit) == (1, first.commit_hash)
    assert (third.parent_iteration, third.parent_commit) == (2, second.commit_hash)
    assert _head(workspace) == third.commit_hash
    assert state.best.iteration == 2
    assert state.best.commit_hash == second.commit_hash
    assert (state.cumulative.kept, state.cumulative.accepted, state.cumulative.reverted) == (1, 2, 0)
    manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    assert (manifest["iteration"], manifest["commit_hash"]) == (2, second.commit_hash)
    events = [e for e in LoopStateStore(str(workspace)).read_events() if e["type"] == "iteration_result"]
    assert [(e["decision"], e["accepted"], e["is_new_best"]) for e in events] == [
        ("ACCEPT", True, False),
        ("KEEP", False, True),
        ("ACCEPT", True, False),
    ]
    assert [e["parent_commit"] for e in events] == [base, first.commit_hash, second.commit_hash]
    meta = CandidateArchive(str(workspace), str(kernel)).load_meta(3)
    assert (meta["decision"], meta["accepted"], meta["parent_iteration"]) == ("ACCEPT", True, 2)


def test_seqany_reverts_back_to_the_accepted_candidate_not_to_the_best(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    base = _head(workspace)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT", 2: "REVERT_VALIDATION"}))

    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=2).run(agent_fn=_editing_agent))

    state = LoopStateStore(str(workspace)).load()
    accepted, reverted = state.candidates
    assert _head(workspace) == accepted.commit_hash
    assert subprocess.run(["git", "diff", "HEAD"], cwd=workspace, capture_output=True, text=True).stdout == ""
    assert (reverted.decision, reverted.commit_hash, reverted.parent_commit) == (
        "REVERT_VALIDATION",
        "",
        accepted.commit_hash,
    )
    assert state.best.commit_hash == ""
    assert state.start_commit == base
    assert not (workspace / "forge_experiments" / "best" / "manifest.json").exists()


def test_seqany_resumes_from_the_accepted_candidate_while_the_best_lies_behind_it(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "KEEP", 2: "ACCEPT", 3: "KEEP"}))
    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=2).run(agent_fn=_editing_agent))
    accepted_head = _head(workspace)

    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=1, resume=True).run(agent_fn=_editing_agent))

    state = LoopStateStore(str(workspace)).load()
    assert [record.decision for record in state.candidates] == ["KEEP", "ACCEPT", "KEEP"]
    assert state.candidates[2].parent_commit == accepted_head
    assert state.best.iteration == 3
    assert _head(workspace) == state.best.commit_hash


def test_seqany_resume_refuses_a_head_that_is_not_the_starting_version(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "KEEP", 2: "ACCEPT"}))
    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=2).run(agent_fn=_editing_agent))
    best = LoopStateStore(str(workspace)).load().best.commit_hash
    subprocess.run(["git", "reset", "--hard", best], cwd=workspace, check=True, capture_output=True)

    with pytest.raises(ValueError, match="HEAD mismatch"):
        asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=1, resume=True).run(agent_fn=_editing_agent))


def test_an_interrupted_accept_is_finished_on_resume_without_publishing_a_best(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT"}))
    first = _seqany_loop(workspace, kernel, driver, iterations=1)

    def interrupt_before_checkpoint(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(first, "_finalize_commit_checkpoint", interrupt_before_checkpoint)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(first.run(agent_fn=_editing_agent))
    root = workspace / "forge_experiments"
    pending = json.loads((root / "pending_keep.json").read_text())
    assert pending["promotes_best"] is False
    committed = _head(workspace)

    async def forbidden_agent(*_args, **_kwargs):
        raise AssertionError("resume started an agent before reconciliation")

    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=0, resume=True).run(agent_fn=forbidden_agent))

    state = LoopStateStore(str(workspace)).load()
    assert [(record.decision, record.commit_hash) for record in state.candidates] == [("ACCEPT", committed)]
    assert state.best.commit_hash == ""
    assert state.cumulative.accepted == 1
    assert not (root / "pending_keep.json").exists()
    assert not (root / "best" / "manifest.json").exists()
    assert CandidateArchive(str(workspace), str(kernel)).load_meta(1)["decision"] == "ACCEPT"


def test_the_session_is_told_its_starting_score_only_when_it_is_not_the_best(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT", 2: "KEEP", 3: "ACCEPT"}))
    sessions: list[tuple[float | None, str]] = []

    async def agent(kernel_path, history, session_sink, best_mean_case_speedup=None, starting_mean_case_speedup=None):
        sessions.append((starting_mean_case_speedup, history))
        return await _editing_agent(kernel_path, history, session_sink)

    asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=3).run(agent_fn=agent))

    starting_scores = [score for score, _history in sessions]
    assert starting_scores == [None, _SLOWER, None]
    assert "Starting version: iter 1 (accepted, not the best)" in sessions[1][1]
    assert "Starting version" not in sessions[2][1]


@pytest.mark.parametrize("starting", [None, 0.9])
def test_the_implementer_scoring_prompt_names_a_starting_score_only_when_given(tmp_path, monkeypatch, starting):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver = tmp_path / "driver.py"
    driver.write_text("raise AssertionError('prompt tests must not execute the driver')\n")
    specs = []

    class RecordingBackend:
        name = "claude"
        capabilities = AgentCapabilities(stop_hooks=True)

        def __init__(self, runtime):
            self.runtime = runtime

        async def run(self, spec, usage=None):
            specs.append(spec)
            return AgentRunResult(text="PLAN: inspect")

    monkeypatch.setattr(agent_module, "create_registered_backend", lambda runtime, **_kw: RecordingBackend(runtime))
    agent_fn = agent_module.make_agent_fn(
        config=Config(gpu_target="gfx950", workspace=str(tmp_path), agent_backend="claude", agent_precheck=False),
        program_md="Optimize the kernel.",
        kernel_backend_name="triton",
        insession_gate=True,
        driver_script=str(driver),
    )

    asyncio.run(
        agent_fn(
            str(kernel),
            "",
            baseline_case_times={"case": 1.0},
            best_mean_case_speedup=1.2,
            starting_mean_case_speedup=starting,
        )
    )

    prompt = " ".join(specs[0].system_prompt.split())
    assert "Current best pristine-relative score: 1.2." in prompt
    if starting is None:
        assert "starts from an accepted version" not in prompt
    else:
        assert "This session starts from an accepted version that is not the best; its score is 0.9." in prompt


def test_planning_evidence_describes_the_accepted_starting_version(tmp_path, monkeypatch):
    """Case timings, current score and commit handed to planning are the kernel the next session edits."""
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT"}))
    loop = _seqany_loop(workspace, kernel, driver, iterations=1, loop_config=SimpleNamespace(gpu_target="gfx942"))
    asyncio.run(loop.run(agent_fn=_editing_agent))

    context = loop._build_orchestration_context()

    assert context.canonical_commit == _head(workspace)
    assert context.current_mean_case_speedup == pytest.approx(_SLOWER)
    assert [case.latency_ms for case in context.cases] == [pytest.approx(1.0 / _SLOWER)]
    assert loop.best_mean_case_speedup == pytest.approx(1.0)


def test_a_head_that_moved_off_the_starting_version_stops_the_campaign(tmp_path, monkeypatch):
    workspace, kernel, driver = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _scripted({1: "ACCEPT", 2: "ACCEPT"}))

    async def committing_agent(kernel_path, history, session_sink):
        await _editing_agent(kernel_path, history, session_sink)
        subprocess.run(["git", "commit", "-qam", "agent commit"], cwd=workspace, check=True)
        return "committed by the agent"

    with pytest.raises(RuntimeError, match="is not the seqany starting version"):
        asyncio.run(_seqany_loop(workspace, kernel, driver, iterations=2).run(agent_fn=committing_agent))
