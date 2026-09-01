# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-case configuration coverage and the shipping of agent-created files."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from kernelforge.loop.runner import (
    IterationConfig,
    IterationLoop,
    IterationResult,
)
from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext
from kernelforge.loop import runner as runner_module
from kernelforge.loop.campaign_config import CampaignConfigStore
from kernelforge.loop.campaign_setup import resolve_campaign
from kernelforge.tests.test_campaign_setup import _base_args, _git_workspace
from kernelforge.tests.test_loop_runner import _make_loop, _unused_supervisor


def _coverage_loop(baseline: dict[str, float], unscored: set[str] | None = None):
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            baseline_wall_ms=10.0,
        ),
        tracker=object(),
        config=object(),
        evolver=object(),
    )
    loop._baseline_case_times = dict(baseline)
    loop._unscored_cases = set(unscored or ())
    return loop


def _keep(iteration: int, case_times: dict[str, float]) -> IterationResult:
    return IterationResult(
        iteration=iteration,
        duration_sec=1.0,
        validation_passed=True,
        validation_summary="ok",
        kept=True,
        bench_detail={"case_times": dict(case_times)},
    )


def _keep_with_runs(
    iteration: int,
    case_times: dict[str, float],
    runs: list[dict[str, float]],
) -> IterationResult:
    """A KEEP that also carries the independent measurements it aggregated."""
    return replace(
        _keep(iteration, case_times),
        bench_detail={
            "case_times": dict(case_times),
            "measurements": [{"success": True, "case_times": dict(run)} for run in runs],
        },
    )


def _revert(iteration: int, case_times: dict[str, float]) -> IterationResult:
    return replace(_keep(iteration, case_times), kept=False)


def _context(workspace: Path, case_ids: tuple[str, ...]) -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize test kernel.",
        source_map_path=str(workspace / "kernel.py"),
        cases=tuple(CaseEvidence(case_id=case_id, latency_ms=1.0) for case_id in case_ids),
    )


def test_case_no_keep_ever_moved_is_reported_as_a_fallback():
    loop = _coverage_loop({"decode-t64": 10.0, "prefill-t4096": 10.0})
    loop.results = [_keep(1, {"decode-t64": 10.0, "prefill-t4096": 8.0})]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {"prefill-t4096": 1}
    assert coverage.fallback == ("decode-t64",)
    assert coverage.unmeasured == ()
    assert coverage.keeps == (1,)


def test_a_keep_that_made_a_case_slower_does_not_cover_it():
    loop = _coverage_loop({"decode-t64": 10.0, "prefill-t4096": 10.0})
    loop.results = [_keep(1, {"decode-t64": 10.5, "prefill-t4096": 8.0})]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {"prefill-t4096": 1}
    assert coverage.fallback == ("decode-t64",)


def test_keep_without_case_timings_is_reported_not_dropped():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [_keep(1, {}), _keep(2, {"decode-t64": 10.0})]

    coverage = loop._case_config_coverage()
    rendered = loop._render_case_config_coverage()

    assert coverage.unreadable == (1,)
    assert coverage.keeps == (2,)
    assert "INCOMPLETE RECORD" in rendered
    assert "iteration(s) 1" in rendered
    assert "config_coverage_partial_record" in (loop._case_config_coverage_flags()["decode-t64"])


def test_only_keep_without_case_timings_never_reads_as_no_keep():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [_keep(1, {})]

    rendered = loop._render_case_config_coverage()

    assert "INCOMPLETE RECORD" in rendered
    assert "No KEEP with per-case timings" in rendered
    assert loop._case_config_coverage_flags() == {"decode-t64": ("config_coverage_partial_record",)}


def test_unscored_case_is_outside_the_coverage_ledger():
    loop = _coverage_loop(
        {"decode-t64": 10.0, "correctness-only": 10.0},
        unscored={"correctness-only"},
    )
    loop.results = [_keep(1, {"decode-t64": 10.0, "correctness-only": 10.0})]

    coverage = loop._case_config_coverage()

    assert coverage.fallback == ("decode-t64",)
    assert "correctness-only" not in coverage.fallback
    assert "correctness-only" not in coverage.unmeasured


def test_reverted_iteration_never_credits_a_case_with_a_configuration():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [_revert(1, {"decode-t64": 2.0})]

    coverage = loop._case_config_coverage()

    assert coverage.keeps == ()
    assert coverage.covered == {}


def test_cases_moved_together_by_every_keep_are_reported_undifferentiated():
    loop = _coverage_loop({"t64": 10.0, "t7211": 10.0})
    loop.results = [
        _keep(1, {"t64": 8.0, "t7211": 8.0}),
        _keep(2, {"t64": 6.0, "t7211": 6.0}),
    ]

    coverage = loop._case_config_coverage()

    assert coverage.undifferentiated == (("t64", "t7211"),)
    rendered = loop._render_case_config_coverage()
    assert "one configuration currently serves them all" in rendered.replace("\n", " ")


def test_case_a_later_keep_separated_is_no_longer_undifferentiated():
    loop = _coverage_loop({"t64": 10.0, "t7211": 10.0})
    loop.results = [
        _keep(1, {"t64": 8.0, "t7211": 8.0}),
        _keep(2, {"t64": 6.0, "t7211": 8.0}),
    ]

    coverage = loop._case_config_coverage()

    assert coverage.undifferentiated == ()
    assert coverage.covered == {"t64": 2, "t7211": 1}


def test_case_no_keep_timed_is_reported_unknown_not_as_a_fallback():
    loop = _coverage_loop({"decode-t64": 10.0, "prefill-t4096": 10.0})
    loop.results = [_keep(1, {"prefill-t4096": 8.0})]

    coverage = loop._case_config_coverage()

    assert coverage.unmeasured == ("decode-t64",)
    assert coverage.fallback == ()
    assert "Coverage unknown" in loop._render_case_config_coverage()


def test_ledger_without_a_keep_says_so_instead_of_reading_as_untuned():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [
        replace(
            _keep(1, {"decode-t64": 9.99}),
            kept=False,
            validation_passed=False,
            validation_summary="failed",
        )
    ]

    rendered = loop._render_case_config_coverage()

    assert "No KEEP with per-case timings is on this session's record" in rendered
    assert "INCOMPLETE RECORD" not in rendered
    assert "decode-t64" in rendered
    assert loop._case_config_coverage_flags() == {}


def test_ledger_names_the_session_it_was_read_from():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [_keep(3, {"decode-t64": 8.0})]

    rendered = loop._render_case_config_coverage()

    assert "Read off KEEP iteration(s) 3" in rendered
    assert "A resumed campaign restarts this record" in rendered


def test_coverage_flags_reach_the_planning_context(tmp_path):
    loop = _coverage_loop({"decode-t64": 10.0, "prefill-t4096": 10.0})
    loop.results = [_keep(2, {"decode-t64": 10.0, "prefill-t4096": 8.0})]

    context = loop._with_case_config_coverage(_context(tmp_path, ("decode-t64", "prefill-t4096")))

    flags = {case.case_id: case.flags for case in context.cases}
    assert "config_coverage_fallback" in flags["decode-t64"]
    assert "config_coverage_keep_2" in flags["prefill-t4096"]


def test_planning_context_is_untouched_before_the_first_keep(tmp_path):
    loop = _coverage_loop({"decode-t64": 10.0})
    context = _context(tmp_path, ("decode-t64",))

    assert loop._with_case_config_coverage(context) is context


def _write_new_files(workspace: Path) -> None:
    (workspace / "configs").mkdir()
    (workspace / "configs" / "tuned.json").write_text('{"BLOCK_M": 64}\n')
    (workspace / "notes.txt").write_text("scratch\n")


def test_keep_commits_only_the_allowlisted_new_file(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    _write_new_files(workspace)

    loop._git_commit("keep: tuned configuration")

    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "configs/tuned.json" in tracked
    assert "notes.txt" not in tracked
    assert "notes.txt" in capsys.readouterr().out


def test_revert_removes_exactly_what_a_keep_would_have_committed(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    _write_new_files(workspace)

    admitted, refused = loop._new_paths()
    assert admitted == ["configs/tuned.json"]
    assert refused == ["notes.txt"]

    loop._git_discard_all_tracked_changes()

    assert not (workspace / "configs" / "tuned.json").exists()
    assert (workspace / "notes.txt").exists()
    assert "notes.txt" in capsys.readouterr().out


def test_staged_new_file_does_not_survive_a_revert(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    _write_new_files(workspace)
    subprocess.run(["git", "add", "configs/tuned.json"], cwd=workspace, check=True)

    loop._git_discard_all_tracked_changes()

    assert not (workspace / "configs" / "tuned.json").exists()


def test_new_file_outside_the_allowlist_is_reported_at_both_sites(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _write_new_files(workspace)

    loop._git_discard_all_tracked_changes()
    discard_out = capsys.readouterr().out
    assert "not removed" in discard_out
    assert "configs/tuned.json" in discard_out and "notes.txt" in discard_out

    (workspace / "kernel.py").write_text("def kernel():\n    return 2\n")
    loop._git_commit("keep: tracked edit only")
    commit_out = capsys.readouterr().out
    assert "not committed" in commit_out

    rendered = loop._render_uncommittable_new_paths()
    assert "notes.txt" in rendered
    assert "a KEEP cannot carry them and a REVERT cannot remove them" in rendered
    assert (workspace / "notes.txt").exists()


def _new_file_agent(workspace: Path, name: str):
    async def agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = "ship a tuned configuration"
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text('{"BLOCK_M": 64}\n')
        return "Added a tuned configuration."

    return agent


def test_new_file_only_candidate_is_named_and_taken_off_the_tree(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])

    asyncio.run(
        loop.run(
            agent_fn=_new_file_agent(workspace, "configs/tuned.json"),
            supervisor_fn=_unused_supervisor,
        )
    )

    assert not (workspace / "configs" / "tuned.json").exists()
    summary = loop.results[-1].validation_summary
    assert "configs/*.json" in summary
    assert "taken off the tree" in summary


def test_new_file_only_candidate_outside_the_allowlist_is_still_reported(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(
        loop.run(
            agent_fn=_new_file_agent(workspace, "notes.txt"),
            supervisor_fn=_unused_supervisor,
        )
    )

    assert (workspace / "notes.txt").exists()
    assert "notes.txt" in loop._render_uncommittable_new_paths()


def test_no_new_files_leaves_nothing_to_report(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    (workspace / "kernel.py").write_text("def kernel():\n    return 3\n")

    loop._git_commit("keep: tracked edit only")

    assert "outside commit_new_paths" not in capsys.readouterr().out
    assert loop._render_uncommittable_new_paths() == ""


@pytest.mark.parametrize("protected", ["test_probe.py", "harness_extra.py"])
def test_allowlist_cannot_admit_a_protected_path(tmp_path, monkeypatch, protected):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["*.py"])
    (workspace / protected).write_text("pass\n")

    admitted, refused = loop._new_paths()

    assert admitted == []
    assert refused == [protected]


def test_allowlist_cannot_admit_the_campaign_driver(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["*.py"])
    subprocess.run(["git", "rm", "--cached", "driver.py"], cwd=workspace, check=True)

    admitted, refused = loop._new_paths()

    assert "driver.py" not in admitted
    assert "driver.py" in refused


def test_loop_output_is_not_reported_as_a_file_the_agent_created(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, build_dir="build")
    (workspace / "forge_experiments").mkdir()
    (workspace / "forge_experiments" / "run_state.json").write_text("{}\n")
    (workspace / "build").mkdir()
    (workspace / "build" / "kernel.so").write_text("binary\n")

    assert loop._new_paths() == ([], [])


def test_unlistable_workspace_raises_instead_of_reporting_no_new_files(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    loop.ic = replace(loop.ic, workspace_dir=str(outside))

    with pytest.raises(RuntimeError, match="could not list new files"):
        loop._new_paths()


def test_allowlist_pattern_does_not_admit_a_file_a_directory_deeper(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "configs" / "generated").mkdir(parents=True)
    (workspace / "configs" / "tuned.json").write_text("{}\n")
    (workspace / "configs" / "generated" / "tuned.json").write_text("{}\n")

    admitted, refused = loop._new_paths()

    assert admitted == ["configs/tuned.json"]
    assert refused == ["configs/generated/tuned.json"]


def test_bare_glob_does_not_admit_the_same_name_in_a_subdirectory(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["*.json"])
    (workspace / "nested").mkdir()
    (workspace / "tuned.json").write_text("{}\n")
    (workspace / "nested" / "tuned.json").write_text("{}\n")

    admitted, refused = loop._new_paths()

    assert admitted == ["tuned.json"]
    assert refused == ["nested/tuned.json"]


@pytest.mark.parametrize("pattern", ["configs/**/*.json", "**/tuned.json", "../outside/*.json", "/etc/*"])
def test_allowlist_refuses_a_pattern_it_cannot_honour(pattern):
    with pytest.raises(ValueError):
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            commit_new_paths=[pattern],
        )


def test_new_file_whose_name_contains_a_newline_is_one_path(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "configs").mkdir()
    weird = "configs/two\nlines.json"
    (workspace / weird).write_text("{}\n")

    admitted, refused = loop._new_paths()

    assert admitted == [weird]
    assert refused == []

    loop._git_discard_all_tracked_changes()
    assert not (workspace / weird).exists()


def _unlistable(loop, monkeypatch):
    """Make new-file enumeration fail the way a broken git repository does."""

    def explode():
        raise RuntimeError("could not list new files: git ls-files exited 128")

    monkeypatch.setattr(loop, "_list_untracked", explode)


def test_discard_still_restores_a_workspace_it_cannot_enumerate(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "kernel.py").write_text("def kernel():\n    return 99\n")
    _unlistable(loop, monkeypatch)

    loop._git_discard_all_tracked_changes()

    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"


def test_unenumerable_workspace_does_not_force_a_discard(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    _unlistable(loop, monkeypatch)

    assert loop._new_paths_need_discard() is False


def test_keep_commit_refuses_a_workspace_it_cannot_enumerate(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    (workspace / "kernel.py").write_text("def kernel():\n    return 2\n")
    _unlistable(loop, monkeypatch)

    with pytest.raises(RuntimeError, match="could not list new files"):
        loop._git_commit("keep: tracked edit only")


def test_revert_leaves_an_allowlisted_file_the_candidate_did_not_create(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "configs").mkdir()
    (workspace / "configs" / "operator.json").write_text('{"kept": true}\n')

    asyncio.run(
        loop.run(
            agent_fn=_new_file_agent(workspace, "configs/tuned.json"),
            supervisor_fn=_unused_supervisor,
        )
    )

    assert not (workspace / "configs" / "tuned.json").exists()
    assert (workspace / "configs" / "operator.json").read_text() == '{"kept": true}\n'


def test_move_below_the_floor_is_not_covered_however_quiet_the_runs():
    """The floor, not the spread: 0.88% clears the KEEP gate and stops here.

    Named for what it checks. The move never reaches the dispersion test --
    ``test_move_smaller_than_the_cases_own_measurement_spread_is_not_covered``
    is the one that does.
    """
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 9.912},
            [
                {"decode-t64": 9.4},
                {"decode-t64": 9.912},
                {"decode-t64": 9.95},
            ],
        )
    ]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {}
    assert coverage.fallback == ("decode-t64",)


def test_move_larger_than_the_spread_in_every_run_is_covered():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 8.0},
            [
                {"decode-t64": 7.95},
                {"decode-t64": 8.0},
                {"decode-t64": 8.05},
            ],
        )
    ]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {"decode-t64": 1}


def test_case_slower_in_one_measurement_is_not_covered_by_the_median():
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 9.0},
            [
                {"decode-t64": 8.0},
                {"decode-t64": 9.0},
                {"decode-t64": 10.4},
            ],
        )
    ]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {}
    assert coverage.fallback == ("decode-t64",)


def test_keep_gate_sized_move_alone_no_longer_counts_as_coverage():
    """A record without per-measurement detail is held to the floor ratio."""
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [_keep(1, {"decode-t64": 9.95})]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {}
    assert coverage.fallback == ("decode-t64",)


def test_floor_ratio_is_configurable_and_not_read_off_the_keep_gate():
    from kernelforge.loop.scoring import KEEP_MIN_MARGIN_FRACTION

    assert runner_module.CONFIG_COVERAGE_MIN_MOVE_RATIO != KEEP_MIN_MARGIN_FRACTION
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.ic = replace(loop.ic, config_coverage_min_move_ratio=0.2)
    loop.results = [_keep(1, {"decode-t64": 9.0})]

    assert loop._case_config_coverage().covered == {}


def test_rendered_ledger_states_the_rule_it_applied():
    """The strong wording only where a per-measurement record backs it.

    Was asserting that a KEEP with no ``measurements`` still rendered "every
    independent measurement" -- pinning in place the claim this change exists
    to remove.
    """
    floor_only = _coverage_loop({"decode-t64": 10.0})
    floor_only.results = [_keep(1, {"decode-t64": 8.0})]

    rendered = floor_only._render_case_config_coverage().replace("\n", " ")

    assert floor_only._case_config_coverage().covered == {"decode-t64": 1}
    assert "every independent measurement" not in rendered
    assert "floor alone" in rendered
    assert "run-to-run spread was never tested" in rendered

    measured = _coverage_loop({"decode-t64": 10.0})
    measured.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 8.0},
            [{"decode-t64": 7.95}, {"decode-t64": 8.05}],
        )
    ]

    strong = measured._render_case_config_coverage().replace("\n", " ")

    assert "every independent measurement" in strong
    assert "floor alone" not in strong


def test_forge_loop_option_reaches_the_campaign_and_survives_resume(tmp_path, monkeypatch):
    """The allowlist has to arrive through the real construction path.

    A field only the tests can set is a field production never sets, and an
    empty allowlist admits nothing: no new file could ship, and none would be
    removed by a REVERT either.
    """
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    workspace, kernel, driver = _git_workspace(tmp_path)
    args = _base_args(workspace, kernel, driver)
    args["commit_new_paths"] = ["configs/*.json", " ", "configs/*.json"]

    resolved = resolve_campaign(**args)

    assert resolved.campaign.commit_new_paths == ["configs/*.json"]
    reloaded = CampaignConfigStore(str(workspace)).load()
    assert reloaded.commit_new_paths == ["configs/*.json"]
    assert IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        commit_new_paths=list(reloaded.commit_new_paths),
    ).commit_new_paths == ["configs/*.json"]


def test_campaign_configuration_refuses_a_recursive_allowlist_pattern(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    workspace, kernel, driver = _git_workspace(tmp_path)
    args = _base_args(workspace, kernel, driver)
    args["commit_new_paths"] = ["configs/**/*.json"]

    with pytest.raises(ValueError, match=r"\*\*"):
        resolve_campaign(**args)


def test_forge_loop_offers_the_option_and_prefers_the_campaigns_copy():
    """The click option itself, plus the one hop nothing else covers.

    ``test_forge_loop_option_reaches_the_campaign_and_survives_resume``
    already exercises CLI value -> campaign -> store -> IterationConfig
    behaviourally. What it cannot see is that ``forge_loop`` then OVERWRITES
    the invocation's value with the campaign's, which is what makes the
    allowlist immutable across a resume. Only that assignment is read off the
    source, by pattern rather than by slicing, so reformatting cannot fail it.
    """
    from kernelforge import cli

    option = next(param for param in cli.forge_loop.params if param.name == "commit_new_paths")
    assert "--commit-new-path" in option.opts
    assert option.multiple

    source = inspect.getsource(cli.forge_loop.callback)
    assert re.search(
        r"commit_new_paths\s*=\s*list\(\s*campaign\.commit_new_paths\s*\)",
        source,
    )


def test_move_smaller_than_the_cases_own_measurement_spread_is_not_covered():
    """The dispersion rule itself: clears the floor, faster everywhere, still noise.

    2% is well over the 1% floor and every run is faster than the 10.0 before,
    so the first two conditions pass and only the spread test can reject it.
    Deleting that test (``CONFIG_COVERAGE_DISPERSION_MULTIPLE = 0.0``) makes
    this the only assertion in the file that notices.
    """
    loop = _coverage_loop({"decode-t64": 10.0})
    loop.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 9.8},
            [
                {"decode-t64": 9.4},
                {"decode-t64": 9.8},
                {"decode-t64": 9.95},
            ],
        )
    ]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {}
    assert coverage.fallback == ("decode-t64",)


def test_floor_only_coverage_is_reported_as_such_to_the_planner():
    """A case admitted without a dispersion test is flagged as one."""
    loop = _coverage_loop({"decode-t64": 10.0, "prefill-t8": 10.0})
    loop.results = [
        _keep_with_runs(
            1,
            {"decode-t64": 8.0, "prefill-t8": 8.0},
            [{"decode-t64": 7.95}, {"decode-t64": 8.05}],
        )
    ]

    coverage = loop._case_config_coverage()

    assert coverage.covered == {"decode-t64": 1, "prefill-t8": 1}
    assert coverage.floor_only == ("prefill-t8",)
    flags = loop._case_config_coverage_flags()
    assert "config_coverage_floor_only" in flags["prefill-t8"]
    assert "config_coverage_floor_only" not in flags["decode-t64"]


def test_v6_campaign_config_still_resumes_with_an_empty_allowlist(tmp_path, monkeypatch):
    """A campaign written before the allowlist existed has to keep resuming.

    ``save`` guards on ``load``, so a version this loop refuses to read is a
    campaign with no way back. 6 differs from 7 only by the missing
    ``commit_new_paths``, and missing means empty.
    """
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    workspace, kernel, driver = _git_workspace(tmp_path)
    resolve_campaign(**_base_args(workspace, kernel, driver))

    store = CampaignConfigStore(str(workspace))
    fresh = store.load()
    payload = fresh.to_dict()
    payload["schema_version"] = 6
    del payload["commit_new_paths"]
    store.path.write_text(json.dumps(payload, indent=2))

    reloaded = store.load()

    assert reloaded.commit_new_paths == []
    assert reloaded == fresh
    # Immutability still holds: the in-memory normalization to 7 must not read
    # as a config that changed under the campaign.
    store.save(reloaded)


def test_unreadable_campaign_config_schema_names_what_it_accepts(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    workspace, kernel, driver = _git_workspace(tmp_path)
    resolve_campaign(**_base_args(workspace, kernel, driver))

    store = CampaignConfigStore(str(workspace))
    payload = store.load().to_dict()
    payload["schema_version"] = 5
    store.path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="unsupported campaign config schema 5"):
        store.load()


def test_matching_refuses_a_pattern_normalization_would_have_rejected():
    """Unreachable through the entry points, and a silent skip if it were not."""
    from kernelforge.loop.new_path_allowlist import (
        AllowlistPatternError,
        matches_commit_new_paths,
    )

    with pytest.raises(AllowlistPatternError):
        matches_commit_new_paths("configs/a/b.json", ["configs/**/*.json"])
    with pytest.raises(AllowlistPatternError):
        matches_commit_new_paths("configs/a.json", [" "])


def test_enumeration_failure_is_reported_instead_of_read_as_silence(tmp_path, monkeypatch):
    """A stale or empty refusal list must not read as "nothing to report"."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "stray.txt").write_text("from the previous iteration\n")

    assert loop._new_paths_need_discard() is False
    assert "stray.txt" in loop._render_uncommittable_new_paths()

    _unlistable(loop, monkeypatch)
    assert loop._new_paths_need_discard() is False

    rendered = loop._render_uncommittable_new_paths()
    assert "stray.txt" not in rendered
    assert "could not enumerate" in rendered
    assert "git ls-files exited 128" in rendered


def test_discard_that_cannot_enumerate_also_refreshes_the_report(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "stray.txt").write_text("from the previous iteration\n")
    assert loop._new_paths_need_discard() is False
    assert loop._refused_new_paths == ["stray.txt"]

    _unlistable(loop, monkeypatch)
    loop._git_discard_all_tracked_changes()

    assert loop._refused_new_paths == []
    assert "could not enumerate" in loop._render_uncommittable_new_paths()


def test_retained_allowlisted_file_reaches_the_next_implementer(tmp_path, monkeypatch):
    """Leaving an operator's file behind is a leak, so it is not only printed."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["configs/*.json"])
    (workspace / "configs").mkdir()
    (workspace / "configs" / "operator.json").write_text('{"kept": true}\n')
    loop._pre_untracked = {"configs/operator.json"}

    loop._git_discard_all_tracked_changes()

    assert (workspace / "configs" / "operator.json").exists()
    rendered = loop._render_uncommittable_new_paths()
    assert "configs/operator.json" in rendered
    assert "did not create" in rendered


def test_resume_recovery_does_not_delete_an_operators_new_file(tmp_path, monkeypatch):
    """Resume recovery discards, and it runs before the first iteration.

    A snapshot taken only at the top of the loop leaves that discard with
    none, and the no-snapshot branch used to clean the whole allowlisted set
    -- ``git clean`` on an untracked file the operator put there, with no way
    back.
    """
    from kernelforge.tests.test_campaign_cross_process import (
        _initialize_workspace,
        _make_loop as _campaign_loop,
        _successful_iteration,
    )
    from kernelforge.tracker import ExperimentTracker

    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    (workspace / "configs").mkdir()
    operator = workspace / "configs" / "operator.json"
    operator.write_text('{"kept": true}\n')
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "uncommitted verified candidate"
        path = Path(kernel_path)
        path.write_text(path.read_text() + "\n# uncommitted candidate\n")
        return "uncommitted candidate"

    first = _campaign_loop(workspace, kernel, driver, tracker, session_count=1)
    first.ic = replace(first.ic, commit_new_paths=["configs/*.json"])

    def interrupt_commit(_message):
        raise KeyboardInterrupt

    monkeypatch.setattr(first, "_git_commit", interrupt_commit)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(first.run(agent_fn=editing_agent))

    assert (workspace / "forge_experiments" / "pending_keep.json").is_file()

    resumed = _campaign_loop(workspace, kernel, driver, tracker, session_count=0, resume=True)
    resumed.ic = replace(resumed.ic, commit_new_paths=["configs/*.json"])
    asyncio.run(resumed.run(agent_fn=None))

    assert operator.read_text() == '{"kept": true}\n'
