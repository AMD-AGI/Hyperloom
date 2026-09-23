# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""How the ceiling reaches a campaign, and the one place it must never reach."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from kernelforge import cli as cli_module
from kernelforge.roofline_ceiling.contract import load_report
from kernelforge.roofline_ceiling.estimate import estimate_ceiling as real_estimate_ceiling
from kernelforge.roofline_ceiling.report import REPORT_FILENAME, WORKSPACE_SUBDIR
from kernelforge.loop.run_state import RunState
from kernelforge.loop.runner import IterationConfig, IterationLoop


def _report(cases=(("decode-t1", 12.8),)):
    return load_report({"cases": {case_id: ideal for case_id, ideal in cases}})


def _publish(report, directory) -> Path:
    """Write what the analyst would have written, so a consumer can read it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_FILENAME
    path.write_text(
        json.dumps({"cases": report.ideal_ms(), "mean_ideal_ms": report.mean_ideal_ms()}),
        encoding="utf-8",
    )
    return path


def _loop(
    path: str = "",
    *,
    target: float = 0.0,
    case_times: dict[str, float] | None = None,
    estimator=None,
    resume: bool = False,
    run_state: RunState | None = None,
) -> IterationLoop:
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            roofline_target=target,
        ),
        tracker=object(),
        config=object(),
        resume=resume,
        ceiling_estimator=estimator,
    )
    # Stands in for a ceiling ``_establish_ceiling`` already published.
    loop._ceiling_report_path = path
    loop._baseline_case_times = dict(case_times or {"decode-t1": 40.0})
    loop._best_case_times = dict(loop._baseline_case_times)
    # ``_run_locked`` loads both before it reaches the ceiling; these stand in for it.
    loop.run_state = run_state or RunState()
    loop.saved_states = []
    loop.state_store = SimpleNamespace(save=lambda state: loop.saved_states.append(state.ceiling_report_path))
    return loop


def test_a_campaign_without_a_ceiling_injects_nothing():
    assert _loop()._render_ceiling_advisory() == ""


def test_a_published_ceiling_reaches_the_implementer_with_its_attainment(tmp_path):
    path = _publish(_report(), tmp_path)

    rendered = _loop(str(path))._render_ceiling_advisory()

    assert "decode-t1" in rendered
    # 12.8 / 40.0
    assert "32.0%" in rendered
    assert "decides no KEEP" in rendered


def test_an_unreadable_ceiling_costs_a_log_line_not_the_campaign(tmp_path, caplog):
    broken = tmp_path / REPORT_FILENAME
    broken.write_text("{not json", encoding="utf-8")

    assert _loop(str(broken))._render_ceiling_advisory() == ""
    assert "ceiling unavailable" in caplog.text


def test_an_unreadable_ceiling_is_not_re_read_every_iteration(tmp_path):
    missing = tmp_path / "absent.json"
    loop = _loop(str(missing))

    loop._render_ceiling_advisory()
    # A second call must not touch the filesystem again: the sentinel records
    # that the lookup already happened and failed.
    assert loop._ceiling_report is None
    assert loop._render_ceiling_advisory() == ""


def test_the_ceiling_never_reaches_the_keep_decision():
    """The one invariant: a derived estimate cannot gate a measured decision.

    Matched on the identifiers rather than the word, because ``scoring`` uses
    "ceiling" in its own unrelated sense for the variance-share cap.
    """
    from kernelforge.loop import scoring

    forbidden = ("kernelforge.roofline_ceiling", "ceiling_report", "_render_ceiling_advisory")
    decision_sites = (
        inspect.getsource(scoring),
        inspect.getsource(IterationLoop.run_one_iteration),
        inspect.getsource(IterationLoop._resolve_keep_sigma),
    )
    for source in decision_sites:
        assert not any(name in source for name in forbidden)


def test_one_reader_loads_the_ceiling_for_the_whole_loop():
    """One loader means one thing to audit if the invariant above ever bends."""
    from kernelforge.loop import runner as runner_module

    source = inspect.getsource(runner_module)
    assert source.count("from kernelforge.roofline_ceiling.report import read_report") == 1
    assert "read_report" in inspect.getsource(IterationLoop._ceiling)
    # Every other user goes through that loader rather than the filesystem.
    for method in (
        IterationLoop._roofline_attainment,
        IterationLoop._render_ceiling_advisory,
        IterationLoop._with_ceiling_standing,
    ):
        assert "self._ceiling()" in inspect.getsource(method)
        assert "read_report" not in inspect.getsource(method)


def _planning_context(*case_ids):
    from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext

    return OrchestrationContext(
        analysis_commit="abc123",
        workspace="/w",
        gpu_target="gfx950",
        objective="minimize latency",
        program_context="one GEMM",
        source_map_path="/w/map.json",
        cases=tuple(CaseEvidence(case_id=case_id, latency_ms=40.0) for case_id in case_ids),
    )


def test_the_planner_sees_how_far_each_case_sits_from_its_ceiling(tmp_path):
    """The planner picks the round's cases, so it is the one the headroom has to reach."""
    path = _publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    loop = _loop(str(path), case_times={"a": 10.0, "b": 8.0})

    context = loop._with_ceiling_standing(_planning_context("a", "b"))
    cases = {case["case_id"]: case for case in context.to_prompt_dict()["cases"]}

    assert cases["a"]["roofline"] == {"ceiling_ms": 9.0, "incumbent_ms": 10.0, "attainment": 0.9, "excluded": ""}
    assert cases["b"]["roofline"]["attainment"] == pytest.approx(0.5)


def test_a_contradicted_ceiling_reaches_the_planner_as_one_with_its_reason(tmp_path):
    """A ceiling above the measured latency is shown as untrustworthy, not as a finished case."""
    path = _publish(_report((("a", 9.0),)), tmp_path)
    loop = _loop(str(path), case_times={"a": 6.0})

    (case,) = loop._with_ceiling_standing(_planning_context("a")).cases

    assert case.roofline.attainment is None
    assert case.roofline.incumbent_ms == 6.0
    assert case.roofline.excluded


def test_without_a_ceiling_the_planner_plans_from_exactly_the_evidence_it_always_did():
    context = _planning_context("a")

    assert _loop()._with_ceiling_standing(context) is context
    assert all("roofline" not in case for case in context.to_prompt_dict()["cases"])


def test_the_standing_is_attached_where_the_round_is_planned():
    """Pinned at the call site: attaching it anywhere after planning reaches only the implementer."""
    source = inspect.getsource(IterationLoop._run_orchestration)
    planned_at = source.index("orchestration_service.run(")

    assert -1 < source.index("self._with_ceiling_standing(") < planned_at


def test_the_ceiling_is_off_unless_an_operator_turns_it_on():
    """An estimate costs a profiler pass and an analyst session, so nobody pays for one by default."""
    option = next(param for param in cli_module.forge_loop.params if param.name == "roofline_ceiling")

    assert option.opts == ["--roofline-ceiling"]
    assert option.default == "off"
    assert list(option.type.choices) == ["on", "off"]
    assert option.type.convert("ON", option, None) == "on"


def test_the_attainment_target_is_off_unless_an_operator_asks_for_it():
    option = next(param for param in cli_module.forge_loop.params if param.name == "roofline_target")

    assert option.opts == ["--roofline-target"]
    assert option.default == 0.0


def test_a_target_above_one_is_refused_because_attainment_cannot_exceed_the_ceiling():
    with pytest.raises(click.BadParameter, match="cannot exceed 1.0"):
        cli_module._validate_roofline_target(86.0, True)


def test_a_target_without_a_ceiling_is_refused_rather_than_never_firing():
    with pytest.raises(click.BadParameter, match="pass --roofline-ceiling on"):
        cli_module._validate_roofline_target(0.86, False)


def test_no_estimator_is_built_while_the_ceiling_is_off():
    assert (
        cli_module._make_ceiling_estimator(
            enabled=False,
            workspace_dir=".",
            driver_script="driver.py",
            source_files=["kernel.py"],
            agent_provider="",
            agent_model="",
            session_timeout_sec=60,
        )
        is None
    )


def test_the_estimator_calls_estimate_ceiling_with_arguments_it_accepts(monkeypatch):
    """The loop's estimator is invoked, not just built.

    Building it proves nothing: the call into ``estimate_ceiling`` is where a
    keyword the signature does not have turns into a ``TypeError``, and that
    only happens once a campaign has already paid for a baseline.
    """
    seen = {}

    async def _estimate(backend, **kwargs):
        seen.update(kwargs)
        return "outcome"

    monkeypatch.setattr("kernelforge.roofline_ceiling.estimate.estimate_ceiling", _estimate)
    monkeypatch.setattr(
        "kernelforge.roofline_ceiling.command.resolve_analyst_backend",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    estimator = cli_module._make_ceiling_estimator(
        enabled=True,
        workspace_dir=".",
        driver_script="driver.py",
        source_files=["kernel.py"],
        agent_provider="",
        agent_model="",
        session_timeout_sec=60,
    )
    assert estimator is not None

    result = asyncio.run(estimator(case_ids=["decode-t1"], case_ms={"decode-t1": 14.0}))

    assert result == "outcome"
    assert seen["known_case_ids"] == ["decode-t1"]
    # Every keyword the estimator sends has to be one the signature declares.
    accepted = set(inspect.signature(real_estimate_ceiling).parameters)
    assert set(seen) <= accepted, sorted(set(seen) - accepted)


def test_the_target_ends_the_campaign_once_the_mean_reaches_it(tmp_path):
    path = _publish(_report(), tmp_path)
    # 12.8 / 14.0 = 91.4%, above the target.
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 14.0})

    assert loop._is_roofline_target_met()


def test_the_target_does_not_fire_while_a_case_is_still_short(tmp_path):
    path = _publish(_report(), tmp_path)
    # 12.8 / 20.0 = 64%.
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 20.0})

    assert not loop._is_roofline_target_met()


def test_the_mean_is_equal_weight_across_cases_like_the_keep_objective(tmp_path):
    path = _publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    # 90% and 80% -> mean 85%, just under the target. A latency-weighted mean
    # would read differently, and would then disagree with what a KEEP scores.
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 5.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.85)
    assert not loop._is_roofline_target_met()


def test_a_ceiling_below_the_measured_latency_cannot_end_the_campaign(tmp_path):
    """The estimate contradicting itself must not read as a finished kernel."""
    path = _publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    # `b`'s ceiling is above its measurement, so it is excluded; `a` alone would
    # average 90% and clear the target on a case set of two.
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 2.0})

    standing = loop._roofline_attainment()
    assert "b" in standing.excluded
    assert standing.mean == pytest.approx(0.9)
    assert not loop._is_roofline_target_met()


def test_a_case_the_ceiling_never_answered_cannot_end_the_campaign(tmp_path):
    path = _publish(_report((("a", 9.0),)), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 5.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.9)
    assert not loop._is_roofline_target_met()


def test_without_a_target_the_ceiling_ends_nothing(tmp_path):
    path = _publish(_report(), tmp_path)
    loop = _loop(str(path), target=0.0, case_times={"decode-t1": 12.8})

    assert loop._roofline_attainment().mean == pytest.approx(1.0)
    assert not loop._is_roofline_target_met()


def test_attainment_follows_the_incumbent_not_the_frozen_anchor(tmp_path):
    path = _publish(_report(), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 40.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.32)
    loop._best_case_times = {"decode-t1": 13.0}
    assert loop._roofline_attainment().mean == pytest.approx(12.8 / 13.0)


def test_the_campaign_estimates_its_ceiling_from_the_baseline_it_measured(tmp_path):
    """The estimator is handed the scored case set and the campaign's own clock."""
    published = _publish(_report(), tmp_path)
    seen = {}

    async def estimator(*, case_ids, case_ms):
        seen["case_ids"] = list(case_ids)
        seen["case_ms"] = dict(case_ms)
        return SimpleNamespace(report=_report(), report_path=published, source="analyst", notes=())

    loop = _loop(target=0.86, case_times={"decode-t1": 40.0}, estimator=estimator)
    asyncio.run(loop._establish_ceiling())

    assert seen["case_ids"] == ["decode-t1"]
    assert seen["case_ms"] == {"decode-t1": 40.0}
    assert loop._ceiling_report_path == str(published)
    assert loop._roofline_attainment().mean == pytest.approx(0.32)
    # Checkpointed, so a resume of this campaign reads it back instead of estimating.
    assert loop.run_state.ceiling_report_path == str(published)
    assert loop.saved_states == [str(published)]


def _estimator_that_counts(published, calls):
    async def estimator(**_kwargs):
        calls.append(1)
        return SimpleNamespace(report=_report(), report_path=published, source="analyst", notes=())

    return estimator


def test_a_resumed_campaign_reads_back_the_ceiling_it_recorded(tmp_path):
    """``--roofline-ceiling on`` with ``--resume``: the run state, not a second estimate, supplies the ceiling."""
    path = _publish(_report(), tmp_path)

    async def estimator(**_kwargs):
        raise AssertionError("a resumed campaign paid for a second estimate")

    loop = _loop(estimator=estimator, resume=True, run_state=RunState(ceiling_report_path=str(path)))
    asyncio.run(loop._establish_ceiling())

    assert loop._ceiling_report_path == str(path)
    assert loop._roofline_attainment().mean == pytest.approx(0.32)


@pytest.mark.parametrize("resume", [False, True])
def test_a_report_this_campaign_did_not_record_is_never_adopted(tmp_path, resume):
    """A report sitting in the workspace may belong to another run, on another box."""
    leftover = _publish(_report((("decode-t1", 1.0),)), tmp_path / "leftover")
    published = _publish(_report(), tmp_path / "fresh")
    calls = []

    loop = _loop(estimator=_estimator_that_counts(published, calls), resume=resume)
    asyncio.run(loop._establish_ceiling())

    assert calls == [1]
    assert loop._ceiling_report_path == str(published) != str(leftover)


def test_a_recorded_ceiling_without_every_scored_case_is_estimated_again(tmp_path):
    partial = _publish(_report((("prefill-t16", 3.0),)), tmp_path / "partial")
    published = _publish(_report(), tmp_path / "fresh")
    calls = []

    loop = _loop(
        estimator=_estimator_that_counts(published, calls),
        resume=True,
        run_state=RunState(ceiling_report_path=str(partial)),
    )
    asyncio.run(loop._establish_ceiling())

    assert calls == [1]
    assert loop.run_state.ceiling_report_path == str(published)


def test_a_recorded_ceiling_that_cannot_be_read_is_estimated_again(tmp_path, capsys):
    published = _publish(_report(), tmp_path / "fresh")
    calls = []

    loop = _loop(
        estimator=_estimator_that_counts(published, calls),
        resume=True,
        run_state=RunState(ceiling_report_path=str(tmp_path / "gone.json")),
    )
    asyncio.run(loop._establish_ceiling())

    assert calls == [1]
    assert "cannot be read back" in capsys.readouterr().out


def test_an_estimate_that_fails_costs_the_target_not_the_campaign(tmp_path, capsys):
    async def estimator(**_kwargs):
        raise RuntimeError("rocprof-compute is not installed")

    loop = _loop(target=0.86, estimator=estimator)
    asyncio.run(loop._establish_ceiling())

    assert loop._ceiling_report_path == ""
    assert not loop._is_roofline_target_met()
    assert "no ceiling for this campaign" in capsys.readouterr().out


def test_no_estimate_is_attempted_before_the_case_set_is_known():
    async def estimator(**_kwargs):
        raise AssertionError("estimated without a scored case set")

    loop = _loop(estimator=estimator, case_times={})
    asyncio.run(loop._establish_ceiling())

    assert loop._ceiling_report_path == ""


def test_an_unscored_case_is_left_out_of_the_standing(tmp_path):
    path = _publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 8.0})
    loop._unscored_cases = {"b"}

    standing = loop._roofline_attainment()
    assert standing.mean == pytest.approx(0.9)
    # `b` is not scored by the objective, so its absence does not block the gate.
    assert loop._is_roofline_target_met()


def test_off_leaves_a_report_sitting_in_the_workspace_unread(tmp_path):
    """Off means off: a report from an earlier run changes nothing the planner sees."""
    _publish(_report(), tmp_path / WORKSPACE_SUBDIR)
    loop = _loop(estimator=None)
    loop.ic.workspace_dir = str(tmp_path)

    asyncio.run(loop._establish_ceiling())

    assert loop._ceiling_report_path == ""
    assert loop._render_ceiling_advisory() == ""


def test_the_ceiling_command_is_registered():
    assert "roofline-ceiling" in cli_module.main.commands
