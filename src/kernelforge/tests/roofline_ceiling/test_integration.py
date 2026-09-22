# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""How the ceiling reaches a campaign, and the one place it must never reach."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import click
import pytest

from kernelforge import cli as cli_module
from kernelforge.roofline_ceiling.contract import Hardware, build_report
from kernelforge.roofline_ceiling.report import REPORT_FILENAME, WORKSPACE_SUBDIR, publish
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_REFERENCE
from kernelforge.loop.runner import IterationConfig, IterationLoop


def _report(cases=(("decode-t1", 12.8),)):
    payload = {
        "cases": [{"case_id": case_id, "t_ideal_ms": ideal, "bound": "memory"} for case_id, ideal in cases],
        "confidence": "high",
        "analysis_md": "# Performance ceiling analysis\n\n"
        + "\n".join(f"Case `{case_id}`: 8e10 B / 6.24 TB/s = {ideal} ms." for case_id, ideal in cases),
    }
    return build_report(
        payload,
        canonical_id="roofline-ceiling:op:gfx950",
        hardware=Hardware(
            arch="gfx950",
            peak_flops={"bf16_mfma": 1.686e15},
            bandwidth={"hbm": 6.24e12},
            peak_source=PEAK_SOURCE_REFERENCE,
            dispatch_floor_s=3.0e-6,
        ),
        expected_case_ids=[case_id for case_id, _ in cases],
    )


def _loop(
    path: str = "",
    *,
    target: float = 0.0,
    case_times: dict[str, float] | None = None,
    estimator=None,
) -> IterationLoop:
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            ceiling_report_path=path,
            roofline_target=target,
        ),
        tracker=object(),
        config=object(),
        ceiling_estimator=estimator,
    )
    loop._baseline_case_times = dict(case_times or {"decode-t1": 40.0})
    loop._best_case_times = dict(loop._baseline_case_times)
    return loop


def test_a_campaign_without_a_ceiling_injects_nothing():
    assert _loop()._render_ceiling_advisory() == ""


def test_a_published_ceiling_reaches_the_implementer_with_its_attainment(tmp_path):
    path = publish(_report(), tmp_path)

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
    for method in (IterationLoop._roofline_attainment, IterationLoop._render_ceiling_advisory):
        assert "self._ceiling()" in inspect.getsource(method)
        assert "read_report" not in inspect.getsource(method)


def test_forge_loop_offers_the_option_and_defaults_to_auto():
    option = next(param for param in cli_module.forge_loop.params if param.name == "ceiling_report")

    assert option.opts == ["--roofline-ceiling"]
    assert option.default == "auto"


def test_the_attainment_target_is_off_unless_an_operator_asks_for_it():
    option = next(param for param in cli_module.forge_loop.params if param.name == "roofline_target")

    assert option.opts == ["--roofline-target"]
    assert option.default == 0.0


def test_compute_defers_the_ceiling_to_the_loop_where_the_baseline_is(tmp_path):
    """``compute`` resolves to no path: the estimate needs the measured case set."""
    assert cli_module._resolve_ceiling_report("compute", str(tmp_path)) == ""


def test_a_target_above_one_is_refused_because_attainment_cannot_exceed_the_ceiling():
    with pytest.raises(click.BadParameter, match="cannot exceed 1.0"):
        cli_module._validate_roofline_target(86.0, "auto")


def test_a_target_without_a_ceiling_is_refused_rather_than_never_firing():
    with pytest.raises(click.BadParameter, match="needs a ceiling"):
        cli_module._validate_roofline_target(0.86, "off")


def test_no_estimator_is_built_unless_compute_was_asked_for():
    assert (
        cli_module._make_ceiling_estimator(
            selection="auto",
            workspace_dir=".",
            driver_script="driver.py",
            source_files=["kernel.py"],
            operator_name="op",
            agent_provider="",
            agent_model="",
            session_timeout_sec=60,
        )
        is None
    )


def test_the_target_ends_the_campaign_once_the_mean_reaches_it(tmp_path):
    path = publish(_report(), tmp_path)
    # 12.8 / 14.0 = 91.4%, above the target.
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 14.0})

    assert loop._is_roofline_target_met()


def test_the_target_does_not_fire_while_a_case_is_still_short(tmp_path):
    path = publish(_report(), tmp_path)
    # 12.8 / 20.0 = 64%.
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 20.0})

    assert not loop._is_roofline_target_met()


def test_the_mean_is_equal_weight_across_cases_like_the_keep_objective(tmp_path):
    path = publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    # 90% and 80% -> mean 85%, just under the target. A latency-weighted mean
    # would read differently, and would then disagree with what a KEEP scores.
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 5.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.85)
    assert not loop._is_roofline_target_met()


def test_a_ceiling_below_the_measured_latency_cannot_end_the_campaign(tmp_path):
    """The estimate contradicting itself must not read as a finished kernel."""
    path = publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    # `b`'s ceiling is above its measurement, so it is excluded; `a` alone would
    # average 90% and clear the target on a case set of two.
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 2.0})

    standing = loop._roofline_attainment()
    assert "b" in standing.excluded
    assert standing.mean == pytest.approx(0.9)
    assert not loop._is_roofline_target_met()


def test_a_case_the_ceiling_never_answered_cannot_end_the_campaign(tmp_path):
    path = publish(_report((("a", 9.0),)), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 5.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.9)
    assert not loop._is_roofline_target_met()


def test_without_a_target_the_ceiling_ends_nothing(tmp_path):
    path = publish(_report(), tmp_path)
    loop = _loop(str(path), target=0.0, case_times={"decode-t1": 12.8})

    assert loop._roofline_attainment().mean == pytest.approx(1.0)
    assert not loop._is_roofline_target_met()


def test_attainment_follows_the_incumbent_not_the_frozen_anchor(tmp_path):
    path = publish(_report(), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"decode-t1": 40.0})

    assert loop._roofline_attainment().mean == pytest.approx(0.32)
    loop._best_case_times = {"decode-t1": 13.0}
    assert loop._roofline_attainment().mean == pytest.approx(12.8 / 13.0)


def test_the_campaign_estimates_its_ceiling_from_the_baseline_it_measured(tmp_path):
    """The estimator is handed the scored case set and the campaign's own clock."""
    published = publish(_report(), tmp_path)
    seen = {}

    async def estimator(*, case_ids, case_ms):
        seen["case_ids"] = list(case_ids)
        seen["case_ms"] = dict(case_ms)
        return SimpleNamespace(report=_report(), report_path=published, source="analyst", notes=())

    loop = _loop(target=0.86, case_times={"decode-t1": 40.0}, estimator=estimator)
    asyncio.run(loop._establish_ceiling())

    assert seen["case_ids"] == ["decode-t1"]
    assert seen["case_ms"] == {"decode-t1": 40.0}
    assert loop.ic.ceiling_report_path == str(published)
    assert loop._roofline_attainment().mean == pytest.approx(0.32)


def test_a_ceiling_already_published_is_not_estimated_again(tmp_path):
    path = publish(_report(), tmp_path)

    async def estimator(**_kwargs):
        raise AssertionError("an estimate was paid for twice")

    loop = _loop(str(path), estimator=estimator)
    asyncio.run(loop._establish_ceiling())

    assert loop.ic.ceiling_report_path == str(path)


def test_an_estimate_that_fails_costs_the_target_not_the_campaign(tmp_path, capsys):
    async def estimator(**_kwargs):
        raise RuntimeError("rocprof-compute is not installed")

    loop = _loop(target=0.86, estimator=estimator)
    asyncio.run(loop._establish_ceiling())

    assert loop.ic.ceiling_report_path == ""
    assert not loop._is_roofline_target_met()
    assert "no ceiling for this campaign" in capsys.readouterr().out


def test_no_estimate_is_attempted_before_the_case_set_is_known():
    async def estimator(**_kwargs):
        raise AssertionError("estimated without a scored case set")

    loop = _loop(estimator=estimator, case_times={})
    asyncio.run(loop._establish_ceiling())

    assert loop.ic.ceiling_report_path == ""


def test_an_unscored_case_is_left_out_of_the_standing(tmp_path):
    path = publish(_report((("a", 9.0), ("b", 4.0))), tmp_path)
    loop = _loop(str(path), target=0.86, case_times={"a": 10.0, "b": 8.0})
    loop._unscored_cases = {"b"}

    standing = loop._roofline_attainment()
    assert standing.mean == pytest.approx(0.9)
    # `b` is not scored by the objective, so its absence does not block the gate.
    assert loop._is_roofline_target_met()


def test_auto_finds_a_ceiling_the_command_published_into_the_workspace(tmp_path):
    publish(_report(), tmp_path / WORKSPACE_SUBDIR)

    resolved = cli_module._resolve_ceiling_report("auto", str(tmp_path))

    assert resolved == str(tmp_path / WORKSPACE_SUBDIR / REPORT_FILENAME)


def test_auto_without_a_published_ceiling_is_simply_no_ceiling(tmp_path):
    assert cli_module._resolve_ceiling_report("auto", str(tmp_path)) == ""


def test_off_declines_a_ceiling_that_is_sitting_right_there(tmp_path):
    publish(_report(), tmp_path / WORKSPACE_SUBDIR)

    assert cli_module._resolve_ceiling_report("off", str(tmp_path)) == ""


def test_an_explicit_path_that_does_not_exist_is_an_error_not_a_shrug(tmp_path):
    """Asking for a specific ceiling and silently getting none hides the typo."""
    with pytest.raises(click.BadParameter, match="not a readable report"):
        cli_module._resolve_ceiling_report(str(tmp_path / "typo.json"), str(tmp_path))


def test_an_explicit_path_is_used_verbatim(tmp_path):
    path = publish(_report(), tmp_path / "elsewhere")

    assert cli_module._resolve_ceiling_report(str(path), str(tmp_path)) == str(path)


def test_the_ceiling_command_is_registered():
    assert "roofline-ceiling" in cli_module.main.commands
