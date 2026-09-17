# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""How the ceiling reaches a campaign, and the one place it must never reach."""

from __future__ import annotations

import inspect

import click
import pytest

from kernelforge import cli as cli_module
from kernelforge.roofline_ceiling.contract import Hardware, build_report
from kernelforge.roofline_ceiling.report import REPORT_FILENAME, WORKSPACE_SUBDIR, publish
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_EMPIRICAL
from kernelforge.loop.runner import IterationConfig, IterationLoop


def _report():
    payload = {
        "cases": [
            {
                "case_id": "decode-t1",
                "stages": [
                    {
                        "name": "gemm",
                        "flops": 2.0e12,
                        "bytes": 8.0e10,
                        "instruction_path": "bf16_mfma",
                        "dispatch_count": 1,
                        "formula_flops": "2*M*N*K",
                        "formula_bytes": "M*K*2",
                    }
                ],
            }
        ],
        "confidence": "high",
    }
    return build_report(
        payload,
        canonical_id="roofline-ceiling:op:gfx950",
        hardware=Hardware(
            arch="gfx950",
            hbm_bw_bytes_per_s=8.0e12,
            peak_flops={"bf16_mfma": 2.0e15},
            peak_source=PEAK_SOURCE_EMPIRICAL,
            dispatch_floor_s=2.0e-6,
        ),
        expected_case_ids=["decode-t1"],
    )


def _loop(path: str = "") -> IterationLoop:
    loop = IterationLoop(
        IterationConfig(kernel_file="kernel.py", driver_script="driver.py", ceiling_report_path=path),
        tracker=object(),
        config=object(),
    )
    loop._baseline_case_times = {"decode-t1": 40.0}
    return loop


def test_a_campaign_without_a_ceiling_injects_nothing():
    assert _loop()._render_ceiling_advisory() == ""


def test_a_published_ceiling_reaches_the_implementer_with_its_headroom(tmp_path):
    path = publish(_report(), tmp_path)

    rendered = _loop(str(path))._render_ceiling_advisory()

    assert "decode-t1" in rendered
    assert "Headroom" in rendered
    assert "advisory" in rendered.lower()


def test_an_unreadable_ceiling_costs_a_log_line_not_the_campaign(tmp_path, caplog):
    broken = tmp_path / REPORT_FILENAME
    broken.write_text("{not json", encoding="utf-8")

    assert _loop(str(broken))._render_ceiling_advisory() == ""
    assert "ceiling advisory unavailable" in caplog.text


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


def test_the_advisory_is_the_only_place_the_loop_reads_a_ceiling():
    """One reader means one thing to audit if the invariant above ever bends."""
    from kernelforge.loop import runner as runner_module

    source = inspect.getsource(runner_module)
    assert source.count("self._render_ceiling_advisory()") == 1
    assert source.count("from kernelforge.roofline_ceiling.report import") == 2  # read_report, render_for_prompt


def test_forge_loop_offers_the_option_and_defaults_to_auto():
    option = next(param for param in cli_module.forge_loop.params if param.name == "ceiling_report")

    assert option.opts == ["--roofline-ceiling"]
    assert option.default == "auto"


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
