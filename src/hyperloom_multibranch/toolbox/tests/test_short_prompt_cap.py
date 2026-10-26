# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt check's cap is sized from the session's measured boots and prompt times (board #126 c2813 point 3),
and is the rule pack's until a boot is measured."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hyperloom.measure.short_prompts.outputs import FAILED, OK, ProcessEntry
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.tests.support import outputs_files
from hyperloom_multibranch.toolbox.short_prompt_cap import CAP_MARGIN_FRACTION, CAP_MARGIN_SEC, CheckTimes
from hyperloom_multibranch.toolbox.short_prompt_cap import landed_processes
from hyperloom_multibranch.toolbox.short_prompts import short_prompt_rules

PLAN = ShortPromptPlan("short-prompts-test", "a" * 64, processes=6, phase_cap_sec=1200.0)


def _entry(status: str, boot: float, prompts: float) -> ProcessEntry:
    return ProcessEntry(0, status, "", boot, prompts, {})


def _cap(processes: int, boot: float, prompts: float) -> float:
    return round(processes * (boot + prompts) * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC, 1)


def test_with_no_boot_measured_the_cap_is_the_rule_packs() -> None:
    times = CheckTimes()

    assert times.sized(PLAN) == PLAN
    assert times.sized(None) is None
    assert short_prompt_rules().phase_cap_sec == 1200.0, "today's fallback, from the pack"


def test_a_measured_boot_sizes_the_cap_for_the_check_s_processes_before_any_prompt_time_is_measured() -> None:
    times = CheckTimes()
    times.saw_boot(196.0)

    sized = times.sized(PLAN)

    assert sized is not None and sized.phase_cap_sec == _cap(6, 196.0, 0.0)
    assert sized.processes == PLAN.processes and sized.trigger == PLAN.trigger, "only the cap changes"


def test_a_landed_check_adds_its_prompt_time_and_its_processes_own_boots() -> None:
    times = CheckTimes()
    times.saw_boot(150.0)
    times.saw_check([_entry(OK, 190.0, 40.0), _entry(OK, 170.0, 55.0)])

    sized = times.sized(replace_processes(PLAN, 8))

    assert sized is not None and sized.phase_cap_sec == _cap(8, 190.0, 55.0)


def test_the_longest_boot_is_kept_not_the_latest() -> None:
    times = CheckTimes()
    times.saw_boot(400.0)
    times.saw_boot(90.0)

    sized = times.sized(PLAN)

    assert sized is not None and sized.phase_cap_sec == _cap(6, 400.0, 0.0)


@pytest.mark.parametrize("seconds", [None, 0.0, -3.0, math.nan, math.inf])
def test_what_is_not_a_measured_time_measures_nothing(seconds: float | None) -> None:
    times = CheckTimes()
    times.saw_boot(seconds)
    times.saw_check([_entry(OK, 0.0, 0.0)])

    assert times.sized(PLAN) == PLAN


def test_a_process_that_did_not_answer_measures_neither_a_boot_nor_a_prompt_time() -> None:
    times = CheckTimes()
    times.saw_check([_entry(FAILED, 900.0, 300.0)])
    assert times.sized(PLAN) == PLAN

    times.saw_boot(100.0)
    sized = times.sized(PLAN)
    assert sized is not None and sized.phase_cap_sec == _cap(6, 100.0, 0.0)


def test_the_landed_processes_are_read_from_the_outputs_file_and_none_without_one(tmp_path: Path) -> None:
    for relative, data in outputs_files(PLAN, ["good", "fail"]).items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(data)

    assert [(e.status, e.boot_seconds, e.query_seconds) for e in landed_processes(tmp_path)] == [
        (OK, 80.0, 20.0),
        (FAILED, 80.0, 20.0),
    ]
    assert landed_processes(tmp_path / "nothing-landed") == ()


def replace_processes(plan: ShortPromptPlan, processes: int) -> ShortPromptPlan:
    return ShortPromptPlan(plan.prompt_set, plan.prompts_sha256, processes=processes, phase_cap_sec=plan.phase_cap_sec)
