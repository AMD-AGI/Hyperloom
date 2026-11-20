# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that profiles chooses what the profiler runs, and keeps any script it writes where its role may write."""

from __future__ import annotations

import pytest

from kernelforge.config.resources import resource_path
from kernelforge.utils.gpu_arch import Profiler
from kernelforge.workflows.forge_loop.roles import agent
from kernelforge.workflows.forge_loop.roles.analysis import PROFILING_METHODOLOGY_FILES
from kernelforge.workflows.forge_loop.roles.analysis_prompts import PROFILED_SYSTEM_PROMPT


def _implementer_section(profiler: Profiler) -> str:
    section = agent._SELF_PROFILING_SECTIONS[profiler].format(profiling_dir="/kb/profiling", driver_run="driver.py")
    return " ".join(section.split())


@pytest.mark.parametrize("profiler", [Profiler.ROCPROF_COMPUTE, Profiler.ROCPROFV3])
def test_the_session_is_told_to_choose_the_command_the_profiler_replays(profiler) -> None:
    prose = _implementer_section(profiler)

    assert "-- <command ...>" in prose
    assert "A mode of `driver.py` that already does this is the cheapest" in prose
    assert "write a short script in a scratch directory outside the workspace" in prose
    assert "do not add such a mode to it" in prose
    assert "--profile-run" not in prose


def test_the_cards_both_roles_read_leave_the_script_location_to_each_role() -> None:
    """The Analysis Agent may write only inside its staging directory, which lies inside the workspace."""
    profiling = resource_path("local_knowledge") / "common_methodology" / "profiling"

    for name in PROFILING_METHODOLOGY_FILES:
        assert "outside the workspace" not in (profiling / name).read_text(encoding="utf-8"), name
    assert "write an adapter in staging/tools" in " ".join(PROFILED_SYSTEM_PROMPT.split())
