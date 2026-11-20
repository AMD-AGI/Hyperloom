# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that profiles chooses what the profiler runs; no driver is assumed to carry a mode for it."""

from __future__ import annotations

import pytest

from kernelforge.utils.gpu_arch import Profiler
from kernelforge.workflows.forge_loop.roles import agent


@pytest.mark.parametrize("profiler", [Profiler.ROCPROF_COMPUTE, Profiler.ROCPROFV3])
def test_the_session_is_told_to_choose_the_command_the_profiler_replays(profiler) -> None:
    section = agent._SELF_PROFILING_SECTIONS[profiler].format(profiling_dir="/kb/profiling", driver_run="driver.py")
    prose = " ".join(section.split())

    assert "-- <command ...>" in prose
    assert "A mode of `driver.py` that already does this is the cheapest" in prose
    assert "write a short script in a scratch directory outside the workspace" in prose
    assert "do not add such a mode to it" in prose
    assert "--profile-run" not in prose
