# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign process never opens the GPU it measures on, so a GPU hang leaves it running."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from kernelforge.tests.conftest import SRC_ROOT


@pytest.mark.skipif(sys.platform != "linux", reason="reads the process's mappings and descriptors from /proc")
def test_a_campaign_that_measures_and_keeps_never_initializes_the_gpu(tmp_path):
    """Every measurement runs in a driver process; the campaign only reads what the drivers print.

    A GPU hang ends every process holding the GPU open, wherever its work ran. A campaign that loaded the GPU runtime
    would be ended by a hang its own candidate did not cause, taking the run with it.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if path)
    completed = subprocess.run(
        [sys.executable, "-m", "kernelforge.tests.workflows.forge_loop.iteration._gpu_free_campaign", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["decisions"] == ["KEEP", "REVERT_PERF"]
    assert report["libraries"] == []
    assert report["descriptors"] == []
