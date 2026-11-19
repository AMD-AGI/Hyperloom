# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A rank that stops waiting stops every rank: a wait stalled by the runtime's fault injection, at
each of the launch's waits, and a rank that never launches. Every launching rank exits non-zero
within the wait deadline and a margin, without finishing its launches."""

import pytest
from ranks import MARGIN_S, RankRun, launch_ranks
from runner import DEFAULT_DEADLINE_S

FAULTY = 2
# the launch's waits in trace order (guard sites): the reduce's poll, the norm's poll, the epoch's end
WAIT_SITES = (1, 2, 3)


def _assert_stopped(run: RankRun) -> None:
    assert run.returncode != 0, f"rank {run.rank} exited 0\n{run.tail()}"
    assert not run.finished, f"rank {run.rank} finished its launches\n{run.tail()}"
    assert run.launched is not None, f"rank {run.rank} never reached its launches\n{run.tail()}"
    took = run.exited - run.launched
    assert took <= DEFAULT_DEADLINE_S + MARGIN_S, f"rank {run.rank} took {took:.1f} s to stop\n{run.tail()}"


@pytest.mark.parametrize("site", WAIT_SITES)
def test_a_stalled_wait_stops_every_rank(site):
    # the stall arms after one launch, so the first launch completes on every rank
    runs = launch_ranks("fault", env={"SGLANG_AMD_MONO_FAULT_INJECT": f"{FAULTY}:{site}:1"}, timeout=900.0)
    for run in runs:
        _assert_stopped(run)


def test_an_absent_rank_stops_the_others():
    runs = launch_ranks("absent", str(FAULTY), timeout=900.0)
    for run in runs:
        if run.rank == FAULTY:
            assert run.returncode == 0, run.tail()
        else:
            _assert_stopped(run)
