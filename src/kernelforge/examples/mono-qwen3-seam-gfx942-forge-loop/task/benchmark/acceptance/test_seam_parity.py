# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TP4 at every width: two back-to-back launches per width, each held to SGLang's path."""

from ranks import TP, launch_ranks

from qwen3_seam import seam

WIDTHS = tuple(range(1, seam.MAX_TOKENS + 1))


def test_every_width_matches_the_reference_on_every_rank():
    runs = launch_ranks("parity", *map(str, WIDTHS), timeout=1800.0)
    for run in runs:
        assert run.returncode == 0 and run.finished, run.tail()
    verdicts = [v for run in runs for v in run.verdicts()]
    assert len(verdicts) == TP * len(WIDTHS) * 2
    failed = [v for v in verdicts if not v["passed"]]
    assert not failed, failed[:8]
