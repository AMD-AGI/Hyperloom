# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading a chatty server log still witnesses every bring-up milestone."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors.baseline import read_bringup_log
from hyperloom.orchestrator.bringup import observe_bringup

#: One line per kernel shape, which is what aiter emits on ROCm. Enough of them
#: and the readiness marker sits past the head window.
_SHAPE_NOISE = "[aiter] shape is M:1, N:6144, K:4096, not found tuned config, will use default config!\n"


def _served_boot_log(path: Path, *, edge_bytes: int) -> None:
    """Write a log whose readiness marker lands in the unread middle."""
    head = "server_args=Namespace(model='m')\nLoading weights\nLoading weights took 42s\nGPU blocks: 1024\n"
    noise = _SHAPE_NOISE * (3 * edge_bytes // len(_SHAPE_NOISE))
    path.write_text(
        head + noise + "INFO: Application startup complete.\n" + noise + "INFO: 127.0.0.1 - POST /v1/completions\n",
        encoding="utf-8",
    )


def test_readiness_past_the_head_window_still_reads_as_booted(tmp_path: Path) -> None:
    """A served boot must not classify as ENGINE_INIT just because the log is long."""
    edge_bytes = 4096
    log = tmp_path / "server.log"
    _served_boot_log(log, edge_bytes=edge_bytes)
    assert log.stat().st_size > edge_bytes * 2

    read = read_bringup_log(log, edge_bytes=edge_bytes)
    observation = observe_bringup(server_log=read.text).observation
    assert observation.stage_reached is LadderStage.HTTP_READY
    assert observation.booted is True


def test_the_tail_wall_still_wins_over_a_middle_milestone(tmp_path: Path) -> None:
    """Carrying milestones out of the middle must not mask a later failure."""
    edge_bytes = 4096
    log = tmp_path / "server.log"
    _served_boot_log(log, edge_bytes=edge_bytes)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "Traceback (most recent call last):\n"
            '  File "/opt/vllm/vllm/v1/engine/core.py", line 88, in run\n'
            "RuntimeError: Engine core initialization failed\n"
        )

    read = read_bringup_log(log, edge_bytes=edge_bytes)
    observation = observe_bringup(server_log=read.text).observation
    assert observation.booted is False
    assert observation.stage_failed is not None


def test_a_short_log_is_still_read_whole(tmp_path: Path) -> None:
    """The middle scan only applies past the two-edge threshold."""
    log = tmp_path / "server.log"
    log.write_text("server_args=Namespace(model='m')\nINFO: Application startup complete.\n", encoding="utf-8")
    read = read_bringup_log(log, edge_bytes=65_536)
    assert "Application startup complete" in read.text
    assert observe_bringup(server_log=read.text).observation.booted is True
