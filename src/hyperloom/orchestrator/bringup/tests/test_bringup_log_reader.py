# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading a chatty server log witnesses every milestone and the root failure, wherever they lie."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors.baseline import read_bringup_log
from hyperloom.orchestrator.bringup import DEGRADED_UNREADABLE, observe_bringup

#: One line per kernel shape, which is what aiter emits on ROCm.
_SHAPE_NOISE = "[aiter] shape is M:1, N:6144, K:4096, not found tuned config, will use default config!\n"
#: A progress bar that repeats a milestone marker once per shard.
_SHARD_PROGRESS = "Loading safetensors checkpoint shards:  42% Completed | 120/284 [00:31<00:44, 3.8it/s]\n"
_NOISE = _SHAPE_NOISE * 2000 + _SHARD_PROGRESS * 300


def _served_boot_log(path: Path) -> None:
    """Write a log whose readiness marker sits hundreds of kilobytes in."""
    head = "server_args=Namespace(model='m')\nLoading weights\nLoading weights took 42s\nGPU blocks: 1024\n"
    path.write_text(
        head + _NOISE + "INFO: Application startup complete.\n" + _NOISE + "INFO: 127.0.0.1 - POST /v1/completions\n",
        encoding="utf-8",
    )


def test_readiness_deep_in_the_log_still_reads_as_booted(tmp_path: Path) -> None:
    """A served boot must not classify as ENGINE_INIT just because the log is long."""
    log = tmp_path / "server.log"
    _served_boot_log(log)

    observation = observe_bringup(server_log=read_bringup_log(log).text).observation
    assert observation.stage_reached is LadderStage.HTTP_READY
    assert observation.booted is True


def test_the_tail_wall_still_wins_over_a_middle_milestone(tmp_path: Path) -> None:
    """A milestone in the middle must not mask a later failure."""
    log = tmp_path / "server.log"
    _served_boot_log(log)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "Traceback (most recent call last):\n"
            '  File "/opt/vllm/vllm/v1/engine/core.py", line 88, in run\n'
            "RuntimeError: Engine core initialization failed\n"
        )

    observation = observe_bringup(server_log=read_bringup_log(log).text).observation
    assert observation.booted is False
    assert observation.stage_failed is not None


def test_a_cause_buried_in_the_middle_outranks_the_tail_consequence(tmp_path: Path) -> None:
    """A worker that dies after readiness is named by its own traceback, not by the engine's re-raise in the tail."""
    log = tmp_path / "server.log"
    worker = "(Worker_TP0 pid=11) ERROR 10-14 11:30:49 [multiproc_executor.py:1053] "
    rejected = 'INFO: 127.0.0.1 - "POST /v1/completions HTTP/1.1" 500 Internal Server Error\n'
    log.write_text(
        "INFO: Application startup complete.\n"
        + _NOISE
        + "(Worker_TP0 pid=11) WARNING 10-14 11:30:40 [x.py:1] Traceback (most recent call last):\n"
        + worker
        + "Traceback (most recent call last):\n"
        + worker
        + '  File "/opt/flydsl/expr/utils/arith.py", line 579, in _to_raw\n'
        + worker
        + "AttributeError: 'int' object has no attribute '_CAPIPtr'\n"
        + rejected * 3000
        + "Traceback (most recent call last):\n"
        + '  File "/opt/vllm/vllm/v1/engine/core_client.py", line 1215, in get_output\n'
        + "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.\n",
        encoding="utf-8",
    )

    observation = observe_bringup(server_log=read_bringup_log(log).text).observation
    assert observation.stage_reached is LadderStage.HTTP_READY
    assert observation.terminal_frame is not None
    assert observation.terminal_frame.exc_type == "AttributeError"
    assert observation.terminal_frame.line == 579


def test_a_missing_log_is_empty_and_an_unreadable_one_is_degraded(tmp_path: Path) -> None:
    """No log written and a log that cannot be served back are different answers."""
    missing = read_bringup_log(tmp_path / "server.log")
    assert (missing.text, missing.degraded) == ("", "")

    unreadable = read_bringup_log(tmp_path)
    assert (unreadable.text, unreadable.degraded) == ("", DEGRADED_UNREADABLE)
