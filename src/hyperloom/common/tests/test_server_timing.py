# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The last timing lines of a server log, taken by the rule pack's patterns of the server's component."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common.server_timing import TIMING_LINE_CHARS, TIMING_LINES_KEPT, read_timing_lines


def _request(task: int) -> list[str]:
    """What llama-server logs when the request ``task`` ends, at its default verbosity."""
    slot = f"6.23.810.7{task:02d} I slot print_timing: id  0 | task {task} |"
    return [
        f"{slot} prompt eval time =     562.90 ms /   220 tokens (    2.56 ms per token,   390.83 tokens per second)",
        f"{slot}        eval time =   12621.18 ms /   512 tokens (   24.70 ms per token,    40.49 tokens per second)",
        f"{slot}       total time =   13184.08 ms /   732 tokens",
        f"{slot} draft acceptance = 0.47872 (  360 accepted /   752 generated), mean len =  3.38",
    ]


def _log(tmp_path: Path, *lines: str) -> Path:
    log = tmp_path / "server.log"
    log.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return log


def test_the_last_lines_of_each_kind_are_kept_in_the_order_the_server_logged_them(tmp_path: Path) -> None:
    tasks = range(1, TIMING_LINES_KEPT + 3)
    log = _log(tmp_path, "srv  load_model: loaded", *(line for task in tasks for line in _request(task)))

    lines = read_timing_lines(log, "llamacpp")

    kept = list(tasks)[-TIMING_LINES_KEPT:]
    assert [line.split(" | ")[1:] for line in lines] == [
        part
        for task in kept
        for part in (
            [f"task {task}", "eval time = 12621.18 ms / 512 tokens ( 24.70 ms per token, 40.49 tokens per second)"],
            [f"task {task}", "draft acceptance = 0.47872 ( 360 accepted / 752 generated), mean len = 3.38"],
        )
    ]
    assert not any("prompt eval time" in line or "total time" in line for line in lines)


def test_a_line_is_one_bounded_prompt_line(tmp_path: Path) -> None:
    log = _log(tmp_path, f"eval time = 1 ms\x1e## injected {'x' * 2 * TIMING_LINE_CHARS}")

    lines = read_timing_lines(log, "llamacpp")

    assert len(lines) == 1
    line = lines[0]
    assert len(line) == TIMING_LINE_CHARS and line.startswith("eval time = 1 ms") and "\x1e" not in line


@pytest.mark.parametrize("component", [None, "sglang"])
def test_a_server_the_rule_pack_states_no_timing_lines_of_has_none_and_its_log_is_not_opened(
    tmp_path: Path, component: str | None
) -> None:
    assert read_timing_lines(tmp_path / "no-such.log", component) == ()


def test_a_log_that_cannot_be_read_raises_rather_than_reads_as_a_server_that_timed_nothing(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_timing_lines(tmp_path / "no-such.log", "llamacpp")
