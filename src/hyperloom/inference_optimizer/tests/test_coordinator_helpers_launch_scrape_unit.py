# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for launch-flag and config-blob helper primitives."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.launch_log_evidence import launch_argv_from_log, split_launch_flags
from hyperloom.orchestrator.loop import coordinator_helpers as ch


# ── _split_env_and_flags ──────────────────────────────────────────────────


def test_split_env_and_flags_mixed_tokens() -> None:
    # Only ``-``-prefixed tokens land in ``flags``; a bare value token following
    # a space-form flag is dropped (equals-form is the only way a value survives).
    envs, flags = ch._split_env_and_flags("FOO=1 BAR=baz --chunked-prefill-size=2048 --disable-radix-cache")
    assert envs == {"FOO": "1", "BAR": "baz"}
    assert flags == "--chunked-prefill-size=2048 --disable-radix-cache"


def test_split_env_and_flags_empty_input() -> None:
    assert ch._split_env_and_flags("") == ({}, "")
    assert ch._split_env_and_flags(None) == ({}, "")


def test_split_env_and_flags_only_env() -> None:
    envs, flags = ch._split_env_and_flags("A=1 B=2")
    assert envs == {"A": "1", "B": "2"}
    assert flags == ""


def test_split_env_and_flags_only_flags() -> None:
    envs, flags = ch._split_env_and_flags("--flag-a --flag-b=1")
    assert envs == {}
    assert flags == "--flag-a --flag-b=1"


def test_split_env_and_flags_falls_back_on_shlex_error() -> None:
    # An unbalanced quote makes shlex.split raise, so the ``.split()`` fallback
    # runs; the unterminated token starts with "-" and lands in ``flags``.
    envs, flags = ch._split_env_and_flags('FOO=1 --flag="unterminated')
    assert envs["FOO"] == "1"
    assert flags == '--flag="unterminated'


# ── _geak_sweep_measured_tput ─────────────────────────────────────────────


def test_geak_sweep_measured_tput_prefers_the_promotion_measurement() -> None:
    res = {
        "promotion_measurement": {"output_throughput": 150.0},
        "points": [{"status": "succeeded", "output_throughput": 999.0}],
    }
    assert ch._geak_sweep_measured_tput(res) == 150.0


def test_geak_sweep_measured_tput_falls_back_to_the_points() -> None:
    res = {
        "promotion_measurement": {},
        "points": [
            {"status": "failed", "output_throughput": 10.0},
            {"status": "succeeded", "output_throughput": 77.5},
        ],
    }
    assert ch._geak_sweep_measured_tput(res) == 77.5


def test_geak_sweep_measured_tput_none_when_not_dict() -> None:
    assert ch._geak_sweep_measured_tput(None) is None
    assert ch._geak_sweep_measured_tput([]) is None  # type: ignore[arg-type]


def test_geak_sweep_measured_tput_none_when_no_positive_throughput() -> None:
    res = {
        "promotion_measurement": {"output_throughput": 0},
        "points": [{"status": "succeeded", "output_throughput": -1}],
    }
    assert ch._geak_sweep_measured_tput(res) is None


# ── split_launch_flags ────────────────────────────────────────────────────


def test_split_launch_flags_strips_run_specific_space_form() -> None:
    argv = "--model-path /models/x --tensor-parallel-size 8 --mem-fraction-static 0.9"
    assert split_launch_flags(argv) == "--mem-fraction-static 0.9"


def test_split_launch_flags_strips_equals_form() -> None:
    argv = "--host=0.0.0.0 --port=30000 --disable-radix-cache"
    assert split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_strips_profiling_flags() -> None:
    argv = "--enable-profile --chunked-prefill-size 2048"
    assert split_launch_flags(argv) == "--chunked-prefill-size 2048"


def test_split_launch_flags_handles_valueless_run_specific_flag() -> None:
    # ``--pid`` followed by another flag: the run-specific flag is dropped
    # without eating the next flag.
    argv = "--pid --disable-radix-cache"
    assert split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_falls_back_on_shlex_error() -> None:
    out = split_launch_flags('--mem-fraction-static 0.9 "unterminated')
    assert "--mem-fraction-static" in out


# ── launch_argv_from_log ──────────────────────────────────────────────────


def test_launch_argv_from_log_extracts_and_strips(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "some preamble\n"
        "+ python3 -m sglang.launch_server --model-path /models/x "
        "--tensor-parallel-size 8 --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "sglang")
    assert flags == "--mem-fraction-static 0.9"


def test_launch_argv_from_log_returns_empty_when_marker_absent(
    tmp_path: Path,
) -> None:
    log = tmp_path / "server.log"
    log.write_text("no engine launch here\n", encoding="utf-8")
    assert launch_argv_from_log(str(log), "sglang") == ""


def test_launch_argv_from_log_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert launch_argv_from_log(str(tmp_path / "nope.log"), "sglang") == ""


def test_launch_argv_from_log_returns_empty_for_unmarked_framework(
    tmp_path: Path,
) -> None:
    # A framework with no registered argv marker never reads the log.
    log = tmp_path / "server.log"
    log.write_text(
        "+ python3 -m sglang.launch_server --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    assert launch_argv_from_log(str(log), "xdit") == ""
    assert launch_argv_from_log(str(log), "") == ""


def test_launch_argv_from_log_falls_back_to_double_dash_scan(
    tmp_path: Path,
) -> None:
    # No regex match, but the line has a "--" run after the marker → the
    # ``line.find("--")`` fallback path is exercised.
    log = tmp_path / "server.log"
    log.write_text(
        "vllm serve --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "vllm")
    assert "--mem-fraction-static 0.9" in flags
