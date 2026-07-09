# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the launch-flag / config-blob helpers in
``coordinator_helpers`` that back the GEAK handoff and resume/revalidation
paths: ``_split_env_and_flags``, ``_geak_sweep_measured_tput``,
``_split_launch_flags``, ``_launch_argv_from_log``, and
``_scrape_resolved_launch_flags``."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.loop import coordinator_helpers as ch


# ── _split_env_and_flags ──────────────────────────────────────────────────


def test_split_env_and_flags_mixed_tokens() -> None:
    # Only ``-``-prefixed tokens land in ``flags``; a bare (non-``=``) value
    # token following a space-form flag is neither an env assignment nor a
    # flag token itself, so it is dropped (equals-form is the only way a
    # flag's value survives the split).
    envs, flags = ch._split_env_and_flags(
        "FOO=1 BAR=baz --chunked-prefill-size=2048 --disable-radix-cache"
    )
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
    # An unbalanced quote makes shlex.split raise ValueError -> the plain
    # ``.split()`` fallback path is exercised instead of raising. The
    # unterminated token still starts with "-", so it lands in ``flags``.
    envs, flags = ch._split_env_and_flags('FOO=1 --flag="unterminated')
    assert envs["FOO"] == "1"
    assert flags == '--flag="unterminated'


# ── _geak_sweep_measured_tput ─────────────────────────────────────────────


def test_geak_sweep_measured_tput_prefers_best_for_each_conc() -> None:
    res = {
        "best_for_each_conc": {"32": {"output_throughput": 150.0}},
        "sweep_grid": [{"status": "succeeded", "output_throughput": 999.0}],
    }
    assert ch._geak_sweep_measured_tput(res) == 150.0


def test_geak_sweep_measured_tput_falls_back_to_sweep_grid() -> None:
    res = {
        "best_for_each_conc": {},
        "sweep_grid": [
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
        "best_for_each_conc": {"32": {"output_throughput": 0}},
        "sweep_grid": [{"status": "succeeded", "output_throughput": -1}],
    }
    assert ch._geak_sweep_measured_tput(res) is None


# ── _split_launch_flags ───────────────────────────────────────────────────


def test_split_launch_flags_strips_run_specific_space_form() -> None:
    argv = "--model-path /models/x --tensor-parallel-size 8 --mem-fraction-static 0.9"
    assert ch._split_launch_flags(argv) == "--mem-fraction-static 0.9"


def test_split_launch_flags_strips_equals_form() -> None:
    argv = "--host=0.0.0.0 --port=30000 --disable-radix-cache"
    assert ch._split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_strips_profiling_flags() -> None:
    argv = "--enable-profile --chunked-prefill-size 2048"
    assert ch._split_launch_flags(argv) == "--chunked-prefill-size 2048"


def test_split_launch_flags_handles_valueless_run_specific_flag() -> None:
    # ``--pid`` immediately followed by another flag (not a value) => the
    # run-specific flag itself is dropped without also eating the next flag.
    argv = "--pid --disable-radix-cache"
    assert ch._split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_falls_back_on_shlex_error() -> None:
    out = ch._split_launch_flags('--mem-fraction-static 0.9 "unterminated')
    assert "--mem-fraction-static" in out


# ── _launch_argv_from_log ─────────────────────────────────────────────────


def test_launch_argv_from_log_extracts_and_strips(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "some preamble\n"
        "+ python3 -m sglang.launch_server --model-path /models/x "
        "--tensor-parallel-size 8 --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = ch._launch_argv_from_log(str(log), "launch_server")
    assert flags == "--mem-fraction-static 0.9"


def test_launch_argv_from_log_returns_empty_when_marker_absent(
    tmp_path: Path,
) -> None:
    log = tmp_path / "server.log"
    log.write_text("no engine launch here\n", encoding="utf-8")
    assert ch._launch_argv_from_log(str(log), "launch_server") == ""


def test_launch_argv_from_log_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert ch._launch_argv_from_log(str(tmp_path / "nope.log"), "launch_server") == ""


def test_launch_argv_from_log_falls_back_to_double_dash_scan(
    tmp_path: Path,
) -> None:
    # No regex match against the marker pattern, but the line still contains
    # a "--" run of flags after the marker text -> the ``line.find("--")``
    # fallback path is exercised.
    log = tmp_path / "server.log"
    log.write_text(
        "vllm serve --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = ch._launch_argv_from_log(str(log), "vllm")
    assert "--mem-fraction-static 0.9" in flags


# ── _scrape_resolved_launch_flags ─────────────────────────────────────────


def _write_bench(
    runs_root: Path, name: str, tput: float | None, marker: str = "launch_server"
) -> Path:
    bench_dir = runs_root / name
    bench_dir.mkdir(parents=True)
    if tput is not None:
        (bench_dir / "inferencex_result.json").write_text(
            json.dumps({"output_throughput": tput}), encoding="utf-8"
        )
    (bench_dir / "server.log").write_text(
        f"+ python3 -m sglang.{marker} --model-path /models/x "
        f"--tensor-parallel-size 8 --chunked-prefill-size 2048\n",
        encoding="utf-8",
    )
    return bench_dir


def test_scrape_resolved_launch_flags_unknown_backend_returns_empty(
    tmp_path: Path,
) -> None:
    assert ch._scrape_resolved_launch_flags(tmp_path, "unknown-backend", 100.0) == ""


def test_scrape_resolved_launch_flags_matches_by_throughput(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_bench(runs_root, "winner", 123.4)
    _write_bench(runs_root, "loser", 50.0)

    flags = ch._scrape_resolved_launch_flags(tmp_path, "sglang", 123.4)
    assert flags == "--chunked-prefill-size 2048"


def test_scrape_resolved_launch_flags_skips_geak_and_overlay_dirs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    # A "geak"-tagged dir happens to match the target throughput exactly, but
    # must be excluded from BOTH the throughput-match and the recency
    # fallback scan, leaving nothing to scrape.
    _write_bench(runs_root, "geak_replay", 200.0)

    flags = ch._scrape_resolved_launch_flags(tmp_path, "sglang", 200.0)
    assert flags == ""


def test_scrape_resolved_launch_flags_prefers_matched_over_other_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    # A "geak"-tagged dir matches the throughput but is excluded; the
    # genuine orchestrator run dir is the one actually scraped.
    _write_bench(runs_root, "geak_replay", 200.0)
    real_dir = _write_bench(runs_root, "orchestrator_run", 200.0)
    (real_dir / "server.log").write_text(
        "+ python3 -m sglang.launch_server --model-path /models/x "
        "--chunked-prefill-size 8192\n",
        encoding="utf-8",
    )

    flags = ch._scrape_resolved_launch_flags(tmp_path, "sglang", 200.0)
    assert flags == "--chunked-prefill-size 8192"


def test_scrape_resolved_launch_flags_falls_back_to_most_recent(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _write_bench(runs_root, "only_run", 999.0)

    # target_tput<=0 => skip throughput matching, go straight to the
    # most-recent-clean-launch fallback.
    flags = ch._scrape_resolved_launch_flags(tmp_path, "sglang", 0.0)
    assert flags == "--chunked-prefill-size 2048"


def test_scrape_resolved_launch_flags_no_runs_dir_returns_empty(
    tmp_path: Path,
) -> None:
    assert ch._scrape_resolved_launch_flags(tmp_path, "sglang", 100.0) == ""


def test_scrape_resolved_launch_flags_tolerates_corrupt_result_json(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    bench_dir = runs_root / "corrupt"
    bench_dir.mkdir(parents=True)
    (bench_dir / "inferencex_result.json").write_text("{not-json", encoding="utf-8")
    (bench_dir / "server.log").write_text(
        "+ python3 -m sglang.launch_server --model-path /models/x "
        "--chunked-prefill-size 4096\n",
        encoding="utf-8",
    )

    # The corrupt result is skipped for throughput matching; falls back to
    # the recency scan, which still finds the same log's flags.
    flags = ch._scrape_resolved_launch_flags(tmp_path, "sglang", 55.0)
    assert flags == "--chunked-prefill-size 4096"
