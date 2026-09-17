# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for launch-flag and config-blob helper primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common.launch_log_evidence import launch_argv_from_log, split_launch_flags
from hyperloom.orchestrator.loop import coordinator_helpers as ch


# ── _split_env_and_flags ──────────────────────────────────────────────────


def test_split_env_and_flags_mixed_tokens() -> None:
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
    # An unbalanced quote makes shlex.split raise, so the ``.split()`` fallback runs; the unterminated token starts
    # with "-" and lands in ``flags``.
    envs, flags = ch._split_env_and_flags('FOO=1 --flag="unterminated')
    assert envs["FOO"] == "1"
    assert flags == '--flag="unterminated'


def test_accepted_config_uses_published_env_map_for_revalidation() -> None:
    from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    config = {
        "flags": "--mem-fraction-static 0.95",
        "env": 'SGLANG_USE_AITER=1 RUN_EVAL=true; BACKEND=sglang; EXTRA_ENV=")." --discarded-prose',
        "env_map": {"SGLANG_USE_AITER": "1", "RUN_EVAL": "true", "BACKEND": "sglang"},
        "env_unparsed": ["EXTRA_ENV=).", "--discarded-prose"],
    }
    flags, envs = KernelPhase._parse_geak_accepted_config({"accepted_config": config})
    variant = GridVariant(name="geak_revalidate", extra_server_args=flags, extra_envs=envs)
    assert variant.extra_server_args == config["flags"]
    assert variant.extra_envs == config["env_map"]
    assert variant.extra_envs["RUN_EVAL"] == "true"
    assert not ch._geak_result_has_material({"accepted_config": config}, prev_best_flags=flags, prev_best_envs=envs)


def test_empty_accepted_env_map_does_not_restore_raw_assignments() -> None:
    config = {"env": "RUN_EVAL=false SGLANG_USE_AITER=1 --disable-cuda-graph", "env_map": {}}
    assert ch._accepted_config_as_variant(config) == ("--disable-cuda-graph", {})
    assert ch._geak_result_has_material({"accepted_config": config}, prev_best_envs={"RUN_EVAL": "true"})


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("unparsed", [False, True])
def test_legacy_flags_and_discarded_text_are_reported(caplog, mapped, unparsed):
    config = {"env": "SGLANG_USE_AITER=1 --disable-cuda-graph"}
    if mapped:
        config["env_map"] = {}
    if unparsed:
        config["env_unparsed"] = ["discarded prose"]
    assert ch._accepted_config_as_variant(config) == (
        "--disable-cuda-graph",
        {} if mapped else {"SGLANG_USE_AITER": "1"},
    )
    assert "retaining them alongside accepted_config.flags" in caplog.text
    assert ("reports discarded source text" in caplog.text) == unparsed
    assert "using only validated env_map" not in caplog.text


@pytest.mark.parametrize("artifact", ["accepted_kernels", "accepted_heads", "final_overlay", "final_patch"])
def test_invalid_config_does_not_hide_artifact_materiality(artifact):
    assert ch._geak_result_has_material({artifact: ["product"], "accepted_config": {"env_map": None}})


def test_invalid_config_is_not_reported_as_no_material():
    with pytest.raises(ValueError, match="env_map"):
        ch._geak_result_has_material({"accepted_config": {"env_map": None}})


@pytest.mark.parametrize("removals", [None, [], ["--disable-radix-cache"]])
def test_complete_config_materiality_inherits_omitted_removals(removals):
    config = {"flags": "--mem-fraction-static 0.95", "env_map": {}, "args_mode": "replace"}
    prior = {"args_mode": "replace", "remove_args": ["--disable-radix-cache"]}
    if removals is not None:
        config["remove_args"] = removals
    assert ch._geak_result_has_material(
        {"accepted_config": config},
        prev_best_flags=config["flags"],
        prev_best_controls=prior,
    ) == (removals == [])


def test_accepted_env_map_preserves_values_and_filters_loader_keys() -> None:
    value = '{"path": "a b", "pattern": "x=y;z"}'
    config = {"env_map": {"SGLANG_TEST_CONFIG": value, "PYTHONPATH": "/untrusted"}}
    assert ch._accepted_config_as_variant(config) == ("", {"SGLANG_TEST_CONFIG": value})


@pytest.mark.parametrize("env_map", [None, "RUN_EVAL=true", [], {"RUN_EVAL": True}, {1: "x"}, {"bad-key": "1"}])
def test_invalid_accepted_env_map_does_not_fall_back_to_raw_string(env_map) -> None:
    with pytest.raises(ValueError, match="env_map must map strings to strings"):
        ch._accepted_config_as_variant({"env": "RUN_EVAL=true;", "env_map": env_map})


def test_legacy_accepted_config_still_parses_env_and_flags() -> None:
    assert ch._accepted_config_as_variant(
        {"flags": "--mem-fraction-static 0.95", "env": "SGLANG_USE_AITER=1 --disable-radix-cache"}
    ) == ("--mem-fraction-static 0.95 --disable-radix-cache", {"SGLANG_USE_AITER": "1"})


@pytest.mark.parametrize("mapped", [False, True])
def test_legacy_flags_keep_values_and_literal_json(mapped):
    import shlex

    value = '{"path": "a b", "pattern": "x=y"}'
    config = {"env": shlex.join(["SGLANG_USE_AITER=1", "--limit", "64", "--config", value])}
    if mapped:
        config["env_map"] = {}
    flags, envs = ch._accepted_config_as_variant(config)
    assert shlex.split(flags) == ["--limit", "64", "--config", value]
    assert envs == ({} if mapped else {"SGLANG_USE_AITER": "1"})


# ── _geak_sweep_measured_tput ─────────────────────────────────────────────


def test_geak_sweep_measured_tput_prefers_the_promotion_measurement() -> None:
    res = {
        "promotion_measurement": {"output_throughput": 150.0},
        "points": [{"status": "succeeded", "output_throughput": 999.0}],
    }
    assert ch._geak_sweep_measured_tput(res) == 150.0


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
    # ``--pid`` followed by another flag: the run-specific flag is dropped without eating the next flag.
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
    # No regex match, but the line has a "--" run after the marker → the ``line.find("--")`` fallback path is
    # exercised.
    log = tmp_path / "server.log"
    log.write_text(
        "vllm serve --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "vllm")
    assert "--mem-fraction-static 0.9" in flags
