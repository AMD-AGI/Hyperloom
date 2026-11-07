# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the small env/server-args helpers in ``roofline_ceiling``."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import roofline_ceiling as rc


@pytest.mark.parametrize("text", ["benchmark:\n  model: /point\n  envs: nope\n", "benchmark: [\n", "other: {}\n"])
def test_a_malformed_baseline_point_falls_back_to_state(tmp_path: Path, text: str):
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(text, encoding="utf-8")
    state = SimpleNamespace(baseline_config_path=str(cfg), model_path="/state", framework="vllm", tp=2, conc=4)

    runtime = rc.resolve_runtime_workload(state)

    assert (runtime.model_path, runtime.framework, runtime.tp, runtime.concurrency) == ("/state", "vllm", 2, 4)


def test_the_baseline_point_wins_over_state(tmp_path: Path):
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(
        "benchmark:\n  model: /point\n  framework: sglang\n  runner_type: mi355x\n"
        "  envs: {TP: 8, CONC: 16, EXTRA_SGLANG_ARGS: --x}\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(baseline_config_path=str(cfg), model_path="/state", framework="vllm", tp=2, conc=4)

    runtime = rc.resolve_runtime_workload(state, arm="baseline")

    assert (runtime.model_path, runtime.framework, runtime.tp, runtime.concurrency) == ("/point", "sglang", 8, 16)
    assert runtime.server_args == "--x"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 0),
        ("8", 8),
        (8, 8),
        (0, 0),
        (-3, 0),
        ("bad", 0),
    ],
)
def test_env_int(raw, expected):
    assert rc._env_int({"K": raw} if raw is not None else {}, "K") == expected


def test_server_args_from_envs():
    assert rc._server_args_from_envs({}) == ""
    envs = {
        "EXTRA_SGLANG_ARGS": " --tp 8 ",
        "EXTRA_VLLM_ARGS": "",
        "EXTRA_ATOM_ARGS": "--foo",
        "IGNORED": "--bar",
    }
    out = rc._server_args_from_envs(envs)
    assert out == "--tp 8 --foo"


def test_server_args_env_override_and_payload():
    assert rc._server_args_env_override("not-a-dict") == ""
    # extra_envs present but not a dict → "".
    assert rc._server_args_env_override({"extra_envs": "nope"}) == ""
    assert rc._server_args_env_override({"extra_envs": {"EXTRA_VLLM_ARGS": "--x"}}) == "--x"

    assert rc._server_args_payload("nope") == ""
    assert rc._server_args_payload({"extra_args": " --z "}) == "--z"
    assert (
        rc._server_args_payload(
            {"candidate_extra_server_args": "--first", "extra_server_args": "--second"},
        )
        == "--first"
    )

    # On the current_best arm an env override wins over the payload overlay.
    entry = {"tput": 1.0, "extra_envs": {"EXTRA_ATOM_ARGS": "--env"}, "extra_args": "--payload"}
    assert rc._collect_runtime_server_args(SimpleNamespace(current_best=entry)) == "--env"
    assert rc._collect_runtime_server_args(SimpleNamespace(current_best={"tput": 1.0, "extra_args": "--payload"})) == (
        "--payload"
    )


def test_read_baseline_server_args(tmp_path: Path):
    assert rc.read_baseline_server_args(SimpleNamespace()) == ""
    assert rc.read_baseline_server_args(SimpleNamespace(baseline_config_path="")) == ""

    cfg = tmp_path / "materialized.yaml"
    cfg.write_text("benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--tp 8'\n", encoding="utf-8")
    assert rc.read_baseline_server_args(SimpleNamespace(baseline_config_path=str(cfg))) == "--tp 8"


def test_baseline_server_args_follow_the_accepted_anchor_not_the_last_attempt(tmp_path: Path):
    """Promotion enqueues the initial roofline before ``last_baseline`` records the anchor being promoted."""
    failed = tmp_path / "failed.yaml"
    failed.write_text("benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--context-length 6144'\n", encoding="utf-8")
    anchor = tmp_path / "anchor.yaml"
    anchor.write_text(
        "benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--dsa-decode-backend tilelang --context-length 6144'\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(
        baseline_config_path=str(anchor),
        last_baseline={"status": "failed", "extras": {"materialized_config": str(failed)}},
    )

    assert rc.read_baseline_server_args(state) == "--dsa-decode-backend tilelang --context-length 6144"
