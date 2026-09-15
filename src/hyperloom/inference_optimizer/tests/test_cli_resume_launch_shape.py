# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch-shape persistence across ``--resume``."""

from __future__ import annotations

import argparse
import json
import os

from hyperloom.inference_optimizer.cli import (
    DEFAULT_MAX_HOURS,
    _export_operator_launch_shape,
    resolve_leg_max_hours,
)
from hyperloom.inference_optimizer.cli.backends import resolve_robustness_options
from hyperloom.inference_optimizer.cli.bootstrap import parse_operator_extra_env
from hyperloom.orchestrator.state.shared_state import SharedState


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_parse_operator_extra_env_keeps_pairs_and_drops_junk():
    """``NAME=VALUE`` pins survive; entries without ``=`` or with a blank name do not."""
    args = _ns(extra_env=["SGLANG_USE_AITER=0", "EMPTY=", "novalue", "=blank"])
    assert parse_operator_extra_env(args) == {"SGLANG_USE_AITER": "0", "EMPTY": ""}


def test_parse_operator_extra_env_missing_attr_is_empty():
    """A namespace without the flag yields no pins rather than raising."""
    assert parse_operator_extra_env(_ns()) == {}


def test_export_operator_launch_shape_sets_env(monkeypatch):
    """Both handoff variables are projected for downstream in-process executors."""
    # setenv, not delenv: the helper writes os.environ directly, so monkeypatch has to have recorded the pre-test
    # value to undo the write on teardown.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")

    _export_operator_launch_shape(
        server_args="--max-num-seqs 512",
        extra_env={"SGLANG_USE_AITER": "0"},
    )

    assert os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] == "--max-num-seqs 512"
    assert json.loads(os.environ["INFERENCE_OPTIMIZER_EXTRA_ENV"]) == {"SGLANG_USE_AITER": "0"}


def test_export_operator_launch_shape_clears_stale_values(monkeypatch):
    """Empty inputs clear the variables so a second session in the same shell can't inherit them."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--stale")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"STALE":"1"}')

    _export_operator_launch_shape(server_args="", extra_env={})

    assert "INFERENCE_OPTIMIZER_SERVER_ARGS" not in os.environ
    assert "INFERENCE_OPTIMIZER_EXTRA_ENV" not in os.environ


def test_robustness_options_fall_back_to_persisted_on_resume():
    """A resume passing no robustness flag keeps the probe silenced by the original launch."""
    state = SharedState(session_id="s", robustness_options={"auto_probe_inference_server": False})
    args = _ns(nodes=1, framework="vllm", robustness_disable_server_probe=None)

    assert resolve_robustness_options(args, state) == {"auto_probe_inference_server": False}


def test_robustness_options_explicit_flag_wins_over_persisted():
    """``--no-robustness-disable-server-probe`` on the resume re-enables the probe."""
    state = SharedState(session_id="s", robustness_options={"auto_probe_inference_server": False})
    args = _ns(nodes=1, framework="vllm", robustness_disable_server_probe=False)

    assert resolve_robustness_options(args, state) == {"auto_probe_inference_server": True}


def test_robustness_options_unrelated_flag_leaves_the_rest_persisted():
    """One unrelated ``--robustness-*`` flag must not reopen the probe the launch closed."""
    state = SharedState(session_id="s", robustness_options={"auto_probe_inference_server": False})
    args = _ns(nodes=1, framework="vllm", robustness_disable_server_probe=None, robustness_llm_rca=True)

    assert resolve_robustness_options(args, state) == {
        "auto_probe_inference_server": False,
        "llm_rca_enabled": True,
    }


def test_robustness_options_empty_state_is_empty():
    """No flags and nothing persisted leaves the runtime on its own defaults."""
    args = _ns(nodes=1, framework="vllm", robustness_disable_server_probe=None)

    assert resolve_robustness_options(args, SharedState(session_id="s")) == {}


def test_launch_shape_survives_a_state_roundtrip():
    """The fields reach disk, which is what a resume reads them back from."""
    state = SharedState(
        session_id="s",
        operator_server_args="--max-num-seqs 512",
        operator_extra_env={"SGLANG_USE_AITER": "0"},
        nodes=4,
        robustness_options={"auto_probe_inference_server": False},
        warm_replay_enabled=False,
        warm_replay_min_confidence=0.55,
        warm_replay_min_reproduce_pct=0.6,
        bypass_scripts_dir="/scripts",
        framework_repo_path="/fw",
        benchmark_backend="bypass",
    )

    restored = SharedState.from_dict(state.to_dict())

    assert restored.operator_server_args == "--max-num-seqs 512"
    assert restored.operator_extra_env == {"SGLANG_USE_AITER": "0"}
    assert restored.nodes == 4
    assert restored.robustness_options == {"auto_probe_inference_server": False}
    assert restored.warm_replay_enabled is False
    assert restored.warm_replay_min_confidence == 0.55
    assert restored.warm_replay_min_reproduce_pct == 0.6
    assert restored.bypass_scripts_dir == "/scripts"
    assert restored.framework_repo_path == "/fw"
    assert restored.benchmark_backend == "bypass"


def test_pre_existing_state_without_the_fields_loads_defaults():
    """Sessions created before these fields existed resume on the documented defaults, not a crash."""
    restored = SharedState.from_dict({"session_id": "old"})

    assert restored.operator_server_args == ""
    assert restored.operator_extra_env == {}
    assert restored.nodes == 1
    assert restored.robustness_options == {}
    assert restored.warm_replay_enabled is True
    assert restored.warm_replay_min_confidence == 0.7
    assert restored.warm_replay_min_reproduce_pct == 0.8
    assert restored.bypass_scripts_dir == ""
    assert restored.framework_repo_path == ""
    assert restored.benchmark_backend == ""


def test_restore_operator_paths_fills_env_from_state(monkeypatch):
    from hyperloom.inference_optimizer.cli import _restore_operator_supplied_paths_from_state

    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "")
    state = SharedState(
        session_id="s",
        framework_repo_path="/archived/fw",
        bypass_scripts_dir="/archived/scripts",
        benchmark_backend="bypass",
    )
    _restore_operator_supplied_paths_from_state(_ns(framework_path=None, benchmark_scripts_dir=None), state)
    assert os.environ["FRAMEWORK_REPO_PATH"] == "/archived/fw"
    assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == "/archived/scripts"
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"


def test_restore_operator_paths_leaves_env_when_cli_repasses(monkeypatch):
    from hyperloom.inference_optimizer.cli import _restore_operator_supplied_paths_from_state

    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "")
    state = SharedState(
        session_id="s",
        framework_repo_path="/archived/fw",
        bypass_scripts_dir="/archived/scripts",
        benchmark_backend="bypass",
    )
    _restore_operator_supplied_paths_from_state(
        _ns(framework_path="/cli/fw", benchmark_scripts_dir="/cli/scripts"),
        state,
    )
    assert os.environ.get("FRAMEWORK_REPO_PATH", "") == ""
    assert os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "") == ""
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"


def test_a_fresh_run_without_max_hours_takes_the_documented_default():
    assert resolve_leg_max_hours(None, session_max_minutes=None) == DEFAULT_MAX_HOURS


def test_an_explicit_max_hours_decides_the_leg_on_every_path():
    assert resolve_leg_max_hours(6.0, session_max_minutes=None) == 6.0
    # A resume may still tighten: that is the documented behavior and the only
    # way an operator caps one leg of a longer session.
    assert resolve_leg_max_hours(1.0, session_max_minutes=600.0) == 1.0


def test_a_resume_without_max_hours_keeps_the_sessions_own_budget():
    """The defect this closes: the parser default silently spent the grant.

    A session granted 10h through --extend-hours resumed against a 2.0h bound it
    had already burned, so every action was skipped as out of time while the
    banner printed 10h, and the leg exited 0 within seconds.
    """
    assert resolve_leg_max_hours(None, session_max_minutes=600.0) == 10.0


def test_a_resume_of_an_unbounded_session_stays_unbounded():
    assert resolve_leg_max_hours(None, session_max_minutes=0.0) == 0.0


def test_the_parser_leaves_max_hours_unset_so_a_resume_can_tell():
    """Without this the resume path cannot distinguish unset from an explicit 2.0."""
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    args = _build_parser().parse_args(["optimize", "--model", "/m"])
    assert args.max_hours is None
    args = _build_parser().parse_args(["optimize", "--model", "/m", "--max-hours", "2"])
    assert args.max_hours == 2.0
