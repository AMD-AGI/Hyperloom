# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch-shape persistence across ``--resume``."""

from __future__ import annotations

import argparse
import json
import os

from hyperloom.inference_optimizer.cli import _export_operator_launch_shape
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


def test_export_operator_launch_shape_exports_each_pin(monkeypatch):
    """A pin reaches the process environment under its own name, not only inside the JSON blob.

    Every Hyperloom control variable is read with a bare ``os.environ.get``. A pin visible only as
    ``INFERENCE_OPTIMIZER_EXTRA_ENV`` is therefore invisible to all of them, which is how one reader can resolve a
    knob differently from the rest.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "")

    _export_operator_launch_shape(server_args="", extra_env={"HYPERLOOM_AGENTIC_BACKEND": "mlperf"})

    assert os.environ["HYPERLOOM_AGENTIC_BACKEND"] == "mlperf"


def test_exported_pin_is_seen_by_the_bare_env_readers(monkeypatch):
    """The workload the switch selects and the axis the session seeds agree once the pin is global."""
    from hyperloom.common.agentx_workload import MLPERF_CLIENT_SCRIPT, agentx_client_script
    from hyperloom.common.perf_metric import GRADED_OUTPUT
    from hyperloom.inference_optimizer.cli.bootstrap import seed_grading

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "")

    _export_operator_launch_shape(server_args="", extra_env={"HYPERLOOM_AGENTIC_BACKEND": "mlperf"})

    assert agentx_client_script() == MLPERF_CLIENT_SCRIPT
    # MLPerf publishes no per-request OSL/E2EL series, so grading it on interactivity would REVERT every round.
    assert seed_grading("sglang", "agentx")["objective"] == GRADED_OUTPUT


def test_export_operator_launch_shape_unsets_pins_dropped_on_resume(monkeypatch):
    """A resume that drops a pin must clear the name the previous launch exported."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("STALE_PIN", "")
    monkeypatch.setenv("KEPT_PIN", "")

    _export_operator_launch_shape(server_args="", extra_env={"STALE_PIN": "1", "KEPT_PIN": "a"})
    _export_operator_launch_shape(server_args="", extra_env={"KEPT_PIN": "b"})

    assert "STALE_PIN" not in os.environ
    assert os.environ["KEPT_PIN"] == "b"


def test_launch_shape_survives_a_state_roundtrip():
    """The fields reach disk, which is what a resume reads them back from."""
    state = SharedState(
        session_id="s",
        operator_server_args="--max-num-seqs 512",
        operator_extra_env={"SGLANG_USE_AITER": "0"},
        nodes=4,
        warm_replay_enabled=False,
        warm_replay_min_confidence=0.55,
        bypass_scripts_dir="/scripts",
        framework_repo_path="/fw",
        benchmark_backend="bypass",
    )

    restored = SharedState.from_dict(state.to_dict())

    assert restored.operator_server_args == "--max-num-seqs 512"
    assert restored.operator_extra_env == {"SGLANG_USE_AITER": "0"}
    assert restored.nodes == 4
    assert restored.warm_replay_enabled is False
    assert restored.warm_replay_min_confidence == 0.55
    assert restored.bypass_scripts_dir == "/scripts"
    assert restored.framework_repo_path == "/fw"
    assert restored.benchmark_backend == "bypass"


def test_pre_existing_state_without_the_fields_loads_defaults():
    """Sessions created before these fields existed resume on the documented defaults, not a crash."""
    restored = SharedState.from_dict({"session_id": "old"})

    assert restored.operator_server_args == ""
    assert restored.operator_extra_env == {}
    assert restored.nodes == 1
    assert restored.warm_replay_enabled is True
    assert restored.warm_replay_min_confidence == 0.7
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
