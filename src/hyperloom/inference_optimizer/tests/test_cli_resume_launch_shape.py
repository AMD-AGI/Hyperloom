# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch-shape persistence across ``--resume``.

``--server-args``, ``--extra-env``, ``--nodes`` and the robustness flags are
rebuilt from argv on every boot, so a resume that re-passes none of them takes
them from SharedState instead of dropping to the defaults.
"""

from __future__ import annotations

import argparse
import json
import os

from hyperloom.inference_optimizer.cli import _export_operator_launch_shape
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
    # setenv, not delenv: the helper writes os.environ directly, so monkeypatch
    # has to have recorded the pre-test value to undo the write on teardown.
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
    """One unrelated ``--robustness-*`` flag must not reopen the probe the launch closed.

    Whole-mapping substitution reintroduced this branch's own bug one flag later.
    """
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
    )

    restored = SharedState.from_dict(state.to_dict())

    assert restored.operator_server_args == "--max-num-seqs 512"
    assert restored.operator_extra_env == {"SGLANG_USE_AITER": "0"}
    assert restored.nodes == 4
    assert restored.robustness_options == {"auto_probe_inference_server": False}
    assert restored.warm_replay_enabled is False
    assert restored.warm_replay_min_confidence == 0.55
    assert restored.warm_replay_min_reproduce_pct == 0.6


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
