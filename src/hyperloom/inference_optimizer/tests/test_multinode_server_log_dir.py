# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RayJob per-rank server logs must land on a cross-node-visible directory.

The launch driver runs on the head; the decode leg runs on another pod. A
node-local /tmp log dir hides decode_0.log from the driver's health-wait, so its
fatal-log fast-fail never fires and its failure tail reads bytes=0. These lock
the resolution that keeps the logs on a shared filesystem instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node.cli import _multinode_server_log_dir


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every env var the resolver consults, for an isolated start."""
    for key in ("HYPERLOOM_MN_SERVER_LOG_DIR", "MULTI_NODE_STATE_FILE", "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"):
        monkeypatch.delenv(key, raising=False)


def test_explicit_shared_dir_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An absolute HYPERLOOM_MN_SERVER_LOG_DIR is honored (matches infera)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "runtime" / "multi_node_state.json"))
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "/wekafs/shared/server_logs")

    assert _multinode_server_log_dir() == "/wekafs/shared/server_logs"


def test_explicit_dir_expands_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``$VAR`` in the override expands, so the pod writes an absolute path."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "runtime" / "multi_node_state.json"))
    monkeypatch.setenv("USER_DATA_PATH", "/wekafs/users/abc")
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "$USER_DATA_PATH/server_logs")

    assert _multinode_server_log_dir() == "/wekafs/users/abc/server_logs"


def test_unresolved_or_relative_override_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-absolute or still-``$``-carrying override falls back to the session dir."""
    _clear_env(monkeypatch)
    session = tmp_path / "sess"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    # $UNSET_VAR does not expand -> stays literal with a '$' -> rejected.
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "$UNSET_VAR/server_logs")

    assert _multinode_server_log_dir() == str(session / "runtime" / "server_logs")


def test_defaults_to_session_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without an override, logs go to the session's shared runtime dir.

    This is unique per run, so concurrent runs' rank logs cannot collide the way
    infera's fixed ``$USER_DATA_PATH/server_logs`` default can.
    """
    _clear_env(monkeypatch)
    session = tmp_path / "sess"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))

    assert _multinode_server_log_dir() == str(session / "runtime" / "server_logs")


def test_state_file_env_also_anchors_the_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit state-file path anchors server_logs beside its runtime dir."""
    _clear_env(monkeypatch)
    state_file = tmp_path / "runtime" / "multi_node_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(state_file))

    assert _multinode_server_log_dir() == str(state_file.parent / "server_logs")


def test_falls_back_to_tmp_when_no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dev/single box with no shared FS configured keeps working on /tmp."""
    _clear_env(monkeypatch)

    result = _multinode_server_log_dir()

    assert result.endswith("multi_node_logs")
    assert Path(result).is_absolute()
