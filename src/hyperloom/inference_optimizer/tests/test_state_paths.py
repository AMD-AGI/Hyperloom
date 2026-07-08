# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for session-scoped multi-node state file resolution."""

from __future__ import annotations

import json
import os
import stat

from hyperloom.inference_optimizer.multi_node import state_paths
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR


def test_resolve_state_file_prefers_explicit_env(tmp_path, monkeypatch):
    p = tmp_path / "custom.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    assert state_paths.resolve_state_file() == p


def test_resolve_state_file_uses_session_runtime(tmp_path, monkeypatch):
    session = tmp_path / "sess"
    session.mkdir()
    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))
    assert state_paths.resolve_state_file() == session / "runtime" / "multi_node_state.json"


def test_bind_state_file_migrates_legacy(tmp_path, monkeypatch):
    session = tmp_path / "sess"
    legacy = tmp_path / "legacy_state.json"
    legacy.write_text(json.dumps({"nodes": 2}), encoding="utf-8")
    legacy.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(legacy))
    target = state_paths.bind_state_file_to_session(session)
    assert target == session / "runtime" / "multi_node_state.json"
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["nodes"] == 2
    assert os.environ["MULTI_NODE_STATE_FILE"] == str(target)


def test_state_file_safe_to_read_rejects_world_writable(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{}", encoding="utf-8")
    mode = p.stat().st_mode
    p.chmod(mode | stat.S_IWOTH)
    assert state_paths.state_file_safe_to_read(p) is False
