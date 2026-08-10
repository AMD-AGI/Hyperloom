# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server-less (scriptable) benchmarks must leave the server-log watchdogs off.

``watchdog_active`` / ``stall_active`` are gated on ``bool(server_log_path)``
alone, so passing a ``server.log`` for a framework that never writes one arms
both against a file that can never exist.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk
from hyperloom.orchestrator.actions.executors.baseline import (
    _config_framework,
    _watchdog_server_log_path,
)


def test_scriptable_frameworks_get_no_watchdog_log(tmp_path):
    for framework in ("custom", "xdit"):
        assert _watchdog_server_log_path(tmp_path, framework) is None


def test_serving_frameworks_keep_the_watchdog_log(tmp_path):
    for framework in ("sglang", "vllm", ""):
        assert _watchdog_server_log_path(tmp_path, framework) == str(tmp_path / "server.log")


def test_config_framework_reads_materialized_yaml(tmp_path):
    cfg = tmp_path / "materialized.yaml"
    cfg.write_text(yaml.safe_dump({"benchmark": {"framework": "Custom"}}), encoding="utf-8")
    assert _config_framework(cfg) == "custom"
    assert _config_framework(tmp_path / "missing.yaml") == ""


def _count_scans(monkeypatch) -> dict[str, int]:
    calls = {"scan": 0, "death": 0}

    def _scan(server_log_path, offsets):
        calls["scan"] += 1
        return False, False, False, False

    def _death(path):
        calls["death"] += 1
        return None

    monkeypatch.setattr(sk, "_scan_logs_increment", _scan)
    monkeypatch.setattr(sk, "_server_log_shows_death", _death)
    return calls


def test_no_server_log_leaves_both_watchdogs_disarmed(monkeypatch):
    calls = _count_scans(monkeypatch)

    proc = sk.run_with_session_kill(["bash", "-c", "sleep 1.2"], timeout=30, server_log_path=None)

    assert proc.returncode == 0
    assert calls == {"scan": 0, "death": 0}


def test_server_log_arms_both_watchdogs(monkeypatch, tmp_path: Path):
    calls = _count_scans(monkeypatch)

    proc = sk.run_with_session_kill(
        ["bash", "-c", "sleep 1.2"],
        timeout=30,
        server_log_path=str(tmp_path / "server.log"),
    )

    assert proc.returncode == 0
    assert calls["scan"] > 0
    assert calls["death"] > 0
