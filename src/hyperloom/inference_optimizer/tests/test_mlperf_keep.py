# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke-vs-full KEEP gate for the MLPerf AgentX backend."""

from __future__ import annotations

import json

from hyperloom.orchestrator.actions.executors._mlperf_keep import (
    clone_variant_for_mlperf_full,
    mlperf_full_keep_block,
    should_validate_mlperf_full,
)
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant


def test_should_validate_only_on_mlperf(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    assert should_validate_mlperf_full() is False
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    assert should_validate_mlperf_full() is True


def test_clone_variant_pins_full_613():
    gv = GridVariant(name="x", extra_envs={"MLPERF_AGENTIC_FLOW": "smoke_test"})
    full = clone_variant_for_mlperf_full(gv)
    assert full.extra_envs["MLPERF_AGENTIC_FLOW"] == "full"
    assert full.extra_envs["AGENTIC_NUM_TRAJECTORIES"] == "613"
    assert full.name.endswith("-mlperf-full")


def _write_mapped(tmp_path, **over):
    payload = {
        "mlperf_complete": True,
        "submission_valid": True,
        "request_error_rate": 0.0,
        "accuracy_score": 0.9,
        "output_throughput": 1000.0,
        "e2e_norm_intvty_p90": 80.0,
    }
    payload.update(over)
    path = tmp_path / "inferencex_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_full_keep_pass(tmp_path):
    assert mlperf_full_keep_block(_write_mapped(tmp_path), status="succeeded") == ""


def test_full_keep_incomplete(tmp_path):
    reason = mlperf_full_keep_block(
        _write_mapped(tmp_path, mlperf_complete=False, submission_valid=False),
        status="succeeded",
    )
    assert reason.startswith("mlperf_full_incomplete")


def test_full_keep_error_rate(tmp_path):
    reason = mlperf_full_keep_block(
        _write_mapped(tmp_path, request_error_rate=25.0),
        status="succeeded",
    )
    assert reason == "mlperf_full_error_rate"


def test_full_keep_accuracy_missing(tmp_path):
    reason = mlperf_full_keep_block(
        _write_mapped(tmp_path, accuracy_score=None),
        status="succeeded",
    )
    assert reason == "mlperf_full_accuracy_unavailable"


def test_full_keep_failed_status(tmp_path):
    assert mlperf_full_keep_block(_write_mapped(tmp_path), status="failed") == "failed"
