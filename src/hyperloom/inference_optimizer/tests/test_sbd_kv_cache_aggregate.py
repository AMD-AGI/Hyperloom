# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the session-level KV aggregate in ``telemetry``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.collectors.telemetry import (
    _aggregate_kv_metrics,
    _scan_kv_artifacts,
)
from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_GLOBS


def _artifact(tmp_path: Path, name: str, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "metrics",
        "available": True,
        "aborted": False,
        "capacity_tokens": 32768.0,
        "capacity_gb": 180.013,
        "retract_delta": None,
        "preempt_delta": None,
        "samples": [],
        "engine": "sglang",
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rows(values, phase="measured"):
    return [{"phase": phase, "active_pool_usage": v} for v in values]


def test_no_artifacts_yields_empty_section(tmp_path):
    assert _aggregate_kv_metrics([], []) == {}


def test_occupancy_is_reported_as_time_above_thresholds(tmp_path):
    """The mean hides the episodes that do the damage; the thresholds do not."""
    # 100 samples: 10 in [0.95, 0.99), 5 at or above 0.99, rest low.
    values = [0.10] * 85 + [0.96] * 10 + [0.995] * 5
    art = _artifact(tmp_path, "a.json", samples=_rows(values))

    out = _aggregate_kv_metrics([art], [])

    assert out["time_at_saturation_pct"] == 15.0
    assert out["time_at_retract_band_pct"] == 5.0
    assert out["active_pool_usage_max"] == 0.995
    assert out["measured_samples"] == 100


def test_only_the_measured_phase_counts(tmp_path):
    """Boot has no traffic and eval drives a different request shape."""
    rows = _rows([0.99, 0.99], phase="warmup") + _rows([0.10]) + _rows([0.99], phase="eval")
    art = _artifact(tmp_path, "a.json", samples=rows)

    out = _aggregate_kv_metrics([art], [])

    assert out["measured_samples"] == 1
    assert out["time_at_saturation_pct"] == 0.0


def test_availability_is_tristate(tmp_path):
    unknown = _aggregate_kv_metrics([_artifact(tmp_path, "u.json", available=None)], [])
    assert unknown["available"] is None

    off = _aggregate_kv_metrics([_artifact(tmp_path, "o.json", available=False)], [])
    assert off["available"] is False

    # One round reaching the endpoint is enough to settle the question.
    mixed = _aggregate_kv_metrics(
        [_artifact(tmp_path, "m1.json", available=False), _artifact(tmp_path, "m2.json", available=True)],
        [],
    )
    assert mixed["available"] is True


def test_pressure_counters_stay_separate_and_absent_is_not_zero(tmp_path):
    """Retract and preemption are the same event under two measurement paths."""
    sg = _artifact(tmp_path, "sg.json", retract_delta=48.0)
    out = _aggregate_kv_metrics([sg], [])
    assert out["retract_total"] == 48.0
    assert out["preempt_total"] is None

    vl = _artifact(tmp_path, "vl.json", preempt_delta=458.0, engine="vllm")
    out = _aggregate_kv_metrics([vl], [])
    assert out["preempt_total"] == 458.0
    assert out["retract_total"] is None


def test_rounds_are_summed_and_aborted_ones_are_counted(tmp_path):
    arts = [
        _artifact(tmp_path, "a.json", retract_delta=10.0, samples=_rows([0.5])),
        _artifact(tmp_path, "b.json", retract_delta=5.0, aborted=True, samples=_rows([0.7])),
    ]

    out = _aggregate_kv_metrics(arts, [])

    assert out["rounds"] == 2
    assert out["aborted_rounds"] == 1
    assert out["retract_total"] == 15.0
    assert out["measured_samples"] == 2


def test_capacity_survives_a_round_whose_server_is_already_gone(tmp_path):
    arts = [
        _artifact(tmp_path, "a.json", capacity_tokens=None, capacity_gb=None),
        _artifact(tmp_path, "b.json"),
    ]

    out = _aggregate_kv_metrics(arts, [])

    assert out["capacity_tokens"] == 32768.0
    assert out["capacity_gb"] == 180.013


def test_physical_occupancy_absent_on_vllm_is_none_not_zero(tmp_path):
    art = _artifact(tmp_path, "a.json", engine="vllm", samples=_rows([0.5]))

    out = _aggregate_kv_metrics([art], [])

    assert out["physical_pool_usage_max"] is None
    assert out["active_pool_usage_max"] == 0.5


def test_malformed_artifact_warns_without_losing_the_good_one(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    good = _artifact(tmp_path, "good.json", samples=_rows([0.42]))
    warnings: list[str] = []

    out = _aggregate_kv_metrics([bad, good], warnings)

    assert out["rounds"] == 1
    assert out["active_pool_usage_max"] == 0.42
    assert warnings


def test_scan_finds_artifacts_under_runs(tmp_path):
    nested = tmp_path / "runs" / "explore" / "v00" / "benchmark_sglang"
    nested.mkdir(parents=True)
    (nested / "kv_metrics.json").write_text("{}", encoding="utf-8")

    found = _scan_kv_artifacts(tmp_path)

    assert len(found) == 1
    assert found[0].name == "kv_metrics.json"


def test_scan_on_a_session_without_runs_is_empty(tmp_path):
    assert _scan_kv_artifacts(tmp_path) == []


def test_artifact_is_in_the_package_globs():
    """The artifact lives in the workspace, so it needs an explicit entry."""
    assert "runs/**/kv_metrics.json" in PACKAGE_GLOBS
