# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Forge E2E PR report."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_REPORT_SCRIPT = _ROOT / ".github" / "scripts" / "forge_e2e_report.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("forge_e2e_report", _REPORT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load_report()


def test_extracts_the_last_machine_readable_result_from_logs() -> None:
    payload = {
        "logs": [
            {"message": 'noise __FORGE_RESULT__{"mean_case_speedup":1.01,"best_ms":0.2}'},
            {"message": 'prefix __FORGE_RESULT__{"mean_case_speedup":1.044164,"best_ms":0.0143} suffix'},
        ]
    }

    assert report.extract_forge_result(payload) == {
        "mean_case_speedup": 1.044164,
        "best_ms": 0.0143,
    }


def test_success_report_includes_timing_and_effect_metrics() -> None:
    detail = {
        "orchestration": {
            "conditions": [
                {"phase": "Queued", "time": "2026-08-31T10:56:35Z"},
                {"phase": "Dispatched", "time": "2026-08-31T10:57:05Z"},
                {"phase": "Succeeded", "time": "2026-08-31T11:36:58Z"},
            ]
        }
    }
    forge_result = {
        "pristine_baseline_ms": 0.0154,
        "best_ms": 0.0143,
        "mean_case_speedup": 1.044163719421481,
        "checkpoint": {"validation_passed": True, "snr_db": 103.68},
        "iteration_count": 3,
        "best_iteration": 2,
        "best_commit": "c85437bb886bc553bdad9aa0f31e2ff67467be54",
        "llm_usage": {"calls": 20, "total_cost_usd": 22.622115},
    }

    body = report.render_report(
        result_label="✅ Succeeded",
        detail=detail,
        forge_result=forge_result,
        max_hours="1.0",
        max_iters="100",
        gpus="1",
        workspace="control-plan-hyperloom-ci",
        head_ref="ci/forge-e2e-path-gate",
        head_sha="f8fb6e2d",
        session_id="d14d455d",
        details_url="https://example.test/run",
    )

    assert "| queue → dispatch | 30s |" in body
    assert "| run time | 39m 53s |" in body
    assert "| total | 40m 23s |" in body
    assert "| baseline → best | 0.0154 ms → 0.0143 ms |" in body
    assert "| speedup | **1.044164x** (+4.42%) |" in body
    assert "| validation | PASS (SNR 103.7 dB) |" in body
    assert "| iterations | 3 (best at iteration 2) |" in body
    assert "LLM usage" not in body
    assert "$22.62" not in body


def test_missing_logs_degrade_to_timing_only() -> None:
    body = report.render_report(
        result_label="❌ Failed",
        detail={},
        forge_result=None,
        max_hours="1.0",
        max_iters="100",
        gpus="1",
        workspace="ci",
        head_ref="branch",
        head_sha="deadbeef",
        session_id="session",
        error="platform failed",
        now=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    assert "| queue → dispatch | – |" in body
    assert "| baseline → best |" not in body
    assert "| reason | platform failed |" in body
