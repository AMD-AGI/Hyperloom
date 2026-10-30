# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The request counts an AgentX round records from Magpie's report: warmup requests are neither asked nor failed."""

from __future__ import annotations

from hyperloom.inference_optimizer.agentx.report import agentx_measurement


def _report(requests: dict) -> dict:
    return {
        "success": True,
        "benchmark_valid": True,
        "publishable": True,
        "agentx_metrics": {"execution_owner": "magpie", "requests": requests, "throughput": {}, "latency_seconds": {}},
    }


def test_warmup_requests_are_not_counted_as_requested() -> None:
    # Magpie's report of a run with 354 warmup records, 3953 profiled and no error (hyperloom-multibranch #159).
    requests = {"total": 4307, "records_total": 4307, "profiled_total": 3953, "successful": 3953, "errors": 0}
    measurement = agentx_measurement(_report({**requests, "warmup_dropped": 354, "error_rate": 0.0}))

    assert measurement["requested_requests"] == 3953
    assert measurement["requested_requests"] - measurement["completed_requests"] == 0


def test_a_profiled_request_that_failed_is_still_counted_as_requested() -> None:
    requests = {"total": 110, "profiled_total": 100, "successful": 97, "errors": 3, "warmup_dropped": 10}
    measurement = agentx_measurement(_report({**requests, "error_rate": 0.03}))

    assert measurement["requested_requests"] - measurement["completed_requests"] == 3
