# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX KEEP evidence is judged by the session's benchmark mode and the report mapper's verdict, not a backend name."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common.perf_metric import constraint_fields, constraint_veto_reason
from hyperloom.common.round_measurement import round_measurement
from hyperloom.common.tests.test_round_measurement import _agentx, _agentx_result
from hyperloom.inference_optimizer.agentx.report import agentx_measurement
from hyperloom.orchestrator.actions.executors.benchmark_result import is_valid_measurement
from inference_suite.contracts import ProfileResult

_AGENTX_SESSION = {"benchmark_mode": "agentx", "agentx_backend": "inferencex"}


def _state(anchor: dict[str, Any], **overrides: str) -> SimpleNamespace:
    return SimpleNamespace(**{**_AGENTX_SESSION, **overrides}, current_best=dict(anchor))


def _valid() -> dict[str, Any]:
    return round_measurement(_agentx(), None)


def _invalid() -> dict[str, Any]:
    return round_measurement(_agentx(_agentx_result(stopped_by_threshold=True)), None)


def _diagnostic() -> dict[str, Any]:
    return round_measurement(
        _agentx(profile=ProfileResult(trace_dirs=("profile/torch_trace",), has_step_traces=False)), None
    )


def test_a_valid_round_of_the_anchor_workload_is_kept_evidence() -> None:
    measurement = _valid()
    assert constraint_veto_reason(measurement, _state(measurement)) == ""
    assert is_valid_measurement(measurement) is True


@pytest.mark.parametrize("build", [_invalid, _diagnostic], ids=["invalid", "diagnostic"])
def test_an_invalid_or_diagnostic_round_is_vetoed_and_not_a_valid_measurement(build) -> None:
    measurement = build()
    assert measurement["output_throughput"] > 0 and measurement["completed_requests"] > 0
    assert constraint_veto_reason(measurement, _state(_valid())) == "agentx_evidence_invalid"
    assert is_valid_measurement(measurement) is False


def test_a_round_of_another_workload_is_vetoed() -> None:
    anchor = {**_valid(), "workload_identity": "sha256:" + "0" * 64}
    assert constraint_veto_reason(_valid(), _state(anchor)) == "agentx_workload_mismatch"


def test_the_veto_reads_the_fields_every_carrier_copies() -> None:
    carried = constraint_fields(_invalid())
    assert constraint_veto_reason(carried, _state(_valid())) == "agentx_evidence_invalid"


@pytest.mark.parametrize(
    "session", [{"benchmark_mode": "fixed"}, {"agentx_backend": "mlperf"}], ids=["fixed", "mlperf"]
)
def test_a_session_that_is_not_an_agentx_replay_is_not_judged(session: dict[str, str]) -> None:
    assert constraint_veto_reason(_invalid(), _state(_valid(), **session)) == ""


_V1_LAUNCH = {"effective_argv": ["python3", "-m", "sglang.launch_server"], "effective_env": {}}


def _v1(**report: object) -> dict[str, Any]:
    metrics = {"execution_owner": "magpie", "server_launch": _V1_LAUNCH, "recipe": {"tp": 8}, "mode": "canonical"}
    metrics["requests"] = {"successful": 90, "error_rate": 0.0}
    metrics["throughput"] = {"output_tokens_per_second": 900.0, "total_tokens_per_second": 9000.0}
    flags = {"success": True, "benchmark_valid": True, "publishable": True, **report}
    return agentx_measurement({"scenario": "agentx", **flags, "agentx_metrics": metrics})


@pytest.mark.parametrize("backend", ["magpie", "inferencex"])
def test_a_v1_archive_is_judged_by_its_own_reader_on_the_old_magpie_session_too(backend: str) -> None:
    valid = _v1()
    state = _state(valid, agentx_backend=backend)
    assert constraint_veto_reason(valid, state) == ""
    assert is_valid_measurement(valid) is True
    unpublishable = _v1(publishable=False)
    assert constraint_veto_reason(unpublishable, state) == "agentx_evidence_invalid"
    assert is_valid_measurement(unpublishable) is False
