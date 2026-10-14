# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX baseline measured with the prefix cache off is refused, whichever launcher started its server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _agentx_prefix_cache as pc
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.state.shared_state import SharedState


def _measured(
    tmp_path: Path,
    *,
    hit: float | None,
    radix_off: bool | None,
    cached: list[int] | None,
    hit_at: str = "raw",
) -> dict[str, Any]:
    """A succeeded measurement whose theoretical hit sits where one launch path or another records it."""
    raw_doc: dict[str, Any] = {}
    if hit is not None and hit_at == "raw":
        raw_doc = {"theoretical_prefix_cache_hit": hit}
    elif hit is not None and hit_at == "aggregate":
        raw_doc = {"request_metrics": {"cache": {"theoretical_cache_hit_rate": hit}}}
    raw = tmp_path / "result.json"
    raw.write_text(json.dumps(raw_doc), encoding="utf-8")
    log = tmp_path / "server.log"
    log.write_text(
        "".join(
            f"Prefill batch. #new-seq: 1, #new-token: 8192, #cached-token: {n}, token usage: 0.01\n" for n in cached
        )
        if cached is not None
        else "server up\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "status": "succeeded",
        "output_throughput": 221.8,
        "raw_result_path": str(raw),
        "server_log_path": str(log),
        "launch_evidence": {"observed_server_config": {} if radix_off is None else {"disable_radix_cache": radix_off}},
    }
    if hit is not None and hit_at == "measurement":
        result["theoretical_prefix_cache_hit"] = hit
    return result


@pytest.mark.parametrize(
    ("hit", "radix_off", "cached", "refused"),
    [
        # The observed server config alone settles it, with the hit in percent or as a fraction.
        (96.8, True, None, True),
        (0.975, True, None, True),
        # The server log alone settles it: every prefill batch reused nothing.
        (96.8, None, [0, 0, 0], True),
        (96.8, False, [0, 0], True),
        # Any reused token clears it.
        (96.8, False, [0, 4096], False),
        # Nothing readable about the cache: left alone.
        (96.8, None, None, False),
        # Below the prefix-heavy line, or no hit reported at all (the synthetic workload): left alone.
        (12.0, True, [0], False),
        (0.12, True, [0], False),
        (49.9, True, [0], False),
        (None, True, [0], False),
    ],
)
def test_prefix_cache_off_evidence(tmp_path, hit, radix_off, cached, refused):
    measured = _measured(tmp_path, hit=hit, radix_off=radix_off, cached=cached)
    assert bool(pc.prefix_cache_off_evidence(measured)) is refused


def test_the_threshold_is_inclusive(tmp_path):
    assert pc.prefix_cache_off_evidence(_measured(tmp_path, hit=50.0, radix_off=True, cached=None))
    assert pc.prefix_cache_off_evidence(_measured(tmp_path, hit=0.5, radix_off=True, cached=None))


@pytest.mark.parametrize("hit_at", ["raw", "aggregate", "measurement"])
@pytest.mark.parametrize("hit", [96.8, 0.975])
def test_the_hit_is_read_wherever_the_launch_path_records_it(tmp_path, hit_at, hit):
    assert pc.prefix_cache_off_evidence(_measured(tmp_path, hit=hit, radix_off=True, cached=None, hit_at=hit_at))
    low = hit / 8
    assert not pc.prefix_cache_off_evidence(_measured(tmp_path, hit=low, radix_off=True, cached=None, hit_at=hit_at))


def test_the_measurement_field_wins_over_the_raw_file(tmp_path):
    measured = _measured(tmp_path, hit=96.8, radix_off=True, cached=None)
    measured["theoretical_prefix_cache_hit"] = 0.12
    assert pc.prefix_cache_off_evidence(measured) == ""


def _run_baseline(
    result: dict[str, Any], *, kind: str = "baseline", params: dict[str, Any] | None = None
) -> tuple[dict[str, Any], SharedState]:
    state = SharedState(enablement_mode="off")
    executor = BaselineExecutor()

    async def _measured_round(ctx, *, recorder=None):
        return result

    executor._run_retrying = _measured_round
    task = SimpleNamespace(task_id="t-bl", kind=kind, params={"framework": "sglang", **(params or {})})
    ctx = SimpleNamespace(task=task, extra={"shared_state": state})
    return asyncio.run(executor._run(ctx)), state


@pytest.mark.parametrize(
    ("radix_off", "cached", "says"),
    [(True, None, "disable_radix_cache=True"), (None, [0, 0], "no prefill batch reused a cached token")],
)
def test_a_baseline_measured_with_the_cache_off_is_refused_and_stops_the_run(tmp_path, radix_off, cached, says):
    result, state = _run_baseline(_measured(tmp_path, hit=96.8, radix_off=radix_off, cached=cached))
    assert result["status"] == "failed"
    assert result["error_class"] == pc.PREFIX_CACHE_OFF_ERROR_CLASS
    assert says in result["error"]
    assert state.stop_reason == "baseline_failed"


def test_a_baseline_that_reused_the_cache_is_kept(tmp_path):
    result, state = _run_baseline(_measured(tmp_path, hit=96.8, radix_off=False, cached=[0, 512]))
    assert result["status"] == "succeeded"
    assert "error_class" not in result
    assert not state.stop_reason


def test_a_baseline_that_already_failed_keeps_its_own_cause(tmp_path):
    measured = _measured(tmp_path, hit=96.8, radix_off=True, cached=[0])
    measured.update(status="failed", error_class="server_boot_failed")
    result, state = _run_baseline(measured)
    assert result["error_class"] == "server_boot_failed"
    assert not state.stop_reason


@pytest.mark.parametrize("kind", ["explore", "profile"])
def test_candidates_are_not_judged(tmp_path, kind):
    result, state = _run_baseline(_measured(tmp_path, hit=96.8, radix_off=True, cached=[0]), kind=kind)
    assert result["status"] == "succeeded"
    assert not state.stop_reason


def test_a_candidate_validated_as_a_baseline_kind_task_is_not_judged(tmp_path):
    measured = _measured(tmp_path, hit=96.8, radix_off=True, cached=[0])
    result, state = _run_baseline(measured, params={"quality_ref_exempt": True})
    assert result["status"] == "succeeded"
    assert not state.stop_reason
