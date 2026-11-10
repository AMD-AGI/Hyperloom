# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX baseline measured with the prefix cache off is refused, on every serving engine and AgentX client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.common.launch_log_evidence import observed_server_config_from_log
from hyperloom.orchestrator.actions.executors import _agentx_prefix_cache as pc
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.state.shared_state import SharedState


def _sglang(radix_off: bool | None) -> dict[str, Any]:
    """A succeeded SGLang measurement whose resolved config says what it says."""
    return {
        "status": "succeeded",
        "output_throughput": 221.8,
        "completed_requests": 64,
        "launch_evidence": {
            "framework": "sglang",
            "observed_server_config": {} if radix_off is None else {"disable_radix_cache": radix_off},
        },
    }


def _vllm(tmp_path: Path, non_default_args: str) -> dict[str, Any]:
    """A succeeded vLLM measurement, its config read by the same parser the launch evidence uses."""
    log = tmp_path / "server.log"
    log.write_text(f"INFO 10-14 api_server.py:1 non-default args: {non_default_args}\n", encoding="utf-8")
    return {
        "status": "succeeded",
        "server_log_path": str(log),
        "launch_evidence": {
            "framework": "vllm",
            "observed_server_config": observed_server_config_from_log(str(log), "vllm"),
        },
    }


def _atom(requested_server_args: str) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "server_log_path": "",
        "launch_evidence": {
            "framework": "atom",
            "observed_server_config": {},
            "requested_server_args": requested_server_args,
        },
    }


@pytest.mark.parametrize(
    ("radix_off", "refused"),
    [
        (True, True),
        (False, False),
        # No resolved setting was read (no record, or one past the cap): left alone, as an unread vLLM record is.
        (None, False),
    ],
)
def test_sglang(radix_off, refused):
    assert bool(pc.prefix_cache_off_evidence(_sglang(radix_off))) is refused


def test_sglang_reads_the_resolved_setting_the_launch_record_parser_returns(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=8)\n", encoding="utf-8")
    measured = _sglang(None)
    measured["launch_evidence"]["observed_server_config"] = observed_server_config_from_log(str(log), "sglang")
    assert "disable_radix_cache=True" in pc.prefix_cache_off_evidence(measured)


@pytest.mark.parametrize(
    ("non_default_args", "refused"),
    [
        ("{'model': 'm', 'enable_prefix_caching': False}", True),
        ("{'model': 'm', 'enable_prefix_caching': True}", False),
        # vLLM records only what the launch changed, so an absent key is its default: on.
        ("{'model': 'm'}", False),
    ],
)
def test_vllm(tmp_path, non_default_args, refused):
    assert bool(pc.prefix_cache_off_evidence(_vllm(tmp_path, non_default_args))) is refused


@pytest.mark.parametrize(
    ("requested_server_args", "refused"),
    [
        ("", True),
        ("--level 2 --kv_cache_dtype fp8", True),
        ("--level 2 --enable_prefix_caching", False),
        ("--enable-prefix-caching", False),
    ],
)
def test_atom(requested_server_args, refused):
    assert bool(pc.prefix_cache_off_evidence(_atom(requested_server_args))) is refused


@pytest.mark.parametrize(
    ("extra_atom_args", "refused"), [("--level 2", True), ("--level 2 --enable_prefix_caching", False)]
)
def test_atom_reads_the_server_args_the_materialized_config_launches_with(tmp_path, extra_atom_args, refused):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {"benchmark": {"framework": "atom", "model": "m", "envs": {"EXTRA_ATOM_ARGS": extra_atom_args}}}
        ),
        encoding="utf-8",
    )
    evidence = build_launch_evidence(RequestedLaunch(config, "atom"), actual_server_log=None, slot=tmp_path)
    measured = {"status": "succeeded", "server_log_path": "", "launch_evidence": evidence}
    assert bool(pc.prefix_cache_off_evidence(measured)) is refused


def test_an_unrecorded_launch_is_left_alone():
    assert pc.prefix_cache_off_evidence({"status": "succeeded"}) == ""
    assert pc.prefix_cache_off_evidence({"status": "succeeded", "launch_evidence": {"framework": "xdit"}}) == ""


def _run_baseline(
    result: dict[str, Any],
    *,
    kind: str = "baseline",
    params: dict[str, Any] | None = None,
    benchmark_mode: str = "agentx",
) -> tuple[dict[str, Any], SharedState]:
    state = SharedState(enablement_mode="off")
    state.benchmark_mode = benchmark_mode
    executor = BaselineExecutor()

    async def _measured_round(ctx, *, recorder=None):
        return result

    executor._run_retrying = _measured_round
    task = SimpleNamespace(task_id="t-bl", kind=kind, params={"framework": "sglang", **(params or {})})
    ctx = SimpleNamespace(task=task, extra={"shared_state": state})
    return asyncio.run(executor._run(ctx)), state


@pytest.fixture(autouse=True)
def _no_ambient_agentx(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


@pytest.mark.parametrize(
    ("build", "says"),
    [
        (lambda p: _sglang(True), "disable_radix_cache=True"),
        (lambda p: _vllm(p, "{'enable_prefix_caching': False}"), "enable_prefix_caching=False"),
        (lambda p: _atom("--level 2"), "without --enable_prefix_caching"),
    ],
)
def test_an_agentx_baseline_measured_with_the_cache_off_is_refused(tmp_path, build, says):
    result, state = _run_baseline(build(tmp_path))
    assert result["status"] == "failed"
    assert result["error_class"] == pc.PREFIX_CACHE_OFF_ERROR_CLASS
    assert says in result["error"]
    # The executor only marks the result; writeback decides what the run does with it.
    assert not state.stop_reason


def test_agentx_enabled_by_the_environment_alone_is_judged(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    result, _ = _run_baseline(_sglang(True), benchmark_mode="")
    assert result["error_class"] == pc.PREFIX_CACHE_OFF_ERROR_CLASS


def test_a_synthetic_baseline_is_not_judged(tmp_path):
    result, state = _run_baseline(_sglang(True), benchmark_mode="")
    assert result["status"] == "succeeded"
    assert not state.stop_reason


def test_a_baseline_that_reused_the_cache_is_kept(tmp_path):
    result, state = _run_baseline(_sglang(False))
    assert result["status"] == "succeeded"
    assert "error_class" not in result
    assert not state.stop_reason


def test_a_baseline_that_already_failed_keeps_its_own_cause(tmp_path):
    measured = _sglang(True)
    measured.update(status="failed", error_class="server_boot_failed")
    result, state = _run_baseline(measured)
    assert result["error_class"] == "server_boot_failed"
    assert not state.stop_reason


@pytest.mark.parametrize("kind", ["explore", "profile"])
def test_candidates_are_not_judged(tmp_path, kind):
    result, state = _run_baseline(_sglang(True), kind=kind)
    assert result["status"] == "succeeded"
    assert not state.stop_reason


def test_a_candidate_validated_as_a_baseline_kind_task_is_not_judged(tmp_path):
    measured = _sglang(True)
    result, state = _run_baseline(measured, params={"quality_ref_exempt": True})
    assert result["status"] == "succeeded"
    assert not state.stop_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor_tput", [0.0, 200.0], ids=["first_baseline", "rebaseline_after_an_anchor"])
async def test_writeback_refuses_to_anchor_and_stops_without_opening_enablement(session_dir, monkeypatch, anchor_tput):
    """The server booted and measured, so the refusal is no boot failure: nothing is left for enablement to author."""
    from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
    from hyperloom.orchestrator.state.task_registry import Task

    from .conftest import make_coordinator

    triggers: list[enablement_event.LaneTrigger] = []
    monkeypatch.setattr(enablement_event, "record_trigger", lambda trigger: triggers.append(trigger))
    coord = make_coordinator(
        session_dir,
        shared_state_overrides={"enablement_mode": "all", "benchmark_mode": "agentx", "baseline_tput": anchor_tput},
    )
    refused = _sglang(True)
    pc.refuse_cacheless_agentx_baseline(refused)
    task = Task(task_id="bl-cache-off", kind="baseline", state="running", params={}, idempotency_key="bl-cache-off")

    # The refused round measured real throughput, so it is a valid measurement; it still must not anchor.
    assert coord.writeback.is_promotable_result("baseline", refused) is False
    assert coord.writeback.is_promotable_result("baseline", _sglang(False)) is True
    await coord.writeback.handle_unpromotable_result(task, refused)

    assert coord.shared_state.stop_reason == "baseline_failed"
    assert triggers == []
    assert not (coord.shared_state.enablement.launch_log or "").strip()
    assert not str(coord.shared_state.enablement.origin or "")
    assert coord.shared_state.baseline_failure_streak == 0
