# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX baseline measured with the prefix cache off is refused, on every serving engine and AgentX client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common.launch_log_evidence import observed_server_config_from_log
from hyperloom.orchestrator.actions.executors import _agentx_prefix_cache as pc
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.state.shared_state import SharedState


def _sglang(tmp_path: Path, *, radix_off: bool | None, cached: list[int] | None) -> dict[str, Any]:
    """A succeeded SGLang measurement whose resolved config and prefill log say what they say."""
    log = tmp_path / "server.log"
    log.write_text(
        "".join(
            f"Prefill batch. #new-seq: 1, #new-token: 8192, #cached-token: {n}, token usage: 0.01\n" for n in cached
        )
        if cached is not None
        else "server up\n",
        encoding="utf-8",
    )
    return {
        "status": "succeeded",
        "output_throughput": 221.8,
        "server_log_path": str(log),
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
    ("radix_off", "cached", "refused"),
    [
        # The observed server config alone settles it.
        (True, None, True),
        (True, [0, 4096], True),
        # The server log alone settles it: every prefill batch reused nothing.
        (None, [0, 0, 0], True),
        (False, [0, 0], True),
        # Any reused token clears it.
        (False, [0, 4096], False),
        # Nothing readable about the cache: left alone.
        (None, None, False),
    ],
)
def test_sglang(tmp_path, radix_off, cached, refused):
    assert bool(pc.prefix_cache_off_evidence(_sglang(tmp_path, radix_off=radix_off, cached=cached))) is refused


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
        (lambda p: _sglang(p, radix_off=True, cached=None), "disable_radix_cache=True"),
        (lambda p: _sglang(p, radix_off=None, cached=[0, 0]), "no SGLang prefill batch reused a cached token"),
        (lambda p: _vllm(p, "{'enable_prefix_caching': False}"), "enable_prefix_caching=False"),
        (lambda p: _atom("--level 2"), "without --enable_prefix_caching"),
    ],
)
def test_an_agentx_baseline_measured_with_the_cache_off_is_refused_and_stops_the_run(tmp_path, build, says):
    result, state = _run_baseline(build(tmp_path))
    assert result["status"] == "failed"
    assert result["error_class"] == pc.PREFIX_CACHE_OFF_ERROR_CLASS
    assert says in result["error"]
    assert state.stop_reason == "baseline_failed"


def test_agentx_enabled_by_the_environment_alone_is_judged(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    result, state = _run_baseline(_sglang(tmp_path, radix_off=True, cached=None), benchmark_mode="")
    assert result["error_class"] == pc.PREFIX_CACHE_OFF_ERROR_CLASS
    assert state.stop_reason == "baseline_failed"


def test_a_synthetic_baseline_is_not_judged(tmp_path):
    result, state = _run_baseline(_sglang(tmp_path, radix_off=True, cached=[0]), benchmark_mode="")
    assert result["status"] == "succeeded"
    assert not state.stop_reason


def test_a_baseline_that_reused_the_cache_is_kept(tmp_path):
    result, state = _run_baseline(_sglang(tmp_path, radix_off=False, cached=[0, 512]))
    assert result["status"] == "succeeded"
    assert "error_class" not in result
    assert not state.stop_reason


def test_a_baseline_that_already_failed_keeps_its_own_cause(tmp_path):
    measured = _sglang(tmp_path, radix_off=True, cached=[0])
    measured.update(status="failed", error_class="server_boot_failed")
    result, state = _run_baseline(measured)
    assert result["error_class"] == "server_boot_failed"
    assert not state.stop_reason


@pytest.mark.parametrize("kind", ["explore", "profile"])
def test_candidates_are_not_judged(tmp_path, kind):
    result, state = _run_baseline(_sglang(tmp_path, radix_off=True, cached=[0]), kind=kind)
    assert result["status"] == "succeeded"
    assert not state.stop_reason


def test_a_candidate_validated_as_a_baseline_kind_task_is_not_judged(tmp_path):
    measured = _sglang(tmp_path, radix_off=True, cached=[0])
    result, state = _run_baseline(measured, params={"quality_ref_exempt": True})
    assert result["status"] == "succeeded"
    assert not state.stop_reason
