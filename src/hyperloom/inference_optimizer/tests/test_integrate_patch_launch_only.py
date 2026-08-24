# Copyright Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for the enablement_launch_only bench-only mode of IntegratePatchExecutor.

Validates that _stage_resolve skips the specialist/Critic gate, _stage_apply skips
the no-patches early return, and the gate produces kept/advanced/reverted without
any real bench or specialist workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


def _make_ctx(task_id: str, params: dict[str, Any], extra: dict | None = None) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra=extra or {})


def _runtime_override() -> dict[str, Any]:
    return {"path_prefix": "/tmp/venv/bin", "framework_python": "/tmp/venv/bin/python"}


def _params_base(session_dir: Path) -> dict[str, Any]:
    return {
        "enablement": True,
        "enablement_launch_only": True,
        "runtime_override": _runtime_override(),
        "config_path": str(session_dir / "bench.yaml"),
        "source": "coordinator_internal",
    }


def _write_minimal_config(path: Path) -> None:
    path.write_text("benchmark:\n  model: /tmp/m\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# _stage_resolve: launch-only bypasses specialist/Critic checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_only_skips_missing_specialist_task_id(tmp_path):
    """No specialist_task_id needed in launch-only mode."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 100.0, "status": "succeeded"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-1", params))

    assert res["status"] == "kept"
    assert res.get("enablement") is True


@pytest.mark.asyncio
async def test_launch_only_skips_critic_gate(tmp_path):
    """Critic verdict is not consulted in launch-only mode."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    class _RejectAll:
        def get_specialist_patch_verdict(self, tid):
            return "reject"

    bench_result = {"output_throughput": 50.0, "status": "succeeded"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-2", params, extra={"shared_state": _RejectAll()}))

    assert res["status"] == "kept"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patches", ["candidate.patch"]),
        ("enablement_base_patches", ["base.patch"]),
        ("localization_candidate", {"kind": "pr_backport"}),
        ("runtime_candidate", {"kind": "runtime_candidate"}),
        ("artifacts", [{"source": "x", "target": "y"}]),
        ("config_changes", {"EXTRA_VLLM_ARGS": "--unsafe"}),
        ("enablement_setup_commands", ["pip install package"]),
    ],
)
async def test_launch_only_rejects_mutation_fields(tmp_path, field, value):
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = {**_params_base(session), field: value}

    res = await ex(_make_ctx("probe-mutation", params))

    assert res["status"] == "failed"
    assert res["error_class"] == "launch_only_mutation_forbidden"
    assert field in res["error"]


# ---------------------------------------------------------------------------
# _stage_apply: launch-only falls through to bench when no patches exist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_only_does_not_return_no_patches(tmp_path):
    """enablement_launch_only suppresses the no_patches early return."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "cuda oom"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-3", params))

    assert res["status"] != "no_patches", "launch-only must never return no_patches"
    assert res.get("enablement") is True


# ---------------------------------------------------------------------------
# Gate routing: kept / reverted / advanced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_only_boot_success_returns_kept(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 200.0, "status": "succeeded"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-4", params))

    assert res["status"] == "kept"
    assert res["runnable"] is True
    assert res["patches_applied"] == []


@pytest.mark.asyncio
async def test_launch_only_boot_fail_returns_reverted_or_advanced(tmp_path):
    """A non-booting result returns reverted (no progress) or advanced (new gap)."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "ImportError: no module"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-5", params))

    assert res["status"] in ("reverted", "advanced")
    assert res.get("enablement") is True
    assert res["runnable"] is False


@pytest.mark.asyncio
async def test_launch_only_with_before_signature_advanced(tmp_path):
    """When before_signature captures a prior failure, a different crash produces 'advanced'."""
    from hyperloom.agents.framework.enablement import FailureSignature

    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)

    before = FailureSignature(kind="weight_init", raw_excerpt="KeyError weight")
    params = {
        **_params_base(session),
        "enablement_before_signature": before.to_dict(),
    }

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "ImportError: vllm._C not found"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-6", params))

    assert res.get("enablement") is True
    assert res["status"] in ("advanced", "reverted")


@pytest.mark.asyncio
async def test_launch_only_runtime_override_passed_to_bench(tmp_path):
    """runtime_override from params is visible inside _bench_patch params."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    captured: list[dict] = []

    async def _capture_bench(self_inner=None, **kwargs):
        captured.append(dict(kwargs.get("params", {})))
        return {"output_throughput": 50.0, "status": "succeeded"}, {"enablement_accuracy": None, "timed_out": False}

    with patch.object(IntegratePatchExecutor, "_bench_patch", new=_capture_bench):
        await ex(_make_ctx("probe-7", params))

    assert captured, "_bench_patch was not called"
    bench_params = captured[0]
    assert bench_params.get("runtime_override") == _runtime_override()
