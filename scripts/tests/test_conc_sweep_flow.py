# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The manual sweep driver follows the production invocation contract."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import hyperloom.orchestrator.actions.executors._grid_runner  # noqa: F401
from hyperloom.orchestrator.kernel.conc_sweep import run_conc_sweep
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def flow():
    path = Path(__file__).resolve().parents[1] / "test_conc_sweep_flow.py"
    spec = importlib.util.spec_from_file_location("conc_sweep_flow_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_flow_driver_uses_the_current_sweep_contract(flow, tmp_path, monkeypatch):
    state = SharedState(session_id="flow", max_minutes=1)
    state.closing_phase = True
    state.stop_reason = "time_exhausted"
    monkeypatch.setattr(flow.SharedState, "load_or_init", lambda _path: state)
    result = {"status": "succeeded"}
    sweep = AsyncMock(spec=run_conc_sweep, return_value=result)
    monkeypatch.setattr(flow, "run_conc_sweep", sweep)

    actual = await flow._run(tmp_path, [8, 4], isl=32, osl=64)

    assert actual == result
    sweep.assert_awaited_once()
    call = sweep.call_args
    inspect.signature(run_conc_sweep).bind(*call.args, **call.kwargs)
    assert call.args == (state, tmp_path)
    assert call.kwargs == {"concs": [8, 4], "total_budget_sec": None, "write_reports": True}
    assert state.isl == 32 and state.osl == 64
    assert state.max_minutes == 0
    assert state.closing_phase is False and state.stop_reason == ""


def test_flow_driver_help_does_not_offer_the_retired_timeout(flow, capsys):
    with pytest.raises(SystemExit) as result:
        flow.main(["--help"])
    assert result.value.code == 0
    assert "--variant-timeout-sec" not in capsys.readouterr().out
