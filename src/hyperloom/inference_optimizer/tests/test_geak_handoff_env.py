# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The GEAK handoff's shell environment preserves current-best values."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["two words", '{"key": "a b", "eq": "a=b"}', "user's config", r"a\b", "", "plain"])
async def test_handoff_preserves_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    expected = {"SGLANG_USE_AITER": "1", "CUSTOM_CONFIG": value}
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path="/models/example",
        current_best={"extra_envs": expected, "optimization_stack": []},
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "sglang")

    def stop_after_handoff(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        stop_after_handoff,
    )
    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["baseline_env_spec"]["config"]["extra_envs"] == expected
    assert dict(token.split("=", 1) for token in shlex.split(handoff["accepted_env"])) == expected
