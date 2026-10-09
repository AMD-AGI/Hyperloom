# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime findings injected into specialist prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.measurement.runtime_findings import persist_runtime_findings, scan_server_log
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_runtime_findings,
    build_specialist_prompts,
)
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.state.shared_state import SharedState


def _make_coord(tmp_path: Path, *, state: SharedState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


def _make_inp(runtime_findings: str) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        runtime_findings=runtime_findings,
    )


@pytest.mark.asyncio
async def test_warm_specialist_params_injects_current_best_findings(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("Unknown vLLM environment variable detected: VLLM_FOO\n", encoding="utf-8")
    slot = tmp_path / "slot"
    persist_runtime_findings(scan_server_log(str(log), "vllm", declared_env=("VLLM_FOO",)), slot=slot)
    state = SharedState(current_best_measurement={"launch_evidence_path": str(slot / "launch_evidence.json")})
    params: dict[str, Any] = {"domain": "serving_specialist"}

    await _make_coord(tmp_path, state=state).specialist_dispatch.warm_specialist_params(params)

    assert params["runtime_findings"].splitlines()[:2] == [
        f"runtime findings for {log} [vllm]",
        "- detected [correctness] vllm.unknown_env VLLM_FOO x1: Unknown vLLM environment variable detected: VLLM_FOO",
    ]


@pytest.mark.asyncio
async def test_warm_specialist_params_skips_without_current_best(tmp_path):
    params: dict[str, Any] = {"domain": "serving_specialist"}

    await _make_coord(tmp_path, state=SharedState()).specialist_dispatch.warm_specialist_params(params)

    assert "runtime_findings" not in params


def test_section_wraps_findings_in_text_block():
    assert _section_runtime_findings(_make_inp("runtime findings for /s/server.log [vllm]")) == [
        "## 4c. RUNTIME FINDINGS (current best server.log, already scanned)",
        "",
        "Do not re-read the server log for these. A disabled or falling-back hot "
        "path is restored first; do not optimize the fallback implementation.",
        "",
        "```text",
        "runtime findings for /s/server.log [vllm]",
        "```",
    ]
    assert _section_runtime_findings(_make_inp("")) == []


def test_build_specialist_prompts_places_section_before_recipe():
    _system, user = build_specialist_prompts(_make_inp("runtime findings for /s/server.log [vllm]"))

    assert user.index("## 4a. ROOFLINE EVIDENCE") < user.index("## 4c. RUNTIME FINDINGS") < user.index("## 5. ")
