# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Language moves must survive the full implementer prompt and its canonical gate."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.orchestrator import agent


@pytest.mark.parametrize("kernel_backend", ["assembly", "flydsl"])
@pytest.mark.parametrize(
    "program",
    ["Optimize the kernel.", "Optimize the kernel. This task requires the implementation to remain in FlyDSL."],
)
def test_gated_implementer_allows_backend_language_moves_subject_to_task_contract(
    tmp_path, monkeypatch, kernel_backend, program
):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import flydsl.compiler as flyc\n")
    driver = tmp_path / "driver.py"
    driver.write_text("raise AssertionError('prompt tests must not execute the driver')\n")
    specs = []

    class RecordingBackend:
        name = "claude"
        capabilities = AgentCapabilities(stop_hooks=True)

        def __init__(self, runtime):
            self.runtime = runtime

        async def run(self, spec, usage=None):
            specs.append(spec)
            return AgentRunResult(text="PLAN: inspect assembly")

    monkeypatch.setattr(agent, "create_registered_backend", lambda runtime, **kwargs: RecordingBackend(runtime))
    agent_fn = agent.make_agent_fn(
        config=Config(
            gpu_target="gfx950",
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-test",
            agent_precheck=False,
        ),
        program_md=program,
        kernel_backend_name=kernel_backend,
        insession_gate=True,
        driver_script=str(driver),
    )
    asyncio.run(agent_fn(str(kernel), ""))

    assert len(specs) == 1
    spec = specs[0]
    prompt = " ".join(spec.system_prompt.split())
    assert "ONE self-correcting session" in prompt
    assert f"Backend Expertise ({kernel_backend})" in prompt
    assert "kernelforge.assembly" in prompt
    assert program in prompt
    assert "Implementation language may change through the supported routes in the selected backend expertise" in prompt
    assert "unless the task explicitly restricts languages" in prompt
    assert "public callable, launch ABI, and the unchanged driver's correctness contract" in prompt
    assert "Keep the kernel in its original backend/DSL" not in prompt
    assert "do not rewrite in another language" not in prompt
    assert "outside the campaign's explicit --commit-new-path allowlist" in prompt
    assert spec.driver_script == str(driver)
    assert spec.hooks is not None and len(spec.hooks.stop) == 1
