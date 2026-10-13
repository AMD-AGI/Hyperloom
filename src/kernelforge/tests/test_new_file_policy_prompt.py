# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The implementer is told the new-file rule the workspace guard will actually enforce.

The shared edit-surface card used to say "you may add new files" unconditionally while the guard refuses every new
file outside --commit-new-path; with an empty allowlist a whole round planned around a new config JSON was discarded.
"""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.kernel_backends.prompt_utils import EDIT_SURFACE_AND_SWEEPS_PROMPT
from kernelforge.orchestrator import agent


def _implementer_prompt(tmp_path, monkeypatch, *, commit_new_paths, gate):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n")
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
            return AgentRunResult(text="PLAN: edit the kernel")

    monkeypatch.setattr(agent, "create_registered_backend", lambda runtime, **kwargs: RecordingBackend(runtime))
    agent_fn = agent.make_agent_fn(
        config=Config(gpu_target="gfx950", workspace=str(tmp_path), agent_backend="claude", agent_precheck=False),
        program_md="Optimize the kernel.",
        kernel_backend_name="triton",
        insession_gate=gate,
        driver_script=str(driver),
        commit_new_paths=commit_new_paths,
    )
    asyncio.run(agent_fn(str(kernel), ""))
    return " ".join(specs[0].system_prompt.split())


def test_the_shared_card_no_longer_grants_new_files_unconditionally():
    card = " ".join(EDIT_SURFACE_AND_SWEEPS_PROMPT.split())
    assert "is editable too, and you may add new files." not in card
    assert "add new files only where the campaign's new-file allowlist admits them" in card


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.parametrize("commit_new_paths", [None, []])
def test_an_empty_allowlist_is_stated_as_no_new_files(tmp_path, monkeypatch, gate, commit_new_paths):
    prompt = _implementer_prompt(tmp_path, monkeypatch, commit_new_paths=commit_new_paths, gate=gate)
    assert "outside the campaign's explicit --commit-new-path allowlist" in prompt
    assert "This campaign admits no new files" in prompt
    assert "goes into the code, not into a new file" in prompt


@pytest.mark.parametrize("gate", [True, False])
def test_a_declared_allowlist_is_spelled_out(tmp_path, monkeypatch, gate):
    prompt = _implementer_prompt(
        tmp_path, monkeypatch, commit_new_paths=["vllm_fused_moe/configs/*.json"], gate=gate
    )
    assert "New files are admitted only where they match: `vllm_fused_moe/configs/*.json`" in prompt
    assert "This campaign admits no new files" not in prompt
