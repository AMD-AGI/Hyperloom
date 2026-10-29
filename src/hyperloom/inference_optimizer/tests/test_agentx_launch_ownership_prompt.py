# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""On a native Magpie AgentX run both prompts point the agents at Magpie's launch protocol.

Magpie rejects a candidate that touches a protected launch input before it is
benchmarked, so an agent that never hears of the rule spends variants, and every
idea bundled into them, on launches that are never measured. The protected lists
are Magpie's: the prompts name where they live and never restate them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir
from hyperloom.orchestrator.prompts.agentx_context import launch_ownership_lines
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
    SessionContext,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import get_domain

_OWNERSHIP = "Magpie own the launch protocol"
_MODULE = "Magpie.modes.benchmark.agentx_launch"
# A value only the installed Magpie file carries: the prompt must point at the file, not copy from it.
_PROTECTED_SENTINEL = "SENTINEL_PROTECTED_ENV"


@pytest.fixture
def magpie_install(tmp_path, monkeypatch) -> Path:
    """A Magpie install whose ``agentx_launch.py`` lies where the session's ``MAGPIE_PATH`` says."""
    launch = tmp_path / "Magpie" / "modes" / "benchmark" / "agentx_launch.py"
    launch.parent.mkdir(parents=True)
    launch.write_text(f'PROTECTED_ENV = frozenset({{"{_PROTECTED_SENTINEL}"}})\n', encoding="utf-8")
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    return launch


def _orchestration_prompt(benchmark_mode: str, backend: str) -> str:
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(benchmark_mode=benchmark_mode, agentx_backend=backend),
    )


def _specialist_prompt(domain: str, benchmark_mode: str, backend: str) -> str:
    inp = SpecialistPromptInputs(
        domain=get_domain(domain),
        benchmark_mode=benchmark_mode,
        agentx_backend=backend,
    )
    return "\n".join(build_specialist_prompts(inp))


def test_a_native_run_is_pointed_at_the_installed_launch_protocol(magpie_install):
    body = "\n".join(launch_ownership_lines("magpie"))
    assert _OWNERSHIP in body
    assert f"`{magpie_install}`" in body
    assert "before it is benchmarked" in body
    assert "every other change bundled into it" in body
    for symbol in ("PROTECTED_ARGS", "PROTECTED_ENV", "PROTECTED_ENV_PREFIXES", "_validate_golden_launch"):
        assert f"`{symbol}`" in body


def test_the_lines_point_at_the_lists_without_copying_them(magpie_install):
    body = "\n".join(launch_ownership_lines("magpie"))
    assert _PROTECTED_SENTINEL not in body
    assert "SGLANG_SIMULATE_ACC" not in body


def test_an_unresolvable_install_still_states_the_rule_with_a_generic_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "missing"))
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    body = "\n".join(launch_ownership_lines("magpie"))
    assert _OWNERSHIP in body
    assert f"`{_MODULE}`" in body
    assert "missing" not in body


@pytest.mark.parametrize("backend", ["mlperf", "aiperf"])
def test_a_backend_other_than_magpie_gets_no_magpie_launch_block(magpie_install, backend):
    """Only a native Magpie launch is Magpie-validated; any other recorded backend gets no block."""
    assert launch_ownership_lines(backend) == []


def test_a_session_without_a_recorded_backend_falls_back_to_the_environment(magpie_install, monkeypatch):
    assert launch_ownership_lines("")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    assert launch_ownership_lines("") == []


def test_the_orchestration_prompt_carries_the_block_only_on_native_agentx(magpie_install):
    assert str(magpie_install) in _orchestration_prompt("agentx", "magpie")
    assert _OWNERSHIP not in _orchestration_prompt("agentx", "mlperf")
    assert _OWNERSHIP not in _orchestration_prompt("", "")
    assert _OWNERSHIP not in _orchestration_prompt("synthetic", "magpie")


@pytest.mark.parametrize("domain", ["research_scout_specialist", "serving_specialist"])
def test_every_specialist_prompt_carries_the_block_only_on_native_agentx(magpie_install, domain):
    assert str(magpie_install) in _specialist_prompt(domain, "agentx", "magpie")
    assert _OWNERSHIP not in _specialist_prompt(domain, "agentx", "mlperf")
    assert _OWNERSHIP not in _specialist_prompt(domain, "", "")


def test_the_specialist_block_is_its_own_paragraph(magpie_install):
    """Without a blank line Markdown folds the block into the grading paragraph above it."""
    lines = _specialist_prompt("serving_specialist", "agentx", "magpie").splitlines()
    start = next(i for i, line in enumerate(lines) if _OWNERSHIP in line)
    assert lines[start - 1] == ""
    assert lines[start - 2].strip()


def test_the_speculative_decoding_reference_defers_to_magpie_on_native_runs():
    doc = (asset_prompt_references_dir() / "speculative_decoding.md").read_text(encoding="utf-8")
    assert "## Native Magpie AgentX runs" in doc
    assert "_validate_golden_launch" in doc
    assert "Hyperloom-managed launch" in doc
