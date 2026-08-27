"""Tests for installations that omit optional provider SDKs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pure_codex_modules_import_without_claude_sdk() -> None:
    """Import all local Codex entry paths while blocking Claude SDK imports."""
    script = r"""
import builtins

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
        raise ModuleNotFoundError("blocked for pure-Codex import test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from kernel_agents.orchestrator import agent as agent_module
import kernel_agents.knowledge.experience_sink
from kernel_agents.config import Config
import kernel_agents.fellows.base

runtime = Config(
    agent_backend="codex",
    agent_precheck=False,
    agent_fallback_provider="",
).agent_runtime()
assert runtime.provider == "codex"
print("PURE_CODEX_IMPORT_OK")
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    assert "PURE_CODEX_IMPORT_OK" in process.stdout


def test_provider_sdks_are_not_core_dependencies() -> None:
    """Keep Claude and Codex SDKs in provider-specific optional extras."""
    project_root = Path(__file__).resolve().parents[1]
    pyproject = (project_root / "pyproject.toml").read_text()
    core_section = pyproject.split("[project.optional-dependencies]", 1)[0]

    assert "claude-agent-sdk" not in core_section
    assert "openai-codex" not in core_section
    assert "claude = [" in pyproject
    assert '"claude-agent-sdk>=0.1.0"' in pyproject
    assert "codex = [" in pyproject
    assert '"openai-codex==0.144.4"' in pyproject
