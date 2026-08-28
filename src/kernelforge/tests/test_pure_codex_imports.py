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

from kernelforge.orchestrator import agent as agent_module
import kernelforge.knowledge.experience_sink
from kernelforge.config import Config
import kernelforge.kernel_backends.base

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


def test_provider_sdks_are_not_core_dependencies(repo_root: Path) -> None:
    """Keep Claude and Codex SDKs in provider-specific optional extras.

    KernelForge used to declare its own ``codex = ["openai-codex==0.144.4"]``
    extra. Inside Hyperloom there is a single distribution, and an exact pin
    alongside Hyperloom's ``openai-codex>=0.144`` would be two contradictory
    specifiers in one metadata file -- an install-time resolution error rather
    than anything a test could catch later. The floor is what matters here.
    """
    pyproject = (repo_root / "pyproject.toml").read_text()
    core_section = pyproject.split("[project.optional-dependencies]", 1)[0]

    assert "claude-agent-sdk" not in core_section
    assert "openai-codex" not in core_section
    assert '"claude-agent-sdk>=0.2.110"' in pyproject
    assert '"openai-codex>=0.144"' in pyproject
    # No exact pin may creep back in: it would conflict with the floor above.
    assert "openai-codex==" not in pyproject
