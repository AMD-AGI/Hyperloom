# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight gate on the binaries GEAK runs in the kernel phase, for the agent backend it will select."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, NoReturn

from hyperloom.common.geak_checkout import resolve_e2e_runner
from hyperloom_common.llm.llm_config import AGENT_BACKEND_CODEX, is_openai_only

#: Every binary variable GEAK reads. An unset GEAK_CLAUDE_BIN leaves the Claude Agent SDK on its bundled CLI.
GEAK_BINARY_ENVS = ("GEAK_NODE_BIN", "GEAK_CODEX_BIN", "GEAK_CLAUDE_BIN")

_REGISTRY_RELATIVE_PATH = Path("runtime") / "engine" / "registry.json"


def _refuse(message: str) -> NoReturn:
    print(f"ERROR: {message} Refusing to start.", file=sys.stderr)
    sys.exit(2)


def _geak_defaults() -> dict[str, str]:
    """Each binary GEAK will run, mapped to the command it runs when the variable is unset ("" for none).

    Mirrors GEAK's agent selection: an explicit ``GEAK_AGENT_BACKEND`` (or the Codex backend derived from an
    OpenAI-only configuration) names the agent, else ``GEAK_AGENT_PROFILE`` names it through GEAK's ``registry.json``;
    that agent's CLI, taken from the registry, and Node are required. With neither, GEAK keeps its native Claude
    path, which runs neither Node nor an agent CLI.
    """
    backend = os.environ.get("GEAK_AGENT_BACKEND", "").strip()
    profile = os.environ.get("GEAK_AGENT_PROFILE", "").strip()
    auto = os.environ.get("GEAK_AGENT_AUTO", "1").strip().lower() not in {"0", "false", "no"}
    if not (backend or profile) and auto and is_openai_only():
        backend = AGENT_BACKEND_CODEX
    defaults = {"GEAK_CLAUDE_BIN": ""}
    selection = backend or profile
    if not selection:
        return defaults
    try:
        # The registry sits next to the runner the kernel phase will run.
        registry_path = Path(resolve_e2e_runner()).parent / _REGISTRY_RELATIVE_PATH
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        agent = registry["agents"][backend or registry["profiles"][profile]["agent"]]
        defaults[agent["bin_env"]] = agent["bin"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _refuse(
            f"GEAK would run the agent {selection!r}, but its registry does not resolve it "
            f"({type(exc).__name__}: {exc}), so its binary cannot be checked."
        )
    defaults["GEAK_NODE_BIN"] = "node"
    return defaults


def _geak_binary_pin(name: str) -> str | None:
    """The value GEAK runs for ``name``, or None when GEAK falls back to its default."""
    value = os.environ.get(name)
    if value is None:
        return None
    # GEAK strips only GEAK_CLAUDE_BIN, and runs an empty GEAK_NODE_BIN instead of defaulting to node.
    if name == "GEAK_CLAUDE_BIN":
        value = value.strip()
    if not value and name != "GEAK_NODE_BIN":
        return None
    return value


def check_geak_binaries() -> dict[str, Any]:
    """Hard-gate the binaries GEAK will run, which are otherwise first run at KERNEL_AGENT."""
    # GEAK hands the value to subprocess, which also resolves a bare command name through PATH.
    broken = {}
    for name, default in _geak_defaults().items():
        pin = _geak_binary_pin(name)
        command = default if pin is None else pin
        if (pin is not None or default) and not shutil.which(command):
            broken[name] = command
    if not broken:
        return {"status": "applied", "skip_reason": None, "target": "GEAK"}
    listed = ", ".join(f"{name}={command}" for name, command in broken.items())
    _refuse(
        f"{listed} is neither an executable file nor a command on PATH (an unset variable stands for the "
        f"command GEAK runs by default). GEAK runs it in the kernel phase, so the session would fail there, hours in. "
        f"Re-run src/hyperloom/inference_optimizer/assets/install.sh and source $KERNEL_AGENT_ENV, or point the "
        f"variable at the installed binary."
    )
