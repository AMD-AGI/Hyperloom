# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight gate on the binaries GEAK runs in the kernel phase, for the agent backend it will select."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from hyperloom_common.llm.llm_config import AGENT_BACKEND_CODEX, is_openai_only

#: Every binary variable GEAK reads. An unset GEAK_CLAUDE_BIN leaves the Claude Agent SDK on its bundled CLI.
GEAK_BINARY_ENVS = ("GEAK_NODE_BIN", "GEAK_CODEX_BIN", "GEAK_CLAUDE_BIN")


def _geak_defaults() -> dict[str, str]:
    """Each binary GEAK will run, mapped to the command it runs when the variable is unset ("" for none).

    Mirrors GEAK's ``interface/run_e2e.py``: an explicit ``GEAK_AGENT_BACKEND`` or ``GEAK_AGENT_PROFILE``, or a
    Codex backend derived from an OpenAI-only configuration, takes GEAK's Node runtime. With neither, GEAK keeps its
    native Claude path, which runs neither Node nor Codex.
    """
    backend = os.environ.get("GEAK_AGENT_BACKEND", "").strip()
    selected = bool(backend or os.environ.get("GEAK_AGENT_PROFILE", "").strip())
    auto = os.environ.get("GEAK_AGENT_AUTO", "1").strip().lower() not in {"0", "false", "no"}
    if not selected and auto and is_openai_only():
        backend, selected = AGENT_BACKEND_CODEX, True
    defaults = {"GEAK_CLAUDE_BIN": ""}
    if selected:
        defaults["GEAK_NODE_BIN"] = "node"
    if backend == AGENT_BACKEND_CODEX:
        defaults["GEAK_CODEX_BIN"] = "codex"
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
    print(
        f"ERROR: {listed} is neither an executable file nor a command on PATH (an unset variable stands for the "
        f"command GEAK runs by default). GEAK runs it in the kernel phase, so the session would fail there, hours in. "
        f"Re-run src/hyperloom/inference_optimizer/assets/install.sh and source $KERNEL_AGENT_ENV, or point the "
        f"variable at the installed binary. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)
