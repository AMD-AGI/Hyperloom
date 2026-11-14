# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rejections that stop a run before its session starts."""

from __future__ import annotations

import argparse
import shutil
import sys

from hyperloom.common.codex_session import (
    CodexSessionError,
    probe_codex_sandbox_capability,
    resolve_codex_sandbox_mode,
)
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom_common.llm.llm_config import AGENT_BACKEND_CODEX


def _preflight_agentx_backend(args: argparse.Namespace) -> None:
    """Reject the combinations where AgentX labels work it did not do."""
    if not agentx_enabled():
        return
    from hyperloom.inference_optimizer import framework_registry

    framework = str(getattr(args, "framework", "") or "").strip().lower()
    if framework and framework_registry.is_scriptable(framework):
        print(
            f"ERROR: HYPERLOOM_AGENTX=1 with --framework {framework!r}, which is "
            "scriptable. The AgentX switch only replaces the benchmark client of a "
            "serving framework, so no trace would be replayed -- but the session "
            "would still be labelled AgentX and run on AgentX budgets. Pick one.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _check_codex_sandbox(backend: str) -> None:
    """Stop before launch when the orchestration ``backend`` is Codex and this host cannot create its sandbox.

    The same probe runs at every orchestration turn and refuses to fall back to bypass,
    so without this gate a run only fails turn by turn after the optimizer has started.
    """
    if backend != AGENT_BACKEND_CODEX:
        return
    try:
        mode = resolve_codex_sandbox_mode()
    except CodexSessionError as exc:
        print(f"Preflight: ERROR — Codex sandbox policy is not usable: {exc}", file=sys.stderr)
        sys.exit(2)
    if mode == "bypass" or probe_codex_sandbox_capability():
        return
    reason = "bwrap is not on PATH" if not shutil.which("bwrap") else "bwrap cannot create the sandbox"
    print(
        f"Preflight: ERROR — Codex sandbox mode {mode!r} needs a working bubblewrap, but {reason}. "
        "Install bubblewrap in the image; in a container run with "
        "--cap-add SYS_ADMIN and --security-opt apparmor=unconfined; or set "
        "HYPERLOOM_CODEX_SANDBOX_MODE=bypass to run without the sandbox.",
        file=sys.stderr,
    )
    sys.exit(2)
