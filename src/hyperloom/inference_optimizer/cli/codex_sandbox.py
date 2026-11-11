# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight gate on the bubblewrap sandbox Codex runs its tools in."""

from __future__ import annotations

import shutil
import sys

from hyperloom.common.codex_session import (
    CodexSessionError,
    probe_codex_sandbox_capability,
    resolve_codex_sandbox_mode,
)
from hyperloom_common.llm.llm_config import AGENT_BACKEND_CODEX


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
        "Install bubblewrap (assets/install.sh does it as root); in a container run with "
        "--cap-add SYS_ADMIN and --security-opt apparmor=unconfined; or set "
        "HYPERLOOM_CODEX_SANDBOX_MODE=bypass to run without the sandbox.",
        file=sys.stderr,
    )
    sys.exit(2)
