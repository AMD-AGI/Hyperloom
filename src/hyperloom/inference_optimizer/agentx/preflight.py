# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX preflight: resolve and capability-check the aiperf binary.

``HYPERLOOM_AGENTX`` needs an aiperf build with AgentX (``weka-trace``) support.
This module verifies *capability*, not mere existence, so a plain mainline
aiperf on ``PATH`` fails loud with actionable guidance instead of erroring deep
inside a benchmark run.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, Mapping, Optional


class AgentXPreflightError(RuntimeError):
    """Raised when the aiperf binary is missing or not AgentX-capable."""


def resolve_aiperf_bin(env: Mapping[str, str]) -> Optional[str]:
    """Return ``AIPERF_BIN`` (operator override) else a PATH lookup else None."""
    override = (env.get("AIPERF_BIN") or "").strip()
    if override:
        return override
    # Resolve against the SAME PATH the benchmark subprocess will use (the child
    # env), not this process's os.environ, so preflight probes the binary that
    # actually runs. Falls back to os.environ PATH when env has none.
    return shutil.which("aiperf", path=env.get("PATH"))


def _default_probe(aiperf_bin: str) -> str:
    """Run ``aiperf profile --help`` and return combined stdout+stderr."""
    out = subprocess.run(
        [aiperf_bin, "profile", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (out.stdout or "") + (out.stderr or "")


def check_aiperf_capability(
    aiperf_bin: Optional[str],
    *,
    probe: Optional[Callable[[str], str]] = None,
) -> None:
    """Raise :class:`AgentXPreflightError` unless ``aiperf_bin`` supports AgentX.

    Args:
        aiperf_bin: Resolved aiperf path (None/empty means "not found").
        probe: Injectable help-text probe (defaults to running the binary);
            lets the check run offline in tests.
    """
    if not aiperf_bin:
        raise AgentXPreflightError(
            "HYPERLOOM_AGENTX is on but aiperf was not found. Install the pinned "
            "SemiAnalysisAI/aiperf (cjq/agentx-v0.3) build via install.sh, or set "
            "AIPERF_BIN to an aiperf with AgentX (weka-trace) support."
        )
    probe = probe or _default_probe
    try:
        help_text = probe(aiperf_bin)
    except Exception as exc:  # noqa: BLE001 — surface as a structured preflight error
        raise AgentXPreflightError(f"aiperf capability probe failed for {aiperf_bin!r}: {exc}") from exc
    # weka-trace alone is not enough: the pre-062126 builds carry it but their
    # scenario allowlist rejects the current corpus, and they have no
    # --benchmark-duration. Probing for the flags the client actually emits
    # turns "your aiperf is too old" into a startup error instead of a failure
    # an hour into a run.
    missing = [
        flag
        for flag in ("weka-trace", "--scenario", "--benchmark-duration")
        if flag not in (help_text or "")
    ]
    if missing:
        raise AgentXPreflightError(
            f"aiperf at {aiperf_bin!r} is not AgentX-capable (missing: "
            f"{', '.join(missing)}); install the pinned SemiAnalysisAI/aiperf "
            "build via install.sh (AIPERF_REF) or point AIPERF_BIN at one."
        )
