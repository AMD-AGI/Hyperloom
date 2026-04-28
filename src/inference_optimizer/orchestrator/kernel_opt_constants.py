"""KERNEL_OPT runtime constants — DESIGN §4.6.

Frozen module-level constants surfaced to GEAK / Codex sub-agents and to the
Conductor. All callers must import; no dynamic mutation. Kept as a separate
module so unit tests can lock the values.

STATUS:
    Concrete values are filled in. ``KERNEL_OPT_IMAGE`` reads from env at
    import time; tests should set the env var.

References:
    - DESIGN §4.5 IR-1..IR-3 / IR-6
    - DESIGN §4.6 KERNEL_OPT runtime constants
"""
from __future__ import annotations

import os
from typing import Final

# --------------------------------------------------------------------------
# §4.6 kernel-opt backend defaults
# --------------------------------------------------------------------------
KERNEL_OPT_BACKENDS: Final[str] = "geak,codex"
"""Both backends submitted in parallel each round (IR-1)."""

OOB_ROUND_ITERATIONS: Final[int] = 3
"""Per kernel-opt invocation we allow 3 OOB rounds before hand-off."""

KERNEL_OPT_WORKSPACE: Final[str] = "control-plane-moe"
"""Default Claw workspace name for kernel-opt sub-agents."""

# --------------------------------------------------------------------------
# §4.6 GEAK budget guards (must hard-fail if violated)
# --------------------------------------------------------------------------
GEAK_STEP_LIMIT: Final[int] = 100
GEAK_MAX_RETRIES: Final[int] = 3
GEAK_MAX_SUBMISSIONS: Final[int] = 15
GEAK_TOP_CANDIDATES: Final[int] = 5
GEAK_CONSECUTIVE_DISCARDS: Final[int] = 5
GEAK_WALL_CLOCK_MIN: Final[int] = 120
GEAK_POLL_INTERVAL_S: Final[int] = 60
GEAK_POLL_TIMEOUT_MIN: Final[int] = 15

# --------------------------------------------------------------------------
# §4.6 / IR-4 / IR-5 process-management constants
# --------------------------------------------------------------------------
MIN_GPU_PCT: Final[int] = 3
"""Minimum GPU memory % a server must be using before we believe it's healthy."""

SERVER_KILL_WAIT_S: Final[int] = 10
"""Seconds to sleep after `kill` before we recheck pgrep (IR-5)."""

FILTERED_TRACE_NAME: Final[str] = "filtered-TP-0.trace.json.gz"
"""Profile output file the analysis pipeline keys off of."""


# --------------------------------------------------------------------------
# §4.6 image / runtime envs (read once at import)
# --------------------------------------------------------------------------
def _read_env(name: str, default: str | None = None) -> str:
    """Read env var; raise if neither env nor default present."""
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(
            f"KERNEL_OPT env {name} not set and no default; "
            "see IMPLEMENTATION-CHECKLIST §1.20"
        )
    return val


# Lazy: pulled at first access so tests can monkey-patch env.
def kernel_opt_image() -> str:
    """Container image for GEAK / Codex kernel-opt sub-agents.

    See IMPLEMENTATION-CHECKLIST §1.20.
    """
    return _read_env("KERNEL_OPT_IMAGE")


def venv_bin_path() -> str:
    """Path prefix prepended for sub-agent shell sessions (IR-4 / §4.7)."""
    return _read_env("INFERENCE_OPTIMIZER_VENV_BIN", "/opt/venv/bin")
