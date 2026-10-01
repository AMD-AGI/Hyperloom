# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Round locations shared by the Meta RSI scripts, read from the environment.

Every location comes from an explicit variable; nothing defaults to a particular machine. The
rsi driver (``python -m meta_rsi.rsi``) sets these variables for each step it runs.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_value(name: str) -> str:
    """Value of a required variable; exits with a message naming it when it is unset or empty."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set; export it or run the step through `python -m meta_rsi.rsi`")
    return value


def round_dir() -> Path:
    """The round's working directory (``PULSE_ROUND_DIR``)."""
    return Path(env_value("PULSE_ROUND_DIR"))


def analysis_dir() -> Path:
    """Where the analysis scripts read and write their tables."""
    return round_dir() / "analysis"


def bundles_dir() -> Path:
    """Where the fetched Pulse archives live (``PULSE_BUNDLES``)."""
    return Path(env_value("PULSE_BUNDLES"))


def recent_since() -> str:
    """First session start date (``YYYY-MM-DD``) that counts as recent (``PULSE_RECENT_SINCE``)."""
    return env_value("PULSE_RECENT_SINCE")
