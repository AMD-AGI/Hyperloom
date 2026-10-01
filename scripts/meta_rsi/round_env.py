# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Round locations shared by the Meta RSI scripts, read from the environment.

Every location comes from an explicit variable; nothing defaults to a particular machine. These
are inputs of the scripts, not Hyperloom settings. The rsi driver (``python -m meta_rsi.rsi``)
runs each script with its own environment plus the round's values from round.yaml.

* Set by the driver from round.yaml: ``PULSE_ROUND_DIR``, ``PULSE_BUNDLES`` and
  ``PULSE_RECENT_SINCE`` (required by the analyses); ``PULSE_LOCAL_SESSIONS`` (``:``-separated
  session roots, required by ``an_local.py``); ``PULSE_MODELS_DIR`` (required by
  ``select_scenario.py``) and ``PULSE_EXCLUDE_ROOTS`` (default none); ``PULSE_API_KEY`` (unless
  already exported) or the key file ``PULSE_KEY_FILE``, ``PULSE_API_BASE`` (default: the hosted
  Pulse API) and ``PULSE_SOCKS`` (default: direct) for ``pulse.py`` and ``fetch_all.sh``.
* Not in round.yaml; export them before running the driver to change them:
  ``PULSE_MIN_MODEL_B`` (default 30, billions of parameters) for ``select_scenario.py``,
  ``PULSE_LOCAL_SPECIALIST_GLOB`` (optional extra specialist logs) for ``an_specialist.py``, and
  ``TIERS`` (default tier1 tier2 tier3), ``JOBS`` (16) and ``MIN_FREE_GB`` (200) for
  ``fetch_all.sh``.
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
