# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Predictor settings, resolved from the environment the CLI exports.

Read on every tick rather than kept on ``SharedState``: the CLI flags and
variables exported in the shell are one interface.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Ask nothing.
MODE_OFF = "off"
#: Ask and log the answer, queue nothing: costs no benchmark time.
MODE_SHADOW = "shadow"
#: Ask and put the answer on the untested-proposal queue.
MODE_ACTIVE = "active"
MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

ENV_ENDPOINT = "HYPERLOOM_PREDICTOR_ENDPOINT"
ENV_MODE = "HYPERLOOM_PREDICTOR_MODE"
ENV_TIMEOUT_SEC = "HYPERLOOM_PREDICTOR_TIMEOUT_SEC"

#: An endpoint alone measures; spending benchmark time takes an explicit ``active``.
DEFAULT_MODE = MODE_SHADOW

#: One round samples eight completions, about two minutes on an 8x MI355X
#: service, and a service shared by several sessions queues them. The request
#: runs off the tick loop, so a generous bound costs nothing while it waits.
DEFAULT_TIMEOUT_SEC = 900.0

#: ``phase.phase`` on the wire. The predictor was trained on sessions in which
#: the configuration arm was a phase of its own under this name.
PHASE_LABEL = "EXPLORE"

#: Frameworks the service has a flag catalogue for; it cannot validate an answer for any other.
SUPPORTED_FRAMEWORKS = frozenset({"sglang", "vllm"})


@dataclass(frozen=True)
class PredictorConfig:
    """Resolved predictor settings."""

    endpoint: str = ""
    mode: str = DEFAULT_MODE
    timeout_sec: float = DEFAULT_TIMEOUT_SEC

    @property
    def enabled(self) -> bool:
        """Whether to ask at all; an unset endpoint is the off switch."""
        return bool(self.endpoint) and self.mode != MODE_OFF

    @property
    def enqueues(self) -> bool:
        """Whether an answer may reach the untested-proposal queue."""
        return self.enabled and self.mode == MODE_ACTIVE


def _timeout_from_env() -> float:
    raw = os.environ.get(ENV_TIMEOUT_SEC, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value >= 1.0):
        log.warning(
            "predictor: %s=%r is not a number of seconds >= 1; using %s", ENV_TIMEOUT_SEC, raw, DEFAULT_TIMEOUT_SEC
        )
        return DEFAULT_TIMEOUT_SEC
    return value


def load() -> PredictorConfig:
    """Resolve settings from the environment; a bad value logs and falls back to its default."""
    mode = os.environ.get(ENV_MODE, "").strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        log.warning("predictor: %s=%r is not one of %s; using %r", ENV_MODE, mode, MODES, DEFAULT_MODE)
        mode = DEFAULT_MODE
    return PredictorConfig(
        endpoint=os.environ.get(ENV_ENDPOINT, "").strip(),
        mode=mode,
        timeout_sec=_timeout_from_env(),
    )
