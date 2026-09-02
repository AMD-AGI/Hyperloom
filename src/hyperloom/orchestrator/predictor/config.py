# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Predictor settings, resolved from the environment the CLI exports.

The pump reads settings through this module rather than off ``SharedState`` so
a resumed session picks up an operator's change of endpoint or mode without
having to re-pass flags. What *is* session state — how many chain steps a
macro-cycle has spent — stays on ``SharedState``.

Everything here fails safe: an unparseable value logs once and falls back to
the default rather than raising into the tick loop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Predict nothing; the pump returns before building a request.
MODE_OFF = "off"
#: Predict, parse and log, but enqueue nothing. Costs no GPU time.
MODE_SHADOW = "shadow"
#: Predict and enqueue.
MODE_ACTIVE = "active"

MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

ENV_ENDPOINT = "HYPERLOOM_PREDICTOR_ENDPOINT"
ENV_MODE = "HYPERLOOM_PREDICTOR_MODE"
ENV_MAX_CHAIN = "HYPERLOOM_PREDICTOR_MAX_CHAIN"
ENV_BUDGET_PCT = "HYPERLOOM_PREDICTOR_BUDGET_PCT"
ENV_TIMEOUT_SEC = "HYPERLOOM_PREDICTOR_TIMEOUT_SEC"
ENV_PHASE_LABEL = "HYPERLOOM_PREDICTOR_PHASE_LABEL"

#: Shadow, not active. The two known mismatches between what Hyperloom reports
#: and what a consumer was trained on are both silent, so the default has to be
#: the mode that measures them instead of the one that spends benchmark cycles
#: on them.
DEFAULT_MODE = MODE_SHADOW

#: Three steps covers the overwhelming majority of observed stack depths, and a
#: chain that has not converged by then is unlikely to.
DEFAULT_MAX_CHAIN = 3

#: Share of the FRAMEWORK budget the chain may spend before it stands down.
DEFAULT_BUDGET_PCT = 25.0

DEFAULT_TIMEOUT_SEC = 120.0

#: The pump feeds the configuration arm, which was its own phase under this
#: name before the merge into FRAMEWORK_AGENT.
DEFAULT_PHASE_LABEL = "EXPLORE"

#: Frameworks with a flag catalogue on the consumer side. Anything else cannot
#: have its answer validated, so the pump declines rather than sending a
#: request whose reply it could not trust.
SUPPORTED_FRAMEWORKS = frozenset({"sglang", "vllm"})


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("predictor_config: %s=%r is not a number; using %s", name, raw, default)
        return default
    if value < minimum:
        log.warning("predictor_config: %s=%r is below %s; using %s", name, raw, minimum, default)
        return default
    return value


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("predictor_config: %s=%r is not an integer; using %s", name, raw, default)
        return default
    if value < minimum:
        log.warning("predictor_config: %s=%r is below %s; using %s", name, raw, minimum, default)
        return default
    return value


@dataclass(frozen=True)
class PredictorConfig:
    """Resolved predictor settings for one tick."""

    endpoint: str = ""
    mode: str = DEFAULT_MODE
    max_chain: int = DEFAULT_MAX_CHAIN
    budget_pct: float = DEFAULT_BUDGET_PCT
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    phase_label: str = DEFAULT_PHASE_LABEL

    @property
    def enabled(self) -> bool:
        """Whether the pump should do anything at all this tick.

        An unset endpoint is the off switch, not an error: a session that never
        configures a predictor behaves exactly as it did before the feature.
        """
        return bool(self.endpoint) and self.mode != MODE_OFF

    @property
    def enqueues(self) -> bool:
        """Whether a prediction may become a task."""
        return self.enabled and self.mode == MODE_ACTIVE

    def supports(self, framework: Any) -> bool:  # noqa: ANN401 - accepts whatever state carries
        """Whether the consumer has a flag catalogue for this framework."""
        return str(framework or "").strip().lower() in SUPPORTED_FRAMEWORKS


def load() -> PredictorConfig:
    """Resolve settings from the environment.

    Returns:
        PredictorConfig: Never raises; bad values log and fall back.
    """
    mode = os.environ.get(ENV_MODE, "").strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        log.warning("predictor_config: %s=%r is not one of %s; using %r", ENV_MODE, mode, MODES, DEFAULT_MODE)
        mode = DEFAULT_MODE
    return PredictorConfig(
        endpoint=os.environ.get(ENV_ENDPOINT, "").strip(),
        mode=mode,
        max_chain=_env_int(ENV_MAX_CHAIN, DEFAULT_MAX_CHAIN),
        budget_pct=_env_float(ENV_BUDGET_PCT, DEFAULT_BUDGET_PCT),
        timeout_sec=_env_float(ENV_TIMEOUT_SEC, DEFAULT_TIMEOUT_SEC, minimum=1.0),
        phase_label=os.environ.get(ENV_PHASE_LABEL, "").strip() or DEFAULT_PHASE_LABEL,
    )
