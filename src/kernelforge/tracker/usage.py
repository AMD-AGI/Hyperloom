# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral LLM token-usage accumulation for autonomous loop runs."""

from __future__ import annotations

import contextlib
import math
from typing import Any

# Canonical four-counter set, mirroring the keys the claude-agent-sdk puts on ``ResultMessage.usage`` (and what
# downstream token ledgers expect).
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class UsageAccumulator:
    """Sum normalized LLM token usage and cost across backend calls."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self._priced_calls = 0

    def add_from_message(self, message: Any) -> bool:
        """Fold one SDK message's usage into the running totals."""
        if not hasattr(message, "total_cost_usd"):
            return False
        usage = getattr(message, "usage", None)
        cost = getattr(message, "total_cost_usd", None)
        return self.add_usage(usage, total_cost_usd=cost)

    def add_usage(
        self,
        usage: dict[str, Any] | None,
        *,
        total_cost_usd: Any = None,
    ) -> bool:
        """Fold one normalized provider usage record into the totals."""
        if isinstance(usage, dict):
            for key in _TOKEN_KEYS:
                with contextlib.suppress(TypeError, ValueError):
                    setattr(self, key, getattr(self, key) + int(usage.get(key) or 0))
        if total_cost_usd is not None and not isinstance(total_cost_usd, bool):
            with contextlib.suppress(TypeError, ValueError):
                cost = float(total_cost_usd)
                if math.isfinite(cost) and cost >= 0:
                    self.total_cost_usd += cost
                    self._priced_calls += 1
        self.calls += 1
        return True

    def totals(self) -> dict[str, Any]:
        """Return the accumulated usage as a plain JSON-serialisable dict."""
        cost_available = self.calls > 0 and self._priced_calls == self.calls
        cost_source = "provider" if cost_available else "partial" if self._priced_calls else "unavailable"
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_available": cost_available,
            "cost_source": cost_source,
            "calls": self.calls,
        }

    def __bool__(self) -> bool:
        """Truthy once at least one LLM call has been counted."""
        return self.calls > 0


__all__ = ["UsageAccumulator"]
