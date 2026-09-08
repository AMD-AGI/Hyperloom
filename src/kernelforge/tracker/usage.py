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

    #: Folded under this name when a caller names no role. A campaign that
    #: reaches production with everything here has an accounting bug, not a
    #: cheap run, and the summary should say so rather than look tidy.
    UNATTRIBUTED = "unattributed"

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self._priced_calls = 0
        # role -> the same six counters, so a campaign total can be taken apart
        # by what was actually being paid for. Ordinary dict: insertion order is
        # first-seen order, which is roughly the order of the run.
        self._by_role: dict[str, dict[str, float]] = {}

    def _role_bucket(self, role: str) -> dict[str, float]:
        """Return the counters for one role, creating them on first sight."""
        name = (role or "").strip() or self.UNATTRIBUTED
        bucket = self._by_role.get(name)
        if bucket is None:
            bucket = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_cost_usd": 0.0,
                "calls": 0,
            }
            self._by_role[name] = bucket
        return bucket

    def add_from_message(self, message: Any, *, role: str = "") -> bool:
        """Fold one SDK message's usage into the running totals.

        Gated on ``total_cost_usd`` so each query is counted once: only the
        terminal ``ResultMessage`` carries it, while ``AssistantMessage`` also
        exposes ``usage`` and would double-bill.
        """
        if not hasattr(message, "total_cost_usd"):
            return False
        usage = getattr(message, "usage", None)
        cost = getattr(message, "total_cost_usd", None)
        return self.add_usage(usage, total_cost_usd=cost, role=role)

    def add_usage(
        self,
        usage: dict[str, Any] | None,
        *,
        total_cost_usd: Any = None,
        role: str = "",
    ) -> bool:
        """Fold one normalized provider usage record into the totals.

        ``role`` names what the call was for. It only ever splits the same
        numbers into buckets -- the campaign total is unchanged by it -- so an
        unnamed call is counted exactly as before, under ``UNATTRIBUTED``.
        """
        bucket = self._role_bucket(role)
        if isinstance(usage, dict):
            for key in _TOKEN_KEYS:
                with contextlib.suppress(TypeError, ValueError):
                    value = int(usage.get(key) or 0)
                    setattr(self, key, getattr(self, key) + value)
                    bucket[key] += value
        if total_cost_usd is not None and not isinstance(total_cost_usd, bool):
            with contextlib.suppress(TypeError, ValueError):
                cost = float(total_cost_usd)
                if math.isfinite(cost) and cost >= 0:
                    self.total_cost_usd += cost
                    bucket["total_cost_usd"] += cost
                    self._priced_calls += 1
        self.calls += 1
        bucket["calls"] += 1
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
            "by_role": self.by_role(),
        }

    def by_role(self) -> dict[str, dict[str, Any]]:
        """Return the same totals split by what the tokens were spent on.

        Ordered most expensive first -- by cost where the provider priced the
        calls, and by input tokens where it did not, because input is what a
        long agent session actually accumulates and it is the only ranking
        available when cost is missing.
        """
        def _weight(item: tuple[str, dict[str, float]]) -> tuple[float, float]:
            counters = item[1]
            return (counters["total_cost_usd"], counters["input_tokens"])

        return {
            name: {
                "input_tokens": int(counters["input_tokens"]),
                "output_tokens": int(counters["output_tokens"]),
                "cache_creation_input_tokens": int(counters["cache_creation_input_tokens"]),
                "cache_read_input_tokens": int(counters["cache_read_input_tokens"]),
                "total_cost_usd": round(counters["total_cost_usd"], 6),
                "calls": int(counters["calls"]),
            }
            for name, counters in sorted(self._by_role.items(), key=_weight, reverse=True)
        }

    def __bool__(self) -> bool:
        """Truthy once at least one LLM call has been counted."""
        return self.calls > 0


__all__ = ["UsageAccumulator"]
