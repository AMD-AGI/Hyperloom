# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral LLM token-usage accumulation for autonomous loop runs.

Claude emits a terminal ``ResultMessage`` per query, while Codex emits a
terminal JSONL usage object. Both are folded into the same canonical counters
so downstream persistence remains provider-independent.

:class:`UsageAccumulator` folds those messages into canonical counters so the
spend can be persisted on the experiment record and read back by external
callers without them having to understand the SDK message types.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

# Canonical four-counter set, mirroring the keys the claude-agent-sdk puts on
# ``ResultMessage.usage`` (and what downstream token ledgers expect).
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class UsageAccumulator:
    """Sum normalized LLM token usage and cost across backend calls.

    Pass an instance to ``make_agent_fn`` (which folds terminal provider usage)
    and to
    :meth:`IterationLoop.run`, which persists :meth:`totals` onto the
    experiment when the run finishes. Cheap and dependency-free so it never
    perturbs the agent path.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self._priced_calls = 0

    def add_from_message(self, message: Any) -> bool:
        """Fold one SDK message's usage into the running totals.

        Only the terminal ``ResultMessage`` carries ``total_cost_usd`` (the
        per-query session rollup); ``AssistantMessage`` also exposes ``usage``
        but counting it as well would double-bill, so we gate on the presence
        of ``total_cost_usd`` to count each query exactly once. Returns ``True``
        when the message was a counted result, ``False`` otherwise. Never
        raises — a malformed usage payload degrades to a best-effort partial
        add so the agent loop is never broken by accounting.
        """
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
        """Return the accumulated usage as a plain JSON-serialisable dict.

        ``calls`` is the number of counted provider calls; ``calls == 0`` means
        no LLM call was observed (callers should treat that as "no usage" rather
        than "zero spend").
        """
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
