# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral LLM token-usage accumulation for autonomous loop runs."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
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

    def __init__(self, on_update: Callable[[dict[str, Any]], None] | None = None) -> None:
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
        # provider -> the same counters plus how many of its calls carried a price, so a partial total names the
        # provider whose cost it lacks.
        self._by_provider: dict[str, dict[str, float]] = {}
        # Called with the new totals after every counted call, so an external ledger survives a hard kill between one
        # call and the next instead of stopping at the run's last coarse checkpoint.
        self._on_update = on_update

    def _bucket(self, table: dict[str, dict[str, float]], name: str) -> dict[str, float]:
        """Return the counters ``table`` keeps under ``name``, creating them on first sight."""
        key = (name or "").strip() or self.UNATTRIBUTED
        bucket = table.get(key)
        if bucket is None:
            bucket = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_cost_usd": 0.0,
                "calls": 0,
                "priced_calls": 0,
            }
            table[key] = bucket
        return bucket

    def add_from_message(self, message: Any, *, role: str = "", provider: str = "") -> bool:
        """Fold one SDK message's usage into the running totals.

        Gated on ``total_cost_usd`` so each query is counted once: only the
        terminal ``ResultMessage`` carries it, while ``AssistantMessage`` also
        exposes ``usage`` and would double-bill.
        """
        if not hasattr(message, "total_cost_usd"):
            return False
        usage = getattr(message, "usage", None)
        cost = getattr(message, "total_cost_usd", None)
        return self.add_usage(usage, total_cost_usd=cost, role=role, provider=provider)

    def add_usage(
        self,
        usage: dict[str, Any] | None,
        *,
        total_cost_usd: Any = None,
        role: str = "",
        provider: str = "",
    ) -> bool:
        """Fold one normalized provider usage record into the totals.

        ``role`` names what the call was for and ``provider`` who served it. They only ever split the same numbers
        into buckets -- the campaign total is unchanged by them -- so an unnamed call is counted exactly as before,
        under ``UNATTRIBUTED``.
        """
        buckets = (self._bucket(self._by_role, role), self._bucket(self._by_provider, provider))
        if isinstance(usage, dict):
            for key in _TOKEN_KEYS:
                with contextlib.suppress(TypeError, ValueError):
                    value = int(usage.get(key) or 0)
                    setattr(self, key, getattr(self, key) + value)
                    for bucket in buckets:
                        bucket[key] += value
        cost = _price(total_cost_usd)
        if cost is not None:
            self.total_cost_usd += cost
            self._priced_calls += 1
        for bucket in buckets:
            bucket["calls"] += 1
            if cost is not None:
                bucket["total_cost_usd"] += cost
                bucket["priced_calls"] += 1
        self.calls += 1
        if self._on_update is not None:
            with contextlib.suppress(Exception):  # broad-suppress: caller-supplied observer
                self._on_update(self.totals())
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
            "by_provider": self.by_provider(),
        }

    def by_role(self) -> dict[str, dict[str, Any]]:
        """Return the same totals split by what the tokens were spent on, most expensive first."""
        return _ranked(self._by_role, priced_calls=False)

    def by_provider(self) -> dict[str, dict[str, Any]]:
        """Return the same totals split by the provider that served them, with how many of its calls were priced."""
        return _ranked(self._by_provider, priced_calls=True)

    def __bool__(self) -> bool:
        """Truthy once at least one LLM call has been counted."""
        return self.calls > 0


def _price(total_cost_usd: Any) -> float | None:
    """A provider-reported cost, or ``None`` when the call carried no usable price."""
    if total_cost_usd is None or isinstance(total_cost_usd, bool):
        return None
    try:
        cost = float(total_cost_usd)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _ranked(table: dict[str, dict[str, float]], *, priced_calls: bool) -> dict[str, dict[str, Any]]:
    """Buckets ordered most expensive first.

    By cost where the provider priced the calls, and by input tokens where it did not, because input is what a long
    agent session actually accumulates and it is the only ranking available when cost is missing.
    """
    keys = (*_TOKEN_KEYS, "calls", *(("priced_calls",) if priced_calls else ()))
    ordered = sorted(table.items(), key=lambda item: (item[1]["total_cost_usd"], item[1]["input_tokens"]), reverse=True)
    return {
        name: {**{key: int(counters[key]) for key in keys}, "total_cost_usd": round(counters["total_cost_usd"], 6)}
        for name, counters in ordered
    }


def combine_usage_totals(
    *records: dict[str, Any] | None,
    incomplete: bool = False,
) -> dict[str, Any]:
    """Combine independently accumulated usage records without losing cost provenance.

    ``incomplete`` states that a contributor's ledger is known to be missing from ``records``. The counters below then
    describe only part of the run, so the combination reports itself as ``partial`` rather than claiming the complete
    provider-priced answer a reader would otherwise bill against.
    """
    combined: dict[str, Any] = dict.fromkeys(_TOKEN_KEYS, 0)
    combined["total_cost_usd"] = 0.0
    combined["calls"] = 0
    all_cost_available = True
    any_priced_usage = False

    for record in records:
        if not isinstance(record, dict):
            continue
        for key in (*_TOKEN_KEYS, "calls"):
            with contextlib.suppress(TypeError, ValueError):
                value = int(record.get(key) or 0)
                if value >= 0:
                    combined[key] += value
        with contextlib.suppress(TypeError, ValueError):
            cost = float(record.get("total_cost_usd") or 0.0)
            if math.isfinite(cost) and cost >= 0:
                combined["total_cost_usd"] += cost

        calls = record.get("calls")
        with contextlib.suppress(TypeError, ValueError):
            if int(calls or 0) > 0:
                available = record.get("cost_available") is True
                all_cost_available = all_cost_available and available
                any_priced_usage = (
                    any_priced_usage
                    or available
                    or record.get("cost_source")
                    in {
                        "provider",
                        "partial",
                    }
                )

    combined["total_cost_usd"] = round(combined["total_cost_usd"], 6)
    combined["cost_available"] = combined["calls"] > 0 and all_cost_available and not incomplete
    combined["cost_source"] = (
        "provider" if combined["cost_available"] else "partial" if any_priced_usage else "unavailable"
    )
    return combined


__all__ = ["UsageAccumulator", "combine_usage_totals"]
