# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement definitions shared by the Meta RSI scripts: token weighting, list-price costs and
what counts as a polling turn.

Weighted tokens count each token class at its list-price ratio to an input token (cache write
1.25, cache read 0.1, output 5), so one number follows spend across models. ``PRICES`` holds
assumed list prices, reported as parameters; GLM is self-hosted and priced at 0.
"""

from __future__ import annotations

import re

TOKEN_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
# USD per 1M tokens: (input, cache write, cache read, output).
PRICES = {
    "opus": (5.0, 6.25, 0.50, 25.0),
    "sonnet": (3.0, 3.75, 0.30, 15.0),
    "gpt": (1.25, 1.25, 0.125, 10.0),
    "gemini": (1.25, 1.25, 0.31, 10.0),
    "glm": (0.0, 0.0, 0.0, 0.0),
}
POLL = re.compile(
    r"\bsleep\s+\d|tail -[fF]\b|\bwatch\b|\bps\s+(-|aux)|rocm-smi|nvidia-smi|curl\s+-s[^|]*(health|v1/models)|heartbeat"
)


def weighted(usage: dict) -> float:
    return (
        (usage.get("input_tokens") or 0)
        + 1.25 * (usage.get("cache_creation_input_tokens") or 0)
        + 0.1 * (usage.get("cache_read_input_tokens") or 0)
        + 5 * (usage.get("output_tokens") or 0)
    )


def context_tokens(usage: dict) -> int:
    """Prompt size of one call: plain, cache-written and cache-read input."""
    return (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )


def price_family(model: str | None) -> str | None:
    """The ``PRICES`` family a model id names as one of its parts (``claude-opus-5`` -> ``opus``);
    None when it names no family or more than one."""
    parts = set(re.split(r"[^a-z0-9]+", (model or "").lower()))
    found = [family for family in PRICES if family in parts]
    return found[0] if len(found) == 1 else None


def cost_usd(usage: dict, family: str | None) -> float:
    """List-price cost; 0 for a model with no priced family, which callers must report as unpriced."""
    if family is None:
        return 0.0
    p = PRICES[family]
    return (
        (usage.get("input_tokens") or 0) * p[0]
        + (usage.get("cache_creation_input_tokens") or 0) * p[1]
        + (usage.get("cache_read_input_tokens") or 0) * p[2]
        + (usage.get("output_tokens") or 0) * p[3]
    ) / 1e6
