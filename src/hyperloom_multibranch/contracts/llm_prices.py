# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an LLM's tokens cost through the AMD LLM gateway, so every recorded spend uses one price sheet.

The sheet is ``llm_prices.json`` beside this module: the gateway's rate per million tokens of each model a session
may run, for uncached input, output, cache reads and cache writes (5-minute, and 1-hour where the gateway lists one).
A model is looked up by its name with case, dots, a date suffix and a ``[...]`` variant suffix ignored, so
``Claude-Opus-5.5``, ``claude-opus-5-5`` and ``claude-opus-5-5-20261001`` are one model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

PRICE_SHEET = Path(__file__).with_name("llm_prices.json")
PER_TOKENS = 1_000_000
_SUFFIXES = re.compile(r"(\[[^\]]*\]|-\d{8})$")


class UnpricedModel(ValueError):
    """A model the price sheet has no rate for."""


@dataclass(frozen=True)
class TokenCounts:
    """Tokens of one or more calls, by how they are billed; ``input`` is the uncached prompt only."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) < 0 for item in fields(self)):
            raise ValueError(f"token counts are at least 0, not {self!r}")

    @property
    def prompt(self) -> int:
        """Every prompt token, cached or not."""
        return self.input + self.cache_read + self.cache_write + self.cache_write_1h

    def __add__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(*(getattr(self, item.name) + getattr(other, item.name) for item in fields(self)))

    @classmethod
    def from_anthropic_usage(cls, usage: Mapping[str, object]) -> TokenCounts:
        """The counts of an Anthropic ``usage`` block. A split ``cache_creation`` bills each cache write by its
        lifetime; without it every cache write is a 5-minute one. A missing or unusable count is 0."""
        split = usage.get("cache_creation")
        split = split if isinstance(split, Mapping) else {}
        written = _count(split, "ephemeral_5m_input_tokens") if split else _count(usage, "cache_creation_input_tokens")
        return cls(
            _count(usage, "input_tokens"),
            _count(usage, "output_tokens"),
            _count(usage, "cache_read_input_tokens"),
            written,
            _count(split, "ephemeral_1h_input_tokens"),
        )


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. A cache write the gateway lists no rate for is billed as the next rate it lists:
    a 1-hour write as a 5-minute one, a 5-minute write as uncached input."""

    input: float
    output: float
    cache_read: float
    cache_write: float | None = None
    cache_write_1h: float | None = None

    def cost_usd(self, tokens: TokenCounts) -> float:
        write = self.input if self.cache_write is None else self.cache_write
        write_1h = write if self.cache_write_1h is None else self.cache_write_1h
        dollars = (
            tokens.input * self.input
            + tokens.output * self.output
            + tokens.cache_read * self.cache_read
            + tokens.cache_write * write
            + tokens.cache_write_1h * write_1h
        )
        return dollars / PER_TOKENS


def model_key(model: str) -> str:
    """The sheet's name for ``model``: lower case, dots as dashes, without a date or ``[...]`` suffix."""
    return _SUFFIXES.sub("", model.strip().lower()).replace(".", "-")


def price_of(model: str) -> ModelPrice:
    """The gateway's rates for ``model``; raises :class:`UnpricedModel` for a model the sheet does not list."""
    found = _sheet().get(model_key(model))
    if found is None:
        raise UnpricedModel(f"the price sheet ({PRICE_SHEET.name}) has no rate for model {model!r}")
    return found


def _count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


@cache
def _sheet() -> dict[str, ModelPrice]:
    models = json.loads(PRICE_SHEET.read_text(encoding="utf-8"))["models"]
    return {name: ModelPrice(**rates) for name, rates in models.items()}


__all__ = ["PER_TOKENS", "PRICE_SHEET", "ModelPrice", "TokenCounts", "UnpricedModel", "model_key", "price_of"]
