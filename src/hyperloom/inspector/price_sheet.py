# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the token timeline is priced by: the built-in price sheet multi-branch spends by
(:mod:`hyperloom_common.llm.llm_prices`), read as it is, so a call costs what multi-branch records for the same tokens:
the same lookup (:func:`price_of`) and the same rates, the cache-write fallback included. The sheet has no price per web
search, so a call that searched is unpriced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloom_common.llm.llm_prices import PER_TOKENS, ModelPrice, model_key, price_of, sheet_source

_SOURCE = "hyperloom_common/llm/llm_prices.json"
_NO_CACHE_WRITE = "input rate: the sheet lists no cache write for it"
_KINDS = ("input", "cache_write", "cache_read", "output")


@dataclass(frozen=True)
class SheetPrice:
    """A model's rates on the sheet, as the token timeline reads a price."""

    id: str
    input: float
    cache_write: float
    cache_read: float
    output: float
    cache_basis: str

    @classmethod
    def of(cls, name: str, price: ModelPrice) -> SheetPrice:
        basis = "listed" if price.cache_write is not None else _NO_CACHE_WRITE
        return cls(name, price.input, price.write_rate, price.cache_read, price.output, basis)

    def rates(self) -> dict[str, float]:
        """USD per token by token kind; reasoning is billed as output."""
        per = {k: getattr(self, k) / PER_TOKENS for k in _KINDS}
        return {**per, "reasoning": per["output"]}


class PriceSheet:
    """The sheet as the token timeline reads it: every model it lists, at its rates, and what the page says of it."""

    source = _SOURCE

    @property
    def built_from(self) -> str:
        """What the sheet says its rates come from."""
        return sheet_source()

    def lookup(self, model: Any) -> SheetPrice | None:
        """The sheet's price of *model*, None for a model it does not list."""
        name = str(model or "")
        price = price_of(name)
        return SheetPrice.of(model_key(name), price) if price is not None else None


SHEET = PriceSheet()
