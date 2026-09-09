# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""USD pricing for a single LLM call.

Two sources exist and they are never mixed inside one total. When the provider
reports what it charged -- the Claude CLI's ``total_cost_usd`` is the only one
that does -- that figure wins outright and the per-bucket split is prorated
from the token counts. Otherwise the cost is derived from the shipped rate card
in ``assets/llm_pricing.yaml``. A model the card does not know yields an
``unavailable`` breakdown with every figure ``None``, so a report can say
"unpriced" instead of quietly totalling zero.

The ``source`` vocabulary matches ``kernelforge/tracker/usage.py``, which
already reports provider/partial/unavailable for the same reason.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

ENV_PRICING_FILE = "HYPERLOOM_LLM_PRICING_FILE"

SOURCE_PROVIDER = "provider"
SOURCE_DERIVED = "derived"
SOURCE_PARTIAL = "partial"
SOURCE_UNAVAILABLE = "unavailable"

_PER_MILLION = 1_000_000.0

# Rate-card keys are normalized model ids; these prefixes and suffixes are the
# routing decoration a gateway adds and never part of the identity we price.
_PROVIDER_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "openai/",
    "openrouter/",
    "litellm/",
    "bedrock/",
    "vertex_ai/",
    "hosted_vllm/",
)


@dataclass(frozen=True)
class ModelRates:
    """USD per million tokens for one model."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def read_rate(self) -> float:
        """Rate charged for a cache-read token (input rate when unpriced)."""
        return self.input if self.cache_read is None else self.cache_read

    def write_rate(self) -> float:
        """Rate charged for a cache-write token (input rate when unpriced)."""
        return self.input if self.cache_write is None else self.cache_write


@dataclass(frozen=True)
class CostBreakdown:
    """What one call cost, split by the bucket that incurred it.

    ``thinking_usd`` is an addend of ``total_usd``, not a share of
    ``output_usd``: the ledger carries ``reasoning_output_tokens`` alongside
    ``output_tokens`` rather than folded into it (see
    :func:`parse_usage.reasoning_output_tokens`), so the two are priced
    separately at the same rate.
    """

    total_usd: float | None = None
    input_usd: float | None = None
    output_usd: float | None = None
    thinking_usd: float | None = None
    cache_usd: float | None = None
    source: str = SOURCE_UNAVAILABLE

    @property
    def available(self) -> bool:
        """True when a figure was produced at all."""
        return self.total_usd is not None


_UNPRICED = CostBreakdown()

_rates_cache: dict[str, ModelRates] | None = None
_rates_cache_key: str | None = None


def normalize_model(model: str | None) -> str:
    """Reduce a provider model id to the identity the rate card is keyed on.

    Strips the routing provider prefix and any ``:tag`` / ``@version`` suffix,
    then lowercases. Returns ``""`` for a missing model.

    Args:
        model: The model id as the provider reported it.

    Returns:
        The normalized id, or ``""`` when there is nothing to normalize.
    """
    name = str(model or "").strip().lower()
    if not name:
        return ""
    for prefix in _PROVIDER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.split(":", 1)[0]
    name = name.split("@", 1)[0]
    return name.strip("/ ")


def pricing_file() -> Path:
    """Return the rate-card path, honouring ``$HYPERLOOM_LLM_PRICING_FILE``."""
    override = os.environ.get(ENV_PRICING_FILE)
    if override and override.strip():
        return Path(override).expanduser()
    from hyperloom.inference_optimizer.session.paths import asset_root

    return asset_root() / "assets" / "llm_pricing.yaml"


def _parse_rates(payload: Any) -> dict[str, ModelRates]:
    """Project a loaded rate-card document onto ``{model: ModelRates}``.

    Entries without a usable input and output rate are dropped with a warning
    rather than priced at zero.

    Args:
        payload: The parsed YAML document.

    Returns:
        The rate map, empty when the document carries no usable entry.
    """
    if not isinstance(payload, Mapping):
        return {}
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), Mapping) else {}
    models = payload.get("models")
    if not isinstance(models, Mapping):
        return {}

    def _rate(entry: Mapping[str, Any], key: str) -> float | None:
        raw = entry.get(key, defaults.get(key))
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    rates: dict[str, ModelRates] = {}
    for name, entry in models.items():
        if not isinstance(entry, Mapping):
            continue
        in_rate = _rate(entry, "input")
        out_rate = _rate(entry, "output")
        if in_rate is None or out_rate is None:
            log.warning("llm_pricing: %s has no usable input/output rate; skipping", name)
            continue
        rates[normalize_model(str(name))] = ModelRates(
            input=in_rate,
            output=out_rate,
            cache_read=_rate(entry, "cache_read"),
            cache_write=_rate(entry, "cache_write"),
        )
    return rates


def load_rates(*, refresh: bool = False) -> dict[str, ModelRates]:
    """Load and memoize the rate card for the current pricing file.

    A missing or malformed card is not fatal: it yields an empty map, and every
    call then prices as ``unavailable``.

    Args:
        refresh: Re-read the file even when it is already cached.

    Returns:
        The model-id to :class:`ModelRates` map.
    """
    global _rates_cache, _rates_cache_key
    path = pricing_file()
    key = str(path)
    if not refresh and _rates_cache is not None and _rates_cache_key == key:
        return _rates_cache
    rates: dict[str, ModelRates] = {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            rates = _parse_rates(yaml.safe_load(handle))
    except FileNotFoundError:
        log.warning("llm_pricing: no rate card at %s; costs will be unavailable", path)
    except (OSError, ImportError) as exc:
        log.warning("llm_pricing: cannot read %s (%s); costs will be unavailable", path, exc)
    except Exception as exc:  # noqa: BLE001 -- yaml raises its own error tree
        log.warning("llm_pricing: %s is not valid YAML (%s)", path, exc)
    _rates_cache = rates
    _rates_cache_key = key
    return rates


def rates_for(model: str | None) -> ModelRates | None:
    """Return the rate entry for ``model`` by longest-prefix match.

    ``claude-opus-4-8-20260101`` matches the card's ``claude-opus-4`` entry;
    the longest matching key wins so a more specific entry always beats a
    family default.

    Args:
        model: The provider model id.

    Returns:
        The matching :class:`ModelRates`, or ``None`` when the card knows no
        prefix of this model.
    """
    name = normalize_model(model)
    if not name:
        return None
    rates = load_rates()
    exact = rates.get(name)
    if exact is not None:
        return exact
    best_key = ""
    for key in rates:
        if name.startswith(key) and len(key) > len(best_key):
            best_key = key
    return rates.get(best_key) if best_key else None


def _count(tokens: Mapping[str, Any] | None, key: str) -> int:
    """Read a non-negative token counter off a usage mapping."""
    if not isinstance(tokens, Mapping):
        return 0
    try:
        value = int(tokens.get(key) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def resolve_cost(
    *,
    model: str | None,
    tokens: Mapping[str, Any] | None,
    provider_usd: float | None = None,
) -> CostBreakdown:
    """Price one call, preferring the provider's own figure over the card.

    When ``provider_usd`` is given it becomes ``total_usd`` verbatim and the
    buckets are prorated by what the card says each bucket is worth -- so the
    total is always the provider's, and the split is only an attribution of it.
    When the card knows nothing about ``model`` and the provider reported
    nothing, every figure is ``None`` and ``source`` is ``unavailable``.

    Args:
        model: The provider model id the call ran on.
        tokens: Usage mapping carrying ``input_tokens``, ``output_tokens``,
            ``cache_creation_input_tokens``, ``cache_read_input_tokens`` and
            ``reasoning_output_tokens``.
        provider_usd: The provider's own charge for this call, when it
            reported one.

    Returns:
        The :class:`CostBreakdown` for the call.
    """
    rates = rates_for(model)
    has_provider = provider_usd is not None and provider_usd >= 0

    if rates is None:
        if not has_provider:
            return _UNPRICED
        return CostBreakdown(total_usd=float(provider_usd), source=SOURCE_PROVIDER)

    in_tok = _count(tokens, "input_tokens")
    out_tok = _count(tokens, "output_tokens")
    think_tok = _count(tokens, "reasoning_output_tokens")
    read_tok = _count(tokens, "cache_read_input_tokens")
    write_tok = _count(tokens, "cache_creation_input_tokens")

    input_usd = in_tok * rates.input / _PER_MILLION
    output_usd = out_tok * rates.output / _PER_MILLION
    thinking_usd = think_tok * rates.output / _PER_MILLION
    cache_usd = (read_tok * rates.read_rate() + write_tok * rates.write_rate()) / _PER_MILLION
    derived_total = input_usd + output_usd + thinking_usd + cache_usd

    if not has_provider:
        return CostBreakdown(
            total_usd=derived_total,
            input_usd=input_usd,
            output_usd=output_usd,
            thinking_usd=thinking_usd,
            cache_usd=cache_usd,
            source=SOURCE_DERIVED,
        )

    total = float(provider_usd)
    scale = (total / derived_total) if derived_total > 0 else 0.0
    return CostBreakdown(
        total_usd=total,
        input_usd=input_usd * scale,
        output_usd=output_usd * scale,
        thinking_usd=thinking_usd * scale,
        cache_usd=cache_usd * scale,
        source=SOURCE_PROVIDER,
    )


__all__ = [
    "ENV_PRICING_FILE",
    "SOURCE_DERIVED",
    "SOURCE_PARTIAL",
    "SOURCE_PROVIDER",
    "SOURCE_UNAVAILABLE",
    "CostBreakdown",
    "ModelRates",
    "load_rates",
    "normalize_model",
    "pricing_file",
    "rates_for",
    "resolve_cost",
]
