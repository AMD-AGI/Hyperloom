# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the whole accepted recipe, and the prose a session left behind.

``mine.py`` reads a parallelism layout out of an accepted server-arg string,
because ranking layouts is the parallelism question the Recipe KB can answer.
But a layout is a small part of a recipe. On the five live Kimi-K3 sessions
every measured win came from somewhere else entirely -- KV cache dtype, mamba
pool sizing, request-pool occupancy, chunked prefill, attention backend, an
AITER env flag and one decode-kernel patch -- and the parallelism extractor saw
none of it, reporting all five as ``framework-default``. Technically true, and
useless.

Two things follow, and this module does both.

First, extract *every* accepted knob and env var, classified by family, so a
recipe is legible even when it re-shards nothing.

Second, treat a knob's value as scope-bound. ``--max-running-requests 64``
measured at concurrency 64 is not a value to copy to concurrency 1, it is a
rule -- size the pool to the load -- that has to be re-applied. Knobs carry
that coupling as a note so a reader re-derives the value instead of inheriting
it. The same applies to the prose: a lesson learned at one concurrency can
invert at another, and one live example says so outright, warning against
importing a vendor's low ``max-num-seqs`` *because* the session ran at 64.

Presence in an accepted recipe is not attribution. A session's gain belongs to
the whole stack it accepted, so a knob seen in a 61% session did not
necessarily earn 61%.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

#: Value recorded for a flag that carries no argument.
BOOLEAN_VALUE = "on"

#: Knob-name fragments mapped to a family, tried in order. Parallelism leads
#: because ``--moe-dense-tp-size`` is a sharding knob that also matches ``moe``;
#: mamba precedes cache because ``--mamba-radix-cache-strategy`` matches both;
#: kv_cache precedes precision because ``--kv-cache-dtype`` matches ``dtype``.
_FAMILY_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "tp_size",
            "tensor_parallel",
            "dp_size",
            "data_parallel",
            "ep_size",
            "expert_parallel",
            "pp_size",
            "pipeline_parallel",
            "moe_dense_tp",
            "decode_tp",
            "prefill_tp",
            "dp_attention",
            "dp_lm_head",
            "ep_moe",
            "deepep_moe",
            "nnodes",
        ),
        "parallelism",
    ),
    (("speculative", "spec_", "draft"), "speculative"),
    (("kv_cache", "kv_offload", "kv_transfer"), "kv_cache"),
    (("mamba",), "mamba"),
    (("cuda_graph", "cudagraph"), "cudagraph"),
    (("attention", "mla"), "attention"),
    (("moe", "expert"), "moe"),
    (("radix_cache", "prefix_caching"), "cache"),
    (("mem_fraction", "memory_utilization", "memory", "mem_"), "memory"),
    (
        (
            "max_running_requests",
            "max_num_seqs",
            "max_num_batched_tokens",
            "chunked_prefill",
            "schedul",
        ),
        "scheduling",
    ),
    # A watchdog is a timeout, not a pool size, so it must not pick up the
    # concurrency-sizing note that the scheduling family carries.
    (("watchdog", "timeout", "keep_alive"), "timeout"),
    (("dtype", "quant"), "precision"),
    (("context_length", "max_model_len", "context"), "context"),
)

#: Families whose correct value is a function of the workload's concurrency. A
#: value from one concurrency is a rule to re-apply, not a setting to copy.
_CONCURRENCY_SIZED = frozenset({"scheduling", "cudagraph", "mamba"})

#: Prose fields a session leaves behind, in the order they are worth reading.
LEARNING_FIELDS = ("what_worked", "what_failed", "lessons", "pitfalls", "remaining_gaps")

_STATEMENT_KEYS = ("statement", "description", "symptom", "name", "variant_name", "error_class")
_DOMAIN_KEYS = ("domain_hint", "phase", "layer", "error_class", "action")
#: Research hints run long and the length is the value; capped only so one
#: pathological record cannot dominate a report.
_MAX_STATEMENT = 2000

_CONC_RE = re.compile(r"concurrenc\w*\s*(?:of\s+|=\s*|is\s+)?(\d+)", re.IGNORECASE)


def _normalize(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def knob_family(name: str) -> str:
    """Classify a knob by what it controls, e.g. ``kv_cache`` or ``mamba``."""
    key = _normalize(name)
    for fragments, family in _FAMILY_PATTERNS:
        if any(fragment in key for fragment in fragments):
            return family
    return "other"


def extract_knobs(server_args: str) -> dict[str, str]:
    """Pull every accepted flag out of a server-arg string.

    Handles ``--flag value`` and ``--flag=value``; a flag with no argument
    records :data:`BOOLEAN_VALUE`. Values are kept as written, since a knob's
    exact spelling is what a reader has to reproduce.
    """
    tokens = str(server_args or "").split()
    knobs: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name, _, inline = token[2:].partition("=")
        key = _normalize(name)
        if not key:
            continue
        value = inline.strip()
        if not value and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1].strip()
        knobs[key] = value or BOOLEAN_VALUE
    return knobs


def scope_coupling(name: str, value: Any, shape: Mapping[str, Any] | None) -> list[str]:
    """Note why a knob's value is bound to the scope it was measured at."""
    notes: list[str] = []
    shape = shape or {}
    for dimension in ("conc", "tp", "isl", "osl"):
        wanted = shape.get(dimension)
        if wanted and str(value).strip() == str(wanted):
            notes.append(f"value {value} equals {dimension}={wanted}, so it was sized to this scope")
    family = knob_family(name)
    conc = shape.get("conc")
    if family in _CONCURRENCY_SIZED and conc:
        notes.append(f"{family} knobs are sized to concurrency; measured at conc={conc}")
    return notes


def concurrency_mentions(text: Any) -> list[int]:
    """Concurrencies a statement names, so a reader sees what it was about."""
    return sorted({int(match) for match in _CONC_RE.findall(str(text or ""))})


def _first_str(item: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _learning(kind: str, item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        statement = item.strip()
        if not statement:
            return None
        return {"kind": kind, "statement": statement[:_MAX_STATEMENT], "gain_pct": None, "severity": None, "domain": ""}
    if not isinstance(item, Mapping):
        return None
    statement = _first_str(item, _STATEMENT_KEYS)
    if not statement:
        return None
    gain = _finite(item.get("gain_pct"))
    if gain is None:
        impact = item.get("measured_impact")
        if isinstance(impact, Mapping):
            gain = _finite(impact.get("gain_pct"))
    severity = _first_str(item, ("severity",)) or ("failed" if kind == "what_failed" else "")
    record: dict[str, Any] = {
        "kind": kind,
        "statement": statement[:_MAX_STATEMENT],
        "gain_pct": gain,
        "severity": severity or None,
        "domain": _first_str(item, _DOMAIN_KEYS),
    }
    provenance = _first_str(item, ("provenance",))
    if provenance:
        record["provenance"] = provenance[:_MAX_STATEMENT]
    return record


def extract_learnings(knowledge: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize the prose fields into comparable records.

    These are the fields that actually answer "what did we learn here", and
    the estimator ignored all five until a cold target made that obvious.
    """
    out: list[dict[str, Any]] = []
    for kind in LEARNING_FIELDS:
        items = knowledge.get(kind)
        if isinstance(items, (Mapping, str)):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            record = _learning(kind, item)
            if record is not None:
                out.append(record)
    return out


__all__ = [
    "BOOLEAN_VALUE",
    "LEARNING_FIELDS",
    "concurrency_mentions",
    "extract_knobs",
    "extract_learnings",
    "knob_family",
    "scope_coupling",
]
