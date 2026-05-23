"""Compose ``gap_description`` + ``keywords`` for the framework_pr arm.

The fa pre-stage hook (legacy, scheduled for removal) took ``--framework-gap``
as a raw command-line string and was easy to typo (a missing ``dense`` token
caused session f219629b to pick a MoE PR for a dense workload). This module
replaces that hand-typed string with a deterministic composer driven by the
structured workload data that is already in ``SharedState`` + ``manifest.json``:

* ``framework``       (sglang / vllm)
* ``gpu_type``        (mi300x / mi355x / h100 / ...)
* ``model_class``     (dense / moe_mla / moe_swa / ...)
* ``precision``       (bf16 / fp8 / awq / ...)  — sourced from manifest.workload
* ``profile_bottleneck`` (op name from latest profile kernel breakdown, if any)

The composer is pure; the framework_pr executor handles all I/O (reading the
profile breakdown JSON, building the manifest dict). Returning ``(gap, keywords)``
lets the executor pick: pass both to fa for service-side AND-search, OR pass
``keywords=[]`` to let fa extract from gap (current default for backward compat
with explicit-keyword override path).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence


log = logging.getLogger(__name__)


# Op-name → canonical bottleneck keyword. Inputs are case-insensitive substrings
# matched against the top-N kernel names from ``last_profile_kernel_breakdown``.
# Keys are the substring (lowercase); values are the keyword fed to fa for the
# rerank. Order matters: first match wins so we always emit one bottleneck
# keyword (not several competing ones that water down the AND-query).
_OP_TO_KEYWORD: tuple[tuple[str, str], ...] = (
    ("attention",   "attention"),
    ("attn",        "attention"),
    ("flash",       "attention"),
    ("moe",         "moe"),
    ("expert",      "moe"),
    ("router",      "moe"),
    ("gemm",        "gemm"),
    ("matmul",      "gemm"),
    ("rmsnorm",     "norm"),
    ("layernorm",   "norm"),
    ("rope",        "rope"),
    ("kvcache",     "kv_cache"),
    ("kv_cache",    "kv_cache"),
    ("sampler",     "sampling"),
    ("allreduce",   "comm"),
    ("nccl",        "comm"),
    ("rccl",        "comm"),
    ("quant",       "quant"),
    ("fp8",         "fp8"),
    ("int8",        "int8"),
)


def _extract_bottleneck_from_breakdown(breakdown_path: str | Path | None) -> str:
    """Read the kernel breakdown JSON and return one canonical bottleneck keyword.

    ``breakdown_path`` is expected to be the value of
    ``SharedState.last_profile_kernel_breakdown`` — the profile / pmc_roofline
    executors write this as a sorted-by-time list (or dict with ``top_kernels``).
    Returns "" when:
      * path is empty / missing / unreadable / unparseable
      * top kernel name does not match any of the substrings in :data:`_OP_TO_KEYWORD`

    The function is best-effort: a malformed payload is logged and the caller
    falls back to manifest-only gap composition.
    """
    if not breakdown_path:
        return ""
    p = Path(str(breakdown_path))
    if not p.is_file():
        return ""
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "_framework_gap_composer: could not read kernel breakdown %s: %s",
            p, exc,
        )
        return ""
    candidates: list[str] = []
    if isinstance(raw, dict):
        items = raw.get("top_kernels") or raw.get("kernels") or raw.get("rows") or []
        if isinstance(items, list):
            for item in items[:5]:
                if isinstance(item, dict):
                    nm = str(item.get("name") or item.get("kernel") or "").strip()
                    if nm:
                        candidates.append(nm.lower())
                elif isinstance(item, str):
                    candidates.append(item.lower())
    elif isinstance(raw, list):
        for item in raw[:5]:
            if isinstance(item, dict):
                nm = str(item.get("name") or item.get("kernel") or "").strip()
                if nm:
                    candidates.append(nm.lower())
            elif isinstance(item, str):
                candidates.append(item.lower())
    if not candidates:
        return ""
    for name in candidates:
        for needle, keyword in _OP_TO_KEYWORD:
            if needle in name:
                return keyword
    return ""


def _normalize_model_class(model_class: str) -> str:
    """Reduce moe_mla / moe-swa / Dense / "" to a canonical lowercase token.

    Matches the convention in scoring.MODEL_CLASS_ACTION_PRIORS so the gap
    string stays grep-friendly across the codebase.
    """
    raw = (model_class or "").strip().lower()
    if not raw:
        return ""
    return raw.replace("-", "_").replace(" ", "_").replace("+", "_")


def _model_class_to_search_token(model_class: str) -> str:
    """Map the IO model_class taxonomy to one fa-friendly architectural token.

    fa's anti-correlation table activates on tokens like ``dense`` / ``moe``
    so the gap MUST carry one of those, not the more granular IO labels
    (``moe_mla`` / ``moe_swa`` / ...).
    """
    mc = _normalize_model_class(model_class)
    if not mc:
        return ""
    if mc.startswith("moe"):
        return "moe"
    if mc == "dense":
        return "dense"
    return mc


def compose_gap(
    *,
    framework: str = "",
    gpu_type: str = "",
    model_class: str = "",
    precision: str = "",
    profile_kernel_breakdown_path: str | Path | None = None,
    tried_refs: Sequence[str] = (),
) -> tuple[str, list[str]]:
    """Build ``(gap_description, keywords)`` for the framework_pr arm.

    Parameters
    ----------
    framework, gpu_type, model_class
        Lifted from :class:`SharedState` (already populated by classify /
        baseline). All optional; missing fields are quietly dropped from the
        composed gap so the executor still has *something* to send to fa.
    precision
        Sourced from ``manifest.json``'s ``workload.precision`` because
        SharedState does not surface it directly. Empty string is fine.
    profile_kernel_breakdown_path
        Path to the JSON dumped by the profile executor (sorted-by-time top
        kernels). When present, the composer adds a bottleneck keyword like
        ``attention`` / ``moe`` / ``gemm``. Missing / unreadable → silent
        fallback to manifest-only gap.
    tried_refs
        Refs already tried this session (passed through; reserved for future
        use to bias the gap away from previously-rejected PR categories).
        Currently unused by the composer, but accepted now so callers stay
        forward-compatible.

    Returns
    -------
    ``(gap_description, keywords)`` where:
      * ``gap_description`` is a free-text phrase fa can pass to
        primus-cortex /v1/search/prs (also fed to extract_keywords as the
        fallback when explicit keywords are empty).
      * ``keywords`` is the explicit keyword list (lowercased, deduped,
        sorted for determinism) the executor passes to fa to bypass
        extract_keywords entirely. Always non-empty when at least one
        of (framework / gpu_type / model_class / bottleneck) is known.
    """
    fw = (framework or "").strip().lower()
    gpu = (gpu_type or "").strip().lower()
    arch = _model_class_to_search_token(model_class)
    prec = (precision or "").strip().lower()
    bottleneck = _extract_bottleneck_from_breakdown(profile_kernel_breakdown_path)

    # Gap text: human-readable phrasing identical to the SKILL.md default
    # template ("improve {fw} {prec} {model_class} throughput on {gpu}") so
    # operators can still cross-reference. Missing pieces drop silently;
    # extra bottleneck appears as a parenthetical so anti-correlation in fa
    # gets the hint without breaking the AND-search query.
    parts: list[str] = ["improve"]
    if fw:
        parts.append(fw)
    if prec:
        parts.append(prec)
    if arch:
        parts.append(arch)
    if bottleneck and bottleneck not in {fw, prec, arch}:
        parts.append(bottleneck)
    parts.append("throughput")
    if gpu:
        parts.extend(["on", gpu])
    gap = " ".join(parts).strip()

    # Keywords: dedup + sort, lowercase. Only include tokens fa's keyword
    # whitelist would have kept anyway (the executor passes these as the
    # ``keywords`` override so fa skips extract_keywords and uses them
    # verbatim).
    kw_pool: list[str] = []
    for tok in (fw, gpu, arch, prec, bottleneck):
        if tok and tok not in kw_pool:
            kw_pool.append(tok)
    keywords = sorted(kw_pool)

    return gap, keywords


__all__ = ["compose_gap"]
