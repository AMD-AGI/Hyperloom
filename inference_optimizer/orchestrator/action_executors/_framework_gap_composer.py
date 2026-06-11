# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Compose ``gap_description`` + ``keywords`` for the framework_pr arm.

Replaces the typo-prone hand-typed ``--framework-gap`` string (session
f219629b once picked a MoE PR for a dense workload from a missing ``dense``
token) with a deterministic composer driven by structured workload data
(framework, gpu_type, model_class, precision, profile bottleneck). Pure;
the executor handles I/O. Returns ``(gap, keywords)`` so the executor can
pass both to fa or pass ``keywords=[]`` to let fa extract from gap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence


log = logging.getLogger(__name__)


# Op-name substring → canonical bottleneck keyword fed to fa. Order matters:
# first match wins so we emit exactly one keyword (not competing ones).
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

    ``breakdown_path`` is ``SharedState.last_profile_kernel_breakdown`` (a
    sorted-by-time list or dict with ``top_kernels``). Best-effort: returns ""
    when the path is empty/unreadable or no kernel matches
    :data:`_OP_TO_KEYWORD`, and the caller falls back to manifest-only gap.
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

    Same canonicalisation rule (lowercase, -/+/space → _) every other
    ``model_class`` consumer uses, keeping the gap token grep-friendly.
    """
    raw = (model_class or "").strip().lower()
    if not raw:
        return ""
    return raw.replace("-", "_").replace(" ", "_").replace("+", "_")


def _model_class_to_search_token(model_class: str) -> str:
    """Map the IO model_class taxonomy to one fa-friendly architectural token.

    fa's anti-correlation table activates on ``dense`` / ``moe``, so the gap
    must carry one of those rather than granular IO labels (``moe_mla`` ...).
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

    All workload fields are optional; missing pieces drop from the gap.
    ``precision`` comes from ``manifest.json``'s ``workload.precision``.
    ``profile_kernel_breakdown_path`` (when present) adds a bottleneck keyword.
    ``tried_refs`` is accepted for forward-compat but currently unused.

    Returns ``(gap_description, keywords)``: a free-text gap phrase for
    fa's PR search, and a lowercased/deduped/sorted explicit keyword list
    (non-empty when any of framework/gpu_type/model_class/bottleneck is known).
    """
    fw = (framework or "").strip().lower()
    gpu = (gpu_type or "").strip().lower()
    arch = _model_class_to_search_token(model_class)
    prec = (precision or "").strip().lower()
    bottleneck = _extract_bottleneck_from_breakdown(profile_kernel_breakdown_path)

    # Gap text mirrors the SKILL.md template
    # ("improve {fw} {prec} {model_class} throughput on {gpu}").
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

    # Keywords: dedup + sort, lowercase; passed as the ``keywords`` override
    # so fa skips extract_keywords and uses them verbatim.
    kw_pool: list[str] = []
    for tok in (fw, gpu, arch, prec, bottleneck):
        if tok and tok not in kw_pool:
            kw_pool.append(tok)
    keywords = sorted(kw_pool)

    return gap, keywords


__all__ = ["compose_gap"]
