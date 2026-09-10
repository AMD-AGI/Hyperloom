# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose ``gap_description`` + ``keywords`` for the framework arm."""

from __future__ import annotations

import json
import logging
from pathlib import Path


log = logging.getLogger(__name__)


# Only the hottest kernels are scanned; the breakdown is time-sorted.
_MAX_KERNELS_SCANNED = 5


# Op-name substring -> canonical bottleneck keyword; first match wins.
_OP_TO_KEYWORD: tuple[tuple[str, str], ...] = (
    ("attention", "attention"),
    ("attn", "attention"),
    ("flash", "attention"),
    ("moe", "moe"),
    ("expert", "moe"),
    ("router", "moe"),
    ("gemm", "gemm"),
    ("matmul", "gemm"),
    ("rmsnorm", "norm"),
    ("layernorm", "norm"),
    ("rope", "rope"),
    ("kvcache", "kv_cache"),
    ("kv_cache", "kv_cache"),
    ("sampler", "sampling"),
    ("allreduce", "comm"),
    ("nccl", "comm"),
    ("rccl", "comm"),
    ("quant", "quant"),
    ("fp8", "fp8"),
    ("int8", "int8"),
)


# Rewrite-evidence category -> canonical gap keyword.
_CATEGORY_TO_KEYWORD: tuple[tuple[str, str], ...] = (
    ("eliminate_host_round_trip", "collective_rendezvous"),
    ("eliminate_host_sync", "host_sync"),
    ("fuse_collectives", "collective_fusion"),
    ("keep_device_resident", "host_to_device"),
    ("memoize_invariant", "recomputation"),
    ("hoist_loop_invariant", "loop_invariant"),
)


def _extract_bottleneck_from_rewrite_evidence(evidence_path: str | Path | None) -> str:
    """Read the host-side rewrite evidence and return its top bottleneck keyword."""
    if not evidence_path:
        return ""
    path = Path(str(evidence_path))
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "_framework_gap_composer: could not read rewrite evidence %s: %s",
            path,
            exc,
        )
        return ""
    candidates = raw.get("candidates") if isinstance(raw, dict) else None
    if not isinstance(candidates, list):
        return ""
    for item in candidates:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower()
        for needle, keyword in _CATEGORY_TO_KEYWORD:
            if category == needle:
                return keyword
    return ""


def _extract_bottleneck_from_breakdown(breakdown_path: str | Path | None) -> str:
    """Read the kernel breakdown JSON and return one canonical bottleneck keyword."""
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
            p,
            exc,
        )
        return ""
    candidates: list[str] = []
    if isinstance(raw, dict):
        items = raw.get("top_kernels") or raw.get("kernels") or raw.get("rows") or []
        if isinstance(items, list):
            for item in items[:_MAX_KERNELS_SCANNED]:
                if isinstance(item, dict):
                    nm = str(item.get("name") or item.get("kernel") or "").strip()
                    if nm:
                        candidates.append(nm.lower())
                elif isinstance(item, str):
                    candidates.append(item.lower())
    elif isinstance(raw, list):
        for item in raw[:_MAX_KERNELS_SCANNED]:
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
    """Reduce moe_mla / moe-swa / Dense / \"\" to a canonical lowercase token."""
    raw = (model_class or "").strip().lower()
    if not raw:
        return ""
    return raw.replace("-", "_").replace(" ", "_").replace("+", "_")


def _model_class_to_search_token(model_class: str) -> str:
    """Map the IO model_class taxonomy to one fa-friendly architectural token."""
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
    rewrite_evidence_path: str | Path | None = None,
) -> tuple[str, list[str]]:
    """Build ``(gap_description, keywords)`` for the framework arm."""
    fw = (framework or "").strip().lower()
    gpu = (gpu_type or "").strip().lower()
    arch = _model_class_to_search_token(model_class)
    prec = (precision or "").strip().lower()
    bottleneck = _extract_bottleneck_from_breakdown(profile_kernel_breakdown_path)
    host_bottleneck = _extract_bottleneck_from_rewrite_evidence(rewrite_evidence_path)

    # Gap text mirrors the SKILL.md template.
    parts: list[str] = ["improve"]
    if fw:
        parts.append(fw)
    if prec:
        parts.append(prec)
    if arch:
        parts.append(arch)
    for token in (bottleneck, host_bottleneck):
        if token and token not in parts:
            parts.append(token)
    parts.append("throughput")
    if gpu:
        parts.extend(["on", gpu])
    gap = " ".join(parts).strip()

    # Keywords: dedup + sort, lowercase; passed as the ``keywords`` override.
    kw_pool: list[str] = []
    for tok in (fw, gpu, arch, prec, bottleneck, host_bottleneck):
        if tok and tok not in kw_pool:
            kw_pool.append(tok)
    keywords = sorted(kw_pool)

    return gap, keywords


__all__ = ["compose_gap"]
