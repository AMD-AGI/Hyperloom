# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose ``gap_description`` + ``keywords`` for the framework arm."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ._framework_rewrite_evidence import (
    CATEGORY_DEVICE_RESIDENT,
    CATEGORY_FUSE_COLLECTIVES,
    CATEGORY_HOIST,
    CATEGORY_HOST_ROUND_TRIP,
    CATEGORY_HOST_SYNC,
    CATEGORY_MEMOIZE,
)


log = logging.getLogger(__name__)


# Rewrite-evidence category -> canonical gap keyword.
_CATEGORY_TO_KEYWORD: dict[str, str] = {
    CATEGORY_HOST_ROUND_TRIP: "collective_rendezvous",
    CATEGORY_HOST_SYNC: "host_sync",
    CATEGORY_FUSE_COLLECTIVES: "collective_fusion",
    CATEGORY_DEVICE_RESIDENT: "host_to_device",
    CATEGORY_MEMOIZE: "recomputation",
    CATEGORY_HOIST: "loop_invariant",
}


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
        keyword = _CATEGORY_TO_KEYWORD.get(str(item.get("category") or "").strip().lower())
        if keyword:
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
    rewrite_evidence_path: str | Path | None = None,
) -> tuple[str, list[str]]:
    """Build ``(gap_description, keywords)`` for the framework arm; an absent evidence path contributes no token."""
    fw = (framework or "").strip().lower()
    gpu = (gpu_type or "").strip().lower()
    arch = _model_class_to_search_token(model_class)
    prec = (precision or "").strip().lower()
    host_bottleneck = _extract_bottleneck_from_rewrite_evidence(rewrite_evidence_path)

    # Gap text mirrors the SKILL.md template.
    parts: list[str] = ["improve"]
    if fw:
        parts.append(fw)
    if prec:
        parts.append(prec)
    if arch:
        parts.append(arch)
    if host_bottleneck and host_bottleneck not in parts:
        parts.append(host_bottleneck)
    parts.append("throughput")
    if gpu:
        parts.extend(["on", gpu])
    gap = " ".join(parts).strip()

    # Keywords: dedup + sort, lowercase; passed as the ``keywords`` override.
    kw_pool: list[str] = []
    for tok in (fw, gpu, arch, prec, host_bottleneck):
        if tok and tok not in kw_pool:
            kw_pool.append(tok)
    keywords = sorted(kw_pool)

    return gap, keywords


__all__ = ["compose_gap"]
