# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""recipe-snapshot v2 HTTP wire constants — single source of truth."""

from __future__ import annotations

import logging
from typing import Final

log = logging.getLogger(__name__)


# No default remote URL by design: Recipe remote mode requires explicit ``KB_STORE_URL`` and ``KB_STORE_TOKEN``
# configuration.

# Request body field names — PUT /recipes/{canonical_id}.
F_AUTHORITY: Final[str] = "authority"
F_CONFIDENCE: Final[str] = "confidence"
F_EVIDENCE_REFS: Final[str] = "evidence_refs"
F_PROVENANCE: Final[str] = "provenance"

# Canonical identity dimensions inside ``labels`` mirroring recipe_canonical_id so /recipes/search can filter by
# dimension.
F_LABEL_MODEL: Final[str] = "model"
F_LABEL_HARDWARE: Final[str] = "hardware"
F_LABEL_FRAMEWORK_NAME: Final[str] = "framework_name"
F_LABEL_FRAMEWORK_VERSION: Final[str] = "framework_version"
F_LABEL_PRECISION: Final[str] = "precision"
F_LABEL_MODEL_TYPE: Final[str] = "model_type"
F_LABEL_ARCHITECTURES: Final[str] = "architectures"

# PUT response fields
F_CANONICAL_ID: Final[str] = "canonical_id"
F_VERSION: Final[str] = "version"

# MetricFilter sub-fields
F_METRIC_MIN: Final[str] = "min"
F_METRIC_MAX: Final[str] = "max"

# Provenance sub-fields
F_PV_DETAILS: Final[str] = "details"


# Enum literals (strict server-side enums; unknown values -> 422).
AUTHORITY_EXPERIENTIAL: Final[str] = "EXPERIENTIAL"

# Search order_by values accepted by ``/recipes/search``.
ORDER_BY_UPDATED_AT_DESC: Final[str] = "updated_at DESC"
ORDER_BY_UPDATED_AT_ASC: Final[str] = "updated_at ASC"
ORDER_BY_CREATED_AT_ASC: Final[str] = "created_at ASC"


# Foreground Recipe KB reads use one fail-fast attempt on the Coordinator loop.
FOREGROUND_HTTP_TIMEOUT_SEC: Final[float] = 2.0  # Coordinator main loop

# Default confidence when caller doesn't override (spec [0.0, 1.0]).
DEFAULT_CONFIDENCE: Final[float] = 0.85


# Default-slug constants for missing identity components.
DEFAULT_MODEL_SLUG: Final[str] = "unknown_model"
DEFAULT_HARDWARE_SLUG: Final[str] = "unknown_hw"
DEFAULT_FRAMEWORK_SLUG: Final[str] = "unknown_framework"
DEFAULT_FRAMEWORK_VERSION_SLUG: Final[str] = "unknown_version"
DEFAULT_PRECISION_SLUG: Final[str] = "unknown_precision"
DEFAULT_MODEL_TYPE_SLUG: Final[str] = "unknown_model_type"
DEFAULT_ARCHITECTURES_SLUG: Final[str] = "unknown_arch"


def _slug(value: str, default: str) -> str:
    """Lowercase + basename + space/tab/slash -> underscore."""
    raw = (value or "").strip()
    if not raw:
        return default
    if "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1] or raw
    cleaned = raw.lower()
    for ch in (" ", "\t", "/"):
        cleaned = cleaned.replace(ch, "_")
    # A dot-only result ("." / "..") is a traversal component, not an identity.
    return cleaned if cleaned.strip(".") else default


def _architectures_slug(value: "str | list[str]") -> str:
    """Serialize an architectures value into a stable slug for canonical_id."""
    if isinstance(value, list):
        parts = sorted(_slug(v, "") for v in value if (v or "").strip())
        return "+".join(parts) if parts else DEFAULT_ARCHITECTURES_SLUG
    return _slug(str(value), DEFAULT_ARCHITECTURES_SLUG)


def recipe_canonical_id(
    *,
    model: str,
    hardware: str,
    framework_name: str,
    model_type: str = "",
    architectures: "str | list[str]" = "",
    framework_version: str,
    precision: str,
) -> str:
    """Build the recipe ``canonical_id``:
    ``inference:{model}:{hardware}:{framework_name}:{model_type}:{architectures}:{framework_version}:{precision}``.
    """
    return (
        f"inference:"
        f"{_slug(model, DEFAULT_MODEL_SLUG)}:"
        f"{_slug(hardware, DEFAULT_HARDWARE_SLUG)}:"
        f"{_slug(framework_name, DEFAULT_FRAMEWORK_SLUG)}:"
        f"{_slug(model_type, DEFAULT_MODEL_TYPE_SLUG)}:"
        f"{_architectures_slug(architectures)}:"
        f"{_slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG)}:"
        f"{_slug(precision, DEFAULT_PRECISION_SLUG)}"
    )


def kb_hardware_slug(
    gpu_type: str,
    *,
    nodes: int = 1,
    gpus_per_node: int = 8,
    pd_mode: str = "aggregated",
    pd_prefill_nodes: int = 0,
    pd_decode_nodes: int = 0,
    tp: int = 0,
    ep: int = 0,
    backend: str = "",
) -> str:
    """Topology-aware hardware dimension for the recipe ``canonical_id``."""
    base = (gpu_type or "").strip()
    try:
        n = int(nodes)
    except (TypeError, ValueError):
        return base
    if n < 2:
        return base
    try:
        ws = n * int(gpus_per_node)
    except (TypeError, ValueError):
        return base
    if ws <= 0:
        return base
    slug = f"{base}_ws{ws}"
    if str(pd_mode or "").strip().lower() == "disaggregated":
        try:
            pn = int(pd_prefill_nodes)
            dn = int(pd_decode_nodes)
        except (TypeError, ValueError):
            pn = dn = 0
        if pn > 0 and dn > 0:
            slug = f"{slug}_pd{pn}p{dn}d"
    try:
        tp_i = int(tp)
    except (TypeError, ValueError):
        tp_i = 0
    if tp_i > 0:
        slug = f"{slug}_tp{tp_i}"
    try:
        ep_i = int(ep)
    except (TypeError, ValueError):
        ep_i = 0
    if ep_i > 1:
        slug = f"{slug}_ep{ep_i}"
    be = _slug(backend, "")
    if be:
        slug = f"{slug}_{be}"
    return slug


def canonical_labels(
    *,
    model: str,
    hardware: str,
    framework_name: str,
    model_type: str = "",
    architectures: "str | list[str]" = "",
    framework_version: str,
    precision: str,
) -> dict[str, str]:
    """Return the 7-key ``labels`` dict mirroring the canonical id, so"""
    return {
        F_LABEL_MODEL: _slug(model, DEFAULT_MODEL_SLUG),
        F_LABEL_HARDWARE: _slug(hardware, DEFAULT_HARDWARE_SLUG),
        F_LABEL_FRAMEWORK_NAME: _slug(framework_name, DEFAULT_FRAMEWORK_SLUG),
        F_LABEL_MODEL_TYPE: _slug(model_type, DEFAULT_MODEL_TYPE_SLUG),
        F_LABEL_ARCHITECTURES: _architectures_slug(architectures),
        F_LABEL_FRAMEWORK_VERSION: _slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG),
        F_LABEL_PRECISION: _slug(precision, DEFAULT_PRECISION_SLUG),
    }


# framework_name slug -> python package whose __version__ is authoritative.
_FRAMEWORK_VERSION_MODULES: Final[dict[str, str]] = {
    "sglang": "sglang",
    "vllm": "vllm",
    "atom": "atom",  # vendor-internal; __version__ is often a git hash
}


def detect_framework_version(framework_name: str) -> str:
    """Best-effort installed version of ``framework_name`` via importing its"""
    fw_slug = _slug(framework_name, "")
    if not fw_slug:
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    module_name = _FRAMEWORK_VERSION_MODULES.get(fw_slug)
    if not module_name:
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    try:
        import importlib

        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — broad on purpose
        log.debug(
            "detect_framework_version: import %r failed (%s); falling back to %r",
            module_name,
            exc,
            DEFAULT_FRAMEWORK_VERSION_SLUG,
        )
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    raw = getattr(mod, "__version__", "") or ""
    if not raw:  # fall back to VERSION
        raw = getattr(mod, "VERSION", "") or ""
    return _slug(str(raw), DEFAULT_FRAMEWORK_VERSION_SLUG)


__all__ = [
    "F_AUTHORITY",
    "F_CONFIDENCE",
    "F_EVIDENCE_REFS",
    "F_PROVENANCE",
    "F_LABEL_MODEL",
    "F_LABEL_HARDWARE",
    "F_LABEL_FRAMEWORK_NAME",
    "F_LABEL_FRAMEWORK_VERSION",
    "F_LABEL_PRECISION",
    "F_LABEL_MODEL_TYPE",
    "F_LABEL_ARCHITECTURES",
    "F_CANONICAL_ID",
    "F_VERSION",
    "F_METRIC_MIN",
    "F_METRIC_MAX",
    "F_PV_DETAILS",
    "AUTHORITY_EXPERIENTIAL",
    "ORDER_BY_UPDATED_AT_DESC",
    "ORDER_BY_UPDATED_AT_ASC",
    "ORDER_BY_CREATED_AT_ASC",
    "FOREGROUND_HTTP_TIMEOUT_SEC",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_FRAMEWORK_SLUG",
    "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "recipe_canonical_id",
    "kb_hardware_slug",
    "canonical_labels",
    "detect_framework_version",
]
