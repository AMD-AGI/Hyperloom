# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""recipe-snapshot v2 HTTP wire constants — single source of truth.

Mirrors the contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md``. The
recipe-KB remote clients (see :mod:`hyperloom.orchestrator.knowledge.recipe_kb`
and :mod:`hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client`) build requests as
plain dicts keyed by these ``Final[str]`` constants so a backend rename surfaces
at one easy-to-grep call-site instead of being scattered across the codebase.
"""

from __future__ import annotations

import logging
from typing import Final

log = logging.getLogger(__name__)


# No DEFAULT_KB_URL by design: the optimizer never silently connects to a
# remote KB; a remote source is used only with --cortex-kb-url / $CORTEX_KB_URL.

# Request body field names — PUT /recipes/{canonical_id}. Top-level fields;
# server validates only authority + provenance (rest are caller-defined).
F_AUTHORITY: Final[str] = "authority"
F_CONFIDENCE: Final[str] = "confidence"
F_EVIDENCE_REFS: Final[str] = "evidence_refs"
F_PROVENANCE: Final[str] = "provenance"

# Canonical identity dimensions inside ``labels`` mirroring
# recipe_canonical_id; stamp them so /recipes/search can filter by dimension.
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

# Search order_by whitelist — exactly 6 values per spec.
ORDER_BY_UPDATED_AT_DESC: Final[str] = "updated_at DESC"
ORDER_BY_UPDATED_AT_ASC: Final[str] = "updated_at ASC"
ORDER_BY_CREATED_AT_ASC: Final[str] = "created_at ASC"


# Defaults — two timeout/retry profiles. Foreground (Coordinator main loop)
# fails fast (2s + 1 retry) and falls through to NDJSON; background (flusher /
# CLOSE drain) uses the larger 10s x 3 budget.
DEFAULT_HTTP_TIMEOUT_SEC: Final[float] = 10.0  # background / flusher
FOREGROUND_HTTP_TIMEOUT_SEC: Final[float] = 2.0  # Coordinator main loop

# Default confidence when caller doesn't override (spec [0.0, 1.0]).
DEFAULT_CONFIDENCE: Final[float] = 0.85


# Canonical id derivation. Default-slug constants for missing identity
# components, named so the audit log + /search corpus stay grep-stable.
DEFAULT_MODEL_SLUG: Final[str] = "unknown_model"
DEFAULT_HARDWARE_SLUG: Final[str] = "unknown_hw"
DEFAULT_FRAMEWORK_SLUG: Final[str] = "unknown_framework"
DEFAULT_FRAMEWORK_VERSION_SLUG: Final[str] = "unknown_version"
DEFAULT_PRECISION_SLUG: Final[str] = "unknown_precision"
DEFAULT_MODEL_TYPE_SLUG: Final[str] = "unknown_model_type"
DEFAULT_ARCHITECTURES_SLUG: Final[str] = "unknown_arch"


def _slug(value: str, default: str) -> str:
    """Lowercase + basename + space/tab/slash -> underscore.

    Slugged for lookup stability (``--model /path/Qwen3`` and ``qwen3`` must
    converge) and filesystem safety in the local KB store. ``/`` resolves to
    the basename first (HF paths collapse to the stem).

    Args:
        value: The raw value to slugify.
        default: Fallback slug returned when ``value`` is empty.

    Returns:
        The slugified value, or ``default`` when empty.
    """
    raw = (value or "").strip()
    if not raw:
        return default
    if "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1] or raw
    cleaned = raw.lower()
    for ch in (" ", "\t", "/"):
        cleaned = cleaned.replace(ch, "_")
    return cleaned or default


def _architectures_slug(value: "str | list[str]") -> str:
    """Serialize an architectures value into a stable slug for canonical_id.

    Accepts a list (from config.json) or a pre-slugged string. Lists are
    sorted for determinism and joined with ``+``.
    """
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

    8 colon-separated segments: 1 prefix + 7 identity dimensions.
    Dimension order reflects fallback priority: model is dropped first
    (cross-model same-architecture reuse), then framework_version.
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
) -> str:
    """Topology-aware hardware dimension for the recipe ``canonical_id``.

    Single-node (``nodes < 2``) returns ``gpu_type`` UNCHANGED, so existing
    single-node recipe keys and their KB data are byte-for-byte preserved.

    Multi-node (``nodes >= 2``) encodes the deployment topology so structurally
    different formations never share a key (and never overwrite each other's
    ``best_config``):

    * ``_ws{world_size}`` (``world_size = nodes * gpus_per_node``) — physical
      scale, always present multi-node.
    * ``_pd{pn}p{dn}d`` — appended ONLY for prefill/decode disaggregation
      (``pd_mode == "disaggregated"``), since a disaggregated config carries
      disagg flags absent from an aggregated one and different prefill/decode
      splits (1P1D vs 3P1D) have different optima. Aggregated keeps just
      ``_ws{N}``.

    Tuning knobs (kv_transfer_backend, per-role tp/ep, server flags) are
    deliberately NOT encoded here — they are the ``best_config`` the KB exists
    to accumulate within one topology.

    The result feeds BOTH :func:`recipe_canonical_id`'s ``hardware=`` segment
    AND the top-level ``hardware`` label used by ``/recipes/search``, so callers
    must apply it to the ``hw`` value ONCE and reuse that result for every KB
    read/write in the session (else read/write keys diverge).

    Args:
        gpu_type: Bare GPU identifier (e.g. ``"MI300X"``). Only the KB identity
            dimension is suffixed; non-KB consumers (Magpie runner / KG) keep
            the raw type.
        nodes: Cluster node count (``>= 2`` selects the multi-node suffix).
        gpus_per_node: GPUs per node, for the ``world_size`` product.
        pd_mode: ``"aggregated"`` or ``"disaggregated"``.
        pd_prefill_nodes: Prefill-group node count (disaggregated only).
        pd_decode_nodes: Decode-group node count (disaggregated only).

    Returns:
        ``gpu_type`` unchanged for single-node; ``{gpu_type}_ws{world_size}``
        (aggregated) or ``{gpu_type}_ws{world_size}_pd{pn}p{dn}d``
        (disaggregated) for multi-node.
    """
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
    """Return the 7-key ``labels`` dict mirroring the canonical id, so
    ``/recipes/search`` can ``label_match`` by individual dimension. Slug
    values match :func:`recipe_canonical_id`.

    Args:
        model: The model identifier.
        hardware: The hardware/GPU identifier.
        framework_name: The serving framework_name name.
        framework_version: The framework_name version.
        precision: The precision/quantization scheme.

    Returns:
        The five-key labels dict mirroring the canonical id.
    """
    return {
        F_LABEL_MODEL: _slug(model, DEFAULT_MODEL_SLUG),
        F_LABEL_HARDWARE: _slug(hardware, DEFAULT_HARDWARE_SLUG),
        F_LABEL_FRAMEWORK_NAME: _slug(framework_name, DEFAULT_FRAMEWORK_SLUG),
        F_LABEL_MODEL_TYPE: _slug(model_type, DEFAULT_MODEL_TYPE_SLUG),
        F_LABEL_ARCHITECTURES: _architectures_slug(architectures),
        F_LABEL_FRAMEWORK_VERSION: _slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG),
        F_LABEL_PRECISION: _slug(precision, DEFAULT_PRECISION_SLUG),
    }


# framework_name slug -> python package whose __version__ is authoritative. Keep
# narrow: every entry must be safe to import at boot (don't import sglang in a
# vLLM-only run).
_FRAMEWORK_VERSION_MODULES: Final[dict[str, str]] = {
    "sglang": "sglang",
    "vllm": "vllm",
    "atom": "atom",  # vendor-internal; __version__ is often a git hash
}


def detect_framework_version(framework_name: str) -> str:
    """Best-effort installed version of ``framework_name`` via importing its
    top-level package and reading ``__version__``. Failures degrade to
    :data:`DEFAULT_FRAMEWORK_VERSION_SLUG` (the optimizer must boot without
    the framework_name importable).

    Args:
        framework_name: The serving framework_name name to probe.

    Returns:
        The detected version slug, or the default on any failure.
    """
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
    if not raw:  # fall back to VERSION; __version__ (PEP 396) wins
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
    "DEFAULT_HTTP_TIMEOUT_SEC",
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
