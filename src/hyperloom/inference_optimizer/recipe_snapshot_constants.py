# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe identity: canonical-id slugs, the hardware dimension and framework-version detection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

from hyperloom.common.provenance import framework_distribution_version

log = logging.getLogger(__name__)


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


@dataclass(frozen=True, kw_only=True)
class RecipeIdentity:
    """The seven serving dimensions a recipe is keyed by."""

    model: str
    hardware: str
    framework_name: str
    model_type: str = ""
    architectures: "str | list[str]" = ""
    framework_version: str
    precision: str


def recipe_canonical_id(identity: RecipeIdentity, scheme: str = "inference") -> str:
    """Build a recipe id as ``{scheme}:{seven serving dimensions}``."""
    if scheme not in {"inference", "agentx"}:
        raise ValueError(f"unsupported recipe identity scheme: {scheme!r}")
    return (
        f"{scheme}:"
        f"{_slug(identity.model, DEFAULT_MODEL_SLUG)}:"
        f"{_slug(identity.hardware, DEFAULT_HARDWARE_SLUG)}:"
        f"{_slug(identity.framework_name, DEFAULT_FRAMEWORK_SLUG)}:"
        f"{_slug(identity.model_type, DEFAULT_MODEL_TYPE_SLUG)}:"
        f"{_architectures_slug(identity.architectures)}:"
        f"{_slug(identity.framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG)}:"
        f"{_slug(identity.precision, DEFAULT_PRECISION_SLUG)}"
    )


def _partition_mode_slug(mode: str) -> str:
    """Slug for a compute-partition mode that actually divides the card, else ``""``.

    SPX collapses to ``""`` along with an unpublished mode, because one partition is the whole card: that is what a
    row recorded before the mode was published was necessarily running on, so suffixing it would re-key the existing
    corpus to say something its key already said.
    """
    from hyperloom.common.gpu_partition import MODE_PARTITION_COUNTS

    canonical = str(mode or "").strip().upper()
    return canonical.lower() if MODE_PARTITION_COUNTS.get(canonical, 1) > 1 else ""


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
    partition_mode: str = "",
    backend: str = "",
) -> str:
    """Topology-aware hardware dimension for the recipe ``canonical_id``.

    ``ep`` and ``partition_mode`` suffix at any node count, because neither is decided by the node count: a card
    divided into CPX partitions is a different machine than a whole one, and experts split across ranks is a
    different formation than dense. Both are fixed at launch rather than explored, so a ``best_config`` tuned under
    one is invalid under the other -- which is the same argument ``_tp{tp}`` already makes. Encoding them here makes
    the mismatch unrepresentable instead of something a reader has to notice and refuse.

    Every suffix is omitted for the default shape (``ep <= 1``, SPX or unpublished mode), so a key recorded before
    these dimensions existed stays byte-identical and nothing in the corpus moves.
    """
    base = (gpu_type or "").strip()
    try:
        ep_i = int(ep)
    except (TypeError, ValueError):
        ep_i = 0
    part = _partition_mode_slug(partition_mode)
    try:
        n = int(nodes)
    except (TypeError, ValueError):
        return base
    if n < 2:
        # Single node still carries the shape suffixes, but nothing else: world size, PD split, tp and backend all
        # describe a cluster. ``tp`` is deliberately absent -- almost every single-node run sets it, so encoding it
        # would re-key the whole corpus rather than only the rows whose shape currently collides.
        slug = f"{base}_ep{ep_i}" if ep_i > 1 else base
        return f"{slug}_{part}" if part else slug
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
    if ep_i > 1:
        slug = f"{slug}_ep{ep_i}"
    if part:
        slug = f"{slug}_{part}"
    be = _slug(backend, "")
    if be:
        slug = f"{slug}_{be}"
    return slug


# framework_name slug -> python package whose __version__ answers when the serving interpreter does not.
_FRAMEWORK_VERSION_MODULES: Final[dict[str, str]] = {
    "sglang": "sglang",
    "vllm": "vllm",
    "atom": "atom",  # vendor-internal; __version__ is often a git hash
}


def detect_framework_version(framework_name: str) -> str:
    """Best-effort installed version of ``framework_name``.

    Read from the interpreter preflight resolved to serve it, else by importing its top-level package and reading
    ``__version__``.
    """
    fw_slug = _slug(framework_name, "")
    if not fw_slug:
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    module_name = _FRAMEWORK_VERSION_MODULES.get(fw_slug)
    if not module_name:
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    # An isolated framework venv is not importable from this interpreter.
    served = framework_distribution_version(os.environ, fw_slug)
    if served:
        return _slug(served, DEFAULT_FRAMEWORK_VERSION_SLUG)
    try:
        import importlib

        mod = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
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
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_FRAMEWORK_SLUG",
    "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "RecipeIdentity",
    "recipe_canonical_id",
    "kb_hardware_slug",
    "detect_framework_version",
]
