# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""recipe-snapshot v2 HTTP wire constants — single source of truth.

Mirrors the contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md``. The
recipe-KB remote clients (see :mod:`inference_optimizer.recipe_kb.remote_client`
and :mod:`inference_optimizer.recipe_kb.gbrain_remote_client`) build requests as
plain dicts keyed by these ``Final[str]`` constants so a backend rename surfaces
at one easy-to-grep call-site instead of being scattered across the codebase.
"""

from __future__ import annotations

import logging
from typing import Final

log = logging.getLogger(__name__)


# No DEFAULT_KB_URL by design: the optimizer never silently connects to a
# remote KB; a remote source is used only with --cortex-kb-url / $CORTEX_KB_URL.

# Mount prefix per spec; client concatenates base_url + MOUNT + endpoint.
MOUNT_PREFIX: Final[str] = "/recipe-snapshot"


# Endpoint paths
PATH_HEALTH:           Final[str] = "/health"  # same service binary, no mount prefix
PATH_OPENAPI:          Final[str] = "/openapi.json"

# Recipe rows. {canonical_id} templates MUST be formatted via
# format_recipe_path so slash-in-id stems (Qwen/Qwen3-30B-A3B) stay verbatim.
PATH_RECIPES_LIST:     Final[str] = MOUNT_PREFIX + "/recipes"
PATH_RECIPES_SEARCH:   Final[str] = MOUNT_PREFIX + "/recipes/search"
PATH_RECIPE_TPL:           Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}"
PATH_RECIPE_HISTORY_TPL:   Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}/history"
PATH_RECIPE_ATTEMPTS_TPL:  Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}/attempts"

# Sessions
PATH_SESSION_ATTEMPTS_TPL: Final[str] = MOUNT_PREFIX + "/sessions/{session_id}/attempts"
PATH_SESSION_SUMMARY_TPL:  Final[str] = MOUNT_PREFIX + "/sessions/{session_id}/summary"


# Request body field names — PUT /recipes/{canonical_id}. Top-level fields;
# server validates only authority + provenance (rest are caller-defined).
F_LABELS:        Final[str] = "labels"
F_BODY:          Final[str] = "body"
F_METRICS:       Final[str] = "metrics"
F_FINDINGS:      Final[str] = "findings"
F_FAILURES:      Final[str] = "failures"
F_PITFALLS:      Final[str] = "pitfalls"
F_LESSONS:       Final[str] = "lessons"
F_GAPS:          Final[str] = "gaps"
F_AUTHORITY:     Final[str] = "authority"
F_CONFIDENCE:    Final[str] = "confidence"
F_EVIDENCE_REFS: Final[str] = "evidence_refs"
F_PROVENANCE:    Final[str] = "provenance"

# Canonical identity dimensions inside ``labels`` mirroring
# recipe_canonical_id; stamp them so /recipes/search can filter by dimension.
F_LABEL_MODEL:             Final[str] = "model"
F_LABEL_HARDWARE:          Final[str] = "hardware"
F_LABEL_FRAMEWORK:         Final[str] = "framework"
F_LABEL_FRAMEWORK_VERSION: Final[str] = "framework_version"
F_LABEL_PRECISION:         Final[str] = "precision"

# PUT response fields
F_CANONICAL_ID:  Final[str] = "canonical_id"
F_VERSION:       Final[str] = "version"
F_CREATED:       Final[str] = "created"

# History response fields
F_HISTORY:       Final[str] = "history"
F_ARCHIVED_AT:   Final[str] = "archived_at"
F_REPLACED_BY:   Final[str] = "replaced_by"
F_SNAPSHOT:      Final[str] = "snapshot"

# Attempts (POST + GET).
F_SESSION_ID:        Final[str] = "session_id"
F_DIFF:              Final[str] = "diff"
F_PREDICTED_DELTA:   Final[str] = "predicted_delta"
F_MEASURED_METRICS:  Final[str] = "measured_metrics"
F_FITNESS:           Final[str] = "fitness"
F_OUTCOME:           Final[str] = "outcome"
F_RATIONALE:         Final[str] = "rationale"
F_ATTEMPT_AT:        Final[str] = "attempt_at"

# Search
F_LABEL_MATCH:    Final[str] = "label_match"
F_METRIC_FILTERS: Final[str] = "metric_filters"
F_UPDATED_SINCE:  Final[str] = "updated_since"
F_ORDER_BY:       Final[str] = "order_by"
F_LIMIT:          Final[str] = "limit"
F_RECIPES:        Final[str] = "recipes"

# MetricFilter sub-fields
F_METRIC_MIN:     Final[str] = "min"
F_METRIC_MAX:     Final[str] = "max"

# EvidenceRef sub-fields
F_EV_KIND: Final[str] = "kind"
F_EV_REF:  Final[str] = "ref"
F_EV_NOTE: Final[str] = "note"

# Provenance sub-fields
F_PV_SOURCE:       Final[str] = "source"
F_PV_GENERATOR:    Final[str] = "generator"
F_PV_GENERATED_AT: Final[str] = "generated_at"
F_PV_DETAILS:      Final[str] = "details"


# Enum literals (strict server-side enums; unknown values -> 422).
AUTHORITY_AUTHORITATIVE: Final[str] = "AUTHORITATIVE"
AUTHORITY_EXPERIENTIAL:  Final[str] = "EXPERIENTIAL"
AUTHORITY_HYPOTHESIZED:  Final[str] = "HYPOTHESIZED"
AUTHORITY_TENTATIVE:     Final[str] = "TENTATIVE"

# AttemptOutcome (POST /recipes/{cid}/attempts — strict enum)
OUTCOME_KEPT:     Final[str] = "kept"
OUTCOME_REVERTED: Final[str] = "reverted"
OUTCOME_FAILED:   Final[str] = "failed"
OUTCOME_SKIPPED:  Final[str] = "skipped"

# EvidenceRef.kind (strict enum)
EV_KIND_URL:          Final[str] = "url"
EV_KIND_COMMIT:       Final[str] = "commit"
EV_KIND_PROFILE_FILE: Final[str] = "profile_file"
EV_KIND_LOG:          Final[str] = "log"

# Search order_by whitelist — exactly 6 values per spec.
ORDER_BY_UPDATED_AT_DESC: Final[str] = "updated_at DESC"
ORDER_BY_UPDATED_AT_ASC:  Final[str] = "updated_at ASC"
ORDER_BY_CREATED_AT_DESC: Final[str] = "created_at DESC"
ORDER_BY_CREATED_AT_ASC:  Final[str] = "created_at ASC"
ORDER_BY_VERSION_DESC:    Final[str] = "version DESC"
ORDER_BY_VERSION_ASC:     Final[str] = "version ASC"

ORDER_BY_WHITELIST: Final[frozenset[str]] = frozenset({
    ORDER_BY_UPDATED_AT_DESC,
    ORDER_BY_UPDATED_AT_ASC,
    ORDER_BY_CREATED_AT_DESC,
    ORDER_BY_CREATED_AT_ASC,
    ORDER_BY_VERSION_DESC,
    ORDER_BY_VERSION_ASC,
})


# NDJSON envelope ops.
OP_PUT_RECIPE:    Final[str] = "put_recipe"
OP_APPEND_ATTEMPT: Final[str] = "append_attempt"


# Defaults — two timeout/retry profiles. Foreground (Coordinator main loop)
# fails fast (2s + 1 retry) and falls through to NDJSON; background (flusher /
# CLOSE drain) uses the larger 10s x 3 budget.
DEFAULT_HTTP_TIMEOUT_SEC:    Final[float] = 10.0  # background / flusher
DEFAULT_MAX_CONCURRENCY:     Final[int]   = 8     # aligned with asyncpg pool
DEFAULT_RETRY_ATTEMPTS:      Final[int]   = 3
DEFAULT_RETRY_BASE_MS:       Final[int]   = 200   # 200ms × {1, 1.4, 4}
FOREGROUND_HTTP_TIMEOUT_SEC: Final[float] = 2.0   # Coordinator main loop
FOREGROUND_RETRY_ATTEMPTS:   Final[int]   = 1     # fail fast → NDJSON

# Max re-attempts for one NDJSON row before it dead-letters (guards against
# infinite retry on permanently-rejected rows, e.g. 422 unknown field).
MAX_FLUSH_ATTEMPTS: Final[int] = 5

# Default provenance.source / generator stamped on PUTs; smoke tests use
# SMOKE_GENERATOR so probe writes don't collide with production data.
DEFAULT_SOURCE:     Final[str] = "hyperloom-inference-optimizer"
DEFAULT_GENERATOR:  Final[str] = "hyperloom"
SMOKE_GENERATOR:    Final[str] = "hyperloom-smoke"

# Default confidence when caller doesn't override (spec [0.0, 1.0]).
DEFAULT_CONFIDENCE: Final[float] = 0.85


# Canonical id derivation. Default-slug constants for missing identity
# components, named so the audit log + /search corpus stay grep-stable.
DEFAULT_MODEL_SLUG:             Final[str] = "unknown_model"
DEFAULT_HARDWARE_SLUG:          Final[str] = "unknown_hw"
DEFAULT_FRAMEWORK_SLUG:         Final[str] = "unknown_framework"
DEFAULT_FRAMEWORK_VERSION_SLUG: Final[str] = "unknown_version"
DEFAULT_PRECISION_SLUG:         Final[str] = "unknown_precision"


def _slug(value: str, default: str) -> str:
    """Lowercase + basename + space/tab/slash -> underscore.

    Slugged for lookup stability (``--model /path/Qwen3`` and ``qwen3`` must
    converge) and filesystem safety in the local KB store. ``/`` resolves to
    the basename first (HF paths collapse to the stem).
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


def recipe_canonical_id(
    *,
    model: str,
    hardware: str,
    framework: str,
    framework_version: str,
    precision: str,
) -> str:
    """Build the recipe ``canonical_id``:
    ``inference:{model}:{hardware}:{framework}:{framework_version}:{precision}``.

    Identity-strength order (strongest -> weakest) so prefix queries are
    useful before all components are known. Keyword-only to prevent
    positional re-ordering; missing components fall back to ``DEFAULT_*_SLUG``
    so the id is always well-formed (6 colon-separated segments).
    """
    return (
        f"inference:"
        f"{_slug(model,             DEFAULT_MODEL_SLUG)}:"
        f"{_slug(hardware,          DEFAULT_HARDWARE_SLUG)}:"
        f"{_slug(framework,         DEFAULT_FRAMEWORK_SLUG)}:"
        f"{_slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG)}:"
        f"{_slug(precision,         DEFAULT_PRECISION_SLUG)}"
    )


def canonical_labels(
    *,
    model: str,
    hardware: str,
    framework: str,
    framework_version: str,
    precision: str,
) -> dict[str, str]:
    """Return the five-key ``labels`` dict mirroring the canonical id, so
    ``/recipes/search`` can ``label_match`` by individual dimension. Slug
    values match :func:`recipe_canonical_id`.
    """
    return {
        F_LABEL_MODEL:             _slug(model,             DEFAULT_MODEL_SLUG),
        F_LABEL_HARDWARE:          _slug(hardware,          DEFAULT_HARDWARE_SLUG),
        F_LABEL_FRAMEWORK:         _slug(framework,         DEFAULT_FRAMEWORK_SLUG),
        F_LABEL_FRAMEWORK_VERSION: _slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG),
        F_LABEL_PRECISION:         _slug(precision,         DEFAULT_PRECISION_SLUG),
    }


# framework slug -> python package whose __version__ is authoritative. Keep
# narrow: every entry must be safe to import at boot (don't import sglang in a
# vLLM-only run).
_FRAMEWORK_VERSION_MODULES: Final[dict[str, str]] = {
    "sglang": "sglang",
    "vllm":   "vllm",
    "atom":   "atom",  # vendor-internal; __version__ is often a git hash
}


def detect_framework_version(framework: str) -> str:
    """Best-effort installed version of ``framework`` via importing its
    top-level package and reading ``__version__``. Failures degrade to
    :data:`DEFAULT_FRAMEWORK_VERSION_SLUG` (the optimizer must boot without
    the framework importable).
    """
    fw_slug = _slug(framework, "")
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
            "detect_framework_version: import %r failed (%s); "
            "falling back to %r",
            module_name, exc, DEFAULT_FRAMEWORK_VERSION_SLUG,
        )
        return DEFAULT_FRAMEWORK_VERSION_SLUG
    raw = getattr(mod, "__version__", "") or ""
    if not raw:  # fall back to VERSION; __version__ (PEP 396) wins
        raw = getattr(mod, "VERSION", "") or ""
    return _slug(str(raw), DEFAULT_FRAMEWORK_VERSION_SLUG)


def format_recipe_path(template: str, canonical_id: str) -> str:
    """Substitute ``{canonical_id}`` into a path template without
    percent-encoding (server treats canonical_id as ``:path``-typed, so HF
    stems like ``Qwen/Qwen3-30B-A3B`` must reach it verbatim).
    """
    return template.replace("{canonical_id}", canonical_id)


__all__ = [
    "MOUNT_PREFIX",
    "PATH_HEALTH",
    "PATH_OPENAPI",
    "PATH_RECIPES_LIST",
    "PATH_RECIPES_SEARCH",
    "PATH_RECIPE_TPL",
    "PATH_RECIPE_HISTORY_TPL",
    "PATH_RECIPE_ATTEMPTS_TPL",
    "PATH_SESSION_ATTEMPTS_TPL",
    "PATH_SESSION_SUMMARY_TPL",
    "F_LABELS", "F_BODY", "F_METRICS",
    "F_FINDINGS", "F_FAILURES", "F_PITFALLS", "F_LESSONS", "F_GAPS",
    "F_AUTHORITY", "F_CONFIDENCE", "F_EVIDENCE_REFS", "F_PROVENANCE",
    "F_LABEL_MODEL", "F_LABEL_HARDWARE", "F_LABEL_FRAMEWORK",
    "F_LABEL_FRAMEWORK_VERSION", "F_LABEL_PRECISION",
    "F_CANONICAL_ID", "F_VERSION", "F_CREATED",
    "F_HISTORY", "F_ARCHIVED_AT", "F_REPLACED_BY", "F_SNAPSHOT",
    "F_SESSION_ID", "F_DIFF", "F_PREDICTED_DELTA", "F_MEASURED_METRICS",
    "F_FITNESS", "F_OUTCOME", "F_RATIONALE", "F_ATTEMPT_AT",
    "F_LABEL_MATCH", "F_METRIC_FILTERS", "F_UPDATED_SINCE",
    "F_ORDER_BY", "F_LIMIT", "F_RECIPES",
    "F_METRIC_MIN", "F_METRIC_MAX",
    "F_EV_KIND", "F_EV_REF", "F_EV_NOTE",
    "F_PV_SOURCE", "F_PV_GENERATOR", "F_PV_GENERATED_AT", "F_PV_DETAILS",
    "AUTHORITY_AUTHORITATIVE", "AUTHORITY_EXPERIENTIAL",
    "AUTHORITY_HYPOTHESIZED", "AUTHORITY_TENTATIVE",
    "OUTCOME_KEPT", "OUTCOME_REVERTED", "OUTCOME_FAILED", "OUTCOME_SKIPPED",
    "EV_KIND_URL", "EV_KIND_COMMIT", "EV_KIND_PROFILE_FILE", "EV_KIND_LOG",
    "ORDER_BY_UPDATED_AT_DESC", "ORDER_BY_UPDATED_AT_ASC",
    "ORDER_BY_CREATED_AT_DESC", "ORDER_BY_CREATED_AT_ASC",
    "ORDER_BY_VERSION_DESC", "ORDER_BY_VERSION_ASC", "ORDER_BY_WHITELIST",
    "OP_PUT_RECIPE", "OP_APPEND_ATTEMPT",
    "DEFAULT_HTTP_TIMEOUT_SEC", "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_RETRY_ATTEMPTS", "DEFAULT_RETRY_BASE_MS",
    "FOREGROUND_HTTP_TIMEOUT_SEC", "FOREGROUND_RETRY_ATTEMPTS",
    "MAX_FLUSH_ATTEMPTS",
    "DEFAULT_SOURCE", "DEFAULT_GENERATOR", "SMOKE_GENERATOR",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MODEL_SLUG", "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_FRAMEWORK_SLUG", "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "recipe_canonical_id", "canonical_labels",
    "detect_framework_version", "format_recipe_path",
]
