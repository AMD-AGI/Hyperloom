"""recipe-snapshot v2 HTTP wire constants — single source of truth.

Mirrors the contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md``. The
client (see :mod:`inference_optimizer.recipe_snapshot_client`) builds
requests as plain dicts keyed by these ``Final[str]`` constants so a
backend rename surfaces at one easy-to-grep call-site instead of being
scattered across the codebase.

This module replaces :mod:`inference_optimizer.cortex_kb_constants` —
the legacy ``/v1/points`` graph surface is being retired in favour of
the dedicated recipe-snapshot resource. See CHANGELOG for the cutover.
"""

from __future__ import annotations

import logging
from typing import Final

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default endpoint
# ---------------------------------------------------------------------------
DEFAULT_KB_URL: Final[str] = "http://kb-service.primus-cortex.svc.cluster.local"

# Mount prefix per the spec (``Conventions`` section). Every endpoint
# path below is the suffix after this prefix. The client concatenates
# ``base_url + MOUNT + endpoint`` once and never substring-matches on
# the prefix again.
MOUNT_PREFIX: Final[str] = "/recipe-snapshot"


# ---------------------------------------------------------------------------
# Endpoint paths
# ---------------------------------------------------------------------------
PATH_HEALTH:           Final[str] = "/health"  # same service binary, no mount prefix
PATH_OPENAPI:          Final[str] = "/openapi.json"

# Recipe rows
PATH_RECIPES_LIST:     Final[str] = MOUNT_PREFIX + "/recipes"
PATH_RECIPES_SEARCH:   Final[str] = MOUNT_PREFIX + "/recipes/search"
# Templates carry ``{canonical_id}`` placeholders — callers MUST format
# them via :func:`format_recipe_path` so the slash-in-id case (HF stems
# like ``Qwen/Qwen3-30B-A3B``) is preserved verbatim instead of being
# percent-encoded.
PATH_RECIPE_TPL:           Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}"
PATH_RECIPE_HISTORY_TPL:   Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}/history"
PATH_RECIPE_ATTEMPTS_TPL:  Final[str] = MOUNT_PREFIX + "/recipes/{canonical_id}/attempts"

# Sessions
PATH_SESSION_ATTEMPTS_TPL: Final[str] = MOUNT_PREFIX + "/sessions/{session_id}/attempts"
PATH_SESSION_SUMMARY_TPL:  Final[str] = MOUNT_PREFIX + "/sessions/{session_id}/summary"


# ---------------------------------------------------------------------------
# Request body field names — PUT /recipes/{canonical_id}
# ---------------------------------------------------------------------------
# Top-level fields. ``labels`` / ``body`` / ``metrics`` / experience
# arrays are caller-defined dicts; the server does NOT validate their
# inner shape. Only ``authority`` + ``provenance`` are required.
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

# Canonical identity dimensions — caller-defined keys inside ``labels``
# that mirror the five components of :func:`recipe_canonical_id`. Server
# does not enforce these (labels are caller-defined under v2), but every
# inference-optimizer caller MUST stamp them so ``/recipes/search`` can
# locate rows by individual dimensions even when the canonical_id format
# evolves (e.g. when a new dimension is appended).
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

# Attempts (POST + GET) — kept here even though Phase 1 doesn't use
# them yet (caller chose ``no_summary_only`` mapping). They land in
# this constants module so the Phase 3 NDJSON flusher / breakdown
# collector can reference them without re-introducing string literals.
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


# ---------------------------------------------------------------------------
# Enum literals
# ---------------------------------------------------------------------------
# Authority (strict enum on the server side; pydantic rejects unknown
# values with ``422`` ``type=enum`` and the value list).
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


# ---------------------------------------------------------------------------
# NDJSON envelope ops
# ---------------------------------------------------------------------------
# Phase 1 only writes ``put_recipe``. The other ops are reserved for
# Phase 3 (attempts append on a flag, history backfill, search-driven
# warm-start). Defining the full set up-front keeps the flusher and
# breakdown collector free of string literals.
OP_PUT_RECIPE:    Final[str] = "put_recipe"
OP_APPEND_ATTEMPT: Final[str] = "append_attempt"


# ---------------------------------------------------------------------------
# Defaults — bypass-mode semantics
# ---------------------------------------------------------------------------
# Two timeout / retry profiles depending on the caller's tolerance for
# being blocked on a slow / unreachable KB:
#
# * **foreground** — Coordinator on the main event loop. Every PUT
#   (one per KEEP / REVERT, ~10 / EXPLORE round) is a sync HTTP call.
#   Operator requirement: "KB is a side-channel — if it's unavailable
#   / slow, do NOT slow the main logic." So the foreground profile
#   fails fast (2s + 1 retry ≈ ~2.5s worst case) and the write falls
#   through to NDJSON on the very first transport hiccup.
#
# * **background** — kb_flusher daemon + CLOSE-time drain. These run
#   outside the main loop so they can afford the legacy retry budget
#   (10s × 3) to maximise the chance a transient KB blip still
#   commits without dead-lettering the row.
DEFAULT_HTTP_TIMEOUT_SEC:    Final[float] = 10.0  # background / flusher
DEFAULT_MAX_CONCURRENCY:     Final[int]   = 8     # aligned with asyncpg pool
DEFAULT_RETRY_ATTEMPTS:      Final[int]   = 3
DEFAULT_RETRY_BASE_MS:       Final[int]   = 200   # 200ms × {1, 1.4, 4}
FOREGROUND_HTTP_TIMEOUT_SEC: Final[float] = 2.0   # Coordinator main loop
FOREGROUND_RETRY_ATTEMPTS:   Final[int]   = 1     # fail fast → NDJSON

# Maximum number of times a single NDJSON row may be re-attempted by
# the drain loop before it is treated as permanent and moved to the
# dead-letter file. Protects against infinite retry loops when an
# input the server permanently rejects (422 unknown top-level field,
# etc.) keeps coming back through the flusher.
MAX_FLUSH_ATTEMPTS: Final[int] = 5

# Default ``provenance.source`` / ``provenance.generator`` stamped on
# every PUT when the caller doesn't supply one. Smoke tests override
# the generator to ``SMOKE_GENERATOR`` so probe writes never collide
# with production data in observability dashboards.
DEFAULT_SOURCE:     Final[str] = "hyperloom-inference-optimizer"
DEFAULT_GENERATOR:  Final[str] = "hyperloom"
SMOKE_GENERATOR:    Final[str] = "hyperloom-smoke"

# Default ``confidence`` when caller doesn't override. Spec accepts
# ``[0.0, 1.0]``; the optimizer's marathon-winner default in the prior
# system was 0.85, kept here for continuity.
DEFAULT_CONFIDENCE: Final[float] = 0.85


# ---------------------------------------------------------------------------
# Canonical id derivation
# ---------------------------------------------------------------------------
# Default-slug constants for missing identity components. Kept as named
# constants (rather than literals scattered across helpers) so the audit
# log and the `/search` corpus stay consistent — operators can grep for
# ``unknown_framework_version`` to find rows where auto-detect failed,
# for example, instead of guessing every spelling variant.
DEFAULT_MODEL_SLUG:             Final[str] = "unknown_model"
DEFAULT_HARDWARE_SLUG:          Final[str] = "unknown_hw"
DEFAULT_FRAMEWORK_SLUG:         Final[str] = "unknown_framework"
DEFAULT_FRAMEWORK_VERSION_SLUG: Final[str] = "unknown_version"
DEFAULT_PRECISION_SLUG:         Final[str] = "unknown_precision"


def _slug(value: str, default: str) -> str:
    """Lowercase + basename + space/dot/colon→underscore.

    Recipe ``canonical_id`` is caller-defined under v2 (the path
    accepts forward slashes raw) so we are no longer subject to the
    strict ``[a-z0-9_-]`` regex the pre-v2 ``/v1/points`` server
    enforced. We still slug here for three reasons:

    1. Lookup stability — two CLI invocations supplying
       ``--model /wekafs/models/Qwen3-30B-A3B`` and
       ``--model qwen3-30b-a3b`` must converge on the same recipe row.
    2. Filesystem safety — the local KB store (Commit 2) maps each
       canonical_id component to a directory level; characters that
       would split a slug across directories (``/``) or that are
       awkward in filenames (``:``, ``.``, whitespace) are normalised
       to ``_`` so ``Qwen/Qwen3`` cannot collide with ``Qwen_Qwen3``.
       NOTE: ``/`` is still resolved to the basename FIRST (so HF
       paths like ``meta-llama/Llama-3.1-8B`` collapse to
       ``llama-3.1-8b``), and only embedded ``/`` survives that step
       in pathological inputs.
    3. Versions like ``0.4.5+abcdef0`` — common for editable installs
       — keep their ``+``/digits but lose dots-as-path-separators on
       Windows-ish filesystems.
    """
    raw = (value or "").strip()
    if not raw:
        return default
    # Path-style → basename (last path component, then forward-slash
    # fallback). Matches the prior helper in ``cortex_kb_client``.
    if "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1] or raw
    # Replace each problematic char individually rather than doing a
    # blanket regex strip — keeps useful chars like '-', '+', '_'
    # untouched while normalising the rest.
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
    """Build the recipe ``canonical_id``.

    Shape:
        ``inference:{model}:{hardware}:{framework}:{framework_version}:{precision}``

    Identity-strength order (strongest → weakest) so the prefix sorts
    nicely and partial-string queries are useful even before five
    components are known:

    1. ``model``             — the workload — ties the row to a
       distinct model architecture / weights.
    2. ``hardware``          — the platform — ties it to one GPU
       generation (mi300x vs mi355x have different tile sizes,
       different best-tp).
    3. ``framework``         — sglang vs vLLM vs atom — different
       schedulers / ``best_config.extra_*_args`` shapes.
    4. ``framework_version`` — same framework can change scheduler
       internals across releases (e.g. sglang 0.4 → 0.5 RadixAttention
       defaults), so two versions deserve separate recipe rows.
    5. ``precision``         — fp8 / fp16 / bf16 / fp4 / int8 —
       changes the optimal tp / ep / kv_cache_dtype because memory
       footprint shifts.

    All five components are ``keyword-only`` to prevent accidental
    positional re-ordering; an empty / missing component falls back
    to the matching ``DEFAULT_*_SLUG`` so the canonical_id is always
    well-formed (5 colons, 6 segments). Callers are encouraged to
    use :func:`detect_framework_version` to fill ``framework_version``
    when the operator did not pass ``--framework-version`` explicitly.
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
    """Return the five-key ``labels`` dict that mirrors the canonical id.

    Stamping these into ``labels`` on every PUT lets ``/recipes/search``
    use ``label_match`` to filter by individual dimensions (e.g. "all
    sglang recipes regardless of model") without parsing the
    canonical_id string. Slug values match the ones used in
    :func:`recipe_canonical_id` so a search round-trip never disagrees
    with the id derivation.
    """
    return {
        F_LABEL_MODEL:             _slug(model,             DEFAULT_MODEL_SLUG),
        F_LABEL_HARDWARE:          _slug(hardware,          DEFAULT_HARDWARE_SLUG),
        F_LABEL_FRAMEWORK:         _slug(framework,         DEFAULT_FRAMEWORK_SLUG),
        F_LABEL_FRAMEWORK_VERSION: _slug(framework_version, DEFAULT_FRAMEWORK_VERSION_SLUG),
        F_LABEL_PRECISION:         _slug(precision,         DEFAULT_PRECISION_SLUG),
    }


# Mapping from ``framework`` slug → top-level python package whose
# ``__version__`` attribute is treated as authoritative. Keep narrow:
# adding a new framework is a one-line append, but every entry has to
# be a package the optimizer process is allowed to import at boot
# (we MUST NOT trigger a sglang import inside a vLLM-only run, etc.).
_FRAMEWORK_VERSION_MODULES: Final[dict[str, str]] = {
    "sglang": "sglang",
    "vllm":   "vllm",
    # ``atom`` is a vendor-internal framework whose ``__version__`` is
    # typically a git short-hash rather than a SemVer tag; we still
    # try the import and fall back to ``unknown_version`` on miss.
    "atom":   "atom",
}


def detect_framework_version(framework: str) -> str:
    """Best-effort: return the installed version of ``framework``.

    Auto-detect path used when the operator didn't pass
    ``--framework-version`` (most common case). Tries to ``import``
    the framework's top-level package and read ``__version__``.
    Failures degrade to :data:`DEFAULT_FRAMEWORK_VERSION_SLUG`
    rather than raising — the optimizer must boot even when the
    framework module isn't importable in the current venv (e.g.
    a dry-run on a CI box without GPU stacks). Callers should treat
    a result equal to the default slug as "operator should pass
    ``--framework-version`` explicitly to scope the recipe row".
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
    # Some frameworks expose ``VERSION`` instead. Tried second so
    # ``__version__`` (the PEP 396 standard) wins when both exist.
    if not raw:
        raw = getattr(mod, "VERSION", "") or ""
    return _slug(str(raw), DEFAULT_FRAMEWORK_VERSION_SLUG)


def format_recipe_path(template: str, canonical_id: str) -> str:
    """Substitute ``{canonical_id}`` into a path template **without**
    percent-encoding the slashes inside the id.

    ``canonical_id`` is ``:path``-typed on the server side (per spec
    Conventions) so a HF stem like ``Qwen/Qwen3-30B-A3B`` must reach
    the server verbatim. ``urllib.parse.quote`` with
    ``safe="/"`` would still encode ``:``, ``@``, etc. — but the
    server accepts those raw too, so this helper does a flat
    substitution and trusts the caller to have already chosen a
    canonical_id from ``recipe_canonical_id``.
    """
    return template.replace("{canonical_id}", canonical_id)


__all__ = [
    "DEFAULT_KB_URL",
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
