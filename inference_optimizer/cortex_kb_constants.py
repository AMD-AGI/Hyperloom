"""Cortex KB HTTP wire constants — single source of truth for endpoint
paths, request field names, and registered enum literals.

Avoiding ``shared/types/*`` Pydantic vendor: the client builds requests
as plain dicts keyed by these ``Final[str]`` constants. CI snapshot test
(``test_v08_kb_http_contract``) diffs them against generated OpenAPI to
catch backend drift.

Reference: ``cortex-kb-http-branch-b-2026-05-20.md``.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Default endpoint
# ---------------------------------------------------------------------------
DEFAULT_KB_URL: Final[str] = "http://kb-service.primus-cortex.svc.cluster.local"

# ---------------------------------------------------------------------------
# Endpoint paths (§1)
# ---------------------------------------------------------------------------
PATH_HEALTH:        Final[str] = "/health"
PATH_PROPOSE_POINT: Final[str] = "/v1/points/propose"
PATH_QUERY_POINT:   Final[str] = "/v1/points/query"
PATH_TRAVERSE:      Final[str] = "/v1/traverse"
# Fact-write endpoints (kg-usage-guide §3.2 / §3.4 / §3.5). Direct-propose
# path used by ``propose_lesson`` / ``propose_pitfall`` / ``update_recipe``
# (session-less; the KB session begin/commit protocol was retired).
PATH_PROPOSE_EDGE:  Final[str] = "/v1/edges/propose"
PATH_NEGATE_EDGE:   Final[str] = "/v1/edges/negate"
PATH_ADD_EVIDENCE:  Final[str] = "/v1/edges/{edge_id}/add_evidence"

# ---------------------------------------------------------------------------
# Request body field names (§1.1–§1.8)
# ---------------------------------------------------------------------------
F_ATTRS:              Final[str] = "attrs"

# propose edge (fact-write surface)
F_FROM_POINT:         Final[str] = "from_point"
F_TO_POINT:           Final[str] = "to_point"
F_EDGE_TYPE:          Final[str] = "edge_type"
F_EVIDENCE_REFS:      Final[str] = "evidence_refs"
F_PROMOTED_EDGE_ID:   Final[str] = "promoted_edge_id"
F_NEGATION_EDGE_ID:   Final[str] = "negation_edge_id"

# propose point
F_CANONICAL_ID:       Final[str] = "canonical_id"
F_KIND:               Final[str] = "kind"
F_ENTITY_TYPE:        Final[str] = "entity_type"
F_AUTHORITY:          Final[str] = "authority"
F_PROVENANCE:         Final[str] = "provenance"
F_PROPOSAL_ID:        Final[str] = "proposal_id"
F_POINT_ID:           Final[str] = "point_id"
F_COMMITTEE:          Final[str] = "committee"
F_STATUS:             Final[str] = "status"

# query / traverse
F_LIMIT:              Final[str] = "limit"
F_NEIGHBOR_PREVIEW:   Final[str] = "neighbor_preview"
F_NEIGHBOR_LIMIT:     Final[str] = "neighbor_limit"
F_ATTRS_FILTER:       Final[str] = "attrs_filter"
F_LENS:               Final[str] = "lens"
F_POINTS:             Final[str] = "points"
F_NEIGHBORS:          Final[str] = "neighbors"
F_START_POINT:        Final[str] = "start_point"
F_BUDGET_STEPS:       Final[str] = "budget_steps"
F_BUDGET_BRANCHES:    Final[str] = "budget_branches"
F_PATHS:              Final[str] = "paths"
F_CANDIDATES:         Final[str] = "candidates"

# response shared
F_LENS_SCHEDULE:      Final[str] = "lens_schedule"

# evidence_ref subfields
F_EV_KIND:            Final[str] = "kind"
F_EV_REF:             Final[str] = "ref"
F_EV_NOTE:            Final[str] = "note"

# provenance subfields
F_PV_SOURCE:          Final[str] = "source"
F_PV_GENERATOR:       Final[str] = "generator"
F_PV_GENERATED_AT:    Final[str] = "generated_at"
F_PV_DETAILS:         Final[str] = "details"

# ---------------------------------------------------------------------------
# Enum literals (§1, §5 — must match registered KB values)
# ---------------------------------------------------------------------------
# EdgeType (Literal[7]) — only the labels still used by the fact-write
# surface (``empirical`` / ``causal`` / ``negation`` / ``structural`` /
# ``evolutionary`` for ``propose_edge.attrs.relation`` pairings). The
# ``investigation`` and ``hypothetical`` labels survive in case the
# fact-write surface grows new edge kinds; pruning them is a follow-up.
EDGE_STRUCTURAL:           Final[str] = "structural"
EDGE_CAUSAL:               Final[str] = "causal"
EDGE_INVESTIGATION:        Final[str] = "investigation"
EDGE_EVOLUTIONARY:         Final[str] = "evolutionary"
EDGE_EMPIRICAL:            Final[str] = "empirical"
EDGE_NEGATION:             Final[str] = "negation"

# Authority (Literal[4])
AUTHORITY_AUTHORITATIVE:   Final[str] = "AUTHORITATIVE"
AUTHORITY_EXPERIENTIAL:    Final[str] = "EXPERIENTIAL"
AUTHORITY_INFERRED:        Final[str] = "INFERRED"
AUTHORITY_HYPOTHESIZED:    Final[str] = "HYPOTHESIZED"

# Provenance.source (Literal[5])
SOURCE_OFFLINE_INGEST:     Final[str] = "offline_ingest"
SOURCE_AGENT_OBSERVATION:  Final[str] = "agent_observation"
SOURCE_CASCADE_DERIVED:    Final[str] = "cascade_derived"
SOURCE_CO_OCCURRENCE:      Final[str] = "co_occurrence"
SOURCE_ANALOGY_SEED:       Final[str] = "analogy_seed"

# EvidenceRef.kind (Literal[6])
EV_KIND_URL:               Final[str] = "url"
EV_KIND_COMMIT:            Final[str] = "commit"
EV_KIND_PROFILE_FILE:      Final[str] = "profile_file"
EV_KIND_LOG:               Final[str] = "log"
EV_KIND_POINT_ID:          Final[str] = "point_id"
EV_KIND_EDGE_ID:           Final[str] = "edge_id"

# Propose response status (Literal[4]) — H1 only emits auto_accepted
STATUS_AUTO_ACCEPTED:      Final[str] = "auto_accepted"
STATUS_PENDING_COMMITTEE:  Final[str] = "pending_committee"
STATUS_PENDING_HUMAN:      Final[str] = "pending_human"
STATUS_REJECTED:           Final[str] = "rejected"

# Registered KB kinds (§5; only the names the fact-write surface uses)
KIND_RECIPE:               Final[str] = "recipe"
KIND_LESSON:               Final[str] = "lesson"
KIND_PITFALL:              Final[str] = "pitfall"

# Edge ``attrs.relation`` — semantic secondary labels (kg-usage-guide §7.3).
# Must pair with a specific ``edge_type``; the propose-edge validator
# rejects mismatched pairs. Carried on every direct edge written by the
# fact-write helpers so cascade / warm-start renderers can follow the
# semantic line.
REL_CITES:                 Final[str] = "cites"            # empirical
REL_GROUNDED_IN:           Final[str] = "grounded_in"      # causal
REL_TESTED:                Final[str] = "tested"           # investigation
REL_ON_RECIPE:             Final[str] = "on_recipe"        # structural
REL_REVEALS:               Final[str] = "reveals"          # causal
REL_HAS_PITFALL:           Final[str] = "has_pitfall"      # negation
REL_BEST_KEEP_FROM:        Final[str] = "best_keep_from"   # causal
REL_NEGATION_OF:           Final[str] = "negation_of"      # negation
REL_SAME_FILE_AS:          Final[str] = "same_file_as"     # structural
REL_SUPERSEDES:            Final[str] = "supersedes"       # evolutionary

# Pitfall severity vocab (kg-usage-guide §7.4). hyperloom maps:
#   crash / oom / hang        → SEVERITY_CRASH
#   gain_pct <= -5%           → SEVERITY_REGRESS
#   anything else REVERT      → no pitfall written (noise)
SEVERITY_CRASH:            Final[str] = "crash"
SEVERITY_REGRESS:          Final[str] = "regress"
SEVERITY_NOOP:             Final[str] = "noop"

# ---------------------------------------------------------------------------
# Defaults — bypass-mode semantics
# ---------------------------------------------------------------------------
# Two timeout / retry profiles depending on the caller's tolerance for
# being blocked on a slow / unreachable KB:
#
# * **foreground** — Coordinator on the main event loop. Every fact-
#   write (one per KEEP / REVERT, ~10 / EXPLORE round) is a sync HTTP
#   call. With the legacy 10s × 3 retries × backoff, a single KB
#   stall could block the optimizer for ~32s per write, snowballing
#   to >5min per EXPLORE round when KB is sick. Operator requirement:
#   "KB is a side-channel — if it's unavailable / slow, do NOT slow
#   the main logic". So the foreground profile fails fast (2s + 1
#   retry ≈ ~2.5s worst case) and the write falls through to NDJSON
#   on the very first transport hiccup. Cost: more rows queued; the
#   kb_flusher daemon picks them up in the background.
#
# * **background** — kb_flusher daemon + CLOSE-time drain. These run
#   outside the main loop so they can afford the legacy retry budget
#   (10s × 3) to maximize the chance a transient KB blip still
#   commits without dead-lettering the row.
DEFAULT_HTTP_TIMEOUT_SEC:  Final[float] = 10.0  # background / flusher
DEFAULT_MAX_CONCURRENCY:   Final[int] = 8       # aligned with asyncpg pool
DEFAULT_RETRY_ATTEMPTS:    Final[int] = 3
DEFAULT_RETRY_BASE_MS:     Final[int] = 200     # 200ms × {1, 1.4, 4}
FOREGROUND_HTTP_TIMEOUT_SEC: Final[float] = 2.0   # Coordinator main loop
FOREGROUND_RETRY_ATTEMPTS:   Final[int] = 1       # fail fast → NDJSON
# Maximum number of times a single NDJSON row may be re-attempted by
# the drain loop before it is treated as permanent and moved to the
# dead-letter counter. Protects against infinite retry loops when a
# dependency (e.g. ``propose_point`` for an edge endpoint) never
# becomes resolvable. ``attempts`` is incremented on every transient
# (``transport`` / ``business:NOT_FOUND`` / ``unknown``) classification.
MAX_FLUSH_ATTEMPTS:        Final[int] = 5
DEFAULT_GENERATOR:         Final[str] = "hyperloom"
SMOKE_GENERATOR:           Final[str] = "hyperloom-smoke"


__all__ = [
    "DEFAULT_KB_URL",
    "PATH_HEALTH",
    "PATH_PROPOSE_POINT",
    "PATH_QUERY_POINT",
    "PATH_TRAVERSE",
    "PATH_PROPOSE_EDGE",
    "PATH_NEGATE_EDGE",
    "PATH_ADD_EVIDENCE",
    "F_ATTRS",
    "F_FROM_POINT",
    "F_TO_POINT",
    "F_EDGE_TYPE",
    "F_EVIDENCE_REFS",
    "F_PROMOTED_EDGE_ID",
    "F_NEGATION_EDGE_ID",
    "F_STATUS",
    "F_CANONICAL_ID",
    "F_KIND",
    "F_ENTITY_TYPE",
    "F_AUTHORITY",
    "F_PROVENANCE",
    "F_PROPOSAL_ID",
    "F_POINT_ID",
    "F_COMMITTEE",
    "F_LIMIT",
    "F_NEIGHBOR_PREVIEW",
    "F_NEIGHBOR_LIMIT",
    "F_ATTRS_FILTER",
    "F_LENS",
    "F_POINTS",
    "F_NEIGHBORS",
    "F_START_POINT",
    "F_BUDGET_STEPS",
    "F_BUDGET_BRANCHES",
    "F_PATHS",
    "F_CANDIDATES",
    "F_LENS_SCHEDULE",
    "F_EV_KIND",
    "F_EV_REF",
    "F_EV_NOTE",
    "F_PV_SOURCE",
    "F_PV_GENERATOR",
    "F_PV_GENERATED_AT",
    "F_PV_DETAILS",
    "EDGE_STRUCTURAL",
    "EDGE_CAUSAL",
    "EDGE_INVESTIGATION",
    "EDGE_EVOLUTIONARY",
    "EDGE_EMPIRICAL",
    "EDGE_NEGATION",
    "REL_CITES",
    "REL_GROUNDED_IN",
    "REL_TESTED",
    "REL_ON_RECIPE",
    "REL_REVEALS",
    "REL_HAS_PITFALL",
    "REL_BEST_KEEP_FROM",
    "REL_NEGATION_OF",
    "REL_SAME_FILE_AS",
    "REL_SUPERSEDES",
    "SEVERITY_CRASH",
    "SEVERITY_REGRESS",
    "SEVERITY_NOOP",
    "AUTHORITY_AUTHORITATIVE",
    "AUTHORITY_EXPERIENTIAL",
    "AUTHORITY_INFERRED",
    "AUTHORITY_HYPOTHESIZED",
    "SOURCE_OFFLINE_INGEST",
    "SOURCE_AGENT_OBSERVATION",
    "SOURCE_CASCADE_DERIVED",
    "SOURCE_CO_OCCURRENCE",
    "SOURCE_ANALOGY_SEED",
    "EV_KIND_URL",
    "EV_KIND_COMMIT",
    "EV_KIND_PROFILE_FILE",
    "EV_KIND_LOG",
    "EV_KIND_POINT_ID",
    "EV_KIND_EDGE_ID",
    "STATUS_AUTO_ACCEPTED",
    "STATUS_PENDING_COMMITTEE",
    "STATUS_PENDING_HUMAN",
    "STATUS_REJECTED",
    "KIND_RECIPE",
    "KIND_LESSON",
    "KIND_PITFALL",
    "DEFAULT_HTTP_TIMEOUT_SEC",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_BASE_MS",
    "FOREGROUND_HTTP_TIMEOUT_SEC",
    "FOREGROUND_RETRY_ATTEMPTS",
    "MAX_FLUSH_ATTEMPTS",
    "DEFAULT_GENERATOR",
    "SMOKE_GENERATOR",
]
