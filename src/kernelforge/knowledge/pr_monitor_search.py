"""Discover, enrich, filter, and rank upstream PR references."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from kernelforge.knowledge.pr_monitor_client import (
    FetchOutcome,
    PRContractError,
    PRMonitorClient,
    PRMonitorError,
    extract_items,
)
from kernelforge.knowledge.pr_query_context import (
    REASON_CONTRACT_ERROR,
    REASON_NO_CANDIDATE,
    REASON_SERVICE_UNREACHABLE,
    REASON_SKIPPED_DEADLINE,
    PRQueryContext,
)

log = logging.getLogger(__name__)

HIT_FILE_PATH = "file_path"
HIT_SEARCH = "search"
HIT_RECENT = "recent_merged"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_CAP = 10
MAX_PATH_QUERIES = 3
MAX_KEYWORD_QUERIES = 4
FALLBACK_LIMIT = 5

# Recent-only candidates need a positive score because no query linked them.
FALLBACK_MIN_WORTH = 0.3
# Established path and keyword hits are unfiltered by default.
DEFAULT_MIN_WORTH = 0.0

# Ranking treats an unknown score as worse than the lowest real one (0.0).
_UNKNOWN_WORTH = -1.0


@dataclass(frozen=True)
class PRReference:
    """One enriched upstream PR, ready for ranking and rendering."""

    repo: str
    number: int
    title: str = ""
    hit_via: tuple[str, ...] = ()
    is_merged: bool = False
    worth_trying: float | None = None
    components: tuple[str, ...] = ()
    mechanisms: tuple[str, ...] = ()
    summary: str = ""
    risk_notes: str = ""
    expected_gain: str = ""
    head_sha: str = ""
    schema_version: str = ""
    updated_at: str = ""
    n_files: int = 0
    distill_absent: bool = False


@dataclass
class SearchOutcome:
    """Ranked references, outcome reason, and request counters."""

    references: tuple[PRReference, ...] = ()
    surfaced_references: tuple[PRReference, ...] = ()
    reason: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def _summary_of(detail: dict) -> dict:
    """Return the summary sub-object, tolerating a flattened payload."""
    summary = detail.get("summary")
    return summary if isinstance(summary, dict) else detail


def _distill_of(detail: dict) -> dict:
    """Return the inlined distill object, or an empty dict when absent."""
    distill = detail.get("distill")
    return distill if isinstance(distill, dict) else {}


def _str_list(value: Any, limit: int = 8) -> tuple[str, ...]:
    """Coerce a payload field into a bounded tuple of non-empty strings."""
    if not isinstance(value, list):
        return ()
    items = [str(item).strip() for item in value if str(item).strip()]
    return tuple(items[:limit])


def _numbers_from(items: list[dict], *, repo: str = "") -> list[int]:
    """Extract PR numbers from list and nested search rows."""
    numbers: list[int] = []
    for item in items:
        row = item
        if "number" not in row and isinstance(item.get("summary"), dict):
            row = item["summary"]
        # A search may be issued without a repo filter; drop foreign rows so a
        # candidate can never come from a repository the caller did not ask for.
        if repo and row.get("repo_name") not in (None, repo):
            continue
        try:
            numbers.append(int(row.get("number")))
        except (TypeError, ValueError):
            continue
    return numbers


def _candidate_requests(
    context: PRQueryContext,
) -> list[tuple[str, dict[str, Any], str]]:
    """Build stage-1 requests as (path, params, hit_kind), all bounded pages."""
    requests: list[tuple[str, dict[str, Any], str]] = []
    for path in context.file_paths[:MAX_PATH_QUERIES]:
        requests.append(
            (
                f"/repos/{context.repo}/prs",
                {"file_path": path, "state": "all", "limit": 5},
                HIT_FILE_PATH,
            )
        )
    for phrase in context.keywords[:MAX_KEYWORD_QUERIES]:
        # One request per phrase: the server ILIKEs the whole query string, so
        # joining phrases would drive the hit count to zero.
        requests.append(
            (
                "/search/prs",
                {"q": phrase, "repo": context.repo, "limit": 20},
                HIT_SEARCH,
            )
        )
    return requests


def remaining_sec(deadline: float) -> float:
    """Seconds left before the shared end-to-end deadline."""
    return deadline - time.monotonic()


def _collect_candidates(
    client: PRMonitorClient,
    context: PRQueryContext,
    *,
    deadline: float,
    stats: dict[str, Any],
) -> tuple[dict[int, set[str]], str]:
    """Run stages 1a and 1b concurrently; fall back to 1c only if both are empty."""
    candidates: dict[int, set[str]] = {}
    failure_reason = ""
    planned = _candidate_requests(context)
    if planned:
        remaining = remaining_sec(deadline)
        if remaining <= 0:
            return candidates, REASON_SKIPPED_DEADLINE
        outcomes = client.get_many(
            [(path, params) for path, params, _ in planned],
            budget_sec=remaining,
        )
        stats["http_calls"] = stats.get("http_calls", 0) + len(outcomes)
        for (_, params, kind), outcome in zip(planned, outcomes):
            failure, seen = _absorb(outcome, kind, candidates, repo=context.repo)
            if failure and (failure == REASON_CONTRACT_ERROR or not failure_reason):
                failure_reason = failure
            if not failure and outcome.payload is not None and seen == 0:
                # A query that provably returned nothing is stable for a fixed
                # target, so record it for the negative cache instead of
                # re-issuing it on every refresh. A query that merely re-hit
                # known candidates is not empty and must not be recorded.
                stats.setdefault("empty_queries", []).append(
                    (kind, str(params.get("file_path") or params.get("q") or ""))
                )

    if not candidates:
        remaining = remaining_sec(deadline)
        if remaining <= 0:
            # The fallback is the least precise stage; it never gets to spend
            # time the caller no longer has.
            return candidates, failure_reason or REASON_SKIPPED_DEADLINE
        stats["fallback_used"] = True
        try:
            items = client.list_recent_prs(context.repo, limit=FALLBACK_LIMIT, timeout_sec=remaining)
            stats["http_calls"] = stats.get("http_calls", 0) + 1
        except PRContractError as error:
            log.error("pr-monitor fallback contract error: %s", error)
            failure_reason = REASON_CONTRACT_ERROR
            items = []
        except PRMonitorError as error:
            log.warning("pr-monitor fallback unavailable: %s", error)
            if not failure_reason:
                failure_reason = REASON_SERVICE_UNREACHABLE
            items = []
        for number in _numbers_from(items):
            candidates.setdefault(number, set()).add(HIT_RECENT)
    return candidates, failure_reason


def _absorb(
    outcome: FetchOutcome,
    kind: str,
    candidates: dict[int, set[str]],
    *,
    repo: str = "",
) -> tuple[str, int]:
    """Fold one response into candidates and return its failure and row count."""
    if isinstance(outcome.error, PRContractError):
        return REASON_CONTRACT_ERROR, 0
    if outcome.error is not None:
        return REASON_SERVICE_UNREACHABLE, 0
    if outcome.payload is None:
        return "", 0
    try:
        items = extract_items(outcome.payload)
    except PRContractError:
        return REASON_CONTRACT_ERROR, 0
    numbers = _numbers_from(items, repo=repo)
    for number in numbers:
        candidates.setdefault(number, set()).add(kind)
    return "", len(numbers)


def _order_candidates(candidates: dict[int, set[str]], cap: int) -> list[int]:
    """Cap the pool, preferring path hits so enrichment spends on the best leads."""
    ranked = sorted(
        candidates,
        key=lambda number: (
            HIT_FILE_PATH in candidates[number],
            HIT_SEARCH in candidates[number],
            number,
        ),
        reverse=True,
    )
    return ranked[:cap]


def worth_floors() -> tuple[float, float]:
    """Read the (global, fallback-only) score floors from the environment."""
    return (
        float(os.environ.get("PR_KB_MIN_WORTH", "").strip() or DEFAULT_MIN_WORTH),
        float(os.environ.get("PR_KB_FALLBACK_MIN_WORTH", "").strip() or FALLBACK_MIN_WORTH),
    )


def _below_worth_floor(reference: PRReference, floors: tuple[float, float]) -> bool:
    """Apply the global or recent-only score floor by provenance."""
    minimum, fallback_minimum = floors
    worth = reference.worth_trying
    if tuple(reference.hit_via) == (HIT_RECENT,):
        return worth is None or worth < fallback_minimum
    return worth is not None and worth < minimum


def _build_reference(repo: str, number: int, detail: dict, hit_via: set[str]) -> PRReference | None:
    """Build a reference, dropping explicit negative distill results."""
    distill = _distill_of(detail)
    status = str(distill.get("status") or "")
    if status and status != "ok":
        return None
    summary = _summary_of(detail)
    worth = distill.get("worth_trying")
    return PRReference(
        repo=repo,
        number=number,
        title=str(summary.get("title") or ""),
        hit_via=tuple(sorted(hit_via)),
        # The service exposes ``is_merged``, not ``merged_at``.
        is_merged=bool(summary.get("is_merged")),
        worth_trying=float(worth) if isinstance(worth, (int, float)) else None,
        components=_str_list(distill.get("components")),
        mechanisms=_str_list(distill.get("mechanisms")),
        summary=str(distill.get("summary") or ""),
        risk_notes=str(distill.get("risk_notes") or ""),
        expected_gain=str(distill.get("expected_gain") or ""),
        head_sha=str(distill.get("head_sha") or summary.get("head_sha") or ""),
        schema_version=str(distill.get("schema_version") or ""),
        updated_at=str(summary.get("pr_updated_at") or ""),
        # ``summary.changed_files`` is null; count the files array.
        n_files=len(detail.get("files") or []),
        distill_absent=not distill,
    )


def component_relevance(components: tuple[str, ...], interest: frozenset[str]) -> float:
    """Return the fraction of components matching query terms."""
    if not components or not interest:
        return 0.0
    matched = 0
    for component in components:
        lowered = component.lower()
        tokens = {token for token in re.split(r"[^0-9a-z]+", lowered) if token}
        if tokens & interest or any(term in lowered for term in interest):
            matched += 1
    return matched / len(components)


def filter_references_by_relevance(references: list[PRReference], interest: frozenset[str]) -> list[PRReference]:
    """Keep exact path history and component-related search results."""
    if not interest:
        return references
    return [
        reference
        for reference in references
        if HIT_FILE_PATH in reference.hit_via
        or reference.distill_absent
        or component_relevance(reference.components, interest) > 0.0
    ]


def rank_references(
    references: list[PRReference], *, components_of_interest: frozenset[str] = frozenset()
) -> list[PRReference]:
    """Order by path hit, component relevance, score, merge state, and recency."""

    def sort_key(ref: PRReference) -> tuple:
        """Rank one reference; every element is descending-better."""
        return (
            HIT_FILE_PATH in ref.hit_via,
            component_relevance(ref.components, components_of_interest),
            ref.worth_trying if ref.worth_trying is not None else _UNKNOWN_WORTH,
            ref.is_merged,
            ref.updated_at,
        )

    return sorted(references, key=sort_key, reverse=True)


def components_of_interest(context: PRQueryContext) -> frozenset[str]:
    """Terms a PR's distill components are matched against for the rank bonus."""
    terms = {token for phrase in context.keywords for token in phrase.split()}
    terms.update(part for path in context.file_paths for part in _path_terms(path))
    return frozenset(term.lower() for term in terms if term)


def _path_terms(path: str) -> list[str]:
    """Directory and stem names from a repo-relative path."""
    parts = [segment for segment in path.split("/") if segment]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return parts


def discover(
    client: PRMonitorClient,
    context: PRQueryContext,
    *,
    top_k: int = 0,
    candidate_cap: int = 0,
    budget_sec: float = 0.0,
    deadline: float | None = None,
) -> SearchOutcome:
    """Discover ranked references; zero-valued limits use ``PR_KB_*`` settings.

    ``deadline`` is the caller's absolute end-to-end cutoff and outranks
    ``budget_sec``, which only seeds one when discovery is the whole operation.
    """
    if context.reason:
        return SearchOutcome(reason=context.reason, stats={"http_calls": 0})
    top_k = top_k or int(os.environ.get("PR_KB_TOP_K", DEFAULT_TOP_K) or DEFAULT_TOP_K)
    candidate_cap = candidate_cap or int(
        os.environ.get("PR_KB_CANDIDATE_CAP", DEFAULT_CANDIDATE_CAP) or DEFAULT_CANDIDATE_CAP
    )
    if deadline is None:
        budget = budget_sec or float(os.environ.get("PR_KB_BUDGET_SEC", "30") or 30)
        deadline = time.monotonic() + budget
    stats: dict[str, Any] = {"http_calls": 0, "fallback_used": False}

    candidates, failure_reason = _collect_candidates(client, context, deadline=deadline, stats=stats)
    stats["candidates"] = len(candidates)
    if not candidates:
        reason = failure_reason or REASON_NO_CANDIDATE
        return SearchOutcome(reason=reason, stats=stats)

    numbers = _order_candidates(candidates, candidate_cap)

    remaining = remaining_sec(deadline)
    if remaining <= 0:
        # Enrichment without time returns nothing but still costs the caller
        # its finalization reserve.
        stats["degraded_reason"] = REASON_SKIPPED_DEADLINE
        return SearchOutcome(reason=REASON_SKIPPED_DEADLINE, stats=stats)
    outcomes = client.get_many(
        [client.pr_request(context.repo, number) for number in numbers],
        budget_sec=remaining,
    )
    stats["http_calls"] += len(outcomes)

    references: list[PRReference] = []
    dropped = 0
    absent = 0
    for number, outcome in zip(numbers, outcomes):
        if isinstance(outcome.error, PRContractError):
            failure_reason = REASON_CONTRACT_ERROR
            continue
        if outcome.error is not None:
            if not failure_reason:
                failure_reason = REASON_SERVICE_UNREACHABLE
            continue
        if outcome.payload is None:
            continue
        if not isinstance(outcome.payload, dict):
            failure_reason = REASON_CONTRACT_ERROR
            continue
        reference = _build_reference(context.repo, number, outcome.payload, candidates[number])
        if reference is None:
            dropped += 1
            continue
        absent += int(reference.distill_absent)
        references.append(reference)

    stats["distill_dropped"] = dropped
    stats["distill_absent"] = absent

    floors = worth_floors()
    kept = [ref for ref in references if not _below_worth_floor(ref, floors)]
    interest = components_of_interest(context)
    relevant = filter_references_by_relevance(kept, interest)
    stats["relevance_dropped"] = len(kept) - len(relevant)
    ranked = rank_references(relevant, components_of_interest=interest)
    stats["surfaced"] = len(ranked)
    if failure_reason == REASON_CONTRACT_ERROR:
        log.error("pr-monitor contract error during discovery for %s", context.repo)
    elif failure_reason == REASON_SERVICE_UNREACHABLE:
        log.warning("pr-monitor request failed during discovery for %s", context.repo)
    if failure_reason:
        stats["degraded_reason"] = failure_reason
    if not ranked:
        reason = failure_reason or REASON_NO_CANDIDATE
        return SearchOutcome(reason=reason, stats=stats)
    return SearchOutcome(
        references=tuple(ranked[:top_k]),
        surfaced_references=tuple(ranked),
        stats=stats,
    )
