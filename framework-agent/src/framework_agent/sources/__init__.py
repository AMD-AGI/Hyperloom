"""PR candidate source dispatcher.

Routes :class:`ExploreRequest` to one or more backends based on
``request.search_modes`` and merges results into a single deduplicated
list of :class:`Candidate` records.

Phase A backends:

* ``primus_cortex`` - internal REST service (hard-fail on errors).

Phase B will add:

* ``github`` - anonymous GitHub Search fallback (best-effort, may be
  rate-limited).

Dispatcher contract:

* If ``search_modes`` is empty -> return ``[]`` (caller must rely on
  ``candidate_refs``).
* If a mode is configured but its config is missing (e.g. ``primus_cortex``
  requested without ``primus_cortex`` block / env var) -> raise
  :class:`SourceConfigError`.
* Per-mode errors propagate per the backend's policy (primus_cortex
  hard-fails; github will be best-effort in Phase B).
"""

from __future__ import annotations

from typing import Iterable

from ..models import Candidate, ExploreRequest
from ._shared import GitHubPr
from .primus_cortex import PrimusCortexError, list_perf_prs


class SourceConfigError(RuntimeError):
    """Raised when a requested search_mode is missing its configuration."""


def _dedupe(items: Iterable[Candidate]) -> list[Candidate]:
    """Stable-deduplicate candidates by ref, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for item in items:
        if item.ref in seen:
            continue
        seen.add(item.ref)
        out.append(item)
    return out


def _pr_to_candidate(pr: GitHubPr, repo_url: str, source: str) -> Candidate:
    """Convert a GitHubPr (any backend) into a downstream Candidate."""
    return Candidate(
        ref=pr.ref,
        repo=repo_url,
        source=source,
        title=pr.title,
        html_url=pr.html_url,
    )


def enumerate_candidates(request: ExploreRequest) -> list[Candidate]:
    """Enumerate candidates per ``request.search_modes`` and union the results.

    Order:
      1. Explicit ``candidate_refs`` (always first; source='explicit').
      2. For each enabled mode in ``request.search_modes``: query the
         backend, map to Candidate(source=mode), append.
      3. Deduplicate by ref, preserving the first occurrence.

    Hard-fails when ``primus_cortex`` is requested without configuration,
    or when the primus-cortex transport fails.
    """
    out: list[Candidate] = []

    for ref in request.candidate_refs:
        out.append(Candidate(ref=ref, repo=request.repo_url, source="explicit"))

    if not request.search_perf_prs:
        return _dedupe(out)

    for mode in request.search_modes:
        if mode == "primus_cortex":
            out.extend(_run_primus_cortex(request))
        elif mode == "github":
            continue
        else:
            raise SourceConfigError(f"unknown search_mode: {mode!r}")

    return _dedupe(out)


def _run_primus_cortex(request: ExploreRequest) -> list[Candidate]:
    """Query primus-cortex; hard-fail on transport/config errors."""
    cfg = request.primus_cortex
    if cfg is None:
        raise SourceConfigError(
            "search_modes contains 'primus_cortex' but no primus_cortex "
            "block was provided (nor PRIMUS_CORTEX_PR_API env var)"
        )
    label = cfg.default_label
    try:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=request.max_search_candidates,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )
    except PrimusCortexError:
        raise
    return [_pr_to_candidate(pr, request.repo_url, "primus_cortex") for pr in prs]


__all__ = [
    "SourceConfigError",
    "enumerate_candidates",
]
