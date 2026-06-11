# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Library entry-points for LLM specialists (Arbor / TBO / Hyperloom).

Three helpers wrapping framework-agent internals without a full
:class:`ExploreRequest` JSON:

* :func:`find_relevant_prs_smart`  - cross-repo PR discovery via
  primus_cortex + (optional) anonymous GitHub Search.
* :func:`fetch_pr_audit_material`  - download ``pr.patches`` +
  ``pr_files.json`` for one PR.
* :func:`evaluate_candidate_outcome` - stateless winner check given
  pre-computed benchmark/accuracy JSON blobs.

The CLI does not depend on this module and vice versa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import Candidate
from ..sources import github as github_backend
from ..sources._shared import GitHubPr, _repo_slug
from ..sources.primus_cortex import (
    PrimusCortexError,
    list_perf_prs,
    pr_files,
    pr_patches,
)


def _pr_to_candidate(pr: GitHubPr, repo_url: str, source: str) -> Candidate:
    """Map a GitHubPr record into the shared Candidate shape.

    Args:
        pr (GitHubPr): Source PR record from a discovery backend.
        repo_url (str): Repo URL to record on the candidate.
        source (str): Origin tag (e.g. ``"primus_cortex"`` or ``"github"``).

    Returns:
        Candidate: A candidate carrying the PR ref, repo, title, and URL.
    """
    return Candidate(
        ref=pr.ref,
        repo=repo_url,
        source=source,
        title=pr.title,
        html_url=pr.html_url,
    )


def find_relevant_prs_smart(
    gap_description: str,
    repos: list[str] | None = None,
    *,
    primus_cortex_url: str | None = None,
    primus_timeout_sec: float = 10.0,
    limit_per_repo: int = 5,
    primus_state: str = "open",
    primus_label: str | None = None,
    include_github: bool = True,
) -> list[Candidate]:
    """Discover candidate PRs across one or more repos.

    Plain-arg version of
    :func:`framework_agent.sources.enumerate_candidates`. Each repo is
    queried via primus_cortex (hard-fail) when ``primus_cortex_url`` is set;
    GitHub Search is a best-effort secondary when ``include_github=True``
    (returns ``[]``, never raises). Results are de-duped by
    ``(repo_url, ref)`` so primus_cortex wins ties.

    Args:
        gap_description: Description used to drive GitHub search relevance.
        repos: Repos to query; ``None``/empty yields ``[]``.
        primus_cortex_url: Primus Cortex base URL; enables that source.
        primus_timeout_sec: Per-request timeout for Primus calls.
        limit_per_repo: Max candidates per repo per source.
        primus_state: PR state filter for Primus (e.g. ``"open"``).
        primus_label: Optional label filter for Primus.
        include_github: Whether to also query GitHub search.

    Returns:
        De-duplicated candidate list across all repos and sources.

    Raises:
        PrimusCortexError: If a Primus query fails when configured.
    """
    if not repos:
        return []
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for repo_url in repos:
        if primus_cortex_url:
            try:
                prs = list_perf_prs(
                    repo_url,
                    base_url=primus_cortex_url,
                    limit=limit_per_repo,
                    state=primus_state,
                    label=primus_label,
                    timeout_sec=primus_timeout_sec,
                )
            except PrimusCortexError:
                raise
            for pr in prs:
                key = (repo_url, pr.ref)
                if key in seen:
                    continue
                seen.add(key)
                out.append(_pr_to_candidate(pr, repo_url, "primus_cortex"))
        if include_github:
            gh_prs = github_backend.search_perf_prs(
                repo_url,
                gap_description=gap_description,
                limit=limit_per_repo,
            )
            for pr in gh_prs:
                key = (repo_url, pr.ref)
                if key in seen:
                    continue
                seen.add(key)
                out.append(_pr_to_candidate(pr, repo_url, "github"))
    return out


def fetch_pr_audit_material(
    repo_url: str,
    pr_number: int,
    *,
    out_dir: Path | str,
    primus_cortex_url: str,
    primus_timeout_sec: float = 30.0,
) -> dict[str, str]:
    """Download a PR's audit material (patches + files) under ``out_dir``.

    Writes ``pr.patches`` (unified diff) and ``pr_files.json``
    (``{repo, number, files}``).

    Args:
        repo_url: Repository URL the PR belongs to.
        pr_number: PR number to fetch.
        out_dir: Directory to write the artifacts into (created if needed).
        primus_cortex_url: Primus Cortex base URL.
        primus_timeout_sec: Per-request timeout for Primus calls.

    Returns:
        A dict of absolute artifact paths (``patches_path``,
        ``files_json_path``).

    Raises:
        PrimusCortexError: On Primus transport / parse errors.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    repo_slug = _repo_slug(repo_url)
    patches_text = pr_patches(
        repo_slug, pr_number, base_url=primus_cortex_url, timeout_sec=primus_timeout_sec
    )
    patches_path = out / "pr.patches"
    patches_path.write_text(patches_text, encoding="utf-8")
    files_payload = pr_files(
        repo_slug, pr_number, base_url=primus_cortex_url, timeout_sec=primus_timeout_sec
    )
    files_path = out / "pr_files.json"
    files_path.write_text(
        json.dumps(
            {
                "repo": repo_slug,
                "number": pr_number,
                "files": files_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "patches_path": str(patches_path),
        "files_json_path": str(files_path),
    }


def _coerce_dict(value: dict | Path | str | None) -> dict:
    """Accept a dict, a Path to a JSON file, or a str path; return dict.

    Args:
        value (dict | Path | str | None): A dict (returned as-is), a path to a
            JSON file, or ``None``.

    Returns:
        dict: The dict value, the parsed JSON object, or ``{}`` when the input
            is missing, unreadable, or not a JSON object.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    path = Path(value) if isinstance(value, (str, Path)) else None
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _metric_float(data: dict, keys: Iterable[str]) -> float | None:
    """Return the first int/float among ``keys`` in ``data``.

    Args:
        data (dict): Mapping to look up.
        keys (Iterable[str]): Candidate keys checked in order.

    Returns:
        float | None: The first numeric value coerced to float, or ``None`` if
            no key holds a numeric value.
    """
    for k in keys:
        v = data.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def evaluate_candidate_outcome(
    benchmark: dict | Path | str | None,
    accuracy: dict | Path | str | None = None,
    *,
    baseline_throughput: float,
    baseline_accuracy: float | None = None,
    min_throughput_ratio: float = 1.05,
    max_accuracy_drop: float = 0.05,
) -> dict:
    """Stateless winner check given pre-computed benchmark/accuracy data.

    Gate logic matches the CLI's winner decision. Inputs accept a dict, a
    Path, or a string path; missing/invalid inputs yield a ``False`` verdict
    with a reason rather than raising.

    Args:
        benchmark: Benchmark data (dict, JSON path, or ``None``).
        accuracy: Accuracy data (dict, JSON path, or ``None``).
        baseline_throughput: Baseline throughput; must be positive.
        baseline_accuracy: Baseline accuracy, if accuracy is gated.
        min_throughput_ratio: Minimum throughput ratio to win.
        max_accuracy_drop: Maximum tolerated accuracy drop.

    Returns:
        A dict with keys ``winner``, ``reason``, ``throughput``,
        ``accuracy``, ``throughput_ratio``, and ``completed``.

    Raises:
        ValueError: If ``baseline_throughput`` is not a positive float.
    """
    if baseline_throughput is None or baseline_throughput <= 0:
        raise ValueError("baseline_throughput must be a positive float")

    bench = _coerce_dict(benchmark)
    acc = _coerce_dict(accuracy)
    throughput = _metric_float(bench, ("throughput", "output_throughput", "tput"))
    acc_value = _metric_float(acc, ("accuracy", "gsm8k", "exact_match", "score"))
    completed = str(bench.get("completed") or bench.get("Completed") or "")

    result: dict = {
        "winner": False,
        "reason": "",
        "throughput": throughput,
        "accuracy": acc_value,
        "throughput_ratio": None,
        "completed": completed,
    }

    if throughput is None or throughput <= 0:
        result["reason"] = "missing throughput"
        return result
    ratio = throughput / baseline_throughput
    result["throughput_ratio"] = ratio
    if ratio < min_throughput_ratio:
        result["reason"] = (
            f"throughput ratio {ratio:.4f} below required {min_throughput_ratio:.4f}"
        )
        return result
    if baseline_accuracy is not None:
        if acc_value is None:
            result["reason"] = "missing accuracy while baseline accuracy is set"
            return result
        drop = baseline_accuracy - acc_value
        if drop > max_accuracy_drop:
            result["reason"] = (
                f"accuracy drop {drop:.4f} exceeds max {max_accuracy_drop:.4f}"
            )
            return result
    if completed and "/" in completed:
        left, _, right = completed.partition("/")
        if left.strip() != right.strip():
            result["reason"] = f"benchmark completed={completed} is incomplete"
            return result
    result["winner"] = True
    result["reason"] = "throughput and accuracy gates passed"
    return result


__all__ = [
    "find_relevant_prs_smart",
    "fetch_pr_audit_material",
    "evaluate_candidate_outcome",
]
