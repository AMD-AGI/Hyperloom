# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR/ref exploration engine; candidates come from ``sources.enumerate_candidates`` (honours ``search_modes``), enrichment + filtering happen here.

Two run modes:

* ``execute=False`` (plan) - drop ``pr.patches`` + ``pr_files.json`` per PR
  candidate and produce a planned ``explore_summary.json``; no worktree /
  venv / build / bench.
* ``execute=True`` - additionally create a detached git worktree +
  per-candidate venv, then run the request's ``build`` / ``benchmark`` /
  ``accuracy`` / ``cleanup`` commands. Promotion stays manual.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json

from .decision import candidate_score, prior_score, winner_decision
from .isolation import (
    WorkspacePaths,
    cleanup_workspace,
    disk_preflight,
    prepare_candidate_workspace,
    prepare_repo_cache,
)
from .kb import read_pr_ledger
from .keywords import extract_keywords
from .logging_setup import get_logger, stage_log
from .models import Candidate, CandidateResult, ExploreRequest, Finding, PrFilter
from hyperloom.common.coerce import first_float as _first_float
from .shell import render_template, run_command
from .sources import pr_monitor
from .sources._shared import _repo_slug


log = get_logger(__name__)


def _coalesce_str(*values: Any) -> str:
    """Return the first non-empty stripped string in ``values``; else ''.

    Args:
        *values (Any): Candidate values; non-strings and blank strings are
            skipped.

    Returns:
        str: The first stripped non-empty string, or ``""`` if none qualify.
    """
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summary_of(detail: dict[str, Any]) -> dict[str, Any]:
    """Return the nested ``summary`` mapping from a PR detail payload.

    pr_monitor wraps PR metadata under a ``summary`` key; this defaults
    to ``{}`` when it is absent or not a dict.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        dict[str, Any]: The nested summary dict, or ``{}`` when missing.
    """
    summary = detail.get("summary")
    return summary if isinstance(summary, dict) else {}


def _extract_head_sha(detail: dict[str, Any]) -> str:
    """Pull head SHA out of any of the known keys in a PR detail payload.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        str: The head commit SHA, or ``""`` when none of the keys carry it.
    """
    summary = _summary_of(detail)
    sha = _coalesce_str(
        summary.get("head_sha"),
        detail.get("git_fetched_head"),
        detail.get("head_sha"),
        detail.get("head_commit_sha"),
        detail.get("head_oid"),
        detail.get("head_ref_sha"),
    )
    if sha:
        return sha
    head = detail.get("head")
    if isinstance(head, dict):
        return _coalesce_str(head.get("sha"), head.get("oid"))
    if isinstance(head, str):
        return head.strip()
    return ""


def _extract_labels(detail: dict[str, Any]) -> tuple[str, ...]:
    """Pull labels out of a PR detail payload (handles dict and string items).

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        tuple[str, ...]: The label names, or an empty tuple when absent.
    """
    summary = _summary_of(detail)
    raw = summary.get("labels") if "labels" in summary else detail.get("labels")
    if raw is None:
        return ()
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                name = _coalesce_str(item.get("name"), item.get("label"))
                if name:
                    out.append(name)
    return tuple(out)


def _extract_author(detail: dict[str, Any]) -> str:
    """Pull author login from any known key in a PR detail payload.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        str: The author login/name, or ``""`` when none can be found.
    """
    summary = _summary_of(detail)
    summary_login = _coalesce_str(summary.get("author_login"), summary.get("author"))
    if summary_login:
        return summary_login
    raw = detail.get("author")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return _coalesce_str(raw.get("login"), raw.get("name"))
    user = detail.get("user")
    if isinstance(user, dict):
        return _coalesce_str(user.get("login"), user.get("name"))
    return _coalesce_str(detail.get("login"), detail.get("author_login"))


def _extract_title(detail: dict[str, Any]) -> str:
    """Pull title from a PR detail payload.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        str: The PR title, or ``""`` when absent.
    """
    summary = _summary_of(detail)
    return _coalesce_str(summary.get("title"), detail.get("title"))


def _extract_updated_at(detail: dict[str, Any]) -> str:
    """Pull the ``updated_at`` timestamp string from a PR detail payload.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        str: The update timestamp string, or ``""`` when absent.
    """
    summary = _summary_of(detail)
    return _coalesce_str(
        summary.get("pr_updated_at"),
        summary.get("updated_at"),
        detail.get("pr_updated_at"),
        detail.get("updated_at"),
        detail.get("updated"),
    )


def _extract_html_url(detail: dict[str, Any]) -> str:
    """Pull html_url from a PR detail payload.

    Args:
        detail (dict[str, Any]): The PR detail payload.

    Returns:
        str: The PR HTML URL, or ``""`` when absent.
    """
    summary = _summary_of(detail)
    return _coalesce_str(summary.get("html_url"), detail.get("html_url"), detail.get("url"))


def _extract_changed_files(detail: dict[str, Any], files_payload: list[dict[str, Any]]) -> tuple[str, ...]:
    """Pull changed-files list, preferring the dedicated files endpoint payload.

    Args:
        detail (dict[str, Any]): The PR detail payload (may embed a file list).
        files_payload (list[dict[str, Any]]): The dedicated files-endpoint
            payload, used in preference to the embedded list.

    Returns:
        tuple[str, ...]: The changed file paths, or an empty tuple when none
            are present.
    """
    out: list[str] = []
    for item in files_payload:
        path = _coalesce_str(item.get("path"), item.get("filename"), item.get("file_path"))
        if path:
            out.append(path)
    if out:
        return tuple(out)
    embedded = detail.get("files") or detail.get("changed_files")
    if isinstance(embedded, list):
        for item in embedded:
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
            elif isinstance(item, dict):
                path = _coalesce_str(item.get("path"), item.get("filename"), item.get("file_path"))
                if path:
                    out.append(path)
    return tuple(out)


def _enrich_candidate_via_pr_monitor(req: ExploreRequest, candidate: Candidate) -> Candidate:
    """Enrich a PR-typed candidate with metadata from PR Monitor.

    Fetches ``pr_get`` + ``pr_files`` for PR candidates; branch / tag /
    commit refs are returned unchanged.

    Args:
        req: The explore request carrying the PR Monitor config.
        candidate: The candidate to enrich.

    Returns:
        The enriched candidate, or the original when enrichment does not
        apply.

    Raises:
        pr_monitor.PRMonitorError: If the repo URL is malformed or a
            required PR Monitor call fails.
    """
    if req.pr_monitor is None:
        return candidate
    number = candidate.pr_number
    if number is None:
        return candidate
    try:
        repo_slug = _repo_slug(req.repo_url)
    except ValueError as exc:
        raise pr_monitor.PRMonitorError(f"cannot enrich {candidate.ref}: bad repo_url={req.repo_url!r}: {exc}") from exc
    base_url = req.pr_monitor.base_url
    timeout_sec = req.pr_monitor.timeout_sec
    detail = pr_monitor.pr_get(repo_slug, number, base_url=base_url, timeout_sec=timeout_sec)
    try:
        files_payload = pr_monitor.pr_files(repo_slug, number, base_url=base_url, timeout_sec=timeout_sec)
    except pr_monitor.PRMonitorError:
        files_payload = []
    return replace(
        candidate,
        head_sha=_extract_head_sha(detail) or candidate.head_sha,
        title=_extract_title(detail) or candidate.title,
        labels=_extract_labels(detail) or candidate.labels,
        author=_extract_author(detail) or candidate.author,
        changed_files=_extract_changed_files(detail, files_payload) or candidate.changed_files,
        updated_at=_extract_updated_at(detail) or candidate.updated_at,
        html_url=_extract_html_url(detail) or candidate.html_url,
    )


def _passes_filter(c: Candidate, f: PrFilter) -> tuple[bool, str]:
    """Apply a :class:`PrFilter` to one candidate.

    Label / author / path constraints fail when the required metadata is
    missing (e.g. enrichment was skipped); the date constraints instead
    pass, since each is guarded on a non-empty ``updated_at``.

    Args:
        c: The candidate to test.
        f: The filter to apply.

    Returns:
        ``(True, "")`` on an empty filter or a pass, otherwise
        ``(False, reason)`` describing the first failing constraint.
    """
    if f.is_empty:
        return True, ""

    labels_lc = {lbl.lower() for lbl in c.labels}
    if f.require_labels:
        required = {lbl.lower() for lbl in f.require_labels}
        if not required.issubset(labels_lc):
            return False, f"missing required label(s): {sorted(required - labels_lc)!r}"
    if f.exclude_labels:
        excluded = {lbl.lower() for lbl in f.exclude_labels}
        if labels_lc & excluded:
            return False, f"has excluded label(s): {sorted(labels_lc & excluded)!r}"

    if f.authors:
        if not c.author:
            return False, "author unknown but pr_filter.authors set"
        if c.author.lower() not in {a.lower() for a in f.authors}:
            return False, f"author {c.author!r} not in pr_filter.authors"

    if f.since and c.updated_at and c.updated_at < f.since:
        return False, f"updated_at {c.updated_at!r} < since {f.since!r}"
    if f.until and c.updated_at and c.updated_at > f.until:
        return False, f"updated_at {c.updated_at!r} > until {f.until!r}"

    if f.include_paths or f.exclude_paths or f.max_changed_files or f.min_changed_files:
        if not c.changed_files:
            return False, "no changed_files metadata (pr_monitor enrichment likely skipped)"

    if f.exclude_paths:
        for path in c.changed_files:
            for prefix in f.exclude_paths:
                if path.startswith(prefix):
                    return False, f"file {path!r} matches exclude_paths prefix {prefix!r}"
    if f.include_paths:
        hit = False
        for path in c.changed_files:
            for prefix in f.include_paths:
                if path.startswith(prefix):
                    hit = True
                    break
            if hit:
                break
        if not hit:
            return False, f"no changed file matched include_paths={list(f.include_paths)!r}"

    n_files = len(c.changed_files)
    if f.min_changed_files and n_files < f.min_changed_files:
        return False, f"changed_files={n_files} < min_changed_files={f.min_changed_files}"
    if f.max_changed_files and n_files > f.max_changed_files:
        return False, f"changed_files={n_files} > max_changed_files={f.max_changed_files}"

    return True, ""


def _enumerate_with_skipped(
    req: ExploreRequest,
) -> tuple[list[Candidate], list[dict[str, str]]]:
    """Enumerate, enrich, and filter candidates.

    Unions the configured sources, enriches PR candidates via PR Monitor, then
    applies ``req.pr_filter``. Explicit candidates bypass the filter
    (operator intent wins) but are still enriched.

    Args:
        req: The explore request.

    Returns:
        A ``(kept, skipped)`` tuple where ``skipped`` entries carry the
        ref, source, and skip reason.
    """
    from .sources import enumerate_candidates as _enum_raw

    raw = _enum_raw(req)
    kept: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    for cand in raw:
        cand = _enrich_candidate_via_pr_monitor(req, cand)
        if cand.source == "explicit":
            kept.append(cand)
            continue
        ok, reason = _passes_filter(cand, req.pr_filter)
        if not ok:
            skipped.append({"ref": cand.ref, "source": cand.source, "reason": reason})
            continue
        kept.append(cand)
    return kept, skipped


def _apply_prior_scores(req: ExploreRequest, candidates: list[Candidate]) -> list[Candidate]:
    """Attach KB prior scores and sort candidates before benchmarking.

    Args:
        req: The explore request carrying framework and gap context.
        candidates: Enumerated candidate list in source order.

    Returns:
        Candidates with ``prior_score`` set. Non-zero scores sort first;
        otherwise original source order is preserved.
    """
    if not candidates:
        return candidates
    ledger = read_pr_ledger()
    gap_keywords = req.keywords or tuple(extract_keywords(req.gap_description))
    scored: list[tuple[int, Candidate]] = []
    for index, cand in enumerate(candidates):
        enriched = replace(
            cand,
            framework=cand.framework or req.framework,
            model_class=cand.model_class or req.model_class,
            gpu_type=cand.gpu_type or req.gpu_type,
            precision=cand.precision or req.precision,
            gap_canonical_id=cand.gap_canonical_id or req.gap_canonical_id,
            gap_description=cand.gap_description or req.gap_description,
            gap_keywords=cand.gap_keywords or tuple(gap_keywords),
        )
        score = prior_score(
            enriched,
            gap_canonical_id=enriched.gap_canonical_id,
            gap_keywords=enriched.gap_keywords,
            ledger=ledger,
        )
        scored.append((index, replace(enriched, prior_score=score)))
    if not any(c.prior_score > 0.0 for _, c in scored):
        return [replace(c, prior_rank=i) for i, (_, c) in enumerate(scored, start=1)]
    scored.sort(key=lambda item: (-item[1].prior_score, item[0]))
    return [replace(c, prior_rank=i) for i, (_, c) in enumerate(scored, start=1)]


def _resolve_output_path(template: str, variables: dict[str, str]) -> Path:
    """Render a path template using the candidate's variable bag.

    Args:
        template (str): The path template with ``{var}`` placeholders.
        variables (dict[str, str]): The variable bag substituted into the
            template.

    Returns:
        Path: The rendered, user-expanded filesystem path.
    """
    return Path(render_template(template, variables)).expanduser()


def _prepare_candidate_workspace_with_artifacts(
    req: ExploreRequest,
    candidate: Candidate,
    *,
    index: int,
    execute: bool,
) -> tuple[WorkspacePaths, dict[str, str]]:
    """Prepare a candidate workspace and drop its audit artifacts.

    Drops audit material (``pr.patches`` / ``pr_files.json``) for the
    candidate regardless of execute mode, then delegates the worktree +
    venv step to :mod:`isolation` when ``execute`` is True.

    Args:
        req: The explore request.
        candidate: The candidate to prepare.
        index: Zero-based candidate index (used in the directory name).
        execute: Whether to materialize the worktree and venv.

    Returns:
        A ``(WorkspacePaths, artifact_paths)`` tuple.
    """
    candidate_dir = req.work_dir / "candidates" / f"{index:02d}_{candidate.slug}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_pr_artifacts(req, candidate, candidate_dir)
    workspace = prepare_candidate_workspace(
        req,
        candidate,
        index=index,
        execute=execute,
    )
    return workspace, artifact_paths


def _write_pr_artifacts(req: ExploreRequest, candidate: Candidate, candidate_dir: Path) -> dict[str, str]:
    """Write ``pr.patches`` + ``pr_files.json`` for a PR candidate.

    No-op when PR Monitor is unconfigured or the candidate is not a PR
    ref; hard-fails on network errors (the CLI converts these to exit
    code 2).

    Args:
        req: The explore request carrying the PR Monitor config and repo URL.
        candidate: The candidate whose artifacts to write.
        candidate_dir: Directory to write the artifacts into.

    Returns:
        A mapping of artifact names to written file paths (empty when the
        candidate is not a PR or PR Monitor is unconfigured).

    Raises:
        pr_monitor.PRMonitorError: If the repo URL is malformed.
    """
    if req.pr_monitor is None:
        return {}
    number = candidate.pr_number
    if number is None:
        return {}
    try:
        repo_slug = _repo_slug(req.repo_url)
    except ValueError as exc:
        raise pr_monitor.PRMonitorError(
            f"cannot drop PR artifacts for {candidate.ref}: bad repo_url={req.repo_url!r}: {exc}"
        ) from exc
    base_url = req.pr_monitor.base_url
    timeout_sec = req.pr_monitor.timeout_sec

    patches_text = pr_monitor.pr_patches(repo_slug, number, base_url=base_url, timeout_sec=timeout_sec)
    patches_path = candidate_dir / "pr.patches"
    patches_path.write_text(patches_text, encoding="utf-8")

    files_payload = pr_monitor.pr_files(repo_slug, number, base_url=base_url, timeout_sec=timeout_sec)
    files_json_path = candidate_dir / "pr_files.json"
    files_json_path.write_text(
        json.dumps(
            {
                "repo": repo_slug,
                "number": number,
                "ref": candidate.ref,
                "head_sha": candidate.head_sha,
                "files": files_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "patches_path": str(patches_path),
        "files_json_path": str(files_json_path),
    }


def _variables(
    req: ExploreRequest,
    candidate: Candidate,
    candidate_dir: Path,
    worktree_dir: Path,
    venv_dir: Path,
) -> dict[str, str]:
    """Build the variable bag passed to render_template for command specs.

    Args:
        req (ExploreRequest): The explore request (framework / repo / work dir).
        candidate (Candidate): The candidate (ref / repo).
        candidate_dir (Path): The candidate's working directory.
        worktree_dir (Path): The candidate's git worktree directory.
        venv_dir (Path): The candidate's virtualenv directory.

    Returns:
        dict[str, str]: The string-valued variable bag for template rendering.
    """
    return {
        "candidate_ref": candidate.ref,
        "candidate_repo": candidate.repo,
        "candidate_dir": str(candidate_dir),
        "worktree_dir": str(worktree_dir),
        "venv_dir": str(venv_dir),
        "venv_bin": str(venv_dir / "bin"),
        "framework": req.framework,
        "repo_url": req.repo_url,
        "work_dir": str(req.work_dir),
    }


def _evaluate_candidate(req: ExploreRequest, variables: dict[str, str]) -> tuple[float | None, float | None, str]:
    """Load post-run benchmark.json + accuracy.json and pull the metrics.

    Args:
        req (ExploreRequest): The explore request supplying output templates.
        variables (dict[str, str]): The candidate variable bag used to resolve
            the output paths.

    Returns:
        tuple[float | None, float | None, str]: ``(throughput, accuracy,
            completed)`` with ``None`` for any metric that is absent.
    """
    benchmark_template = req.outputs.get("benchmark_json", "{candidate_dir}/benchmark.json")
    accuracy_template = req.outputs.get("accuracy_json", "{candidate_dir}/accuracy.json")
    benchmark = read_json(_resolve_output_path(benchmark_template, variables), default={}, require_dict=True)
    accuracy = read_json(_resolve_output_path(accuracy_template, variables), default={}, require_dict=True)
    throughput = _first_float(*(benchmark.get(k) for k in ("throughput", "output_throughput")))
    acc = _first_float(*(accuracy.get(k) for k in ("accuracy", "gsm8k", "exact_match", "score")))
    completed = str(benchmark.get("completed") or benchmark.get("Completed") or "")
    return throughput, acc, completed


def _run_single_candidate(
    req: ExploreRequest,
    candidate: Candidate,
    *,
    index: int,
    execute: bool,
) -> CandidateResult:
    """Run a single candidate end-to-end (workspace + commands + decision).

    Synchronous helper used by both the serial and async paths; side-effect
    free beyond the workspace it owns. Concurrency safety: callers must
    ensure two candidates never share an ``index`` (slug collisions could
    overwrite material).

    Args:
        req: The explore request.
        candidate: The candidate to run.
        index: Unique candidate index (must be distinct across callers).
        execute: When False, plan only; when True, build and benchmark.

    Returns:
        The :class:`CandidateResult` for the candidate.
    """
    workspace, artifact_paths = _prepare_candidate_workspace_with_artifacts(
        req,
        candidate,
        index=index,
        execute=execute,
    )
    candidate_dir = workspace.candidate_dir
    variables = _variables(
        req,
        candidate,
        candidate_dir,
        workspace.worktree_dir,
        workspace.venv_dir,
    )
    if not execute:
        return CandidateResult(
            candidate=candidate,
            candidate_dir=str(candidate_dir),
            worktree_dir=str(workspace.worktree_dir),
            venv_dir=str(workspace.venv_dir),
            status="planned",
            reason="run with --execute to build and benchmark this candidate",
            patches_path=artifact_paths.get("patches_path", ""),
            files_json_path=artifact_paths.get("files_json_path", ""),
        )

    command_results = []
    status = "succeeded"
    reason = ""
    for name in ("build", "benchmark", "accuracy", "cleanup"):
        spec = req.commands.get(name)
        if spec is None:
            continue
        command = render_template(spec.command, variables, shell_quote=True)
        with stage_log(
            log,
            name,
            candidate=candidate.ref,
            timeout_sec=spec.timeout_sec,
        ) as ctx:
            result = run_command(
                name,
                command,
                cwd=candidate_dir,
                timeout_sec=spec.timeout_sec,
            )
            ctx["ok"] = bool(result.ok)
            ctx["returncode"] = int(result.returncode)
            ctx["timed_out"] = bool(result.timed_out)
        command_results.append(result)
        if spec.required and not result.ok:
            status = "failed"
            reason = f"{name} command failed"
            break

    throughput, accuracy, completed = _evaluate_candidate(req, variables)
    winner, gate_reason = winner_decision(req, throughput, accuracy, completed)
    if status == "succeeded":
        reason = gate_reason
    score = candidate_score(req, throughput, accuracy)
    log.info(
        "candidate %s: status=%s winner=%s score=%.4f reason=%s",
        candidate.ref,
        status,
        winner,
        score,
        reason,
    )
    return CandidateResult(
        candidate=candidate,
        candidate_dir=str(candidate_dir),
        worktree_dir=str(workspace.worktree_dir),
        venv_dir=str(workspace.venv_dir),
        status=status,
        throughput=throughput,
        accuracy=accuracy,
        completed=completed,
        winner=winner if status == "succeeded" else False,
        reason=reason,
        commands=command_results,
        patches_path=artifact_paths.get("patches_path", ""),
        files_json_path=artifact_paths.get("files_json_path", ""),
    )


async def _run_candidates_concurrent(
    req: ExploreRequest,
    candidates: list[Candidate],
    *,
    execute: bool,
) -> list[CandidateResult]:
    """Run candidates concurrently, bounded by a ``build_concurrency`` semaphore.

    Each task wraps :func:`_run_single_candidate` in
    :func:`asyncio.to_thread`; bench/accuracy stay inside the worker so two
    candidates only overlap when concurrency > 1 (the explicit user knob).

    Args:
        req: The explore request.
        candidates: Candidates to run.
        execute: Whether to build and benchmark (vs plan only).

    Returns:
        The per-candidate results in submission order.
    """
    semaphore = asyncio.Semaphore(max(1, req.build_concurrency))

    async def _bounded(idx: int, cand: Candidate) -> CandidateResult:
        """Run one candidate under the concurrency semaphore.

        Args:
            idx (int): The unique 1-based candidate index.
            cand (Candidate): The candidate to run.

        Returns:
            CandidateResult: The result of running the candidate in a thread.
        """
        async with semaphore:
            return await asyncio.to_thread(
                _run_single_candidate,
                req,
                cand,
                index=idx,
                execute=execute,
            )

    tasks = [_bounded(i, c) for i, c in enumerate(candidates, start=1)]
    return await asyncio.gather(*tasks)


def _maybe_disk_preflight(req: ExploreRequest, n_candidates: int, *, execute: bool) -> None:
    """Run disk_preflight when execute mode is on, ``disk_min_free_gb`` is not 0,
    and either it is set explicitly or ``prepare_candidate_env`` is True.

    Args:
        req (ExploreRequest): The explore request (work dir + threshold).
        n_candidates (int): The number of candidates to size the check by; a
            non-positive count returns early with nothing to preflight.
        execute (bool): Whether the run is in execute mode; preflight is
            skipped entirely in plan mode.
    """
    if not execute or n_candidates <= 0:
        return
    if req.disk_min_free_gb == 0:
        log.debug("disk_preflight: skipped (disk_min_free_gb=0)")
        return
    if req.disk_min_free_gb is None and not req.prepare_candidate_env:
        log.debug("disk_preflight: skipped (prepare_candidate_env=False, no explicit disk_min_free_gb)")
        return
    disk_preflight(
        req.work_dir,
        n_candidates,
        min_free_gb=req.disk_min_free_gb,
    )


def _cleanup_losers(
    req: ExploreRequest,
    results: list[CandidateResult],
    *,
    execute: bool,
) -> None:
    """Apply keep_winner_only cleanup over all non-winner results.

    Args:
        req (ExploreRequest): The explore request (supplies keep_winner_only).
        results (list[CandidateResult]): The per-candidate results to clean up.
        execute (bool): Whether the run is in execute mode; cleanup is skipped
            in plan mode.
    """
    if not execute or not req.keep_winner_only:
        return
    repo_dir: Path | None = None
    try:
        repo_dir = prepare_repo_cache(req)
    except Exception:  # noqa: BLE001 — repo cache may be unavailable in tests
        log.debug("cleanup_losers: repo cache unavailable; using rmtree fallback")
    for result in results:
        if result.status != "succeeded" and not result.winner:
            keep = False
        else:
            keep = result.winner
        cleanup_workspace(
            WorkspacePaths(
                candidate_dir=Path(result.candidate_dir),
                worktree_dir=Path(result.worktree_dir),
                venv_dir=Path(result.venv_dir),
            ),
            is_winner=keep,
            keep_winner_only=True,
            repo_dir=repo_dir,
        )


def explore(req: ExploreRequest, *, execute: bool = False) -> dict[str, Any]:
    """Main entry: enumerate, optionally build/bench, return summary dict.

    Behaviour matrix:

    * ``execute=False`` — plan mode; drops audit material, never builds.
    * ``execute=True`` + ``ranking_mode=False`` — run candidates serially
      (or via ``build_concurrency`` async gather), stop on first winner.
    * ``execute=True`` + ``ranking_mode=True`` — run every candidate, sort by
      :func:`candidate_score` descending; ``winner_ref`` is the top scorer
      that passed all gates.
    * ``keep_winner_only=True`` — after the run, drop worktree+venv for every
      non-winner candidate to reclaim disk.
    * ``build_concurrency > 1`` — fan out build via ``asyncio.gather``;
      bench/accuracy stay serial within a candidate task.

    Disk preflight runs first, but only when ``execute=True``,
    ``disk_min_free_gb != 0``, and either ``disk_min_free_gb`` is set
    explicitly or ``prepare_candidate_env`` is True; failure raises
    :class:`isolation.DiskPreflightError`.

    Args:
        req: The explore request driving the run.
        execute: When False, plan only; when True, build and benchmark.

    Returns:
        A summary dict describing the run, candidates, winner, and KB
        contribution.

    Raises:
        isolation.DiskPreflightError: If the disk preflight check fails.
    """
    log.info(
        "explore start framework=%s repo=%s work_dir=%s execute=%s "
        "ranking=%s build_concurrency=%d keep_winner_only=%s kb_domain=%r",
        req.framework,
        req.repo_url,
        req.work_dir,
        execute,
        req.ranking_mode,
        req.build_concurrency,
        req.keep_winner_only,
        req.kb_domain,
    )
    req.work_dir.mkdir(parents=True, exist_ok=True)

    with stage_log(log, "enumerate") as ctx:
        candidates, skipped_candidates = _enumerate_with_skipped(req)
        candidates = _apply_prior_scores(req, candidates)
        ctx["n_candidates"] = len(candidates)
        ctx["n_skipped"] = len(skipped_candidates)

    _maybe_disk_preflight(req, len(candidates), execute=execute)

    if execute and req.ranking_mode and req.build_concurrency > 1:
        log.info(
            "explore: ranking_mode + build_concurrency=%d -> asyncio.gather",
            req.build_concurrency,
        )
        results: list[CandidateResult] = asyncio.run(_run_candidates_concurrent(req, candidates, execute=execute))
    else:
        # Serial path; early-stops on first winner unless ranking_mode is on.
        results = []
        for index, candidate in enumerate(candidates, start=1):
            result = _run_single_candidate(
                req,
                candidate,
                index=index,
                execute=execute,
            )
            results.append(result)
            if execute and result.winner and not req.ranking_mode:
                log.info(
                    "explore: early-stop after winner %s (ranking_mode=False)",
                    candidate.ref,
                )
                break

    # Ranking mode only changes display order; the winner flag still comes
    # from the gate logic. Failed candidates score 0 and fall to the tail.
    if execute and req.ranking_mode:
        results.sort(
            key=lambda r: candidate_score(req, r.throughput, r.accuracy),
            reverse=True,
        )

    winner_result = next((r for r in results if r.winner), None)
    _cleanup_losers(req, results, execute=execute)

    audit_materials = {
        "patch_files_present": sum(1 for r in results if r.patches_path),
        "files_json_present": sum(1 for r in results if r.files_json_path),
        "policy": "patches_and_files_only",
    }
    kb_contribution = _contribute_findings_to_kb(req, winner_result, execute=execute)
    summary = {
        "ok": True,
        "mode": "execute" if execute else "plan",
        "framework": req.framework,
        "repo_url": req.repo_url,
        "work_dir": str(req.work_dir),
        "baseline": {
            "throughput": req.baseline.throughput,
            "accuracy": req.baseline.accuracy,
            "completed": req.baseline.completed,
        },
        "thresholds": {
            "min_throughput_ratio": req.thresholds.min_throughput_ratio,
            "max_accuracy_drop": req.thresholds.max_accuracy_drop,
        },
        "ranking_mode": req.ranking_mode,
        "prior_ranking": {
            "enabled": True,
            "ranked_candidates": sum(1 for c in candidates if c.prior_score > 0.0),
            "top_prior_score": max((c.prior_score for c in candidates), default=0.0),
        },
        "build_concurrency": req.build_concurrency,
        "keep_winner_only": req.keep_winner_only,
        "winner_ref": winner_result.candidate.ref if winner_result else None,
        "winner_dir": winner_result.candidate_dir if winner_result else None,
        "promotion_policy": "manual_only",
        "promotion_hint": (
            "No main-environment mutation was performed. Inspect winner_dir and promote manually."
            if winner_result
            else "No winner found."
        ),
        "pr_filter_applied": asdict(req.pr_filter),
        "skipped_candidates": skipped_candidates,
        "audit_materials": audit_materials,
        "kb_contribution": kb_contribution,
        "candidates": [r.to_dict() for r in results],
    }
    log.info(
        "explore done winner=%s n_results=%d kb=%s",
        summary["winner_ref"],
        len(results),
        kb_contribution.get("status"),
    )
    return summary


def _contribute_findings_to_kb(
    req: ExploreRequest,
    winner: CandidateResult | None,
    *,
    execute: bool,
) -> dict[str, object]:
    """Append a Finding to ``${KB}/<domain>/empirical_kb.md`` when warranted.

    Fires only when ``execute=True``, ``req.kb_domain`` is non-empty, and a
    ``winner`` exists. Best-effort: any KB write error is captured into the
    returned metadata dict so the explore summary stays usable even if the
    KB directory is read-only.

    Args:
        req: The explore request (supplies KB domain and baseline).
        winner: The winning candidate result, if any.
        execute: Whether the run was in execute mode.

    Returns:
        A ``kb_contribution`` metadata dict with a ``status`` field.
    """
    if not execute or not req.kb_domain or winner is None:
        return {"status": "skipped", "reason": "execute+kb_domain+winner required"}
    from .kb import contribute_to_kb, synthesize_findings

    metrics: dict[str, float] = {}
    if winner.throughput is not None:
        metrics["throughput"] = float(winner.throughput)
        metrics["baseline_throughput"] = float(req.baseline.throughput)
        if req.baseline.throughput > 0:
            metrics["throughput_ratio"] = winner.throughput / req.baseline.throughput
    if winner.accuracy is not None:
        metrics["accuracy"] = float(winner.accuracy)
    finding = Finding(
        title=f"{req.framework} winner {winner.candidate.ref}",
        body=f"Reason: {winner.reason or 'gates passed'}",
        source="fa explore --execute",
        session_id=Path(req.work_dir).name or "session",
        candidate_ref=winner.candidate.ref,
        metrics=metrics,
    )
    body = synthesize_findings(req.kb_domain, [finding], with_llm=False)
    try:
        path = contribute_to_kb(
            domain=req.kb_domain,
            finding=body,
            source=finding.source,
            session_id=finding.session_id,
        )
    except OSError as exc:  # noqa: BLE001 - any disk error must not fail explore
        return {"status": "failed", "reason": str(exc)}
    return {
        "status": "appended",
        "domain": req.kb_domain,
        "path": str(path),
        "finding_title": finding.title,
    }


__all__ = ["explore"]
