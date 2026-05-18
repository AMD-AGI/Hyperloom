"""PR/ref exploration engine.

Adapted from zhenggong/framework-agent with two structural changes:

* Imports are rerouted to ``sources.primus_cortex`` / ``sources.github`` /
  ``sources._shared`` (Phase A reorg).
* Candidate enumeration goes through ``sources.enumerate_candidates``
  to honour ``ExploreRequest.search_modes``; enrichment + filtering
  still happen in this module.

Two run modes:

* ``execute=False`` (plan)    - drop ``pr.patches`` + ``pr_files.json``
  per PR candidate under ``candidate_dir`` and produce a planned
  ``explore_summary.json``. **No** worktree / venv / build / bench.
* ``execute=True``            - additionally create a detached git
  worktree + per-candidate venv, then run the request's ``build`` /
  ``benchmark`` / ``accuracy`` / ``cleanup`` commands. Promotion stays
  manual (``promotion_policy=manual_only``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .models import Candidate, CandidateResult, ExploreRequest, Finding, PrFilter
from .shell import render_template, run_command
from .sources import primus_cortex
from .sources._shared import _repo_slug


def _coalesce_str(*values: Any) -> str:
    """Return the first non-empty stripped string in ``values``; else ''."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summary_of(detail: dict[str, Any]) -> dict[str, Any]:
    """primus-cortex wraps PR metadata under 'summary'; default to {} otherwise."""
    summary = detail.get("summary")
    return summary if isinstance(summary, dict) else {}


def _extract_head_sha(detail: dict[str, Any]) -> str:
    """Pull head SHA out of any of the known keys in a PR detail payload."""
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
    """Pull labels out of a PR detail payload (handles dict and string items)."""
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
    """Pull author login from any known key in a PR detail payload."""
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
    """Pull title from a PR detail payload."""
    summary = _summary_of(detail)
    return _coalesce_str(summary.get("title"), detail.get("title"))


def _extract_updated_at(detail: dict[str, Any]) -> str:
    """Pull updated_at timestamp string."""
    summary = _summary_of(detail)
    return _coalesce_str(
        summary.get("pr_updated_at"),
        summary.get("updated_at"),
        detail.get("pr_updated_at"),
        detail.get("updated_at"),
        detail.get("updated"),
    )


def _extract_html_url(detail: dict[str, Any]) -> str:
    """Pull html_url from a PR detail payload."""
    summary = _summary_of(detail)
    return _coalesce_str(summary.get("html_url"), detail.get("html_url"), detail.get("url"))


def _extract_changed_files(
    detail: dict[str, Any], files_payload: list[dict[str, Any]]
) -> tuple[str, ...]:
    """Pull changed-files list, preferring the dedicated files endpoint payload."""
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


def _enrich_candidate_via_primus(req: ExploreRequest, candidate: Candidate) -> Candidate:
    """Fetch pr_get + pr_files for PR-typed candidates; hard-fails on primus errors.

    Branch / tag / commit refs are returned unchanged. Explicit refs that
    look like ``PR:N`` are also enriched so the audit artifact dump can
    target them.
    """
    if req.primus_cortex is None:
        return candidate
    number = candidate.pr_number
    if number is None:
        return candidate
    try:
        repo_slug = _repo_slug(req.repo_url)
    except ValueError as exc:
        raise primus_cortex.PrimusCortexError(
            f"cannot enrich {candidate.ref}: bad repo_url={req.repo_url!r}: {exc}"
        ) from exc
    base_url = req.primus_cortex.base_url
    timeout_sec = req.primus_cortex.timeout_sec
    detail = primus_cortex.pr_get(
        repo_slug, number, base_url=base_url, timeout_sec=timeout_sec
    )
    try:
        files_payload = primus_cortex.pr_files(
            repo_slug, number, base_url=base_url, timeout_sec=timeout_sec
        )
    except primus_cortex.PrimusCortexError:
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
    """Apply :class:`PrFilter` to a single candidate.

    Returns ``(True, '')`` on empty filter or pass; ``(False, reason)``
    otherwise. Path/label/author/date constraints that need metadata fail
    when that metadata is missing (typically because enrichment skipped).
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
            return False, "no changed_files metadata (primus enrichment likely skipped)"

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
    """Return ``(kept, skipped)`` after enrichment + filtering.

    Uses :mod:`framework_agent.sources` to do the multi-source union
    (primus_cortex + github + explicit), then enriches PR-typed
    candidates via primus and applies ``req.pr_filter``. Explicit
    candidates bypass the filter (operator intent wins) but are still
    enriched.
    """
    from .sources import enumerate_candidates as _enum_raw

    raw = _enum_raw(req)
    kept: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    for cand in raw:
        cand = _enrich_candidate_via_primus(req, cand)
        if cand.source == "explicit":
            kept.append(cand)
            continue
        ok, reason = _passes_filter(cand, req.pr_filter)
        if not ok:
            skipped.append({"ref": cand.ref, "source": cand.source, "reason": reason})
            continue
        kept.append(cand)
    return kept, skipped


def _load_json(path: Path) -> dict[str, Any]:
    """Best-effort JSON loader; returns {} when missing or invalid."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/bad metric files don't kill the run
        return {}
    return data if isinstance(data, dict) else {}


def _metric_float(data: dict[str, Any], *keys: str) -> float | None:
    """Return the first int/float value among ``keys`` in ``data``."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _resolve_output_path(template: str, variables: dict[str, str]) -> Path:
    """Render a path template using the candidate's variable bag."""
    return Path(render_template(template, variables)).expanduser()


def _repo_cache_dir(req: ExploreRequest) -> Path:
    """Stable per-repo cache directory under work_dir/_repos."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in req.repo_url.lower()).strip("-")
    return req.work_dir / "_repos" / (safe or "repo")


def _run_subprocess(
    args: list[str], *, cwd: Path | None = None, timeout_sec: int = 1800
) -> None:
    """Run a subprocess with a timeout; raise CalledProcessError on non-zero."""
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True, timeout=timeout_sec)


def _run_git(args: list[str], *, cwd: Path | None = None, timeout_sec: int = 1800) -> None:
    """Run a git command with a timeout; thin wrapper over :func:`_run_subprocess`."""
    _run_subprocess(args, cwd=cwd, timeout_sec=timeout_sec)


def _ensure_repo_cache(req: ExploreRequest) -> Path:
    """Mirror-clone the repo into the cache dir; fetch when already present."""
    repo_dir = _repo_cache_dir(req)
    if repo_dir.exists():
        _run_git(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["git", "clone", "--mirror", req.repo_url, str(repo_dir)])
    return repo_dir


def _worktree_ref(candidate: Candidate) -> str:
    """Choose the ref to materialise in a detached worktree."""
    if candidate.head_sha:
        return candidate.head_sha
    if candidate.ref.startswith("PR:"):
        number = candidate.ref.split(":", 1)[1]
        return f"refs/pull/{number}/head"
    return candidate.ref


def _fetch_candidate_ref(repo_dir: Path, candidate: Candidate) -> None:
    """Pre-fetch the candidate's ref into the cache mirror."""
    if candidate.head_sha:
        _run_git(["git", "fetch", "origin", candidate.head_sha], cwd=repo_dir)
        return
    if not candidate.ref.startswith("PR:"):
        return
    number = candidate.ref.split(":", 1)[1]
    _run_git(
        [
            "git",
            "fetch",
            "origin",
            f"refs/pull/{number}/head:refs/pull/{number}/head",
        ],
        cwd=repo_dir,
    )


def _prepare_candidate_workspace(
    req: ExploreRequest,
    candidate: Candidate,
    *,
    index: int,
    execute: bool,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Create candidate_dir, drop audit artifacts; build worktree+venv when execute."""
    candidate_dir = req.work_dir / "candidates" / f"{index:02d}_{candidate.slug}"
    worktree_dir = candidate_dir / "worktree"
    venv_dir = candidate_dir / "venv"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_pr_artifacts(req, candidate, candidate_dir)
    if not execute or not req.prepare_candidate_env:
        return candidate_dir, worktree_dir, venv_dir, artifact_paths

    repo_dir = _ensure_repo_cache(req)
    _fetch_candidate_ref(repo_dir, candidate)
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir)
    _run_git(
        [
            "git",
            "--git-dir",
            str(repo_dir),
            "worktree",
            "add",
            "--detach",
            str(worktree_dir),
            _worktree_ref(candidate),
        ]
    )
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    _run_subprocess(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        timeout_sec=600,
    )
    return candidate_dir, worktree_dir, venv_dir, artifact_paths


def _write_pr_artifacts(
    req: ExploreRequest, candidate: Candidate, candidate_dir: Path
) -> dict[str, str]:
    """Drop ``pr.patches`` + ``pr_files.json`` per PR candidate.

    No-op (returns empty dict) when primus_cortex is unconfigured or when
    the candidate is not a PR ref. Hard-fails on network errors via the
    primus_cortex backend's policy; the CLI's outer ``except Exception``
    converts this to exit code 2 with a clear message.
    """
    if req.primus_cortex is None:
        return {}
    number = candidate.pr_number
    if number is None:
        return {}
    try:
        repo_slug = _repo_slug(req.repo_url)
    except ValueError as exc:
        raise primus_cortex.PrimusCortexError(
            f"cannot drop PR artifacts for {candidate.ref}: bad repo_url={req.repo_url!r}: {exc}"
        ) from exc
    base_url = req.primus_cortex.base_url
    timeout_sec = req.primus_cortex.timeout_sec

    patches_text = primus_cortex.pr_patches(
        repo_slug, number, base_url=base_url, timeout_sec=timeout_sec
    )
    patches_path = candidate_dir / "pr.patches"
    patches_path.write_text(patches_text, encoding="utf-8")

    files_payload = primus_cortex.pr_files(
        repo_slug, number, base_url=base_url, timeout_sec=timeout_sec
    )
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
    """Build the variable bag passed to render_template for command specs."""
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


def _evaluate_candidate(
    req: ExploreRequest, variables: dict[str, str]
) -> tuple[float | None, float | None, str]:
    """Load post-run benchmark.json + accuracy.json and pull the metrics."""
    benchmark_template = req.outputs.get("benchmark_json", "{candidate_dir}/benchmark.json")
    accuracy_template = req.outputs.get("accuracy_json", "{candidate_dir}/accuracy.json")
    benchmark = _load_json(_resolve_output_path(benchmark_template, variables))
    accuracy = _load_json(_resolve_output_path(accuracy_template, variables))
    throughput = _metric_float(benchmark, "throughput", "output_throughput", "tput")
    acc = _metric_float(accuracy, "accuracy", "gsm8k", "exact_match", "score")
    completed = str(benchmark.get("completed") or benchmark.get("Completed") or "")
    return throughput, acc, completed


def _winner_decision(
    req: ExploreRequest,
    throughput: float | None,
    accuracy: float | None,
    completed: str,
) -> tuple[bool, str]:
    """Apply throughput/accuracy/completed gates; return (winner, reason)."""
    if throughput is None or throughput <= 0:
        return False, "missing throughput"
    ratio = throughput / req.baseline.throughput
    if ratio < req.thresholds.min_throughput_ratio:
        return (
            False,
            f"throughput ratio {ratio:.4f} below required {req.thresholds.min_throughput_ratio:.4f}",
        )
    if req.baseline.accuracy is not None:
        if accuracy is None:
            return False, "missing accuracy while baseline accuracy is set"
        drop = req.baseline.accuracy - accuracy
        if drop > req.thresholds.max_accuracy_drop:
            return (
                False,
                f"accuracy drop {drop:.4f} exceeds max {req.thresholds.max_accuracy_drop:.4f}",
            )
    if completed and "/" in completed:
        left, _, right = completed.partition("/")
        if left.strip() != right.strip():
            return False, f"benchmark completed={completed} is incomplete"
    return True, "throughput and accuracy gates passed"


def explore(req: ExploreRequest, *, execute: bool = False) -> dict[str, Any]:
    """Main entry: enumerate, optionally build/bench, return summary dict."""
    req.work_dir.mkdir(parents=True, exist_ok=True)
    candidates, skipped_candidates = _enumerate_with_skipped(req)
    results: list[CandidateResult] = []
    for index, candidate in enumerate(candidates, start=1):
        (
            candidate_dir,
            worktree_dir,
            venv_dir,
            artifact_paths,
        ) = _prepare_candidate_workspace(
            req, candidate, index=index, execute=execute
        )
        variables = _variables(req, candidate, candidate_dir, worktree_dir, venv_dir)
        if not execute:
            results.append(
                CandidateResult(
                    candidate=candidate,
                    candidate_dir=str(candidate_dir),
                    worktree_dir=str(worktree_dir),
                    venv_dir=str(venv_dir),
                    status="planned",
                    reason="run with --execute to build and benchmark this candidate",
                    patches_path=artifact_paths.get("patches_path", ""),
                    files_json_path=artifact_paths.get("files_json_path", ""),
                )
            )
            continue

        command_results = []
        status = "succeeded"
        reason = ""
        for name in ("build", "benchmark", "accuracy", "cleanup"):
            spec = req.commands.get(name)
            if spec is None:
                continue
            command = render_template(spec.command, variables, shell_quote=True)
            result = run_command(
                name,
                command,
                cwd=candidate_dir,
                timeout_sec=spec.timeout_sec,
            )
            command_results.append(result)
            if spec.required and not result.ok:
                status = "failed"
                reason = f"{name} command failed"
                break
        throughput, accuracy, completed = _evaluate_candidate(req, variables)
        winner, gate_reason = _winner_decision(req, throughput, accuracy, completed)
        if status == "succeeded":
            reason = gate_reason
        results.append(
            CandidateResult(
                candidate=candidate,
                candidate_dir=str(candidate_dir),
                worktree_dir=str(worktree_dir),
                venv_dir=str(venv_dir),
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
        )
        if winner:
            break
    winner_result = next((r for r in results if r.winner), None)
    audit_materials = {
        "patch_files_present": sum(1 for r in results if r.patches_path),
        "files_json_present": sum(1 for r in results if r.files_json_path),
        "policy": "patches_and_files_only",
    }
    kb_contribution = _contribute_findings_to_kb(req, winner_result, execute=execute)
    return {
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
        "winner_ref": winner_result.candidate.ref if winner_result else None,
        "winner_dir": winner_result.candidate_dir if winner_result else None,
        "promotion_policy": "manual_only",
        "promotion_hint": (
            "No main-environment mutation was performed. "
            "Inspect winner_dir and promote manually."
            if winner_result
            else "No winner found."
        ),
        "pr_filter_applied": asdict(req.pr_filter),
        "skipped_candidates": skipped_candidates,
        "audit_materials": audit_materials,
        "kb_contribution": kb_contribution,
        "candidates": [r.to_dict() for r in results],
    }


def _contribute_findings_to_kb(
    req: ExploreRequest,
    winner: CandidateResult | None,
    *,
    execute: bool,
) -> dict[str, object]:
    """Append a Finding to ``${KB}/<domain>/empirical_kb.md`` when warranted.

    Hook fires only when all of the following are true:

    * ``execute=True`` (plan mode never writes anything outside work_dir);
    * ``req.kb_domain`` is non-empty (explicit opt-in);
    * a ``winner`` candidate exists.

    The hook is best-effort - any KB write error is captured into the
    returned dict so the explore summary stays usable even if the KB
    directory is read-only. Returns a metadata dict that gets folded
    into ``explore_summary.json`` under the ``kb_contribution`` key.
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
