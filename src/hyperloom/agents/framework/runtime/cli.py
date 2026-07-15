# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Framework Agent CLI entry point.

Subcommands:

* ``fa schema``     - placeholder schema summary.
* ``fa candidates`` - enumerate PR/ref candidates from the configured
  sources (primus_cortex + github).
* ``fa explore``    - run the full exploration pipeline; defaults to
  ``--plan`` mode (drop audit material only); ``--execute`` adds
  worktree + venv + build/benchmark/accuracy commands.
* ``fa kb``         - knowledge-base operations: ``list``, ``show``,
  ``search``, ``contribute``, ``synthesize``. Defaults to pure-Python
  digest; ``synthesize --with-llm`` lazy-imports ``claude_agent_sdk``.
* ``fa phase-discover`` - Hyperloom FRAMEWORK_AGENT phase entry point.
  Reads a JSON ``--request`` and writes a JSON ``--out``
  (critic-agent style).
* ``fa phase-audit`` - static local-source judging of whether a candidate
  PR is already present in the framework source roots; optional
  opt-in single chat-completion refine. Reads/writes JSON like
  ``phase-discover``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.subprocess_bridge import RuntimeAdapterError as RuntimeAdapterError
from hyperloom.common.subprocess_bridge import emit_json

from ..pr_kb_slug import normalise_repo as _normalise_pr_kb_repo

if TYPE_CHECKING:
    from ..models import ExploreRequest


def _load_request(path: str) -> "ExploreRequest":
    """Load and parse a JSON request file into an ExploreRequest.

    Args:
        path (str): Path to the JSON request file.

    Returns:
        ExploreRequest: The parsed request.

    Raises:
        RuntimeAdapterError: If the file is missing, not valid JSON, or its root
            is not a JSON object.
    """
    from ..models import ExploreRequest

    req_path = Path(path).expanduser()
    if not req_path.exists():
        raise RuntimeAdapterError(f"request file not found: {req_path}")
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeAdapterError(f"request file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(f"request file root must be a JSON object, got {type(raw).__name__}")
    return ExploreRequest.from_dict(raw)


def _cmd_schema(args: argparse.Namespace) -> None:
    """Print the ExploreRequest schema summary.

    Args:
        args (argparse.Namespace): Parsed CLI args (unused).
    """
    del args
    emit_json(
        {
            "required": ["framework", "repo_url", "baseline"],
            "subcommands_available": [
                "schema",
                "candidates",
                "explore",
                "kb",
                "phase-discover",
                "phase-audit",
            ],
            "subcommands_planned": [],
            "search_modes_supported": ["gbrain_pr_kb", "primus_cortex", "github"],
            "modes": {
                "plan": "drop audit material only (pr.patches + pr_files.json)",
                "execute": "additionally create worktree+venv and run build/bench commands",
            },
            "promotion_policy": "manual_only",
            "kb_subcommands": ["list", "show", "search", "contribute", "synthesize"],
            "phase_audit": {
                "purpose": "static (default) local-source judging of whether a candidate PR is already present; optional opt-in LLM refine",
                "semantic_status_values": [
                    "already_equivalent",
                    "already_superset",
                    "partially_present",
                    "not_present",
                    "unknown",
                ],
                "applicability_values": [
                    "direct_apply",
                    "needs_rewrite",
                    "not_applicable",
                    "needs_human_review",
                ],
                "recommended_next_step_values": [
                    "skip",
                    "direct_framework",
                    "author_via_specialist",
                ],
                "patch_sources": ["diff_text", "patches_path", "primus_cortex"],
                "llm": "opt-in via request.use_llm; needs SAFE_API_KEY + OPENAI_BASE_URL; best-effort, evidence-gated",
            },
        },
        "-",
        make_parents=True,
    )


def _cmd_explore(args: argparse.Namespace) -> None:
    """Run the full exploration; plan by default, build/bench when --execute.

    Args:
        args (argparse.Namespace): Parsed CLI args with ``request``,
            ``execute``, and ``out``.
    """
    from ..explorer import explore

    request = _load_request(args.request)
    summary = explore(request, execute=bool(args.execute))
    emit_json(summary, args.out, make_parents=True)


def _cmd_candidates(args: argparse.Namespace) -> None:
    """Enumerate candidates per request.search_modes and emit JSON.

    Args:
        args (argparse.Namespace): Parsed CLI args with ``request`` and ``out``.
    """
    from ..sources import enumerate_candidates

    request = _load_request(args.request)
    candidates = enumerate_candidates(request)
    payload = {
        "framework": request.framework,
        "repo_url": request.repo_url,
        "search_modes": list(request.search_modes),
        "search_perf_prs": request.search_perf_prs,
        "max_search_candidates": request.max_search_candidates,
        "count": len(candidates),
        "candidates": [asdict(c) for c in candidates],
    }
    emit_json(payload, args.out, make_parents=True)


def _read_json_request(path: str) -> dict[str, Any]:
    """Load a JSON request file for the ``phase-*`` subcommands; enforces a
    dict at the top level since every ``phase-*`` request is an object.

    Args:
        path (str): Path to the JSON request file.

    Returns:
        dict[str, Any]: The decoded request object.

    Raises:
        RuntimeAdapterError: If the file is missing, not valid JSON, or its root
            is not a JSON object.
    """
    req_path = Path(path).expanduser()
    if not req_path.exists():
        raise RuntimeAdapterError(f"request file not found: {req_path}")
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeAdapterError(f"request file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(f"request file root must be a JSON object, got {type(raw).__name__}")
    return raw


def _pr_url_for(repo: str, pr_number: int | str) -> str:
    """Build a canonical GitHub PR URL from a repo slug and PR number.

    Args:
        repo (str): Repository slug in ``owner/name`` form.
        pr_number (int | str): PR number; only an ``int`` yields a URL.

    Returns:
        str: The PR URL, or an empty string when ``repo`` is empty or
            ``pr_number`` is not an int.
    """
    if repo and isinstance(pr_number, int):
        return f"https://github.com/{repo}/pull/{pr_number}"
    return ""


def _extract_pr_number(text: str) -> str:
    """Extract a bare PR number from a ``PR:1234`` ref or a ``.../pull/1234`` URL.

    Args:
        text: A candidate ref (``PR:<n>``) or a PR URL.

    Returns:
        The bare PR number as a string, or ``""`` when none is present.
    """
    import re as _re

    s = str(text or "").strip()
    if not s:
        return ""
    head, sep, tail = s.partition(":")
    if sep and head.strip().upper() == "PR":
        tail = tail.strip()
        if tail.isdigit():
            return tail
    m = _re.search(r"/pull/(\d+)", s)
    return m.group(1) if m else ""


def _candidate_excluded_by_memory(
    *,
    pr_url: str,
    ref: str,
    pr_number: int | str,
    excluded_ids: set[str],
    excluded_pr_numbers: set[str],
) -> bool:
    """True when a discovered candidate matches the session's exclusion memory.

    Drops a candidate whose ``pr_url`` / ``ref`` is an excluded id, or whose PR
    number matches an excluded / already-failed PR.

    Args:
        pr_url: The candidate's PR URL.
        ref: The candidate's ref (e.g. ``PR:1234``).
        pr_number: The candidate's parsed PR number (``int`` or ``""``).
        excluded_ids: pr_url / ref ids the caller already saw or finalised.
        excluded_pr_numbers: bare PR numbers derived from excluded ids +
            failed-candidate context.

    Returns:
        ``True`` when the candidate should be excluded from the batch.
    """
    if pr_url and pr_url in excluded_ids:
        return True
    if ref and ref in excluded_ids:
        return True
    cand_num = str(pr_number) if isinstance(pr_number, int) else _extract_pr_number(ref or pr_url)
    return bool(cand_num) and cand_num in excluded_pr_numbers


def _cmd_phase_discover(args: argparse.Namespace) -> None:
    """Discover one batch of PR candidates for the FRAMEWORK_AGENT phase.

    Request shape:
        {"model": str, "framework": str, "gpu_type": str,
         "gaps": [{"gap_canonical_id": str, "gap_description": str}, ...],
         "repo_url": str (optional), "work_dir": str (optional),
         "max_search_candidates": int (optional, default 5),
         "batch_id": str (optional; defaults to "batch-<uuid8>"),
         "excluded_candidate_ids": [str, ...] (optional; pr_url / ref /
             "PR:<n>" the caller already saw or finalised — hard-filtered),
         "failed_candidate_context": [{"ref", "status", "gain_pct", "why"},
             ...] (optional; same-PR numbers are de-prioritised to a drop)}

    Output shape:
        {"batch_id": str, "framework": str, "excluded_count": int,
         "candidates": [{"pr_url", "repo", "ref", "pr_number", "title",
                          "summary", "score", "prior_score", "diff_url",
                          "gap_canonical_id"}, ...]}

    Args:
        args (argparse.Namespace): Parsed CLI args with ``request`` and ``out``.

    Raises:
        RuntimeAdapterError: If no repo URL can be resolved for the framework.
    """
    import uuid as _uuid
    from dataclasses import asdict as _asdict

    from ..decision import prior_score
    from ..kb import read_pr_ledger
    from ..keywords import extract_keywords
    from ..models import ExploreRequest
    from ..sources import enumerate_candidates

    request = _read_json_request(args.request)
    framework = str(request.get("framework") or "sglang").strip().lower()
    repo_url = str(request.get("repo_url") or "").strip()
    if not repo_url:
        from ..repo_map import repo_url_for_framework

        repo_url = repo_url_for_framework(framework)
    if not repo_url:
        raise RuntimeAdapterError(f"phase-discover: no repo_url for framework={framework!r}")
    work_dir = str(request.get("work_dir") or "/tmp/framework-agent")
    max_candidates = int(request.get("max_search_candidates") or 5)
    batch_id = str(request.get("batch_id") or f"batch-{_uuid.uuid4().hex[:8]}")
    gaps = request.get("gaps") or []
    if not isinstance(gaps, list) or not gaps:
        gaps = [{"gap_canonical_id": "", "gap_description": ""}]

    # Forced cross-framework candidates: pinned upstream PRs surfaced as
    # source='explicit'. Scoped to repo_scope; supports request fields and
    # FRAMEWORK_AGENT_FORCE_PR_REFS / FRAMEWORK_AGENT_FORCE_PR_REPO env fallback.
    forced_refs_raw = request.get("forced_candidate_refs")
    if not forced_refs_raw:
        env_refs = (os.environ.get("FRAMEWORK_AGENT_FORCE_PR_REFS") or "").strip()
        forced_refs_raw = [r for r in env_refs.replace(";", ",").split(",")] if env_refs else []
    forced_repo_scope = str(
        request.get("forced_candidate_repo_scope")
        or os.environ.get("FRAMEWORK_AGENT_FORCE_PR_REPO")
        or ""
    ).strip().lower()

    def _norm_ref(r: str) -> str:
        r = str(r or "").strip()
        if not r:
            return ""
        if r.startswith("PR:"):
            return r
        # tolerate a full PR URL or bare number
        if r.isdigit():
            return f"PR:{r}"
        if "/pull/" in r:
            tail = r.rstrip("/").rsplit("/pull/", 1)[-1].split("/")[0]
            if tail.isdigit():
                return f"PR:{tail}"
        return r

    forced_refs: list[str] = []
    if forced_repo_scope and forced_repo_scope not in repo_url.strip().lower():
        # Queried repo out of scope for the forced refs; skip.
        forced_refs = []
    else:
        seen_fr: set[str] = set()
        for raw in forced_refs_raw or []:
            nr = _norm_ref(raw)
            if nr and nr not in seen_fr:
                seen_fr.add(nr)
                forced_refs.append(nr)

    # Hard-dedup: drop candidates already discovered/finalised, and collapse
    # "same PR number as a failed candidate" into a drop.
    excluded_ids: set[str] = {
        str(x).strip() for x in (request.get("excluded_candidate_ids") or []) if str(x).strip()
    }
    excluded_pr_numbers: set[str] = {n for n in (_extract_pr_number(x) for x in excluded_ids) if n}
    failed_ctx = request.get("failed_candidate_context") or []
    if isinstance(failed_ctx, list):
        for f in failed_ctx:
            if isinstance(f, dict):
                n = _extract_pr_number(str(f.get("ref") or ""))
                if n:
                    excluded_pr_numbers.add(n)
    excluded_count = 0

    # Enable the primus_cortex mode only when a URL is available; GitHub
    # (anonymous, best-effort) is always kept so discovery degrades gracefully.
    primus_url = str(request.get("primus_cortex_url") or os.environ.get("PRIMUS_CORTEX_PR_API") or "").strip()
    if primus_url:
        search_modes = ["primus_cortex", "github"]
        primus_block: dict[str, Any] = {"primus_cortex": {"base_url": primus_url}}
    else:
        search_modes = ["github"]
        primus_block = {}

    # gbrain PR KB is the primary discovery backend when enabled + configured;
    # prepend it so it ranks ahead of primus_cortex/github fallbacks.
    pr_kb_enabled = (os.environ.get("PR_KB_ENABLE", "1") or "1").strip() != "0"
    pr_kb_configured = bool(
        (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
        and (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    )
    if pr_kb_enabled and pr_kb_configured and "gbrain_pr_kb" not in search_modes:
        search_modes = ["gbrain_pr_kb", *search_modes]

    seen_refs: set[tuple[str, str]] = set()
    out_cands: list[dict[str, Any]] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        gap_id = str(gap.get("gap_canonical_id") or "")
        gap_desc = str(gap.get("gap_description") or "")
        raw_gap_keywords = gap.get("gap_keywords") or request.get("keywords") or []
        if isinstance(raw_gap_keywords, list):
            gap_keywords = [str(k).strip().lower() for k in raw_gap_keywords if str(k).strip()]
        else:
            gap_keywords = []
        if not gap_keywords and gap_desc.strip():
            gap_keywords = extract_keywords(gap_desc)
        req = ExploreRequest.from_dict(
            {
                "framework": framework,
                "repo_url": repo_url,
                "work_dir": work_dir,
                "baseline": {"throughput": 1.0},
                "gap_description": gap_desc,
                "gap_canonical_id": gap_id,
                "model_class": str(request.get("model_class") or request.get("model") or ""),
                "gpu_type": str(request.get("gpu_type") or ""),
                "precision": str(request.get("precision") or ""),
                # Must be True: else enumerate_candidates short-circuits to
                # explicit-refs-only and returns 0 candidates here.
                "search_perf_prs": True,
                "search_modes": search_modes,
                "pr_states": request.get("pr_states") or ["open"],
                "max_search_candidates": max_candidates,
                "keywords": gap_keywords,
                # Forced refs are emitted first (source='explicit'); empty by default.
                "candidate_refs": tuple(forced_refs),
                **primus_block,
            }
        )
        try:
            cands = enumerate_candidates(req)
        except Exception as exc:  # noqa: BLE001 — best-effort per gap
            print(
                f"WARN: phase-discover gap={gap_id!r} enumerate failed: {exc!r}",
                file=sys.stderr,
            )
            continue
        for cand in cands:
            entry = _asdict(cand)
            repo = str(entry.get("repo") or "")
            # Explicit forced candidates carry repo=repo_url (a .git URL).
            # Derive the owner/name slug for pr_url/diff_url but KEEP the repo
            # field as the canonical .git URL for cross-discover origin re-tagging.
            repo_slug = repo
            if str(entry.get("source") or "") == "explicit" and (
                repo.startswith("http") or repo.endswith(".git")
            ):
                slug = _normalise_pr_kb_repo(repo)
                if slug:
                    repo_slug = slug  # used only for pr_url/diff_url
            ref = str(entry.get("ref") or "")
            key = (repo, ref)
            if not ref or key in seen_refs:
                continue
            seen_refs.add(key)
            pr_number: int | str = ""
            if ref.startswith("PR:"):
                try:
                    pr_number = int(ref.split(":", 1)[1])
                except (ValueError, IndexError):
                    pr_number = ""
            html_url = str(entry.get("html_url") or "")
            diff_url = (
                f"{html_url}.diff"
                if html_url and isinstance(pr_number, int)
                else (f"https://github.com/{repo_slug}/pull/{pr_number}.diff" if repo_slug and isinstance(pr_number, int) else "")
            )
            pr_url = html_url or _pr_url_for(repo_slug, pr_number)
            # Drop candidates already seen/finalised or equivalent to a failed PR.
            if _candidate_excluded_by_memory(
                pr_url=pr_url,
                ref=ref,
                pr_number=pr_number,
                excluded_ids=excluded_ids,
                excluded_pr_numbers=excluded_pr_numbers,
            ):
                excluded_count += 1
                continue
            labels = entry.get("labels") or ()
            try:
                score = float(entry.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out_cands.append(
                {
                    "pr_url": pr_url,
                    "repo": repo,
                    "ref": ref,
                    "pr_number": pr_number,
                    "title": str(entry.get("title") or ""),
                    "summary": ", ".join(str(l) for l in labels) if labels else "",
                    "score": score,
                    "diff_url": diff_url,
                    "labels": [str(l) for l in labels] if labels else [],
                    "author": str(entry.get("author") or ""),
                    "framework": framework,
                    "model_class": str(request.get("model_class") or request.get("model") or ""),
                    "gpu_type": str(request.get("gpu_type") or ""),
                    "precision": str(request.get("precision") or ""),
                    "gap_canonical_id": gap_id,
                    "gap_description": gap_desc,
                    "gap_keywords": gap_keywords,
                    "changed_files": [str(f) for f in (entry.get("changed_files") or [])],
                }
            )
    ledger = read_pr_ledger()
    scored_cands: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in enumerate(out_cands):
        candidate["prior_score"] = prior_score(
            candidate,
            gap_canonical_id=str(candidate.get("gap_canonical_id") or ""),
            gap_keywords=candidate.get("gap_keywords") or [],
            ledger=ledger,
        )
        scored_cands.append((index, candidate))
    if any(float(c.get("prior_score") or 0.0) > 0.0 for _, c in scored_cands):
        scored_cands.sort(key=lambda item: (-float(item[1].get("prior_score") or 0.0), item[0]))
        out_cands = [c for _, c in scored_cands]
    for rank, candidate in enumerate(out_cands, start=1):
        candidate["prior_rank"] = rank

    emit_json(
        {
            "batch_id": batch_id,
            "framework": framework,
            "repo_url": repo_url,
            "model": str(request.get("model") or ""),
            "gpu_type": str(request.get("gpu_type") or ""),
            "candidate_count": len(out_cands),
            "excluded_count": excluded_count,
            "prior_ranking": {
                "enabled": bool(out_cands),
                "ledger_records": len(ledger),
                "ranked_candidates": sum(1 for c in out_cands if float(c.get("prior_score") or 0.0) > 0.0),
            },
            "candidates": out_cands,
        },
        args.out,
        make_parents=True,
    )


def _cmd_phase_audit(args: argparse.Namespace) -> None:
    """Statically audit whether a candidate PR is already present in local source.

    Request shape:
        {"candidate": {repo, pr_number, ref, diff_url, pr_url, ...},
         "framework": str, "framework_source_roots": [str, ...],
         "repo_url": str (optional), "work_dir": str (optional),
         "diff_text" | "patches_path" | "primus_cortex_url" (optional patch source),
         "use_llm": bool (optional, default false), "model": str (optional)}

    Output shape (``semantic_audit.json``):
        {"candidate_id", "semantic_status", "applicability", "confidence",
         "evidence": [...], "risks": [...], "recommended_next_step",
         "layer", "metrics", "ts"}

    Args:
        args (argparse.Namespace): Parsed CLI args with ``request`` and ``out``.
    """
    from ..audit import run_phase_audit

    request = _read_json_request(args.request)
    result = run_phase_audit(request)
    emit_json(result, args.out, make_parents=True)


def _cmd_kb(args: argparse.Namespace) -> None:
    """Dispatch ``fa kb <op>`` to the appropriate kb-module helper.

    Args:
        args (argparse.Namespace): Parsed CLI args carrying ``kb_op`` and the
            op-specific options (``domain``, ``query``, ``body``, etc.).

    Raises:
        RuntimeAdapterError: On an unknown op, a missing domain/body, or an
            invalid ``--findings`` file.
    """
    from .. import kb as kb_mod
    from ..models import Finding

    op = args.kb_op
    if op == "list":
        emit_json(
            {
                "kb_root": str(kb_mod._resolve_kb_root()),
                "domains": kb_mod.list_domains(),
            },
            args.out,
            make_parents=True,
        )
        return
    if op == "show":
        files = kb_mod.get_domain_files(args.domain)
        if not files:
            raise RuntimeAdapterError(f"domain {args.domain!r} not found under {kb_mod._resolve_kb_root()}")
        emit_json(
            {
                "domain": args.domain,
                "files": [{"path": str(p), "size_bytes": p.stat().st_size} for p in files if p.is_file()],
            },
            args.out,
            make_parents=True,
        )
        return
    if op == "search":
        hits = kb_mod.search_kb(args.query, domains=args.domain or None)
        emit_json(
            {
                "query": args.query,
                "domain_filter": list(args.domain) if args.domain else None,
                "count": len(hits),
                "hits": [{"domain": h.domain, "path": str(h.path)} for h in hits],
            },
            args.out,
            make_parents=True,
        )
        return
    if op == "contribute":
        if not args.body and not args.body_file:
            raise RuntimeAdapterError("fa kb contribute requires --body or --body-file")
        text = args.body or Path(args.body_file).read_text(encoding="utf-8")
        path = kb_mod.contribute_to_kb(
            domain=args.domain,
            finding=text,
            source=args.source,
            session_id=args.session_id,
        )
        emit_json({"status": "appended", "path": str(path)}, args.out, make_parents=True)
        return
    if op == "synthesize":
        findings: list[Finding] = []
        if args.findings:
            raw = json.loads(Path(args.findings).read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise RuntimeAdapterError("--findings file must contain a JSON array of Finding objects")
            for item in raw:
                if not isinstance(item, dict):
                    continue
                findings.append(
                    Finding(
                        title=str(item.get("title") or ""),
                        body=str(item.get("body") or ""),
                        source=str(item.get("source") or ""),
                        session_id=str(item.get("session_id") or ""),
                        candidate_ref=str(item.get("candidate_ref") or ""),
                        metrics={
                            str(k): float(v)
                            for k, v in (item.get("metrics") or {}).items()
                            if isinstance(v, (int, float))
                        },
                    )
                )
        markdown = kb_mod.synthesize_findings(
            args.domain,
            findings,
            with_llm=bool(args.with_llm),
            model=args.model,
        )
        if args.out and args.out != "-":
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(markdown, encoding="utf-8")
        else:
            sys.stdout.write(markdown)
            if not markdown.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        return
    raise RuntimeAdapterError(f"unknown kb op: {op!r}")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for framework-agent CLI.

    Returns:
        argparse.ArgumentParser: The configured parser with all subcommands and
            global logging flags registered.
    """
    parser = argparse.ArgumentParser(
        prog="framework-agent",
        description="Explore serving frameworks/refs in isolated worktrees.",
    )
    # Global logging flags on the top-level parser so every subcommand picks them up.
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "Override log level (DEBUG/INFO/WARNING/ERROR). "
            "Env fallback: FRAMEWORK_EXPLORER_LOG_LEVEL or "
            "FRAMEWORK_AGENT_LOG_LEVEL. Default INFO."
        ),
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=False,
        help=("Emit one JSON object per record (machine-friendly). Env fallback: FRAMEWORK_AGENT_LOG_JSON=1."),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=("Append log records to this path in addition to stderr. Env fallback: FRAMEWORK_AGENT_LOG_FILE."),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    schema_p = sub.add_parser("schema", help="Print the request schema summary")
    schema_p.set_defaults(func=_cmd_schema)

    cand_p = sub.add_parser(
        "candidates",
        help="Enumerate PR/ref candidates per request.search_modes (no build/bench)",
    )
    cand_p.add_argument(
        "--request",
        required=True,
        help="Path to a JSON ExploreRequest file",
    )
    cand_p.add_argument(
        "--out",
        default="-",
        help="Output path (default '-' = stdout)",
    )
    cand_p.set_defaults(func=_cmd_candidates)

    explore_p = sub.add_parser(
        "explore",
        help="Run the exploration pipeline (plan by default; --execute to build/bench)",
    )
    explore_p.add_argument(
        "--request",
        required=True,
        help="Path to a JSON ExploreRequest file",
    )
    explore_p.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Run build/benchmark commands (default: plan-only with audit material)",
    )
    explore_p.add_argument(
        "--out",
        default="-",
        help="Output path (default '-' = stdout)",
    )
    explore_p.set_defaults(func=_cmd_explore)

    # ----- Hyperloom FRAMEWORK_AGENT phase entry points -----
    pd_p = sub.add_parser(
        "phase-discover",
        help="Discover one batch of PR candidates for the FRAMEWORK_AGENT phase",
    )
    pd_p.add_argument("--request", required=True, help="JSON request file path")
    pd_p.add_argument("--out", default="-", help="Output path (default stdout)")
    pd_p.set_defaults(func=_cmd_phase_discover)

    pa_p = sub.add_parser(
        "phase-audit",
        help="Statically audit whether a candidate PR is already present in local source",
    )
    pa_p.add_argument("--request", required=True, help="JSON request file path")
    pa_p.add_argument("--out", default="-", help="Output path (default stdout)")
    pa_p.set_defaults(func=_cmd_phase_audit)

    kb_p = sub.add_parser(
        "kb",
        help="Knowledge-base operations (list / show / search / contribute / synthesize)",
    )
    kb_sub = kb_p.add_subparsers(dest="kb_op", required=True)

    kb_list_p = kb_sub.add_parser("list", help="List available KB domains")
    kb_list_p.add_argument("--out", default="-", help="Output path (default stdout)")
    kb_list_p.set_defaults(func=_cmd_kb)

    kb_show_p = kb_sub.add_parser("show", help="Show files within a KB domain")
    kb_show_p.add_argument("--domain", required=True)
    kb_show_p.add_argument("--out", default="-", help="Output path (default stdout)")
    kb_show_p.set_defaults(func=_cmd_kb)

    kb_search_p = kb_sub.add_parser("search", help="Search KB content (case-insensitive)")
    kb_search_p.add_argument("--query", required=True)
    kb_search_p.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Restrict search to this domain (repeatable)",
    )
    kb_search_p.add_argument("--out", default="-", help="Output path (default stdout)")
    kb_search_p.set_defaults(func=_cmd_kb)

    kb_contrib_p = kb_sub.add_parser(
        "contribute",
        help="Append a finding to ${KB}/<domain>/empirical_kb.md",
    )
    kb_contrib_p.add_argument("--domain", required=True)
    kb_contrib_p.add_argument("--body", default="", help="Finding markdown body")
    kb_contrib_p.add_argument("--body-file", default="", help="Read finding body from this file")
    kb_contrib_p.add_argument("--source", default="manual")
    kb_contrib_p.add_argument("--session-id", default="manual")
    kb_contrib_p.add_argument("--out", default="-", help="Output path (default stdout)")
    kb_contrib_p.set_defaults(func=_cmd_kb)

    kb_syn_p = kb_sub.add_parser(
        "synthesize",
        help="Synthesise a markdown digest from a list of Finding records",
    )
    kb_syn_p.add_argument("--domain", required=True)
    kb_syn_p.add_argument(
        "--findings",
        default="",
        help="JSON file containing a list of Finding objects (optional; empty -> empty digest)",
    )
    kb_syn_p.add_argument(
        "--with-llm",
        action="store_true",
        default=False,
        help="Route through claude_agent_sdk (lazy-imported); default is pure-Python",
    )
    kb_syn_p.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="LLM model identifier (only used with --with-llm)",
    )
    kb_syn_p.add_argument("--out", default="-", help="Output path (default stdout)")
    kb_syn_p.set_defaults(func=_cmd_kb)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point invoked by both `framework-agent` and `fa` scripts.

    Args:
        argv (list[str] | None): Argument vector to parse; ``None`` uses
            ``sys.argv``.

    Returns:
        int: Process exit code: ``0`` on success, ``2`` on a handled or
            unexpected error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    from ..logging_setup import configure_logging, get_logger

    configure_logging(
        level=args.log_level,
        json_output=args.log_json or None,
        log_file=args.log_file,
    )
    log = get_logger("cli")
    log.debug("fa cli start cmd=%s argv=%r", args.cmd, argv)
    try:
        args.func(args)
    except RuntimeAdapterError as exc:
        log.error("RuntimeAdapterError: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        log.exception("unexpected framework-agent failure")
        print(f"ERROR: unexpected framework-agent failure: {exc}", file=sys.stderr)
        return 2
    log.debug("fa cli done cmd=%s", args.cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
