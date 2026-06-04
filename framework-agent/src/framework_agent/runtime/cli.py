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
* ``fa phase-discover`` - Hyperloom FRAMEWORK_PR phase entry point.
  Reads a JSON ``--request`` and writes a JSON ``--out``
  (critic-agent style).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


class RuntimeAdapterError(RuntimeError):
    """Raised when CLI input or runtime adaptation fails."""


def _emit_json(obj: Any, out: str | None) -> None:
    """Serialize obj as JSON to stdout or a file path."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _load_request(path: str) -> "ExploreRequest":
    """Load and parse a JSON request file into an ExploreRequest."""
    from ..models import ExploreRequest

    req_path = Path(path).expanduser()
    if not req_path.exists():
        raise RuntimeAdapterError(f"request file not found: {req_path}")
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeAdapterError(f"request file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(
            f"request file root must be a JSON object, got {type(raw).__name__}"
        )
    return ExploreRequest.from_dict(raw)


def _cmd_schema(args: argparse.Namespace) -> None:
    """Print the ExploreRequest schema summary."""
    del args
    _emit_json(
        {
            "required": ["framework", "repo_url", "baseline"],
            "subcommands_available": [
                "schema", "candidates", "explore", "kb",
                "phase-discover", "phase-fetch", "phase-emit-proposal",
            ],
            "subcommands_planned": [],
            "search_modes_supported": ["primus_cortex", "github"],
            "modes": {
                "plan": "drop audit material only (pr.patches + pr_files.json)",
                "execute": "additionally create worktree+venv and run build/bench commands",
            },
            "promotion_policy": "manual_only",
            "kb_subcommands": ["list", "show", "search", "contribute", "synthesize"],
        },
        "-",
    )


def _cmd_explore(args: argparse.Namespace) -> None:
    """Run the full exploration; plan by default, build/bench when --execute."""
    from ..explorer import explore

    request = _load_request(args.request)
    summary = explore(request, execute=bool(args.execute))
    _emit_json(summary, args.out)


def _cmd_candidates(args: argparse.Namespace) -> None:
    """Enumerate candidates per request.search_modes and emit JSON."""
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
    _emit_json(payload, args.out)


def _read_json_request(path: str) -> dict[str, Any]:
    """Load a JSON request file for the ``phase-*`` subcommands.

    Mirrors :func:`critic-agent.runtime.cli._read_json` but enforces a
    dict at the top level since every ``phase-*`` request is an object.
    """
    req_path = Path(path).expanduser()
    if not req_path.exists():
        raise RuntimeAdapterError(f"request file not found: {req_path}")
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeAdapterError(f"request file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(
            f"request file root must be a JSON object, got {type(raw).__name__}"
        )
    return raw


def _pr_url_for(repo: str, pr_number: int | str) -> str:
    if repo and isinstance(pr_number, int):
        return f"https://github.com/{repo}/pull/{pr_number}"
    return ""


def _cmd_phase_discover(args: argparse.Namespace) -> None:
    """Discover one batch of PR candidates for the FRAMEWORK_PR phase.

    Request shape:
        {"model": str, "framework": str, "gpu_type": str,
         "gaps": [{"gap_canonical_id": str, "gap_description": str}, ...],
         "repo_url": str (optional), "work_dir": str (optional),
         "max_search_candidates": int (optional, default 5),
         "batch_id": str (optional; defaults to "batch-<uuid8>")}

    Output shape:
        {"batch_id": str, "framework": str,
         "candidates": [{"pr_url", "repo", "ref", "pr_number", "title",
                          "summary", "score", "diff_url",
                          "gap_canonical_id"}, ...]}
    """
    import uuid as _uuid
    from dataclasses import asdict as _asdict

    from ..models import ExploreRequest
    from ..sources import enumerate_candidates

    request = _read_json_request(args.request)
    framework = str(request.get("framework") or "sglang").strip().lower()
    repo_url = str(request.get("repo_url") or "").strip()
    if not repo_url:
        # Fall back to the standalone repo_map (no reverse-import of
        # inference_optimizer; framework-agent is a standalone package).
        from framework_agent.repo_map import repo_url_for_framework
        repo_url = repo_url_for_framework(framework)
    if not repo_url:
        raise RuntimeAdapterError(
            f"phase-discover: no repo_url for framework={framework!r}"
        )
    work_dir = str(request.get("work_dir") or "/tmp/framework-agent")
    max_candidates = int(request.get("max_search_candidates") or 5)
    batch_id = str(request.get("batch_id") or f"batch-{_uuid.uuid4().hex[:8]}")
    gaps = request.get("gaps") or []
    if not isinstance(gaps, list) or not gaps:
        gaps = [{"gap_canonical_id": "", "gap_description": ""}]

    seen_refs: set[tuple[str, str]] = set()
    out_cands: list[dict[str, Any]] = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        gap_id = str(gap.get("gap_canonical_id") or "")
        gap_desc = str(gap.get("gap_description") or "")
        req = ExploreRequest.from_dict({
            "framework": framework,
            "repo_url": repo_url,
            "work_dir": work_dir,
            "baseline": {"throughput": 1.0},
            "gap_description": gap_desc,
            "search_modes": ["primus_cortex", "github"],
            "max_search_candidates": max_candidates,
        })
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
                f"{html_url}.diff" if html_url and isinstance(pr_number, int)
                else (
                    f"https://github.com/{repo}/pull/{pr_number}.diff"
                    if repo and isinstance(pr_number, int) else ""
                )
            )
            pr_url = html_url or _pr_url_for(repo, pr_number)
            labels = entry.get("labels") or ()
            try:
                score = float(entry.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out_cands.append({
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
                "gap_canonical_id": gap_id,
            })
    _emit_json(
        {
            "batch_id": batch_id,
            "framework": framework,
            "repo_url": repo_url,
            "model": str(request.get("model") or ""),
            "gpu_type": str(request.get("gpu_type") or ""),
            "candidate_count": len(out_cands),
            "candidates": out_cands,
        },
        args.out,
    )


def _cmd_kb(args: argparse.Namespace) -> None:
    """Dispatch ``fa kb <op>`` to the appropriate kb-module helper."""
    from .. import kb as kb_mod
    from ..models import Finding

    op = args.kb_op
    if op == "list":
        _emit_json(
            {
                "kb_root": str(kb_mod._resolve_kb_root()),
                "domains": kb_mod.list_domains(),
            },
            args.out,
        )
        return
    if op == "show":
        files = kb_mod.get_domain_files(args.domain)
        if not files:
            raise RuntimeAdapterError(
                f"domain {args.domain!r} not found under {kb_mod._resolve_kb_root()}"
            )
        _emit_json(
            {
                "domain": args.domain,
                "files": [
                    {"path": str(p), "size_bytes": p.stat().st_size}
                    for p in files
                    if p.is_file()
                ],
            },
            args.out,
        )
        return
    if op == "search":
        hits = kb_mod.search_kb(args.query, domains=args.domain or None)
        _emit_json(
            {
                "query": args.query,
                "domain_filter": list(args.domain) if args.domain else None,
                "count": len(hits),
                "hits": [
                    {"domain": h.domain, "path": str(h.path)} for h in hits
                ],
            },
            args.out,
        )
        return
    if op == "contribute":
        if not args.body and not args.body_file:
            raise RuntimeAdapterError(
                "fa kb contribute requires --body or --body-file"
            )
        text = args.body or Path(args.body_file).read_text(encoding="utf-8")
        path = kb_mod.contribute_to_kb(
            domain=args.domain,
            finding=text,
            source=args.source,
            session_id=args.session_id,
        )
        _emit_json({"status": "appended", "path": str(path)}, args.out)
        return
    if op == "synthesize":
        findings: list[Finding] = []
        if args.findings:
            raw = json.loads(Path(args.findings).read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise RuntimeAdapterError(
                    "--findings file must contain a JSON array of Finding objects"
                )
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
    """Construct the top-level argparse parser for framework-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="framework-agent",
        description="Explore serving framework PRs/refs in isolated worktrees.",
    )
    # Global flags wired into logging_setup.configure_logging. Kept on the
    # top-level parser so every subcommand picks them up uniformly.
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
        help=(
            "Emit one JSON object per record (machine-friendly). "
            "Env fallback: FRAMEWORK_AGENT_LOG_JSON=1."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Append log records to this path in addition to stderr. "
            "Env fallback: FRAMEWORK_AGENT_LOG_FILE."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    schema_p = sub.add_parser("schema", help="Print the request schema summary")
    schema_p.set_defaults(func=_cmd_schema)

    # Sibling-skill `fa agent` subcommand -- P2 PR-D. Used by
    # inference_optimizer's FrameworkAgentBackend.
    # NOTE: ``framework_agent.agent`` is not always shipped (e.g. PR branch
    # ``framework-agent-pr`` ships runtime/ but not agent/). Tolerate the
    # missing module so the core ``fa candidates`` / ``fa explore`` verbs
    # (used by the inference_optimizer ``framework_pr`` bandit arm) still
    # work; the ``fa agent`` subcommand simply becomes unavailable.
    try:
        from ..agent.cli import register_subparser as _register_agent_subparser
        _register_agent_subparser(sub)
    except ModuleNotFoundError as _exc:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "framework_agent.agent not installed; skipping 'fa agent' subcommand (%s)",
            _exc,
        )

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

    # ----- Hyperloom FRAMEWORK_PR phase entry points -----
    pd_p = sub.add_parser(
        "phase-discover",
        help="Discover one batch of PR candidates for the FRAMEWORK_PR phase",
    )
    pd_p.add_argument("--request", required=True, help="JSON request file path")
    pd_p.add_argument("--out", default="-", help="Output path (default stdout)")
    pd_p.set_defaults(func=_cmd_phase_discover)

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
    kb_contrib_p.add_argument(
        "--body-file", default="", help="Read finding body from this file"
    )
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
    """Entry point invoked by both `framework-agent` and `fa` scripts."""
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
