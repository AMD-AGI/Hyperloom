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
            "subcommands_available": ["schema", "candidates", "explore", "kb"],
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
    from ..agent.cli import register_subparser as _register_agent_subparser
    _register_agent_subparser(sub)

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
