"""Framework Agent CLI entry point.

PR A ships two subcommands:

* ``fa schema``     - placeholder schema summary (real schema lands in PR-C).
* ``fa candidates`` - enumerate PR/ref candidates from the configured
  sources (primus_cortex in PR-A; github backend in PR-B).

``fa explore`` and ``fa kb`` land in subsequent PRs per the implementation
plan in ``claw-dev/docs-zh/framework-agent-hyperloom-implementation-plan.md``.
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
    """Print the ExploreRequest schema summary (PR-A placeholder).

    Full schema will land in the explore PR together with explorer.py.
    """
    del args
    _emit_json(
        {
            "status": "placeholder",
            "pr": "A",
            "note": (
                "PR-A ships schema + candidates only. Full ExploreRequest "
                "schema is delivered in subsequent PRs (explore + isolation)."
            ),
            "required": ["framework", "repo_url", "baseline"],
            "subcommands_available": ["schema", "candidates"],
            "subcommands_planned": ["explore", "kb"],
            "search_modes_supported": ["primus_cortex"],
            "search_modes_planned": ["github"],
        },
        "-",
    )


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


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for framework-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="framework-agent",
        description="Explore serving framework PRs/refs in isolated worktrees.",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point invoked by both `framework-agent` and `fa` scripts."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeAdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        print(f"ERROR: unexpected framework-agent failure: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
