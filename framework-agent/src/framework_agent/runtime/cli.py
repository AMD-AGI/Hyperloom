"""Framework Agent CLI entry point.

PR 1 ships only the `schema` subcommand as a placeholder so the install
step has something concrete to smoke-test (`fa schema`). Other subcommands
(`explore`, `candidates`, `kb`) are added in later PRs per the
implementation plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


class RuntimeAdapterError(RuntimeError):
    """Raised when CLI input or runtime adaptation fails."""


def _emit_json(obj: Any, out: str | None) -> None:
    """Serialize obj as JSON to stdout or a file path."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        from pathlib import Path

        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _cmd_schema(args: argparse.Namespace) -> None:
    """Print the ExploreRequest schema summary.

    Full schema lands in PR 3 together with models.py and explorer.py.
    PR 1 returns a minimal placeholder that callers can inspect to confirm
    the CLI is wired correctly.
    """
    del args
    _emit_json(
        {
            "status": "placeholder",
            "pr": "1",
            "note": "Full schema is delivered in subsequent PRs (models + explorer).",
            "required": ["framework", "repo_url", "baseline"],
            "subcommands_planned": ["explore", "candidates", "schema", "kb"],
        },
        "-",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for framework-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="framework-agent",
        description="Explore serving framework PRs/refs in isolated worktrees.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    schema_p = sub.add_parser("schema", help="Print the request schema summary")
    schema_p.set_defaults(func=_cmd_schema)

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
