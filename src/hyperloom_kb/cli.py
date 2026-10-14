# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-kb``: operate the Experience KB service ``HYPERLOOM_KB_URL`` names, a local or a global one alike.

A tool of its own embeds the same commands with ``add_commands`` and runs them with ``run_command``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from hyperloom_kb.remote import RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.schema import JsonValue

_Run = Callable[[RemoteClient, argparse.Namespace], dict[str, JsonValue]]
_SYNC_COMMANDS = ("push", "pull")


def _schema_option(parser: argparse.ArgumentParser, default: str | None, unset: str | None) -> None:
    """``--schema``, defaulting to ``default``; without one, ``unset`` says what the command acts on, or ``None`` that
    it must name a schema."""

    required = default is None and unset is None
    parser.add_argument(
        "--schema",
        dest="schema_ref",
        default=default,
        required=required,
        help="The schema_ref to act on" + ("." if required else f" (default: {default or unset})."),
    )


def _page_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--after", type=int, default=0, help="The previous page's next_cursor.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-excluded", action="store_true", help="Also name the ones reads do not see.")


def _list(client: RemoteClient, args: argparse.Namespace) -> dict[str, JsonValue]:
    page = client.list_experiences(
        after=args.after, limit=args.limit, schema_ref=args.schema_ref, include_excluded=args.include_excluded
    )
    return {"items": list(page.items), "next_cursor": page.next_cursor, "has_more": page.has_more}


def _export(client: RemoteClient, args: argparse.Namespace) -> dict[str, JsonValue]:
    page = client.export_page(
        after=args.after, limit=args.limit, schema_ref=args.schema_ref, include_excluded=args.include_excluded
    )
    result: dict[str, JsonValue] = {
        "items": list(page.items),
        "next_cursor": page.next_cursor,
        "has_more": page.has_more,
        "head": page.head,
    }
    if page.declaration is not None:
        result["declaration"] = page.declaration.to_dict()
        result["state"] = page.state
    return result


def add_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser], *, schema_ref: str | None = None
) -> None:
    """Register every ``hyperloom-kb`` command whose name ``commands`` does not hold yet.

    A tool replaces a command by registering its own under that name first. A command that names no schema acts on
    ``schema_ref``; when that is ``None``, a pull must name one, list and export cover every schema, and the rest act
    on the service's default schema.
    """

    def command(name: str, description: str, run: _Run) -> argparse.ArgumentParser | None:
        if name in commands.choices:
            return None
        parser = commands.add_parser(name, help=description)
        parser.set_defaults(run=run)
        return parser

    default_schema = "the schema the service reads by default"
    command("health", "The service's identity, and what its reads see per schema.", lambda c, a: c.health())
    command("push", "Send what was written here and not pushed yet to the global KB.", lambda c, a: c.push())
    command(
        "rebind",
        "Forget the global KB synced with, such as one redeployed at the same URL, so the next push and pull start "
        "over with whichever KB answers there; what was pulled stays pulled and is never pushed.",
        lambda c, a: c.rebind(),
    )
    if pull := command(
        "pull",
        "Bring one schema to everything the global KB holds of it; a state no label holds is labelled first, so "
        "restoring that label undoes the pull.",
        lambda c, a: c.pull(a.schema_ref),
    ):
        _schema_option(pull, schema_ref, None)
    if labels := command(
        "labels",
        "List a schema's labels, its current label, and whether its state changed since.",
        lambda c, a: c.labels(schema_ref=a.schema_ref),
    ):
        _schema_option(labels, schema_ref, default_schema)
    if label := command(
        "label",
        "Label a schema's current state, so a restore can return to it.",
        lambda c, a: c.create_label(schema_ref=a.schema_ref, name=a.name),
    ):
        _schema_option(label, schema_ref, default_schema)
        label.add_argument("--name", default="", help="A name for people; the label is identified by its label_id.")
    if restore := command(
        "restore",
        "Make a label's state current; a current state no label holds is labelled first.",
        lambda c, a: c.restore(a.label_id),
    ):
        restore.add_argument("label_id")
    if exclude := command(
        "exclude",
        "Hide an Experience from reads; one written here is no longer pushed either.",
        lambda c, a: c.exclude(a.experience_id, reason=a.reason),
    ):
        exclude.add_argument("experience_id")
        exclude.add_argument("--reason", required=True, help="Why it is excluded; kept in the exclusion history.")
    if include := command(
        "include",
        "Let reads see an Experience again: lift its exclusion, and put it back into the state if a restore set it "
        "outside.",
        lambda c, a: c.include(a.experience_id),
    ):
        include.add_argument("experience_id")
    if exclusions := command(
        "exclusions",
        "List a schema's exclusions and the history of every exclude and include.",
        lambda c, a: c.exclusions(schema_ref=a.schema_ref),
    ):
        _schema_option(exclusions, schema_ref, default_schema)
    if listing := command("list", "Summaries of the Experiences written here, one page in write order.", _list):
        _schema_option(listing, schema_ref, "every schema")
        _page_options(listing)
    if export := command(
        "export", "Complete records of the Experiences written here, one page in write order.", _export
    ):
        _schema_option(export, schema_ref, "every schema")
        _page_options(export)


def run_command(client: RemoteClient, args: argparse.Namespace) -> int:
    """Run a command ``add_commands`` registered against ``client``'s service, print its JSON result, and return the
    exit status.

    A push or pull that stopped early, was refused, or rejected an Experience exits 1 like a failed request.
    """

    try:
        result = args.run(client, args)
    except RemoteClientError as exc:
        print(f"Experience KB {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in _SYNC_COMMANDS and (result["status"] != "completed" or result["rejected"]):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyperloom-kb",
        description="Operate an Experience KB service, local or global. Connects with HYPERLOOM_KB_URL and "
        "HYPERLOOM_KB_TOKEN; prints each result as JSON.",
    )
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        config = RemoteConfig.from_env()
    except RemoteClientError as exc:
        print(f"hyperloom-kb: {exc}", file=sys.stderr)
        return 2
    if config is None:
        print("hyperloom-kb: HYPERLOOM_KB_URL must be configured", file=sys.stderr)
        return 2
    return run_command(RemoteClient(config), args)


__all__ = ["add_commands", "main", "run_command"]


if __name__ == "__main__":
    raise SystemExit(main())
