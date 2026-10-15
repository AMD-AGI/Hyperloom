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
from pathlib import Path

from hyperloom_kb.distill.directory import sync_directory
from hyperloom_kb.remote import RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.schema import JsonValue

_Run = Callable[[RemoteClient, argparse.Namespace], dict[str, JsonValue]]
#: Whether a command that answered still did not do what it was asked.
_Failed = Callable[[dict[str, JsonValue]], bool]
_Command = Callable[..., "argparse.ArgumentParser | None"]
_DEFAULT_SCHEMA = "the schema the service reads by default"
_DRAFT_HELP = "The draft: a JSON file, or - for stdin."


class CommandInputError(ValueError):
    """Raised when a command's input, such as a draft file, cannot be read."""


def _json_value(text: str, name: str) -> JsonValue:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise CommandInputError(f"{name} is not JSON: {exc}") from exc


def _json_file(path: str, name: str) -> JsonValue:
    """The JSON value in the file ``path``, or on stdin for ``-``."""

    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CommandInputError(f"cannot read the {name} {path}: {exc}") from exc
    return _json_value(text, f"the {name} {path}")


def _sync_failed(result: dict[str, JsonValue]) -> bool:
    return result["status"] != "completed" or bool(result["rejected"])


def _default_schema(client: RemoteClient, args: argparse.Namespace) -> str:
    return str(args.schema_ref or client.health()["schema_ref"])


def _adder(commands: argparse._SubParsersAction[argparse.ArgumentParser], prefix: str = "") -> _Command:
    """Register a command in ``commands`` unless it holds that name; a command without ``run`` groups others."""

    def command(
        name: str, description: str, run: _Run | None = None, *, failed: _Failed | None = None
    ) -> argparse.ArgumentParser | None:
        if name in commands.choices:
            return None
        parser = commands.add_parser(name, help=description, description=description)
        if run is not None:
            parser.set_defaults(run=run, failed=failed, label=f"{prefix}{name}")
        return parser

    return command


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

    A tool replaces a command by registering its own under that name first; ``schema``, ``experience``, and
    ``distill`` each group their subcommands, replaced together. A command that names no schema acts on
    ``schema_ref``; when that is ``None``, a pull must name one, list, export, and the Distill changes and sync cover
    every schema, and the rest act on the service's default schema.
    """

    command = _adder(commands)
    command("health", "The service's identity, and what its reads see per schema.", lambda c, a: c.health())
    command(
        "push",
        "Send what was written here and not pushed yet to the global KB, then the latest version of each visible "
        "Distill.",
        lambda c, a: c.push(),
        failed=_sync_failed,
    )
    command(
        "rebind",
        "Forget the global KB synced with, such as one redeployed at the same URL, so the next push and pull start "
        "over with whichever KB answers there; what was pulled stays pulled and is never pushed.",
        lambda c, a: c.rebind(),
    )
    if pull := command(
        "pull",
        "Bring one schema to everything the global KB holds of it, its Distills included; a state no label holds is "
        "labelled first, so restoring that label undoes the pull.",
        lambda c, a: c.pull(a.schema_ref),
        failed=_sync_failed,
    ):
        _schema_option(pull, schema_ref, None)
    _add_state_commands(command, schema_ref)
    _add_record_commands(command, schema_ref)
    if distill := command("distill", "Explore Experiences, and write, read, and sync the Distills drawn from them."):
        subcommands = distill.add_subparsers(dest="distill_command", required=True)
        _add_distill_commands(_adder(subcommands, "distill "), schema_ref)


def _add_state_commands(command: _Command, schema_ref: str | None) -> None:
    if labels := command(
        "labels",
        "List a schema's labels, its current label, and whether its state changed since.",
        lambda c, a: c.labels(schema_ref=a.schema_ref),
    ):
        _schema_option(labels, schema_ref, _DEFAULT_SCHEMA)
    if label := command(
        "label",
        "Label a schema's current state, so a restore can return to it.",
        lambda c, a: c.create_label(schema_ref=a.schema_ref, name=a.name),
    ):
        _schema_option(label, schema_ref, _DEFAULT_SCHEMA)
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
        _schema_option(exclusions, schema_ref, _DEFAULT_SCHEMA)


def _add_record_commands(command: _Command, schema_ref: str | None) -> None:
    if listing := command("list", "Summaries of the Experiences written here, one page in write order.", _list):
        _schema_option(listing, schema_ref, "every schema")
        _page_options(listing)
    if export := command(
        "export", "Complete records of the Experiences written here, one page in write order.", _export
    ):
        _schema_option(export, schema_ref, "every schema")
        _page_options(export)
    if schema := command("schema", "Look at a schema."):
        show = _adder(schema.add_subparsers(dest="schema_command", required=True), "schema ")(
            "show",
            "A schema's declaration, the paths a Distill's scope may name, its roles, and how many Experiences, "
            "Distills, and forms it holds.",
            lambda c, a: c.schema(_default_schema(c, a)),
        )
        _schema_option(show, schema_ref, _DEFAULT_SCHEMA)
    if experience := command("experience", "Look at one Experience."):
        get = _adder(experience.add_subparsers(dest="experience_command", required=True), "experience ")(
            "get",
            "One held Experience in full, whether reads see it, and where the service holds its files.",
            lambda c, a: c.experience(a.experience_id),
        )
        get.add_argument("experience_id")


def _new_draft(client: RemoteClient, args: argparse.Namespace) -> JsonValue:
    """The draft file of a new Distill, of the default schema when it names none."""

    draft = _json_file(args.draft, "draft")
    if isinstance(draft, dict) and "schema_ref" not in draft:
        draft["schema_ref"] = _default_schema(client, args)
    return draft


def _preview(client: RemoteClient, args: argparse.Namespace) -> dict[str, JsonValue]:
    draft = _json_file(args.draft, "draft") if args.entry else _new_draft(client, args)
    return client.preview_distill(draft, entry_id=args.entry)


def _sync(client: RemoteClient, args: argparse.Namespace) -> dict[str, JsonValue]:
    schemas = [args.schema_ref] if args.schema_ref else list(client.health()["distills"])
    return sync_directory(client, Path(args.dir), schemas, representatives=args.representatives)


def _add_distill_commands(command: _Command, schema_ref: str | None) -> None:
    every_schema = "every schema"
    if explore := command(
        "explore",
        "The group a scope selects among a schema's visible Experiences: counts, runs, decisions, and outcome "
        "statistics, crossed over one or two fields with --group-by.",
        lambda c, a: c.explore(
            _default_schema(c, a),
            scope=_json_value(a.scope, "--scope"),
            group_by=a.group_by,
            member_limit=a.member_limit,
        ),
    ):
        _schema_option(explore, schema_ref, _DEFAULT_SCHEMA)
        explore.add_argument(
            "--scope", default="{}", help='Conditions as JSON, e.g. \'{"identity.model": ["qwen3"]}\'.'
        )
        explore.add_argument("--group-by", action="append", default=[], help="A path to cross by; at most twice.")
        explore.add_argument("--member-limit", type=int, help="How many member ids each group names.")
    if forms := command(
        "forms", "The forms a schema's Distills are written in.", lambda c, a: c.distill_forms(_default_schema(c, a))
    ):
        _schema_option(forms, schema_ref, _DEFAULT_SCHEMA)
    if add_form := command(
        "add-form",
        "Register a form, the claim fields a Distill fills in, from a JSON file (- for stdin).",
        lambda c, a: c.register_distill_form(_default_schema(c, a), _json_file(a.form, "form")),
    ):
        _schema_option(add_form, schema_ref, _DEFAULT_SCHEMA)
        add_form.add_argument("form", help="The form declaration: a JSON file, or - for stdin.")
    if preview := command(
        "preview",
        "What creating the draft in a JSON file (- for stdin), or with --entry updating that Distill to it, would "
        "store; nothing is stored, and a draft the KB would refuse exits 1 with its errors.",
        _preview,
        failed=lambda result: result["valid"] is not True,
    ):
        _schema_option(preview, schema_ref, f"the draft's, else {_DEFAULT_SCHEMA}")
        preview.add_argument("draft", help=_DRAFT_HELP)
        preview.add_argument("--entry", help="The Distill an update would change.")
    if create := command(
        "create",
        "Create a Distill from the draft in a JSON file (- for stdin); it is visible at once.",
        lambda c, a: c.create_distill(_new_draft(c, a)),
    ):
        _schema_option(create, schema_ref, f"the draft's, else {_DEFAULT_SCHEMA}")
        create.add_argument("draft", help=_DRAFT_HELP)
    if update := command(
        "update",
        "Add a version to a Distill: the draft in a JSON file (- for stdin) in full, its evidence drawn again.",
        lambda c, a: c.update_distill(a.entry_id, _json_file(a.draft, "draft")),
    ):
        update.add_argument("entry_id")
        update.add_argument("draft", help=_DRAFT_HELP)
    if get := command(
        "get",
        "One version of a Distill: its claim and evidence, how the evidence moved, and the members representing it.",
        lambda c, a: c.distill(
            a.entry_id,
            version=a.version,
            follow_evidence=a.follow_evidence,
            after=a.after,
            limit=a.limit,
            representatives=a.representatives,
        ),
    ):
        get.add_argument("entry_id")
        get.add_argument("--version", type=int, help="A version other than the one the Distill shows.")
        get.add_argument("--follow-evidence", action="store_true", help="Add a page of the Experiences it rests on.")
        get.add_argument("--after", type=int, default=0, help="The evidence page's previous next.")
        get.add_argument("--limit", type=int, help="How many Experiences an evidence page holds.")
        get.add_argument("--representatives", type=int, help="How many representing members to name (default 5).")
    if draft := command(
        "draft",
        "The draft that writes a Distill's version again over the KB as it is, to edit and pass to update.",
        lambda c, a: c.distill(a.entry_id, version=a.version)["draft"],
    ):
        draft.add_argument("entry_id")
        draft.add_argument("--version", type=int, help="A version other than the one the Distill shows.")
    if listing := command(
        "list",
        "A schema's Distills, newest first, each with how its evidence moved since it was written.",
        lambda c, a: c.list_distills(_default_schema(c, a), include_deleted=a.include_deleted),
    ):
        _schema_option(listing, schema_ref, _DEFAULT_SCHEMA)
        listing.add_argument("--include-deleted", action="store_true", help="Also list the deleted ones.")
    if changes := command(
        "changes",
        "The visible Distills whose evidence moved in a way their author should look at, and what moved.",
        lambda c, a: c.distill_changes(schema_ref=a.schema_ref),
    ):
        _schema_option(changes, schema_ref, every_schema)
    if delete := command(
        "delete",
        "Hide a Distill from listings and syncs, with a reason; it stays readable and restore-entry brings it back.",
        lambda c, a: c.exclude(a.entry_id, reason=a.reason),
    ):
        delete.add_argument("entry_id")
        delete.add_argument("--reason", required=True, help="Why it is deleted; kept in the exclusion history.")
    if restore_entry := command("restore-entry", "Bring back a deleted Distill.", lambda c, a: c.include(a.entry_id)):
        restore_entry.add_argument("entry_id")
    if sync := command(
        "sync",
        "Mirror the visible Distills into a directory, one markdown page each, for a Curated Wiki to search; pages "
        "of Distills no longer visible are removed.",
        _sync,
    ):
        _schema_option(sync, schema_ref, "every schema with Distills")
        sync.add_argument("--dir", required=True, help="The directory the KB owns; only its Distill pages change.")
        sync.add_argument("--representatives", type=int, help="How many representing members a page names (default 5).")


def run_command(client: RemoteClient, args: argparse.Namespace) -> int:
    """Run a command ``add_commands`` registered against ``client``'s service, print its JSON result, and return the
    exit status.

    A command that answered without doing what it was asked exits 1 like a failed request: a push or pull that
    stopped early, was refused, or rejected a record, and a preview of a draft the KB would refuse.
    """

    try:
        result = args.run(client, args)
    except (RemoteClientError, CommandInputError) as exc:
        print(f"Experience KB {args.label} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if args.failed is not None and args.failed(result) else 0


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


__all__ = ["CommandInputError", "add_commands", "main", "run_command"]


if __name__ == "__main__":
    raise SystemExit(main())
