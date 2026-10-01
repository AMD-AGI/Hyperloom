---
name: hyperloom-kb
description: Operates an Experience KB service, local or global, with the hyperloom-kb CLI. Checks its health, labels its state, restores a label to roll back or undo a pull, excludes or includes an Experience with a reason, pushes to and pulls one schema from its global KB, and lists or exports what was written to it. Use when asked to label, roll back, restore, undo a pull, exclude, include, push, pull, list, or export Experiences, or to embed these commands in another tool.
---

# Operate an Experience KB

`hyperloom-kb` talks to one Experience KB service: the one `HYPERLOOM_KB_URL`
names, authenticated with `HYPERLOOM_KB_TOKEN`. A local and a global service
are the same service and take the same commands. Run it as
`hyperloom-kb <command>`, or as `python3 -m hyperloom_kb.cli <command>` where
the package is importable but the command is not on `PATH`.

Load both variables from wherever the user keeps them without printing them.
Never echo the token, and do not ask the user to paste it into chat.

Every command prints its result as JSON and exits:

- `0` when it worked;
- `1` when the service refused the request (the message on stderr names the
  HTTP status), or a push or pull stopped early, was refused, or rejected an
  Experience;
- `2` when the URL or the token is not configured.

When a tool embeds these commands (see the last section), run them through
that tool instead: it supplies its own schema, and may push and pull its own
way.

## The state reads see

Each schema on a service has one current state: the Experiences stored there,
written to it or pulled, minus the excluded ones. Reads see exactly that
state. Nothing is ever deleted.

```bash
hyperloom-kb health                  # kb_id, name, and what reads see per schema
hyperloom-kb labels [--schema REF]   # labels newest first, current_label_id, modified
```

`modified: true` means the state changed since its current label, or that a
non-empty state has no label yet. A command that names no schema acts on the
service's default schema, the `schema_ref` in `health`.

## Label and restore

```bash
hyperloom-kb label [--schema REF] [--name "before tuning"]
hyperloom-kb restore LABEL_ID
```

- A label is identified by its `label_id`. Its `name` is only for people and
  need not be unique, so always restore by `label_id`.
- A restore makes the label's state current: the Experiences and the
  exclusions it held. Experiences written after the label leave the state, and
  later writes add to the restored one.
- When the current state is not what its label saved, the restore first labels
  it (reason `before_restore`) and answers that label as `saved`. Report its
  `label_id`: restoring it brings the state back.

## Exclude and include

```bash
hyperloom-kb exclude EXPERIENCE_ID --reason "measured on a noisy node"
hyperloom-kb include EXPERIENCE_ID
hyperloom-kb exclusions [--schema REF]
```

- An exclusion hides an Experience from reads, whether it was written here or
  pulled. Ask for the reason when the user gave none; it stays in the history.
- An excluded Experience written here is not pushed until `include` releases
  it, even when a restore to a label without the exclusion lets reads see it
  again. One already pushed stays on the global KB: an exclusion acts only on
  this service.
- `include` answers `not_excluded` when the Experience was neither excluded
  nor waiting for an include to be pushed.
- `exclusions` lists the current exclusions and every exclude and include.

## Push and pull

Only a service started with a global KB pushes and pulls; any other answers
`sync_unavailable`.

```bash
hyperloom-kb push
hyperloom-kb pull --schema REF
```

- A push sends the Experiences written to this service and not pushed yet.
  Pulled ones count as `skipped`; excluded ones, and ones a restore set aside,
  count as `held_back` and go with the first push after reads see them again.
- A pull names one schema and brings its state to everything the global KB
  holds of it, including Experiences the global KB shows again after hiding
  them; reads see the result at once, and exclusions stand. When the
  state was unlabelled, the pull first labels it (reason `before_pull`) and
  answers that label as `saved`. Report its `label_id`: restoring it undoes
  the pull.
- `status` is `completed`; `incomplete`, with an `error`, when the global KB
  stopped answering, and running the command again resumes it; or `refused`,
  with the reason, when the URL answers as another KB than before, holds less
  than was pulled, or reports no identity. A refused sync changes nothing:
  report the reason rather than pointing the service at another global KB.
- `rejected` lists the Experiences the global KB refused for good.

## List and export

```bash
hyperloom-kb list   [--schema REF] [--include-excluded] [--after N] [--limit N]
hyperloom-kb export [--schema REF] [--include-excluded] [--after N] [--limit N]
```

Both name only Experiences written to this service, never pulled ones, and of
those only what reads see unless `--include-excluded` asks for the rest.
Without `--schema` they cover every schema. Each call returns one page: repeat
with `--after <next_cursor>` while `has_more` is true. `list` returns
summaries; `export` returns complete records and, with `--schema`, that
schema's declaration.

## Embed the commands in another tool

A tool with its own CLI gets every command above, with its own default schema:

```python
from hyperloom_kb import RemoteClient
from hyperloom_kb.cli import add_commands, run_command

commands = parser.add_subparsers(dest="command", required=True)
commands.add_parser("push", help="The tool's own push.")  # registered first, so it replaces the generic one
add_commands(commands, schema_ref=TOOL_SCHEMA_REF)
args = parser.parse_args()
if hasattr(args, "run"):  # one of the commands add_commands registered
    raise SystemExit(run_command(RemoteClient(tool_config), args))  # tool_config: the RemoteConfig of its service
```

`add_commands` registers every command whose name the tool has not registered
yet. Each one that names no schema acts on `schema_ref`, `list`, `export`, and
`pull` included. `run_command` prints the same JSON and returns the same exit
status as `hyperloom-kb`.
