---
name: hyperloom-kb
description: Operates an Experience KB service with the hyperloom-kb CLI. Checks its health, labels its state, restores a label to roll back, excludes or includes an Experience with a reason, and lists or exports what was written to it. Use when asked to label, roll back, restore, exclude, include, list, or export Experiences, or to embed these commands in another tool.
---

# Operate an Experience KB

`hyperloom-kb` talks to one Experience KB service: the one `HYPERLOOM_KB_URL`
names, authenticated with `HYPERLOOM_KB_TOKEN`. Run it as
`hyperloom-kb <command>`, or as `python3 -m hyperloom_kb.cli <command>` where
the package is importable but the command is not on `PATH`.

Load both variables from wherever the user keeps them without printing them.
Never echo the token, and do not ask the user to paste it into chat.

Every command prints its result as JSON and exits:

- `0` when it worked;
- `1` when the service refused the request (the message on stderr names the
  HTTP status);
- `2` when the URL or the token is not configured.

When a tool embeds these commands (see the last section), run them through
that tool instead: it supplies its own schema.

## The state reads see

Each schema on a service has one current state: the Experiences stored there,
minus the excluded ones. Reads see exactly that state. Nothing is ever
deleted.

```bash
hyperloom-kb health                  # kb_id, name, and what reads see per schema
hyperloom-kb labels [--schema REF]   # labels newest first, current_label_id, modified
```

`modified: true` means the state changed since its current label, or that a
non-empty state has no label yet. A command that names no schema acts on the
service's default schema, the `schema_ref` in `health`.

## Health and metrics

`hyperloom-kb health` reports the KB's identity and what its reads see per
schema. Three endpoints answer without a token, for an orchestrator, a
metrics scraper, or a quick look:

```bash
curl -s "$HYPERLOOM_KB_URL/readyz"    # ready, and which check failed when not
curl -s "$HYPERLOOM_KB_URL/metrics"   # Prometheus metrics: requests, latency, writes, storage
```

A failure `hyperloom-kb` reports names its request id; the service logs one
JSON line per request with that `request_id`, so search its log for it.

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

- An exclusion hides an Experience from reads. Ask for the reason when the
  user gave none; it stays in the history.
- `include` lifts the exclusion, and also puts back an Experience a restore
  set outside the state, so one no label holds any more, after its label was
  deleted, can still be brought back. It answers `not_excluded` when the
  Experience was neither excluded nor outside the state.
- `exclusions` lists the current exclusions and every exclude and include.

## List and export

```bash
hyperloom-kb list   [--schema REF] [--include-excluded] [--after N] [--limit N]
hyperloom-kb export [--schema REF] [--include-excluded] [--after N] [--limit N]
```

Both name the Experiences written to this service, and of those only what
reads see unless `--include-excluded` asks for the rest. Without `--schema`
they cover every schema. Each call returns one page: repeat with
`--after <next_cursor>` while `has_more` is true. `list` returns summaries;
`export` returns complete records and, with `--schema`, that schema's
declaration.

## Embed the commands in another tool

A tool with its own CLI gets every command above, with its own default schema:

```python
from hyperloom_kb import RemoteClient
from hyperloom_kb.cli import add_commands, run_command

commands = parser.add_subparsers(dest="command", required=True)
commands.add_parser("list", help="The tool's own list.")  # registered first, so it replaces the generic one
add_commands(commands, schema_ref=TOOL_SCHEMA_REF)
args = parser.parse_args()
if hasattr(args, "run"):  # one of the commands add_commands registered
    raise SystemExit(run_command(RemoteClient(tool_config), args))  # tool_config: the RemoteConfig of its service
```

`add_commands` registers every command whose name the tool has not registered
yet. Each one that names no schema acts on `schema_ref`, `list` and `export`
included. `run_command` prints the same JSON and returns the same exit status
as `hyperloom-kb`.
