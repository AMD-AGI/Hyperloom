"""python3 -m arbor run|master|worker|review|conflicts|restore|pause|checkpoint|iterate|version|schema|directions|
interface [--payload FILE]"""

import sys

# Force line-buffered stdout so tee/pipes see each line immediately.
# Must happen before any other imports so the buffer mode is set correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

USAGE = """Usage:
  python3 -m arbor run --config <path>             # run a campaign (a new one waits for a review)
  python3 -m arbor run --config <path> --no-review-gate   # start without the review
  python3 -m arbor run --config <path> --dry-run   # the loop on synthetic numbers: no GPU, no agent
  python3 -m arbor run --config <path> --split processes  # the master and each worker as a command, in turn
  python3 -m arbor master --config <path>          # the master's side, until a work order is ready or the end
  python3 -m arbor worker --order <work_order.json>       # one worker's stretch, from its order to its report
  python3 -m arbor review --session <dir> show|approve|reject   # review a new campaign before it runs
  python3 -m arbor conflicts --session <dir> show|pick|merge    # decide two skills that write the same thing
  python3 -m arbor restore --config <path> [--checkpoint REF]   # lay a checkpoint into the workspace, any node
  python3 -m arbor pause --config <path> [--reason TEXT]        # pause at the next pass boundary (exit 76)
  python3 -m arbor checkpoint list|show|verify|create|fork ...  # a session's checkpoints; a new session from one
  python3 -m arbor iterate init|begin|gate         # internal harness API
  python3 -m arbor version [--json]                # the version, and each interface's, for callers
  python3 -m arbor directions generate|split ...  # the map step a host runs (python3 -m arbor.directions)
  python3 -m arbor <command> --payload <file>      # the command's parameters as one JSON or YAML document
  python3 -m arbor --payload <file>                # the same, the command named in the file (command: run)
  python3 -m arbor schema [<command>] [--markdown]  # the payload's JSON Schema, from the commands' parsers
  python3 -m arbor interface [--json]              # the host contract: ends, events, records, timing (docs/interface.md)
  python3 -m arbor --version
Documentation: README.md and docs/ in this repository (docs/cli.md: every command's parameters)."""

# A payload (arbor/cli/payload.py) becomes the command's own flags before anything reads them.
if any(a == "--payload" or a.startswith("--payload=") for a in sys.argv[1:]):
    from arbor.cli.payload import run_expanded
    _expanded = run_expanded(sys.argv[1:])
    if isinstance(_expanded, int):
        sys.exit(_expanded)
    sys.argv[1:] = _expanded

if len(sys.argv) < 2:
    print(USAGE, file=sys.stderr)
    sys.exit(1)
if sys.argv[1] in ("-h", "--help", "help"):
    print(USAGE)
    sys.exit(0)
if sys.argv[1] in ("-V", "--version"):
    from arbor.version import line
    print(line())
    sys.exit(0)

cmd = sys.argv[1]

if cmd == "version":
    from arbor.version import main
    sys.exit(main(sys.argv[2:]))

if cmd == "run":
    if "--dry-run" in sys.argv:
        from arbor.dry_run import main
        rest = [a for a in sys.argv[2:] if a != "--dry-run"]
        sys.exit(main(rest))
    from arbor.driver import main
    sys.exit(main(sys.argv[2:]))

if cmd == "restore":
    from arbor.checkpoints import main_restore
    sys.exit(main_restore(sys.argv[2:]))

if cmd == "pause":
    from arbor.checkpoints import main_pause
    sys.exit(main_pause(sys.argv[2:]))

if cmd == "checkpoint":
    from arbor.checkpoints import main as main_checkpoint
    sys.exit(main_checkpoint(sys.argv[2:]))

if cmd == "master":
    from arbor.driver import main_master
    sys.exit(main_master(sys.argv[2:]))

if cmd == "worker":
    from arbor.driver import main_worker
    sys.exit(main_worker(sys.argv[2:]))

if cmd == "review":
    from arbor.review import main
    sys.exit(main(sys.argv[2:]))

if cmd == "conflicts":
    from arbor.conflicts import main
    sys.exit(main(sys.argv[2:]))

if cmd == "iterate":
    from arbor.cli.iterate import main
    sys.exit(main(sys.argv[2:]))

if cmd == "schema":
    from arbor.cli.payload import main
    sys.exit(main(sys.argv[2:]))

if cmd == "directions":
    from arbor.directions import main
    sys.exit(main(sys.argv[2:]))

if cmd == "interface":
    from arbor.interface import main
    sys.exit(main(sys.argv[2:]))

print(f"Unknown command: {cmd}", file=sys.stderr)
sys.exit(1)
