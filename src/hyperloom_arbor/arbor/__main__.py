"""python3 -m arbor run|master|worker|review|conflicts|iterate|version"""

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
  python3 -m arbor iterate init|begin|gate         # internal harness API
  python3 -m arbor version [--json]                # the version, and each interface's, for callers
  python3 -m arbor --version
Documentation: README.md and docs/ in this repository."""

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

print(f"Unknown command: {cmd}", file=sys.stderr)
sys.exit(1)
