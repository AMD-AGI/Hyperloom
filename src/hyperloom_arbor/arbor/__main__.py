"""python3 -m arbor run|iterate"""

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
  python3 -m arbor review --session <dir> show|approve|reject   # review a new campaign before it runs
  python3 -m arbor iterate init|begin|gate         # internal harness API
Documentation: README.md and docs/ in this repository."""

if len(sys.argv) < 2:
    print(USAGE, file=sys.stderr)
    sys.exit(1)
if sys.argv[1] in ("-h", "--help", "help"):
    print(USAGE)
    sys.exit(0)

cmd = sys.argv[1]

if cmd == "run":
    if "--dry-run" in sys.argv:
        from arbor.dry_run import main
        rest = [a for a in sys.argv[2:] if a != "--dry-run"]
        sys.exit(main(rest))
    from arbor.driver import main
    sys.exit(main(sys.argv[2:]))

if cmd == "review":
    from arbor.review import main
    sys.exit(main(sys.argv[2:]))

if cmd == "iterate":
    from arbor.cli.iterate import main
    sys.exit(main(sys.argv[2:]))

print(f"Unknown command: {cmd}", file=sys.stderr)
sys.exit(1)
