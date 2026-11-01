---
myst:
    html_meta:
        "description": "Review the agent map before a multi-branch session opens its Arbor branches: when MB pauses, what it shows, what you may edit, how to approve or reject, and how to skip the pause for unattended runs."
        "keywords": "Hyperloom, multi-branch, MB, Arbor, review gate, agent map, dossier, approve, reject, no-review-gate"
---
# Reviewing the agent map before the Arbor branches open

A multi-branch session (MB) with Arbor agents makes a map of directions before its Arbor branches open: the dossier
pass's map, a map generated from the root's profile, or one you gave with `--agent-directions`. With the review on,
the default, MB then **pauses** until a person approves or rejects that map and its dossier. Nothing runs while it
waits: the pause holds no GPU lease.

## When MB pauses

- **Where:** after the map step, before it splits the map among the Arbors and before any Arbor branch opens.
- **Which sessions:** any with Arbor agents (`--autonomous-agent` naming an Arbor checkout). That includes a session
  with one Arbor, which then takes the reviewed map whole, and a session given `--agent-directions`, whose map is
  shown too.
- **Never:** with `--no-review-gate`, or with `--stop-after-map`. The latter's stop is the review point: it keeps
  the map and ends, for a review offline (the `points` command's first stage; see
  {doc}`Run several operating points with Arbor </how-to/multibranch-points>`).

## What you see

MB writes two files under the session directory and records the decision `review requested`:

- **`MB-review/REVIEW.md`:** the map file and its directions (id, status, title), the dossier directory and its parts,
  and the commands to decide.
- **`MB-review/review.json`:** `status` (`pending`, `approved` or `rejected`), who decided and when, the note, the
  map and dossier paths, and, after an approval, `changed`: the files you edited, by their sha256.

## What you may edit

Edit the map file and the dossier where `REVIEW.md` names them. On approval MB reads both again from disk, splits the
map as edited and copies the dossier as edited into each Arbor branch's checkout.

## Approve or reject

```bash
hyperloom-multibranch review --session <MB session dir> show
hyperloom-multibranch review --session <MB session dir> approve --note "why"
hyperloom-multibranch review --session <MB session dir> reject --note "why"
```

- **`--by NAME`** records who decided (the login name by default).
- **Exit codes:** 0 when done; 2 when there is no review, or it is already decided.
- **Where to run it:** anywhere that sees the session directory. The command only reads and writes `review.json`, so
  a login node on the same shared storage will do. Without MB's virtualenv there, its module needs only the standard
  library:

  ```bash
  PYTHONPATH=<hyperloom checkout>/src python3 -c 'import argparse, sys
  from hyperloom_multibranch.scheduling import review
  p = argparse.ArgumentParser(); review.add_parser(p.add_subparsers(dest="command"))
  sys.exit(review.main(p.parse_args(["review", *sys.argv[1:]])))' approve --session <dir> --note "why"
  ```

- **After an approval:** MB records `review approved`, with how long it paused, and opens the branches.
- **After a rejection:** MB records `review rejected` and ends the session with stop reason `review_rejected`
  (exit 0), opening no branch.

## The clock

The pause does not count against `--max-hours`. MB moves the session's deadline, and EXPLORE's end, by the time it
paused. A batch job's own time limit still runs, though: approve promptly, or skip the pause.

## Skipping the pause

For an unattended run, pass `hyperloom-multibranch run --no-review-gate`. MB records `review skipped
(--no-review-gate)`, and each Arbor it starts is told not to wait either (`"review_gate": false` in its config).

## What Arbor is told

Arbor has a review gate of its own, at the start of a new campaign. Under MB the host owns the review:

- **The review on:** MB writes `component.review: "host"` in every Arbor config it starts, the dossier pass's
  included, and Arbor's gate passes at once, recording that the host reviewed.
- **The review off:** MB writes `"review_gate": false`.
- **An Arbor without the gate** (before Arbor `1fa2069`) gets neither key: MB reads whether the checkout's
  `arbor/component.py` lists `review` among its block keys.

Which side does what in a session: {doc}`Arbor in multi-branch </conceptual/multibranch-arbor>`.
