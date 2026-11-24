---
myst:
    html_meta:
        "description": "Review the agent map before a multi-branch session opens its agent branches: when MB waits, what it shows, the clock, what you may edit, how to approve or reject, and how to skip the review for unattended runs."
        "keywords": "Hyperloom, multi-branch, MB, review gate, agent map, dossier, approve, reject, no-review-gate"
---
# Reviewing the agent map before the agent branches open

A multi-branch session (MB) with component agents makes a map of directions before their branches open: the dossier
pass's map, a map generated from the root's profile, or one you gave with `--agent-directions`. A component agent is
an `--autonomous-agent` whose checkout an installed adapter recognises. With the review on, the default, MB then
waits until a person approves or rejects that map and its dossier, holding no GPU lease for it.

## When MB waits

- **Where:** after the map step, before it splits the map among the component agents and before any of their branches opens.
- **Which sessions:** any with component agents. That includes a session with one, which then takes the reviewed
  map whole, and a session given `--agent-directions`, whose map is shown too.
- **What waits:**
  - **A map the session makes** (the dossier pass's, or one generated from the root's profile) is made by a mapping
    branch in EXPLORE. Only that branch waits: the other branches go on, and the deadline stays.
  - A map you gave with `--agent-directions` is reviewed before EXPLORE. The session pauses, and its deadline
    moves by the pause.
- **Never:** with `--no-review-gate`, or with `--stop-after-map`. The latter's stop is the review point: it keeps
  the map and ends, for a review offline. A second run then takes the reviewed map and dossier with
  `--agent-directions` and `--agent-dossier`, and passes `--no-review-gate`.

## What you see

MB writes two files under the session directory and records the decision `review requested`:

- **`MB-review/REVIEW.md`:** the map file and its directions (id, status, title), the dossier directory and its parts,
  and the commands to decide. A review in EXPLORE adds the clock (below).
- **`MB-review/review.json`:** `status` (`pending`, `approved` or `rejected`), who decided and when, the note, the
  map and dossier paths, and, after an approval, `changed`: the files you edited, by their sha256.

## What you may edit

Edit the map file and the dossier where `REVIEW.md` names them. On approval MB reads both again from disk, splits the
map as edited and copies the dossier as edited into each share branch's checkout.

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

- **After an approval:** MB records `review approved`, with how long it waited, and opens the branches: in EXPLORE,
  one share branch per component agent, if there is still time for them (below).
- **After a rejection:** in EXPLORE, MB records `review rejected`, opens no share branch, and the session goes on.
  Before EXPLORE (a map you gave), it ends the session with stop reason `review_rejected` (exit 0), opening no branch.

## The clock

- **A review in EXPLORE does not stop the clock.** EXPLORE ends when it would have, whenever you decide, and each share
  must still have `--agent-share-minutes` (63 by default) before its cutoff to open: the time one
  iteration of a component agent may need.
  - **On the page:** `REVIEW.md` says when it was asked, by when to decide for the shares to open, and when EXPLORE
    ends; `review.json` keeps `open_by_iso` and `explore_end_iso`.
  - **Time left:** `review show` says how many minutes are left, or that the time has passed.
  - **Too late:** MB checks before the review and after it. Below that floor it opens no share and records a `SHARE`
    decision, `agent map: too little time left for the shares`, with the minutes a share would have had.
    `--agent-share-minutes 0` opens them whatever the time.
- **A review before EXPLORE** (a map you gave) does not count against `--max-hours`. MB moves the session's deadline,
  and EXPLORE's end, by the time it paused.
- **Either way,** a batch job's own time limit still runs: approve promptly, or skip the review.

## Skipping the pause

For an unattended run, pass `hyperloom-multibranch run --no-review-gate`. MB records `review skipped
(--no-review-gate)`, and each component agent it starts is told not to wait either.

## What a component agent is told

A component agent may have a review gate of its own. Under MB the host owns the review, and MB tells the agent's
adapter which case holds, for the agent's config:

- **The review on:** the host reviews, so the agent's own gate has nothing to wait for. The dossier pass is told the
  same.
- **The review off:** nobody reviews, and the agent does not wait either.
