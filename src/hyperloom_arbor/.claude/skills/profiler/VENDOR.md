# ruppell, vendored

`ruppell/` is a copy of ruppell, the graph-aware tracer, taken whole at one
commit so that this skill works where it is copied (a node, a container)
without a second checkout.

source: ruppell's repository (AMD internal) (since 2026-10-19; it moved from
`<fork>/ruppell` with its whole history, so the commit below is in both)
commit: 5a5e146b3fe1bf1fd56cfec59168dbf86518b499
date: 2026-10-23

Left out: `tests/p[0-9]*/`, `tests/d4/` and `tests/drafter/`, the job
scripts of the runner's plans (about 1 MB). Nothing else in ruppell refers to
them; the counter jobs [counters.md](counters.md) cites are in the ruppell
repository.

## Rules

- **Do not edit `ruppell/`.** A fix, an adapter or a new limitation goes to
  ruppell first, as a pull request with its evidence (its `AGENTS.md`, and
  `ruppell/.claude/skills/ruppell-profile/new-target.md`, "How to
  contribute"). Then re-vendor.
- **Re-vendor only a commit the remote has**, so the copy can be rebuilt from
  it:

  ```bash
  bash sync-ruppell.sh <ruppell checkout> [<commit>]   # default: origin/main
  ```

  It replaces `ruppell/` with that commit's tree, minus the folders above,
  and rewrites `commit:` and `date:` here. Commit the result as one change,
  `profiler: ruppell <old> -> <new>`.
