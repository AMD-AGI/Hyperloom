<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Prior-art gotchas

Consulted from the steps of [`../SKILL.md`](../SKILL.md) that name them. Each is a way a real
match, or a real absence, turns into a wrong verdict.

## G1 — Issue and PR numbers share a namespace

`#N` in a body, a comment or a commit subject can be an issue as well as a pull request: GitHub
numbers both from one counter per repository. A "fixes #N" link usually names the issue the PR
closes, not another PR; open it before following it as a successor. The counter is also per
repository, so `#N` in one repository and `#N` in another, a fork included, are unrelated. The
inclusion probe matches the PR number in commit subjects, so run it only against a tree of the PR's
own repository or a fork that carries it, and compare the matched subject with the PR's title.

## G2 — A duplicate may fix a different failure mode

Two PRs can touch the same kernel, the same file and the same title words and still be different
changes: one fixes a crash on a shape, the other a slowdown on another; one adds a fast path, the
other removes a bad fallback. Classify on what the diff changes and on which condition it reaches,
never on the title. A PR whose description reports your symptom but whose diff changes another
code path is OVERLAPPING at most.

## G3 — A merged PR is not in an image built before it

"Merged upstream" says nothing about the tree or container you measure. A release, a nightly or a
vendor image is cut from some commit; a PR merged after it is absent however old the image looks,
and a branch cut before the merge does not gain it later. Decide inclusion from the commit, with
the probe, never from dates. An open or closed-unmerged PR has no merge commit: `mergeCommit`
in `pr.json` stays null until it merges, while REST's `merge_commit_sha` names a test merge that no
branch contains.

## G4 — An installed package is not a tree

A framework installed into a container usually has no `.git`. Its version often carries the
commit it was built from as a local version segment, `<version>+g<sha>`; read it with
`python -c 'import importlib.metadata as m; print(m.version("<distribution>"))'`. Clone the
upstream repository and pass that sha as `--commit`. A `--filter=blob:none` clone answers by
subject and by merge-commit ancestry; a cherry-pick matched by patch-id needs the blobs, and the
probe, which fetches nothing, answers `unknown` without them. A version without `+g<sha>` names no
commit: write `SKIPPED: inclusion -- <distribution> <version> names no commit`.

## G5 — A fork may carry a cherry-pick without ancestry

A fork that cherry-picks an upstream PR, or squashes it into a merge of its own, never contains
the upstream merge commit. The probe still finds the carry when a commit subject names the PR, a
trailing `(#N)` or `Merge pull request #N from`, which is how most cherry-picks record their origin,
and, given the merge commit, when every commit the PR brought in has the same patch-id in the tree.
A backport that was edited, had a conflict resolved, applied next to lines the fork changed, or was
folded into a larger commit matches neither and reads `unknown`. So Step 6 searches the tree's log
for the change itself before a verdict calls it unchecked: `git log -S<symbol>` for a function or
constant the diff adds, and `git log --grep=<words from the title>`. An open upstream PR can be
carried the same way; it has no merge commit, so only its subject or that log search finds it, and
a fork that carries it gets `EXISTS-OPEN -- in your tree`, not a second measurement.

## G6 — A validated operating point may differ from yours

A PR's benchmark covers its author's model, shapes, batch sizes, data types, GPU architecture and
parallelism. Merged and included does not mean active for you: the change may sit behind an
architecture check, a dtype check, an environment variable or a config flag, or win only on shapes
yours never reach. Read the condition that selects the new path in the diff and check that your
workload satisfies it. A DUPLICATE whose condition excludes your operating point is OVERLAPPING,
and the next step names the gap.

## G7 — Paths differ between a package, a fork and upstream

Step 3 compares paths as upstream spells them. An installed package drops the repository's
top-level layout (`site-packages/<pkg>/x.py` is often `<pkg>/x.py`, sometimes `python/<pkg>/x.py`
or `src/<pkg>/x.py`), and a fork can move files. Map every planned path to the upstream default
branch, and check it exists there: a path upstream never had matches nothing, and the scan still
says `COMPLETE`. A rename counts under both names in Step 3, which reads an open PR's old paths
from REST, but not in `pathlog.txt`: `git log -- <path>` stops at a merged move, so when upstream
moved a planned file after your commit, log the old and the new path.

## G8 — The target repository may require a duplicate-work statement

Some upstream projects ask a contributor to search for duplicate or related work before opening a
PR, and to say so in the description; some close duplicates on sight. Read the target repository's
contributor guide, agent guide and PR template before opening anything, and quote the report's
Coverage part where they ask for it.
