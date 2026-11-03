---
name: campaign-archive
description: Archive a running Arbor campaign to git and rebuild any campaign's exact source tree from the archive. Use when syncing campaign patches/ledgers to the Arbor repo, when asked to recover or reproduce a campaign's code, or when a landed.patch will not apply.
---

# Campaign archive and exact recovery

An Arbor campaign's source tree lives **inside a Docker container** on a compute node
(`/usr/local/lib/python3.12/dist-packages/vllm`, made into a git repo). It is not on NFS and not
bind-mounted. It dies with the container. This skill covers getting the reproducible part into git, and
rebuilding the tree byte-for-byte from what git holds.

Verified 2026-09-17: rebuild from the archive reproduces the live container tree, **15/15 files identical by
sha256**.

## Layout

| Path | Contents |
|---|---|
| `/data/dsv41/sessions/<campaign>/` | live session: `session.json`, `ledger.jsonl`, `landed.patch`, `stock_base/`, `iter_N/` |
| `scripts/dsv41-campaign/` | `sync_campaign_to_git.sh`, `check_review_consistency.py`, the sbatch files |
| `results/dsv41flash-agentx-c32/runs/<campaign>/` | the archived ~2 MB slice |
| `results/dsv41flash-agentx-c32/CAMPAIGN_REVIEW.md` | hand-written analysis |

Session folders are ~2.7 GB, but **99.96% is replayable aiperf telemetry**. Everything with intellectual
content — patches, ledgers, verdicts, stock base — is ~2 MB. Archive that; leave `agentic_run/` behind.

## Sync

```bash
bash /data/dsv41/sync_campaign_to_git.sh
```

Copies `session.json`, `ledger.jsonl`, `landed.patch`, `stock_base/` and every `iter_N/patch.diff` into
`results/dsv41flash-agentx-c32/runs/`, runs the review consistency check, commits and pushes to
`huasha_dsv41_flash`. Safe against running campaigns: read-only except for appending a missing trailing
newline to `landed.patch`. Idempotent — prints `nothing changed` when there is nothing new.

Runs every 20 minutes on a cron while campaigns are live.

## Recover a campaign's exact tree

```bash
R=results/dsv41flash-agentx-c32/runs/<campaign>     # dsv41flash-c32 | dsv41flash-c32-v2

cp -r "$R/stock_base" rebuild && cd rebuild
git init -q . && git add -A && git -c user.name=r -c user.email=r@r commit -qm stock
git apply --whitespace=nowarn "$R/landed.patch"
```

`git status --short` should list ~14 modified/new files. To verify against a live container:

```bash
find . -name '*.py' -not -path './.git/*' | sort | xargs sha256sum | sed 's|\./||' > /tmp/repro.sha
srun --jobid=<JOBID> --overlap -n1 bash -c '
  while read -r h f; do
    live=$(docker exec <CONTAINER> bash -lc "cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum \"$f\" | cut -d\" \" -f1")
    [ "$h" = "$live" ] && echo "MATCH  $f" || echo "DIFFER $f"
  done < /tmp/repro.sha'
```

To rebuild a single iteration instead of the cumulative state, apply `iter_N/patch.diff` on top of the
parent state rather than `landed.patch`.

## Refresh `stock_base` for a new campaign

`landed.patch` is a **diff against the root commit, not a snapshot**, so it is meaningless without the exact
base it was cut against. The full container tree is 2.0 GB; only the paths the patches touch are needed
(~579 KB).

```bash
# union of every path any patch references
{ grep -h '^+++ b/' */landed.patch */iter_*/patch.diff
  grep -h '^--- a/' */landed.patch */iter_*/patch.diff; } \
  | sed -e 's|^+++ b/||' -e 's|^--- a/||' | grep -v '^/dev/null' | sort -u > paths.txt

srun --jobid=<JOBID> --overlap -n1 bash -c '
  ROOT=$(docker exec <CONTAINER> bash -lc "cd /usr/local/lib/python3.12/dist-packages/vllm && git rev-list --max-parents=0 HEAD")
  echo "$ROOT" > <session>/stock_base_root_commit.txt
  while read -r f; do
    mkdir -p "<session>/stock_base/$(dirname "$f")"
    docker exec <CONTAINER> bash -lc "cd /usr/local/lib/python3.12/dist-packages/vllm && git show $ROOT:\"$f\"" \
      > "<session>/stock_base/$f" 2>/dev/null
  done < paths.txt'
```

Paths absent at root are the files the campaign created; the patch carries those in full, so skipping them
is correct.

## Traps — every one of these has already cost real work

**`landed.patch` must end with a newline.** `git()` strips it, so `save_landed_patch` re-appends it
(`arbor/execute/workspace.py`). Without it `git apply` reports `corrupt patch at line <last>` and rejects the
**whole file** even when the hunk arithmetic is complete — and `git apply` is atomic, so nothing lands. This
was not cosmetic: a rejected `landed.patch` made the sbatch restore block fall back to replaying
per-iteration patches, and that fallback globbed `$SESSIONS/*/session.json` — every campaign — so V2's
container restored ORIGINAL's iterations 1, 12, 13, 15 and 18. One missing byte cross-contaminated a
campaign. If a patch will not apply, check the last byte before anything else:

```bash
[ -n "$(tail -c1 landed.patch)" ] && printf '\n' >> landed.patch
```

**`stock_base` is per-session, never shared.** The two campaigns have different root commits (`d4b6ca7` vs
`9ce4108`). Content is identical today only because both containers came from the same pinned image. A
container rebuilt from a newer image diverges silently, and then one campaign's base corrupts the other's
patches.

**Use `cp -r`, not `cp -a`.** Copying from NFS to a local filesystem, `cp -a` fails on
`preserving permissions: Operation not supported`, returns non-zero, and under `set -e` aborts the copy
half-done. The apply then fails with a misleading `No such file or directory` per file.

**Never name an archive directory `sessions/`.** `.gitignore` has a bare `sessions/`, which git matches at
*any* depth. An entire archive subtree was silently swallowed once — `git add -A` reported nothing and 25
files vanished. Hence `runs/`.

**Commit from `/home/<colleague>/Arbor`, never `/data/dsv41/Arbor`.** The `/apps` tree is bind-mounted
into both live containers as `/workspace/Arbor`, sits at an older commit, and carries local modifications.
Running git operations there while drivers read packs and validators out of it risks the running jobs.

**Exclude `__pycache__`/`*.pyc`.** Regenerated bytecode made `has_changes()` fire while `commit()` staged
nothing, so `git commit` exited 1 and a whole iteration was discarded — ledger records included.

**Never `docker rm`/`run`/`restart` during an iteration.** The tree lives in the container's writable layer;
recreating it discards every landed KEEP and leaves `session.json` pointing at a tree that no longer exists.

## Pinned image

`vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`

`stock_base` covers only patched paths. A full rebuild of the whole tree needs this image. **If the tag
becomes unpullable and no local copy survives, the unpatched remainder is unrecoverable** — the patches
themselves stay appliable, since their base is archived.

## Review consistency

```bash
python3 scripts/dsv41-campaign/check_review_consistency.py
```

Cross-checks the hand-written `CAMPAIGN_REVIEW.md` against the machine-written `session.json`: every KEEP has
a table row, no row is a non-KEEP, header counts and `best_metric` match, no column left as `—`. It exists
because V2 iteration 23 — the campaign best at the time — was logged correctly by Arbor in all five automated
places and still went missing from the review table, while the file's own header said 7 KEEPs over a 6-row
table. It cannot catch a wrong *claim*, only a wrong count.
