---
name: knowledge-collector
description: Share what an Arbor campaign learned so the next campaign on the model starts from it, and read what earlier campaigns learned. Collects each iteration's outcome and why (kept, reverted, probed, not measured), the ledger's facts, assumptions and dead ends, profile reports, the dossier, and the patches, with a manifest, and pushes them to the shared volume. Use at the end of a campaign or chunk, after a profile or a kept change, when asked to share or archive a campaign's results, or before starting a campaign, to see what others learned on the same model.
---

# Knowledge collector: what a campaign learned, for the next one

Audience: whoever runs or ends an Arbor campaign, and whoever starts the next
one on the same model. The collector is `python3 -m arbor.collect knowledge`
(`arbor/collect/knowledge.py`, standard library only). Access to the share,
and the project, user and campaign names, are set up as in the
`realtime-collector` skill; the same flags and variables apply here.

## What it collects

One directory per type, staged in `<session>/collector/knowledge/<type>/` and
pushed to `/shared/arbor/projects/<project>/<user>/<campaign>/knowledge/<type>/`:

| Type | Files | From |
| --- | --- | --- |
| `summary` | `SUMMARY.md`; `manifest.json` (every file with size and sha256, and what was skipped) | everything below |
| `outcomes` | `OUTCOMES.md`, `outcomes.jsonl`: each iteration kept, reverted, probed or not measured, with what changed and why | `session.json`'s lessons |
| `findings` | `FINDINGS.md`, `findings.jsonl`: facts, assumptions, unknowns and dead ends, with evidence, falsifier and when to revisit | `ledger.jsonl`, the latest record per id |
| `profiles` | per profile: its report and small files (`PROFILE.md`, `profile.json`, `census_summary.json`, `LIMITATIONS.md`, `*_check.json`, `*.csv`) | `<session>/profiles/*/`, and each `--profiles` |
| `dossier` | the dossier's `*.md` | `--dossier` |
| `artifacts` | `session.json`, `ledger.jsonl`, `landed.patch`, `ABORTED.md`, `REPORT.md`, every `iter_N/patch.diff`, and `bench_output.json` of the iterations kept | the session |

Raw traces and records never go: a file over 5 MB is listed in the manifest as
skipped. The summary quotes each profile report's title and its
`## Candidates` (or findings) section, so a reader sees the leads without
opening the profiles.

## Run it

On the host that holds the session, from an Arbor checkout:

```bash
python3 -m arbor.collect project --session <session> --config <config>     # the names, first time
python3 -m arbor.collect knowledge --session <session> --config <config> \
    [--dossier <dossier dir>] [--profiles <dir>]... [--types outcomes,findings]
```

- `--no-push` stages only; read `<session>/collector/knowledge/` before sharing.
- **When:** at the end of a campaign or of each chunk; after a profile; after
  a kept change others should see. It is cheap and can be run again at any
  time. Each run restages the types it collects and replaces those files on
  the share.
- **It never deletes on the share.** A file the latest collection no longer
  stages stays there. `summary/manifest.json` lists what the latest
  collection holds, and the dashboard marks anything else "from an earlier
  collection".
- **Everyone with access to the workspace reads the share.** A session
  directory must not hold keys; if one does, fix that before collecting.
- Exit status: 0 when pushed (or staged, with `--no-push`); 1 when the share
  could not be reached or written (the staged copy stays); 2 for bad arguments
  or an unknown type.

## Profiles and the dossier

- **Profiles** are found at `<session>/profiles/<name>/`, where the `profiler`
  skill writes a campaign's profile report as `PROFILE.md`, with the small
  files beside it. Pass a profile kept elsewhere with `--profiles <dir>`.
- **The dossier** the campaign worked from (`docs/dossier/<model>-<workload>-<gpu>/`,
  found by the `dossier` skill) goes with `--dossier`. Once shared, the dossier
  skill's lookup finds it on the share too, so the next campaign on the same
  model, workload and GPU type starts from it instead of writing one.

## Reading what others learned

Before a campaign, or when asked what is known about a model:

1. **Find the campaigns**: the dashboard's global mode, filtered by project
   (the `dashboard` skill); or
   `ssh arbor-share 'ls /shared/arbor/projects/<project>/*/*/knowledge/summary/SUMMARY.md'`;
   on a Cluster B node, the same path directly.
2. **Read** `SUMMARY.md`, then `OUTCOMES.md` (what was tried and why it was
   kept or reverted) and `FINDINGS.md` (dead ends, with when to revisit each).
3. **Use it as leads, with their labels.** A result holds for its pins, node,
   GPU type, seed and operating points (the campaign's `artifacts/session.json`
   and its summary say which). MI350 never stands in for MI355X. A revert at
   another operating point, or on other pins, is not a dead end here unless its
   reason carries over; a dead end's `revisit_if` says when it reopens.
4. **A kept change's code** is `artifacts/iter_N/patch.diff`. `landed.patch`
   is the campaign's cumulative diff against its own base, and applies only to
   that base (the `campaign-archive` skill).
