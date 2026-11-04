# The round's dossier (parallel mode): inputs and deliverables

A parallel run (runner decision 0003, `arbor/parallel/`) maps each round before it fans
out: this campaign writes the dossier skill's update round uK for the round's base tree,
in two halves, while a node profiles that tree. Read this before the first iteration.

## The workspace

Built by `scripts/parallel/dossier_round.sh` from the controller's request; the harness
commits after every iteration.

| Path | What |
| --- | --- |
| `reports/` | the deliverables, `NN.uK-<part>.md`, and each rewrite's `11.uK-plan-<region>.md` |
| `inputs/dossier/` | the run's dossier: its parts, its earlier rounds, its index |
| `inputs/tree/vllm/` | the round's base tree: the last round's integrated tree (or the run's base) |
| `inputs/ROUND.md` | the last round's record: lanes and their clocks, items and verdicts, the integration table, what is reopenable, the directions its lanes proposed, the rewrites awaiting approval |
| `inputs/lanes/<lane>/` | each earlier lane's session as the knowledge collector stages it: `summary/`, `outcomes/OUTCOMES.md`, `findings/FINDINGS.md`, `artifacts/` (the kept iterations' patches, the ledger) |
| `inputs/profile/`, `inputs/bar.json` | part B only: the round's profile capture and its bar, measured on one node |

`inputs/` is read-only (the correctness check). Everything in it can be cited by path.

## Part A (beside the profile)

- **03.uK, the code read again** on `inputs/tree/`: each in-path method with its
  opportunity label and its history (the skill's step 1).
- **10.uK, the lessons** of every lane of the last round (the skill's step 4): its KEEPs
  and where they stand (integrated, dropped at the integration and why, conflicting); its
  kills and what each rests on, marked reopenable or not; what its probes established;
  the map against the work; the counts.

## Part B (after the profile)

- **06.uK, the bar**: `inputs/bar.json` (its replays and spread), and the floor it gives.
- **07.uK, the profile**: `inputs/profile/`, plan then results.
- **11.uK, the directions**, ranked from 03, 06, 07 and 10, ending in its `## Map` line
  with the lane fields: `lane`, `plan` (relative to `reports/`), `depends_on`, `nodes`,
  `upstream`. A direction that died earlier in the run (killed, below its floor, dropped
  or conflicting at an integration) comes back only with `reopens`. Rewrites are the
  operator's: rank them with their kernel plans, status `plan`; a lane starts only for
  one the operator approves.

## Done when

`score_dossier.py` gives every part its 10 and the map its 10, `check-dossier.py reports
--round uK --no-index` passes, and `arbor.directions check ... --dossier reports` prints
the order the dependencies allow. The controller runs those checks again, with the
reopen rule, before any lane starts.
