---
myst:
    html_meta:
        "description": "How to optimize one model at several concurrencies with hyperloom-multibranch points: one session and one Arbor per operating point, mapped first, reviewed by a person, then run, on one machine or several."
        "keywords": "Hyperloom, multi-branch, Arbor, operating point, concurrency, points, review, map, dossier, container"
---
# Running several operating points, each with its own Arbor

A multi-branch session measures, gates and merges at one operating point, its `--conc`. To optimize a model at
several concurrencies, run one session per point. `hyperloom-multibranch points` does that in two stages, with a
person's review between them:

1. **map**: each point's session runs PRELUDE and its Arbor's map, then stops.
2. **review**: a person reads each point's map, edits it if they wish, and approves or rejects the point.
3. **run**: each approved point's session runs on its reviewed map and dossier.

Each point has its own Arbor, and that Arbor plans its own directions at its own operating point. No map is shared
among the points, and each point ends with its own recipe: nothing merges across the points. No machine waits for
the review: the stages are separate commands, which can run as separate jobs. How the sessions and their Arbors divide
the work is in [Arbor in multi-branch, who runs what](../conceptual/multibranch-arbor.md).

## What you need

- **Run arguments** that run Arbor: `--autonomous-agent <Arbor checkout>` and its `--agent-launch`.
- **A points directory on storage** that every stage and the reviewer can reach.
- **Points that share a machine:** PRELUDE's main flow runs the Magpie installed in the environment, and it must take
  its GPUs from the device mask it inherits and its server port from `PORT`. Magpie at the pin `install.sh` names
  (`MAGPIE_REF`) does neither yet.
  - **Install one that does,** with `MAGPIE_REPO` and `MAGPIE_REF` set for `install.sh`.
  - **Unset both before the session starts.** Set, they make the session pin your commit for its measurement runners,
    which fetch Magpie from upstream and cannot find it there.

## The points directory

```text
<dir>/
├── points.json          the run arguments the points share (written by map)
├── REVIEW.md            the review page
└── c<conc>/             one directory per point
    ├── point.json       its cards, port and Arbor port; each stage's session and exit code
    ├── map.log, run.log each stage's output
    ├── review.json      the person's decision
    └── userdata/        the point's USER_DATA_PATH: its sessions
```

A map session keeps its map in `MB-agent-map/`: `directions.json`, the dossier, and `agent-map.json`, which names
both.

## Stage 1: map every point

```shell
hyperloom-multibranch points map --dir /shared/points/r1 --conc 8 --conc 64 -- \
    --workload inferencex --isl 1024 --osl 1024 --framework vllm --model /models/Qwen3-8B --tp 1 \
    --gpu-type mi350x --precision bf16 --max-hours 3 --max-branches 1 \
    --autonomous-agent /opt/arbor --agent-launch /shared/agent-launch.json
```

- **What runs:** each point's session at once, on its own share of the machine's cards (`--cards-per-point` sets the
  share). Each session runs PRELUDE, then the dossier pass for its Arbor (one Arbor too), keeps the map, and stops
  with the reason `mapped`.
- **Its own ports:** a point gets `PORT` 8888 and Arbor port blocks from 31100, each 100 apart per point. Points in
  separate processes on one machine take separate `--slot`s (below).
- **Run arguments:** everything after `--` is shared by the points. Each stage sets `--conc`, `--agent-port`,
  `--stop-after-map`, `--no-review-gate`, `--reuse-prelude`, `--agent-directions` and `--agent-dossier` itself, so the
  command refuses them there.
- **Its end:** the command writes `REVIEW.md` and prints its path. It exits 1 when any point's session failed.

## Review the maps

`REVIEW.md` lists each point's session, its map's directions (rank, id, kind, direction, ceiling) and its dossier,
with the commands to decide on it. Edit a map file in place if you wish, then decide on each point:

```shell
hyperloom-multibranch points approve --dir /shared/points/r1 --conc 8
hyperloom-multibranch points reject --dir /shared/points/r1 --conc 64 --why "the map misses the MoE"
hyperloom-multibranch points review --dir /shared/points/r1     # write the review page again
```

- **What they write:** only `c<conc>/review.json` and the review page. They need no GPU and no installed
  environment, just the checkout:
  `PYTHONPATH=<checkout>/src python3 -m hyperloom_multibranch points approve ...`. A login node will do.
- **Rejecting:** needs `--why`.
- **A point without a map** cannot be decided on: read its session's `MB-reports/MB-report.md`.

## Stage 2: run the approved points

```shell
hyperloom-multibranch points run --dir /shared/points/r1
```

- **Each approved point's session:**
  - reuses its map session's PRELUDE (`--reuse-prelude`) and measures the baseline again on its own cards;
  - opens its Arbor on the reviewed map and dossier (`--agent-directions`, `--agent-dossier`);
  - skips MB's own review gate (`--no-review-gate`): the points' review was it.
- **The run arguments:** `points.json`'s, unless you give new ones after `--`, which replace them whole.
- **Which points:** `--conc` limits the stage to the points named. A rejected point, or one nobody reviewed, does not
  run, and the others go on. The command prints each point's outcome.
- **Running a stage again** starts a new session per point and records it in `point.json`. Earlier sessions stay on
  disk.

## Merge the points' trees

Each point ends with the tree its FINAL chose, its best tree, and different points may use different trees. The merge
puts every point's tree into one vLLM patch, behind one variable: `HYPERLOOM_OPERATING_POINT=<conc>` runs that point's
tree, and unset, the root's runs. One image then serves each point as its own session measured it.

```shell
hyperloom-multibranch points merge --dir /shared/points/r1 --aiter <the image's site-packages>/aiter
HYPERLOOM_OPERATING_POINT=64 vllm serve /models/Qwen3-8B ...      # with the patch applied to the image's vLLM
```

- **On by default:** `points run` ends with the merge when it ran every approved point itself; `--no-merge` skips it.
  A run limited with `--conc`, one container per point, leaves it to one `points merge` once every point has run.
- **What a point changed, as code under the variable:**
  - **server arguments** become the defaults of vLLM's parser, so an argument given on the command line still wins;
  - **environment** is set when vLLM is imported, where a variable is not set already;
  - **kernel tables** ship in the patch, laid over the image's defaults as the point's tests read them, and the
    variable that loads them names them;
  - **code** runs under `if` the point is the one named, widened to the whole statements that hold the change. Where
    two points changed the same statements, the branches are `if`/`elif`; points that changed them alike share one.
- **Where it writes,** below `<dir>/merged/`:
  - `merged.patch`, the patch;
  - `recipe.json`, the root's recipe carrying it;
  - `merged.json` and `MERGED.md`: what each point brought, and which files more than one point changed.
- **Run it in the image:** a rows-only kernel table is laid over the image's aiter, which `--aiter` names.
- **What it refuses, saying why:** a point started from another root, a change outside Python source, a new or deleted
  file, or a change that leaves no whole statements. A point whose run ended without FINAL's choice is left out.
- **Not done here:** whether one point could use another's changes. That is a review of its own, later.

## Points that share a machine: one container each

Magpie ends a benchmark by killing every serving process it can see. Points in one process namespace kill each other's
servers: one point's cleanup takes down another point's server mid-measurement. Give each point a container of its
own:

- **One `points` command per container,** for its own point:
  `points map --conc <its conc> --slot <k>`, and later `points run --conc <its conc> --slot <k> -- <its run arguments>`.
  Each container's map writes `points.json` over the last, so give each container's run stage its own run arguments.
- **Ports:** with host networking, the containers share the machine's ports, so each takes its own `--slot`: 0, 1, ...
- **Cards:** the container's device mask (`ROCR_VISIBLE_DEVICES`) is its point's cards. Set it after `install.sh`:
  `install.sh` imports Ray, and Ray refuses that mask at import.
- **No Ray head:**
  - `install.sh` runs with `SKIP_RAY_START=1`, and the session with `INFERENCE_OPTIMIZER_RAY_EXEC=0`.
  - Ray heads in several containers collide on the machine's network, and one shared head offers one serving slot for
    the whole machine.
  - Without Ray, PRELUDE serves its baseline on its own leased card.
- **Installs:** run `install.sh` in one container at a time when the containers share an environment on storage.
  Installs in parallel race on it.

## Budgets

`--max-hours` sets when the session's deadline falls, not when the session ends:

- **EXPLORE** ends one root test, plus a short CLOSE grace, before the deadline.
- **FINAL** then measures each candidate with a sweep at conc/2, conc and 2×conc, and a reproduction at conc. It waits
  for those tests, past the deadline if need be.
  - An InferenceX test has a 60 min timeout.
  - A sweep that runs out of its time is not rerun: FINAL ranks its candidate on the reproduction, and the choice's
    reason says so.
- **The agent cutoff:** Arbor stops early enough before EXPLORE ends for MB to measure its last keep and confirm it
  on Arbor's branch. That is the tests in flight plus two of the session's longest test: about 50 min at C8 and 20 at
  C64 in the example below, and 40 min before any test was measured. Merging that keep into the integration head takes
  two tests more, which may not fit; the keep then reaches FINAL as the head of Arbor's branch.

On a cluster that caps a job's time, size each point's `--max-hours` so the session ends before the cap. Count the
container and install time first. Then add FINAL's length beyond its reserve, and CLOSE. With a container per point,
each point can have its own budget.

| Qwen3-8B BF16, 1k/1k, TP1, MI350X, one 8-card node | C8 | C64 |
| --- | --- | --- |
| One test (boot, warmup, GSM8K, measure) | 25 min | 5–10 min |
| Stage 1, the map | 37 min, both points | 37 min, both points |
| Stage 2, `--max-hours` | 3.0 | 3.25 |
| FINAL | 60 min (one sweep timed out, one took 57) | 22 min |
| Stage 2's session ended, from the job's start | 3 h 36 min | 3 h 24 min |
| Arbor's run in stage 2 | 71 min, 12 iterations | 113 min, 12 iterations |
