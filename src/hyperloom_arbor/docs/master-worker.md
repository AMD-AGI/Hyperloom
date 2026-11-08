# The master and its workers

An Arbor session has two sides:
- **The master** sets the campaign up, writes the map, ends each round and ends the campaign.
- **A worker** runs the rounds: it iterates on the map the master set, and reports what it did.

This page says which side does what, the two records that cross between them, and where the split stands.

## Which side runs what

| Side | Chapters (`arbor/lifecycle.py`, `ROLES`) | Code |
| --- | --- | --- |
| **Master** | `review`, `init`, `benchmark`, `accuracy`, `map.discover`, `map.measure`, `map.directions`, `round_end`, `end` | `arbor/master.py`: the setup (refusals, the review gate, skill conflicts, `init`, the baseline, a host's run, the root's accuracy), what each pass must clear, a round end's A/B, the campaign's end |
| **Worker** | `iterate` | `arbor/worker.py`: a stretch of passes, from a work order to a report |

**Which side runs the next pass** follows the map (`roles.role_of`), as that pass's own start will leave it.

| The map | The next pass is the |
| --- | --- |
| none yet, in a campaign that maps (the pass starts one) | master's |
| being written (`bootstrap`, `unmapped`) or updated (`updating`, `fresh`) | master's |
| set and waiting to start (`pending`) | worker's: it is that worker's first pass |
| `active`, with a direction open | worker's |
| `active` with every direction closed, or waiting on its round end's A/B (`verifying`) | master's: the next pass starts the update round |
| a round end pending | master's |
| no map rounds (a pack that does not map) | worker's, after the setup |
| a document campaign (its work is the dossier) | master's throughout |

## The records

| Record | Written | Holds |
| --- | --- | --- |
| `work_order.json` (`arbor.work_order/1`) | when a worker's stretch starts | its `id`; the campaign and its session; the stretch's first iteration and its round; the map (its phase, and the directions the stretch works, as DIRECTIONS entries: a set map's, else the active map's open ones); the base (the workspace and its HEAD); the dossier and the benchmark; the skills loaded; the bar; the target; the limits (iterations, time budget, a host's deadline) |
| `worker_report.json` (`arbor.worker_report/1`), and `WORKER_REPORT.md` | when it ends | the `id` of the order it answers; why it ended (`round_end`, `map`, `spent`, or the run's own reason); its iterations; each decision's count; its KEEPs; the bar it ended on; the directions it closed (with why), left open and added; the directions it proposed to a host; a summary; `account`, when the worker wrote its own |
| `WORKER_SUMMARY.md` | at its end, by the `worker-summary` skill, when the pack loads it | the worker's own account: what it tried, what held and why, what did not and what that showed, the directions, and what the next map should weigh, each claim with its evidence |

Each stretch's records are also kept under `<session>/roles/`, by first iteration: `0047-work_order.json`,
`0047-worker_report.json`, and `0047-WORKER_SUMMARY.md` when it has one.

**The worker's own account:**
- **When:** a stretch's end fires `stretch_end`, once its report is written, with the order's id, the stretch's
  iterations, why it ended and the report's path.
- **Who writes it:** the `worker-summary` skill (`.claude/skills/worker-summary/`), a blocking trigger skill on that
  event. A pack loads it (`lifecycle.load`; `serving-m10` does).
- **Its time:** the worker waits for it before handing over, within the time the run has left (a host's deadline less
  its margin, or the time budget), and skips it with under a minute left. The journal says `ran`, `skipped` or
  `timed_out`.
- **The report names it** (`account`) only when that end wrote it.

## The commands

| Command | Runs | Stops |
| --- | --- | --- |
| `python3 -m arbor run --config C` | both sides, in one process, switching between passes | when the campaign ends |
| `python3 -m arbor run --config C --split processes` | `arbor master`, then `arbor worker --order`, in turn, each a process of its own | when the master ends the campaign |
| `python3 -m arbor run --config C --split workers` | `arbor master --deal`, then several workers at once, one per slot of the config's `workers` block, then the integration (below) | when the master ends the campaign |
| `python3 -m arbor master --config C [--deal]` | the master's passes: the setup, the map, a round end and its update round. With `--deal`, it starts the set map itself, and ends the campaign when only the integration's reserve is left | once the next pass is a worker's, with the work order written (`ordered`); or when the campaign ends (`ended`) |
| `python3 -m arbor worker --order W [--config C]` | one worker's stretch, from the order: in the master's session, or with `--config` in a session of its own (below) | once the next pass is the master's, or the run ends; with the report written (`reported`) |
| `python3 -m arbor.roles split --order W --shares N --out DIR` | nothing: cuts an order into N orders, `DIR/work_order-K.json`, each with a share of its directions (those joined by `depends_on` stay together) and an id of its own | |
| `python3 -m arbor.roles deal (--order W \| --map M) [--reports R ...] [--held ID ...] [--held-order O ...] [--size 2] [--format order\|share] --out FILE` | nothing: the next order for a free worker, the best-ranked directions of the master's order, or a host's map, that no report settled and no running worker holds (below) | one JSON line saying where every direction stands. With `--order`: a work order, written when there is one, `null` when nothing is left. With `--map` (a host): a share, `{"directions": [...]}`, always written, the list empty when nothing is left |

- **How each ended:** `<session>/role_exit.json` (`arbor.role_exit/1`): the side; the outcome (`ordered`, `reported`
  or `ended`); the record it wrote; and, when the run itself ended (`STOP`, the time budget, `--max-iters`, an abort),
  the run's own reason (`done`, `aborted`, ...). Under a host, `component_exit.json` says `ordered` or `reported` too
  (component interface 2).
- **`--split processes` from a host:** `ARBOR_SPLIT=processes` makes it the default, so a host that launches
  `arbor run` runs the split with no change of its own. A SIGTERM to `arbor run` goes on to the command that runs, which
  exits as it would have (in component mode, `sigterm`), and no command starts after it.
- **Run twice, they do nothing twice.** A master whose last order has no report yet hands that order over again, and
  runs nothing: it is the worker's turn. A worker whose order has its report hands it over again.
- **A worker:**
  - runs in its master's session, unless `--config` names a session of its own. Without `--config`, an order whose
    session is not the one its config names is refused (exit 2);
  - never waits at the review gate: its master's campaign was reviewed;
  - takes its time budget's start from the order, so a budget bounds the whole run, not each command. `--split
    processes` passes the run's start to each master.
- **Events:** the campaign's `end` is the master's: a hand-over fires none, and a stretch's end fires
  `stretch_end`. Each command fires `session_start`, as a
  resumed `arbor run` does. `map_set` fires once per map round in the session, whichever command sees the round
  start: a round the journal has announced is not announced again.

**Dealing, rather than splitting.** A fixed share leaves a worker whose share runs dry
with nothing to do while others still have work. So the master keeps the map and deals from it:
- **What a free worker gets:** up to `--size` directions that no report settled and no running worker holds. Fresh
  ones come before ones reported back open, then come by rank and the map's order.
- **Dependencies:** a direction comes with the free ones it depends on, and waits while one it depends on is held.
- **Exhausted:** a direction reported back open `MAX_RETURNS` (2) times is not dealt again in the round.
- **`roles.fold`** writes a report's closures (with why, the order that closed them, and `spent` when a spent share
  closed them) and its proposals into the master's own map.
- **Under a host:** MB calls `deal` for a freed slot.

**A worker of its own session** (`--config C`, C's session not the master's), as a host runs one per share:
- **Its map** is a lane of the order's directions (`map.directions`), active at once, as a host's `component.plan`
  makes one. The stretch ends when every direction is closed. A config that also gives a `component.plan` is refused,
  and so is an order without directions (exit 2).
- **A spent share goes back.** Before each pass, the worker checks every open direction of its
  map. When each was worked `map_worked_flag` times (4 by default) since its last KEEP:
  - it closes each `killed`, marked `spent`, with the iterations that worked it;
  - it reports `ended: "spent"` instead of running the next pass;
  - the master's `fold` closes them in its map with the mark, so no one else is dealt them this round, and its update
    round may bring them back.

  A worker in its master's session never hands a share back spent: there, the map's prompt asks the agent to close a
  direction worked that often, as before.
- **Its dossier** is the master's, copied to its own place when that is empty.
- **Its order** is the master's, renumbered from its session's next iteration, with `from` saying where the master
  wrote it. The id stays, so its report answers the master's order.
- **Its records** stay in its own session: `work_order.json`, `worker_report.json` and `role_exit.json`. Run again, it
  hands its report over and runs nothing.
- **Its own init and baseline:** a fresh session runs `init` and measures its baseline on its node, as any session
  does. Its report gives its bar with its scale.
- **Its report and its tree back into the master's session:** a host does that itself; a standalone master does it with
  `--split workers` (below).

## Several workers per round

`arbor run --split workers` (`arbor/dealer.py`): one master and several workers per round, standalone, in one job on one
node.

**Slots** (`arbor/slots.py`). Each worker runs in a slot: its own container (the workspace is the container's installed
tree), its cards (`tp` each) and its port. The launcher starts the containers from the campaign's image, each with its
workspace git-inited, as it starts the master's. The config says where they are:

```json
"workers": {"count": 2, "gpu_ids": "0,1", "container": "{container}-w{k}", "port_step": 10, "size": 2,
            "reserve_minutes": 45}
```

| Key | What | Default |
| --- | --- | --- |
| `count` | how many slots | as many as the cards allow, `len(gpu_ids) // tp` |
| `gpu_ids` | the cards the slots share out, `tp` each | 0-7 |
| `container` | each slot's container, `{container}` and `{k}` filled | `{container}-w{k}` |
| `port_step` | slot k serves on the config's port plus k times this | 10 |
| `slots` | each slot in full instead: `docker_container`, `gpu_ids`, `port`, and `workspace_root` when it differs | |
| `size` | the directions dealt in one order | 2 |
| `reserve_minutes` | the run's time kept for the integration: no order is dealt with less left, and a dealt order's budget ends that much before the run's | 45 |
| `ab_rounds` | the measured pairs of each integration A/B | `arbor.ab`'s, 3 |

Slots that share a card, a container or a port, or that are the master's own workspace, are refused (exit 2). The
master's own GPU steps (its map, the A/Bs, its update round) run while no worker runs, so the slots may take every card.

**A round:**
1. **The master maps** (`arbor master --deal`): the setup, the dossier, the map. It starts the map itself, since no
   pass runs in its session while its workers work, and exits `ordered`, its order listing the map's open directions.
2. **Dealing** (`roles.deal`): each free slot gets an order of up to `size` open directions, by rank, as their
   dependencies allow. The slot's workspace is reset to its first commit (the image) and the round's base applied as
   one commit (`arbor base: ...`): the master's tree as `landed.patch` holds it, with its `arbor/config.json`.
3. **The workers run at once:** each `arbor worker --order W --config C`, in a session of its own,
   `<session>/workers/<order>/session`, with its config, order and `worker.log` beside it. It runs its own init and
   baseline on the base.
4. **A slot frees when its worker exits:**
   - its report folds into the master's map (`roles.fold`);
   - what it kept comes back from its workspace (`slots.addition`: the diff from its base commit, with its config at
     both ends);
   - its server is stopped, and the slot is dealt again.
5. **Drain:** nothing is left to deal and no worker runs. A direction still open was handed back too often, or waits
   on one that was: it is closed `killed`, marked `unsettled`, saying why. When only the reserve is left, they stay
   open, and the master ends the campaign instead of starting a round.
6. **The integration** (`arbor/integrate.py`), with every card free. The additions go onto the master's tree by rank:
   - **merged:** the patch by `git apply --3way` (the image's blobs are in every workspace, so the base it was made on
     is the ancestor), the server flags and env per key;
   - **measured:** an A/B on this node of the tree before against the tree after (`arbor.ab`);
   - **confirmed:** kept; **not confirmed:** the tree goes back; **a conflict:** dropped, with its paths or keys.

   `<session>/integrate/round-<n>/integration.json` holds each outcome, and each A/B has its own directory.
7. **The round's report** answers the master's order (`worker_report.json` in its session: `ended` is `drained`, or
   `time`), with `workers` and `integration`. The master's next command takes the map on: its update round, then the
   next round.

**What the master's session records:** each worker's stretch as a NOTE at an iteration of its own, and each measured
addition as a KEEP (with the A/B's mean) or a REVERT, citing the order's directions. The round's record counts both, so
two rounds without a KEEP still stop the map. `landed.patch` follows the integrated tree.

**Not yet:** editable packages other than the workspace (`arbor/execute/packages.py`) do not cross between workspaces;
a worker's account (`worker-summary`) stays in its own session.

## Where the split stands

- **First step (done):** the two sides in code, one process. What runs, in which order, is what it was.
- **Second step (done):**
  - `arbor master`, `arbor worker`, `arbor run --split processes`, `role_exit.json`;
  - a worker in a session of its own, from an order (`--config`), and `arbor.roles split` for shares;
  - the worker's own account of each stretch (`stretch_end`, `worker-summary`);
  - checked on a GPU node under MB (with `ARBOR_SPLIT=processes`), and standalone (several map rounds, a master and a
    worker in turn).
  - **Where a standalone master runs:** in the same job and on the same node as its workers, as `arbor run` always
    ran.
- **Third step (done):** dealing (`roles.deal`, `roles.fold`), the spent hand-over, slots, the dealer and the
  integration, above; the pack-generic A/B (`arbor.ab`).
  - **Checked on a GPU node:** two TP1 workers on one node, two rounds:
    - every worker ran in its own container and session;
    - the integration kept additions its same-node A/B confirmed, and dropped one that measured slower;
    - each round's report took the master to its update round;
    - the campaign ended when the time left was the integration's reserve.
  - **What the earlier checks found:**
    - a dealt worker was prompted as a plan-file lane;
    - the A/B passed `gpu_ids` as a string to the server stop;
    - every server the harness started set `ROCR_VISIBLE_DEVICES` alone, which Ray, imported by vLLM 0.27, refuses.
- **Under a host:**
  - **Hyperloom stays the master that owns the session.** It would call:
    1. `arbor master` (today's `component.stop_at: "map"` pass), which exits `ordered`;
    2. `python3 -m arbor.roles split` on its order (today's `arbor.directions split`), or `deal` for a freed slot;
    3. `arbor worker --order W_k --config C_k` per share, each in its branch's session (today's `component.plan`),
       which exits `reported`, or the run's own reason.

    That is an adapter change in Hyperloom, to be agreed with Hyperloom first.
  - **Already the master's alone:** a dossier pass stops before any worker pass, and writes no work order.

## For callers: the version

`python3 -m arbor --version` prints the release line and the checkout. `python3 -m arbor version --json` gives what a
caller checks before relying on a checkout:
- its commands (`master` and `worker` from `cli` 2; `run --split workers` and `master --deal` from `cli` 3);
- each interface's version: `cli`, `pack`, `component` and `roles`;
- the component block's keys, and the exit record's reasons (`ordered` and `reported` from `component` 2,
  `retraction_conflict` from `component` 3);
- the records' schemas (`role_exit` from `roles` 2; a report that ends `spent` from `roles` 3);
- the chapters, the events, and which side runs each chapter.

An interface's version goes up when it changes in a way an old caller would break on.
