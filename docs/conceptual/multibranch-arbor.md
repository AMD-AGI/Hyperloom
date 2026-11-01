---
myst:
    html_meta:
        "description": "Arbor run by Hyperloom's multi-branch mode (MB) in component mode: which side runs each step of a session, the files that pass between them, Arbor's exit codes against what MB does, and the known gaps."
        "keywords": "Hyperloom, multi-branch, MB, Arbor, component mode, autonomous agent, dossier pass, KEEP snapshot, lease, review gate"
---
# Arbor in multi-branch: what runs in Arbor, what runs in MB

In component mode, Hyperloom's multi-branch mode (MB) runs Arbor as an autonomous branch.
- **MB owns** the node, the GPUs, the clock, the measurement of record and the merge.
- **Arbor owns** the optimization loop, inside the cards and the time MB gives it.

This page says which side runs each step of an MB session, and what passes between the two.
- **As of 2026-11-01:**
  - Hyperloom `jiaweche/Hyperloom` `arbor-adapter-v2` `cc5fc7dca`, merging into `multibranch-dev`;
  - Arbor `main` `1fa2069`.
- **Arbor's copy of this page:** [`docs/who-runs-what.md`](https://github.com/AMD-AGI/Arbor/blob/main/docs/who-runs-what.md).
- **The contract key by key, from Arbor's side:** [`docs/hosts.md`](https://github.com/AMD-AGI/Arbor/blob/main/docs/hosts.md).
- **The design and its decisions:** [`docs/component-mode.md`](https://github.com/AMD-AGI/Arbor/blob/main/docs/component-mode.md).

In the Hyperloom paths below, `src/hyperloom_multibranch/` is left out. Arbor's paths are in the Arbor repository.

## At a glance

| Concern | MB (Hyperloom) | Arbor |
|---|---|---|
| **Node, container, image** | Owns them. Starts Arbor on the host as its own user (`docker_container: "@host"`), after installing pyyaml and claude-agent-sdk. | Never calls docker. |
| **GPUs** | Leases `--tp` cards per Arbor from its pool. Sets `ROCR_VISIBLE_DEVICES` (absolute ids) and `HIP_VISIBLE_DEVICES`. Kills any process that reaches past the lease. | Serves on the cards it is given. Never sets or passes a mask. |
| **Ports** | A block of 10 per Arbor, from 31100. | Serves on the config's `port` only. |
| **Clock** | Owns the session's deadline. Gives each Arbor `component.deadline` = EXPLORE's end minus 40 min. | Starts an iteration only if it fits, and exits 0 `deadline`. |
| **Operating point** | One per session: `--conc`, ISL/OSL, and its gate's client. | Runs MB's client, `component.bench_command`, against its own server. |
| **Root and baseline** | PRELUDE: the root commit, the baseline tests, and the root profile (a TraceLens job). | None. |
| **The map** | Without `--agent-directions`, for two or more Arbors, or for one with the review on: the dossier pass, an Arbor run MB starts and reads. Without a map from it, `python3 -m arbor.directions generate`. Then `split`. Each Arbor gets its share as `component.plan`. | In the dossier pass: writes the dossier and `directions.json`, then exits `mapped`. In a branch: plans from its share. |
| **Human review** | On by default: pauses after the map step, before the split, holding no lease, until `hyperloom-multibranch review --session D approve`. `--no-review-gate` skips it. | Its gate at session start passes at once when the host reviewed (`component.review: "host"`). |
| **The optimization loop** | None. | Agent turns, its own server, the bench, GSM8K and the short-prompt check, and its own KEEP or REVERT. |
| **A KEEP** | Imports the snapshot, tests it on its own measurement pool, judges it, lands it on the branch and merges it into integration. | Writes an atomic snapshot to `keeps/NNNN/`. |
| **Numbers of record** | MB's toolbox: InferenceX or AgentX, through Magpie. | Its own measurements decide only its own KEEPs. |
| **Feedback** | Rewrites `component.notes` whenever integration moves. Reads the ledger's FACT and DEAD_END records as learnings. | Reads the notes at every iteration. Writes the ledger. |
| **Liveness** | Polls every 5 s. A stall is 30 min with no output, transcript or file write. Stops with SIGTERM, then SIGKILL after 30 s. | Writes a heartbeat every 5 min, exits within 25 s of SIGTERM, and records why it ended. |
| **LLM** | Passes its environment on: all of it locally; in a hosted sandbox, only the gateway, proxy and CA variables. | Its agents use that environment. |
| **FINAL** | Sweeps the integration head and the best branch head, and picks one. | None. |

## One session, step by step

1. **MB starts the session.**
   - **The run:** `hyperloom-multibranch run ... --autonomous-agent <Arbor checkout>`, once per Arbor.
   - **What counts as an Arbor checkout:** one with `arbor/component.py`, `arbor/__main__.py` and `docs/hosts.md`.
   - **Its copy:** each source is fetched once into `MB-agent-plans/sources/<i>`.
   - **Code:** `cli.py`, `scheduling/agent_plans.py`.
2. **MB runs PRELUDE.**
   - **The root:** it commits the root recipe and measures the baseline.
   - **The profile:** it starts the root profile, TraceLens job `prelude:<run id>`, without waiting for it.
   - **Code:** `scheduling/prelude.py`.
3. **MB reads the root's shares.** This and steps 4 and 5 happen only without `--agent-directions`: for two or more Arbors, or for one with the review on, which then takes the map whole.
   - **The wait:** up to 10 min, for the profile job.
   - **What it takes:** the root's op shares; or, when TraceLens degraded, the kernel shares from the job's result.
   - **Code:** `scheduling/root_shares.py`.
4. **The dossier pass.** It runs on MB's side and inside Arbor.
   - **MB sets it up:**
     - it leases `--tp` cards, charged to integration;
     - it syncs a worktree to the start commit;
     - it writes `hyperloom-mb/dossier.json` with `component.stop_at: "map"` and the shares as `component.profile`.
   - **Arbor:**
     - runs its dossier skill;
     - writes `docs/dossier/<name>/` and `<session>/directions.json`;
     - exits 0 with reason `mapped`.
   - **MB takes the map and the dossier,** then gives the cards back.
   - **Time:** up to `--agent-dossier-minutes` (90 by default), counted as EXPLORE's.
   - **Code:** `specialist/dossier_pass.py`.
5. **MB splits the map.**
   - **Without a map from the pass,** MB runs `python3 -m arbor.directions generate` in the Arbor checkout, on its own host.
   - **The split:** `split --shares N` writes `MB-agent-plans/shares/share-<k>.json`.
   - **Code:** `specialist/arbor_adapter.py`.
6. **MB pauses for a human,** unless `--no-review-gate`. It also pauses on a map given with `--agent-directions`. A run with `--stop-after-map` never pauses: it keeps the map and ends, for a review offline.
   - **The state:** `MB-review/review.json`.
   - **The decision:** `hyperloom-multibranch review --session D approve|reject`.
   - **Code:** `scheduling/review.py`.
7. **MB opens each Arbor branch.**
   - **What it prepares:**
     - a lease and a worktree at the start commit;
     - `link_built`, which links the installed framework's compiled files into the worktree;
     - the dossier, copied into the branch's own Arbor checkout;
     - the start tables: aiter rows laid over private copies of the image's tables;
     - the config `hyperloom-mb/config.json`.
   - **The start:** `python3 -m arbor run --config hyperloom-mb/config.json`. The process must stay up for 300 s.
   - **Code:** `specialist/autonomous_session.py`, `specialist/agent_onboarding.py`, `specialist/arbor_adapter.py`.
8. **Arbor runs.**
   - **Before the first iteration:** the review gate, then the init: it pins the validators and seeds the workspace.
   - **Each iteration:** an agent turn, its server and the bench, the correctness checks, and its own gate.
   - **A KEEP:** written as a snapshot (Arbor's `docs/hosts.md`, "What to read back").
   - **Code:** Arbor's `arbor/driver.py`, `arbor/component.py` (`snapshot_keep`).
9. **MB takes each KEEP.**
   - **The trigger:** a new `keeps/LATEST`.
   - **The import:** MB reads the snapshot, chained back to the root, and builds a delta: the patch increment, a config delta and the kernel tables as artifacts.
   - **The test:** on MB's measurement pool, at its operating point.
   - **The rule gate:** for InferenceX, a gain over the parent above max(0.6 %, 2× the spread), then a confirm on another node with accuracy.
   - **Landing:** a KEEP that passes is committed on the branch, and the integrator merges it into integration after its own test, confirm and regression set.
   - **Code:** `specialist/agent_import.py`, `specialist/judging.py`, `rulegate/`, `scheduling/integrator.py`.
10. **The end of a branch.**
    - **Arbor:** exits at its deadline.
    - **MB:**
      - keeps importing until the last KEEP has settled, then stops the branch;
      - at session end, stops any Arbor still running (SIGTERM, then SIGKILL after 30 s) and releases its lease.
    - **Code:** `scheduling/stops.py`, `specialist/agent_watch.py`.
11. **MB's FINAL.** MB sweeps 0.5x, 1x and 2x the concurrency for the integration head and the best branch head, and picks at the primary concurrency. Arbor plays no part. **Code:** `scheduling/final.py`.

## What passes between them

`<checkout>` is the branch's Arbor checkout, `MB-branches/<b>/session/agent/src`. `<session>` is
`<checkout>/hyperloom-mb/session`.

| File | Written by | Read by | What |
|---|---|---|---|
| `<checkout>/hyperloom-mb/config.json` | MB | Arbor | the config and its `component` block (Arbor's `docs/hosts.md`, "The config") |
| `<checkout>/hyperloom-mb/plan.json` | MB | Arbor | the branch's share of the map (`component.plan`) |
| `agent/mb-integration.md` | MB, whenever integration moves | Arbor, every iteration | `component.notes`: the integration head, its launch, the merged changes |
| `<session>/op_shares.json` | MB | Arbor | `component.profile`: the root's op shares |
| `<checkout>/docs/dossier/<name>/` | Arbor in the dossier pass; MB copies it into each branch | Arbor | the dossier |
| `<session>/directions.json` | Arbor | MB, from the dossier pass | the map |
| `<session>/keeps/LATEST`, `keeps/NNNN/` | Arbor | MB | the KEEP snapshots |
| `<session>/ledger.jsonl` | Arbor | MB, as learnings | FACT and DEAD_END records |
| `<session>/heartbeat.json` | Arbor | nobody by name; any file write counts for MB's stall watch | liveness |
| `<session>/component_exit.json`, `component_refused.json` | Arbor | MB | why the run ended, or why it never started |

## Exit codes, and what MB does

| Arbor exits | MB |
|---|---|
| 0: `done`, `deadline`, `map_exhausted`, `sigterm`, `aborted` or `review_rejected` | Reads the reason, learns the ledger, settles the pending KEEP, stops the branch. |
| 0 `mapped` (the dossier pass) | Takes the map and the dossier. |
| 2 (refused) | Never restarts it. The branch ends with the reasons from `component_refused.json`. |
| 3 (`unreachable`), or any other non-zero | Restarts it with the same command and the same deadline. It gives up after 3 crashes in 10 min, or 2 stalls in a row. |

## Known gaps (2026-11-01)

- **The review pauses unattended runs:** it is on by default. A batch run passes `--no-review-gate`, or a person
  approves from a login node.
- **An older MB under a newer Arbor:** an MB before `cc5fc7dca` doesn't write `component.review`. Its Arbors at
  `1fa2069` or later wait at Arbor's own gate, holding their leases.
- **One operating point per session:** several concurrencies means several MB sessions (ArborDevelopment #27).
- **A dropped launch-file flag stays:**
  - A flag set by the `--agent-launch` file, which a KEEP drops, is not removed on import.
  - A recipe's launch cannot record such a removal either (`boards/hyperloom-multibranch#168` c4583).
- **No profile on CUDA-graph runs:** a branch's `component.profile` takes op shares only. On a run where TraceLens degraded, the branches get none, though the dossier pass got kernel shares.
- **One Arbor without the review:** with `--no-review-gate` and no `--agent-directions`, a lone Arbor gets no
  dossier pass and makes its own dossier inside its branch.
