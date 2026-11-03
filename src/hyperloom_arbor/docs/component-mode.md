# Component mode: Arbor inside a host

Status: **slices 1 to 3 built** (CORE-02, 2026-10-28; decisions in §7, as built in §8), and run
on a GPU node (Cluster A jobs 102861 and 102998, `scripts/component/`).

Arbor has run one way so far, **standalone**: it owns a node, its container and its clock. Its
launchers recreate the container, its harness stops any server it finds and waits for every GPU
to empty, it rolls from node to node, ends rounds with its own same-node A/B and rebases its bar.

**Component mode** is Arbor run by a host that owns all of that: Hyperloom's multibranch (MB)
today, which runs Arbor as an "autonomous branch" on a lease of GPUs, under its own deadline, and
gates and merges what Arbor keeps. MB already runs Arbor this way (its board topics #100, #126,
#148), with a sandbox built around what standalone Arbor did wrong inside it: stopping MB's
servers with a pattern `pkill`, running on cards outside its lease, iterating an hour after the
session died, hanging for 511 minutes, loading kernel tables from paths outside the workspace.
Component mode makes Arbor do the right thing itself.

Which side runs each step of an MB session, and the files between the two:
[who-runs-what.md](who-runs-what.md).

## 1. Choosing the mode

`"mode": "component"` in the campaign config, beside a `"component"` block; the default is
`"standalone"`, and a standalone config is unchanged. The mode is read once in `load_campaign`,
recorded in `session.json`, and a resumed session keeps it.

```json
"mode": "component",
"component": {
  "host": "hyperloom-mb",
  "deadline": "2026-10-29T03:20:00Z",
  "seconds": 25200,
  "notes": "/abs/path/mb-integration.md",
  "bench_command": "python3 benchmark_serving.py ... --base-url http://127.0.0.1:{port} --result-dir {out}",
  "outbox": "keeps"
}
```

- **`deadline`** (absolute) or **`seconds`** (a duration from the first launch). MB passes a
  duration today, frozen when it onboards Arbor; Arbor turns it into an absolute deadline at the
  first launch and keeps that in `session.json`, so a resume does not restart the clock.
- **`notes`**: what the host has already measured and merged. Read at every iteration; the agent
  never redoes what it lists.
- **`bench_command`**: the host's own benchmark client. When given, every bench and A/B uses it,
  so Arbor ranks candidates the way the host's gate does (MB: "a client of the agent's own sends
  other prompts, which route MoE tokens otherwise").
- **`outbox`**: where each KEEP's snapshot goes (§4), relative to the session.
- `docker_container` is `"@host"`: the host runs Arbor where the workspace is, and Arbor never
  calls docker.

## 2. What changes, by behaviour

| Behaviour | Standalone (today) | Component |
|---|---|---|
| GPUs | the launcher's container sees the cards; config `server_env` may set ROCr masks | the host sets the masks; Arbor never sets or passes `ROCR_/HIP_/CUDA_VISIBLE_DEVICES` and drops them from `server_env` when it launches |
| Stopping a server | `pkill -TERM -f` every framework's patterns in the container | stop the server Arbor launched, by the PID the launch wrote, and its children; no pattern |
| GPU memory | wait until every GPU of `gpu_ids` is under 10 GB; anything held counts as ours and is killed | no memory reading and no kill for memory: the cards are the host's; only the server Arbor launched is stopped, by its PID file |
| Containers | `docker exec`, `docker cp`, `docker inspect`, `docker run --rm` from Python | none: the serving path runs on `@host` |
| Ports and logs | one port, `/tmp/server.log` | the config's port; the log and the PID file in the session |
| Clock | `time_budget_minutes` from process start, checked between iterations | the absolute deadline: an iteration starts only if its estimated length (the rolling handoff's estimator) fits; every phase has a timeout that ends at the deadline at the latest |
| Signals | none handled; servers outlive the driver | SIGTERM stops Arbor's own server, records the iteration as not measured, and exits within the host's grace (MB: 30 s) |
| Watchdog | none | each phase (agent turn, server start, bench) has a timeout; a heartbeat file every 5 minutes, so a host's stall check sees a live run |
| Exit | many stop files | exit 0 when done or out of time, after writing `component_exit.json` with the reason (§7); exit 3, with the record too, when the not-measured streak ends the run while the model cannot be reached (`unreachable`), so the host resumes it; exit 2 when refusing a config; any other non-zero only for a crash. MB ends the branch on exit 0 and resumes it otherwise |
| Review gate | a new session waits for `arbor review ... approve` before anything runs (on by default) | the same, on the host's clock: heartbeat `phase: review`, SIGTERM handled, exit 0 with `deadline` if the deadline passes while waiting, `review_rejected` on a rejection; a host that reviews first sets `component.review: "host"` and the gate passes at once, a host with no human sets `review_gate: false` ([hosts.md](hosts.md#launch-and-resume)) |
| Rolling chunks | `ARBOR_ROLL=1` | refused |
| Target, round ends | target exits or opens a round-end A/B; `round_ab` hook | both ignored, with a log line: the host gates and merges |
| Rebased bar, calibration | the launcher's | not used: the host's bar |
| Map rounds | on with the dossier skill | unchanged: Arbor still plans from its dossier; an exhausted map starts the update round |
| A KEEP | one commit; `landed.patch` is cumulative and also carries PROBE and unmeasured commits; config edits after the commit miss it | an atomic snapshot per KEEP in the outbox (§4), written before the counter the host polls moves |
| Kernel tables | any path | under the workspace; a KEEP whose env points at a table outside it is flagged (the host loads only what is under the worktree) |
| Correctness | GSM8K | GSM8K plus a short-prompt check: two processes of the same tree at each KEEP (slice 3, §8) |

## 3. What the host gives, and what it reads back

**Gives:** the config (with the block above); a checkout to run from and a workspace (the
framework tree) to edit; the GPU masks in the environment; a port; a deadline; optionally its
notes and bench command; the LLM gateway environment.

**Reads back:**
- `session.json`: the `lessons` with `"decision": "KEEP"`, whose count is the keep signal (MB
  polls it every 5 s);
- `<workspace>/arbor/config.json`: `server_flags` and `server_env`;
- `ledger.jsonl`: FACT and DEAD_END records, as learnings;
- the outbox's snapshots (§4), when the host reads them instead of the live workspace;
- the exit code, and the heartbeat.

## 4. The KEEP snapshot

On a KEEP, before `session.json` records it:

```
<session>/keeps/0007/          written to keeps/.0007.tmp, then renamed
  keep.json                    number, previous, iteration, metric, change, intent,
                               root, parent, commit, ts
  change.patch                 this KEEP alone: applies with `git apply` at `parent`
  served.patch                 the served tree: applies with `git apply` at `root`
  config.json                  server_flags and server_env after the KEEP
  tables/                      every file an env var of the config points at, by name
  outside.json                 the links the tree makes outside itself (arbor/execute/outside.py)
<session>/keeps/LATEST         "0007", replaced atomically
```

- `previous`: the number of the KEEP before this one, or `null` for the first, so a host tells
  a skipped snapshot from a gap after a crash (MB's request).
- `root`: the commit the host gave the workspace (`component.root`, else the repository's first
  commit); `parent`: the tree this KEEP was made on, the previous KEEP's or `root` for the
  first. Both patches use the served pathspec (`workspace.landed_pathspec`): no probes, bytecode or `arbor/config.json`.

A host that reads the snapshot never sees a half-written keep or the next candidate's edits. It
is an addition: MB's live-workspace import keeps working for agents without a snapshot. MB
agreed to read `keeps/LATEST` when it exists, take `change.patch` as the commit's change,
`config.json` as its launch and `tables/` as kernel artifacts.

## 5. The docs a host's onboarding reads

MB's onboarding model reads only Arbor's README and docs to write its manifest. The README gets
a short "Running Arbor inside a host" section: the config block, the launch and resume commands,
the keep signal, the config source, the ledger, and the rules above. That shapes every future
manifest without either side writing host-specific code.

## 6. Slices

1. **The switch and the safety:** the mode and its validation; the deadline, the iteration
   estimate, phase timeouts, SIGTERM; stopping by PID; no masks, no memory kills, no docker on
   `@host`; the heartbeat; target, round ends and rolling refused or ignored. Tested on `@host`
   fakes; then once on a node with a lease of 4 cards and something else on the other 4.
2. **The KEEP snapshot,** the notes and the bench command in the prompts, the tables check.
3. **The short-prompt check** (built 2026-10-28; the README section came with slice 2).

## 7. Decisions (operator, 2026-10-28)

1. **An exhausted map with no time left for its update round: exit 0, and say why.** Every exit
   in component mode writes `<session>/component_exit.json` first: `{reason, iteration, keeps,
   map, deadline, ts}`, with `reason` one of `deadline`, `map_exhausted` (the map is closed and
   its update round does not fit), `done` (STOP, max iterations), `sigterm`, and since 2026-10-30
   `aborted` (the harness stopped measuring: `ABORTED.md`, its message as `detail`), and since
   2026-10-31 `review_rejected` (a human rejected the campaign at the review gate); the same line
   goes to the log and the heartbeat. A host reads why Arbor ended instead of guessing from an exit code.
   **One exception exits non-zero** (operator, 2026-10-29, on Cluster C's gateway outages): `unreachable`, a streak that ended while the model could not be reached, exits 3 after
   writing its record. The host resumes it as it does a crash, instead of retiring the branch; the
   resumed run moves that `ABORTED.md` aside and starts a fresh not-measured streak.
2. **The snapshot format is proposed to MB now,** on the board; slice 2 starts after slice 1 and
   at least an hour after the proposal, whether or not they have answered, and changes in flight
   if they object.
3. **Component mode refuses to start without a deadline** (`deadline` or `seconds`), and refuses
   a `docker_container` other than `"@host"` and `ARBOR_ROLL=1`: exit 2, before it touches the
   session, with the reason on stderr.

## 8. As built (slice 1, 2026-10-28)

The switch and the safety are in `arbor/component.py` and the sites the table names. Where the
build settled what the design left open:
- **The deadline:** `deadline` as unix seconds or ISO 8601 with a timezone (MB passes unix
  seconds); with both `deadline` and `seconds`, the earlier wins; 180 s are kept
  in hand before it, by the iteration gate and the phase timeouts alike. `component.json` keeps
  it across resumes; without that file the clock falls back to session.json's `start_epoch`,
  never to now. A session recorded in one mode and resumed in the other exits 2.
- **Exit 2 is refusal only** (2026-10-29, for the 11-04 release: MB
  resumed an exit 2 like a crash). Two more ways of exiting 2 were added, so that no config a
  restart cannot fix ever crashes: a config `load_campaign` cannot load (unparsable JSON, no
  `pack`, a missing or broken `pack.yaml`), and in component mode a campaign that
  `validate_campaign` rejects, or one without a `session_root`. Read errors that may pass (an
  NFS hiccup) stay crashes. Every component-mode refusal also writes
  `<session>/component_refused.json` (`{reasons, ts}`), the one file a refusal writes, for the
  host to show; `Run.start` removes a stale one, as it does `component_exit.json`.
- **Timed-out phases:** the agent, server and bench phases make the iteration NOT_MEASURED; the
  agent's uncommitted edits are discarded, so a host never reads a half-written
  `arbor/config.json`. A timed-out critic leaves the gate's verdict standing, as a crashed one
  does: it could only downgrade, and the host gates every KEEP again.
- **SIGTERM:** the exit record is written and our server's group signalled at once, the server
  is given 15 s, and a 25 s watchdog ends the process if unwinding stalls: inside MB's 30 s grace.
- **A crash** (a non-zero exit other than 2 and 3) writes no exit record; our server still gets SIGTERM first.
- **The stop** leaves a PID alone that started after the PID file was written or leads no group:
  it was reused.
- Standalone records `"mode": "standalone"` in session.json; its commands, prompts and stops are
  byte for byte as before.

### Slice 2 (2026-10-28)

The KEEP snapshot (§4, in the layout MB agreed), the tables check, the host's notes and bench
command in the prompts (`arbor/component.py`, `arbor/policy/claude_loop.py`), and the README's
"Running Arbor inside a host" section (§5). Where the build settled what the design left open:
- **`number`** is the KEEP count session.json reaches with this KEEP, so `keeps/N` is the
  snapshot of the N-th KEEP a host counts. `previous` and `parent` come from the outbox itself
  (the newest snapshot numbered below), so a resume continues the chain.
- **A snapshot that was never recorded** (the run stopped between the snapshot and
  session.json): the next KEEP takes its number, moves it aside as `.NNNN.unrecorded.*` and
  chains to the last recorded one; its change, if the tree kept it, is in the next
  `change.patch`. A snapshot that failed leaves a gap `previous` shows, and the next
  `change.patch` carries both KEEPs.
- **LATEST** holds the four digits with no newline.
- **`config.json`** is the live `arbor/config.json` at the KEEP (the server and bench agents'
  edits included), without the GPU masks the launch drops.
- **Tables:** an env var's value is a table when it is an absolute path to an existing regular
  file; one over 512 MiB is listed with `skipped`, not copied. `tables_outside_workspace` is
  `{VAR: path}`, symlinks resolved.
- **Notes** go into the main agent's prompt before the host's rules, which stay last; the first
  20 KB, with a line naming the file for the rest. The server and bench agents do not get them.
- **`bench_command`:** `{out}` is a directory per replay (`iter_N/host_bench/`, or `run<i>/`
  under it), made empty by the harness; the bench agent reads the metric from it into
  `bench_output.json` with a fixed one-line command, and with `bench_repeats` the fixed mean
  block aggregates, as with the pack's bench. Since 2026-10-30 the same command also takes the
  pack's `secondary_metric` when the client gives it; before, the gate saw only the primary
  (a Cluster C job: `secondary: None` on every KEEP).
- **`component.root`** (MB's request): MB's worktree is at the branch's start
  commit with the experiment repository's history behind it, so the first commit is not what MB
  gave. A sha in `component.root` is the base of `served.patch`, the first `change.patch`,
  `keep.json`'s `root`, `landed.patch` and `scratch.patch`
  (`workspace.base_commit`); one the workspace lacks is refused at start (exit 2) and malformed
  is a config error. Without it, the first commit stays the base, for standalone use.

### Slice 3 (2026-10-28): the short-prompt check

Decided by the operator (2026-10-28): a second launch at each KEEP, component mode only.
`arbor/measure/short_prompts.py`; the probe after correctness and the check after the critic in
`arbor/policy/claude_loop.py`. Where the build settled what the design left open:
- **What it answers:** MB's #126. AITER's vectorized_5d `mha_batch_prefill` gave wrong output on
  short sequences in about one server process in three, and GSM8K did not see it. The design's
  "against the base server" was dropped: MB compared with the parent first and stopped
  (Hyperloom `b315477e5`), because exact greedy matching fires on the largest gains (fp8 KV,
  quantized GEMMs). A defect wrong in every process is one GSM8K already sees, so the check
  compares two processes of the same tree.
- **The probe** has MB's shape (Hyperloom `hl-short-prompts/1`'s lengths), so the two floors
  measure the same thing: 64 prompts of 1 to 64 tokens, each starting with a token of its own,
  32 sent one at a time and 32 at once; `/v1/completions` with token ids, temperature 0,
  32 tokens, `logprobs` 1 and `return_tokens_as_token_ids`. Compared by token id when every
  answer has them, by text otherwise. The prompts are built once per session through the
  server's `/tokenize` (`<session>/short_prompts/prompts.json`), so every process gets the same
  ids. It runs after a passed correctness check on every measured iteration's live server,
  under the `short_prompts` cap (10 minutes) and the deadline, and is recorded only.
- **At a KEEP** (after the critic, before anything is booked): the harness stops its own server
  (by its process group), boots the same tree with `_start_server` (no agent turn), waits for
  health under the `server` cap, and probes again (`iter_N/short_prompts_2.json`). The
  session's first KEEP sets the floor (`<session>/short_prompts/floor.json`); a later KEEP whose
  processes differ on more prompts sent one at a time than the floor becomes REVERT, with the
  first differing positions in the reason. The batched half is recorded and refuses nothing
  (operator, 2026-10-28, from the GPU run, Cluster A job 102998): an unchanged tree's two boots
  differed on 0 of the 32 prompts sent one at a time each time, and on 23, 19 and 20 of the 32
  batched ones in their first tokens, because which prompts share a prefill batch changes
  between boots and that flips near-tied greedy picks; pooled, that noise hid a small
  single-request defect and could refuse a good KEEP. The second launch's server log is kept
  as `iter_N/server_2.log`. The record is `iter_N/short_prompts_check.json`; `keep.json` gets its
  brief, and a snapshot carries the record when the check ran.
- **What refuses nothing:** an iteration probe that did not finish (no boot is spent then), a
  second boot that does not come up, no time left, a second probe that does not finish. The
  KEEP stands and `keep.json` says why, as MB's check does; MB re-gates every KEEP.
- **Odds and cost:** two processes catch a one-in-three defect 4 times in 9 per KEEP, and every
  later KEEP that carries it is another chance. The cost is one server boot and two probes per
  KEEP (about 5 minutes on Model M at TP4), which a host should allow for in its deadline.

### The host's plan (2026-10-29)

`component.plan` (release of 2026-11-04): a host
running several Arbors from one set of directions gives each its share. `component.plan_entries`
validates it with the same check a `DIRECTIONS` line gets (`directions._entry`); every entry
open, ids unique. `component.seed_plan`, called from `Run.start`, seeds the session's map once as a
lane (`scope: lane`, `lane.kind: host-plan`), reusing parallel mode's lane semantics
(`arbor/directions.py`): no bootstrap or update round, a `DIRECTIONS` restatement refused, the
agent's additions to `proposals` for the host, closes on checked evidence, and `LANE_CLOSED.json`
and a STOP when all are closed, so the run exits 0 with `done`. `stages.render_host_plan` tells the
agent it works its host's share and nothing else. Standalone ignores the key.

`python3 -m arbor.directions split MAP --shares N` (`directions.split`) makes those shares from
one map: its open directions, disjoint, each a valid `component.plan`. Directions joined by
`depends_on` stay together, since `deps_met` counts a dependency missing from a map as met; the
groups are dealt in rank order to the share with the fewest directions ([hosts.md](hosts.md)).

### The dossier pass (2026-10-29)

Agreed with MB (the dossier question, (b) with (c)). Before its branches
open, the host runs one Arbor as a dossier pass, splits that pass's map, and gives each branch a
share; directions an Arbor finds outside its share come back as `proposals`.

- **`component.stop_at: "map"`:** the run ends once its map is set.
  - **The check:** `component.map_ready`, called by the driver before each iteration. A pending
    map (the bootstrap's `DIRECTIONS` list) is first promoted with `directions.advance`, as the
    next iteration's begin would promote it.
  - **The exit:** 0 with reason `mapped`; the exit record's `detail` lists the open directions.
  - **What it needs:** map rounds (`directions.enabled`: the dossier skill, a workload that is
    not a document, `map_rounds` above 0), or the config is refused. A `plan` beside it is a
    config error.
- **`component.profile`:** the host's profile file, which MB writes from the root's `OpShares`.
  It goes into the prompt at every iteration through the same capped reader as
  the notes (`_read_head`), and the agent profiles only what it does not cover. When MB has no
  shares (GLM-5.3 AgentX: its head profile is refused as too large), the key is left out.
- **The agent is told what the pass is for** (`dossier_pass_block`): LEARN only, every direction
  one Arbor's work, `depends_on` only where needed (joined directions go to one Arbor), every
  direction ranked.
  - The prompt is the only guard: the gate does not refuse a change in a pass.
  - The host ignores a pass's snapshots: it splits the map and does not import the KEEPs.
