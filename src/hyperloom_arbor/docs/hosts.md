# Running Arbor inside a host (component mode)

A host that owns the node, its GPUs and the clock (Hyperloom's multibranch today) can run Arbor as
a component: Arbor optimizes the worktree the host gives it, on the host's GPU lease and under the
host's deadline, and the host gates and merges what Arbor keeps. The design, the decisions and
what is built: [component-mode.md](component-mode.md). Tested on a GPU node with a stand-in host
([scripts/component/](../scripts/component/README.md)), and run inside Hyperloom's MB on Cluster A since
2026-10-30. Which side runs each step of an MB session: [who-runs-what.md](who-runs-what.md).

## The config

An Arbor campaign config (`pack`, `identity`, `goal`, `framework`, `model_path`, `model_name`,
`server_env`, `server_flags`, `bench_script`, `correctness_script`; [reference.md](reference.md)),
with:

```json
"mode": "component",
"docker_container": "@host",
"workspace_root": "/abs/path/to/the/worktree",
"session_root": "/abs/path/to/the/session",
"port": 8006,
"component": {
  "host": "hyperloom-mb",
  "deadline": 1793244000,
  "notes": "/abs/path/to/merged-notes.md",
  "bench_command": "python3 bench.py --base-url http://127.0.0.1:{port} --result-dir {out}",
  "outbox": "keeps",
  "root": "<the sha of the commit the worktree starts at>"
}
```

| Key | Meaning |
|---|---|
| `mode` | `"component"`. The default, `"standalone"`, is Arbor owning its own node. A session keeps the mode it started in. |
| `docker_container` | `"@host"`, required: Arbor runs the server and the bench where it runs, and never calls docker. |
| `workspace_root` | the worktree Arbor edits and commits in: a git repository at the commit the host gave (`component.root`). |
| `port` | the one port Arbor's server listens on. |
| `gpu_ids` | optional: the leased cards' physical numbers, a string or a list. |
| `directives` | optional: lines every agent reads in its program, at every iteration. The host's rules go here, each starting `Host rule`: what it holds fixed, what it refuses and how it judges a change. The dossier copies them into part 4's rules table (the dossier skill's `## The benchmark's rules`). |
| `component.host` | the host's name, for the log and the agents' rules. |
| `component.deadline` | when the run must be over: unix seconds, or ISO 8601 with a timezone (`"2026-10-29T03:20:00Z"`). |
| `component.seconds` | or a duration in seconds from the first launch; a resume does not restart it. With both, the earlier wins; `0` means none, so `deadline` decides. |
| `component.notes` | a file of what the host has already measured and merged. Read at every iteration; the agent never redoes what it lists (the first 20 KB go into its prompt). |
| `component.bench_command` | the host's benchmark client, run unchanged for every measurement, with `{port}` (Arbor's server) and `{out}` (an empty directory per replay, for its results) filled in. Without it, the pack's bench. Warm the server first: the first reading after a start reads low. The agent reads the pack's metric from the client's results, and its second metric too when the pack names one (`p90_interactivity` as 1000 / p90 TPOT in ms if the client reports latencies): without it Arbor's gate is one-axis. |
| `component.outbox` | where the KEEP snapshots go, relative to the session (default `keeps`). |
| `component.root` | the commit the host gave the worktree (a sha): every patch and snapshot is measured against it. Without it, the repository's first commit. A `root` the workspace lacks is refused. |
| `component.plan` | optional: this campaign's share of the host's directions, when the host runs several Arbors from one set ([below](#the-hosts-plan)). A list of directions, `{"directions": [...]}`, or the path of a JSON file holding either. |
| `component.stop_at` | optional, `"map"`: the host's dossier pass ([below](#the-dossier-pass)). The run writes the dossier and its map, and exits 0 with `mapped` once the map is set. Not with `plan`. |
| `component.profile` | optional: the path of a file holding the host's profile of the tree (PRELUDE's op shares for Hyperloom). Read into the agent's prompt at every iteration, like `notes`; the agent cites it and profiles only what it does not cover. |
| `component.review` | optional, `"host"`: the host reviews the campaign before it starts this run (MB pauses after its dossier pass, holding no lease), so Arbor's review gate passes at once and records that the host reviewed ([below](#launch-and-resume)). Without it, a new session waits for `python3 -m arbor review ... approve`. |
| `component.phase_minutes` | caps on each phase, in minutes: `agent` 90, `server` 45, `bench` 60, `critic` 15, `correctness` 30, `short_prompts` 10. An agent, server, bench or correctness phase cut off by its cap or the deadline leaves the iteration NOT_MEASURED; a critic cut off leaves the gate's verdict standing; a short-prompt probe cut off refuses nothing. |

**A deadline is required:** `deadline` or `seconds`. Arbor refuses to start (exit 2) without one, or
in any of these cases:
- without a `session_root`;
- with a `docker_container` other than `"@host"`;
- with `ARBOR_ROLL=1`;
- with a bad `component` block;
- with a config that does not load (JSON that does not parse, no `pack`, a missing or broken
  `pack.yaml`);
- with a campaign that does not validate (a missing `workspace_root`, `bench_script` or
  `correctness_script`, a broken pack, or a `results/` directory inside the pack without
  `--allow-pack-results` on the command line, [packs.md](packs.md));
- with an empty `server_flags`: the launch the host measures, which defaults are not;
- when the framework's package imports but its compiled extension does not (`vllm._C`,
  `sgl_kernel`), as in a source-only worktree: link or install the image's extension modules into
  it. It is checked with the environment Arbor starts in, the `server_env`'s `PYTHONPATH` over it;
  a framework that is not found at all is left to the server's launch to report.

The reasons go to stderr and to `<session>/component_refused.json` (`{reasons, ts}`), the only
file a refusal writes. The next start that is not refused removes it. **Exit 2 means refused and
nothing else does:** a crash never exits 2, so a host must not restart an exit 2, and must resume
any other non-zero exit. The host gives Arbor the GPU masks
(`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`) and the LLM gateway's
variables in its environment. A fresh session starts no iteration with less than about 63 minutes
left (its first estimate is 60 minutes, plus a 180 s margin).

## The host's plan

A host that runs several Arbors from one set of directions gives each its share in
`component.plan`. Each entry has the form of a `DIRECTIONS` line:

```json
"plan": [
  {"id": "m-attn", "kind": "increment", "title": "attention decode path", "rank": 1},
  {"id": "m-moe",  "kind": "increment", "title": "MoE GEMM tables", "rank": 2, "ceiling": "3%"}
]
```

`id` is a map ID (`m-<component>`), `kind` is `increment` or `rewrite`, and `title` is required;
`why`, `rank` and `ceiling` are optional. Every entry is open and every id unique, or the config is
refused. At the first start, Arbor seeds the session's map with these directions (a resume keeps
the map it has), and then:

- works these directions and nothing else; its dossier serves them, and a `DIRECTIONS` line of its
  own is refused;
- closes a direction only on evidence the harness checks; a done increment needs a KEEP that cites
  it;
- sends a direction it finds worth working, but not on its list, to the host: `proposals` in
  `<session>/directions.json`;
- stops (exit 0, `done`) when every direction is closed, writing `LANE_CLOSED.json`.

Without `component.plan`, Arbor maps its own directions from its dossier, as before.

## The dossier pass

The map a host splits comes from one Arbor run first, the dossier pass:
`"component": {"stop_at": "map", ...}`, on one lease, before the host opens its branches.

- **What it does:** it runs the pack's dossier (LEARN iterations: read, measure, profile) and writes
  the map. Its agent is told that this is the dossier pass, that it changes nothing in the served
  tree, and that each direction must be one Arbor's work. Once the map is set, the pass exits 0
  with reason `mapped`, before any iteration works the map. `component_exit.json`'s `detail` names
  the open directions, and the map is `<session>/directions.json`. A resume whose map is already
  set exits at once.
- **The host's profile:** give it with `component.profile`, so the pass profiles less. Without it,
  the pass profiles on its own lease.
- **What it needs:** map rounds, so a pack with the `dossier` skill (the serving template has it),
  a workload other than `document`, and `map_rounds` above 0. Otherwise the config is refused
  (exit 2).
- **When there is no map:** a pass that writes none before its deadline exits with `deadline`, or
  with `done` when the map step gives up (`map_fresh_max_iters`). Either way, no `mapped`.

**Splitting one map into shares:**

```bash
python3 -m arbor.directions split <map> --shares 2 --out <dir>   # <dir>/share-1.json, share-2.json
```

`<map>` is a session's `directions.json`, a JSON list or `{"directions": [...]}`, or a dossier's
part 11 (its `DIRECTIONS` list). Only open directions are split; closed ones are left out. Each
share is a valid `component.plan`: set `component.plan` to its path.

- **Dependencies:** directions joined by `depends_on` stay in one share. A dependency missing from
  a map counts as met, so split apart, one Arbor would start what another has not finished. A
  dependency on a closed direction is dropped.
- **Order:** the groups are dealt in rank order, each to the share with the fewest directions, so
  every share starts with high-ranked work.
- **Errors:** fewer independent groups than shares is an error (exit 1), as is a map that does
  not validate. Without `--out`, the shares go to stdout as `{"shares": [{"directions": [...]},
  ...]}`.

**A map without a dossier pass:**

```bash
python3 -m arbor.directions generate --profile <root profile> --out <map> [--min-share 2]
```

The map at once, with no GPU time, from the root profile a host already has after its baseline:
the file it gives Arbor as `component.profile`, `{"unit": "percent", "shares": {<op>: <share>}}`
(shares as fractions are read as such). Split it as any map.
- **The directions** are the profile's op families, as Hyperloom's MB reads them for its open areas:
  attention, linear attention, MoE, dense GEMM, collectives, sampling, and the small kernels (norm,
  RoPE, activation, elementwise) as one. Ops of no family are reported, not mapped.
- **Kernel names map too,** for a host whose profile has no op decomposition (MB's, when TraceLens
  degrades on a CUDA-graph trace, gives its kernel shares): hipBLASLt's `Cijk_` GEMMs, vLLM's
  `wvSplitK`, `reshape_and_cache` and `_fwd_kernel` are placed as well; copies and memcpys are not.
- **Each is framed by the v2 pack's roadmap** (`packs/agentx-dsv41-2`), its title naming the gap that
  roadmap names, model-independent.
- **Rank** follows the family's share, largest first. **Ceiling** is its Amdahl ceiling, the most
  the run could gain with the family's time at zero. A family under `--min-share` percent is left
  out.
- **Exit 1:** the profile cannot be used (unreadable, no shares, nothing above the cut), and nothing
  is written. A host then runs its Arbors without shares, as without a map.

The dossier pass (above) measures a better map; this one is generic and costs nothing.

## Launch and resume

```bash
python3 -m arbor run --config <cfg>
```

The same command resumes a session that stopped (a crash, a host restart): the session, the
deadline of the first launch and the KEEP numbering carry over. `touch <session>/STOP` ends the
run after the iteration in flight.

**The review gate is on by default here too** ([how-it-works.md](how-it-works.md#before-the-first-iteration-the-review-gate)).
A new session waits, before anything runs, for a human to run
`python3 -m arbor review --session <session> approve` (or `reject`). While it waits, the
deadline runs, the heartbeat stays alive with `phase: review`, SIGTERM is handled as in a run,
and a deadline reached while waiting exits 0 with `deadline` and the detail
`the deadline passed while awaiting review`. A rejection exits 0 with `review_rejected`. A host
that reviews the campaign itself before it starts Arbor sets `"review": "host"` in the
`component` block, and the gate passes at once (it holds no lease while a human reads). A host
with no human and no review of its own turns the gate off: `"review_gate": false` in the
config it writes, `ARBOR_REVIEW_GATE=0` in the launch environment, or `run --no-review-gate`.
A resumed session is never gated again.

## What to read back

- **The keep signal:** `<session>/session.json`, the entries of `lessons` with `"decision":
  "KEEP"`. Their count moves by one per KEEP.
- **The KEEP snapshot,** written whole before that count moves. `<session>/keeps/LATEST` holds the
  newest one's number (`0007`, no newline):

  ```
  keeps/0007/
    keep.json      number, previous (the KEEP before, null for the first), iteration,
                   metric, change, intent, root, parent, commit, carries, ts, tables,
                   tables_outside_workspace, outside (counts, when there are links),
                   short_prompts (verdict, sequential, batched, differ, compared, floor, reason),
                   left_out (each file over 10 MB kept out of both patches, with its size),
                   notes (when previous's commit is not in the workspace: change.patch
                   is then against root)
    change.patch   this KEEP alone: `git apply` at `parent` (the previous KEEP's commit, or root)
    served.patch   the whole served tree: `git apply` at `root`
    config.json    server_flags and server_env as measured (without GPU masks)
    tables/<VAR>/  every file an env var of that config names, its sha256 in keep.json;
                   a `:`-separated list (aiter's tables) as tables/<VAR>/<k>-<name>,
                   listed in order in keep.json's tables.<VAR>.files
    outside.json   when the tree links outside itself: each link's target, sha256 and size
    short_prompts.json  the short-prompt check, when it ran
  ```

  The patches hold the served tree only: no probes (`arbor/eval/`, `eval/` at the workspace's
  top), logs, bytecode or `arbor/config.json`, and no file over 10 MB (named in `left_out`). A
  `previous` that is not `number - 1` means a snapshot was not written; that `change.patch` then
  carries both KEEPs' changes.
- **What a KEEP carries besides its own change.**
  - **Why it happens:** an iteration that was not measured keeps its tree, so its change rides
    into the next KEEP. That KEEP's gate judges both together; nothing judged the earlier change
    on its own.
  - **`carries`** lists each such change since `parent`, oldest first:
    `{iteration, change, commit, verdict, reason}`, with the verdict and reason from session.json.
  - **What counts:** every `iter-N` commit that no revert undid and that changes the served tree
    or `arbor/config.json`. It is empty when the KEEP holds its own change alone.
  - **How to read it:** a non-empty `carries` means the KEEP's metric is the bundle's. Gate it as
    one, or measure the carried change apart before merging it.
- **A table variable given as a list.** A `:`-separated list, such as aiter's tables, is recorded
  as `tables.<VAR> = {path, files}`. `path` is the whole value. `files` holds one entry per part
  that is a file (`path`, `file`, `sha256`, `bytes`), in list order. Parts that are not files stay
  in `path` only. A single-file value keeps the plain shape (`path`, `file`, `sha256`, `bytes`).
  The agent is told to add to a list the host gave, never to replace it: it keeps every part in
  order and appends its own rows file. Hyperloom gives its start tables this way (#155
  c4101/c4102).
- **Binaries outside the tree are rebuilt, not shipped.** A tree that links to JIT modules carries
  the links only; `outside.json` says what each served. Start the server with the KEEP's
  `config.json` on the same image and GPU arch, and AITER rebuilds a missing module at its first
  call. Run `python3 -m arbor.execute.outside prepare --container @host --workspace W` first, so git
  never stages a rebuilt module, and `check --record outside.json` after.
- **The short-prompt check.** Each measured iteration probes its live server with 64 short prompts
  (1 to 64 tokens, 32 one at a time and 32 at once, greedy, 32 tokens), recorded in
  `iter_N/short_prompts.json`. At a KEEP, Arbor restarts its own server on the same tree, probes
  again and compares the two processes. Only the prompts sent one at a time gate: the first KEEP's
  count of them that differ is the session's floor, and above it the KEEP becomes REVERT. The
  batched half moves with the batch each boot happens to form, so it is recorded and refuses
  nothing. A check that cannot finish refuses nothing and says why in `keep.json`. Allow one more
  server boot per KEEP (about 5 minutes for a large model) in the deadline.
- **The live config:** `<workspace>/arbor/config.json`, `server_flags` and `server_env`.
- **The ledger:** `<session>/ledger.jsonl`; the `FACT` and `DEAD_END` records are what Arbor
  learned and ruled out.
- **The heartbeat:** `<session>/heartbeat.json` (`ts`, `epoch`, `pid`, `iteration`, `phase`), every
  5 minutes and at every phase change.
- **Why it ended:** `<session>/component_exit.json`, `{reason, iteration, keeps, map, deadline,
  ts}`, written before every exit 0 and exit 3.
- **Why it never started:** `<session>/component_refused.json`, `{reasons, ts}`, written before
  every exit 2 whose config names a session.

| Exit code | When | `reason` |
|---|---|---|
| 0 | the work is done: a STOP file, or `--max-iters` | `done` |
| 0 | the next iteration would not end before the deadline | `deadline` |
| 0 | the map is explored and its update round does not fit | `map_exhausted` |
| 0 | SIGTERM from the host: Arbor's own server stopped, the iteration in flight booked as not measured, out within 25 s | `sigterm` |
| 0 | a dossier pass (`stop_at: "map"`) has set its map | `mapped` |
| 0 | a human rejected the campaign at the review gate, before anything ran; `detail` is who and the note | `review_rejected` |
| 0 | STOP, or the deadline, while the review gate waited; `detail` says so | `done`, `deadline` |
| 0 | the harness stopped measuring (`ABORTED.md`: consecutive not-measured iterations, such as a bench that never ran); `detail` is its message with the last error. Read it as broken, not finished | `aborted` |
| 3 | the same, while the model could not be reached (`ABORTED.md` starts `# Model unreachable`): a gateway outage longer than the agents' retries. Resume it with the same command, as a crash; the resumed run moves that `ABORTED.md` aside and starts a fresh streak | `unreachable` |
| 2 | the config is refused (above): never restart it, show `component_refused.json` | none |
| other non-zero | a crash: resume with the same command | none |

A command line `arbor run` cannot parse (no `--config`) also exits 2, with a usage message and no
record.

## What Arbor does inside a host

- **With the pack's accuracy gate on and no `baseline_record`,** a new session first serves the root
  and runs the full GSM8K once, before iteration 1. The gate holds a numerics change to that score
  less `policy.accuracy_tolerance`, on Arbor's own eval.
  - **The cost:** one server boot and 1319 questions, within the deadline.
  - **Skipped** in a dossier pass, on a resume, or with under 30 minutes left; `GSM8K_MIN` is then the
    floor.
  - **The record:** `<session>/baseline_accuracy.json` (`score`, or `error`), and
    `root_accuracy` in iteration 1's row of `session.json`.
  - **A host whose rule allows a different drop** sets `policy.accuracy_tolerance` in the config.
- **Every KEEP is held to the full GSM8K** (since v0.9.1), not only a numerics change: a host imports
  what Arbor keeps, and a change nobody called numerics can still break the served model
  (boards/hyperloom-multibranch#155 c5075).
  - **Only an arm the gate would keep pays for it:** the gate decides without the eval first.
  - **Below the root's score less the tolerance** the KEEP is a REVERT. The result is in the KEEP's
    snapshot (`keep.json` `accuracy`, `by` naming "a KEEP in component mode").
  - **Only on a GSM8K workload:** the pack's accuracy gate is on (`serving`, `balanced`), and the
    launch does not simulate speculative acceptance. An AgentX launch does (`SGLANG_SIMULATE_ACC_*`, or
    vLLM's `rejection_sample_method: synthetic`), so GSM8K would score the simulation; neither the
    root's set nor any KEEP's runs there. A host can also turn it off with `policy.accuracy_gate:
    false`.
- **A KEEP that lands a switch that defaults off is served with it on** (S4, since v0.9.1;
  `arbor/measure/switches.py`).
  - **Which:** each `SWITCH: NAME=VALUE` the agent declares, and any new variable the change reads
    that the tree did not name, which the arm leaves unset. An undeclared one refuses the KEEP.
  - **Refused when:** the arm or the launch sets the switch on; the tree does not serve with it on;
    the accuracy check fails with it on (the full GSM8K on a GSM8K workload, the pack's correctness
    check on any other); a file appears in a JIT cache once that server is healthy; or its log has no
    line naming the switch from every rank.
  - **Not run** (no time left, an eval with no score): the KEEP stands, the switch off, as measured.
  - **The record:** `keep.json` `switches` (`verdict`, `reasons`, `accuracy`, `jit`, `dispatch`), and
    `<session>/switches.json`. A later change that turns the switch on is held to the accuracy gate.
  - **For a host that turns a kept switch on itself:** read `switches.verdict` first. Only `pass`
    means Arbor served it on and found it correct.
- It stops only the server it launched, by the PID file in `<session>/server/`: never a pattern
  `pkill`, never `kill -9`.
- It never sets or passes the GPU masks: its launch drops them from `server_env`.
- It never calls docker, and serves on its own port only.
- It keeps every file in the workspace or the session; a KEEP whose env names a file outside the
  workspace is flagged in its snapshot (`tables_outside_workspace`).
- It reads no target and runs no round-end A/B: the host gates and merges.
- An agent that cannot reach its model (a refused connection, a proxy error, a 5xx or an overloaded
  API) is started again with backoff, for up to 15 minutes per turn (`ARBOR_AGENT_CONNECT_WAIT_S`),
  before the iteration is booked not measured; a 4xx is not retried. Three not-measured iterations
  in a row end the run: `unreachable` (exit 3) when the last agent could not reach its model,
  else `aborted` (exit 0).
- **The LLM gateway must stay reachable for a run of several hours.** Arbor rides out an outage of
  up to 15 minutes inside one agent turn. A longer one costs not-measured iterations, and after
  three (at least 45 minutes of waiting) Arbor exits 3. A host that resumes non-zero exits, as MB
  does, restarts it, and the run carries on once the gateway answers. Each restart into a
  continuing outage takes as long again, so it never fills a crash window like MB's (3 in 10
  minutes); the deadline still ends the run. Sites that reach the gateway through a tunnel or a
  proxy should expect drops (Lux: 7 in four days, 2026-10).
