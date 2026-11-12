# Operations

Running campaigns on clusters: what a launcher does, rolling chunks, parallel mode, dashboards and
the component-mode GPU test. The loop itself: [how-it-works.md](how-it-works.md).

## Before a launch

- **Arbor on the cluster.** A cluster's nodes may not be able to read this repository directly.
  Clone from a mirror they can read, and point the launcher at it and at the branch to run.
- **Credentials.** The launchers source `~/.arbor/gateway.env` (mode 600) for the agents' gateway.
- **The image and weights.** Launchers load the serving image from an archive and check its ID, and
  check the weights as the container sees them.
- **Python.** The driver runs on the host with a Python that has PyYAML and `claude-agent-sdk`
  ([getting-started.md](getting-started.md)).

## What a launcher does

A launcher is specific to a cluster: its scheduler, partitions, paths and image. It clones or
fetches Arbor at a ref and submits a job with the campaign's config. One job is one chunk:

1. load the image and start the container, with the profiler's tools mounted;
2. lay the optional base tree (`base_vmu`) and check it file by file;
3. make the served framework's site a git repository with a baseline commit;
4. restore the session's `landed.patch`, `scratch.patch` and landed packages;
5. build the profiler's tracer (not fatal);
6. run `python3 -m arbor run --config ... --cooldown 30`.

A launcher's usual inputs are a cap on chunks, the round-end A/B hook (on or off, and its operating
points), a calibration and profiles. The round-end A/B is a launcher's own script, set through
`ARBOR_ROUND_AB` or the config's `round_ab.command`
([how-it-works.md](how-it-works.md)); so is a rebase of the bar on a new node (dossier campaigns
only). A document campaign runs on the host, with no GPU.

## Rolling chunks

On a cluster whose jobs have a short wall (the defaults assume four hours; `ARBOR_ROLL_WALL_MIN`
and its siblings in `arbor/cluster/handoff.py` change them), a campaign longer than one job moves
from node to node (`ARBOR_ROLL=1`, set by the launcher):

- at 180 minutes the running chunk submits its successor (no dependency, excluding its node and
  `chain/bad_nodes`), which provisions and waits (`handoff wait`, up to 90 minutes);
- the driver starts an iteration only if it ends before the switch, 15 minutes before the 240-minute
  wall, then hands off with exit 75 and a lease, so two drivers never share a session;
- exit 73 means the lease is held, 78 a bad roll config; any other exit ends the chain.

```bash
PYTHONPATH=<arbor> python3 -m arbor.cluster.handoff status --session <session>
touch <session>/chain/CHAIN_STOP   # this chunk finishes with no successor
touch <session>/STOP               # stop after the iteration in flight
```

## Parallel mode

Rounds of a map step, lanes and integration. Each lane is a campaign on its own node, seeded with
one direction or plan; at the round's end each lane's tree is merged onto the base with a same-node
A/B (B C C B), and the kept tree becomes the next base.

```bash
bash scripts/parallel/start_ctrl.sh <run.json> <unit>   # init, then the keeper as a user unit
python3 -m arbor.parallel status --config <run.json>
touch <store>/<run_id>/keeper/STOP
```

The run config (`arbor/parallel/config.py`) requires `run_id`, `store`, `arbor_root`,
`template_config`, `dossier_dir`, `gates` and `lane_sessions`; optional keys include `max_lanes`
(4), `max_rounds`, `no_keep_rounds` (2), `seed`, `base_vmu`, and `cluster` (partition, account,
time limit, scripts). State lives under `<store>/<run_id>/`. Tested on fakes only;
`configs/parallel/*.json` are examples, never launched.

## Checkpoints, pause and fork

A session writes a checkpoint at every pass boundary (`arbor/checkpoints.py`): after each pass and after a round end.
A failure is printed and never stops the run; `"checkpoints": false` in the config turns them off. On a checkpoint
you can continue a campaign on another node (`restore`), stop it cleanly and continue it later (`pause`), or start
new branches from it (`fork`). [quickstart-checkpoints.md](quickstart-checkpoints.md) walks through each.

**What a checkpoint holds** (`<session>/checkpoints/NNNN.json`, `arbor.checkpoint/2`, the checkpoint after iteration
NNNN):
- the session's files that are rewritten in place (`session.json`, `directions.json`, `stages.json`, ...), and the
  length of each append-only log (`ledger.jsonl`, `lifecycle.jsonl`);
- the served tree: patches against the root (`tree.patch`, `scratch.patch`) with the live `arbor/config.json` and
  `outside.json`. Unlike `landed.patch`, written at KEEPs only, they hold the change a NOT MEASURED iteration carries;
- the tree's history: the root commit and a Git bundle of `root..HEAD`, so a restore rebuilds the exact `HEAD`;
- the instructions: the campaign config with its credentials redacted, `pack.yaml`, and the agent's and critic's
  templates in use, with their digests;
- the dossier's files, which live in the Arbor checkout;
- a fingerprint (model, framework, image, operating point, pack, objective, the root commit's tree), the bars, the
  landed packages, and for a fork its parent.

**Immutable and verified.** Each file is stored once under `objects/<sha256>`, named by its digest. The manifest
carries `id`, the digest of its own content. `checkpoint verify` rechecks every object, the id, the format, and that
no captured text holds a credential-shaped value. `restore` and `fork` verify first and refuse a checkpoint that
fails. A served tree whose patches or live config hold a credential is not captured at all: pass credentials in the
environment, not in `arbor/config.json`.

**Named checkpoints** are frozen copies under `checkpoints/names/`, never moved: `checkpoint create --name`, the
pauses' `pause-NNNN`, and a fork's first checkpoint, `fork`. Wherever a command takes a checkpoint, `REF` is an
iteration, a name, or `latest`.

```bash
python3 -m arbor checkpoint list --config <cfg>                 # checkpoints, names, the session's lineage
python3 -m arbor checkpoint show --config <cfg> [REF]           # one manifest
python3 -m arbor checkpoint verify --config <cfg> [REF | --all]
python3 -m arbor checkpoint create --config <cfg> --name before-routing-work   # no driver may be running
```

### Continue on another node

Start the container from the same image, make its served tree a git repository with the image as its first commit,
then:

```bash
python3 -m arbor restore --config <cfg>      # the latest checkpoint; --checkpoint REF for another
python3 -m arbor run --config <cfg>          # carries on
```

`restore` rebuilds the checkpoint's exact `HEAD`, commit for commit, from its history, or lays the patches when it has
none. It checks the tree it built against the checkpoint's, byte for byte, and records each restore in
`restores.jsonl`. It also puts the dossier into this checkout when the checkout has none.

It refuses:
- another image (the root tree);
- a workspace that is not at its root (`--force` resets it);
- a checkpoint outside this Arbor's range;
- a checkpoint that fails `verify`;
- a session a driver holds.

### Pause and continue

```bash
python3 -m arbor pause --config <cfg> [--reason "try another prompt"]    # --cancel takes it back
```

`pause` writes `<session>/PAUSE`. At the next pass boundary, after the iteration in flight has its verdict, the
driver:
1. stops its server;
2. checks that the tree is committed;
3. writes the checkpoint `pause-NNNN`;
4. records `paused.json` and clears `PAUSE`;
5. exits **76** (`paused`), not 0, so a launcher can tell a pause from an end.

Under a host the exit record's reason is `paused`. To continue, run the same `arbor run --config <cfg>`, after
`restore` if the container is gone. No pause inside an iteration: `STOP` and a host's SIGTERM are unchanged.

### Fork

```bash
python3 -m arbor checkpoint fork --from <parent cfg | session> [--checkpoint REF] --config <fork cfg> \
    [--branch prompt-a] [--prompt-profile <dir>] \
    [--fork-policy exploratory --directions <directions.json> --reset null_streak ...] [--no-restore]
```

The fork's config names its own `session_root`, container and workspace, and its own `dossier`. A fork refuses to
share any of them with the parent; `--force` overrides the workspace and the dossier.

The fork:
- **copies the parent's session as of the checkpoint:** the state files, the logs cut back to their lengths, the
  iterations up to it, and the other evidence written before it. It copies; it never links.
- **leaves out the transient files:** `STOP`, `PAUSE`, `ABORTED.md` and the other stop markers, the heartbeat, the
  exit records, `clock.json` (the fork has a clock of its own), the chain lease, the server records and the parent's
  `checkpoints/`.
- **writes its lineage** to `fork.json`: the branch id, and the parent's session, checkpoint, id, identity and
  iteration. It also writes the policy and every change, the prompts' digest, and a `NOTE` row on the history.
- **writes its own checkpoint `fork`.**
- **restores the tree** into its workspace, unless `--no-restore`, in which case run `arbor restore` later.

Then `arbor run --config <fork cfg>` starts at the parent's next iteration.

- **`exact` (the default):** the bars, the counters, the map and the iteration are the parent's. Only the prompt
  profile may change.
- **`exploratory`:** may also replace the map (`--directions`) and set scheduling counters to 0 (`--reset`:
  `null_streak`, `probe_streak`, `tier_iters`, `barren_streak`, `secondary_only_streak`, `invalid_streak`, or
  `all`). Each change is recorded.

Branches never update one another. To compare two, run a same-node A/B of their trees; to combine them, integrate
explicitly and measure again.

### Prompts as inputs

Each role reads one template, the first of:
1. the config's `prompt_overrides` (`{"main": FILE, "critic": FILE}`);
2. the config's `prompt_profile` (a directory with `claude_loop_prompt.md` and/or `critic_prompt.md`, the built-ins'
   names);
3. `<session>/prompt_profile/` (a fork's `--prompt-profile`, copied there);
4. the built-in `arbor/policy/`.

A file the config names that is missing refuses the start.

Every lesson row records `provenance`: `prompt_digest`, `pack_digest`, `config_digest` and `arbor_commit`. A verdict
after a fork or a prompt change names the instructions it ran under.

### Which Arbor continues which session

`python3 -m arbor version --json` gives `interfaces.session`, the format this Arbor writes (2), and
`continues.session`, the range it continues ([0, 2]). A session's format is its latest checkpoint's, or 0 without
one. `run`, `master` and `worker` refuse (exit 2) a session outside the range.

**The session lock** (`<session>/session.lock`): a driver holds it shared. `checkpoint create`, `restore` and `fork`
take it exclusive, so they refuse while a driver runs on that session.

**Next:** a replay of the logs, to check a checkpoint and to fork from an iteration with none.

## Dashboards and collectors

```bash
node arbor/dashboard/server.mjs local --host <host|local> --session <session>   # port 8766
node arbor/dashboard/server.mjs global                                            # port 8767
node arbor/dashboard/server.mjs parallel --host <host> --store <store>/<run_id>   # port 8768
```

The dashboard binds 127.0.0.1 and reads a session over ssh through `arbor/collect/realtime.py`
([arbor/dashboard/README.md](../arbor/dashboard/README.md)). The collectors
(`python3 -m arbor.collect realtime|project|knowledge`) write `<session>/collector/` and push to a
shared directory (`ARBOR_SHARE_HOST`, `ARBOR_SHARE_ROOT`: [reference.md](reference.md)).

## Component mode's GPU test

`scripts/component/` holds a one-job test of component mode on one 8-GPU node, as Hyperloom runs
Arbor: a lease of cards 4-7, a decoy server on 0-3, a docker shim, a deadline run and a SIGTERM
run, and a PASS/FAIL summary ([scripts/component/README.md](../scripts/component/README.md)).

## Trajectory traces

`.claude/settings.json` runs `evals/arbor_trace.py hook` on every Claude Code event in this
repository, and the loop marks each iteration; `evals/` turns the record into a session log and a
report ([evals/README.md](../evals/README.md)).
