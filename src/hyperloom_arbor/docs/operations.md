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
