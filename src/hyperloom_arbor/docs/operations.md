# Operations

Running campaigns on clusters: the launchers, rolling chunks, parallel mode, dashboards and the
component-mode GPU test. The loop itself: [how-it-works.md](how-it-works.md).

## Before a launch

- **Arbor on the cluster.** Clusters often cannot read `AMD-AGI/Arbor` directly. Clone
  from a mirror you can read and point launchers at it with `ARBOR_URL`. **The launchers' defaults
  still name the retired repository and branches** (`<fork>/Arbor`, `<old branch>`,
  `<fork>/model-m`): always set `ARBOR_URL` and the branch.
- **Credentials.** The launchers source `~/.arbor/gateway.env` (mode 600) for the agents' gateway.
- **The image and weights.** Launchers load the serving image from an archive and check its ID, and
  check the weights as the container sees them.
- **Python.** The driver runs on the host with a Python that has PyYAML and `claude-agent-sdk`
  ([getting-started.md](getting-started.md)).

## Model M campaigns on Cluster A (`scripts/model-m-campaign/`)

Cluster A: Slurm partition `<partition>`, 4-hour jobs, shared `/apps`, node-local `/scratch`.

```bash
ARBOR_URL=<a clone URL Cluster A can read> CAMPAIGN_BRANCH=main \
  bash scripts/model-m-campaign/launch_model_m.sh <campaign> [ref]
```

`launch_model_m.sh` clones or fetches into `Arbor-<campaign>` at the ref and submits
`campaign_model_m.sbatch` (`CAMPAIGN=<campaign>`; config `configs/mi350x/model-m-c32-<campaign>.json`
unless `CAMPAIGN_CONFIG` names another). One job is one chunk:

1. load the image and start the container (`common.sh`);
2. lay the optional base tree (`base_vmu`) and check it file by file;
3. make the vLLM site a git repository with a baseline commit;
4. restore the session's `landed.patch`, `scratch.patch` and landed packages;
5. nothing to build for the profiler (TraceLens comes with the environment);
6. run `python3 -m arbor run --config ... --cooldown 30`.

Inputs: `MAX_CHUNKS` (12), `ROUND_AB` (1: the round-end A/B hook on), `ROUND_AB_POINTS`,
`CALIBRATE`, `PROFILES`. `round_ab.py` is the round-end A/B; `rebase_bar.py` rebases the bar on a
new node (dossier campaigns only). `plan_campaign.sh` runs a document campaign on the host, with no
GPU. The Cluster B variant (`launch_model_m_cluster_b.sh`, `campaign_model_m_cluster_b.sbatch`: Spur, `<account>`,
24 h) does not chain and has no `CAMPAIGN_CONFIG` or `base_vmu`.

## Rolling chunks (Cluster A)

A campaign longer than one 4-hour job moves from node to node (`ARBOR_ROLL=1`, set by the
launcher for the users in `CHUNK_ROLL_USERS`; `CHUNK_ROLL=1|0` forces it):

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

`scripts/cluster/roll_smoke.sh` rolls four short jobs as a smoke test (set its repository and ref).

## DSV4.1 campaigns (`scripts/dsv41-campaign/`)

The DeepSeek-V4.1-Flash campaign line, one sbatch per pack variant. These scripts write under
another user's `/apps` paths and were last changed on 2026-10-07; adapt one before reuse.

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
time limit, scripts). State lives under `<store>/<run_id>/`. Built on 2026-10-20 and tested on
fakes only; `configs/parallel/*.json` are examples, never launched.

## Dashboards and collectors

```bash
node arbor/dashboard/server.mjs local --host <host|local> --session <session>   # port 8766
node arbor/dashboard/server.mjs global                                            # port 8767
node arbor/dashboard/server.mjs parallel --host <host> --store <store>/<run_id>   # port 8768
```

The dashboard binds 127.0.0.1 and reads a session over ssh through `arbor/collect/realtime.py`
([arbor/dashboard/README.md](../arbor/dashboard/README.md)). The collectors
(`python3 -m arbor.collect realtime|project|knowledge`) write `<session>/collector/` and push to
`/shared/arbor`.

## Component mode's GPU test

`scripts/component/` holds a one-job test of component mode on a Cluster A node, as Hyperloom runs
Arbor: a lease of cards 4-7, a decoy server on 0-3, a docker shim, a deadline run and a SIGTERM
run, and a PASS/FAIL summary ([scripts/component/README.md](../scripts/component/README.md)).

## Trajectory traces

`.claude/settings.json` runs `evals/arbor_trace.py hook` on every Claude Code event in this
repository, and the loop marks each iteration; `evals/` turns the record into a session log and a
report ([evals/README.md](../evals/README.md)).
