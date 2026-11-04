# Getting started

From a laptop dry run to a first campaign on a GPU node.

## What you need

- **Linux** (or WSL) and **Python 3.11 or newer**: the harness uses POSIX calls.
- **Python packages:** `pyyaml` (every command) and `claude-agent-sdk` (to run agents; it brings
  its own Claude CLI). Arbor has no requirements file yet:

  ```bash
  python3 -m pip install pyyaml claude-agent-sdk
  ```

  The SDK runs its bundled CLI, not one on PATH, and that CLI must know the model the config
  names: `claude-opus-5-5` needs CLI 2.1.280 or newer (SDK 0.2.174, CLI 2.1.310, is known
  good; 0.2.153, CLI 2.1.273, is too old). Check with
  `python3 -c "from claude_agent_sdk._cli_version import __cli_version__ as v; print(v)"`.

- **`git`** for everything. On a GPU node also `docker`, `curl` and `rocm-smi`; on a cluster, its
  scheduler (`sbatch`, `squeue`).
- **Credentials for the agents:** Claude, through your gateway. Put the gateway's variables
  (`ANTHROPIC_BASE_URL` and the token it needs) in the environment; the cluster launchers source
  them from `~/.arbor/gateway.env` (mode 600). Never commit them.
- **Access:** `AMD-AGI/Arbor` is private; ask its maintainers for access.

## 1. A dry run (no GPU)

```bash
git clone git@github.com:AMD-AGI/Arbor.git && cd Arbor
python3 -m arbor run --config configs/agentx-glm52-mi355x.json --dry-run
```

The dry run drives the real loop (the program, the gate, escalation, the records) on synthetic
bench numbers in a temporary git repository: no GPU, container, server or agent. It prints each
iteration's verdict and ends with a summary of KEEPs and REVERTs. Use it to check a new config or
pack before spending a node.

## 2. Choose a pack and write a config

- **The pack** says what is optimized and how it is measured ([packs.md](packs.md)). Start from the
  nearest existing one: `model-m-c8-tp4` (vLLM at TP4), `balanced` (every rule and five skills).
- **The config** says where: the pack, the container and the workspace inside it, the model, the
  server's flags and env, the bench and correctness scripts, and the session directory. Copy a
  config from `configs/` ([configs/README.md](../configs/README.md)) and change the paths;
  [reference.md](reference.md) lists every key.

## 3. Run on a GPU node

Arbor drives a container that has the serving stack (vLLM or SGLang), the model's weights and the
pack's validators. The served tree must be a git repository; the launchers make the installed
framework's directory one, with a baseline commit.

- **With a launcher** (recommended): the launchers provision the node (the image, the container,
  the workspace, a restore of earlier KEEPs) and start the driver. For Model M on Cluster A:

  ```bash
  ARBOR_URL=<a clone URL the cluster can read> CAMPAIGN_BRANCH=main \
    bash scripts/model-m-campaign/launch_model_m.sh <campaign>
  ```

  Set `ARBOR_URL` and the branch: the launchers' defaults still name retired repositories.
  [operations.md](operations.md) covers the launchers, rolling chunks and parallel mode.
- **By hand**, on a node whose container is up: `docker_container` names it, `workspace_root` is
  the served tree inside it, and

  ```bash
  python3 -m arbor run --config <cfg>
  ```

  from the Arbor checkout on the host. The same command resumes the session after a crash.

**Then review it.** A new campaign stops before it runs anything and prints where its
`REVIEW.md` is: what it will optimize, the bar, the rules, the roadmap and the program the agent
will read. Edit the pack, its roadmap or the config if needed, then, from another shell:

```bash
python3 -m arbor review --session <session_root> approve --note "reviewed"
```

The run reloads the files and starts. `reject` ends it instead. An unattended launch (a batch
job, a host) passes `--no-review-gate` or sets `ARBOR_REVIEW_GATE=0`
([how-it-works.md](how-it-works.md#before-the-first-iteration-the-review-gate)).

## 4. Watch it

- The driver's log: `[arbor] Gate: KEEP (B) — reason` lines, one per iteration.
- `<session>/session.json`: `lessons` has one row per verdict.
- The dashboard: `node arbor/dashboard/server.mjs local --host <host> --session <session>`
  ([operations.md](operations.md)).
- `touch <session>/STOP` stops after the iteration in flight.

## 5. What you get

- `<session>/landed.patch`: every kept change, as one patch against the workspace's first commit;
  apply it to a fresh tree with `git apply`.
- `<session>/ledger.jsonl`: what the agents learned and ruled out.
- `<session>/session.json`: the bars, the baseline and every verdict.

[how-it-works.md](how-it-works.md) explains the loop, the gate and the records.
