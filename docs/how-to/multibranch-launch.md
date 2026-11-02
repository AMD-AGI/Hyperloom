---
myst:
    html_meta:
        "description": "Run several multi-branch sessions on one machine, one per operating point, with hyperloom-multibranch-launch: the cards, ports and containers each point gets, its review, and how to follow and stop them."
        "keywords": "Hyperloom, multi-branch, MB, operating point, concurrency, launcher, containers, Arbor, review"
---
# Running several operating points on one machine

A multi-branch session (MB) has one operating point, its `--conc`: every gate, merge and FINAL measures there. To
optimize C8 and C64, run two sessions. `hyperloom-multibranch-launch` starts them on one machine. Each point gets one
ordinary `hyperloom-multibranch run` with:

- its own share of the cards;
- its own server port and Arbor port base;
- by default, its own container.

The launcher then waits for them all. Each session runs its own PRELUDE, maps and reviews its own Arbor directions,
and ends with its own recipe. Nothing is merged across the points.

## Before you start

- **Magpie with the per-session changes.** Magpie serves every session on port 8888 and picks its cards from the
  whole machine, unless it has the two changes of `boards/hyperloom-multibranch#165`:
  - its GPU selection keeps to the inherited mask;
  - its server port comes from `PORT`.

  Until upstream Magpie has them, install a Magpie that does (`MAGPIE_REPO`, `MAGPIE_REF`) before running several
  points on one machine.
- **Shared storage at the same path.** The launcher reads each session's directory, as the session prints it, to
  show its state and its review. Mount the storage `USER_DATA_PATH` names at the same path in the containers
  (`--mount`).
- **The machine's own Python, 3.10 or later.** The launcher uses only the standard library. Run it on the machine,
  outside the image: `python3 -m hyperloom_multibranch.launch` with `PYTHONPATH=<checkout>/src`, or the
  `hyperloom-multibranch-launch` entry point of an installed Hyperloom.

## Launch

Give the points, a new directory for the launch, and the image. After `--`, give the run arguments every point
shares:

```bash
hyperloom-multibranch-launch --point 8 --point 64 --point-hours 8=3.0 --point-hours 64=3.25 \
  --out /shared/launches/qwen-c8-c64 \
  --image rocm/vllm:<tag> --mount /shared --workdir /shared/hyperloom \
  --prepare 'bash src/hyperloom/inference_optimizer/assets/install.sh' \
  --setup '. /shared/env.sh' \
  -- --workload inferencex --model /shared/models/Qwen3-8B --framework vllm --tp 1 --isl 1024 --osl 1024 \
     --precision bf16 --gpu-type mi355x --max-hours 3.0 --autonomous-agent /shared/arbor
```

The launcher sets `--conc` and `--agent-port` for each point and refuses them after `--`. It also refuses, before
starting anything:

- run arguments with no `--tp`;
- points the cards cannot hold;
- a directory that is not empty;
- several points without `--image` (below).

`--dry-run` prints every command each point would run, and starts nothing.

## What each point gets

| | Point 1 | Point 2 | Point *n* |
| --- | --- | --- | --- |
| **Cards** | the first share of `--cards` (default: every card; an equal share unless `--cards-per-point`) | the next | ... |
| **Server port** (`PORT`) | 8888 | 8988 | 100 on per point |
| **Arbor port base** (`--agent-port`) | 31100 | 31200 | 100 on per point |
| **`--max-hours`** | `--point-hours` for that conc, else the run arguments' | | |

Each session is also given:

- **its cards** as `ROCR_VISIBLE_DEVICES`, for the session alone;
- **`INFERENCE_OPTIMIZER_RAY_EXEC=0`**: no Ray. One Ray head per container would collide on the machine's network,
  and one shared head has a single serving slot for the machine.

## Containers

Each point runs in a container of its own: Magpie ends a benchmark by killing every serving process it can see, so
sessions in one PID namespace kill each other's servers. The launcher:

1. **starts each container** (`docker run`) with the GPUs, the host's network and IPC, your `--mount`s, and
   `--docker-arg`s, keeping it up with `tail -f /dev/null`;
2. **prepares the containers one at a time:**
   - `--root-prepare`, as root, for container-local fixes such as a passwd entry for your user;
   - then `--prepare`, with `SKIP_RAY_START=1`. Parallel installs race on a shared virtualenv, and Ray refuses
     `ROCR_VISIBLE_DEVICES` at import, so the mask is not set yet;
3. **starts every session together:** `--setup`, then `hyperloom-multibranch run`, in each container;
4. **removes the containers** when the sessions have ended.

A container whose preparation fails runs no session; the other points go on.

**Credentials** reach the containers by name, never on a command line: the gateways' variables
(`ANTHROPIC_*`, `OPENAI_*`, `CLAUDE_CODE_OAUTH_TOKEN`), `HF_TOKEN`, the CA bundle, `USER_DATA_PATH`,
`HYPERLOOM_RUNNER_CACHE` and `HYPERLOOM_RSI_ARCHIVE`, when set. Add others with `--forward NAME`.

**Without `--image`** the sessions are processes of the launcher. That is safe for one point. For several, Magpie's
cleanup reaches across them, so the launcher refuses unless you accept it with `--same-host`.

## Follow the points

The launch directory holds:

- **`STATUS.md`:** each point's cards, ports, state (`waiting`, `preparing`, `running`, `review`, `done` or `failed`),
  session directory, stop reason and FINAL choice. Below that: the review commands of each session that waits for
  one, and every error.
- **`launch.json`:** the same, for scripts.
- **`c<conc>.log`:** each point's preparation and session output.

The launcher also prints a line when a point starts, waits for a review, or ends.

## Review each point's map

Each session reviews its own agent map in EXPLORE, as one session does
({doc}`Review the agent map before the Arbor branches open </how-to/multibranch-review-gate>`): only its mapping
branch waits, and its clock runs. While a point waits, `STATUS.md` names its `REVIEW.md` and the command:

```bash
hyperloom-multibranch review --session <that point's session> approve --note "why"
```

For runs nobody watches, pass `--no-review-gate` in the run arguments.

## Stop

A SIGTERM or SIGINT to the launcher (or `scancel` of its job) stops every session with SIGTERM, which MB answers with
its CLOSE. `--stop-grace` (180 s by default) later, the launcher kills what is left and removes the containers.

## Exit codes

- **0:** every session exited 0.
- **1:** at least one did not, or its container could not be prepared.
- **2:** the launch was refused before anything started.
