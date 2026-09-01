---
myst:
  html_meta:
    "description": "How to run a KernelForge optimization campaign: prepare a git workspace, launch kernelforge forge-loop on a kernel and its driver, and review the measured result."
    "keywords": "KernelForge, run campaign, kernelforge forge-loop, workspace, driver, kernel backend, gfx950, forge_experiments"
---

# Run a campaign

A campaign is one `kernelforge forge-loop` run over one kernel. Each iteration
proposes a change, measures it against the task's driver, and keeps it only if
the measurement improves.

## Prerequisites

- Hyperloom installed (`pip install -e ".[forge]"`; see
  {doc}`Quickstart </kernelforge/install/quickstart>`).
- Claude credentials: a logged-in `claude` CLI for in-session mode, or
  `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_API_KEY` / a gateway's
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` for headless runs.
- A ROCm environment with the target GPU (for example `gfx950`).
- A git workspace holding the kernel and its driver.

## Prepare the workspace

`forge-loop` edits the kernel in place and relies on git to keep an improvement
and restore everything else, so the campaign needs a workspace that is a git
repository with an initial commit. A workspace holds:

| File | Role | Needed |
|:-----|:-----|:-------|
| kernel source | What the loop optimizes — the `--kernel` anchor | required |
| `driver.py` | Correctness oracle and benchmark; protected, never edited | required |
| `graph_harness.py` | CUDA/HIP graph timing harness the driver benches through | recommended |
| `program.md` | Free-form guidance handed to the agent | recommended |

Keep build artifacts and `forge_experiments/` untracked so a revert never fails
on a dirtied tree. Every `src/kernelforge/data/examples/<task>/run_example.sh` sets
up exactly this and launches the loop; copying the closest one is the fastest
way to start a new task.

## Run the loop

```bash
W=/tmp/forge_run

kernelforge forge-loop \
    --kernel "$W/softmax_kernel.py" \
    --driver "$W/driver.py" \
    --workspace "$W" \
    --program-md-file "$W/program.md" \
    --experiments-dir "$W/forge_experiments" \
    --result-json "$W/forge_experiments/forge_result.json" \
    --kernel-backend triton \
    --gpu-target gfx950 \
    --snr-threshold 30.0 \
    --max-hours 8 \
    --git-branch forge-optimize \
    --target-functions "softmax,_softmax_kernel"
```

The flags that decide what a campaign is:

- `--kernel` — the anchor the driver exercises and the agent edits.
- `--driver` — the measurement driver. The loop treats it as a black box,
  talks to it over stdout, and blocks edits to it.
- `--kernel-backend` — which backend's domain knowledge is injected into the agent's
  prompt: one of `ck`, `flydsl`, `triton`, `gluon`, `aiter`, `hip`, or
  `hipblaslt`, written as the bare `<backend>` key.
- `--snr-threshold` — the correctness gate in dB, fixed for the campaign.
- `--max-hours` — the wall-clock budget (minimum 1.0). The campaign is
  time-driven; it does not stop at a fixed iteration count. It stops when what
  remains can no longer finish a round — measured once its planning has
  returned, see {doc}`the autonomous loop </kernelforge/how-to/autonomous-loop>` — so the
  last hour of a budget buys a narrower round rather than one that is killed
  halfway.
- `--git-branch` — the development branch the kept commits land on.

Each iteration, the agent works through the Bash tool inside the workspace: it
reads the kernel, edits it, compiles, runs the driver, and profiles. The loop
then runs the driver-owned complete correctness suite and the canonical
benchmark itself, commits a measured improvement as the new best, and restores
every other candidate. See the
{doc}`Optimization loop </kernelforge/conceptual/optimization-loop>` for the gates each
change has to clear.

For a multi-file operator or a whole repository (for example AITER), add
`--task-type repository` and list the implementation entry points with
`--source-files a.py,b.hip,...`. Those paths seed orientation, profiling and
knowledge-base identity; `--kernel` stays the anchor.

Before the first iteration, the loop checks the driver against the contract it
enforces at run time and repairs it if needed. When that fails the run aborts
with `task_preparation_failed`; see
{doc}`Debug task preparation </kernelforge/how-to/debug-task-preparation>`.

## Watch, stop, and resume a run

The loop prints its per-iteration progress to stdout, so a headless run is
usually launched with the output redirected to a log:

```bash
kernelforge forge-loop --workspace "$W" ... > /tmp/forge.log 2>&1
tail -F /tmp/forge.log
```

To end a run early, drop a stop file in the workspace; the loop checks for it at
the next iteration boundary and finalizes with the best it has. A campaign
interrupted that way — or by a crash — continues from the same workspace:

```bash
touch "$W/.stop"                                     # stop at the next iteration boundary
kernelforge forge-loop --workspace "$W" --resume   # continue the campaign in that workspace
```

Only the existence of `.stop` is checked, so removing it before resuming
continues the campaign.

## Review results

The best kept kernel is checked out in the workspace. Everything the campaign
measured is under the experiments directory:

```bash
cat "$W/forge_experiments/forge_result.json"   # baseline_ms, best_ms, mean_case_speedup, improved
ls  "$W/forge_experiments/candidates/"         # per-iteration kernel, diff, measurements, profile
ls  "$W/forge_experiments/lessons/"            # per-iteration factual records
git -C "$W" log --oneline forge-optimize       # the commits the loop kept
```

Lessons distilled from the run accumulate under
`knowledge_base/<backend>/learned/`, rooted at `$KERNELFORGE_PROJECT_ROOT`
(default `~/.cache/hyperloom/kernelforge`) -- not in the installed package.

For the two ways to launch and bill a run — Claude Code in-session and the
unattended autonomous loop — see
{doc}`Deployment modes </kernelforge/reference/deployment-modes>`.
