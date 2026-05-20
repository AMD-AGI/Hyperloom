---
name: inference_optimizer.multi_node
description: |
  Multi-node companion to the inference_optimizer skill. Use when the user
  prompt asks for inference optimization that needs more GPU memory than a
  single pod provides (i.e. ``nodes >= 2``), e.g. "8x MI300X across 2 nodes",
  "TP=16 across two pods", "multi-node sglang serving", or any time a model
  cannot fit on one pod's GPUs. Drives a session-scoped SaFE RayJob through the
  ``inference_optimizer.multi_node`` Python CLI.
globs:
  - "**/multi_node/**"
  - "**/multi-node/**"
---

# Multi-Node RayJob Skill

Drive every RayJob lifecycle action through the Python CLI below. **Never
`ray.init`, `kubectl`, or raw `curl` to SaFE / Ray Dashboard.** All state
is in `/tmp/multi_node_state.json`, rewritten by the CLI.

| Action          | Use                                       | Not                                                       |
|-----------------|-------------------------------------------|-----------------------------------------------------------|
| Create RayJob   | `create-rayjob`                           | `curl POST /workloads`, `kubectl create -f rayjob.yaml`   |
| Check phase     | `create-rayjob` (idempotent — resumes)    | `curl /workloads/{id}`, `kubectl get rayjob`              |
| Restart server  | `restart-server`                          | `kubectl exec ... sglang.launch_server`                   |
| Stop            | `stop-rayjob`                             | `curl POST .../stop`                                      |

Bypassing loses: idempotency, `ownerId` cascade cleanup, exit-2 +
`MULTI_NODE_FAILURE_SNAPSHOT={...}` failure detection, cross-subcommand
state, and `BENCHMARK_BASE_URL` plumbing for Magpie.

## The Five Subcommands

```bash
python3 -m inference_optimizer.multi_node create-rayjob   --image <rayjob-image> --nodes <N>
python3 -m inference_optimizer.multi_node bootstrap       [--print-logs]
python3 -m inference_optimizer.multi_node verify
python3 -m inference_optimizer.multi_node restart-server  --framework sglang --model <path-or-id> --tp <N>  [--extra-args "..."]
python3 -m inference_optimizer.multi_node stop-rayjob     [--clear-state]
```

Run `<subcommand> --help` for the full flag set. **Do not invent flags.**

### Parameter sources

* **From sandbox env** (do NOT pass on CLI / read from prompt):
  `SAFE_API_URL`, `SAFE_API_KEY`, `SAFE_WORKSPACE`, `DISPLAY_NAME`,
  `WORKLOAD_ID` — Brain / SaFE injects at sandbox start.
* **From user prompt** (verbatim): `--image` ← `RayJob image:`;
  `--nodes` ← `Nodes=N` / 多节点说法; `--cpus-per-node` /
  `--mem-per-node` / `--ephemeral-per-node` ← `RayJob resource:`;
  `--tp` (restart-server only) ← `TP=N`; `--extra-env KEY=VAL` (repeatable)
  ← prompt `env:` block (skip `*_API_KEY` / `*_BASE_URL` /
  `RAY_JOB_ENTRYPOINT` — auto-injected).
* **Defaults** (omit when prompt is silent): `--workspace`→`$SAFE_WORKSPACE`,
  `--gpus-per-node`→`8`, `--display-name`→`$DISPLAY_NAME` else
  `multi_node_<unix-ts>`, `--owner-id`→`$WORKLOAD_ID`.

## Call Order

1. **`create-rayjob`** — once. Persists `rayjob_id` before polling
   (overlapping retries never spawn a second RayJob), then fills
   `head_pod_ip` / `service_url` once phase is `Running`.
2. **`bootstrap`** — once. Submits `bootstrap.sh` via Ray Dashboard REST
   to install oob / claude / codex / tracelens on the head pod.
3. **`verify`** — once. Confirms toolchain on PATH; bail if it fails.
4. **`restart-server`** — every framework / model / TP / flag change.
   Kills previous via PID file (no `pkill -f`), relaunches under `nohup`
   so Ray pods never restart and the aiter JIT cache survives.
   * `nodes == 1`: bash entrypoint — local PID kill + fresh
     `sglang.launch_server :8888`.
   * `nodes >= 2`: one Python Ray driver per restart sweeps every pod's
     PID files via NodeAffinity-pinned actors, then spawns one
     `sglang.launch_server --nnodes N --node-rank K
     --dist-init-addr <head_pod_ip>:<port>` per pod (port from
     `$RAYJOB_DIST_INIT_PORT`, default 5000). Agent issues ONE
     invocation; the driver fans out.
5. **`stop-rayjob`** — at session end. Always call explicitly for an
   auditable release. `ownerId` cascade is a safety net, not a
   substitute. `inference_optimizer optimize --nodes N>=2` does **not**
   call it on exit.

After step 4 route all benchmark / OOB / Magpie traffic to
`state.service_url` (head pod ClusterIP `:8888`). Re-read state every
turn; never cache across actions.

## Hard Rules

* **ADDENDUM-09** (bash budget): each CLI invocation polls ~110s then
  exits. Timeout → rerun the same subcommand. Never wrap in `sleep` /
  `while true ...; sleep 60; done`.
* **ADDENDUM-13** (credentials): `SAFE_API_URL` / `SAFE_API_KEY` must be
  in sandbox env at CLI start (Brain injects). CLI fans them out (plus
  `OOB_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `*_BASE_URL`)
  to RayJob env at `create-rayjob`. **Never pass keys on the command line.**
* **ADDENDUM-02** (no Ray Python client in orchestration layer):
  `multi_node/cli.py` and `multi_node/_internal/` MUST use Ray Dashboard
  REST only — never `import ray` / `ray.init(address=...)` against the
  inference RayJob. (`pip install ray` in sandbox is fine for unrelated
  stacks. Code submitted as a Ray Dashboard entrypoint and running
  inside RayJob pods is exempt — those pods *are* the Ray cluster.)
* **ADDENDUM-14** (sandbox never runs the inference server): When
  `nodes >= 2`, sglang / vllm lives only on RayJob pods; the sandbox is
  the client. Every Magpie launch MUST inherit
  `BENCHMARK_BASE_URL=<state.service_url>` (forces `PHASE=client`).
  Missing it → Magpie defaults to `PHASE=all` → `python3 -m
  sglang.launch_server` on the CPU sandbox → `ModuleNotFoundError`.
  Always fix orchestrator env propagation; never `pip install sglang`
  in the sandbox. (`/etc/profile.d/hyperloom.sh` global export is a
  shell-level backstop only — head pod IP changes per RayJob recreate.)
* **ADDENDUM-15** (kernel-agent fan-out): kernel-agent runs in the
  sandbox but writes to source under `/sgl-workspace/{aiter,sglang,vllm}/`
  which is per-pod local fs — sandbox edits do NOT reach the RayJob
  pods. Use the three multi-node subcommands instead:
  * `apply-patch` — fan-out a kernel patch to every pod (head + workers)
    via `kernel_patch_multinode.py`; per-host backups are written under
    `--backup-dir` and returned to the caller for revert.
  * `revert-patch` — inverse of apply, takes the per-host backup map.
  * `kernel-bench` — run a kernel micro-benchmark on a GPU-bearing pod
    (the sandbox is CPU-only in multi-node mode); stages helper files
    + bench script onto the pod, runs `bash --bench-command`, reads
    back result artifacts. Used by kernel_optimization.py prompt
    template; not invoked directly by the agent.
  After every `apply-patch` the integrate path calls
  `restart_server_for_round(force_full_restart=True)` so sglang
  re-imports the patched modules — resume fast-path is bypassed for
  this one call only. RayJob recreate replays applied patches
  automatically (see `_replay_kernel_patches_for_multi_node` in
  inference_optimizer/cli.py).

## Exit Codes (for the controller / agent)

| Code | Meaning                                | Controller action                              |
|-----:|----------------------------------------|------------------------------------------------|
| 0    | success                                | continue                                       |
| 1    | transient (poll timeout / SaFE 5xx /   | safe to rerun the SAME subcommand to retry     |
|      | network error / unknown exception)     |                                                |
| 2    | workload entered Failed/Stopped/Cancelled | DO NOT retry; cluster is unusable. Stderr     |
|      |                                        | also carries `MULTI_NODE_FAILURE_SNAPSHOT={...}` |
|      |                                        | with the structured failure detail             |
| 3    | config error: SaFE 4xx (image not      | DO NOT retry as-is; fix args / env and rerun   |
|      | found, quota, missing workspace, bad   |                                                |
|      | label) / missing env / missing arg     |                                                |
| 130  | SIGINT / Ctrl-C                        | user aborted                                   |

## When Something Looks Wrong

* `create-rayjob` times out → rerun; state already has `rayjob_id`,
  rerun resumes polling.
* `bootstrap` fails → `bootstrap --print-logs` once for the trace, fix
  script, rerun. RayJob stays alive.
* `restart-server` hangs in health probe → rerun with `--no-wait-health`
  to detach, then `verify` / `curl state.service_url/health` to debug.
* Cluster-side cleanup → `stop-rayjob --delete --clear-state` (hard
  delete = SaFE `DELETE /workloads/{id}`).
* `ModuleNotFoundError: No module named 'sglang'` + stderr shows
  `setsid python3 -m sglang.launch_server` on the sandbox →
  ADDENDUM-14 trip. Fix orchestrator env propagation;
  `/etc/profile.d/hyperloom.sh` export is an interim unblock.
* `restart-server` driver SUCCEEDED then 1800s `/health` timeout →
  legacy launch_multinode swallowed framework early-exit. Patched: on
  rank-0 pid death the driver now exits 2 + writes
  `MULTI_NODE_FAILURE_SNAPSHOT={kind:"framework_early_exit",...}`; the
  Ray Dashboard job flips to FAILED and grid_runner skips the variant
  in seconds. Stderr `rank_0.log` tail names the cause.
* Known sglang launcher-flag denylist for current RayJob image
  (`sgl-workspace` commit `e714db81a`): `--enable-fused-moe`,
  `--enable-custom-ar` are NOT recognized — `backends` grid variants
  `enable_fused_moe` / `custom_ar` always fail at argparse. Pass
  `SKIP_VARIANTS=enable_fused_moe,custom_ar` (plus your existing
  `attn_aiter,decode_aiter` if DSr1 TP=16) in the prompt or CLI.
  Re-probe after every RayJob image rebuild via
  `python3 -m sglang.launch_server --help | grep -- --enable-fused-moe`.
