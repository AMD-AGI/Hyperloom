---
name: inference_optimizer.multi_node
description: |
  Multi-node companion to the inference_optimizer skill. Use when the user
  prompt asks for inference optimization that needs more GPU / memory than
  a single pod provides (i.e. ``nodes >= 2``) — typical prompt signals are
  ``Nodes=N`` / ``N pods`` / ``TP=N`` larger than one pod's GPU count, or
  any model that cannot fit on one pod's GPUs. Drives a session-scoped
  SaFE RayJob through the ``inference_optimizer.multi_node`` Python CLI.
globs:
  - "**/multi_node/**"
  - "**/multi-node/**"
---

# Multi-Node RayJob Skill

Drive every RayJob lifecycle action through the Python CLI below. **Never
`ray.init`, `kubectl`, or raw `curl` to SaFE / Ray Dashboard.** All state
is in `/tmp/multi_node_state.json` (sandbox-local; lost on sandbox
recreate — re-running any subcommand reads back from SaFE / Ray
Dashboard and rewrites the file), maintained by the CLI.

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
python3 -m inference_optimizer.multi_node restart-server  --framework <sglang|vllm> --model <path-or-id> --tp <N>  [--extra-args "..."]
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
   Kills the previous server via PID file (never `pkill -f`), relaunches
   under `nohup` so Ray pods do NOT restart and the aiter JIT cache
   survives. Issue ONE invocation per change; the CLI fans out across
   all pods on multi-node runs. Never issue per-pod invocations.
5. **`stop-rayjob`** — at session end. Always call explicitly for an
   auditable release. `ownerId` cascade is a safety net (sandbox
   deletion removes the SaFE workload and tears the RayJob down via
   owner-ref), not a substitute. `inference_optimizer optimize
   --nodes N>=2` does **not** call it on exit; nor does sandbox idle /
   hard-TTL GC distinguish "session in progress" from "session
   abandoned" — when the sandbox dies the RayJob is collateral, so an
   in-flight optimize loses access to head_pod_ip / service_url even
   if RayJob teardown lags a few seconds behind sandbox pod removal.

After step 4 route all benchmark / OOB / Magpie traffic to
`state.service_url` (head pod ClusterIP `:8888`). Re-read
`/tmp/multi_node_state.json` every turn; never cache `head_pod_ip` /
`service_url` across actions — RayJob recreate (or `stop-rayjob` then
`create-rayjob` again) reassigns the head pod and rewrites both keys.

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
  Integrate path auto-restarts the server after `apply-patch`
  (bypassing the resume fast-path) and RayJob recreate auto-replays
  applied patches — do not invoke either step manually.

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
* `bootstrap` fails → `bootstrap --print-logs` once for the trace.
  The script (`multi_node/scripts/bootstrap.sh`) is repo-owned and
  should not be edited from the agent — failures usually point at one
  of: missing credentials in the RayJob env (verify the ADDENDUM-13
  fan-out so `SAFE_API_KEY` / `OOB_API_KEY` / `*_BASE_URL` actually
  reached the head pod), the BYOI image lacking a toolchain package,
  or the head pod failing to reach an upstream package / model
  registry. Fix the root cause (env / image / network), then rerun;
  RayJob stays alive across the retry.
* `restart-server` hangs in health probe → rerun with `--no-wait-health`
  to detach, then `verify` / `curl state.service_url/health` to debug.
* Cluster-side cleanup → `stop-rayjob --delete --clear-state` (hard
  delete = SaFE `DELETE /workloads/{id}`).
* `ModuleNotFoundError: No module named 'sglang'` + stderr shows
  `setsid python3 -m sglang.launch_server` on the sandbox →
  ADDENDUM-14 trip. Fix orchestrator env propagation;
  `/etc/profile.d/hyperloom.sh` export is an interim unblock.
* `restart-server` driver SUCCEEDED then `/health` wait timed out
  (default `DEFAULT_HEALTH_TIMEOUT_S = 900s`; override per-run via
  `HYPERLOOM_MN_HEALTH_WAIT_S`) → legacy launch_multinode swallowed
  framework early-exit. Patched: on rank-0 pid death the driver now
  exits 2 + writes
  `MULTI_NODE_FAILURE_SNAPSHOT={kind:"framework_early_exit",...}`; the
  Ray Dashboard job flips to FAILED and grid_runner skips the variant
  in seconds. Stderr `rank_0.log` tail names the cause.

* **Variant silently aborts with no benchmark output** — read
  `failed_variants` inside the round's `<action>_attempts.extras`
  (Coordinator surfaces per-variant aborts there). For post-mortem
  forensics, the per-variant `abort_reason.json` written by
  grid_runner under
  `${USER_DATA_PATH}/runs/<action>/<task>/<variant>/` carries
  `error_class` plus a truncated error tail.

* **`baseline_accuracy=0.0` + `accuracy gate skipped`** is the expected
  default, not a bug. `_workload_envs.py` sets `RUN_EVAL=false` because
  Magpie's `--concurrent-requests N` is rejected by InferenceX's current
  `run_lm_eval`; `_accuracy_gate.py` therefore returns True for
  high-risk variants. To opt in, export `RUN_EVAL=true` in the
  **sandbox** env before launch (`_workload_envs.py` reads it via
  `os.environ.get`); the multi-node CLI's `--extra-env` flag feeds
  RayJob env and is the **wrong scope**. Before opting in, confirm
  InferenceX accepts the flag via
  `grep -nR concurrent-requests "$INFERENCEX_PATH/benchmarks"` — if the
  grep is empty, opting in will fail every variant including baseline.

* **Image-level launcher-flag denylist (probe each boot,
  framework-aware, model-agnostic)**. Any `backends` grid variant
  whose corresponding launcher CLI flag is not registered on the
  current RayJob image will fail at argparse regardless of model /
  TP. Image rebuilds happen out-of-band; do not carry a hard-coded
  skip list across sessions. On each fresh sandbox boot, after
  `bootstrap` succeeds, probe the framework launcher from the RayJob
  head pod (the sandbox does not have the inference framework
  installed) and cross-reference its flag set against the grid for
  that framework in
  `inference_optimizer/orchestrator/action_executors/backends.py`:
  * **sglang** — probe via `python3 -m sglang.launch_server --help`;
    grid = `DEFAULT_BACKENDS_GRID` plus the multi-node tier
    additions.
  * **vllm** — probe via `python3 -m vllm.entrypoints.openai.api_server --help`
    (older builds) or `vllm serve --help` (v0.5+); grid =
    `DEFAULT_VLLM_BACKENDS_GRID`.

  For each variant in the active framework's grid whose flag the
  probe reports missing, add the variant name to `--skip-variants`
  on `inference_optimizer optimize` (or `SKIP_VARIANTS=...` in the
  prompt). Drop entries the moment a probe shows the flag accepted
  again. The other framework's grid is irrelevant to this run and
  MUST NOT be probed against the wrong launcher.

* **Model-specific variant incompatibilities are NOT a default skip
  list**. A variant that fails on one `(model, TP)` pair is not, by
  itself, evidence that it will fail on another. Do NOT auto-extend a
  known-bad list across model classes by analogy — re-probe per model
  and let the runtime's own rejected-variant ledger
  (`backends_search.rejected` / `params_search.rejected` in
  `state.json`) accumulate evidence instead. If you genuinely need to
  skip a variant for the *current* model, capture the failure first
  (let it run once and surface the per-variant abort marker the grid
  runner writes), then state that decision in the prompt with the
  model name spelled out — never propagate the skip silently into the
  next session.
