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

## Dynamo backend (`--mn-backend dynamo`)

Alternative multi-node backend: same "long-lived idle pod + external server
restart" loop as RayJob, but on a SaFE **DynamoDeployment** with **SSH** as the
control plane instead of the Ray Dashboard. Only active when `--nodes >= 2`;
single-node runs are unaffected. Select via `optimize --mn-backend dynamo`
(or `$INFERENCE_OPTIMIZER_MN_BACKEND=dynamo`).

Worker pods deploy **idle** (`mn-idle.sh` → sshd + block); `restart-server`
SSHes in to (re)launch `dynamo.sglang`/`dynamo.vllm`, so the aiter JIT cache
survives across restarts. Benchmarks always target the **Dynamo frontend
:8000** (`state.service_url` → `BENCHMARK_BASE_URL`), never sglang rank-0 :8888.

Requirements & behaviour:

* **Image** must carry the sshd layer (`docker/dynamo/Dockerfile.sshd`); sshd
  runs on `$MN_SSH_PORT` (default 2222, not 22).
* **Aggregated** (default): `serviceRoles=[frontend, worker]`,
  `multinodeRoles=[worker]`, `worker.replica = nodes`.
* **PD disaggregation**: pass `optimize --pd-mode disaggregated
  --pd-prefill-nodes N --pd-decode-nodes M [--pd-prefill-tp/--pd-decode-tp]`
  (same flags as the RayJob backend). Produces `serviceRoles=[frontend,
  prefill, decode]`; a role becomes a LeaderWorkerSet (multi-node) only when
  its TP exceeds one pod's GPUs, otherwise its replica is independent
  single-node instances.

Subcommands (`restart-server` / `kill-inference` auto-route by `state.backend`;
no `bootstrap` / `verify` step):

```bash
python3 -m inference_optimizer.multi_node create-dynamo --image <img-with-sshd> --nodes <N> \
  [--pd-mode disaggregated --pd-prefill-nodes N --pd-decode-nodes M] [--kv-transfer-backend nixl]
python3 -m inference_optimizer.multi_node restart-server --framework sglang --model <path> --tp <N> [--ep <N>] [--extra-args "..."]
python3 -m inference_optimizer.multi_node kill-inference
python3 -m inference_optimizer.multi_node stop-rayjob [--clear-state]
```

Native params via `restart-server --extra-args` (standard sglang knobs:
`--ep-size`, `--enable-dp-attention`, `--attention-backend aiter`,
`--mem-fraction-static`) and `dynamo.frontend --router-mode {round-robin,kv}`.
`--kv-transfer-backend {nixl,mori,mooncake}` selects the PD KV plane.

Kernel-agent on the Dynamo backend (no Ray): GEAK and OOB (claude/codex/cursor)
run on a GPU pod over SSH (`KERNEL_AGENT_GPU_PLACEMENT=ssh`, injected only when
`backend==dynamo`); `apply-patch` / `revert-patch` / `kernel-bench` fan out over
SSH (routed by `state.backend`). The provisioner installs both toolchains on
the pods once (`install-geak` + `install-oob`, from the shared `$HYPERLOOM_ROOT/
geak` and `$OOB_SRC` checkouts) — skipped under `--no-kernel`. vLLM multi-node
bootstraps Ray across the pods pod-side.

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
  generated (see `DISPLAY_NAME` section below), `--owner-id`→`$WORKLOAD_ID`.

### Map the user Environment block → CLI (do not re-ask)

When the user prompt already lists topology / workload / kernel knobs, the
launcher **maps** them — it does not need the user to repeat them in chat.
Typical prompt fields and where they land:

| User prompt | Launcher action |
|---|---|
| `Nodes=N` / `N nodes` | `create-rayjob --nodes N`; `optimize --nodes N` |
| `RayJob image: …` / `Dynamo image: …` | `create-rayjob`/`create-dynamo --image …`; `optimize --rayjob-image …` |
| `TP=N`, `EP=…` | `restart-server --tp N`; `optimize --tp` / `--ep`. **Always set `--tp`** (default is 1); for PD set it to the per-role TP. |
| `MN_BACKEND=dynamo` | `optimize --mn-backend dynamo` (selects the idle DynamoDeployment + SSH backend; default `rayjob`). |
| `PD_MODE=disaggregated` | `optimize --pd-mode disaggregated` — **must be passed as a flag**; `$PD_MODE` env is deliberately ignored (stale-env guard). Omit ⇒ aggregated. |
| `PD_PREFILL_NODES` / `PD_DECODE_NODES` | `optimize --pd-prefill-nodes N --pd-decode-nodes M` (or export `$PD_PREFILL_NODES`/`$PD_DECODE_NODES` — read as flag defaults). |
| `PD_PREFILL_TP` / `PD_DECODE_TP` | `optimize --pd-prefill-tp N --pd-decode-tp M` (or export; default = `--tp`). A PD role spans nodes (LWS) only when its TP > GPUs-per-pod. |
| `PD_PREFILL_EP` / `PD_DECODE_EP` | **Dynamo PD only.** export `$PD_PREFILL_EP` / `$PD_DECODE_EP` (read by `restart-server` as defaults). Per-role expert-parallel size; `0` (default) ⇒ fall back to the shared `--ep`. Lets prefill run EP1 while decode runs EP8 (InferenceX disagg recipe). Ignored by RayJob/aggregated/single-node. |
| `PD_PREFILL_EXTRA_ARGS` / `PD_DECODE_EXTRA_ARGS` | **Dynamo PD only.** export these; appended to the **per-role** sglang launch AFTER the shared `--extra-args` base (role-specific wins on duplicate keys). Used to give prefill vs decode different server flags (e.g. decode `--enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal --moe-dense-tp-size 1 --enable-dp-lm-head`; prefill `--mem-fraction-static 0.8 --disable-radix-cache`). Empty (default) ⇒ both roles use only the shared `--extra-args`. **Sandbox-only** (do NOT `--rayjob-extra-env`). |
| `PD_TRANSFER_BACKEND` | `optimize --pd-transfer-backend nixl` (or export `$PD_TRANSFER_BACKEND`); `nixl|mori|mooncake`. |
| `ISL` / `OSL` / `CONC` / `PRECISION` | `export` + `optimize --isl` / `--osl` / `--conc` / `--precision` |
| `KERNEL_OPT_*` / `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX` | `export` before `install.sh` / `optimize` |
| prompt `env:` block lines (e.g. `PATH_TO_AINIC_TAR_PACKAGE=…`, `PATH_TO_BNXT_TAR_PACKAGE=…`, `NCCL_DEBUG=INFO`) | `create-rayjob --extra-env K=V` (one per line, repeatable); `optimize --rayjob-extra-env K=V` (same shape). Skip `*_API_KEY` / `*_BASE_URL` (credential fanout auto-injects) and `RAY_JOB_ENTRYPOINT` (reserved). CLI owns no defaults — values come verbatim from the prompt. **Do NOT forward sandbox-side tool source fields** (`OOB_SRC` / `INFERENCEX_PATH` / `TRACELENS_ROOT`) here — they are sandbox-only; see `inference_optimizer/SKILL.md` "Tool source fields". |
| MoE JIT cold-start (often omitted in prompt) | `export HYPERLOOM_MN_POLL_TIMEOUT_S=1800` and `HYPERLOOM_MN_HEALTH_WAIT_S=1800` — see below |

If the prompt already contains the first rows, **do not** claim the
“environment block is incomplete”; wire them into `setsid nohup optimize`
and `multi_node` subcommands. Only add exports the prompt did not cover
(chiefly `HYPERLOOM_MN_*` for 30 min polls on large MoE RayJobs).

**DO NOT `--rayjob-extra-env` these (sandbox-only):**

These are consumed by `install.sh` / `inference_optimizer optimize` /
`_workload_envs.py` running inside the **sandbox**; nothing inside the
RayJob pod reads them. Forwarding them pollutes the pod env and risks
shadowing real values.

- `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX`, `KERNEL_OPT_*`
- `NODE_TLS_REJECT_UNAUTHORIZED`
- `RANDOM_RANGE_RATIO`, `RUN_EVAL`
- `MODEL_PATH`, `FRAMEWORK`, `TP`, `EP`, `ISL`, `OSL`, `CONC`, `PRECISION`,
  `TARGET_GAIN`, `MAX_HOURS`, `GPU_TYPE`, `NODES` (already passed as
  `optimize` CLI flags)
- `MN_BACKEND`, `PD_MODE`, `PD_PREFILL_NODES`, `PD_DECODE_NODES`,
  `PD_PREFILL_TP`, `PD_DECODE_TP`, `PD_PREFILL_EP`, `PD_DECODE_EP`,
  `PD_PREFILL_EXTRA_ARGS`, `PD_DECODE_EXTRA_ARGS`, `PD_TRANSFER_BACKEND`
  (sandbox-side `optimize` flags / env; the Dynamo deployment is created by
  `create-dynamo` from these / consumed by `restart-server`, NOT injected
  into pods)
- `HYPERLOOM_MN_POLL_TIMEOUT_S`, `HYPERLOOM_MN_HEALTH_WAIT_S` (sandbox
  CLI poll budget, not a pod env)

Forward to `--rayjob-extra-env` **only** the prompt `env:` block lines
(`NCCL_DEBUG`, `PATH_TO_*` etc.). `OOB_SRC` / `INFERENCEX_PATH` /
`TRACELENS_ROOT` are sandbox-only and **must NOT** be forwarded (the
RayJob pod does not consume them — kernel-bench is NodeAffinity-pinned
to the head pod and the head pod does not invoke OOB CLI).

Example `optimize` tail (**example only** — map each flag from the user
Environment block / `setup_env.sh`; do not treat literals below as defaults):

```bash
# Multi-node poll budget (large MoE RayJobs; see "MoE JIT poll budget" below)
export HYPERLOOM_MN_POLL_TIMEOUT_S=1800
export HYPERLOOM_MN_HEALTH_WAIT_S=1800
# Optional kernel exports — only when the prompt specifies them
export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-claude}"
export KERNEL_AGENT_BUILD_GEAK_RAG_INDEX="${KERNEL_AGENT_BUILD_GEAK_RAG_INDEX:-0}"

setsid nohup inference_optimizer --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --gpu-type "${GPU_TYPE:?set from prompt}" \
  --nodes "${NODES:?set from prompt Nodes=N}" \
  --rayjob-image "${INFERENCE_OPTIMIZER_RAYJOB_IMAGE:?set from prompt RayJob image}" \
  --tp "${TP:?set from prompt TP=N}" \
  ${EP:+--ep "$EP"} \
  --conc "${CONC:?set from prompt}" \
  --isl "${ISL:?set from prompt}" \
  --osl "${OSL:?set from prompt}" \
  --precision "${PRECISION:?set from prompt}" \
  --target-gain "${TARGET_GAIN:?set from prompt}" \
  --max-hours "${MAX_HOURS:?set from prompt}" \
  ${KERNEL_CLAUDE:+--kernel-claude} \
  ${CLAUDE_MODEL:+--claude-model "$CLAUDE_MODEL"} \
  $(for kv in "${RAYJOB_EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && printf -- '--rayjob-extra-env %q ' "$kv"; done) \
  > "$RUN_LOG" 2>&1 < /dev/null &
```

### `DISPLAY_NAME` (SaFE workload create only)

Only `create-rayjob` sets the SaFE workload name. Resolution order:

1. If `$DISPLAY_NAME` is set, use it as-is — do **not** pass `--display-name`.
2. Otherwise generate one that satisfies the SaFE admission webhook:
   length **1–36**, lowercase letters / digits / hyphens only (`[a-z0-9-]`),
   start with a letter, end with alphanumeric. Good: `hl-run-$(date +%m%d%H%M)`.
   Bad: `hyperloom-sglang-2node-20260522_022937` (too long / underscores).

### SaFE workload `phase` (source of truth)

`create-rayjob` polls **SaFE GetWorkload `phase`** until it is **`Running`**.
Do **not** treat individual pod `phase=Running` as a substitute while the
workload is still `Pending` — bootstrap / `head_pod_ip` / benchmarks must
wait for the workload object to flip. If poll times out with `phase=Pending`,
re-run the **same** `create-rayjob` (idempotent resume) with a longer
`HYPERLOOM_MN_POLL_TIMEOUT_S` or `--poll-timeout`; inspect
`conditions` / `message` on the workload for queue or dispatch issues.

### MoE JIT poll budget (110s default is too short)

First `restart-server` on a multi-node RayJob often needs **20–30 minutes**
(weight load + aiter JIT). The default per-invocation poll is **~110s**
(ADDENDUM-09). Export before `restart-server` / `optimize --nodes >=2`:

```bash
export HYPERLOOM_MN_POLL_TIMEOUT_S=1800
export HYPERLOOM_MN_HEALTH_WAIT_S=1800
```

On timeout, re-run the **same** subcommand (no `while sleep` wrapper).
`restart-server` checkpoints `last_restart_submission_id` so retries can
resume an in-flight launch (`MULTI_NODE_RESTART_RESUME_RUNNING=1`, default).

## Call Order

1. **`create-rayjob`** — once. Persists `rayjob_id` before polling
   (overlapping retries never spawn a second RayJob), then fills
   `head_pod_ip` / `service_url` once phase is `Running`.
2. **`bootstrap`** — once. Submits `bootstrap.sh` via Ray Dashboard REST
   to install oob / claude / codex / tracelens on the head pod.
3. **`verify`** — once. Checks `ray` on PATH on the head pod.
   On `MISSING:`, re-run `bootstrap --print-logs`.
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

* **ADDENDUM-09** (bash budget): each CLI invocation polls until
  `--poll-timeout` (default 110s unless `HYPERLOOM_MN_POLL_TIMEOUT_S`
  is set) then exits. For MoE JIT cold-start on RayJob pods, export
  `HYPERLOOM_MN_POLL_TIMEOUT_S=1800` and `HYPERLOOM_MN_HEALTH_WAIT_S=1800`
  before `restart-server` / `optimize --nodes >=2`. Timeout → rerun the
  **same** subcommand (resume uses `last_restart_submission_id`). Never
  wrap in `sleep` / `while true ...; sleep 60; done`.
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
* **ADDENDUM-16** (robustness LocalProbe is sandbox-scoped): the
  `robustness-agent` backend's `LocalProbeSource` family probes
  sandbox-local resources only — `ray status`, the inference server
  health URL (`http://127.0.0.1:8888`), GPU / FD / disk / shm metrics,
  the local log-error scanner, etc. On `--nodes >= 2` every one of
  those resources lives in a separate Kubernetes pod (head pod /
  worker pod / RayJob submitter, on a different subnet from the
  sandbox in some clusters), so each probe surfaces as a HIGH-severity
  false positive (`ray_head_dead`, `local_server_unreachable`,
  `gpu_memory_leaked`, ...). The CLI
  auto-downgrades `--robustness-agent` to `--robustness-mock`
  (heartbeat-only) when `args.nodes >= 2` and prints a WARNING.
  Operators who want to suppress the WARNING pass `--robustness-mock`
  explicitly. Until `robustness-agent` grows multi-node-aware probe
  targeting (probe head pod over the cluster service URL, route
  GPU / log probes through `kubectl exec` or a sidecar), the
  multi-node path keeps robustness on the mock heartbeat.

## Robustness limitation in multi-node mode

`inference_optimizer.cli._resolve_robustness_choice` enforces the
contract above:

```python
# pseudo-code mirroring the actual logic
if args.nodes >= 2 and chosen == "agent":
    if explicit:
        print("WARN: ... auto-downgrading to --robustness-mock ...",
              file=sys.stderr)
    chosen = "mock"
```

Operator-visible effects:

* All robustness intents are heartbeats — no `alert(HIGH)`,
  no `escalate_strategy_change`, no `delegate(report)` /
  `delegate(recover)` / `delegate(server_lifecycle)`,
  no `prune_branch`, no `force_dispatch`, no `kill_task`.
* The `<session_dir>/robustness-workdir/` and
  `<session_dir>/agents/robustness/` directories stay empty (mock
  backend does not write them).
* Long-run health monitoring still works at the **shell** level via
  `optimizer_runs/robustness_monitor.sh` (polls `state.json`,
  detects terminal `stop_reason`, auto-resumes a dead optimizer);
  that monitor is independent of the in-process robustness backend.

The auto-downgrade is unconditional on `args.nodes >= 2`. The
explicit-flag WARNING is the only operator signal (silent when the
default `--robustness-agent` was selected via
`DEFAULT_ROBUSTNESS_BACKEND` rather than an explicit CLI flag).

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
  (`explore_search.rejected` in `state.json`) accumulate evidence
  instead. If you genuinely need to
  skip a variant for the *current* model, capture the failure first
  (let it run once and surface the per-variant abort marker the grid
  runner writes), then state that decision in the prompt with the
  model name spelled out — never propagate the skip silently into the
  next session.
