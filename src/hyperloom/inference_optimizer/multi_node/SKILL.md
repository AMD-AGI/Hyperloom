---
name: hyperloom.inference_optimizer.multi_node
description: |
  Multi-node companion to the inference_optimizer skill. Use when the user
  prompt asks for inference optimization that needs more GPU / memory than
  a single pod provides (i.e. ``nodes >= 2``) — typical prompt signals are
  ``Nodes=N`` / ``N pods`` / ``TP=N`` larger than one pod's GPU count, or
  any model that cannot fit on one pod's GPUs. Drives a session-scoped
  SaFE RayJob through the ``hyperloom.inference_optimizer.multi_node`` Python CLI.
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
| Stop            | `stop-multi-job`                             | `curl POST .../stop`                                      |

Bypassing loses: idempotency, `ownerId` cascade cleanup, exit-2 +
`MULTI_NODE_FAILURE_SNAPSHOT={...}` failure detection, cross-subcommand
state, and `BENCHMARK_BASE_URL` plumbing for Magpie.

## Infera backend (`--mn-backend infera`)

Alternative multi-node backend: same "long-lived idle pod + external server
restart" loop as RayJob, but on a SaFE **InferaDeployment** with **SSH** as the
control plane instead of the Ray Dashboard. Only active when `--nodes >= 2`;
single-node runs are unaffected. Select via `optimize --mn-backend infera`
(or `$INFERENCE_OPTIMIZER_MN_BACKEND=infera`).

Worker pods deploy **idle** (`mn-idle.sh` → sshd + block); `restart-server`
SSHes in to (re)launch `infera.engine.sglang`/`infera.engine.vllm`, so the aiter JIT cache
survives across restarts. Benchmarks always target the **Infera frontend
:8000** (`state.service_url` → `BENCHMARK_BASE_URL`), never sglang rank-0 :8888.

Requirements & behaviour:

* **Image** must carry the sshd layer (`docker/infera/Dockerfile.sshd`); sshd
  runs on `$MN_SSH_PORT` (default base 2222, not 22). Under hostNetwork, each
  GPU **role** binds a distinct port (prefill/worker `2222+N`, decode
  `2232+N` via `LWS_WORKER_INDEX`) so co-located roles on one node do not
  collide.
* **Aggregated** (default): `serviceRoles=[frontend, worker]`,
  `multinodeRoles=[worker]`, `worker.replica = nodes`.
* **PD disaggregation**: pass `optimize --pd-mode disaggregated
  --pd-prefill-nodes N --pd-decode-nodes M [--pd-prefill-tp/--pd-decode-tp]`
  (same flags as the RayJob backend). Produces `serviceRoles=[frontend,
  prefill, decode]`; a role becomes a LeaderWorkerSet (multi-node) only when
  its TP exceeds one pod's GPUs, otherwise its replica is independent
  single-node instances.

Subcommands (`restart-server` / `kill-inference` auto-route by `state.backend`;
no `init-env` / `verify` step):

```bash
python3 -m hyperloom.inference_optimizer.multi_node create-infera --image <img-with-sshd> --model <path-or-hf-id> --nodes <N> \
  [--pd-mode disaggregated --pd-prefill-nodes N --pd-decode-nodes M] [--kv-transfer-backend mooncake]
python3 -m hyperloom.inference_optimizer.multi_node restart-server --framework sglang --model <path> --tp <N> [--ep <N>] [--extra-args "..."]
python3 -m hyperloom.inference_optimizer.multi_node kill-inference
python3 -m hyperloom.inference_optimizer.multi_node stop-multi-job [--clear-state]
```

Native params via `restart-server --extra-args` (standard sglang knobs:
`--ep-size`, `--enable-dp-attention`, `--attention-backend aiter`,
`--mem-fraction-static`) and `infera.server --router-mode {round-robin,kv}`.
`--kv-transfer-backend {nixl,mori,mooncake}` selects the PD KV plane.
Prefer `mooncake` for sglang on this RoCE/bnxt fabric: `nixl` completes
requests with HTTP 200 but produces 0 output tokens (prefill OK, decode emits
nothing — KV handoff via UCX/nixl fails to register/transfer). `mooncake`
auto-detects the RDMA device and is the sglang framework default.

Kernel-agent on the Infera backend (no Ray): GEAK runs on a GPU pod over SSH
(`KERNEL_AGENT_GPU_PLACEMENT=ssh`, injected only when `backend==infera`);
`apply-patch` / `revert-patch` / `kernel-bench` fan out over SSH (routed by
`state.backend`). The provisioner installs GEAK on the pods once
(`install-geak`, from the shared `$HYPERLOOM_ROOT/geak` checkout) — skipped
under `--no-kernel`. vLLM multi-node bootstraps Ray across the pods pod-side.

## External mode (SaFE-less: env-provided cluster)

When SaFE is **unavailable** (`SAFE_API_URL` / `SAFE_API_KEY` not both set) and
`HYPERLOOM_MN_EXT_SERVICE_URL` is set, the optimizer **skips all SaFE
create/init** and synthesizes the multi-node state from env vars, then
benchmarks (and, when SSH/head is supplied, restarts + GPU-samples) an
already-provisioned cluster. When both `SAFE_API_*` are present these external
vars are ignored (normal SaFE flow).

| Env var | Backend | Req? | Purpose |
| --- | --- | --- | --- |
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | HTTP frontend for benchmarks (-> `BENCHMARK_BASE_URL`); presence triggers external mode |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` / `_WORKER_IPS` | infera | infera | comma-separated GPU pod IPs (topology / PD / GPU sampling) |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | infera | private key that can SSH into the pods (you supply it; no SaFE to inject one) |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | no | SSH base port (default 2233; decode is role-offset) |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | no | known_hosts path (else lax host-key check) |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | rayjob | Ray head pod IP -> Dashboard `:8265` (submit) + GCS `:6379` (derived) |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | no | Ray Dashboard auth token (only if the dashboard is authenticated) |

Reused as-is: `INFERENCE_OPTIMIZER_NODES`, `INFERENCE_OPTIMIZER_GPUS_PER_NODE`,
`PD_MODE`, `INFERENCE_OPTIMIZER_MN_BACKEND`, `SAFE_WORKSPACE` (passthrough).

Behavior:

* **infera external REQUIRES SSH** (`_SSH_KEY` + at least one `*_IPS`); if missing
  the run **fails fast** (`sys.exit(2)`) rather than degrading. With SSH: full
  SSH restart + on-pod GPU sampling + pd/by-role telemetry, exactly like a
  SaFE-created infera deployment.
* **rayjob external** uses Ray via `_HEAD_IP` (not SSH). Without `_HEAD_IP` it is
  benchmark-only (per-round restart no-ops); the SSH rule does not apply to it.
* The SSH keypair is normally hyperloom-generated and its **public** key is
  injected into the pods by SaFE at create time. With SaFE absent you must
  pre-authorize your own key on the external pods and pass its private path via
  `_SSH_KEY`.

Example (SaFE assumed absent, infera PD-disaggregated):

```bash
unset SAFE_API_URL SAFE_API_KEY
export HYPERLOOM_MN_EXT_SERVICE_URL=http://<frontend-host>:8000
export HYPERLOOM_MN_EXT_PREFILL_IPS=<prefill-ip>  HYPERLOOM_MN_EXT_DECODE_IPS=<decode-ip>
export HYPERLOOM_MN_EXT_SSH_KEY=/path/to/id_ed25519
export INFERENCE_OPTIMIZER_NODES=2 PD_MODE=disaggregated
inference_optimizer optimize --model <path> --nodes 2 \
  --mn-backend infera --pd-mode disaggregated --tp 8 --ep 8 ...
```

## The Five Subcommands

```bash
python3 -m hyperloom.inference_optimizer.multi_node create-rayjob   --image <rayjob-image> --nodes <N>
python3 -m hyperloom.inference_optimizer.multi_node init-env       [--print-logs]
python3 -m hyperloom.inference_optimizer.multi_node verify
python3 -m hyperloom.inference_optimizer.multi_node restart-server  --framework <sglang|vllm> --model <path-or-id> --tp <N>  [--extra-args "..."]
python3 -m hyperloom.inference_optimizer.multi_node stop-multi-job     [--clear-state]
```

Run `<subcommand> --help` for the full flag set. **Do not invent flags.**

### Parameter sources

* **From sandbox env** (do NOT pass on CLI / read from prompt):
  `SAFE_API_URL`, `SAFE_API_KEY`, `SAFE_WORKSPACE`, `DISPLAY_NAME`,
  `WORKLOAD_ID` — Brain / SaFE injects at sandbox start.
* **From user prompt** (verbatim): `--image` ← `RayJob image:`;
  `--nodes` ← `Nodes=N` / multi-node phrasing; `--cpus-per-node` /
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
| `RayJob image: …` / `Infera image: …` | `create-rayjob`/`create-infera --image …`; `optimize --mn-image …` |
| `TP=N`, `EP=…` | `restart-server --tp N`; `optimize --tp` / `--ep`. **Always set `--tp`** (default is 1); for PD set it to the per-role TP. |
| `MN_BACKEND=infera` | `optimize --mn-backend infera` (selects the idle InferaDeployment + SSH backend; default `rayjob`). |
| `PD_MODE=disaggregated` | `optimize --pd-mode disaggregated` — **must be passed as a flag**; `$PD_MODE` env is deliberately ignored (stale-env guard). Omit ⇒ aggregated. |
| `PD_PREFILL_NODES` / `PD_DECODE_NODES` | `optimize --pd-prefill-nodes N --pd-decode-nodes M` (or export `$PD_PREFILL_NODES`/`$PD_DECODE_NODES` — read as flag defaults). |
| `PD_PREFILL_TP` / `PD_DECODE_TP` | `optimize --pd-prefill-tp N --pd-decode-tp M` (or export; default = `--tp`). A PD role spans nodes (LWS) only when its TP > GPUs-per-pod. |
| `PD_PREFILL_EP` / `PD_DECODE_EP` | **Infera PD only.** export `$PD_PREFILL_EP` / `$PD_DECODE_EP` (read by `restart-server` as defaults). Per-role expert-parallel size; `0` (default) ⇒ fall back to the shared `--ep`. Lets prefill run EP1 while decode runs EP8 (InferenceX disagg recipe). Ignored by RayJob/aggregated/single-node. |
| `PD_PREFILL_EXTRA_ARGS` / `PD_DECODE_EXTRA_ARGS` | **Infera PD only.** export these; appended to the **per-role** sglang launch AFTER the shared `--extra-args` base (role-specific wins on duplicate keys). Used to give prefill vs decode different server flags (e.g. decode `--enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal --moe-dense-tp-size 1 --enable-dp-lm-head`; prefill `--mem-fraction-static 0.8 --disable-radix-cache`). Empty (default) ⇒ both roles use only the shared `--extra-args`. **Sandbox-only** (do NOT `--rayjob-extra-env`). |
| `PD_TRANSFER_BACKEND` | `optimize --pd-transfer-backend mooncake` (or export `$PD_TRANSFER_BACKEND`); `nixl|mori|mooncake`. **Use `mooncake` for sglang** — `nixl` returns 200 OK but 0 output tokens on this RoCE/bnxt fabric (decode KV handoff fails). |
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
  (sandbox-side `optimize` flags / env; the Infera deployment is created by
  `create-infera` from these / consumed by `restart-server`, NOT injected
  into pods)
- `HYPERLOOM_MN_POLL_TIMEOUT_S`, `HYPERLOOM_MN_HEALTH_WAIT_S` (sandbox
  CLI poll budget, not a pod env)

Forward to `--rayjob-extra-env` **only** the prompt `env:` block lines
(`NCCL_DEBUG`, `PATH_TO_*` etc.). `OOB_SRC` / `INFERENCEX_PATH` /
`TRACELENS_ROOT` are sandbox-only and **must NOT** be forwarded (the
RayJob pod does not consume them — kernel-bench is NodeAffinity-pinned
to the head pod).

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
  --mn-image "${INFERENCE_OPTIMIZER_MN_IMAGE:?set from prompt multi-node image}" \
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
workload is still `Pending` — init-env / `head_pod_ip` / benchmarks must
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
2. **`init-env`** — once. Submits `init_rayjob_env.sh` via Ray Dashboard REST
   to verify `/opt/venv` and write `hyperloom-env.sh` (PATH only) on the
   head pod.
3. **`verify`** — once. Checks `ray` on PATH on the head pod.
   On `MISSING:`, re-run `init-env --print-logs`.
4. **`restart-server`** — every framework / model / TP / flag change.
   Kills the previous server via PID file (never `pkill -f`), relaunches
   under `nohup` so Ray pods do NOT restart and the aiter JIT cache
   survives. Issue ONE invocation per change; the CLI fans out across
   all pods on multi-node runs. Never issue per-pod invocations.
5. **`stop-multi-job`** — at session end. Always call explicitly for an
   auditable release. `ownerId` cascade is a safety net (sandbox
   deletion removes the SaFE workload and tears the RayJob down via
   owner-ref), not a substitute. `inference_optimizer optimize
   --nodes N>=2` does **not** call it on exit; nor does sandbox idle /
   hard-TTL GC distinguish "session in progress" from "session
   abandoned" — when the sandbox dies the RayJob is collateral, so an
   in-flight optimize loses access to head_pod_ip / service_url even
   if RayJob teardown lags a few seconds behind sandbox pod removal.

After step 4 route all benchmark / Magpie traffic to
`state.service_url` (head pod ClusterIP `:8888`). Re-read
`/tmp/multi_node_state.json` every turn; never cache `head_pod_ip` /
`service_url` across actions — RayJob recreate (or `stop-multi-job` then
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
  in sandbox env at CLI start (Brain injects). LLM keys are consumed in the
  sandbox only — not via `create-rayjob` workload env or `init_rayjob_env.sh`.
  **Never pass keys on the command line.**
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
  no `prune_branch`, no `kill_task`.
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
* `init-env` fails → `init-env --print-logs` once for the trace.
  The script (`multi_node/scripts/init_rayjob_env.sh`) is repo-owned and
  should not be edited from the agent — failures usually point at one
  of: the RayJob image lacking `/opt/venv`, or the head pod failing to
  reach an upstream package / model registry. Fix the root cause (image /
  network), then rerun; RayJob stays alive across the retry.
* `restart-server` hangs in health probe → rerun with `--no-wait-health`
  to detach, then `verify` / `curl state.service_url/health` to debug.
* Cluster-side cleanup → `stop-multi-job --delete --clear-state` (hard
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

* **Accuracy eval defaults on.** `_workload_envs.py` now defaults
  `RUN_EVAL=true`; setting `RUN_EVAL=false` is an explicit disable path and
  emits a warning. If a session records `baseline_accuracy=0.0`, the accuracy
  gate still skips because there is no baseline to compare against, so treat
  that as a missing-evidence warning rather than a clean accuracy pass. Before
  relying on multi-node accuracy eval, confirm InferenceX accepts the relevant
  flags via `grep -nR concurrent-requests "$INFERENCEX_PATH/benchmarks"` — if
  the grep is empty, eval may fail every variant including baseline.

* **Image-level launcher-flag denylist (probe each boot,
  framework-aware, model-agnostic)**. Any `backends` grid variant
  whose corresponding launcher CLI flag is not registered on the
  current RayJob image will fail at argparse regardless of model /
  TP. Image rebuilds happen out-of-band; do not carry a hard-coded
  skip list across sessions. On each fresh sandbox boot, after
  `init-env` succeeds, probe the framework launcher from the RayJob
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

## Interpreting a low-gain result

* A small validated gain with `kernel_opt_outcome="skip"` / `totals.attempted=0`
  (`reports/kernel_optimization_summary.json`) means the profiled step was
  host-bound. Kernel candidates come only from the TraceLens `analysis.md`, and
  `tracelens_analysis._evaluate_high_idle_gate` suppresses them when the step is
  mostly GPU-idle; only structural levers (GPU-graph capture + batching) apply.
* Under PD-disaggregation + DP-attention the per-rank steady-state batch can be
  bs1 even at high client concurrency, so TraceLens splits into
  `*_steady_state_..._bs1_conc1` windows. If every file in
  `tracelens/trace_split/` is `bs1_conc1`, the trace is host-bound and kernel_opt
  skips by design.
* Kernel candidates require a compute-bound profile (steady-state bsN, N>1). The
  lever is the per-rank step batch (DP/PD topology), not client concurrency. On
  multi-node this is automatic: a host-bound profile (high GPU idle + 0
  candidates) triggers one compute-bound re-profile with DP-attention/dp-size
  stripped (single DP rank, full per-step batch); candidates found are still
  validated on the served config. Disable with
  `HYPERLOOM_PROFILE_AUTO_COMPUTE_BOUND=0`.

## Disaggregated + DP-attention prerequisites

* `--enable-dp-attention` / `--enable-dp-lm-head` are no-ops without `--dp-size N`
  (N>1); sglang force-disables them at `dp_size==1`.
  `launch_infera_node._build_sglang_cmd` auto-injects `--dp-size = tp` when a
  dp-attention flag is present and `--dp-size` is absent (explicit value wins).
* With `dp_size>1` sglang binds one kv-events ZMQ socket per DP rank at
  `base+rank`; the image must ship `infera.common.net.free_tcp_port_block`. An
  image without it allocates a single port and decode crash-loops with
  `zmq.error.ZMQError: Address already in use` — rebuild the image from current
  Optimus.
* KV transfer backend default is `mori`.

## Readiness gate

`_wait_for_server_health_async` requires, in order: `/health` 200 ->
`/v1/models` non-empty -> `/v1/completions` (`max_tokens>=2`, `ignore_eos`)
returning `completion_tokens>=2`, twice. It fast-fails when `/v1/models` stays
empty for `HYPERLOOM_MN_MODELS_EMPTY_GRACE_S` (default 600s) after `/health` is
up (workers crashed on launch). Tunables:
`HYPERLOOM_MN_COMPLETION_PROBE_{COUNT,TOKENS,MIN_TOKENS}`,
`HYPERLOOM_MN_MODELS_EMPTY_GRACE_S`.
