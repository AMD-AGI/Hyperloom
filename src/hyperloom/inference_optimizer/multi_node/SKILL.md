---
name: hyperloom.inference_optimizer.multi_node
description: |
  Multi-node companion to the inference_optimizer skill. Use when the user
  prompt asks for inference optimization that needs more GPU / memory than
  a single pod provides (i.e. ``nodes >= 2``) — typical prompt signals are
  ``Nodes=N`` / ``N pods`` / ``TP=N`` larger than one pod's GPU count, or
  any model that cannot fit on one pod's GPUs. Drives a platform-provisioned
  multi-node cluster through the ``hyperloom.inference_optimizer.multi_node``
  Python CLI.
globs:
  - "**/multi_node/**"
  - "**/multi-node/**"
---

# Multi-Node Skill (infera + rayjob backends)

**The cluster is not created here.** The platform provisions it (an
`InferaDeployment` or a `RayJob`) before the optimizer starts and hands it over
through the `HYPERLOOM_MN_EXT_*` env vars (see "Cluster hand-off"). This CLI
drives an **already-running** cluster and never creates or tears one down; the
platform reclaims it when the session ends.

Drive every action through the Python CLI. **Never `ray.init`, `kubectl`, or
raw `curl` to a pod.** All state lives in the file resolved from
`$MULTI_NODE_STATE_FILE`, else
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/runtime/multi_node_state.json` — one
of the two MUST be set or the CLI raises (sandbox-local; synthesized from the
hand-off env on first use, rewritten by each subcommand). Re-read it
every turn — never cache `service_url` / `head_pod_ip` across actions (a
re-provisioned cluster rewrites them).

| Action         | Use               | Never                                     |
|----------------|-------------------|-------------------------------------------|
| Restart server | `restart-server`  | `kubectl exec … sglang.launch_server`     |
| Kill server    | `kill-inference`  | `pkill -f sglang`                         |

Bypassing loses exit-2 + `MULTI_NODE_FAILURE_SNAPSHOT={…}` failure detection,
cross-subcommand state, and `BENCHMARK_BASE_URL` plumbing for Magpie.

## Backends

Select via `optimize --mn-backend {infera|rayjob}` (default `rayjob`) or
`$INFERENCE_OPTIMIZER_MN_BACKEND`. Only active when `--nodes >= 2`; single-node
runs are unaffected. `restart-server` / `kill-inference` auto-route by
`state.backend`.

**infera** — `InferaDeployment` with **SSH** as the control plane. GPU pods
deploy idle (`mn-idle.sh` → sshd + block); `restart-server` SSHes in to
relaunch `infera.engine.{sglang,vllm}`, so the aiter JIT cache survives
restarts. No `bootstrap` / `verify` step. Benchmarks target the **Infera
frontend `:8000`** (`state.service_url`), never sglang rank-0.
* Image must carry the sshd layer (`mn-sshd-init.sh`, started by `mn-idle.sh`);
  sshd runs on `$MN_SSH_PORT` (base **2233**, not 22 — avoids colliding with
  node sshd on :2222). Under hostNetwork each GPU role binds a distinct port —
  prefill/worker `2233+N`, decode `2243+N` (via `LWS_WORKER_INDEX`) — so
  co-located roles don't collide.
* **Aggregated** (default): `serviceRoles=[frontend, worker]`,
  `worker.replica = nodes`.
* **PD-disaggregated**: `--pd-mode disaggregated --pd-prefill-nodes N
  --pd-decode-nodes M [--pd-prefill-tp/--pd-decode-tp]` →
  `serviceRoles=[frontend, prefill, decode]`. A role becomes a multi-node
  LeaderWorkerSet only when its TP exceeds one pod's GPUs.

**rayjob** — `RayJob` driven via the **Ray Dashboard** (`:8265` submit, `:6379`
GCS); no SSH. Uses `bootstrap` → `verify` → `restart-server` (below).

The topology (aggregated vs PD, per-role TP/EP, image, RDMA) is fixed by the
platform at provision time; the flags here only describe what `restart-server`
launches on the pods it was handed.

## Cluster hand-off (env-provided)

`HYPERLOOM_MN_EXT_SERVICE_URL` is what marks a cluster as available: without it
a `--nodes >= 2` run has nothing to drive and exits 2.

**Common (both backends)**

| Env var | Req? | Purpose |
| --- | --- | --- |
| `HYPERLOOM_MN_EXT_SERVICE_URL` | **yes** | HTTP(S) frontend for benchmarks (→ `BENCHMARK_BASE_URL`); its presence marks the cluster available |

**infera (`--mn-backend infera`)**

| Env var | Req? | Purpose |
| --- | --- | --- |
| `HYPERLOOM_MN_EXT_SSH_KEY` | **yes** | path to a private key already authorised on the pods (platform writes the file + bakes its public half at pod-create; never refreshed) |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` / `_WORKER_IPS` | **yes** (≥1) | comma-separated GPU pod IPs. PD uses `_PREFILL_IPS` + `_DECODE_IPS`; aggregated uses `_WORKER_IPS` |
| `HYPERLOOM_MN_EXT_SSH_PORT` | no | SSH base port (default **2233**; decode role-offset +10) |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | no | known_hosts path (else lax host-key check) |

infera **requires** SSH (`_SSH_KEY` + ≥1 `*_IPS`); missing → fails fast
(**exit 3**, config error), no degrade.

Which list is read is decided by `$PD_MODE` alone, never guessed from which
lists you set: unset means aggregated, so a PD cluster that exports
`_PREFILL_IPS` / `_DECODE_IPS` without `PD_MODE=disaggregated` reads as having
no GPU pods at all. `optimize` exports it from `--pd-mode`; a standalone
`hyperloom-mn` subcommand needs it in the environment.

**rayjob (`--mn-backend rayjob`)** — ignores the infera `_SSH_*` / `*_IPS` vars.

| Env var | Req? | Purpose |
| --- | --- | --- |
| `HYPERLOOM_MN_EXT_HEAD_IP` | recommended | Ray head IP → Dashboard `:8265` + GCS `:6379`; enables per-round `restart-server`. Omit ⇒ **benchmark-only** (restarts no-op). Must be the `<rayCluster>-head-svc`, not the workload's single-port Service |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | no | Dashboard auth token (only if authenticated) |

**Companion vars** (set by `optimize` from CLI flags; usually leave alone):
`INFERENCE_OPTIMIZER_NODES`, `INFERENCE_OPTIMIZER_GPUS_PER_NODE`, `PD_MODE`,
`PD_PREFILL_NODES` / `PD_DECODE_NODES`, `INFERENCE_OPTIMIZER_MN_BACKEND`.

Example (infera PD-disaggregated):

```bash
export HYPERLOOM_MN_EXT_SERVICE_URL=http://<frontend-host>:8000
export HYPERLOOM_MN_EXT_PREFILL_IPS=<prefill-ip> HYPERLOOM_MN_EXT_DECODE_IPS=<decode-ip>
export HYPERLOOM_MN_EXT_SSH_KEY=/path/to/id_ed25519
inference_optimizer optimize --model <path> --nodes 2 \
  --mn-backend infera --pd-mode disaggregated --tp 8 --ep 8 ...
```

Example (rayjob with per-round restart):

```bash
export HYPERLOOM_MN_EXT_SERVICE_URL=http://<ray-serve-or-head-url>:<port>
export HYPERLOOM_MN_EXT_HEAD_IP=<ray-head-svc-host>
inference_optimizer optimize --model <path> --nodes 2 --mn-backend rayjob --tp 8 --ep 8 ...
```

## Subcommands

```bash
# rayjob only — infera skips these two:
python3 -m hyperloom.inference_optimizer.multi_node bootstrap [--print-logs]  # verify /opt/venv + write PATH env on head
python3 -m hyperloom.inference_optimizer.multi_node verify                    # check `ray` on PATH on head
# both backends:
python3 -m hyperloom.inference_optimizer.multi_node restart-server --framework <sglang|vllm> --model <path> --tp <N> [--ep <N>] [--extra-args "…"]
python3 -m hyperloom.inference_optimizer.multi_node kill-inference
```

Run `<subcommand> --help` for the full flag set. **Do not invent flags.** There
is no create/stop subcommand. Issue ONE `restart-server` per framework / model /
TP / flag change — the CLI fans out across all pods; never per-pod invocations.

> **If you are running the `optimize` CLI** (handed `--target-gain` /
> `--max-hours` / `--isl/--osl/--conc` FLAGS), `optimize` performs the entire
> flow internally (adopt cluster → bootstrap → restart per round). Run
> **`optimize` ONLY** — do not also run standalone `bootstrap` / `restart-server`
> alongside it, or the two resolve different state files and fight over the pods.
> The manual subcommands are only for driving a cluster *without* `optimize`.

### Prompt → CLI mapping (do not re-ask; do not repeat in chat)

| User prompt field | Launcher action |
|---|---|
| `Nodes=N` | `optimize --nodes N` (must match the handed-over cluster) |
| `TP=N`, `EP=…` | `optimize --tp N` / `--ep`; `restart-server --tp N`. **Always set `--tp`** (default 1); for PD use the per-role TP |
| `MN_BACKEND=infera` | `optimize --mn-backend infera` |
| `PD_MODE=disaggregated` | `optimize --pd-mode disaggregated` — **must be a flag**; `$PD_MODE` env is ignored (stale-env guard). Omit ⇒ aggregated |
| `PD_PREFILL_NODES` / `PD_DECODE_NODES` | `--pd-prefill-nodes` / `--pd-decode-nodes` (or export) |
| `PD_PREFILL_TP` / `PD_DECODE_TP` | `--pd-prefill-tp` / `--pd-decode-tp` (default = `--tp`) |
| `PD_PREFILL_EP` / `PD_DECODE_EP` | **infera PD only**, export as `restart-server` defaults; `0` ⇒ shared `--ep` |
| `PD_PREFILL_EXTRA_ARGS` / `PD_DECODE_EXTRA_ARGS` | **infera PD only**, export; appended per-role AFTER shared `--extra-args` (role wins on dup keys) |
| `PD_TRANSFER_BACKEND` | `--pd-transfer-backend {nixl\|mori\|mooncake}`. **Use `mooncake` for sglang** (see below) |
| `ISL`/`OSL`/`CONC`/`PRECISION` | `optimize --isl/--osl/--conc/--precision` |
| `--mn-image …` / `RayJob image:` / `Infera image:` | **Not an `optimize` flag** — the image is the platform's input; pods already exist by the time the optimizer runs |
| prompt `env:` lines (`NCCL_DEBUG`, `PATH_TO_*_TAR_PACKAGE`, …) | Pod env is baked by the platform at provision time — pass to the platform, not this CLI |

**Sandbox-only exports** (consumed by `install.sh` / `optimize` /
`_workload_envs.py` in the **sandbox**; never ask the platform to bake them into
pod env — nothing in the pods reads them, and they shadow real pod values):
`KERNEL_AGENT_BUILD_GEAK_RAG_INDEX`, `KERNEL_OPT_*`, `RANDOM_RANGE_RATIO`,
`RUN_EVAL`, the `optimize`-flag mirrors (`MODEL_PATH`, `TP`, `EP`, …), all
`PD_*` knobs, and `HYPERLOOM_MN_POLL_TIMEOUT_S` / `HYPERLOOM_MN_HEALTH_WAIT_S`.

### KV transfer backend (PD)

`--pd-transfer-backend` selects the PD KV plane, on `optimize` or directly on
`restart-server`; `optimize` mirrors it to `$PD_TRANSFER_BACKEND`, which later
rounds fall back on. There is no state-file source: the cluster is handed over
rather than created here, so nothing records a backend on your behalf.

Leaving it unset is fine, and usually right. The flag is then simply not passed
and sglang applies its own default, which on the RoCE/bnxt fabric is `mooncake`
-- the one to **prefer** here, since it auto-detects the RDMA device. Set the
flag only to move off that: `nixl` returns HTTP 200 but 0 output tokens (decode
KV handoff via UCX/nixl fails), and `mori` is an alternative but not the sglang
default here.

### Poll budget (MoE JIT cold-start)

First `restart-server` on a large MoE often needs **20–30 min** (weight load +
aiter JIT), but the default per-invocation poll is ~110s. Export before
`restart-server` / `optimize --nodes >= 2`:

```bash
export HYPERLOOM_MN_POLL_TIMEOUT_S=1800
export HYPERLOOM_MN_HEALTH_WAIT_S=1800
```

On timeout, re-run the **same** subcommand (no `while sleep` wrapper);
`restart-server` checkpoints `last_restart_submission_id` and resumes the
in-flight launch (`MULTI_NODE_RESTART_RESUME_RUNNING=1`, default).

## Hard Rules

* **Sandbox never runs the inference server.** When `nodes >= 2`, sglang/vllm
  live only on the pods; the sandbox is the client. Every Magpie launch MUST
  inherit `BENCHMARK_BASE_URL=<state.service_url>` (forces `PHASE=client`).
  Missing it → Magpie runs `sglang.launch_server` on the CPU sandbox →
  `ModuleNotFoundError`. Fix orchestrator env propagation; never
  `pip install sglang` in the sandbox.
* **Credentials stay in the sandbox** — never forward LLM keys into pod env /
  `bootstrap.sh`, never pass keys on the command line.
* **No Ray Python client in the orchestration layer** — `multi_node/` uses Ray
  Dashboard REST only, never `import ray` / `ray.init(address=…)` against the
  inference RayJob. (Code running *inside* RayJob pods is exempt.)
* **Kernel-agent fan-out** — kernel edits in the sandbox do NOT reach the pods
  (per-pod local fs). Use `apply-patch` / `revert-patch` / `kernel-bench` (route
  by `state.backend`; on infera GEAK runs on a GPU pod over SSH, installed once
  per cluster via `install-geak`). The integrate path auto-restarts the server
  after `apply-patch` — do not restart manually.
* **Robustness runs the real agent on `nodes >= 2`, with LocalProbe off.** The
  probe only sees sandbox-local resources, so on multi-node every probe-derived
  symptom (`ray_head_dead`, `local_server_unreachable`, `gpu_memory_leaked`, …)
  would be a false positive; `disable_local_probe` defaults to True there and
  swaps the probe for a silent stub. What remains is the node-agnostic set the
  agent reads straight off the Coordinator prompt and inbox: the deadline /
  budget ladder, `gain_plateau`, `no_levers_found`, crash escalation,
  `phase_budget_nearly_exhausted`, `conversation_no_progress`, plus the
  inbox-driven `agent_stall` / `repeated_failure` / `repeated_policy_denied`
  family. Pass `--robustness-mock` for heartbeat-only. Shell-level health
  monitoring (`optimizer_runs/robustness_monitor.sh`, auto-resume on terminal
  `stop_reason`) is unaffected.

## Exit Codes

| Code | Meaning | Controller action |
|-----:|---------|--------------------|
| 0 | success | continue |
| 1 | transient (poll timeout / network / unknown) | rerun the SAME subcommand |
| 2 | cluster unusable (no hand-off, infera without SSH, pods gone) | DO NOT retry; stderr may carry `MULTI_NODE_FAILURE_SNAPSHOT={…}` |
| 3 | config error (missing env / arg) | fix args/env, then rerun |
| 130 | SIGINT | user aborted |

## When Something Looks Wrong

* **Exit 2 "no cluster"** → hand-off env missing. Check
  `HYPERLOOM_MN_EXT_SERVICE_URL` (+ infera `_SSH_KEY` and one `*_IPS`);
  provisioning is upstream of this CLI.
* **`ModuleNotFoundError: sglang`** + stderr shows `sglang.launch_server` on the
  sandbox → the client-only rule above tripped; fix env propagation.
* **`restart-server` succeeded then `/health` timed out** → framework early-exit;
  the driver exits 2 + `MULTI_NODE_FAILURE_SNAPSHOT={kind:"framework_early_exit"}`
  and the `rank_0.log` tail names the cause.
* **Cross-node NCCL/RCCL collective hangs → watchdog abort, weights never load**
  (server GPU mem stays near 0) → the pods lack a usable RDMA device
  (`ibv_devices` empty, no `/dev/infiniband`) while NCCL is forced onto IB
  (`NCCL_IB_HCA` set). This is a **platform provisioning** issue — the pods must
  request `rdma/hca` (a partial-GPU multi-node task must still get RDMA). To
  confirm/unblock, force TCP by exporting
  `HYPERLOOM_MN_EXTRA_FWD_ENV='{"NCCL_IB_DISABLE":"1"}'` before `optimize`: that
  is the channel both backends forward to the pods (rayjob via the job
  `runtime_env`, infera via the SSH fan-out). Slow but works. **Not**
  `--extra-env` — that reaches the variant filter only and never leaves the
  sandbox, so using it here looks like a clean result and rules nothing out.
* **Variant aborts with no benchmark output** → read `failed_variants` in the
  round's `<action>_attempts.extras`, or the per-variant `abort_reason.json`
  under `<session_dir>/runs/<action>/<task_id>/variant_<NN>_<name>/`
  (`error_class` + error tail); `session_dir` is the per-run timestamped dir
  (`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`), not `$USER_DATA_PATH` itself.
* **Launcher-flag rejection** → an explore grid variant whose CLI flag isn't
  on the current image fails at argparse. Probe the launcher from a GPU pod
  (`sglang.launch_server --help` / `vllm serve --help`) and `--skip-variants`
  the missing ones; drop the skip when the probe shows the flag again. Do not
  carry a hard-coded skip list across sessions or model classes.

## Interpreting a low-gain result

* Small validated gain with a rewrite-controller
  `status="no_opportunity"` / `patch_count=0` → inspect the controller summary
  and TraceLens evidence. A host-bound, GPU-idle step usually favors structural
  levers such as graph capture and batching over source-level kernel rewrites.
* Under PD + DP-attention the per-rank steady-state batch can be bs1 even at
  high client concurrency; if every `tracelens/trace_split/` file is
  `bs1_conc1`, the trace is host-bound and offers little compute-bound rewrite
  evidence. Multi-node auto-re-profiles with DP-attention stripped (disable with
  `HYPERLOOM_PROFILE_AUTO_COMPUTE_BOUND=0`).

## Disaggregated + DP-attention prerequisites

* `--enable-dp-attention` / `--enable-dp-lm-head` are no-ops without
  `--dp-size N` (N>1); `launch_infera_node` auto-injects `--dp-size = tp` when a
  dp-attention flag is present and `--dp-size` is absent (explicit value wins).
* With `dp_size>1` sglang binds one kv-events ZMQ socket per DP rank; the image
  must ship `infera.common.net.free_tcp_port_block`, else decode crash-loops
  with `zmq.error.ZMQError: Address already in use` — rebuild from current
  Optimus.

## Readiness gate

`_wait_for_server_health_async` requires, in order: `/health` 200 →
`/v1/models` non-empty → `/v1/completions` (`max_tokens>=2`, `ignore_eos`)
returning `completion_tokens>=2`, twice. It fast-fails when `/v1/models` stays
empty for `HYPERLOOM_MN_MODELS_EMPTY_GRACE_S` (default 600s) after `/health` is
up (workers crashed on launch). Tunables:
`HYPERLOOM_MN_COMPLETION_PROBE_{COUNT,TOKENS,MIN_TOKENS}`,
`HYPERLOOM_MN_MODELS_EMPTY_GRACE_S`, `HYPERLOOM_MN_HEALTH_WAIT_S`.
