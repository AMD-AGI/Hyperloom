# Hyperloom Remote Demo — Multi-Node Inference Optimization

Optimize inference on a **multi-node** GPU cluster (`--nodes >= 2`): Hyperloom
benchmarks a long-lived remote server, restarts it with new settings each round,
and improves throughput.

**How you run it:** hand the agent (or Primus-Claw) the **workload skill** at
`examples/hyperloom-remote-mn-qwen3-30b/SKILL.md` — it carries the exact
`optimize` flags and environment for either backend, and the agent launches and
monitors the run for you. This page explains what that skill contains and what
each variable means.

## Two backends

Pick one; both serve the same model, differ in control plane:

- **infera** — GPU pods idle with sshd; Hyperloom SSHes in to (re)launch sglang
  each round. Supports **PD disaggregation** (separate prefill / decode pods).
- **rayjob** — pods run under Ray; restarts go through the Ray dashboard.

Requires `--mn-image` pointing at an operator-supplied image
(`<INFERA_SSHD_IMAGE>` must include sshd; `<RAYJOB_IMAGE>` is a standard Ray
image).

## Two ways to connect the cluster

- **Path A — SaFE-managed** *(default)*: the optimizer runs in a **Primus-SaFE**
  sandbox with `SAFE_API_URL` + `SAFE_API_KEY` (both auto-injected by the
  platform). Hyperloom creates the GPU pods (`create-infera` / `create-rayjob`)
  and tears them down with `stop-multi-job`.
  **[Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE)** is AMD's open-source
  training and inference management platform.
- **Path B — External**: the cluster is **already running** with **no SaFE
  API**; you point Hyperloom at it with `HYPERLOOM_MN_EXT_*` env vars (see below).

If both SaFE creds and `HYPERLOOM_MN_EXT_*` are set, **SaFE wins** (Path A).

## Shared filesystem (mandatory, both paths)

Multi-node **cannot run** without a cluster-wide shared mount (`NFS_SHARED_ROOT`,
NFS or equivalent) visible at the **same absolute path** on the sandbox and every
GPU pod. It holds model weights, tool checkouts, and session artifacts /
profiler traces; `USER_DATA_PATH` normally lives under it too, so both sides read
the same files.

```text
${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B/   # --model
${NFS_SHARED_ROOT}/InferenceX/             # INFERENCEX_PATH
${NFS_SHARED_ROOT}/Magpie/                 # MAGPIE_PATH
${NFS_SHARED_ROOT}/TraceLens/              # TRACELENS_ROOT
${NFS_SHARED_ROOT}/TraceLens-internal/     # TRACELENS_INTERNAL_ROOT (optional)
```

## What the agent produces

A session under `$USER_DATA_PATH/<model_basename>/<UTC_timestamp>/` with launcher
logs and a persisted `state.json` (holds `phase`, `cumulative_gain`,
`crash_count`, `stop_reason`) for status and `--resume`. `$USER_DATA_PATH` comes
from the environment (platform-injected under SaFE).

---

## The workload skill (what you hand the agent)

Give the agent the pinned skill:

```text
Use the skill at examples/hyperloom-remote-mn-qwen3-30b/SKILL.md
```

It contains two ready-to-run blocks — **Workload A (infera + PD)** and
**Workload B (rayjob)** — each a `FLAGS` list plus an `Environment` block. The
agent runs `inference_optimizer optimize` with them, reports session id / log /
PID, and monitors `state.json` until a terminal `stop_reason`.

### Variables in those blocks

**FLAGS** (passed to `optimize`):

| Flag | Meaning |
|------|---------|
| `--model` | Model path under `${NFS_SHARED_ROOT}/models/...` (or HF id) |
| `--nodes` | Node count (`>= 2` for multi-node) |
| `--mn-backend` | `infera` or `rayjob` |
| `--mn-image` | Container image (`<INFERA_SSHD_IMAGE>` needs sshd; `<RAYJOB_IMAGE>`) |
| `--gpus-per-node` / `--cpus-per-node` / `--mem-per-node` | Per-pod resources (default GPUs 8) |
| `--tp` / `--ep` | Tensor / expert parallel (MoE: often `ep == tp`) |
| `--framework` | `sglang` |
| `--target-gain` / `--max-hours` | Optimization goal and time budget |
| `--pd-mode disaggregated` + `--pd-prefill-*` / `--pd-decode-*` | infera PD split (prefill + decode roles; node counts sum to `--nodes`) |
| `--pd-transfer-backend mooncake` | PD KV transfer plane (prefer `mooncake`; `nixl` can yield 0 output tokens) |
| `--server-args "..."` | rayjob shared sglang args (applied each restart) |
| `--no-framework-agent` | Skip the framework-tuning agent |

**Environment** (exported before `optimize`; sandbox-side):

| Var | Meaning |
|-----|---------|
| `GPU_TYPE` / `PRECISION` | Target GPU (e.g. `mi325x`) and dtype (`bf16`) |
| `ISL` / `OSL` / `CONC` / `RANDOM_RANGE_RATIO` | Benchmark input/output length, concurrency, length jitter |
| `INFERENCEX_PATH` / `MAGPIE_PATH` / `TRACELENS_ROOT` | Tool checkouts under `${NFS_SHARED_ROOT}` |
| `TRACELENS_INTERNAL_ROOT` | Optional `TraceLens-internal` checkout |
| `SGLANG_USE_AITER` / `SGLANG_AITER_MLA_PERSIST` | Enable + persist the aiter kernel path |
| `SGLANG_DISAGGREGATION_*_TIMEOUT` | PD bootstrap / wait timeouts (infera PD only) |

Platform-injected (do **not** set): `SAFE_API_URL`, `SAFE_API_KEY`,
`SAFE_WORKSPACE`, `WORKLOAD_ID`, `DISPLAY_NAME`, and `USER_DATA_PATH`.

---

## Path B — External mode (no SaFE)

Same run as Path A, but you point Hyperloom at an already-running cluster instead
of letting SaFE provision it. Unset the SaFE creds, set
`HYPERLOOM_MN_EXT_SERVICE_URL`, and Hyperloom skips provisioning and benchmarks
that URL directly. The same FLAGS / Environment blocks apply.

| Variable | Backend | Required | Meaning |
|----------|---------|----------|---------|
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | Benchmark frontend URL (`http(s)://…`; infera frontend typically `:8000`) |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` | infera | PD | Prefill / decode pod IPs (comma-separated) |
| `HYPERLOOM_MN_EXT_WORKER_IPS` | infera | aggregated | Worker pod IPs (comma-separated) |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | **yes** | Private SSH key already authorized on the pods |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | no | SSH base port (default `2233`; decode +10) |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | no | known_hosts path (else relaxed host-key check) |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | no | Ray head IP (Dashboard `:8265`, GCS `:6379`); omit → benchmark-only |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | no | Ray Dashboard auth token if required |

- **infera** requires `SSH_KEY` + at least one `*_IPS`, else `optimize` fails
  fast (`sys.exit(2)`).
- **rayjob** uses `HEAD_IP` for restarts (not SSH); without it, benchmark-only.

Example (infera + PD disaggregated):

```bash
unset SAFE_API_URL SAFE_API_KEY
export HYPERLOOM_MN_EXT_SERVICE_URL=http://<frontend-host>:8000
export HYPERLOOM_MN_EXT_PREFILL_IPS=<prefill-ip>  HYPERLOOM_MN_EXT_DECODE_IPS=<decode-ip>
export HYPERLOOM_MN_EXT_SSH_KEY=/path/to/id_ed25519

inference_optimizer optimize --model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
  --nodes 2 --mn-backend infera --pd-mode disaggregated \
  --pd-prefill-nodes 1 --pd-decode-nodes 1 --tp 8 --ep 8 \
  --pd-transfer-backend mooncake ...
```

For rayjob, swap `--mn-backend rayjob` and set `HYPERLOOM_MN_EXT_HEAD_IP` instead
of the SSH/IPs vars.

---

## Further reading

- **Workload skill (copy-paste flags/env):**
  `examples/hyperloom-remote-mn-qwen3-30b/SKILL.md`
- **Primus-SaFE:** [github.com/AMD-AGI/Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE)
- **Multi-node CLI, SSH & external-mode semantics:**
  `src/hyperloom/inference_optimizer/multi_node/SKILL.md`
- **Local single-node counterpart:** `examples/hyperloom-local-demo.md`
