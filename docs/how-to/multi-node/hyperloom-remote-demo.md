# Hyperloom Remote Demo — Multi-Node Inference Optimization

Optimize inference on a multi-node GPU cluster (`--nodes >= 2`): Hyperloom
benchmarks a long-lived remote server, restarts it with new settings each round,
and improves throughput.

To run it, hand the agent (or Primus-Claw) the
[workload skill](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md),
which carries the exact `optimize` flags and environment for either backend,
and the agent launches and monitors the run for you. This page explains what
that skill contains and what each variable means.

## Two backends

Pick one; both serve the same model, differ in control plane:

- **infera**: GPU pods idle with sshd; Hyperloom SSHes in to (re)launch sglang
  each round. Supports *PD disaggregation* (separate prefill / decode pods).
- **rayjob**: pods run under Ray; restarts go through the Ray dashboard.

## Hyperloom does not create the cluster

The platform provisions the GPU pods before the optimizer starts — normally
**[Primus-Claw](https://github.com/AMD-AGI/Primus-Claw)** — and hands the
running cluster over through `HYPERLOOM_MN_EXT_*` env vars. Hyperloom
adopts it, benchmarks it, and restarts the server on it each round. It never
creates or releases the cluster, so the container image and per-pod CPU/memory
are the platform's inputs, not `optimize` flags.

Without a hand-off, a `--nodes >= 2` run exits 2.

| Variable | Backend | Required | Meaning |
|----------|---------|----------|---------|
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | Benchmark frontend URL (`http(s)://…`); use the port in this URL — the platform assigns it (not a fixed `:8000`) |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` | infera | PD | Prefill / decode pod IPs (comma-separated, in rank order) |
| `HYPERLOOM_MN_EXT_WORKER_IPS` | infera | aggregated | Worker pod IPs (comma-separated, leader first) |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | **yes** | Path to a private key authorized on the pods |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | no | SSH base port (default `2233`; decode is role-offset `+10`, and each pod of a multi-node role adds its rank) |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | no | known_hosts path (else relaxed host-key check) |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | no | Host serving the Ray control plane — Dashboard `:8265` + GCS `:6379`, i.e. KubeRay's `<rayCluster>-head-svc`; omit → benchmark-only |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | no | Ray Dashboard auth token if required |

- **infera** requires `SSH_KEY` + at least one `*_IPS`, else `optimize` fails
  fast (`sys.exit(2)`).
- **rayjob** uses `HEAD_IP` for restarts (not SSH); without it, benchmark-only.

## Shared filesystem (mandatory)

Multi-node *can't run* without a cluster-wide shared mount (`NFS_SHARED_ROOT`,
NFS or equivalent) visible at the *same absolute path* on the sandbox and every
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
from the environment (platform-injected).

---

## The workload skill (what you hand the agent)

Give the agent the pinned skill:

```text
Use the skill at docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md
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
| `--nodes` | Node count (`>= 2` for multi-node); must match the cluster handed over |
| `--mn-backend` | `infera` or `rayjob` |
| `--gpus-per-node` | GPUs per pod (default 8); feeds the TP feasibility check |
| `--tp` / `--ep` | Tensor / expert parallel (MoE: often `ep == tp`) |
| `--isl` / `--osl` / `--conc` | Benchmark input/output length and concurrency (flags only — `ISL`/`OSL`/`CONC` env vars are ignored and overwritten) |
| `--gpu-type` / `--precision` | Target GPU (e.g. `mi325x`) and dtype (`bf16`) — flags (env is not an authoritative source; `--gpu-type` is re-exported for subprocesses) |
| `--framework` | `sglang` |
| `--target-gain` / `--max-hours` | Optimization goal and time budget |
| `--pd-mode disaggregated` + `--pd-prefill-*` / `--pd-decode-*` | infera PD split; must describe the topology the platform actually provisioned |
| `--pd-transfer-backend mooncake` | PD KV transfer plane (prefer `mooncake`; `nixl` can yield 0 output tokens) |
| `--server-args "..."` | Shared sglang args applied on every restart |
| `--no-framework-agent` | Skip the framework-tuning agent |

**Environment** (exported before `optimize`; sandbox-side):

| Var | Meaning |
|-----|---------|
| `RANDOM_RANGE_RATIO` | Benchmark sequence-length jitter (env has a fallback; `ISL`/`OSL`/`CONC`/`GPU_TYPE`/`PRECISION` are flags — see FLAGS above) |
| `INFERENCEX_PATH` / `MAGPIE_PATH` / `TRACELENS_ROOT` | Tool checkouts under `${NFS_SHARED_ROOT}` |
| `SGLANG_DISAGGREGATION_*_TIMEOUT` | PD bootstrap / wait timeouts (Workload A only) |
| `FORGE_PATH` | Kernel-Forge checkout, for the Kernel-Forge kernel backend (Workload B) |

Platform-injected (do **not** set): every `HYPERLOOM_MN_EXT_*` var above, plus
`USER_DATA_PATH`.

Example (infera + PD disaggregated) once the platform has handed a cluster over:

```bash
inference_optimizer optimize --model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
  --nodes 2 --mn-backend infera --pd-mode disaggregated \
  --pd-prefill-nodes 1 --pd-decode-nodes 1 --tp 8 --ep 8 \
  --pd-transfer-backend mooncake ...
```

For rayjob, swap `--mn-backend rayjob`; the platform supplies
`HYPERLOOM_MN_EXT_HEAD_IP` instead of the SSH/IPs vars.

---

## Related topics

- [Workload skill (copy-paste flags/env)](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md)
- [Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) — provisions the cluster and hands it over
- [Multi-node CLI, SSH & hand-off semantics](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/multi_node/SKILL.md)
- [Local single-node counterpart](https://github.com/AMD-AGI/Hyperloom/blob/main/examples/README.md)
