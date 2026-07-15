---
name: hyperloom-remote-mn-qwen3-30b
description: |
  Run a 4-hour multi-node Hyperloom Qwen3-30B-A3B optimization on Primus SaFE
  (Infera PD-disaggregated or RayJob aggregated). Use when the user wants a
  remote SaFE sandbox demo with --nodes 2 and sglang MoE tuning on MI325X.
---

# Hyperloom Remote Multi-Node — Qwen3-30B-A3B (SaFE)

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Before launching:

1. Optimizer skill (mandatory): `@${HYPERLOOM_SKILL_PATH}` — if unset, fall back
   to `@../../src/hyperloom/inference_optimizer/SKILL.md` relative to this repo.
2. Multi-node companion (when `--nodes >= 2`):
   `@../../src/hyperloom/inference_optimizer/multi_node/SKILL.md`
3. Full walkthrough: `@../hyperloom-remote-demo.md`

Paths resolve from `HYPERLOOM_ROOT`, `HYPERLOOM_SKILL_PATH`, `NFS_SHARED_ROOT`
(see below), or operator-provided values in the prompt.

This skill pins the **workload**, **CLI flags**, and **launch constraints** for
a SaFE remote run. Valid flags are those listed by `optimize --help` and the
`multi_node` subcommand `--help`.

## NFS shared filesystem (mandatory)

Remote multi-node **cannot run** without a **cluster-wide NFS (or equivalent
shared) mount** visible at the **same absolute path** from:

| Consumer | Why |
|----------|-----|
| Optimizer sandbox | `USER_DATA_PATH`, session artifacts, torch profiler traces, server logs |
| SaFE GPU pods | Model weights (`--model`), shared tool checkouts |

The platform (or operator) must export a single root, e.g. `NFS_SHARED_ROOT`.
All paths below are **under that root** — replace `<nfs>` with the operator's
mount point (example layout only):

```text
<nfs>/models/Qwen3-30B-A3B/     # --model
<nfs>/InferenceX/               # INFERENCEX_PATH
<nfs>/Magpie/                   # MAGPIE_PATH
<nfs>/TraceLens/                # TRACELENS_ROOT
<nfs>/TraceLens-internal/       # TRACELENS_INTERNAL_ROOT (optional)
```

`NFS_SHARED_ROOT` (or equivalent) is required; local-only paths break
multi-node restart, profiling, and Magpie client benchmarks.

Verify from the sandbox before launch:

```bash
test -n "$NFS_SHARED_ROOT" && test -d "$NFS_SHARED_ROOT" && echo "NFS_SHARED_ROOT OK: $NFS_SHARED_ROOT" \
  || echo "ERROR: NFS_SHARED_ROOT missing — cannot run remote multi-node"
```

## Run Mode

**Path A (default):** SaFE-managed remote sandbox. **Path B:** external cluster
— env reference: `@../hyperloom-remote-demo.md` § Path B; CLI semantics:
`multi_node/SKILL.md` § External mode.

| Check | Path A | Path B |
|-------|--------|--------|
| `USER_DATA_PATH` | Platform-injected, writable (kept as-is) | Writable dir (operator choice) |
| `SAFE_API_URL` + `SAFE_API_KEY` | Must be present | Must be **unset** |
| `SAFE_WORKSPACE` | Must be present for workload create | N/A |
| Benchmark / restart | SaFE provision + `BENCHMARK_BASE_URL` | `HYPERLOOM_MN_EXT_*` env vars |
| Optimizer runs on | CPU sandbox pod; GPUs on SaFE workloads | Sandbox or host; GPUs pre-provisioned |

Session layout:

```text
$USER_DATA_PATH/                          # workspace root (platform)
└── <model_basename>/<UTC_ts>/           # session_dir (optimizer)
    ├── manifest.json, state.json
    └── runtime/multi_node_state.json
```

## Backend Selection

Two backends are supported:

| Backend | Inputs | Image |
|---------|--------|-------|
| **infera** (default for this skill's PD recipe) | `--mn-backend infera`; Workload A adds `--pd-mode disaggregated` and PD flags | `--mn-image <INFERA_SSHD_IMAGE>` |
| **rayjob** | `--mn-backend rayjob` | `--mn-image <RAYJOB_IMAGE>` |

When `--mn-backend infera`, use `--pd-transfer-backend mooncake` for PD
disaggregation.

---

## Workload A — Infera + PD disaggregation

Use when `--mn-backend infera` and `--pd-mode disaggregated`.

### FLAGS (keep as `--flags`)

```text
--model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
--target-gain 30 \
--max-hours 4 \
--mn-backend infera \
--framework=sglang \
--nodes 2 \
--gpus-per-node 8 \
--cpus-per-node 90 \
--mem-per-node 1024 \
--tp 8 --ep 8 \
--mn-image <INFERA_SSHD_IMAGE> \
--pd-mode disaggregated \
--pd-prefill-nodes 1 --pd-decode-nodes 1 \
--pd-prefill-tp 8 --pd-decode-tp 8 \
--pd-prefill-ep 8 --pd-decode-ep 8 \
--pd-transfer-backend mooncake \
--pd-prefill-extra-args "--attention-backend aiter --mem-fraction-static 0.78 --disable-radix-cache --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --trust-remote-code" \
--pd-decode-extra-args "--attention-backend aiter --mem-fraction-static 0.82 --enable-dp-attention --deepep-mode normal --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024 --trust-remote-code" \
--no-framework-agent
```

### Environment (sandbox — export before `optimize`)

```text
GPU_TYPE=mi325x
PRECISION=bf16
ISL=1024
OSL=1024
CONC=128
RANDOM_RANGE_RATIO=0.8
KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
SGLANG_USE_AITER=1
SGLANG_AITER_MLA_PERSIST=1
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
TRACELENS_INTERNAL_ROOT=${NFS_SHARED_ROOT}/TraceLens-internal
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
NODE_TLS_REJECT_UNAUTHORIZED=0
```

Optional MoE cold-start poll budget (large models):

```bash
export HYPERLOOM_MN_POLL_TIMEOUT_S=1800
export HYPERLOOM_MN_HEALTH_WAIT_S=1800
```

---

## Workload B — RayJob aggregated

Use when `--mn-backend rayjob` (no PD flags).

### FLAGS (keep as `--flags`)

```text
--model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
--target-gain 30 \
--max-hours 4 \
--mn-backend rayjob \
--framework=sglang \
--nodes 2 \
--gpus-per-node 8 \
--cpus-per-node 90 \
--mem-per-node 1024 \
--tp 8 --ep 8 \
--no-framework-agent \
--mn-image <RAYJOB_IMAGE> \
--server-args "--attention-backend aiter --mem-fraction-static 0.8 --enable-dp-attention --deepep-mode normal --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024 --trust-remote-code"
```

### Environment (sandbox — export before `optimize`)

```text
GPU_TYPE=mi325x
PRECISION=bf16
ISL=1024
OSL=1024
CONC=128
RANDOM_RANGE_RATIO=0.8
KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0
SGLANG_USE_AITER=1
SGLANG_AITER_MLA_PERSIST=1
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
TRACELENS_INTERNAL_ROOT=${NFS_SHARED_ROOT}/TraceLens-internal
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
NODE_TLS_REJECT_UNAUTHORIZED=0
```

---

## Minimum Multi-Node Inputs

| Parameter | Infera | RayJob |
|-----------|--------|--------|
| `NFS_SHARED_ROOT` | **required** (shared NFS mount) | **required** |
| `--nodes` | `2` | `2` |
| `--mn-backend` | `infera` | `rayjob` |
| `--mn-image` | `<INFERA_SSHD_IMAGE>` | `<RAYJOB_IMAGE>` |
| `--gpus-per-node` | `8` | `8` |
| `--tp` / `--ep` | `8` / `8` | `8` / `8` |
| `--model` | `${NFS_SHARED_ROOT}/models/...` | same |
| PD flags | see Workload A | N/A (aggregated) |
| Pod env at create | `create-infera --extra-env` (operator-provided) | `--rayjob-extra-env` (operator-provided) |

Platform-injected (not passed on the CLI): `SAFE_API_URL`, `SAFE_API_KEY`,
`SAFE_WORKSPACE`, `WORKLOAD_ID`, `DISPLAY_NAME`.

---

## Launch & State

- Artifacts are written under `$USER_DATA_PATH`; the session dir is
  `$USER_DATA_PATH/<model_basename>/<UTC_ts>/` and is writable.
- `inference_optimizer optimize` runs in the background (`setsid nohup`).
  `session ID`, `log path`, and `PID` come from the `HYPERLOOM_LAUNCH` line or
  the `--launch-info-file` JSON.
- Run status is available in `state.json` under the session dir: `phase`,
  `cumulative_gain`, `crash_count`, and `stop_reason`. Terminal `stop_reason`
  values are `target_reached`, `global_converged`, `time_exhausted`,
  `max_ticks`.
- Crash recovery uses `optimize --resume` on the same session dir; a second
  `optimize` is not started for the same job. Resume past a terminal
  `stop_reason` requires `--force-resume`.
- `USER_DATA_PATH` is kept unchanged.
- After the run, the SaFE workload is released:

  ```bash
  python3 -m hyperloom.inference_optimizer.multi_node stop-multi-job --delete --clear-state
  ```

## Pre-launch Checklist

- [ ] `NFS_SHARED_ROOT` set; same mount path on sandbox **and** GPU pods
- [ ] Path A: `SAFE_API_URL` and `SAFE_API_KEY` present; Path B: `HYPERLOOM_MN_EXT_SERVICE_URL` set, SaFE creds unset
- [ ] `USER_DATA_PATH` writable; `${NFS_SHARED_ROOT}/models/...` exists
- [ ] `<INFERA_SSHD_IMAGE>` / `<RAYJOB_IMAGE>` supplied by operator
- [ ] Backend chosen; `--mn-image` set when `--mn-backend infera`
- [ ] `install.sh` already run under `$USER_DATA_PATH/runtime/`
- [ ] `ulimit -Sn 65536` before launch (avoid "too many open files")
- [ ] Robustness: multi-node auto-downgrades to mock unless
      `--robustness-server-url` is set (expected on SaFE)
