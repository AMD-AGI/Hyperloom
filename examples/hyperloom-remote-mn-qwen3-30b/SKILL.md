---
name: hyperloom-remote-mn-qwen3-30b
description: |
  Run a 4-hour multi-node Hyperloom Qwen3-30B-A3B optimization (Infera
  PD-disaggregated or RayJob aggregated) with --nodes 2 and sglang MoE tuning on
  MI325X. Hand this skill to the agent to launch and monitor the run.
---

# Hyperloom Remote Multi-Node — Qwen3-30B-A3B

Pinned workload for a remote multi-node `optimize` run. Pick **one** backend
below and pass its `FLAGS` + `Environment` block to `inference_optimizer
optimize`; the agent launches it in the background and monitors `state.json`
until a terminal `stop_reason` (`target_reached`, `global_converged`,
`time_exhausted`, `max_ticks`). Concept and variable reference:
`@../hyperloom-remote-demo.md`. Optimizer skill: `@${HYPERLOOM_SKILL_PATH}`
(fallback `@../../src/hyperloom/inference_optimizer/SKILL.md`).

## Prerequisites

- **`NFS_SHARED_ROOT`** — cluster-wide shared mount at the **same absolute path**
  on the sandbox and every GPU pod (model, tool checkouts, session artifacts).
  Required; local-only paths break multi-node restart and benchmarks.
- **`--mn-image`** — operator-supplied: `<INFERA_SSHD_IMAGE>` (must include sshd)
  for infera, `<RAYJOB_IMAGE>` for rayjob.
- **SaFE (Path A)** — `SAFE_API_URL`, `SAFE_API_KEY`, `SAFE_WORKSPACE` are
  platform-injected; not passed on the CLI. For an external cluster with no SaFE,
  see `@../hyperloom-remote-demo.md` § Path B.

Layout under `NFS_SHARED_ROOT` (replace `${NFS_SHARED_ROOT}` with the real mount):

```text
${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B/   # --model
${NFS_SHARED_ROOT}/InferenceX/             # INFERENCEX_PATH
${NFS_SHARED_ROOT}/Magpie/                 # MAGPIE_PATH
${NFS_SHARED_ROOT}/TraceLens/              # TRACELENS_ROOT
${NFS_SHARED_ROOT}/TraceLens-internal/     # TRACELENS_INTERNAL_ROOT (optional)
```

---

## Workload A — Infera + PD disaggregation

`--mn-backend infera` with `--pd-mode disaggregated`.

### FLAGS

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
--isl 1024 --osl 1024 --conc 128 \
--gpu-type mi325x \
--precision bf16 \
--mn-image <INFERA_SSHD_IMAGE> \
--pd-mode disaggregated \
--pd-prefill-nodes 1 --pd-decode-nodes 1 \
--pd-prefill-tp 8 --pd-decode-tp 8 \
--pd-prefill-ep 8 --pd-decode-ep 8 \
--pd-transfer-backend mooncake \
--pd-prefill-extra-args "--attention-backend aiter --mem-fraction-static 0.78 --disable-radix-cache --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --trust-remote-code" \
--pd-decode-extra-args "--attention-backend aiter --mem-fraction-static 0.82 --enable-dp-attention --deepep-mode normal --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024 --trust-remote-code" 
```

### Environment

```text
RANDOM_RANGE_RATIO=0.8
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
TRACELENS_INTERNAL_ROOT=${NFS_SHARED_ROOT}/TraceLens-internal
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
```

Large-model MoE cold start may need a longer poll budget:
`export HYPERLOOM_MN_POLL_TIMEOUT_S=1800 HYPERLOOM_MN_HEALTH_WAIT_S=1800`.

---

## Workload B — RayJob aggregated

`--mn-backend rayjob` (no PD flags). Same model / topology / Environment as
Workload A; the differences are:

- Drop all `--pd-*` flags.
- Replace `--pd-*-extra-args` with a single `--server-args` (applied each restart).
- `--mn-image <RAYJOB_IMAGE>` (standard Ray image, no sshd requirement).
- Drop the `SGLANG_DISAGGREGATION_*` env vars (PD-only).

### FLAGS

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
--isl 1024 --osl 1024 --conc 128 \
--gpu-type mi325x \
--precision bf16 \
--no-framework-agent \
--mn-image <RAYJOB_IMAGE> \
--server-args "--attention-backend aiter --mem-fraction-static 0.8 --enable-dp-attention --deepep-mode normal --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024 --trust-remote-code"
```

### Environment

```text
RANDOM_RANGE_RATIO=0.8
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
TRACELENS_INTERNAL_ROOT=${NFS_SHARED_ROOT}/TraceLens-internal
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
```

---

## Run notes

- Session lands in `$USER_DATA_PATH/<model_basename>/<UTC_ts>/`; `USER_DATA_PATH`
  is platform-injected and kept unchanged.
- Crash recovery: `optimize --resume` on the **same** session dir (never a second
  `optimize`; resume past a terminal `stop_reason` needs `--force-resume`).
- Release the cluster when done:
  `python3 -m hyperloom.inference_optimizer.multi_node stop-multi-job --delete --clear-state`.
