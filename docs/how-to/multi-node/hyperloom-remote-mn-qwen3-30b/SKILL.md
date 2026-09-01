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
until a terminal `stop_reason` — success (`target_reached`, `global_converged`,
`time_exhausted`, `max_ticks`) or early failure (`baseline_failed`,
`baseline_accuracy_failed`) that ends the run before optimizing. Concept and variable reference:
`@../../../reference/multi-node.md`. Optimizer skill: `@${HYPERLOOM_SKILL_PATH}`
(fallback `@../../../../src/hyperloom/inference_optimizer/SKILL.md`).

## Prerequisites

- **`NFS_SHARED_ROOT`**: cluster-wide shared mount at the **same absolute path**
  on the sandbox and every GPU pod (model, tool checkouts, session artifacts).
  Required; local-only paths break multi-node restart and benchmarks.
- **A provisioned cluster**: the platform creates the GPU pods and hands them
  over via `HYPERLOOM_MN_EXT_*` (platform-injected; never passed on the CLI).
  `optimize` adopts that cluster and exits 2 without it.
  See `@../../../reference/multi-node.md`.

### One FLAGS block, two readers

The blocks below are written for Primus-Claw *and* `optimize`: Claw parses them to
size and build the cluster, then the agent runs `optimize` with the same block.
Four flags configure the cluster Claw provisions and **can be dropped when
composing the `optimize` command**. `optimize` accepts them so a real platform
prompt parses, but nothing reads them once the cluster exists:

| Claw-only flag | What Claw does with it |
|---|---|
| `--mn-image` | Container image for the GPU pods (infera needs the sshd layer) |
| `--cpus-per-node` / `--mem-per-node` | Per-pod CPU / memory request |
| `--extra-env K=V` | Baked into pod env at create time (repeatable) |

They are listed last in each block so the `optimize` command is the block with
its tail cut off.

Everything else (`--nodes`, `--mn-backend`, `--gpus-per-node`, `--model`,
`--framework`, `--pd-*`) is read by both and stays on the `optimize` command.

Layout under `NFS_SHARED_ROOT` (replace `${NFS_SHARED_ROOT}` with the real mount):

```text
${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B/   # --model
${NFS_SHARED_ROOT}/InferenceX/             # INFERENCEX_PATH
${NFS_SHARED_ROOT}/Magpie/                 # MAGPIE_PATH
${NFS_SHARED_ROOT}/TraceLens/              # TRACELENS_ROOT
```

If your deployment has the `TraceLens-internal` checkout, set
`TRACELENS_INTERNAL_ROOT` separately. Leave it unset for the public open-source
path.

---

## Workload A — Infera + PD disaggregation

`--mn-backend infera` with `--pd-mode disaggregated`.

### FLAGS

```text
--gpu-type mi325x \
--model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
--target-gain 30 \
--max-hours 4 \
--mn-backend infera \
--framework=sglang \
--nodes 2 \
--gpus-per-node 8 \
--tp 8 --ep 8 \
--isl 1024 --osl 1024 --conc 128 \
--precision bf16 \
--pd-mode disaggregated \
--pd-prefill-nodes 1 --pd-decode-nodes 1 \
--pd-prefill-tp 8 --pd-decode-tp 8 \
--pd-prefill-ep 8 --pd-decode-ep 8 \
--pd-transfer-backend mooncake \
--pd-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
--pd-prefill-extra-args "--attention-backend aiter --mem-fraction-static 0.78 --disable-radix-cache --load-balance-method round_robin --watchdog-timeout 3600 --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192" \
--pd-decode-extra-args "--attention-backend aiter --mem-fraction-static 0.82 --enable-dp-attention --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024" \
--mn-image <INFERA_SSHD_IMAGE> \
--cpus-per-node 90 \
--mem-per-node 1024 \
--extra-env MC_GID_INDEX=3 \
--extra-env NCCL_IB_GID_INDEX=3 \
--extra-env SGLANG_USE_AITER_AR=0
```

### Environment

```text
RANDOM_RANGE_RATIO=0.8
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
```

Large-model MoE cold start may need a longer poll budget:
`export HYPERLOOM_MN_POLL_TIMEOUT_S=1800 HYPERLOOM_MN_HEALTH_WAIT_S=1800`.

---

## Workload B — RayJob aggregated

`--mn-backend rayjob` (no PD flags). Same model / Environment as Workload A; the
differences are:

- Drop all `--pd-*` flags.
- Replace `--pd-*-extra-args` with a single `--server-args` (applied each restart).
- `--mn-image` is a standard Ray image (no sshd layer needed).
- Drop the `SGLANG_DISAGGREGATION_*` env vars (PD-only).
- **`--tp` covers the whole cluster, not one node.** One server spans every pod
  here, so `--tp` must be `nodes * gpus-per-node` (16) — sglang receives
  `--tp <n> --nnodes 2` and splits the ranks, so `--tp 8` would leave half of
  each pod's GPUs idle. Workload A differs because each PD role owns one node,
  where `--pd-*-tp 8` fills all 8 of its GPUs.
- Pod-side env reaches the GPU pods only through `--extra-env`, which Claw bakes
  in at create time (Claw-only, see the table above — drop it from the `optimize`
  command). Cross-node tensor parallel needs the RoCE GID there. The
  `Environment` block below is sandbox-side only and never reaches the pods.

### FLAGS

```text
--model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
--target-gain 30 \
--max-hours 4 \
--mn-backend rayjob \
--framework=sglang \
--nodes 2 \
--gpus-per-node 8 \
--tp 16 --ep 16 \
--isl 1024 --osl 1024 --conc 128 \
--gpu-type mi325x \
--precision bf16 \
--server-args "--attention-backend aiter --mem-fraction-static 0.8 --enable-dp-attention --deepep-mode normal --ep-dispatch-algorithm fake --load-balance-method round_robin --watchdog-timeout 3600 --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 8192 --max-running-requests 1024" \
--mn-image <RAYJOB_IMAGE> \
--cpus-per-node 90 \
--mem-per-node 1024 \
--extra-env NCCL_IB_GID_INDEX=3
```

Optional model pins (omit to use the deployment defaults):
`--claude-model <model>` / `--codex-model <model>`.

### Environment

```text
RANDOM_RANGE_RATIO=0.8
INFERENCEX_PATH=${NFS_SHARED_ROOT}/InferenceX
TRACELENS_ROOT=${NFS_SHARED_ROOT}/TraceLens
MAGPIE_PATH=${NFS_SHARED_ROOT}/Magpie
```

The Kernel-Forge kernel backend needs no checkout and no path variable:
KernelForge ships inside Hyperloom. Select it with `KERNEL_OPT_BACKEND_ORDER=forge`
(Workload B only). Behind a TLS-terminating proxy add
`NODE_TLS_REJECT_UNAUTHORIZED=0` and `ANTHROPIC_SKIP_TLS_VERIFY=true`.

---

## Run notes

- Session lands in `$USER_DATA_PATH/<model_basename>/<UTC_ts>/`; `USER_DATA_PATH`
  is platform-injected and kept unchanged.
- Crash recovery: `optimize --resume-from "$SESSION_DIR"` on the **same** session
  dir (never a second `optimize`; resume past a terminal `stop_reason` needs
  `--force-resume`).
- Releasing the cluster is the platform's job, not the optimizer's — it happens
  when the session ends.
