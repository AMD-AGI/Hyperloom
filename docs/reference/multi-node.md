---
myst:
    html_meta:
        "description": "Reference for running Hyperloom on a multi-node GPU cluster. Covers the infera and rayjob backends, cluster hand-off variables, pod image requirements, shared filesystem setup, and troubleshooting."
        "keywords": "Hyperloom, multi-node, cluster, infera, rayjob, AMD GPU, ROCm, MI300X, MI325X, SGLang, PD disaggregation, NFS, Primus-Claw, inference optimization"
---

# Multi-node inference optimization

This page is the reference for running Hyperloom across a multi-node GPU cluster
(`--nodes >= 2`). It covers both supported backends, the cluster hand-off
contract, pod image requirements, the shared filesystem requirement, the workload
skill variables, and common failure patterns.

For the corresponding single-node flow, see
[Run a Hyperloom optimization](../how-to/optimize.md).

## Backends

Hyperloom supports two multi-node control planes. Both serve the same model and
differ only in how Hyperloom restarts the inference server each optimization round.

| Backend | Control plane | Notes |
|---------|--------------|-------|
| `infera` | SSH | Hyperloom SSHes into each pod to launch or restart SGLang. Supports prefill/decode (PD) disaggregation. |
| `rayjob` | Ray Dashboard REST API | Pods form a KubeRay cluster. Restarts go through the Ray Dashboard; no SSH required. |

## Cluster hand-off

Hyperloom never provisions or releases a cluster. The provisioning platform
(typically [Primus-Claw](https://github.com/AMD-AGI/Primus-Claw)) creates the GPU
pods and hands the running cluster over through `HYPERLOOM_MN_EXT_*` environment
variables. Hyperloom adopts it, benchmarks it, and restarts the server on it each
round.

Without a hand-off, a `--nodes >= 2` run exits immediately with code 2.

The following table lists all hand-off variables.

| Variable | Backend | Required | Meaning |
|----------|---------|----------|---------|
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | Benchmark frontend URL (`http(s)://…`). Use the port the platform assigns — it is not always `:8000`. |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | **yes** | Path to a private key authorized on the pods. |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` | infera | PD only | Prefill pod IPs, comma-separated, in rank order. |
| `HYPERLOOM_MN_EXT_DECODE_IPS` | infera | PD only | Decode pod IPs, comma-separated, in rank order. |
| `HYPERLOOM_MN_EXT_WORKER_IPS` | infera | aggregated | Worker pod IPs, comma-separated, leader first. At least one of `_PREFILL_IPS`, `_DECODE_IPS`, or `_WORKER_IPS` is required for infera. |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | no | SSH base port (default `2233`). The decode role adds offset `+10`; each pod in a multi-node role adds its rank. |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | no | `known_hosts` path. Omit to use a relaxed host-key check. |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | no | Ray head IP — Dashboard on `:8265`, GCS on `:6379` (KubeRay's `<rayCluster>-head-svc`). Omit for benchmark-only mode with no per-round restarts. |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | no | Ray Dashboard auth token, if the dashboard requires authentication. |

The container image and per-pod CPU or memory are the provisioning platform's
inputs, not `optimize` flags.

## Pod image requirements

Neither backend runs the inference engine at pod start. Pods come up idle and
Hyperloom launches or relaunches the server each round.

### infera

The image must provide an SSH control plane so Hyperloom can reach the pods:

- `openssh-server` installed, with `/run/sshd` present and host keys baked in
  at build time (`ssh-keygen -A`) so sshd starts without write access at runtime.
- An entrypoint that starts sshd and then blocks. The platform injects the
  authorized public key as `MN_SSH_AUTHORIZED_KEY` and the port as `MN_SSH_PORT`.

```bash
mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
printf '%s\n' "$MN_SSH_AUTHORIZED_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A                              # no-op when host keys are baked in
printf 'Port %s\nPermitRootLogin prohibit-password\nPasswordAuthentication no\n' \
  "${MN_SSH_PORT:-2233}" > /etc/ssh/sshd_config.d/mn.conf
/usr/sbin/sshd -e                          # detaches, so pid 1 must block below
exec tail -f /dev/null
```

A dedicated port rather than `:22` is required: pods run with `hostNetwork` for
RDMA, where `:22` collides with the node's own sshd.

### rayjob

The image must have Ray installed and on `PATH` at the version the KubeRay
cluster expects. KubeRay runs `ray start` in the head and worker containers. Pin
Ray's CLI dependency explicitly:

```bash
python -m pip install "ray[default]==2.44.1" "click==8.1.8"
```

Keep `click` below 8.2 regardless of what Ray's declared dependency allows — Ray
2.44.1 accepts `click>=7.0`, but click 8.2 changed `Sentinel` and breaks Ray's
CLI on import, preventing `ray start` from running.

The cluster idles through the RayJob submitter: the platform sets its entrypoint
to `tail -f /dev/null` so the cluster stays up between rounds.

## Shared filesystem

A cluster-wide shared mount is required. It must be visible at the **same
absolute path** on the sandbox and every GPU pod. `USER_DATA_PATH` must also live
under it so both the sandbox and the pods read and write the same session
artifacts.

```text
${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B/   # --model
${NFS_SHARED_ROOT}/InferenceX/             # INFERENCEX_PATH
${NFS_SHARED_ROOT}/Magpie/                 # MAGPIE_PATH
${NFS_SHARED_ROOT}/TraceLens/              # TRACELENS_ROOT
${NFS_SHARED_ROOT}/TraceLens-internal/     # TRACELENS_INTERNAL_ROOT (optional)
```

A path that exists only on the sandbox causes silent failures at benchmark time:
the pod resolves the same absolute path to a different (or missing) file.

## Session output

Each run produces a session directory under
`$USER_DATA_PATH/<model_basename>/<UTC_timestamp>/`. The key files are:

| File | Contents |
|------|----------|
| `state.json` | Live status: `phase`, `cumulative_gain_validated`, `crash_count`, `stop_reason` |
| `manifest.json` | Session manifest; required for `--resume-from` |
| `session_breakdown.json` | Final downstream contract; written at CLOSE |

`$USER_DATA_PATH` is platform-injected. See
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md) for the
full schema.

## Workload skill reference

The pinned workload skill at
[`docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md)
contains two ready-to-run blocks — **Workload A (infera + PD)** and **Workload B
(rayjob)** — each a `FLAGS` list and an `Environment` block. The agent runs
`inference_optimizer optimize` with those blocks and monitors `state.json` until a
terminal `stop_reason`.

### `optimize` flags

The following flags are used in multi-node workload skill blocks.

| Flag | Meaning |
|------|---------|
| `--model` | Model path under `${NFS_SHARED_ROOT}/models/...` or a HuggingFace repo ID |
| `--nodes` | Node count; must be `>= 2` and match the cluster handed over |
| `--mn-backend` | `infera` or `rayjob` |
| `--gpus-per-node` | GPUs per pod (default 8); feeds the TP feasibility check |
| `--tp` | Tensor parallelism degree |
| `--ep` | Expert parallelism degree (MoE; often `ep == tp`) |
| `--isl`, `--osl`, `--conc` | Benchmark input length, output length, and concurrency. These must be flags — the corresponding env vars are ignored and overwritten |
| `--gpu-type`, `--precision` | Target GPU (for example `mi325x`) and dtype (`bf16`). Flags are authoritative; env vars are not |
| `--framework` | `sglang` |
| `--target-gain`, `--max-hours` | Optimization goal and time budget |
| `--pd-mode disaggregated` | Enable PD disaggregation (infera only) |
| `--pd-prefill-nodes`, `--pd-prefill-tp`, `--pd-prefill-ep`, `--pd-prefill-extra-args` | Prefill topology (infera PD) |
| `--pd-decode-nodes`, `--pd-decode-tp`, `--pd-decode-ep`, `--pd-decode-extra-args` | Decode topology (infera PD) |
| `--pd-transfer-backend` | KV transfer plane for PD. Use `mooncake`; `nixl` can produce 0 output tokens |
| `--server-args "..."` | Extra sglang args applied on every server restart |
| `--no-framework-agent` | Skip the framework-tuning agent phase |

### Environment variables

The following variables are exported on the sandbox side before calling `optimize`.

| Variable | Meaning |
|----------|---------|
| `RANDOM_RANGE_RATIO` | Sequence-length jitter for the benchmark |
| `INFERENCEX_PATH` | InferenceX checkout path under `${NFS_SHARED_ROOT}` |
| `MAGPIE_PATH` | Magpie checkout path under `${NFS_SHARED_ROOT}` |
| `TRACELENS_ROOT` | TraceLens checkout path under `${NFS_SHARED_ROOT}` |
| `SGLANG_DISAGGREGATION_*_TIMEOUT` | PD bootstrap and wait timeouts (Workload A only) |

Forge is not in that list and does not need to be: it ships inside the Hyperloom
wheel, so there is no checkout to export. `FORGE_PATH` is removed. Set
`KERNELFORGE_PROJECT_ROOT` only to point forge's writable state, or a resource
override, somewhere other than the default.

Do not set `HYPERLOOM_MN_EXT_*` variables or `USER_DATA_PATH` — the platform
injects those.

### Example launch command

The following example shows an infera + PD disaggregated launch after the
platform has handed a cluster over:

```bash
inference_optimizer optimize \
  --model ${NFS_SHARED_ROOT}/models/Qwen3-30B-A3B \
  --nodes 2 --mn-backend infera \
  --pd-mode disaggregated \
  --pd-prefill-nodes 1 --pd-decode-nodes 1 \
  --tp 8 --ep 8 \
  --pd-transfer-backend mooncake ...
```

For rayjob, replace `--mn-backend infera` with `--mn-backend rayjob` and omit
the `--pd-*` flags. The platform supplies `HYPERLOOM_MN_EXT_HEAD_IP` instead of
the SSH and IP variables.

## Troubleshooting

### Run exits immediately with code 2

**Cause**: No cluster hand-off. `HYPERLOOM_MN_EXT_SERVICE_URL` is unset or the
infera backend is missing its required SSH key and IP list.

**Fix**: Confirm the platform has provisioned a cluster and injected the
`HYPERLOOM_MN_EXT_*` variables before calling `optimize`. Check that
`HYPERLOOM_MN_EXT_SSH_KEY` is set and that at least one of
`HYPERLOOM_MN_EXT_PREFILL_IPS`, `HYPERLOOM_MN_EXT_DECODE_IPS`, or
`HYPERLOOM_MN_EXT_WORKER_IPS` is populated for infera.

---

### PD run produces 0 output tokens

**Cause**: `--pd-transfer-backend nixl` was used. nixl can yield zero output
tokens on some configurations.

**Fix**: Switch to `--pd-transfer-backend mooncake`.

---

### Ray `--num-gpus` rejected (rayjob)

**Symptom**: `ray start --head --num-gpus=N` fails with `Error: no such option:
--num-gpus` or Click-related import errors.

**Cause**: Click ≥ 8.2 is incompatible with Ray 2.44. The rayjob image was built
with a newer Click version.

**Fix**:

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

---

### Ray tasks stuck pending, GPU usage 0% (rayjob)

**Cause**: Ray was started with `--num-gpus=0` or the flag was omitted. Tasks
submitted with `num_gpus>=1` queue indefinitely.

**Fix**:

```bash
RAY_NUM_GPUS="$(python3 -c 'import torch; print(torch.cuda.device_count() or 1)')"
ray stop --force || true
ulimit -Sn 65536 2>/dev/null || true
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

If a stale Ray head is already running, stop it first so Hyperloom can create a
fresh head with the correct GPU and resource configuration.

---

### Raylet zombie or SIGABRT (rayjob)

**Symptom**: The raylet aborts on startup, or `ray stop` leaves zombie `raylet`
processes.

**Cause**: The container's `ulimit -n` (open files) is too low. On a large node
with many workers, the raylet needs `nofile >= 65536`.

**Fix**: Launch the container with a high open-files hard limit. This cannot be
raised from inside the container:

```bash
docker run --ulimit nofile=1048576 ...
```

The runtime raises the soft limit automatically before `ray start`; the hard cap
must be set at container launch time.

---

### Shared filesystem path mismatch

**Symptom**: Benchmark fails to find the model or tool checkouts on the pod side,
even though they exist on the sandbox.

**Cause**: The NFS mount is not at the same absolute path on both sides, or
`USER_DATA_PATH` is not under the shared mount.

**Fix**: Confirm that `echo $NFS_SHARED_ROOT` resolves to the same string on the
sandbox and on a pod. Confirm `USER_DATA_PATH` is a subpath of `NFS_SHARED_ROOT`.

---

### Resume fails after a crash

**Symptom**: `--resume-from` exits with `manifest.json not found` or
`state.json missing`.

**Cause**: `USER_DATA_PATH` differs from the original session, or the session
failed before writing `manifest.json`.

**Fix**:

```bash
# Locate the session directory
find "$USER_DATA_PATH" -name manifest.json

# Resume with the exact session directory
python3 -m hyperloom.inference_optimizer.cli optimize \
  --resume-from /path/to/session
```

If `manifest.json` never existed, resume is not possible — restart with a fresh
launch.

---

For issues not covered here, see [Troubleshooting Hyperloom](troubleshooting.md).

## Related topics

- [Workload skill](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/SKILL.md) — copy-paste flags and environment for Qwen3-30B-A3B
- [Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) — provisions the cluster and hands it over
- [Multi-node CLI, SSH, and hand-off semantics](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/multi_node/SKILL.md)
- [Environment variables](environment-variables.md) — full list of `HYPERLOOM_MN_EXT_*` variables
- [Run a Hyperloom optimization](../how-to/optimize.md) — single-node counterpart
