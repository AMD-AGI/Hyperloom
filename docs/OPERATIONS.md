# Operations & Self-Host Runbook

> **Audience.** Site reliability and platform engineers self-hosting
> Hyperloom on their own AMD GPU infrastructure (Kubernetes, bare
> metal, or a managed PaaS). For the hosted PrimusClaw experience
> ([core42.primus-safe.amd.com/hyperloom](https://core42.primus-safe.amd.com/hyperloom/))
> AMD owns operations; this document does **not** apply.

This page covers Kubernetes sizing, `USER_DATA_PATH` backup and
retention, the auth-proxy supervisor, log/metrics surface, and a
disaster-recovery runbook.

For application-level configuration see
[`CONFIGURATION_REFERENCE.md`](CONFIGURATION_REFERENCE.md); for
credential setup see [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md); for
recurring symptoms see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## 1. Sizing & resource requirements

### Per-session sandbox

A single Hyperloom optimization session is a long-running Python
process that drives benchmarks; the heavy GPU work happens in the
inference server it benchmarks (sglang / vllm) and in the Ray-scheduled
GEAK / OOB workers. The Coordinator pod itself is small.

| Component                          | CPU       | RAM       | GPU                                       | Disk                                                                                       |
|------------------------------------|-----------|-----------|-------------------------------------------|--------------------------------------------------------------------------------------------|
| Coordinator + Orchestration        | 4 cores   | 16 GiB    | none                                      | minimal                                                                                    |
| Critic (subprocess)                | 1 core    | 2 GiB     | none                                      | <100 MB (KB drafts)                                                                        |
| Robustness (subprocess)            | 1 core    | 2 GiB     | none                                      | <100 MB (findings JSONL)                                                                   |
| Kernel-agent + Ray head            | 4 cores   | 16 GiB    | none for head; workers below              | varies                                                                                     |
| Ray worker (GEAK / OOB attempt)    | 8 cores   | 32 GiB    | 1 × MI300X / MI325X / MI355X              | ~10 GB per attempt for build artefacts                                                     |
| Inference server (sglang / vllm)   | 16 cores  | 128 GiB   | 1–8 × MI300X / MI325X / MI355X (matches TP)| weights + KV cache; depends on model                                                       |
| GEAK RAG index (first build)       | 4 cores   | 16 GiB    | 1 × any GPU (CPU is hours-slow)           | ~1.3 GB BGE embedding model + index in `~/.cache/amd-ai-devtool/semantic-index/`           |

**Minimum viable node:** one AMD GPU (MI300X / MI325X / MI355X) with
≥ 256 GiB system RAM, 32 cores, and 500 GB local fast disk for the
session dir + GEAK build artefacts.

### Storage for `USER_DATA_PATH`

| Workload                  | Typical session size              | Retention recommendation                |
|---------------------------|-----------------------------------|-----------------------------------------|
| 2-hour explore-only run   | 5–10 GB                           | 30 days (then archive `session_breakdown.json` only) |
| 24-hour full run with kernel-opt | 50–100 GB                  | 14 days (then archive selectively)      |
| Multi-day run             | 200 GB+                           | 7 days (move artefacts to cold storage) |

The largest contributors are:

* `runs/<task_id>/` Magpie outputs (per-benchmark trace + result.json).
* `agents/<agent>/runs/<session>/` kernel optimization attempt
  artefacts (especially GEAK's `optimization_report.md` + per-task
  patches).
* `tracelens/` per-session traces (compressed but still GB-scale).

If you only need long-term observability, the only file you must
preserve is `session_breakdown.json` (1–10 MB; see
[`INTEGRATION_SESSION_BREAKDOWN.md`](INTEGRATION_SESSION_BREAKDOWN.md)).

---

## 2. Kubernetes layout

Hyperloom does **not** ship its own Helm chart. Recommended layout for
self-hosters:

```
namespace: hyperloom
├── Job: hyperloom-session-<session_id>   # short-lived, one per optimization run
│   ├── Pod: coordinator                  # Python CLI
│   ├── (subprocess) critic-agent
│   ├── (subprocess) robustness-agent
│   └── (subprocess) kernel-agent + Ray head
├── PersistentVolumeClaim: user-data       # mounted at /workspace/hyperloom
├── PersistentVolumeClaim: weka-tracelens  # read-only mount of TraceLens-internal
├── Secret: hyperloom-creds                # SAFE_API_KEY, CURSOR_API_KEY
└── ConfigMap: hyperloom-env               # path env, KB env, observability env
```

Notes:

* Ray workers are launched as **child processes** of the kernel-agent,
  not as separate pods. Hyperloom does not require Ray's Kubernetes
  operator. (Hosted PrimusClaw deployments do use RayJob for multi-node
  scale-out; that is internal to the PrimusClaw control plane.)
* Pin the pod to a single node with `nodeSelector` matching your AMD
  GPU labels; Ray currently expects all GPUs visible to the head.
* Mount `USER_DATA_PATH` on a fast local SSD or NVMe (RWO). Network
  storage (NFS, WekaFS) works but adds latency to the per-tick
  state.json reads.
* The auth-proxy binds on `127.0.0.1:4002` inside the pod — no Service
  / NetworkPolicy required.

### Lifecycle

| Phase           | Trigger                                            | Action                                                  |
|-----------------|----------------------------------------------------|---------------------------------------------------------|
| Session start   | API call / Job creation                            | Coordinator writes `manifest.json`, `state.json`.       |
| Heartbeat       | Every 60 s                                         | Coordinator writes `state.json.tmp` → atomic rename.    |
| Session end     | `target_reached` / `time_exhausted` / `global_converged` | Coordinator writes `session_breakdown.json`, exits 0. |
| Crash recovery  | Pod OOM / preemption                               | Re-launch with `--resume`; reads `manifest.json` + `state.json`. |

---

## 3. Backup & retention

### What to back up

| Artefact                                | Source path                                                       | Retention                                                                                                |
|-----------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Session manifest + state                | `$USER_DATA_PATH/manifest.json`, `$USER_DATA_PATH/state.json`     | Until the session ends; not normally needed afterwards.                                                  |
| `session_breakdown.json` (downstream contract) | `$USER_DATA_PATH/session_breakdown.json`                   | **Permanent.** This is the canonical record consumed by `claw-stats-service` and downstream notebooks.   |
| KB contributions                        | `$INFERENCE_OPTIMIZER_KB_ROOT/*.jsonl`                            | **Permanent.** Append-only; backup before any cleanup of `USER_DATA_PATH`.                               |
| Robustness findings                     | `$USER_DATA_PATH/agents/robustness/findings/*.jsonl`              | 30 days minimum; longer if your incident process needs it.                                               |
| Kernel-opt attempts                     | `$USER_DATA_PATH/agents/kernel/runs/<session>/optimization_attempts.jsonl` | 14 days unless an attempt was promoted; keep promoted attempts permanently.                       |
| Per-attempt artefacts (full)            | `$USER_DATA_PATH/agents/kernel/runs/<session>/{logs,results,verification}/` | 7–14 days. Cold-archive only if you need full reproducibility.                                  |

### Suggested cron

```bash
# Daily: ship session_breakdown.json + KB to S3
find "$USER_DATA_PATH" -name session_breakdown.json -mtime -1 \
  -exec aws s3 cp {} s3://my-bucket/hyperloom/sessions/ \;
aws s3 sync "$INFERENCE_OPTIMIZER_KB_ROOT" s3://my-bucket/hyperloom/kb/

# Weekly: prune session dirs older than 14 days
find "$USER_DATA_PATH" -maxdepth 1 -name 'sess-*' -mtime +14 -exec rm -rf {} \;
```

---

## 4. Auth-proxy supervision

The OOB auth-proxy (`127.0.0.1:4002`) is a single Python child of the
kernel-agent. If it dies (OOM, port conflict, stale tcp state),
**every** subsequent `claude` / `codex` CLI call returns HTTP 401.

`kernel-agent/scripts/ensure_auth_proxy.sh` is idempotent and safe to
run from a sidecar / liveness probe:

```bash
bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"
```

It TCP-probes `:4002`, then HTTP-probes via `curl`. If the port is
open but the probe times out (stuck proxy), it kills the existing
`auth_proxy.py` process and relaunches. If `:4002` is healthy, it
noops.

Recommended liveness probe: every 60 s, exit non-zero if
`curl --max-time 2 http://127.0.0.1:4002/healthz` fails.

---

## 5. Observability

Hyperloom does not ship a metrics endpoint of its own; observability
is JSONL-on-disk + (optional) downstream collectors.

| Signal                         | File / location                                                                  | Format          |
|--------------------------------|----------------------------------------------------------------------------------|-----------------|
| Per-tick Coordinator state     | `$USER_DATA_PATH/state.json`                                                     | JSON, snapshot  |
| Session breakdown (final)      | `$USER_DATA_PATH/session_breakdown.json`                                         | JSON, snapshot  |
| Robustness findings            | `$USER_DATA_PATH/agents/robustness/findings/<session>.jsonl`                     | JSONL, append   |
| Critic verdicts                | `$USER_DATA_PATH/critic-session-memory/<session>/emit-*.json`                    | JSON per call   |
| Kernel-opt attempts            | `$USER_DATA_PATH/agents/kernel/runs/<session>/optimization_attempts.jsonl`       | JSONL, append   |
| Coordinator logs               | `$USER_DATA_PATH/coordinator.log`                                                | text            |
| Inference server logs          | `$USER_DATA_PATH/runs/<task>/server.log`                                         | text            |

Recommended pipeline: `vector` / `fluentbit` tailing the JSONL files
and forwarding to your observability stack of choice (Datadog, Loki,
Elastic, …). `session_breakdown.json` is the highest-signal artefact —
ingest it whole on session end.

---

## 6. Disaster recovery

### Scenario A: pod was OOM-killed mid-session

1. Verify the PV is intact: `ls $USER_DATA_PATH/state.json`.
2. Relaunch with `--resume`:
   ```bash
   inference_optimizer optimize --resume
   ```
3. Coordinator reads `manifest.json` + `state.json`, re-enters the
   loop at the last completed action. The current in-flight action
   (if any) is re-played from scratch.
4. Robustness writes a fresh `findings/<session>.jsonl` segment; old
   segments remain.

### Scenario B: PV lost or corrupted

1. The session is unrecoverable. Restart from scratch with a fresh
   `--model …` invocation.
2. KB is unaffected if `$INFERENCE_OPTIMIZER_KB_ROOT` lives on a
   different volume (recommended). The next run gets the same priors
   as before.

### Scenario C: auth-proxy stuck

1. Liveness probe should already have caught this.
2. Manual: `bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"`.
3. If 401s persist, rotate `SAFE_API_KEY` (rare — the key is
   long-lived) and re-run.

### Scenario D: KB write corrupted

1. KB JSONL files are append-only. If the last record is partial,
   truncate to the last newline:
   ```bash
   sed -i '$d' $INFERENCE_OPTIMIZER_KB_ROOT/lessons.jsonl  # drop last line only if invalid
   ```
2. Restart Critic; it will regenerate `index.json` on next ingest.

### Scenario E: Ray won't start (`--num-gpus` rejected)

The Ray 2.44 CLI is incompatible with Click ≥ 8.3:

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

---

## 7. Upgrading

See [`UPGRADING.md`](UPGRADING.md) for per-version migration steps.
The summary policy: `USER_DATA_PATH` is forward-compatible across
patch releases; minor releases may add new fields to
`session_breakdown.json` (backwards-compatible) without bumping
`schema_version`.

---

## 8. Capacity planning checklist

Before going to production with self-hosted Hyperloom:

- [ ] AMD GPU pool sized to your concurrent-session count (1 session
  = 1–8 GPUs depending on workload TP).
- [ ] `USER_DATA_PATH` PV ≥ 200 GB per active session, ideally local
  NVMe.
- [ ] `$INFERENCE_OPTIMIZER_KB_ROOT` on a separate PV with daily
  backup.
- [ ] `SAFE_API_KEY` rotation runbook (key is long-lived; rotation
  requires only re-export + `install.sh` re-run).
- [ ] Liveness probe for auth-proxy on `127.0.0.1:4002`.
- [ ] Daily ship of `session_breakdown.json` to long-term storage.
- [ ] Weekly prune of `USER_DATA_PATH` for completed sessions
  > 14 days old.
- [ ] Pager rotation for "Coordinator process exit code ≠ 0".
