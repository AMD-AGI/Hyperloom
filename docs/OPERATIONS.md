# Operations & Self-Host Runbook

> **Audience.** Site reliability and platform engineers self-hosting
> Hyperloom on their own AMD GPU infrastructure (Kubernetes, bare
> metal, or a managed PaaS). For the hosted PrimusClaw experience
> ([core42.example-internal-host.invalid/hyperloom](https://core42.example-internal-host.invalid/hyperloom/))
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

By default, each optimization session lives under a per-model timestamped
directory:

```text
$USER_DATA_PATH/<model_basename>/<YYYYMMDDTHHMMSSZ>/
```

Commands below use `$SESSION_DIR` for that concrete session directory.

| Workload                  | Typical session size              | Retention recommendation                |
|---------------------------|-----------------------------------|-----------------------------------------|
| 2-hour explore-only run   | 5–10 GB                           | 30 days (then archive `session_breakdown.json` only) |
| 24-hour full run with kernel-opt | 50–100 GB                  | 14 days (then archive selectively)      |
| Multi-day run             | 200 GB+                           | 7 days (move artefacts to cold storage) |

The largest contributors are:

* `$SESSION_DIR/runs/<action>/<task_id>/` Magpie outputs (per-benchmark trace
  + result.json).
* `$SESSION_DIR/kernel-agent/runs/<session_id>/` kernel optimization and
  TraceLens artefacts (especially GEAK reports, prompts, traces, and patches).
* `$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/` per-session traces
  (compressed but still GB-scale).

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
| Session start   | API call / Job creation                            | Coordinator creates `$SESSION_DIR` and writes `manifest.json`, `state.json`. |
| Heartbeat       | Every 60 s                                         | Coordinator writes `state.json.tmp` → atomic rename inside `$SESSION_DIR`. |
| Session end     | `target_reached` / `time_exhausted` / `global_converged` | Coordinator writes `session_breakdown.json`, exits 0. |
| Crash recovery  | Pod OOM / preemption                               | Re-launch with `--resume` / `--resume-from`; reads `manifest.json` + `state.json`. |

---

## 3. Backup & retention

### What to back up

| Artefact                                | Source path                                                       | Retention                                                                                                |
|-----------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Session manifest + state                | `$SESSION_DIR/manifest.json`, `$SESSION_DIR/state.json`           | Until the session ends; not normally needed afterwards.                                                  |
| `session_breakdown.json` (downstream contract) | `$SESSION_DIR/session_breakdown.json`                       | **Permanent.** This is the canonical record consumed by `claw-stats-service` and downstream notebooks.   |
| Local recipe KB                         | `${HYPERLOOM_LOCAL_KB_ROOT:-$USER_DATA_PATH/kb}`                  | **Permanent.** Backup before cleanup of `USER_DATA_PATH`.                                                |
| Robustness findings                     | `$USER_DATA_PATH/agents/robustness/findings/*.jsonl`              | 30 days minimum; longer if your incident process needs it.                                               |
| Kernel-opt attempts                     | `$SESSION_DIR/kernel-agent/runs/<session_id>/optimization_attempts.jsonl` | 14 days unless an attempt was promoted; keep promoted attempts permanently.                       |
| Per-attempt artefacts (full)            | `$SESSION_DIR/kernel-agent/runs/<session_id>/{logs,results,verification}/` | 7–14 days. Cold-archive only if you need full reproducibility.                                  |

### Suggested cron

```bash
# Daily: ship session_breakdown.json + KB to S3
find "$USER_DATA_PATH" -name session_breakdown.json -mtime -1 \
  -exec aws s3 cp {} s3://my-bucket/hyperloom/sessions/ \;
aws s3 sync "${HYPERLOOM_LOCAL_KB_ROOT:-$USER_DATA_PATH/kb}" s3://my-bucket/hyperloom/kb/

# Weekly: prune session dirs older than 14 days
find "$USER_DATA_PATH" -mindepth 2 -maxdepth 2 -type d -name '20??????T??????Z' -mtime +14 -exec rm -rf {} \;
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
| Per-tick Coordinator state     | `$SESSION_DIR/state.json`                                                        | JSON, snapshot  |
| Session breakdown (final)      | `$SESSION_DIR/session_breakdown.json`                                            | JSON, snapshot  |
| Robustness findings            | `$USER_DATA_PATH/agents/robustness/findings/<session>.jsonl`                     | JSONL, append   |
| Critic verdicts                | `$USER_DATA_PATH/critic-session-memory/<session>/emit-*.json`                    | JSON per call   |
| Kernel-opt attempts            | `$SESSION_DIR/kernel-agent/runs/<session_id>/optimization_attempts.jsonl`        | JSONL, append   |
| Inference server logs          | `$SESSION_DIR/runs/<action>/<task>/server.log`                                  | text            |

Recommended pipeline: `vector` / `fluentbit` tailing the JSONL files
and forwarding to your observability stack of choice (Datadog, Loki,
Elastic, …). `session_breakdown.json` is the highest-signal artefact —
ingest it whole on session end.

---

## 6. Disaster recovery

### Scenario A: pod was OOM-killed mid-session

1. Locate the affected session directory and verify the PV is intact:
   `ls "$SESSION_DIR/state.json"`.
2. Relaunch with `--resume`:
   ```bash
   inference_optimizer optimize --resume --resume-from "$SESSION_DIR"
   ```
3. Coordinator reads `manifest.json` + `state.json`, re-enters the
   loop at the last completed action. The current in-flight action
   (if any) is re-played from scratch.
4. Robustness writes a fresh `findings/<session>.jsonl` segment; old
   segments remain.

### Scenario B: PV lost or corrupted

1. The session is unrecoverable. Restart from scratch with a fresh
   `--model …` invocation.
2. KB is unaffected if `HYPERLOOM_LOCAL_KB_ROOT` lives on a different
   volume (recommended). The next run gets the same local recipe store.

### Scenario C: auth-proxy stuck

1. Liveness probe should already have caught this.
2. Manual: `bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"`.
3. If 401s persist, rotate `SAFE_API_KEY` (rare — the key is
   long-lived) and re-run.

### Scenario D: Local KB store corrupted

1. Move the selected local KB root aside before starting a new run:
   ```bash
   mv "${HYPERLOOM_LOCAL_KB_ROOT:-$USER_DATA_PATH/kb}" \
      "${HYPERLOOM_LOCAL_KB_ROOT:-$USER_DATA_PATH/kb}.corrupt.$(date -u +%Y%m%dT%H%M%SZ)"
   ```
2. Restart the optimizer with the same `--local-kb-root` (or env default). The
   local store is recreated lazily on first write.

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
- [ ] `HYPERLOOM_LOCAL_KB_ROOT` (or `$USER_DATA_PATH/kb`) on persistent
  storage with daily backup.
- [ ] `SAFE_API_KEY` rotation runbook (key is long-lived; rotation
  requires only re-export + `install.sh` re-run).
- [ ] Liveness probe for auth-proxy on `127.0.0.1:4002`.
- [ ] Daily ship of `session_breakdown.json` to long-term storage.
- [ ] Weekly prune of `USER_DATA_PATH` for completed sessions
  > 14 days old.
- [ ] Pager rotation for "Coordinator process exit code ≠ 0".
