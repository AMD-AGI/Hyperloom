---
myst:
    html_meta:
        "description": "Operations guide for self-hosted Hyperloom. Covers Kubernetes sizing, USER_DATA_PATH backup, LLM gateway credential checks, observability, disaster recovery, and capacity planning."
        "keywords": "Hyperloom, self-hosting, operations, Kubernetes, USER_DATA_PATH, backup, LLM gateway, observability, disaster recovery, AMD GPU, ROCm, capacity planning"
---
# Hyperloom self-hosting and operations guide

This topic covers Kubernetes sizing, `USER_DATA_PATH` backup and
retention, LLM gateway credential checks, log/metrics surface, and a
disaster-recovery runbook.

```{note}
This topic is intended for Site reliability and platform engineers self-hosting
Hyperloom on their own AMD GPU infrastructure (Kubernetes, bare metal, or a managed platform as a service (PaaS)). For the hosted Primus-Claw experience AMD owns operations; this document does not apply.
```

For application-level configuration see
[Environment variables](environment-variables.md); for
credential setup see [Hyperloom authentication and credentials](authentication.md); for
recurring symptoms see [Troubleshooting Hyperloom](troubleshooting.md).

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.
```

---

## Sizing and resource requirements

The following resource requirements apply to a single optimization session.

### Per-session sandbox

A single Hyperloom optimization session is a long-running Python
process that drives benchmarks; the heavy GPU work happens in the
inference server it benchmarks (sglang or vllm) and in the Ray-scheduled
GEAK workers. The Coordinator pod itself is small.

| Component                          | CPU       | RAM       | GPU                                       | Disk                                                                                       |
|------------------------------------|-----------|-----------|-------------------------------------------|--------------------------------------------------------------------------------------------|
| Coordinator + Orchestration        | 4 cores   | 16 GiB    | none                                      | minimal                                                                                    |
| Critic (subprocess)                | 1 core    | 2 GiB     | none                                      | <100 MB (knowledge base (KB) drafts)                                                       |
| Robustness (subprocess)            | 1 core    | 2 GiB     | none                                      | <100 MB (findings JSONL)                                                                   |
| GEAK + Ray head (kernel optimization) | 4 cores   | 16 GiB    | none for head; workers below              | varies                                                                                     |
| Ray worker (GEAK attempt)          | 8 cores   | 32 GiB    | 1 × MI300X, MI325X, or MI355X             | ~10 GB per attempt for build artifacts                                                     |
| Inference server (sglang or vllm)  | 16 cores  | 128 GiB   | 1–8 × MI300X, MI325X, or MI355X (matches TP)| weights + KV cache; depends on model                                                      |
| GEAK retrieval-augmented generation (RAG) index (first build) | 4 cores   | 16 GiB    | 1 × any GPU (CPU is hours-slow)           | ~1.3 GB BGE embedding model + index in `~/.cache/amd-ai-devtool/semantic-index/`           |

Minimum viable node: one AMD GPU (MI300X, MI325X, or MI355X) with
≥ 256 GiB system RAM, 32 cores, and 500 GB local fast disk for the
session dir + GEAK build artifacts.

### Storage for `USER_DATA_PATH`

By default, each optimization session lives under a per-model timestamped
directory:

```text
$USER_DATA_PATH/<model_basename>/<YYYYMMDDTHHMMSSZ>/
```

Commands below use `$SESSION_DIR` for that concrete session directory.

| Workload                  | Typical session size              | Retention recommendation                |
|---------------------------|-----------------------------------|-----------------------------------------|
| 2-hour `--no-kernel` run  | 5–10 GB                           | 30 days (then archive `session_breakdown.json` only) |
| 24-hour full run with kernel-opt | 50–100 GB                  | 14 days (then archive selectively)      |
| Multi-day run             | 200 GB+                           | 7 days (move artifacts to cold storage) |

The largest contributors are:

* `$SESSION_DIR/runs/<action>/<task_id>/` Magpie outputs (per-benchmark trace
  + result.json).
* `$SESSION_DIR/kernel-agent/runs/<session_id>/` kernel optimization and
  TraceLens artifacts (especially GEAK reports, prompts, traces, and patches).
* `$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/` per-session traces
  (compressed but still GB-scale).

If you only need long-term observability, the only file you must
preserve is `session_breakdown.json` (1–10 MB; see
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md)).

---

## Kubernetes layout

```{seealso}
To run the same workload under a batch scheduler instead of Kubernetes, see
[Install for Slurm](../install/slurm.md).
```

Hyperloom does not ship its own Helm chart. Recommended layout for
self-hosters:

```
namespace: hyperloom
├── Job: hyperloom-session-<session_id>   # short-lived, one per optimization run
│   ├── Pod: coordinator                  # Python CLI
│   ├── (subprocess) critic-agent
│   ├── (subprocess) robustness-agent
│   └── Ray head (launched by kernel request handlers; GEAK runs as Ray workers)
├── PersistentVolumeClaim: user-data       # mounted at /workspace/hyperloom
├── PersistentVolumeClaim: tracelens-extension  # optional read-only private extension mount
├── Secret: hyperloom-creds                # OPENAI_API_KEY
└── ConfigMap: hyperloom-env               # path env, KB env, observability env
```

Notes:

* Ray is launched directly by the Coordinator's programmatic kernel request
  handlers (`orchestrator/kernel/request_handlers.py`) when GEAK runs, not by a
  separate kernel-agent process. Hyperloom does not require Ray's Kubernetes
  operator. (Hosted Primus-Claw deployments do use RayJob for multi-node
  scale-out; that is internal to the Primus-Claw control plane.)
* Pin the pod to a single node with `nodeSelector` matching your AMD
  GPU labels; Ray currently expects all GPUs visible to the head.
* Mount `USER_DATA_PATH` on a fast local SSD or NVMe (RWO). Network
  storage (NFS, WekaFS) works but adds latency to the per-tick state.json reads.
  (`RWO` = ReadWriteOnce, a persistent volume (PV) access mode.)
* LLM calls go directly to the configured upstream gateway; no in-pod
  auth-proxy Service or NetworkPolicy is required.

### Lifecycle

The following table describes the key lifecycle events for a Hyperloom session.

| Phase           | Trigger                                            | Action                                                  |
|-----------------|----------------------------------------------------|---------------------------------------------------------|
| Session start   | API call / Job creation                            | Coordinator creates `$SESSION_DIR` and writes `manifest.json`, `state.json`. |
| Heartbeat       | Every Coordinator tick (`--tick-interval-sec`, default `0` = no sleep) | Coordinator atomically rewrites `state.json` (temp `.state.json.*.tmp` + `os.replace`) inside `$SESSION_DIR`. |
| Session end     | `target_reached` / `time_exhausted` / `global_converged` | Coordinator writes `session_breakdown.json`, exits 0. |
| Crash recovery  | Pod OOM / preemption                               | Re-launch with `--resume-from "$SESSION_DIR"`; reads `manifest.json` + `state.json`. |

---

## Backup and retention

The following guidelines cover what to back up and for how long.

### What to back up

Back up the following artifacts from each session.

| Artifact                                | Source path                                                       | Retention                                                                                                |
|-----------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Session manifest + state                | `$SESSION_DIR/manifest.json`,<br>`$SESSION_DIR/state.json`        | Until the session ends; not normally needed afterwards.                                                  |
| `session_breakdown.json` (downstream contract) | `$SESSION_DIR/`<br>`session_breakdown.json`                | Permanent. This is the canonical record consumed by downstream dashboards and notebooks.                 |
| Local knowledge root                    | `${KNOWLEDGE_LOCAL_ROOT:-$USER_DATA_PATH/knowledge}` (otherwise `~/.cache/hyperloom/knowledge`) | Permanent. Backup before cleanup of `USER_DATA_PATH`. |
| Robustness findings                     | `$SESSION_DIR/agents/`<br>`robustness/findings/`<br>`<session_id>.jsonl` | 30 days minimum; longer if your incident process needs it.                                        |
| Kernel-opt attempts                     | `$SESSION_DIR/kernel-agent/`<br>`runs/<session_id>/`<br>`optimization_attempts.jsonl` | 14 days unless an attempt was promoted; keep promoted attempts permanently.          |
| Per-attempt artifacts (full)            | `$SESSION_DIR/kernel-agent/`<br>`runs/<session_id>/`<br>`{logs,results,verification}/` | 7–14 days. Cold-archive only if you need full reproducibility.                    |

### Suggested cron

Use the following cron jobs to automate session backup and cleanup.

```bash
# Daily: ship session_breakdown.json + KB to S3
find "$USER_DATA_PATH" -name session_breakdown.json -mtime -1 \
  -exec aws s3 cp {} s3://my-bucket/hyperloom/sessions/ \;
KB_ROOT="${KNOWLEDGE_LOCAL_ROOT:-${HYPERLOOM_LOCAL_KB_ROOT:-${USER_DATA_PATH:-$HOME/.cache/hyperloom}/knowledge}}"
aws s3 sync "$KB_ROOT" s3://my-bucket/hyperloom/knowledge/

# Weekly: prune session dirs older than 14 days
find "$USER_DATA_PATH" -mindepth 2 -maxdepth 2 -type d -name '20??????T??????Z' -mtime +14 -exec rm -rf {} \;
```

---

## LLM gateway credential checks

GEAK and forge workers call the configured upstream gateway
directly using the aliases generated by preflight and the kernel-agent installer.

Operational checks:

```bash
test -n "${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}}"
test -n "${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-}}"
bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh" --check-only
```

Child processes inherit the gateway settings prepared by preflight; keep
operator-facing gateway configuration in `OPENAI_API_KEY` / `OPENAI_BASE_URL`
or the split Anthropic/OpenAI credentials.

---

## Observability

Hyperloom does not ship a metrics endpoint of its own; observability
is JSONL-on-disk + (optional) downstream collectors.

| Signal                         | File / location                                                                  | Format          |
|--------------------------------|----------------------------------------------------------------------------------|-----------------|
| Per-tick Coordinator state     | `$SESSION_DIR/state.json`                                                        | JSON, snapshot  |
| Session breakdown (final)      | `$SESSION_DIR/session_breakdown.json`                                            | JSON, snapshot  |
| Robustness findings            | `$SESSION_DIR/agents/robustness/findings/<session_id>.jsonl`                     | JSONL, append   |
| Critic verdicts                | `$SESSION_DIR/critic-workdir/<turn>/emit.json` (memory in `$SESSION_DIR/critic-session-memory/`) | JSON per call   |
| Kernel-opt attempts            | `$SESSION_DIR/kernel-agent/runs/<session_id>/optimization_attempts.jsonl`        | JSONL, append   |
| Inference server logs          | `$SESSION_DIR/runs/<action>/<task>/server.log`                                  | text            |

Recommended pipeline: `vector` / `fluentbit` tailing the JSONL files
and forwarding to your observability stack of choice (Datadog, Loki,
Elastic, …). `session_breakdown.json` is the highest-signal artefact —
ingest it whole on session end.

---

## Disaster recovery

### Scenario A: pod was OOM-killed mid-session

1. Locate the affected session directory and verify the PV is intact:
   `ls "$SESSION_DIR/state.json"`.
2. Relaunch with `--resume-from`:
   ```bash
   python3 -m hyperloom.inference_optimizer.cli optimize --resume-from "$SESSION_DIR"
   ```
3. Coordinator reads `manifest.json` + `state.json`, re-enters the
   loop at the last completed action. The current in-flight action
   (if any) is re-played from scratch.
4. Robustness writes a fresh `findings/<session>.jsonl` segment; old
   segments remain.

To rebuild only the `session_breakdown` (and push it to Langfuse) for a run
that exited abnormally — without re-running the optimization loop — use the
dedicated subcommand instead of `--resume-from`:

```bash
  python3 -m hyperloom.inference_optimizer.cli recover-session --session-dir "$SESSION_DIR" [--force] [--backfill-trace]
```

`--force` re-runs even when the session already looks complete;
`--backfill-trace` replays `reports/trace/llm_calls.jsonl` as Langfuse
generations (use only when the live emitter never ran, or it duplicates
generations). `--resume-from` = keep optimizing; `recover-session` = rebuild
the breakdown artifact.

### Scenario B: PV lost or corrupted

1. The session is unrecoverable. Restart from scratch with a fresh
   `--model …` invocation.
2. KB is unaffected if `HYPERLOOM_LOCAL_KB_ROOT` lives on a different
   volume (recommended). The next run gets the same local recipe store.

### Scenario C: Gateway 401 after credential or config drift

1. Confirm the pod has a current key and base URL (`OPENAI_API_KEY` /
   `OPENAI_BASE_URL`, or split Anthropic/OpenAI credentials).
2. Re-run `bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh" --check-only`
   and then without `--check-only` if it reports missing aliases.
3. Inspect `~/.claude/config.json`; `customApiUrl` must point at the upstream
   gateway.

### Scenario D: Local KB store corrupted

1. Move the selected local KB root aside before starting a new run:
   ```bash
   KB_ROOT="${KNOWLEDGE_LOCAL_ROOT:-${HYPERLOOM_LOCAL_KB_ROOT:-${USER_DATA_PATH:-$HOME/.cache/hyperloom}/knowledge}}"
   mv "$KB_ROOT" "$KB_ROOT.corrupt.$(date -u +%Y%m%dT%H%M%SZ)"
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

## Upgrading

See [Upgrade Hyperloom version](upgrade.md) for per-version migration steps.
The summary policy: `USER_DATA_PATH` is forward-compatible across
patch releases; minor releases might add new fields to
`session_breakdown.json` (backwards-compatible) without bumping
`schema_version`.

---

## Capacity planning checklist

Before going to production with self-hosted Hyperloom, ensure:

- AMD GPU pool sized to your concurrent-session count (1 session
  = 1–8 GPUs depending on workload TP).
- `USER_DATA_PATH` PV ≥ 200 GB per active session, ideally local
  NVMe.
- `KNOWLEDGE_LOCAL_ROOT` (or the default `$USER_DATA_PATH/knowledge`) on persistent
  storage with daily backup.
- `OPENAI_API_KEY` rotation runbook (key is long-lived; rotation
  requires only re-export + `install.sh` re-run).
- LLM gateway credential and endpoint check in the session startup probe.
- Daily ship of `session_breakdown.json` to long-term storage.
- Weekly prune of `USER_DATA_PATH` for completed sessions
  > 14 days old.
- Pager rotation for "Coordinator process exit code ≠ 0".
