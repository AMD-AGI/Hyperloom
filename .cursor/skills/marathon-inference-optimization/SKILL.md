---
name: marathon-inference-optimization
description: |
  Marathon-phase (3h–24h+) autonomous inference optimization for LLM serving on AMD GPUs.
  Drives a 3-pane tmux architecture (watchdog + orchestrator + kernel-manager), each an
  independent `claude` CLI working against a shared NFS session directory, to push
  `tok/s/GPU` via deep-kernel analysis, operator autotuning, framework rebuilds,
  compiler tuning, and communication optimization — picking up from a Sprint-optimized
  baseline.
globs:
  - "**/marathon*"
  - "**/inference*optim*long*"
  - "**/kernel*rewrite*"
---

# Marathon Inference Optimization — 3-Pane Autonomous Long-Run Skill

## Your role as the skill-invoking agent (READ THIS CAREFULLY — it replaces the usual per-prompt Rules section)

You are the **launcher + poller**, NOT the optimizer. The actual marathon work
happens in 3 independent `claude` CLI processes inside a tmux session (`watchdog`
/ `orchestrator` / `kernel-mgr`), coordinating through JSONL files on shared
storage. The user's prompt supplies only the env vars. Everything else is your
responsibility, defined here once so the prompt can stay short.

### Exactly these four steps

1. **Collect env vars** from the user prompt. Required: `MODEL_NAME`, `BASE_DIR`.
   All other vars (see "Required env" / "Optional env" below) should come from
   the user too when they know them; otherwise `run.sh` uses defaults.
   `STRICT=1` is recommended so `run.sh` fails loudly (exit 20) if any critical
   var is unset.

2. **Launch in BACKGROUND** — never foreground:
   ```bash
   <all the env exports from the user> \
   bash $SKILL_ROOT/scripts/launcher/run.sh > /tmp/marathon.log 2>&1 &
   echo "marathon_pid=$!"
   ```
   A 24h foreground call would exhaust your context. The background approach
   hands control back immediately so you can poll.

3. **Poll every 60s** with a SEPARATE shell tool call. Always pair the tail
   with a liveness check on the marathon PID; otherwise an early `run.sh`
   death can look like a quiet long-running marathon for hours:
   ```bash
   sleep 60 && (kill -0 $marathon_pid 2>/dev/null || echo MARATHON_DEAD) && tail -200 /tmp/marathon.log
   ```
   The monitor inside `run.sh` already writes a progress block every 60s
   containing `state.json` fields (`phase`, `tput`, `gain%`, `completed`,
   `crash`, `dream`, `km`) **plus the last 3 activity lines from EACH of
   the three pane logs** (`orchestrator` / `kernel-mgr` / `watchdog` —
   `assistant text`, `tool: <name> <cmd>`, `result: ...`). 200 lines covers
   several progress blocks, enough buffer for autocompact gaps. Do NOT
   separately tail the three per-pane logs on every poll — that duplicates
   output. Only do a one-off `tail -200 $SESSION_DIR/logs/<pane>.log` when
   you need deeper detail on a specific pane.

   If the liveness check prints `MARATHON_DEAD` before `[run.sh] Done`,
   immediately classify as "MARATHON DIED EARLY", forward the last 200
   lines plus a `ps -ef | grep -E 'run.sh|claude'` snapshot, and stop polling.

4. **Stop polling** when `/tmp/marathon.log` contains the line `[run.sh] Done`.
   That line is the authoritative finish signal. When you see it, forward the
   full `=== SESSION_REPORT.md === ... === END SESSION_REPORT ===` block
   (or, if absent, the `=== NO SESSION_REPORT.md — final state.json ===`
   block) to the user.

### Hard rules

- **No cron, no &-loops, no scheduled tasks.** Use multi-turn shell tool
  polling. The panes themselves also forbid cron (see their prompts).
- **Don't interrupt** the marathon process before `MAX_HOURS` expires. If the
  user explicitly asks to stop, `touch $SESSION_DIR/STOP_PANE_{watchdog,orchestrator,kernel-mgr}`
  and wait for `[run.sh] Done`.
- **Don't interpret** `$SPEC_ROOT/SKILL.md` / `MARATHON_PROTOCOL.md` / any
  actions/*.md yourself. Those are for the 3 panes.
- **Cleanup logs are expected**: lines like `killing inference server pid=...`
  or `killing stray inference process` near the end of `/tmp/marathon.log`
  are the cleanup trap doing its job (zombie prevention). Not an error.

### `$SKILL_ROOT`

The directory containing this `SKILL.md`. When paths need to be absolute:
`/shared_nfs/xiaofei/Hyperloom/.cursor/skills/marathon-inference-optimization`
(or wherever your Hyperloom checkout is — `$SKILL_ROOT` is safer).

## Why 3 panes

Each `claude` CLI has its own 200K context window and can run indefinitely via
`--continue`. One agent cannot survive a 24h run (context fills in ~8–12h). The
3-pane split also gives genuine parallelism:

- **orchestrator** — owns the DFS loop and the inference server lifecycle
- **kernel-mgr**  — runs OOB backends (GEAK / Codex / Claude) for kernel rewrites (async)
- **watchdog**   — tails `event_log.jsonl` and produces RCA findings (async)

Coordination is entirely file-based (see `MARATHON_PROTOCOL.md` §IPC).

## Required env

The user prompt must supply:

| Var | Meaning |
|-----|---------|
| `MODEL_NAME` | identifier, e.g. `DeepSeek-R1-0528`, `Qwen3-8B` |
| `BASE_DIR`   | baseline / sprint-handoff / prior-session root (auto-created if missing) |

## Sandbox-injected env (agent need not set)

These are pre-populated by the Claw GPU sandbox. For local / non-sandbox runs, the
user must export them before invoking the skill.

| Var | Source |
|-----|--------|
| `ANTHROPIC_AUTH_TOKEN` | sandbox (copied to `ANTHROPIC_API_KEY` by `run.sh`) |
| `ANTHROPIC_BASE_URL`   | sandbox → OCI LLM gateway |
| `SAFE_API_KEY`         | sandbox (falls back to `ANTHROPIC_AUTH_TOKEN` if unset) |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | sandbox (`claude-sonnet-4-6`) |

## Optional env (defaults shown)

| Var | Default | Notes |
|-----|---------|-------|
| `MAX_HOURS` | 24 | wall-clock budget; hard kill at end |
| `FRAMEWORK` | sglang | `sglang` \| `vllm` |
| `MODEL_CLASS` | moe_mla | `dense` \| `moe_mla` \| `moe_swa` \| `moe_mla_nsa` |
| `GPU_COUNT` / `GPU_TYPE` | `8` / `MI355X` | |
| `TP` / `EP` / `PRECISION` | `8` / `1` / `fp8` | |
| `CONC` / `ISL` / `OSL` | `64` / `1024` / `1024` | benchmark workload |
| `MODEL_PATH` | (empty) | absolute path to weights (optional for warm-start) |
| `IMAGE` | (empty) | container image for GEAK; empty → geak backend skipped |
| `KERNEL_OPT_WORKSPACE` | `control-plane-sandbox` | SaFE workspace id |
| `KERNEL_OPT_BACKENDS` | `geak,claude,codex` | OOB allowlist |
| `DRY_RUN` | 0 | `1` = preflight only, no tmux |
| `REPORT_INTERVAL_S` | 60 | monitor cadence |
| `PANE_CLAUDE_TIMEOUT_S` | `1800` (30 min) | per-`claude --print` wallclock cap; on expiry the pane launcher restarts with `--continue` |
| `PANE_MAX_RESTARTS` | `200` | restart-loop cap per pane |
| `PANE_CLAUDE_DEBUG` | `0` | `1` injects `ANTHROPIC_LOG=debug` into inner `claude --print` calls for hang diagnosis |

## How `run.sh` self-adapts

The script does the choosing for you:

| Concern | Detection | Behaviour |
|---------|-----------|-----------|
| sandbox vs local | `ENGINE_TYPE=claude` AND `/app` exists | sandbox: `SESSION_DIR=/workspace/hyperloom/marathon-sessions/<ts>/` (S3-synced); local: `SESSION_DIR=${WORKSPACE_ROOT:-/shared_nfs/xiaofei/marathon-sessions}/<ts>/` |
| `tmux` / `jq` / `curl` missing | `command -v` check | local mode: `apt-get install -y` |
| `claude` CLI missing | `command -v claude` | local mode: `npm install -g @anthropic-ai/claude-code`; sandbox always has it |
| `STRICT=1` | env flag | `run.sh` exits 20 if any required var (MAX_HOURS / FRAMEWORK / MODEL_CLASS / GPU_COUNT / GPU_TYPE / TP / EP / PRECISION / CONC / ISL / OSL) is unset — catches "agent only exported MODEL_NAME and silently fell back to defaults" |

## Prompt template — just the params, Rules live here in SKILL.md

The "Your role" section above is the full operating contract for the agent.
The user's prompt therefore only needs to supply the env vars (what changes
per run) and a trigger pointing at this skill. No Rules to repeat.

### Cursor / local GPU host

```
@marathon-inference-optimization

PANE_CLAUDE_DEBUG=1 \
STRICT=1 MODEL_NAME=deepseek-ai/DeepSeek-R1-0528 \
BASE_DIR=/shared_nfs/xiaofei/marathon-runs/dsr1-$(date +%m%d-%H%M) \
MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528 \
MAX_HOURS=2 FRAMEWORK=sglang MODEL_CLASS=moe_mla \
GPU_COUNT=8 GPU_TYPE=MI355X TP=8 EP=1 PRECISION=fp8 \
CONC=64 ISL=1024 OSL=1024 \
IMAGE=harbor.oci-slc.example-internal-host.invalid/custom/lmsysorg/sglang:202603270958 \
KERNEL_OPT_WORKSPACE=control-plane-sandbox KERNEL_OPT_BACKENDS=claude \
INFERENCEX_PATH=/hyperloom/InferenceX \
bash $SKILL_ROOT/scripts/launcher/run.sh > /tmp/marathon.log 2>&1 &
echo "marathon_pid=$!"
```

### Claw sandbox

Same body, plus one sandbox header:

```
@marathon-inference-optimization

SandboxImage: harbor.oci-slc.example-internal-host.invalid/custom/lmsysorg/sglang:202603270958

STRICT=1 MODEL_NAME=... BASE_DIR=/workspace/hyperloom/... MODEL_PATH=... \
MAX_HOURS=... (...same as above...)
```

The agent reads this SKILL.md when `@marathon-inference-optimization` is
triggered (or when the glob `**/marathon*` matches), follows the "Your role"
section, constructs the `bash $SKILL_ROOT/scripts/launcher/run.sh > /tmp/marathon.log 2>&1 &`
invocation from the env vars, polls with the `kill -0` + `tail -200` command above, and forwards
the final SESSION_REPORT when `[run.sh] Done` appears.

### Sandbox vs local — what differs

| Field | Cursor (local GPU) | Claw (sandbox) |
|-------|--------------------|-----------------|
| Skill trigger mechanism | `@marathon-inference-optimization` or glob match — SKILL.md auto-loaded into agent context | Sandbox agent reads SKILL.md explicitly; include `SandboxImage:` header so Claw provisions the right pod |
| `BASE_DIR` | any path the GPU host can write | `/workspace/hyperloom/...` to get S3 sync |
| `MODEL_PATH` / `INFERENCEX_PATH` | NFS paths (same as sandbox, e.g. `/hyperloom/...`) | same |
| `claude` CLI / `tmux` / `jq` | `run.sh` auto-installs if missing | preinstalled in sandbox image |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL` / `SAFE_API_KEY` | user must export before triggering skill | sandbox injects automatically |

`run.sh` auto-detects sandbox vs local via `ENGINE_TYPE=claude && /app exists`,
so the script itself needs no mode flag.

## Observability surface (on shared NFS)

```
$SESSION_DIR/
├── state.json                  # orchestrator writes every ~60s
├── SESSION_REPORT.md           # final report
├── checkpoints/latest/         # every 30min + every KEEP
├── kernel_manager/
│   ├── work_queue.jsonl        # orch → km
│   ├── results.jsonl           # km  → orch
│   ├── merge_ready/<id>/       # km  → orch (patch dirs)
│   ├── event_log.jsonl         # orch+km → watchdog
│   ├── findings.jsonl          # watchdog → orch+km
│   └── rca_reports/<id>/
└── logs/{watchdog,orchestrator,kernel-mgr}.log
```

The panes read/write these files directly — you observe them through the progress
lines that `run.sh` emits.

## Graceful stop

`run.sh` installs an `EXIT/INT/TERM` trap that touches `STOP_PANE_<name>` files
so each pane's while-loop exits cleanly after its current `claude` call, then
`tmux kill-session`. If the user sends SIGTERM (e.g. cursor cancels the skill),
every pane's in-flight claude call finishes within ~60s and the final
`SESSION_REPORT.md` is written if the orchestrator sees the STOP.

## What NOT to do

- Do NOT `exec_on_gpu`-wrap any command. The panes run natively on GPU hosts
  (sandbox or local); `modes/CLAW.md` is legacy and the pane prompts explicitly
  override it.
- Do NOT write to `/shared_nfs/.../state.json` directly from your monitoring
  code. The orchestrator pane owns it.
- Do NOT kill server / `pkill -9` / `fuser -k`. The orchestrator pane owns
  all server lifecycle (it records the authoritative PID to
  `/tmp/.marathon_server.pid` and kills rogues on each launch).

## Protocol spec is referenced, not copied

This skill is **thin** (~800 LoC in total): just `SKILL.md` + `scripts/launcher/`.
The actual Marathon protocol (`SKILL.md`, `actions/`, `kernel-manager/`,
`watchdog/`, `modes/`, `kb/`, `scripts/` with `common.sh` / `run_baseline.sh`
etc., `KNOWLEDGE-BASE.md`) is **referenced in-place** from Neha's upstream
location, defaulting to:

```
SPEC_ROOT=/hyperloom/Hyperloom/marathon_optimization/marathon_harness/skills
```

`run.sh` falls back to `/shared_nfs/xiaofei/Hyperloom/marathon_optimization/...`
if `/hyperloom/Hyperloom/` is not mounted (e.g. developer machines). Override
`SPEC_ROOT` env to point elsewhere.

**Why** — so (a) skill packages stay small and trivial to zip/upload, (b) Neha's
upstream bug fixes and KB updates flow through without a skill republish, and
(c) there's a single source of truth for the DFS protocol.

The three pane launcher prompts (`scripts/launcher/pane_*.md`) tell their
`claude` CLI to `Load and follow $SPEC_ROOT/SKILL.md` (and
`$SPEC_ROOT/kernel-manager/SKILL.md` / `$SPEC_ROOT/watchdog/SKILL.md`
respectively), and explicitly IGNORE `$SPEC_ROOT/modes/CLAW.md` since the
pane IS on the GPU host natively.

## Deeper reading (for when the user asks "what is marathon doing now")

- `$SPEC_ROOT/SKILL.md` — the full DFS / IPC / Iron Rules / scoring protocol
- `$SPEC_ROOT/kernel-manager/SKILL.md` — OOB dispatch, deep guidance loop, merge-ready
- `$SPEC_ROOT/watchdog/SKILL.md` — RCA methodology, findings schema
- `$SPEC_ROOT/KNOWLEDGE-BASE.md` — validated lessons from prior runs
- `$SPEC_ROOT/actions/*.md` — per-action execution details (deep-kernel-analysis,
  operator-tuning, framework-rebuild, comm-optimization, compiler-tuning, dream,
  checkpoint, re-explore, recover, sweep, report)
