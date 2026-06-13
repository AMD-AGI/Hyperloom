---
name: inference_optimizer
description: |
  Launches and monitors Hyperloom's multi-agent inference optimizer for LLM
  serving on AMD GPUs. Use when the user asks to optimize an inference model,
  run Magpie benchmarks/profiles, resume an inference_optimizer session, tune
  SGLang/vLLM serving parameters, run TraceLens/kernel-agent, or validate
  end-to-end throughput gains in a new inference environment.
globs:
  - "**/inference*optim*"
  - "**/inference_optimizer*"
---

# Inference Optimizer Skill

You are the **launcher and monitor**. The optimizer itself is the Python
`inference_optimizer` runtime under this repository. Do not manually optimize in
chat unless debugging; launch the CLI, poll persisted state, and report
objective progress.

## Quick Start

The whole job is three steps:

1. **Install** (IR-2): run `inference_optimizer/scripts/install.sh`, then source
   the generated `runtime/kernel-agent.env.sh` in the same shell.
2. **Launch**: verify GPU cleanliness (IR-1), then start
   `inference_optimizer --verbose optimize` under `setsid nohup` with
   `--launch-info-file`.
3. **Monitor & report**: learn `SESSION_DIR` from launch-info, poll
   `state.json` every 300s (unless debugging startup), surface lifecycle lines,
   and resume only unexpected crashes via same-session
   `optimize --resume --resume-from "$SESSION_DIR"`.

A user request to optimize a model is approval to run install + launch on a
fresh node; do not stop for an extra confirmation (the one exception:
environment mutation during kernel apply; see [launcher/kernel.md](launcher/kernel.md)).

Before executing install / launch / resume / monitor commands, read
[launcher/operations.md](launcher/operations.md). Keep the main file lean; the
launcher templates and helper scripts live under `launcher/`.

## Reference Index (read on demand)

| When you need… | Read |
|---|---|
| Install, preflight, launch, resume, monitor templates | [launcher/operations.md](launcher/operations.md) |
| Quantization prelude (`--quantize`) | [launcher/quantization.md](launcher/quantization.md) |
| Session/workspace layout, where `manifest.json`/`state.json` live, path rules | [launcher/paths.md](launcher/paths.md) |
| Framework choice (sglang/vllm/atom), GPU runner type, `--gpu-type` | [launcher/frameworks.md](launcher/frameworks.md) |
| Benchmark YAML fields, workload-contract reuse, leak-path salvage | [launcher/benchmark.md](launcher/benchmark.md) |
| Critic backend modes, `--critic-mock`, critic env | [launcher/critic.md](launcher/critic.md) |
| Cold-start / aiter JIT / cache timeouts | [launcher/cache.md](launcher/cache.md) |
| Kernel harness, apply safety, E2E retry discipline | [launcher/kernel.md](launcher/kernel.md) |
| Recovery, auth/SDK drift, error tables, `stop_reason` meanings | [launcher/troubleshooting.md](launcher/troubleshooting.md) |
| Coordinator internals: EXPLORE (IR-4/6/7), FRAMEWORK_PR, retired modules, expected flow | [launcher/internals.md](launcher/internals.md) |
| Multi-node (`--nodes >= 2`) | [multi_node/SKILL.md](multi_node/SKILL.md) |

## What This Skill Runs

The CLI starts a Python Coordinator that coordinates orchestration, kernel,
critic, and robustness agents:

- **Orchestration** chooses `baseline`, `explore`, `specialist`,
  `integrate_patch`, `sweep`, kernel requests, and `report`.
- **Kernel** handles `trace_analyze`, `run_optimization`, and `integrate`.
- **Critic** reviews proposals; default is `--critic-agent`.
- **Robustness** monitors in-loop health; default is `--robustness-agent`.
  Launcher crash recovery is separate and may only relaunch the same session via
  `optimize --resume --resume-from "$SESSION_DIR"`.

For `--nodes >= 2`, robustness auto-downgrades to `--robustness-mock` because
local probes would target the wrong pod and flood the bus with false positives.
Pass `--robustness-mock` explicitly to suppress the warning. Read
[multi_node/SKILL.md](multi_node/SKILL.md) before multi-node launches.

State lives under a **session directory** (per run); the **workspace root** is
`$USER_DATA_PATH` (default `/workspace/hyperloom`). Always prefer
`manifest.json` / `state.json` / `coordinator.db` under the session dir over
terminal logs. Full layout and path-resolution rules:
[launcher/paths.md](launcher/paths.md).

## Iron Rules

SKILL-level constraints the launcher MUST satisfy **before** `inference_optimizer
optimize` is even spawned.

### IR-1 — GPU MUST be unoccupied before every launch

Before every `optimize` invocation, fresh or `--resume`, verify every visible GPU
has **zero foreign serving PIDs and ≲ 500 MiB VRAM in use**. Leftover
`sglang.launch_server`, `vllm.entrypoints`, or `Magpie` processes silently
degrade the next baseline; `current_best` cannot detect this after the fact. The
portable check lives in [launcher/operations.md](launcher/operations.md).

> The in-loop equivalent is Kernel-agent IR-4 (`kill_server` +
> `check_gpu_memory` before every server (re)start — see
> `orchestrator/system_prompts/kernel.md`). IR-1 is the *outer* gate.

### IR-2 — install.sh MUST succeed before every launch

Run `bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"` and source the
regenerated
`${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}`
in the **same shell** that spawns `optimize`. Skipping install strikes silently
*after* `baseline` succeeds: missing TraceLens/GEAK/OOB CLI → `trace_analyze` /
`kernel_opt` fail; no live Ray head → `kernel_opt` hangs; missing
`kernel-agent.env.sh` → first claude/codex call returns `401`.
`install.sh --check-only` is a *diagnostic*, never a substitute.

**Resume carve-out.** `optimize --resume` may skip install only when ALL hold:
(1) `install.sh` exited 0 earlier in the *same shell*; (2) `kernel-agent.env.sh`
is still sourced; (3) the selected session has `manifest.json` and `state.json`.
Any failure → treat as fresh launch and re-run `install.sh`.

> The in-loop equivalent is `_preflight()` (drift repair, not a substitute).

### IR-3 — KB + PR Monitor reachability (in-loop, soft degrade)

`_preflight()` runs `bash inference_optimizer/scripts/preflight_kb.sh`. IR-3
**never aborts launch**: exit `0` = both reachable; exit `1` = the CLI
auto-enables the matching `--degraded-*` and records `*_degraded_reason=ir3_auto`
in `manifest.json`. Operator opt-out: `--degraded-kb` / `--degraded-pr` (records
`reason=explicit_flag`); both together short-circuit IR-3.

### IR-8 — `--framework atom` is single-node only

`_apply_atom_auto_tighten` rejects `--nodes >= 2` with `SystemExit(2)` (atom
upstream has no multi-node TP wiring). No other flag is auto-flipped —
kernel-agent / framework-agent / profile / roofline / TraceLens all run on atom.
Details: [launcher/frameworks.md](launcher/frameworks.md).

> **EXPLORE contracts (IR-4/6/7) and the FRAMEWORK_PR phase are
> Coordinator-internal** — the launcher never proposes or drives them. See
> [launcher/internals.md](launcher/internals.md).

## Required Workflow

Read [launcher/operations.md](launcher/operations.md) before running these
commands. This section is only the control flow.

1. Set credentials and runtime roots. `SAFE_API_KEY` and `OPENAI_BASE_URL` must
   be present in the shell environment before install or launch. Optional
   source-root overrides (`OOB_SRC`, `INFERENCEX_PATH`, `TRACELENS_ROOT`,
   `TRACELENS_INTERNAL_ROOT`) are local sandbox inputs only.
2. Run install and source the generated env in the same shell. Do not manually
   pip-install SDKs, edit `~/.claude/config.json`, start Ray, or repair
   `runtime/source-mirrors/`; `_preflight()` and `install.sh` own those.
3. Optionally write `$USER_DATA_PATH/model_arch.json` if the model architecture
   is known. It is advisory only; skip rather than guessing.
4. Run the IR-1 portable preflight after install. Never print tokens.
5. Launch with `setsid nohup`, `--launch-info-file`, and `$USER_DATA_PATH` set to
   the workspace root, not a session dir.
6. Read `SESSION_DIR` from launch-info (`jq -r .session_dir "$LAUNCH_INFO_FILE"`).
   Refuse to guess by timestamp.
7. For runs longer than 5 min, copy the robustness monitor template from
   `inference_optimizer/launcher/robustness_monitor.sh.example` to the runtime
   `optimizer_runs/robustness_monitor.sh` and start it. Its only allowed
   relaunch is same-session `optimize --resume --resume-from "$SESSION_DIR"`
   after an unexpected crash; no fresh run and no latest-session auto-pick.

## Report Back To User

Report concise status:

- session id (from `manifest.json`), `SESSION_DIR`, and log path
- `cumulative_gain` and `current_best`
- explore accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why

Surface lifecycle lines from `state.json` verbatim. For internal expected flow,
read [launcher/internals.md](launcher/internals.md). For `stop_reason`
meanings and runtime signals, read
[launcher/troubleshooting.md](launcher/troubleshooting.md).
