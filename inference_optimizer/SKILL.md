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

You are the launcher and monitor. The optimizer itself is the Python
`inference_optimizer` runtime under this repository. Do not manually optimize
inside chat unless debugging; launch the CLI, poll persisted state, and report
objective progress.

## What This Skill Runs

The CLI starts a Python Coordinator that coordinates:

- Orchestration: decides next actions (`baseline`, `profile`, `backends`, `params`, `sweep`, Kernel requests, `report`).
- Kernel: responder path for `select_kernels`, `run_optimization`, `integrate`.
- Critic: proposal review (default: `--critic-agent` — drives the
  `critic-agent/` skill runtime with KB priors / session memory /
  `review_constraints`-gated verdicts). `--critic-mock` for offline /
  smoke tests; `--critic-codex-bare` for debugging the LLM layer
  without the runtime layer.
- Robustness: default `--robustness-agent` — drives the `robustness-agent/`
  subprocess runtime for health monitoring, RCA, and scheduling-police
  intents. `--robustness-mock` for offline / smoke tests.

State lives in **one fixed session directory** — `/workspace/hyperloom`
by default. v0.6.1 collapses the previous `<root>/<session_id>/` layout
to a flat directory because every sandbox is single-use; there is no
session_id in the path. Override only for tests via
`$INFERENCE_OPTIMIZER_SESSION_DIR`.

```text
/workspace/hyperloom/                     # session_dir (fixed)
├── manifest.json                         # Python-written session resume tag
├── state.json                            # SharedState (Coordinator-owned)
├── storage/coordinator.db                # SQLite WAL
├── agents/{orchestration,kernel,critic,robustness}/
│   ├── inbox.jsonl  outbox.jsonl
│   ├── persona.md
│   └── system_prompt.snapshot.md
├── personas/  checkpoints/  findings/  kb/
├── runs/                                 # data-plane (executor outputs)
│   ├── baseline/<task_id>/
│   ├── profile/<task_id>/
│   ├── backends/<task_id>/{variant_NN_*/, result.json}
│   ├── params/<task_id>/{variant_NN_*/, combo/, result.json}
│   ├── sweep/<task_id>/
│   ├── integrate/<task_id>/
│   └── kernel_opt/<kernel_id>/<task_id>/
├── kernel-agent-workspace/<kernel_id>/   # cross-task GEAK/OOB artefacts
├── patches/<kernel_id>/                  # KEEP'd patches + backup
├── reports/                              # `report` action output
└── logs/                                 # cli + reactor + auth-proxy logs
```

Paths emitted by agents must resolve under `$SESSION_DIR` — PolicyGate
enforces this (with a framework-source allowlist for `source_file`:
`/sgl-workspace/{aiter,sglang,vllm}/`).

Always prefer `manifest.json` / `state.json` / `coordinator.db` over
guessing from terminal logs.

Session dir resolution order (`inference_optimizer/paths.py`):
1. `$INFERENCE_OPTIMIZER_SESSION_DIR` env → use as-is.
2. Default `/workspace/hyperloom`.

Path helpers (don't string-concat):

| Helper | Returns |
|---|---|
| `paths.session_dir()` | `/workspace/hyperloom` (or env override) |
| `paths.make_session_dir()` | session dir + full skeleton, idempotent |
| `paths.db_path_for(sd)` | `<sd>/storage/coordinator.db` |
| `session_paths.runs_dir(sd, kind, task_id)` | `<sd>/runs/<kind>/<task_id>/` |
| `session_paths.kernel_workspace(sd, kernel_id)` | `<sd>/kernel-agent-workspace/<kernel_id>/` |
| `session_paths.patches_dir(sd, kernel_id)` | `<sd>/patches/<kernel_id>/` |
| `session_paths.agent_log(sd, role)` | `<sd>/logs/<role>.log` |
| `session_paths.agent_prompt_snapshot(sd, role)` | `<sd>/agents/<role>/system_prompt.snapshot.md` |
| `manifest.write_manifest(sd, args)` / `load_manifest(sd)` | manifest.json read/write |

## Setup

This skill is **two commands**. Do NOT replicate setup steps inside chat —
both commands are idempotent, do auto-detection, and re-run safely.

### Credentials (env only)

`SAFE_API_KEY` and `OPENAI_BASE_URL` are the only credentials this skill
needs and must be exported in the calling shell before running install
or the CLI (typically by sourcing `$HYPERLOOM_KERNEL_AGENT_ROOT/env.sh`
after Step 1). `install.sh` and the CLI's `_preflight()` read them from
`os.environ` only — no `.env` files are loaded.


### Step 1 — Install (one-time per pod / venv rebuild)

```bash
export REPO_ROOT="$(pwd)"   # repo root containing kernel-agent/ + inference_optimizer/ + .env
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"   # pod-local runtime env
```

`inference_optimizer/scripts/install.sh` is the only install entrypoint for
full inference optimization. It installs the optimizer / Magpie / InferenceX
first, then chains to `kernel-agent/scripts/install.sh` for the kernel
optimization environment. `kernel-agent/scripts/install.sh` remains valid for
standalone kernel-agent debugging, but should not be the main entrypoint for a
full inference optimizer session.

The install phase always initializes the full Hyperloom runtime. Even if the
user later passes `--no-kernel` at runtime, the installer still prepares
kernel-agent / TraceLens / GEAK / OOB / auth-proxy; `--no-kernel` only means
that this `optimize` run skips the kernel optimization phase.

`kernel-agent/scripts/install.sh` installs everything in one shot (no
`--with-*` flags to remember):

| Component | Provided by |
|---|---|
| `ray==2.44.1` + `click<8.3.0` | pip |
| TraceLens internal (perf-report CLI) | `ensure_tracelens` (`cp -r` from read-only WekaFS mount to `${HYPERLOOM_ROOT}/TraceLens-internal`) |
| GEAK CLI + `${HYPERLOOM_ROOT}/geak-config/local.yaml` | `ensure_geak` |
| Node.js/npm + OOB CLI + claude/codex npm CLIs + `@cursor/sdk` global install + `~/.claude/config.json` + `~/.codex/auth.json` | `ensure_node` + `ensure_oob` (mirrors `${HYPERLOOM_BUNDLE}/OOB` → `${HYPERLOOM_ROOT}/OOB/oob_cli`) |
| OOB auth-proxy on `127.0.0.1:4002` (rewrites `x-api-key` → `Authorization: Bearer`; without it Claude SDK returns 401) | `ensure_auth_proxy.sh` |
| `CURSOR_API_KEY` / `CURSOR_DEFAULT_MODEL` exported to `kernel-agent.env.sh` if set in env (cursor backend uses Cursor's own gateway, not the `:4002` proxy) | `write_env_file` |

`${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}` is
regenerated by `install.sh` and contains the proxy-rewritten URLs, auth aliases,
GEAK config path, and InferenceX path. Source it (don't try to derive these by
hand). Generated env/config state is written to the pod-local runtime directory,
not back into a shared WekaFS source checkout.

**Multi-node escape hatch**: if `$TRACELENS_ROOT` / `$OOB_SRC` / `$GEAK_REPO` /
`$WORKSPACE_ROOT/Magpie` / `$INFERENCEX_PATH` may move or differ across nodes,
`rsync -a` them into `$SESSION_DIR/vendor/<name>/` and override the matching
env vars BEFORE running `install.sh`. Single-node WekaFS-mount setups (the
production default) need none of this — `ensure_tracelens` / `ensure_oob`
already handle the read-only-source case.

### Step 2 — Launch

```bash
inference_optimizer optimize \
  --model "$MODEL_PATH" \
  --framework vllm \           # or sglang (default)
  --gpu-type MI300X \          # or omit for rocm-smi auto-detect
  --max-hours 2
```

Install or validate the optimizer + downstream stack with the bundled
installer. It is idempotent and chains to `kernel-agent/scripts/install.sh`,
so a single call covers: inference_optimizer + `claude_agent_sdk` extras,
Magpie, InferenceX detection, Ray (with a live ray head started), Node.js/npm,
TraceLens CLI, GEAK + OOB CLI, the OOB auth-proxy on `:4002`, and the pod-local
`kernel-agent.env.sh`.
A user request to optimize a model is approval to run this on a fresh node;
do not stop for an extra confirmation:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"

export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"
"$PYTHON" -m inference_optimizer.cli --help
```

Notes: with `set -u` active, assign dependent vars on separate lines (Bash
expands RHS before assigning, so chained `export A=... B=$A` can fail with
`unbound variable` on a clean environment). The installer leaves a live Ray
head; `ray status` must succeed because `select_kernels` submits Ray tasks
with `num_gpus>=1` — never restart Ray with `--num-gpus=0`.

The CLI runs `_preflight()` on every launch as a safety net for `install.sh`.
Steps 1–9 run before `_preflight()` returns; 10–12 run before Coordinator
boots. Cite the linked section for fixes:

| #  | Check | On-fail / Reference |
|----|---|---|
| 1  | Re-export auth aliases (`ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`, ...) from `SAFE_API_KEY`. `OOB_BASE_URL` / `GEAK_BASE_URL` / `LLM_API_BASE` inherit `OPENAI_BASE_URL` directly (Bearer-native, no proxy). | — |
| 2  | Auto-`pip install` missing `claude-agent-sdk>=0.1.65`, `openai>=1.50`, `httpx>=0.27` into `sys.executable` | `## Failure Handling` — `claude-agent-sdk not installed` |
| 3  | Bootstrap `auth_proxy.py` source from `$OOB_SRC` (or `/wekafs/fully-local/{,inference_optimization/}OOB`) into `${HYPERLOOM_ROOT:-/opt/hyperloom}/OOB/oob_cli/` if missing | `## Failure Handling` — `Claude SDK exit code 1` |
| 4  | Re-run `ensure_auth_proxy.sh`; rewrite `~/.claude/config.json` `customApiUrl` and force-override `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` to the proxy URL (overriding any shell/`.env`/k8s value, logged on stdout). On retry-fail restores originals + WARN. | `### Recovery` |
| 5  | ROCm hygiene (WARN-only): pop `HIP_VISIBLE_DEVICES` if `ROCR_VISIBLE_DEVICES` also set; visible-GPU count vs `$TP` via `rocm-smi --showid`; `/dev/shm` free ≥ 16 GiB | — |
| 6  | Auto-install missing `ray` / `Magpie` / `InferenceX` (pod rebuild recovery) | — |
| 7  | Auto-detect `--gpu-type` if not given | `## GPU Runner Type` |
| 8  | WARN-only presence check: `node` / `claude` / `codex` CLIs + `@cursor/sdk` (resolved via `node -e "require.resolve('@cursor/sdk')"` against `$(npm root -g)`) | — |
| 9  | Emit canonical `Preflight diagnostics:` block (`asset_root`, `session_dir` + resolving env var, `magpie_python`, `INFERENCEX_PATH`, aiter jit cache WARM/COLD + `.so` count + path, cold/warm timeouts, proxy URL). Paste verbatim into status reports. | `## Cold-start Discipline` |
| 10 | Hard model gate: `--claude-model` ∈ {`claude-opus-4-7` (preferred), `claude-opus-4-6` (fallback)}; probe `GET <OPENAI_BASE_URL>/models` with Bearer (3 retries 1s/3s/5s); rewrite to `4-6` if `4-7` missing; abort if neither present or gateway unreachable | `## Failure Handling` — model-gate errors |
| 11 | Codex smoke-test (WARN-only): `--codex-model` checked when codex actually used (`--critic-agent` / `--critic-codex-bare` / `--kernel-codex`) | — |
| 12 | Critic-agent runtime probe (when `--critic-agent` active): resolve `CRITIC_AGENT_ROOT` (env > sibling `$REPO_ROOT/critic-agent/` > abort), `python -m runtime.cli --help` (5s timeout); abort rc=2 if it fails. Default-sets `WORKSPACE_PATH` / `CRITIC_SESSION_MEMORY_DIR` / `CRITIC_KB_CLIENT_MODE`. | `## Critic Backend Selection` |

`install.sh` is the canonical bring-up; `_preflight()` catches drift mid-run.
Don't manually pip-install SDKs, edit `~/.claude/config.json`, start Ray, or
`curl /v1/models` — `_preflight()` owns all of these. See `kernel-agent/SKILL.md`
for the chained installer truth.

### Recovery

If the CLI exits with `Claude SDK exit code 1` or `Primus.00009 token not present`,
the auth-proxy died. Re-run the supervisor and retry — both are idempotent:

```bash
bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"   # noop if healthy
inference_optimizer optimize ... # rerun
```

If `_preflight()` itself fails, run install in `--check-only` mode to see
which piece is missing, then re-run full install:

```bash
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh" --check-only
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
```

In sandboxes where `/workspace/hyperloom` is unwritable, override the
session location with a single env var:

```bash
export INFERENCE_OPTIMIZER_SESSION_DIR="$RUN_ROOT/optimizer-session"
mkdir -p "$INFERENCE_OPTIMIZER_SESSION_DIR"
```

The CLI calls `make_session_dir()` once at startup; that creates the
full subdirectory skeleton in place (idempotent — safe to re-run).

## Portable Preflight

Before every new model run, verify the model path, GPU visibility, and duplicate
processes. Never print tokens.

```bash
export MODEL_PATH=/path/to/model
test -d "$MODEL_PATH"

"$PYTHON" - <<'PY'
import os
try:
    import torch
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])

patterns = ("inference_optimizer.cli", "Magpie", "sglang.launch_server")
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    text = cmd.replace(b"\0", b" ").decode("utf-8", "ignore")
    if text and any(p in text for p in patterns):
        print(f"existing_process {pid}: {text[:300]}")
PY
```

## Benchmark Config

Default configs live here:

```bash
inference_optimizer/scripts/configs/baseline_sglang.yaml
inference_optimizer/scripts/configs/baseline_vllm.yaml
inference_optimizer/scripts/configs/profile_sglang.yaml
inference_optimizer/scripts/configs/profile_vllm.yaml
```

Two fields in each YAML are **fallback only** — the optimizer overrides
them at runtime:

- `benchmark.model` <- `--model` / `$MODEL_PATH`
- `benchmark.runner_type` <- `--gpu-type` / `$GPU_TYPE` / rocm-smi auto-detect

`benchmark.benchmark_script` is deliberately NOT set in the shipped
YAMLs. With `runner_type` injected at run time, Magpie picks
`{framework}_{runner_type}.sh` itself (e.g. `sglang_mi300x.sh` /
`sglang_mi355x.sh`). Each YAML has a commented `# benchmark_script: ...`
template right under `framework:` for manual debug overrides.

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.
- `benchmark.envs.PATH`: must lead with the launcher Python's bin dir
  (`$(dirname "$PYTHON")`).

### Workload-contract reuse (baseline → params/backends/sweep)

`baseline` materializes its YAML once with the operator's process env (`CONC` /
`ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL` /
`ROCR_VISIBLE_DEVICES` plus adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it
as `baseline_config.with_envs.yaml`, and forwards the path on
`SharedState.baseline_config_path` as `task.params["config_path"]` to every
`params` / `backends` / `sweep` task. Variants thus benchmark the **same
workload baseline ran**; without this contract they would render from the
YAML's smoke defaults (`TP=1` / `CONC=8` / `ISL=256` / `OSL=256`) and produce
~10x lower throughput (the historical fairness bug). Downstream actions
re-materialize on top of `config_path`; per-variant `extra_envs` (e.g. sweep's
explicit `CONC`/`ISL`/`OSL`) still win because `_grid_runner._build_variant_yaml`
applies them last.

## Critic Backend Selection

The Critic role has three backend modes, picked by mutually-exclusive
CLI flags. Default is `--critic-agent` (no flag needed).

| Flag | Backend class | Behaviour |
|---|---|---|
| (none) / `--critic-agent` | `CriticAgentBackend` | Drives the standalone `critic-agent/` skill runtime via `python -m runtime.cli prepare-review` → Codex chat completion → `python -m runtime.cli commit-review`. Adds KB priors lookup (with circuit-breaker for unreachable services), per-session memory + idempotent `reviewed_msg_ids` (no double-verdict), `judge_bundle.review_constraints` injected into the LLM prompt, and `needs_review` / `critic_unavailable` source when context is missing. |
| `--critic-mock` | `MockCriticBackend` | Always-approve adapter. Use for offline / smoke tests when Codex creds aren't available. |
| `--critic-codex-bare` | `CodexBackend` | Legacy direct chat-completion path with no KB / session memory / `review_constraints`. Available for debugging the LLM layer in isolation. (`--critic-real` is a hidden back-compat alias.) |

Default is overridable per pod via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` (one of `mock` / `agent` /
`codex_bare`).

### Required env when `--critic-agent` is active

| Var | Purpose | Default |
|---|---|---|
| `CRITIC_AGENT_ROOT` | Path to the directory containing `runtime/cli.py`. | sibling `$REPO_ROOT/critic-agent/` |
| `CRITIC_KB_CLIENT_MODE` | `inmemory` keeps KB writes / reads off the wire. `live` requires `KB_BASE_URL`. | `inmemory` |
| `KB_BASE_URL` | KB service URL when `CRITIC_KB_CLIENT_MODE=live`. | unset (live mode aborts at start if absent) |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | Forwarded to the runtime; see `critic-agent/AGENTS.md`. | runtime defaults |
| `CRITIC_SESSION_MEMORY_DIR` | Where the runtime persists per-session decisions / reviewed_msg_ids. | `$SESSION_DIR/critic-session-memory` (auto-set by the optimizer; co-located with the Coordinator session and cleaned up alongside it). |
| `WORKSPACE_PATH` | Skill root the critic-agent runtime resolves prompt assets against. | `$REPO_ROOT` (auto-set). |

`_preflight()` checks `CRITIC_AGENT_ROOT` resolves to a real directory
with `runtime/cli.py`, then runs `python -m runtime.cli --help` (5s
timeout) before the Coordinator boots. Missing or broken runtime
aborts the run with a clear error pointing at `--critic-mock` /
`--critic-codex-bare` as bypasses.

### Per-turn artefacts (audit trail)

Each Critic turn writes:

```text
$SESSION_DIR/critic-workdir/<turn_idx 6-digit>/
├── request.json         # raw_prompt + session_id passed to runtime.cli
├── judge_bundle.json    # output of prepare-review (proposals, KB priors,
│                          review_constraints, kb_read_skipped_reason)
├── review.json          # LLM's verdicts (extracted JSON envelope)
└── emit.json            # output of commit-review (intent_envelope +
                           kb_writes); the Coordinator consumes
                           intent_envelope verbatim.

$SESSION_DIR/critic-session-memory/<session_id>/
├── context.json          decisions.jsonl   events.jsonl
└── kb_priors_cache.json  reviewed_msg_ids.json
```

The backend prunes everything older than the latest 50 turn workdirs
on every tick to avoid unbounded growth.


## Framework Selection

A session is single-framework. Pick `sglang` (default) or `vllm` via
`--framework` or `$FRAMEWORK`:

```bash
inference_optimizer optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to
  (`baseline_sglang.yaml` / `baseline_vllm.yaml`,
  `profile_sglang.yaml` / `profile_vllm.yaml`)
- Which params grid `params` action runs (`DEFAULT_VLLM_PARAMS_GRID`
  vs `DEFAULT_PARAMS_GRID`)
- Which extra-args env name `_grid_runner` writes
  (`EXTRA_VLLM_ARGS` vs `EXTRA_SGLANG_ARGS`)
- Which Marathon KB partition orchestration reads for hints

Mixing sglang and vllm in a single session is not supported; the CLI
locks `$FRAMEWORK` for the run. Resume re-reads `$FRAMEWORK` from the
shell — set it when you resume a vLLM session.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without
either, the optimizer auto-detects via `rocm-smi --showproductname`
(falling back to `torch.cuda.get_device_properties(0).gcnArchName`).

```bash
inference_optimizer optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi325x`, `mi355x`. **`mi325x` is mapped to
`mi300x`** with a warning, since the two GPUs share the same arch and
Magpie has not shipped `sglang_mi325x.sh` / `vllm_mi325x.sh` yet. If you
need a true MI325X-specific script, uncomment the `benchmark_script:`
template in the relevant YAML and point it at your script under
`InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.

## SGLang Parameter Search

Validate SGLang first, add vLLM once SGLang is stable. `params` writes
candidates through `EXTRA_SGLANG_ARGS` and `benchmark.envs`; don't hard-code
a flag as default unless A/B keeps it across the target workload.

Default grid covers cuda graph batch caps, continuous decode steps, memory
fraction, scheduling conservativeness, chunked prefill, and max prefill tokens.
Also test the InferenceX-derived candidates:

- Cache/scheduler: `--disable-radix-cache`, `--max-running-requests 128/256`.
- Tokenization/streaming: `--tokenizer-worker-num 8/16`, `--stream-interval 30/50`.
- ROCm/TileLang envs: `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1`,
  `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`, `SGLANG_OPT_USE_TILELANG_INDEXER=true`.

Speculative decoding is model-specific — only enable `SGLANG_ENABLE_SPEC_V2=1` /
`--speculative-*` when the model has the required draft path or MTP support,
and benchmark with chat-formatted prompts (`--dsv4` for DeepSeek-V4 style)
because random prompts skew acceptance-rate results.

Judge candidates over **{1k/1k, 8k/1k} × {low CONC, high CONC}** (high-CONC
only when the model fits); KEEP only when throughput improves without
unacceptable TTFT/E2E or correctness regression. Coordinator long runs default
`max_candidates_per_round=5`; direct runner calls may pass `0` for the full grid.

### Per-Run Asset Override (advanced)

To override shipped configs without editing them, materialize a per-run asset
root and pass `--asset-root`. `mkdir -p "$ASSET_ROOT/scripts/configs"`,
`ln -sfn` `actions/` / `kernel_opt/` / `orchestrator/` and the two
`scripts/ab_torch_compile_*.py` from `$REPO_ROOT/inference_optimizer/`, then
copy + edit the relevant `baseline_*.yaml` / `profile_*.yaml`. Reach for this
only when `_workload_envs.materialize_config_with_envs` defaults don't fit
(e.g. per-yaml `profiler.torch_profiler.enabled`); otherwise `--model` /
`--gpu-type` overrides are enough.

## Launch a New Optimization

Assumes Step 1 (install) already ran. Session lives at `/workspace/hyperloom`
(override `$INFERENCE_OPTIMIZER_SESSION_DIR`); there is no `--session-name`.
For sandboxes that don't persist `export`s across shell calls (Cursor agents),
copy `inference_optimizer/scripts/setup_env.sh.example` to
`optimizer_runs/setup_env.sh`, fill in the workload block, and `.` it each call.
After `setsid nohup ... &`, locate the optimizer via
`pgrep -af 'inference_optimizer.*optimize'` — `$!` may be a wrapper PID.

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.log"
export PID_FILE="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.pid"
mkdir -p "$REPO_ROOT/optimizer_runs"

setsid nohup inference_optimizer --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

`setsid nohup ... &` is required for runs > 5 min — Cursor's background
shell can die on SSH disconnect.

Critic defaults to `--critic-agent`; Robustness defaults to `--robustness-agent`.
See [Critic Backend Selection](#critic-backend-selection) for `--critic-mock` /
`--critic-codex-bare` overrides; pod-level overrides via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` /
`INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND`.

After launching, do a short health check:

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"
session_dir="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
test -f "$session_dir/manifest.json" && echo "manifest_present=true"
test -f "$session_dir/state.json" && echo "state_exists=true" \
  && python3 -c "import json; print(json.load(open('$session_dir/state.json')).get('stop_reason'))"
```

Healthy = optimizer process alive + `manifest.json` + `state.json`
exist + no early `stop_reason`.

## Resume Existing Session

`--resume` is a flag (no argument); it picks up `/workspace/hyperloom`
(override: `$INFERENCE_OPTIMIZER_SESSION_DIR`). The CLI refuses to start
if `manifest.json` or `state.json` is missing.

Reuse the Launch template above with these diffs: drop `--model`, add
`--resume`, set `RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`. Resume preserves
baseline, current best, params-search state, event history, and kernel-agent
artifacts; the CLI clears stale `stop_reason` and `crash_count` before retrying.

## Robustness Monitor for Long Runs

For runs > 5 min, start a monitor in its own `setsid nohup` process. It polls
`state.json` every 5 min, exits on terminal `stop_reason` (`target_reached` /
`no_more_leverage` / `time_exhausted` / `max_ticks`), and resumes via
`--resume` when the optimizer dies unexpectedly.

```bash
cp "$REPO_ROOT/optimizer_runs/robustness_monitor.sh.example" \
   "$REPO_ROOT/optimizer_runs/robustness_monitor.sh"
chmod +x "$REPO_ROOT/optimizer_runs/robustness_monitor.sh"
setsid nohup bash "$REPO_ROOT/optimizer_runs/robustness_monitor.sh" \
  > "$REPO_ROOT/optimizer_runs/robustness_monitor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 < /dev/null &
```

Reads `$REPO_ROOT`, `$PID_FILE`, and (optional) `$INFERENCE_OPTIMIZER_SESSION_DIR`
/ `$MAX_HOURS` / `$TARGET_GAIN`. Edit the example before copying if defaults
need to change. `stop_reason` interpretation matches the `## Monitoring` reader.

## Monitoring

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "last_select_kernels", "last_sweep"):
    print(f"{k}: {s.get(k)}")
print("params_search_last_round:", s.get("params_search", {}).get("last_round"))
PY
```

Recent action counts from SQLite (last 500 events grouped by category):

```bash
python3 "$REPO_ROOT/inference_optimizer/scripts/event_counts.py" "$SESSION"
```

## Expected Flow

The optimizer should:

1. Establish or reuse `baseline_tput`.
2. Run `profile` only when the active server args differ from
  `last_profile_args`; otherwise reuse `last_profile_trace`.
3. Run `select_kernels` once per trace/config and cache the result in
  `last_select_kernels`.
4. Pick only `reusable_native_kernel_ids` for `run_optimization`.
5. Require compile + correctness + microbench/E2E evidence before KEEP.
6. Use `params_search` to test parameters incrementally and remember rejected
  candidates across resume.
7. Use `optimization_stack` so backend + params + kernel changes do not
  overwrite each other.
8. Use `sweep` to understand workload-specific results beyond the smoke
  workload.

## Cache Topology & Cold-start Discipline

SGLang/vLLM on ROCm route hot fused kernels (RMSNorm / attention / MoE / GEMM /
RoPE) through `aiter`, which JIT-compiles per-shape variants on first sight
and caches `.so` on disk. First launch of a fresh (model, dtype, TP,
`max_model_len`, `max_num_seqs`, `gpu_memory_utilization`) signature can spend
30+ min in `hipcc` for 671B FP8 MoE; later launches reuse the cache in seconds.

### Cache locations

| Cache | Path | Clear |
|---|---|---|
| aiter JIT (primary cold-start cost) | `<aiter pkg root>/jit/` (resolved via `import aiter`; wheel installs hold ~80 pre-built `.so` here, plus runtime-JIT staging under `jit/build/<module>/build/`) | `rm -rf <aiter pkg root>/jit/build/` (clears JIT staging only; do NOT delete `jit/*.so` — those are wheel-bundled) |
| Triton | `~/.triton/cache/` (resolves via `$HOME`) | `rm -rf ~/.triton/cache` |
| torch.compile / Inductor | `/tmp/torchinductor_<user>/` (override `$TORCHINDUCTOR_CACHE_DIR`) | `rm -rf /tmp/torchinductor_root` |

`sgl_kernel` (`site-packages/sgl_kernel/common_ops.*.so`) is build-time only;
only `kernel_opt` / `integrate` may rebuild it.

### Cold-start triggers

First launch on this pod; change to `--max-model-len` / `--max-num-seqs` /
`--gpu-memory-utilization` / `--cuda-graph-max-bs` / `--quantization` /
`--enable-torch-compile`; pod rebuild; manual cache `rm`; aiter source patch.

### Auto-detection + timeout

`BaselineExecutor` (and `ProfileExecutor`, which subclasses it) counts `.so`
files recursively under aiter's `jit/` dir, resolved via
`importlib.util.find_spec("aiter")` (with `jit/build/` as secondary
candidate and `baseline.py:AITER_JIT_PROBE_PATHS` as legacy fallback).
Threshold: **< 20 = COLD**. Override via `INFERENCE_OPTIMIZER_AITER_JIT_DIR`.

| Condition | `subprocess.run(timeout=...)` |
|---|---|
| `task.params['timeout_sec']` set | task value (always wins) |
| Probe `found` + COLD | `BASELINE_COLD_START_TIMEOUT_SEC` (3600s; override `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=N`) |
| Probe `found` + WARM | `BASELINE_DEFAULT_TIMEOUT_SEC` (2400s) |
| Probe `not_found` / `error` | 2400s + WARN |

Every launch logs one `baseline_executor: ...` marker (COLD_START / WARM /
explicit / not located); grep `optimizer_runs/run_*.log` to verify. Resolved
cache state also lands in the boot `Preflight diagnostics:` block. If
COLD_START repeats across retries, JIT was killed mid-`hipcc`; bump
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400` instead of relaunching.

## Kernel Apply Safety

Kernel optimization may modify `/sgl-workspace/aiter`, `/sgl-workspace/sglang`,
or compiled artifacts. Before applying a patch:

- Back up source files.
- Back up compiled `.so` / `.co` artifacts when available.
- On REVERT, restore compiled artifacts first, then source files, then restart
  the server. Avoid a rebuild on revert when the original compiled artifact was
  backed up.
- Only KEEP when correctness and E2E are acceptable.

If the user has not explicitly approved environment mutation, stop before real
apply/rebuild and ask. Dry-run and analysis are safe.

## Kernel E2E Retry Discipline

Microbench speedups are not enough. After `run_optimization` returns a candidate
kernel patch, `integrate` must validate the patch with E2E Magpie throughput and
record every attempt in `state.json`.

For the same `kernel_id + patch_path + EXTRA_SGLANG_ARGS`:

- `KEEP`: accept only when E2E gain clears the configured threshold.
- `REVERT`: reject that patch immediately and do not run it again.
- `NEEDS_REVIEW`: allow at most 3 E2E attempts. If none clears the KEEP
  threshold, reject that patch and move on to params search or a different
  reusable native kernel.

Do not repeatedly integrate the same patch because its microbench was strong.
If E2E results are unstable around zero gain, the correct action is to mark the
patch rejected, preserve the artifacts for human review, and spend the remaining
budget on untested params/backend candidates or the next kernel.

## Failure Handling

Auth / SDK drift (`Claude SDK exit code 1`, `Primus.00009 token not present`,
`ANTHROPIC_AUTH_TOKEN not set`, `BackendError: claude-agent-sdk not installed`,
`Fatal error in message reader`) is owned by `_preflight()`; see
`## Setup → Recovery` for the supervisor + install rerun loop. Manual SDK
fallback if frozen pip blocks `_ensure_python_sdks()`:
`python -m pip install 'claude-agent-sdk>=0.1.65' 'openai>=1.50' 'httpx>=0.27'`.
Transient SDK errors retry/resume up to the Coordinator emergency threshold.

### Model-gate errors (preflight #10)

Allowlist: `claude-opus-4-7` (preferred) → `claude-opus-4-6` (fallback). The
gate is intentional — opus-4-5 / haiku silently degraded prior runs.

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | Drop `--claude-model` / `$CLAUDE_MODEL`. Update `_CLAUDE_ALLOWED_MODELS` in `cli.py` only when a successor is blessed. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` (offline / smoke) or `--critic-codex-bare` (legacy
direct Codex). See `## Critic Backend Selection`.

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/critic-agent`, or `git -C "$REPO_ROOT" submodule update --init critic-agent`. |
| `runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `critic-agent/AGENTS.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `runtime.cli ... timed out after 30s` | KB stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory`. Reproducing in `inmemory` is a bug — that path must not block on I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `select_kernels` with unchanged trace/config: bug — reuse `last_select_kernels`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must contain explicit correctness evidence.
- `stop_reason=no_more_leverage`: stop and report; only resume if the user changes workload / search space / model / strategy.
- `stop_reason=time_exhausted`: resume same session (`--resume`); do not start fresh.

## Report Back To User

Report concise status:

- session id (from `manifest.json`) and log path
- `cumulative_gain` and `current_best`
- params accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
