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
- Robustness: mock robustness monitor in this branch.

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

Always prefer `manifest.json` / `state.json` / `coordinator.db` over
guessing from terminal logs.

Session dir resolution order (`inference_optimizer/paths.py`):
1. `$INFERENCE_OPTIMIZER_SESSION_DIR` env → use as-is.
2. Default `/workspace/hyperloom`.

## Setup

This skill is **two commands**. Do NOT replicate setup steps inside chat —
both commands are idempotent, do auto-detection, and re-run safely.

### Credentials (env > .env, env always wins)

`SAFE_API_KEY` and `OPENAI_BASE_URL` are the only credentials this skill
needs. Resolution order, applied by both `install.sh` and the CLI's
`_preflight()`:

1. If both vars are already in env → use them, do not touch `.env`.
2. Otherwise, source `$REPO_ROOT/.env` for any **missing** keys only;
   keys already in env are protected and never overwritten by `.env`.
3. If neither env nor `.env` provides them → fail fast.

Caller's only responsibility: either `export REPO_ROOT=<hyperloom_repo_root>`
or invoke from the repo root (so `$(pwd)` is the fallback). Do NOT manually
`source .env` from chat — `install.sh` and the CLI both do it for you with
the correct env-wins semantics.

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

- `ray==2.44.1` + `click<8.3.0`
- TraceLens internal (perf-report CLI)
- GEAK CLI + `/workspace/hyperloom/runtime/geak-config/local.yaml`
- OOB CLI + claude/codex npm CLIs + `~/.claude/config.json` + `~/.codex/auth.json`
- **OOB auth-proxy on `127.0.0.1:4002`** (rewrites `x-api-key` → `Authorization: Bearer`
  for the AMD primus-safe gateway; without it Claude SDK returns 401)

**Tip — rsync install sources into the session dir before `install.sh`:** the
above pip-installs reference source trees on shared paths (e.g.
`TRACELENS_ROOT=/wekafs/hyperloom/TraceLens-internal`, `OOB_SRC`, `GEAK_REPO`,
`WORKSPACE_ROOT/Magpie`, `INFERENCEX_PATH`). If those paths may move,
disappear, or differ across nodes the session runs on, `rsync -a` them into a
session-local mirror (e.g. `$SESSION_DIR/vendor/{TraceLens-internal,OOB,GEAK,Magpie,InferenceX}/`)
and override the matching env vars to point at the mirror BEFORE invoking
`install.sh`. This binds the install to the session and avoids mid-run
failures like "TraceLens root not found" / "OOB source not found".

For the common single-node case where `$TRACELENS_ROOT` is simply on a
read-only WekaFS mount (the production default), no manual rsync is
needed: `kernel-agent/scripts/install.sh:ensure_tracelens` detects the
read-only source and `cp -r`s it to `${HYPERLOOM_ROOT}/TraceLens-internal`
(parallel to how `ensure_oob` mirrors `${HYPERLOOM_BUNDLE}/OOB` to
`${HYPERLOOM_ROOT}/OOB/oob_cli`), then `write_env_file` re-exports
`TRACELENS_ROOT` so any subsequent `inference_optimizer optimize` /
`select_kernels` subprocess inherits the writable mirror. The rsync-into-
session-dir tip above is still the right escape hatch for multi-node /
fast-moving-upstream scenarios.

`${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}` is
regenerated by `install.sh` and contains the proxy-rewritten URLs, auth aliases,
GEAK config path, and InferenceX path. Source it (don't try to derive these by
hand). Generated env/config state is written to the pod-local runtime directory,
not back into a shared WekaFS source checkout.

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
Magpie, InferenceX detection, Ray (with a live ray head started), TraceLens
CLI, GEAK + OOB CLI, the OOB auth-proxy on `:4002`, and the pod-local
`kernel-agent.env.sh`.
A user request to optimize a model is approval to run this on a fresh node;
do not stop for an extra confirmation:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"

# Prefer the launcher Python's bin dir, then standard system paths. Do NOT
# hardcode /opt/venv/bin: in bare images that path may not exist.
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
export PATH="${PYTHON_BIN_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"
"$PYTHON" -m inference_optimizer.cli --help
```

Do not collapse dependent exports into a single command when `set -u` is active.
Bash expands every right-hand side before assigning the left-hand sides, so
`export HYPERLOOM_KERNEL_AGENT_ROOT=... KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"`
can fail with `unbound variable` on a clean environment. Assign and export
dependent variables on separate lines as shown above.

The installer leaves a live Ray head running; `ray status` should succeed.
`select_kernels` and downstream kernel agents need this — they submit Ray
tasks with `num_gpus>=1`. Do not pass `--num-gpus=0` if you ever restart Ray
manually; that leaves kernel optimization pending forever even when ROCm
sees idle GPUs.

The CLI also runs `_preflight()` on every launch as a safety net for the
above install. It will:

1. Re-export auth aliases (`ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`, ...)
   from `SAFE_API_KEY`. `OOB_BASE_URL` / `GEAK_BASE_URL` / `LLM_API_BASE`
   inherit upstream from `OPENAI_BASE_URL` (those clients speak Bearer
   natively and do not need the proxy).
2. **Auto-installs missing Python SDKs** into the running interpreter
   (`sys.executable`): `claude-agent-sdk>=0.1.65`, `openai>=1.50`,
   `httpx>=0.27`. These are declared in `pyproject.toml` but a sandbox
   that only pulled the source tree without resolving deps lands here
   without them, causing `BackendError: claude-agent-sdk not installed`
   on the first reactor tick after baseline has already burned wall time.
3. **Bootstraps `auth_proxy.py` source** from `$OOB_SRC` (or
   `/wekafs/fully-local/OOB` / `/wekafs/fully-local/inference_optimization/OOB`)
   into `${HYPERLOOM_ROOT:-/opt/hyperloom}/OOB/oob_cli/` if missing. This
   is the file `ensure_auth_proxy.sh` actually executes; without it the
   supervisor warns, returns 1, and `:4002` stays dead.
4. Re-runs `ensure_auth_proxy.sh`, **rewrites `~/.claude/config.json`**
   so its `customApiUrl` points at `127.0.0.1:4002`, **and force-overrides
   `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` in the running process to the
   proxy URL** (any pre-set values from shell rc, `.env`, k8s secret, or
   container env are replaced; the override is logged on stdout). If the
   auth-proxy cannot be brought up after one retry, the original env values
   are restored so the two vars stay consistent and a WARNING is printed —
   Claude/Codex CLIs may then 401 against the gateway directly.
5. **ROCm env hygiene** (WARN-only): pops `HIP_VISIBLE_DEVICES` when
   `ROCR_VISIBLE_DEVICES` is also set (mixing the two makes
   `torch.cuda.is_available()` return false inside the Magpie subprocess);
   counts visible GPUs vs `$TP` via `rocm-smi --showid`; checks
   `/dev/shm` free space >= 16 GiB.
6. Auto-installs missing `ray` / `Magpie` / `InferenceX` if the pod was rebuilt.
7. Auto-detects `--gpu-type` if not given.
8. WARN-only presence check on `node` / `claude` / `codex` CLIs.
9. Emits a single canonical **`Preflight diagnostics:`** block with
   `asset_root`, `session_dir` (and the env var that resolved it),
   `magpie_python`, `INFERENCEX_PATH`, aiter jit cache state (WARM/COLD
   + `.so` count + path), cold/warm timeout caps, and the active proxy URL.
   Launchers should paste this block verbatim into status reports rather
   than grepping the source for env names.

After `_preflight()` returns, but BEFORE Coordinator boots, the CLI runs:

10. **Hard model gate** for `--claude-model`. The arg must equal
    `claude-opus-4-7` (preferred) or `claude-opus-4-6` (fallback). Anything
    else aborts with `sys.exit(2)` because orchestration drift on opus-4-5 /
    haiku silently degraded prior runs. Then the gateway catalog is probed
    (`GET <OPENAI_BASE_URL>/models` with Bearer, `verify=False`,
    3 retries with exponential backoff at 1s/3s/5s); if the chosen model
    is missing but `claude-opus-4-6` is present, the arg is rewritten and a
    WARNING is printed. If neither allowed model is in catalog OR the gateway
    is unreachable after all retries, refuse to start.
11. **Codex smoke-test** (WARN-only). `--codex-model` is checked against
    the same catalog when codex is actually used (`--critic-agent` /
    `--critic-codex-bare`, or `--kernel-codex` with kernel enabled).
12. **Critic-agent runtime probe** (only when `--critic-agent` is active —
    it's the default). Resolves `critic_agent_root`: env
    `CRITIC_AGENT_ROOT` > sibling `$REPO_ROOT/critic-agent/` > abort.
    Then `python -m runtime.cli --help` (5s timeout, `cwd=root`) must
    exit 0; if not, the optimizer aborts with rc=2 and a recovery hint
    pointing at `--critic-mock` / `--critic-codex-bare`. Default-sets
    `WORKSPACE_PATH=$REPO_ROOT`, `CRITIC_SESSION_MEMORY_DIR=$SESSION_DIR/critic-session-memory`,
    and `CRITIC_KB_CLIENT_MODE=inmemory`; `live` mode additionally
    requires `KB_BASE_URL` to be exported.

The install.sh-based bring-up is the canonical entry point; `_preflight()`
only catches drift mid-run. `kernel-agent/SKILL.md` (`Installation`,
`TraceLens Requirements`, `Backend Selection`) is the source of truth for
what the chained installer covers; read it if you need to debug the
kernel-agent layer.

You do NOT need to manually `pip install claude-agent-sdk`, copy
`auth_proxy.py` from a bundle path, `export ANTHROPIC_*`, manually start
ray, manually pip install Magpie, manually source `.env`, manually edit
`~/.claude/config.json`, or manually `curl /v1/models` to pick a model
name. Doing any of those by hand is the exact failure mode that produces
"Claude SDK exit code 1" / HTTP 401 / "claude-sonnet-4 not in catalog" /
"customApiUrl points to a local proxy that isn't running".

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
  (`$(dirname "$PYTHON")` — typically `/opt/venv/bin` in hyperloom containers,
  fall back to `$(dirname $(which python3))` on bare images).

### Workload-contract reuse (baseline → params/backends/sweep)

The `baseline` executor materializes its YAML once with the operator's
process env (`CONC` / `ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION`
/ `RUN_EVAL` / `ROCR_VISIBLE_DEVICES` plus adaptive `NUM_PROMPTS` /
`NUM_WARMUPS`) and writes it as `baseline_config.with_envs.yaml` next to
the baseline workspace. The Coordinator stashes that path on
`SharedState.baseline_config_path` and plumbs it forward as
`task.params["config_path"]` for every subsequent `params`, `backends`,
and `sweep` task.

This means downstream variants benchmark the **same workload baseline
ran**. Without this reuse the grid runner used to render variants from
the shipped YAML's smoke defaults (`TP=1` / `CONC=8` / `ISL=256` /
`OSL=256`) and produced ~10x lower throughput than baseline — the
"baseline 4367 tok/s vs variants ~360 tok/s" benchmark fairness bug.

`params` / `backends` / `sweep` also re-run materialization on top of
whatever `config_path` they receive, so the contract still holds for
direct test invocations or operators who delegate one of these actions
before `baseline`. Sweep variants' explicit `CONC` / `ISL` / `OSL`
overrides still win because `_grid_runner._build_variant_yaml` applies
per-variant `extra_envs` last.

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

### Required env (only when `--critic-agent` is active)

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

This project should first validate SGLang improvements, then add vLLM once the
SGLang path is stable. `params` writes candidates through `EXTRA_SGLANG_ARGS`
and `benchmark.envs`; do not hard-code a flag as default unless A/B results keep
it across the target workload.

The default SGLang search already covers cuda graph batch caps, continuous
decode steps, memory fraction, scheduling conservativeness, chunked prefill, and
max prefill tokens. It should also test the InferenceX-derived candidates:

- Cache/scheduler: `--disable-radix-cache`, `--max-running-requests 128/256`.
- Tokenization/streaming: `--tokenizer-worker-num 8/16`, `--stream-interval 30/50`.
- ROCm/TileLang envs: `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1`,
  `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`,
  `SGLANG_OPT_USE_TILELANG_INDEXER=true`.

Treat speculative decoding as model-specific until validated. For MTP/EAGLE,
use a custom grid with `SGLANG_ENABLE_SPEC_V2=1` and the appropriate
`--speculative-*` flags only when the model has the required draft path or MTP
support. Benchmark with chat-formatted prompts (`--dsv4` for DeepSeek-V4 style
runs) because raw random prompts can make acceptance-rate results misleading.

When judging a SGLang candidate, compare at least `1k/1k` and `8k/1k`, and
include both low and high concurrency if the model fits. Keep parameters only
when throughput improves without unacceptable TTFT/E2E or correctness regressions.
Coordinator-managed long runs test params incrementally with
`max_candidates_per_round=5` by default; direct runner calls may pass `0` to run
the full grid.

### Per-Run Asset Override (advanced)

To run a model with custom workload envs without editing the shipped YAMLs,
materialize a per-run asset root and pass it via `--asset-root`:

```bash
export ASSET_ROOT="$REPO_ROOT/optimizer_runs/assets_$(basename "$MODEL_PATH")_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ASSET_ROOT/scripts/configs"
for d in actions kernel_opt orchestrator; do
  ln -sfn "$REPO_ROOT/inference_optimizer/$d" "$ASSET_ROOT/$d"
done
ln -sfn "$REPO_ROOT/inference_optimizer/scripts/ab_torch_compile_magpie.py"  "$ASSET_ROOT/scripts/"
ln -sfn "$REPO_ROOT/inference_optimizer/scripts/ab_torch_compile_kernels.py" "$ASSET_ROOT/scripts/"
# Copy + edit baseline_*.yaml and profile_*.yaml under "$ASSET_ROOT/scripts/configs/" for
# this run's TP/CONC/ISL/OSL/MAX_MODEL_LEN/ROCR_VISIBLE_DEVICES. The
# `_workload_envs.materialize_config_with_envs` helper applies most of these
# from process env automatically; you only need a custom asset root for
# fields it does not touch (e.g. profiler.torch_profiler.enabled per yaml).
inference_optimizer optimize --asset-root "$ASSET_ROOT" --model "$MODEL_PATH" ...
```

For most cases the shipped YAMLs + `--model` / `--gpu-type` overrides are
enough; reach for `--asset-root` only when defaults don't fit the workload.

## Launch a New Optimization

Single command — assumes Step 1 (install) already ran in this pod.
There is no `--session-name`; the session lives at the canonical
`/workspace/hyperloom` (override with `$INFERENCE_OPTIMIZER_SESSION_DIR`):

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

`setsid nohup ... &` is required for runs > 5 min. Cursor's background
shell alone is not enough; it can die on SSH disconnect.

The Critic now defaults to `--critic-agent` (the real critic-agent
runtime — KB priors / session memory / `review_constraints`-gated
verdicts). Pass `--critic-mock` to fall back to the always-approve
adapter for offline / smoke runs, or `--critic-codex-bare` to run the
legacy direct-Codex path with no runtime layer (for debugging the LLM
in isolation). See [Critic Backend Selection](#critic-backend-selection).

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

`--resume` is a flag (no argument); it picks up whatever lives at the
canonical session_dir. The CLI refuses to start if `manifest.json` or
`state.json` is missing.

```bash
export RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$REPO_ROOT/optimizer_runs/resume_${RUN_TAG}.log"
export PID_FILE="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.pid"

setsid nohup inference_optimizer --verbose optimize \
  --resume \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Resume preserves baseline, current best, params search state, event history, and
kernel-agent artifacts. The CLI clears stale `stop_reason` and `crash_count`
before retrying.

## Robustness Monitor for Long Runs

For any run longer than 5 minutes, start a robustness monitor in its own
`setsid nohup` process. It must poll no more often than every 5 minutes,
stop when the session has a terminal `stop_reason`, and resume the
session if the optimizer exits unexpectedly.

```bash
export ROBUSTNESS_MONITOR_SCRIPT="$REPO_ROOT/optimizer_runs/robustness_monitor.sh"
export ROBUSTNESS_MONITOR_LOG="$REPO_ROOT/optimizer_runs/robustness_monitor_$(date +%Y%m%d_%H%M%S).log"
export ROBUSTNESS_MONITOR_PID_FILE="$REPO_ROOT/optimizer_runs/robustness_monitor.pid"

cat > "$ROBUSTNESS_MONITOR_SCRIPT" <<'SH'
#!/usr/bin/env bash
set -u
session_dir="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
deadline=$(( $(date +%s) + (${MAX_HOURS:-5} + 1) * 3600 ))
read_stop_reason() {
  python3 -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print((json.loads(p.read_text()).get('stop_reason') or '').strip() if p.exists() else '')" "$session_dir/state.json"
}
while [ "$(date +%s)" -lt "$deadline" ]; do
  pid=""
  [ -f "$PID_FILE" ] && read -r pid < "$PID_FILE" || true
  stop_reason="$(read_stop_reason)"
  case "$stop_reason" in
    target_reached|no_more_leverage|time_exhausted|max_ticks)
      echo "[robustness] terminal stop_reason=$stop_reason $(date -Is)"
      exit 0 ;;
  esac
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    echo "[robustness] alive pid=$pid stop_reason=${stop_reason:-none} $(date -Is)"
    sleep 300; continue
  fi
  echo "[robustness] optimizer stopped; resuming $(date -Is)"
  resume_log="$REPO_ROOT/optimizer_runs/resume_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup inference_optimizer --verbose optimize \
    --resume \
    --target-gain "${TARGET_GAIN:-10}" --max-hours "${MAX_HOURS:-5}" \
    --tick-interval-sec 30 --kernel-claude \
    > "$resume_log" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 300
done
echo "[robustness] deadline reached $(date -Is)"
SH

chmod +x "$ROBUSTNESS_MONITOR_SCRIPT"
setsid nohup bash "$ROBUSTNESS_MONITOR_SCRIPT" > "$ROBUSTNESS_MONITOR_LOG" 2>&1 < /dev/null &
echo $! > "$ROBUSTNESS_MONITOR_PID_FILE"
```

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

Recent action counts from SQLite:

```bash
python3 - <<'PY'
import json, os, pathlib, sqlite3
from collections import Counter
db = pathlib.Path(os.environ["SESSION"]) / "storage" / "coordinator.db"
con = sqlite3.connect(db)
c = Counter()
for fa, ta, topic, payload in con.execute(
    "select from_agent,to_agent,topic,payload from events order by seq desc limit 500"
):
    try:
        p = json.loads(payload)
    except Exception:
        continue
    if topic == "proposal":
        c["proposal:" + str(p.get("action_name"))] += 1
    if topic == "delegated_result":
        c["delegated:" + str(p.get("kind")) + ":" + str(p.get("state"))] += 1
    if topic == "request" and ta == "kernel":
        c["kernel_request:" + str(p.get("kind"))] += 1
    if topic == "response" and fa == "kernel":
        c["kernel_response:" + str(p.get("kind")) + ":" + str(p.get("status"))] += 1
print(dict(c))
PY
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

## Cache Topology

Why this matters: SGLang/vLLM on ROCm route hot fused kernels (RMSNorm,
attention, fused MoE, a8w8 blockscale GEMM, RoPE, ...) through `aiter`,
which JIT-compiles per-shape variants the first time it sees them and
caches the resulting `.so` on disk. The first `vllm serve` / `sglang
launch_server` against a fresh combination of (model, dtype, TP,
`max_model_len`, `max_num_seqs`, `gpu_memory_utilization`) can spend
30+ minutes inside `hipcc` on a 671B FP8 MoE class workload (e.g.
DeepSeek-R1-0528). Subsequent launches reuse the cached `.so` in
seconds. The optimizer needs to know where these caches live so it
can (a) interpret long-running cold starts correctly instead of
treating them as hangs, and (b) auto-bump the baseline timeout when
the cache is empty.

### aiter — JIT cache (primary cold-start cost)

```text
Source:     /sgl-workspace/aiter/aiter/
Git repo:   /sgl-workspace/aiter/

JIT cache:  /sgl-workspace/aiter/aiter/jit/build/
            (also: /usr/local/lib/python3.{10,12}/site-packages/aiter/jit/build/
                   /opt/venv/lib/python3.{10,12}/site-packages/aiter/jit/build/)
            Each kernel has its own build/<kernel_name>/build/<kernel_name>.so

Tuned configs:  aiter/configs/a8w8_blockscale_tuned_gemm.csv
                aiter/configs/tuned_fmoe.csv
RoPE source:    aiter/rotary_embedding.py
GEMM dispatch:  aiter/ops/gemm.py
MoE dispatch:   aiter/fused_moe.py

Clear (specific kernel): rm -rf /sgl-workspace/aiter/aiter/jit/build/<kernel>/
Clear (all):             rm -rf /sgl-workspace/aiter/aiter/jit/build/
```

### Triton cache

```text
Path:   ~/.triton/cache/    (resolves via $HOME, NOT $TRITON_CACHE_DIR
                             unless explicitly exported)
Clear: rm -rf ~/.triton/cache
```

### torch.compile / Inductor cache

```text
Path:   /tmp/torchinductor_<user>/    (default; override via
                                        $TORCHINDUCTOR_CACHE_DIR)
Clear: rm -rf /tmp/torchinductor_root
```

### sgl_kernel pre-compiled .so (not cold-start, build-time only)

```text
Location: /opt/venv/lib/python3.{10,12}/site-packages/sgl_kernel/
Compiled: common_ops.cpython-3{10,12}-*-linux-gnu.so  (built with image)
Source:   /sgl-workspace/sglang/sgl-kernel/
Build:    cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install
```

This one is informational. The optimizer never rebuilds `sgl_kernel`
during a baseline / params / sweep run; only `kernel_opt` / `integrate`
may touch it via the kernel-agent path.

## Cold-start Discipline

Cold-start triggers (any one of these makes the next baseline a cold
run):

- First launch of a (model, dtype, TP) combination on this pod.
- Any change to `--max-model-len`, `--max-num-seqs`,
  `--gpu-memory-utilization`, `--cuda-graph-max-bs`, or `--quantization`
  vs. the previous live server (changes the shape signature aiter
  hashes against).
- `--enable-torch-compile` toggled on or off.
- Container / pod rebuild that wiped `aiter/jit/build/`.
- Manual `rm -rf` of any of the cache trees above.
- aiter source-level patch (kernel_opt / integrate just landed on a
  kernel under `/sgl-workspace/aiter/aiter/`).

`BaselineExecutor` auto-detects cold start by counting `.so` files
under `aiter/jit/build/`. Threshold: **`< 20` files = COLD**, otherwise
WARM. The first existing path under the probe list (see
`baseline.py:AITER_JIT_PROBE_PATHS`) wins. Override paths by exporting
`INFERENCE_OPTIMIZER_AITER_JIT_DIR=/abs/path/to/jit/build`.

The same probe runs once at boot inside `_emit_preflight_diagnostics()`
so the resolved cache state appears in the canonical preflight block:

```
Preflight diagnostics:
  ...
  aiter jit cache     = 98 .so / 887 MB (WARM) at /sgl-workspace/aiter/aiter/jit/build
  cold_start_timeout  = 3600s
  warm_timeout        = 1500s
  proxy URLs          = http://127.0.0.1:4002/api/v1/llm-proxy (auth-proxy alive)
```

Launchers should read this block instead of grepping `cli.py` /
`baseline.py` for env var names. The `cold_start_timeout` line reflects
any active `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC` override.

Timeout selection (one-shot, per baseline `__call__`):

| Condition | Resulting `subprocess.run(timeout=...)` |
| --- | --- |
| `task.params['timeout_sec']` set | task value (always wins) |
| Cache probe `found` AND `kernel_count < 20` | `BASELINE_COLD_START_TIMEOUT_SEC` (default 3600s; override via `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=N`) |
| Cache probe `found` AND `kernel_count >= 20` | `BASELINE_DEFAULT_TIMEOUT_SEC` (1500s) |
| Cache probe `not_found` / `error` | `BASELINE_DEFAULT_TIMEOUT_SEC` (1500s) + WARN log |

Every baseline launch logs exactly one of these markers — grep
`optimizer_runs/run_*.log` to verify which path fired:

- `baseline_executor: COLD_START detected — aiter jit/build/ at <path> has N .so (< 20 threshold), M MB. Bumping timeout 1500s -> 3600s. ...`
- `baseline_executor: WARM start — aiter jit/build/ at <path> has N .so, M MB. Using default timeout=1500s.`
- `baseline_executor: timeout=Ns (explicit task param)`
- `baseline_executor: aiter jit cache not located (probe_status=...). Using default timeout=1500s. Cold-start auto-bump disabled for this run.`

If you see repeated COLD_START markers across baseline retries, the
JIT was likely killed mid-`hipcc` by the previous timeout (leaving
`.so` half-written) — extend the cold cap further via
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400` rather than relaunching.
`ProfileExecutor` inherits the same logic (it subclasses
`BaselineExecutor`), so `profile` actions get the same auto-bump.

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

- `ERROR: --claude-model=... is not allowed`: the static gate rejected
  the chosen model. Orchestration must use `claude-opus-4-7` (preferred)
  or `claude-opus-4-6` (fallback). Drop or change `--claude-model` /
  `$CLAUDE_MODEL` and re-run. This is **intentional** — opus-4-5 / haiku
  silently degraded prior runs and the operator pinned the allowlist.
- `ERROR: gateway catalog unreachable after retries`: the
  `GET <base_url>/models` probe failed all 4 attempts (initial + 3
  exponential-backoff retries at 1s/3s/5s). Reproduce manually with the
  command in `terminals/6.txt`:
  ```bash
  curl -k -H "Authorization: Bearer $SAFE_API_KEY" \
       "$OPENAI_BASE_URL/models" | jq '.data[].id' | sort
  ```
  If the gateway answers, the proxy / SSL path is the problem; if it
  doesn't, the gateway itself is down. We deliberately fail-fast here
  rather than launch a baseline that will 401 ~5 minutes in.
- `ERROR: neither claude-opus-4-7 nor claude-opus-4-6 present in gateway catalog`:
  catalog reachable but neither allowed model is listed. Either the
  gateway dropped them (escalate to operator) or this is a wrong
  endpoint. Don't bypass the gate — change the catalog or update the
  allowlist constant `_CLAUDE_ALLOWED_MODELS` in `cli.py` if a successor
  model has been blessed.
- `WARNING — claude-opus-4-7 not in gateway catalog; falling back to claude-opus-4-6`:
  expected when 4-7 is rotated out of the gateway. The run continues on
  4-6; performance characteristics are nearly identical.
- `Claude SDK exit code 1` / `Primus.00009 token not present`: auth-proxy
  is dead. Run `bash $REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh`
  and retry the CLI. Do NOT manually rewrite `~/.claude/config.json` —
  `_preflight()` owns it. If the supervisor warns that `auth_proxy.py`
  is missing, set `OOB_SRC` to a directory that contains it (or land
  one of `/wekafs/fully-local/OOB`, `/wekafs/fully-local/inference_optimization/OOB`)
  so `_ensure_oob_proxy_source()` can bootstrap it next run.
- `ERROR: --critic-agent selected but critic-agent runtime not found`:
  resolution order is `$CRITIC_AGENT_ROOT` env > sibling
  `$REPO_ROOT/critic-agent/`. Fix one of:
  ```bash
  export CRITIC_AGENT_ROOT=/path/to/critic-agent
  # or:
  test -f "$REPO_ROOT/critic-agent/runtime/cli.py" || \
    git -C "$REPO_ROOT" submodule update --init critic-agent
  ```
  Bypass with `--critic-mock` (offline / smoke) or `--critic-codex-bare`
  (legacy direct Codex path) if a fix isn't available immediately.
- Every critic verdict comes back as `('needs_review',
  'critic_unavailable')` with `kb_skipped=missing_critical_context` and
  `required_context=['model', 'framework', ...]` (see the
  `critic_agent_backend turn=...` log line and
  `BackendTurnResult.metadata['required_context']`): in older versions
  this was a real bug — `CriticAgentBackend` shipped `request.context={}`
  unconditionally, so the runtime's `CRITICAL_CONTEXT_KEYS=("model","framework")`
  gate fired on every proposal and the orchestrator could never get an
  `approve`. As of this fix the backend reads `manifest.json` once in
  `__post_init__` (`_load_static_context_from_manifest()`) and injects
  model / framework / gpu_type / model_path / tp / workload / precision
  into every `prepare-review` request. If you see this symptom again,
  check `manifest.json` for non-empty `model_name` and `framework`
  (`build_manifest()` is the writer) and grep `logs/cli.log` for
  `critic_agent_backend static_context source=... keys=[...]` — the keys
  list shows what was actually loaded. To bypass while debugging, restart
  with `--critic-mock` (always-approve, no review safety) or pass an
  explicit `static_context=` if you're invoking the backend programmatically.
- `BackendError: critic-agent runtime.cli prepare-review/commit-review
  exited rc=2`: the critic-agent runtime aborted with an adapter bug —
  per `critic-agent/AGENTS.md` §Exit codes, rc=2 means schema or
  validation failure inside the runtime. Inspect
  `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`
  for the offending payload, then either fix the upstream issue or
  retry with `--critic-mock` so the run can keep moving while the
  runtime bug is debugged.
- `BackendError: critic-agent runtime.cli ... timed out after 30s`:
  prepare-review / commit-review usually return in <1s. A timeout
  indicates a stuck KB call or a heavy KB write fan-out. If
  `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory` for the rest of the
  run (no kill switch needed; the next process inherits the lower mode).
  If the timeout reproduces in `inmemory` mode, capture the runtime
  logs and file a bug — that path should not block on I/O.
- `BackendError: claude-agent-sdk not installed`: should not happen
  after `_ensure_python_sdks()` lands, but if it does (frozen pip, no
  network) install manually:
  `python -m pip install claude-agent-sdk>=0.1.65 openai>=1.50 httpx>=0.27`.
- `ANTHROPIC_AUTH_TOKEN not set`: re-source `${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}`.
- `Fatal error in message reader`: retry/resume; transient Claude CLI failures
  are tolerated up to the Coordinator emergency threshold.
- `No accelerator`: ensure Magpie subprocess `PATH` leads with the launcher
  Python's bin dir (`$(dirname "$PYTHON")`, or set `MAGPIE_PYTHON` to the
  correct interpreter) and use `ROCR_VISIBLE_DEVICES`, not
  `HIP_VISIBLE_DEVICES`.
- Repeated `select_kernels`: check `last_select_kernels`; if trace/config did
  not change, this is a bug. Reuse cached candidates and run optimization.
- `correctness_passed=false`: do not integrate. Inspect the kernel-agent report;
  the report must contain explicit correctness evidence.
- `no_more_leverage`: stop the run and report results; do not resume the same
  session unless the user changes workload, search space, model, or strategy.
- `time_exhausted`: resume the same session id; do not start from scratch.

## Report Back To User

Report concise status:

- session id (from `manifest.json`) and log path
- `cumulative_gain` and `current_best`
- params accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why

## Session Layout (cheat sheet)

The CLI flattens everything into a single fixed directory at
`/workspace/hyperloom` (override: `$INFERENCE_OPTIMIZER_SESSION_DIR`).
Python owns every mkdir; agents reference paths via the injected
`SESSION_DIR` token. PolicyGate refuses path-like fields that escape
the session_dir (or, for `source_file`, the framework allowlist
`/sgl-workspace/{aiter,sglang,vllm}/`).

```text
$SESSION_DIR/                            # /workspace/hyperloom by default
├── manifest.json                        # written first; v1 schema; resume tag
├── state.json                           # SharedState — Coordinator-owned
├── storage/coordinator.db               # SQLite WAL (events/leases/cursors/tasks)
├── agents/<role>/                       # orchestration / kernel / critic / robustness
│   ├── inbox.jsonl  outbox.jsonl
│   ├── persona.md
│   └── system_prompt.snapshot.md        # snapshot of the prompt at boot
├── personas/  checkpoints/  findings/  kb/
├── runs/                                # data-plane (executor outputs)
│   ├── baseline/<task_id>/              # Magpie workspace + materialized YAML
│   ├── profile/<task_id>/               # baseline + torch_trace/
│   ├── backends/<task_id>/{variant_NN_*/, result.json}
│   ├── params/<task_id>/{variant_NN_*/, combo/, result.json}
│   ├── sweep/<task_id>/
│   ├── integrate/<task_id>/             # patch → re-baseline workspace
│   └── kernel_opt/<kernel_id>/<task_id>/
├── kernel-agent-workspace/<kernel_id>/  # GEAK / OOB cross-task artefacts
├── patches/<kernel_id>/                 # KEEP-promoted patches + backup/
├── reports/                             # `report` action output (final.{md,json})
└── logs/                                # cli.log / coordinator.log / <role>.log
```

Resolution helpers (use these — don't string-concat):

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
