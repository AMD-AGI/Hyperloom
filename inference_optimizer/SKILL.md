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
- Critic: proposal review; can be real Codex or `--critic-mock` when Codex credentials are unavailable.
- Robustness: mock robustness monitor in this branch.

State lives in one session directory. Production defaults to
`/hyperloom/inference_optimizer-sessions`, but portable launches should set
`INFERENCE_OPTIMIZER_SESSION_ROOT` to a writable run-local directory:

```bash
/hyperloom/inference_optimizer-sessions/<session_id>/
├── state.json
├── storage/coordinator.db
├── results/
└── kernel-agent-workspace/
```

Always prefer `state.json` and `coordinator.db` over guessing from terminal logs.

## Portable Environment Setup

Start from the repository root, but do not assume a fixed checkout path. Resolve
the root once and use variables for every path that follows:

```bash
export REPO_ROOT="$(pwd)"
export PYTHON="${PYTHON:-/opt/venv/bin/python}"
test -x "$PYTHON" || export PYTHON="$(command -v python3)"
export RUN_ROOT="$REPO_ROOT/optimizer_runs"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
mkdir -p "$RUN_ROOT" "$WORKSPACE_ROOT"
```

Load LLM credentials without printing secrets. The canonical credentials are
`OPENAI_BASE_URL` and `SAFE_API_KEY` for the user's LiteLLM-compatible endpoint.
If they are not already exported, source `$REPO_ROOT/.env`. Export compatibility
aliases for the Python optimizer, Claude/Codex OOB, GEAK, and any legacy code
that still reads Anthropic/OpenAI-style env names.

```bash
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set in env or .env}"
: "${SAFE_API_KEY:?SAFE_API_KEY must be set in env or .env}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-$SAFE_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$SAFE_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$SAFE_API_KEY}"
export OOB_API_KEY="${OOB_API_KEY:-$SAFE_API_KEY}"
export GEAK_API_KEY="${GEAK_API_KEY:-$SAFE_API_KEY}"
export LLM_API_KEY="${LLM_API_KEY:-$SAFE_API_KEY}"
export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-$SAFE_API_KEY}"

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$OPENAI_BASE_URL}"
export OOB_BASE_URL="${OOB_BASE_URL:-$OPENAI_BASE_URL}"
export GEAK_BASE_URL="${GEAK_BASE_URL:-$OPENAI_BASE_URL}"
export LLM_API_BASE="${LLM_API_BASE:-$OPENAI_BASE_URL}"

"$PYTHON" - <<'PY'
import os
required = ["OPENAI_BASE_URL", "SAFE_API_KEY"]
missing = [k for k in required if not os.environ.get(k)]
print("env_required_present=", not missing)
if missing:
    print("missing=", ",".join(missing))
print("compat_aliases_present=", all(bool(os.environ.get(k)) for k in [
    "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OOB_API_KEY", "GEAK_API_KEY"
]))
PY
```

Install or validate the optimizer package in the same Python environment that
will launch the long run:

```bash
"$PYTHON" -m pip install -e "${REPO_ROOT}[test]"
"$PYTHON" -m inference_optimizer.cli --help
```

Kernel-agent is a reusable downstream skill. Before launching
`inference_optimizer.cli`, read and follow `$REPO_ROOT/kernel-agent/SKILL.md`,
especially `Installation`, `TraceLens Requirements`, and `Backend Selection`.
This makes the same kernel-agent work both standalone and under
inference_optimizer. A user request to optimize a model is approval to install
the required kernel-agent runtime on a fresh node; do not stop for an extra
confirmation before running the installer. The installer may set up TraceLens,
OOB, Node/npm, Claude/Codex CLIs, local auth files, and the OOB auth-proxy.
Invoke it instead of duplicating backend setup logic:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
export PATH="/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

bash "$HYPERLOOM_KERNEL_AGENT_ROOT/scripts/install.sh" --with-oob
. "$HYPERLOOM_KERNEL_AGENT_ROOT/env.sh"
```

Do not collapse dependent exports into a single command when `set -u` is active.
Bash expands every right-hand side before assigning the left-hand sides, so
`export HYPERLOOM_KERNEL_AGENT_ROOT=... KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"`
can fail with `unbound variable` on a clean environment. Assign and export
dependent variables on separate lines as shown above.

Kernel backends submit Ray tasks with `num_gpus>=1`, so Ray must advertise all
visible GPUs. Do not start Ray with `--num-gpus=0`; that leaves kernel
optimization pending forever even when ROCm sees idle GPUs.

```bash
export RAY_NUM_GPUS="${RAY_NUM_GPUS:-$("$PYTHON" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}"
command -v ray >/dev/null
ray stop --force || true
ray start --head --disable-usage-stats \
  --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

Use the `ray` CLI from `PATH`, not `"$PYTHON" -m ray`; Ray does not expose a
`ray.__main__` module in this environment, so `python -m ray` fails before the
optimizer can launch.

Magpie must be importable by the launcher Python. If it is missing, install it
before launching:

```bash
"$PYTHON" - <<'PY' || {
import Magpie
print("Magpie OK")
PY
  git clone https://github.com/AMD-AGI/Magpie "$WORKSPACE_ROOT/Magpie"
  "$PYTHON" -m pip install -e "$WORKSPACE_ROOT/Magpie"
}
```

TraceLens is used by `select_kernels`. Prefer the internal mount because it
contains the standalone TraceLens skills required by `kernel-agent`; clone the
open-source repo only as a fallback when the internal mount is unavailable:

```bash
if [ -d "/wekafs/hyperloom/TraceLens-internal" ]; then
  export TRACELENS_ROOT="/wekafs/hyperloom/TraceLens-internal"
elif [ -d "$WORKSPACE_ROOT/TraceLens" ]; then
  export TRACELENS_ROOT="$WORKSPACE_ROOT/TraceLens"
else
  git clone https://github.com/AMD-AGI/TraceLens "$WORKSPACE_ROOT/TraceLens"
  export TRACELENS_ROOT="$WORKSPACE_ROOT/TraceLens"
fi
test -f "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md"
```

Kernel-agent is expected in the same repository. Resolve it relative to
`REPO_ROOT` and treat its skill as the source of truth for TraceLens, Ray, GEAK,
and OOB setup:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
test -d "$HYPERLOOM_KERNEL_AGENT_ROOT"
test -f "$HYPERLOOM_KERNEL_AGENT_ROOT/SKILL.md"
```

Use a repo-local session root by default so the skill works in sandboxes where
`/hyperloom` is absent:

```bash
export INFERENCE_OPTIMIZER_SESSION_ROOT="$RUN_ROOT/inference_optimizer-sessions"
mkdir -p "$INFERENCE_OPTIMIZER_SESSION_ROOT"
```

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
inference_optimizer/scripts/configs/baseline_qwen3_8b_sglang.yaml
inference_optimizer/scripts/configs/profile_qwen3_8b_sglang.yaml
```

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.
- `benchmark.envs.PATH`: must put `/opt/venv/bin` first.
- `benchmark.benchmark_script`: usually `sglang_mi300x.sh`.

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

Do not edit the default Qwen YAML for a new model. Materialize a run-specific
asset root and override only the benchmark configs for that run:

```bash
export MODEL_NAME="$(basename "$MODEL_PATH")"
export RUN_TS="$(date +%Y%m%d_%H%M%S)"
export ASSET_ROOT="$RUN_ROOT/assets_${MODEL_NAME}_${RUN_TS}"
export TP="${TP:-$( "$PYTHON" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$(seq -s, 0 $((TP - 1)))}"

"$PYTHON" - <<'PY'
from pathlib import Path
import os, yaml

repo = Path(os.environ["REPO_ROOT"])
asset = Path(os.environ["ASSET_ROOT"])
src = repo / "inference_optimizer"
(asset / "scripts" / "configs").mkdir(parents=True, exist_ok=True)

for name in ["actions", "kernel_opt", "orchestrator"]:
    target = asset / name
    if not target.exists():
        target.symlink_to(src / name, target_is_directory=True)
for helper in ["ab_torch_compile_magpie.py", "ab_torch_compile_kernels.py"]:
    target = asset / "scripts" / helper
    if not target.exists():
        target.symlink_to(src / "scripts" / helper)

def write_config(src_name, dst_name, profile_enabled, timeout_seconds):
    cfg = yaml.safe_load((src / "scripts" / "configs" / src_name).read_text())
    bench = cfg.setdefault("benchmark", {})
    bench["model"] = os.environ["MODEL_PATH"]
    bench["precision"] = os.environ.get("PRECISION", "bf16")
    bench["timeout_seconds"] = timeout_seconds
    envs = bench.setdefault("envs", {})
    envs.update({
        "TP": int(os.environ["TP"]),
        "CONC": int(os.environ.get("CONC", "8")),
        "ISL": int(os.environ.get("ISL", "256")),
        "OSL": int(os.environ.get("OSL", "256")),
        "RANDOM_RANGE_RATIO": os.environ.get("RANDOM_RANGE_RATIO", "1"),
        "MAX_MODEL_LEN": int(os.environ.get("MAX_MODEL_LEN", "8192")),
        "ROCR_VISIBLE_DEVICES": os.environ["ROCR_VISIBLE_DEVICES"],
        "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    bench["gpu_selection"] = {"auto": False}
    profiler = bench.setdefault("profiler", {})
    profiler.setdefault("torch_profiler", {})["enabled"] = bool(profile_enabled)
    profiler.setdefault("system_profiler", {})["enabled"] = False
    profiler.setdefault("tracelens", {})["enabled"] = False
    (asset / "scripts" / "configs" / dst_name).write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

write_config("baseline_qwen3_8b_sglang.yaml",
             "baseline_qwen3_8b_sglang.yaml",
             False, int(os.environ.get("BASELINE_TIMEOUT_SEC", "1800")))
write_config("profile_qwen3_8b_sglang.yaml",
             "profile_qwen3_8b_sglang.yaml",
             True, int(os.environ.get("PROFILE_TIMEOUT_SEC", "2400")))
print("asset_root=", asset)
PY

export INFERENCE_OPTIMIZER_ASSET_ROOT="$ASSET_ROOT"
```

## Launch A New Optimization

Use this for a fresh model/session:

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
. "$HYPERLOOM_KERNEL_AGENT_ROOT/env.sh"
export PATH="/opt/venv/bin:/usr/local/bin:$PATH"

export SESSION_NAME="${MODEL_NAME}-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$RUN_ROOT/run_${SESSION_NAME}.log"
export PID_FILE="$RUN_ROOT/run_${SESSION_NAME}.pid"
export TARGET_GAIN="${TARGET_GAIN:-10}"
export MAX_HOURS="${MAX_HOURS:-5}"

setsid nohup "$PYTHON" -m inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --session-name "$SESSION_NAME" \
  --target-gain "$TARGET_GAIN" \
  --max-hours "$MAX_HOURS" \
  --tick-interval-sec 30 \
  --kernel-claude \
  --critic-mock \
  > "$RUN_LOG" \
  2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Use real Critic only when Codex/OpenAI credentials are available:

```bash
# omit --critic-mock when ANTHROPIC_AUTH_TOKEN/OPENAI_API_KEY for Codex is valid
```

`setsid nohup ... &` is required for long runs. Cursor background shell alone is
not enough; it can die on SSH disconnect.

After launching, perform a short health check. The run is healthy only when the
optimizer process is alive, `state.json` exists, and the first benchmark process
or SGLang server has started.

```bash
"$PYTHON" - <<'PY'
import json, os, pathlib
pid_file = pathlib.Path(os.environ["PID_FILE"])
pid = pid_file.read_text().strip()
print("optimizer_running=", pathlib.Path("/proc", pid).exists())
session = pathlib.Path(os.environ["INFERENCE_OPTIMIZER_SESSION_ROOT"]) / os.environ["SESSION_NAME"]
state_path = session / "state.json"
print("state_exists=", state_path.exists())
if state_path.exists():
    state = json.loads(state_path.read_text())
    print("stop_reason=", state.get("stop_reason"))
    print("baseline_tput=", state.get("baseline_tput"))
    print("cumulative_gain=", state.get("cumulative_gain"))
patterns = ("Magpie", "sglang.launch_server")
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        text = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        continue
    if text and any(p in text for p in patterns):
        print(f"child_process {proc.name}: {text[:300]}")
PY
```

## Resume Existing Session

Use resume instead of starting over:

```bash
cd "$REPO_ROOT"
set -a; . "$REPO_ROOT/.env"; set +a

export SESSION_NAME=<session_id>
export RUN_LOG="$RUN_ROOT/resume_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"
export PID_FILE="$RUN_ROOT/run_${SESSION_NAME}.pid"
export TARGET_GAIN="${TARGET_GAIN:-10}"
export MAX_HOURS="${MAX_HOURS:-5}"

setsid nohup "$PYTHON" -m inference_optimizer.cli --verbose optimize \
  --resume "$SESSION_NAME" \
  --target-gain "$TARGET_GAIN" \
  --max-hours "$MAX_HOURS" \
  --tick-interval-sec 30 \
  --kernel-claude \
  --critic-mock \
  > "$RUN_LOG" \
  2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Resume preserves baseline, current best, params search state, event history, and
kernel-agent artifacts. The CLI clears stale `stop_reason` and `crash_count`
before retrying.

## Robustness monitor For Long Runs

For any run longer than 5 minutes, start a robustness monitor in its own `setsid nohup`
process. It must poll no more often than every 5 minutes, stop when the session
has a terminal `stop_reason`, and resume the same session if the optimizer exits
unexpectedly.

```bash
export ROBUSTNESS_MONITOR_SCRIPT="$RUN_ROOT/robustness_monitor_${SESSION_NAME}.sh"
export ROBUSTNESS_MONITOR_LOG="$RUN_ROOT/robustness_monitor_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"
export ROBUSTNESS_MONITOR_PID_FILE="$RUN_ROOT/robustness_monitor_${SESSION_NAME}.pid"

cat > "$ROBUSTNESS_MONITOR_SCRIPT" <<'SH'
#!/usr/bin/env bash
set -u
deadline="$("$PYTHON" - <<'PY'
import os, time
print(int(time.time() + (float(os.environ.get("MAX_HOURS", "5")) + 1.0) * 3600))
PY
)"
read_stop_reason() {
  "$PYTHON" - "$INFERENCE_OPTIMIZER_SESSION_ROOT/$SESSION_NAME/state.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("")
else:
    try:
        print(str((json.loads(p.read_text()).get("stop_reason") or "")).strip())
    except Exception:
        print("")
PY
}
while [ "$(date +%s)" -lt "$deadline" ]; do
  pid=""
  [ -f "$PID_FILE" ] && read -r pid < "$PID_FILE" || true
  stop_reason="$(read_stop_reason)"
  case "$stop_reason" in
    target_reached|no_more_leverage|time_exhausted|max_ticks)
      echo "[robustness monitor] terminal stop_reason=$stop_reason $(date -Is)"
      exit 0
      ;;
  esac
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    echo "[robustness monitor] alive pid=$pid stop_reason=${stop_reason:-none} $(date -Is)"
    sleep 300
    continue
  fi
  echo "[robustness monitor] optimizer stopped; resuming $SESSION_NAME $(date -Is)"
  resume_log="$RUN_ROOT/resume_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"
  set -a; . "$REPO_ROOT/.env"; set +a
  setsid nohup "$PYTHON" -m inference_optimizer.cli --verbose optimize \
    --resume "$SESSION_NAME" \
    --target-gain "${TARGET_GAIN:-10}" \
    --max-hours "${MAX_HOURS:-5}" \
    --tick-interval-sec 30 \
    --kernel-claude \
    --critic-mock \
    > "$resume_log" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 300
done
echo "[robustness monitor] deadline reached $(date -Is)"
SH

chmod +x "$ROBUSTNESS_MONITOR_SCRIPT"
setsid nohup bash "$ROBUSTNESS_MONITOR_SCRIPT" > "$ROBUSTNESS_MONITOR_LOG" 2>&1 < /dev/null &
echo $! > "$ROBUSTNESS_MONITOR_PID_FILE"
```

## Monitoring

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION="$INFERENCE_OPTIMIZER_SESSION_ROOT/$SESSION_NAME"
"$PYTHON" - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
print("stop_reason:", s.get("stop_reason"))
print("baseline_tput:", s.get("baseline_tput"))
print("cumulative_gain:", s.get("cumulative_gain"))
print("current_best:", s.get("current_best"))
print("params_search:", s.get("params_search", {}).get("last_round"))
print("last_kernel_opt:", s.get("last_kernel_opt"))
print("last_select_kernels:", s.get("last_select_kernels"))
print("last_sweep:", s.get("last_sweep"))
PY
```

Use SQLite for recent action counts:

```bash
"$PYTHON" - <<'PY'
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

- `ANTHROPIC_AUTH_TOKEN not set`: source `.env`.
- `Fatal error in message reader`: retry/resume; transient Claude CLI failures
are tolerated up to the Coordinator emergency threshold.
- `No accelerator`: ensure Magpie subprocess PATH includes `/opt/venv/bin` and
use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `select_kernels`: check `last_select_kernels`; if trace/config did
not change, this is a bug. Reuse cached candidates and run optimization.
- `correctness_passed=false`: do not integrate. Inspect the kernel-agent report;
the report must contain explicit correctness evidence.
- `no_more_leverage`: stop the run and report results; do not resume the same
  session unless the user changes workload, search space, model, or strategy.
- `time_exhausted`: resume the same session id; do not start from scratch.

## Report Back To User

Report concise status:

- session id and log path
- `cumulative_gain` and `current_best`
- params accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
