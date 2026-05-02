---
name: inference-optimizer
description: |
  Launches and monitors Hyperloom's multi-agent inference optimizer for LLM
  serving on AMD GPUs. Use when the user asks to optimize an inference model,
  run Magpie benchmarks/profiles, resume an inference-optimizer session, tune
  SGLang/vLLM serving parameters, run TraceLens/kernel-agent, or validate
  end-to-end throughput gains in a new inference environment.
globs:
  - "**/inference*optim*"
  - "**/inference-optimizer*"
---

# Inference Optimizer Skill

You are the launcher and monitor. The optimizer itself is the Python
`inference-optimizer` runtime under this repository. Do not manually optimize
inside chat unless debugging; launch the CLI, poll persisted state, and report
objective progress.

## What This Skill Runs

The CLI starts a Python Conductor that coordinates:

- Orchestration: decides next actions (`baseline`, `profile`, `backends`, `params`, `sweep`, Kernel requests, `report`).
- Kernel: responder path for `select_kernels`, `run_optimization`, `integrate`.
- Critic: proposal review; can be real Codex or `--critic-mock` when Codex credentials are unavailable.
- Robustness: mock watchdog in this branch.

State lives in one session directory:

```bash
/hyperloom/inference-optimizer-sessions/<session_id>/
├── state.json
├── storage/conductor.db
├── results/
└── kernel-agent-workspace/
```

Always prefer `state.json` and `conductor.db` over guessing from terminal logs.

## New Environment Setup

Run these from the Hyperloom repo root:

```bash
cd /wekafs/xiaofei/Hyperloom
```

Load credentials. The expected `.env` keys are `ANTHROPIC_BASE_URL` and
`ANTHROPIC_AUTH_TOKEN`.

```bash
set -a
. ./.env
set +a
```

Install the optimizer package into the active Python environment:

```bash
/opt/venv/bin/python -m pip install -e ".[test]"
/opt/venv/bin/python -m inference_optimizer.cli --help
```

Magpie must be importable by the same Python that launches the optimizer:

```bash
/opt/venv/bin/python -c "import Magpie; print('Magpie OK')"
```

If Magpie is missing, install it in the environment before launching:

```bash
git clone https://github.com/AMD-AGI/Magpie /workspace/Magpie
/opt/venv/bin/python -m pip install -e /workspace/Magpie
```

TraceLens is used by `select_kernels`. Prefer the open-source repo unless the
environment already provides an internal mount:

```bash
git clone https://github.com/AMD-AGI/TraceLens /workspace/TraceLens
export TRACELENS_ROOT=/workspace/TraceLens
```

Kernel-agent is expected next to `inference-optimizer`:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT=/wekafs/xiaofei/Hyperloom/kernel-agent
```

Optional but recommended session root in a new sandbox:

```bash
export INFERENCE_OPTIMIZER_SESSION_ROOT=/hyperloom/inference-optimizer-sessions
mkdir -p "$INFERENCE_OPTIMIZER_SESSION_ROOT"
```

## Benchmark Config

Default configs live here:

```bash
inference-optimizer/scripts/configs/baseline_qwen3_8b_sglang.yaml
inference-optimizer/scripts/configs/profile_qwen3_8b_sglang.yaml
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

## Launch A New Optimization

Use this for a fresh model/session:

```bash
cd /wekafs/xiaofei/Hyperloom
set -a; . ./.env; set +a

setsid nohup /opt/venv/bin/python -m inference_optimizer.cli --verbose optimize \
  --model /path/to/model \
  --target-gain 10 \
  --max-hours 5 \
  --tick-interval-sec 30 \
  --kernel-claude \
  --critic-mock \
  > /wekafs/xiaofei/Hyperloom/optimizer_runs/run_$(date +%Y%m%d_%H%M%S).log \
  2>&1 < /dev/null &
```

Use real Critic only when Codex/OpenAI credentials are available:

```bash
# omit --critic-mock when ANTHROPIC_AUTH_TOKEN/OPENAI_API_KEY for Codex is valid
```

`setsid nohup ... &` is required for long runs. Cursor background shell alone is
not enough; it can die on SSH disconnect.

## Resume Existing Session

Use resume instead of starting over:

```bash
cd /wekafs/xiaofei/Hyperloom
set -a; . ./.env; set +a

setsid nohup /opt/venv/bin/python -m inference_optimizer.cli --verbose optimize \
  --resume <session_id> \
  --target-gain 10 \
  --max-hours 5 \
  --tick-interval-sec 30 \
  --kernel-claude \
  --critic-mock \
  > /wekafs/xiaofei/Hyperloom/optimizer_runs/run_$(date +%Y%m%d_%H%M%S).log \
  2>&1 < /dev/null &
```

Resume preserves baseline, current best, params search state, event history, and
kernel-agent artifacts. The CLI clears stale `stop_reason` and `crash_count`
before retrying.

## Monitoring

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION=/hyperloom/inference-optimizer-sessions/<session_id>
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
print("stop_reason:", s.get("stop_reason"))
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
python3 - <<'PY'
import sqlite3, json
from collections import Counter
db = "/hyperloom/inference-optimizer-sessions/<session_id>/storage/conductor.db"
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

## Failure Handling

- `ANTHROPIC_AUTH_TOKEN not set`: source `.env`.
- `Fatal error in message reader`: retry/resume; transient Claude CLI failures
  are tolerated up to the Conductor emergency threshold.
- `No accelerator`: ensure Magpie subprocess PATH includes `/opt/venv/bin` and
  use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `select_kernels`: check `last_select_kernels`; if trace/config did
  not change, this is a bug. Reuse cached candidates and run optimization.
- `correctness_passed=false`: do not integrate. Inspect the kernel-agent report;
  the report must contain explicit correctness evidence.
- `time_exhausted`: resume the same session id; do not start from scratch.

## Report Back To User

Report concise status:

- session id and log path
- `cumulative_gain` and `current_best`
- params accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
