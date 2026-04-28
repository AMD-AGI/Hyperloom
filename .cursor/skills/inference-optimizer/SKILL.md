# Inference Optimizer (v0.8 — Python ↔ shell ActionExecutor bridge)

> **STATUS**: ``MockBackend`` (dry-run), ``ClaudeBackend`` (real Claude
> via ``claude-agent-sdk``), and ``CodexBackend`` (real Codex / GPT via
> the ``openai`` SDK with ``validated_json_output``) all work. The
> Conductor spawns one reactor per role (executor + critic + sage +
> watchdog) according to ``ExecutionMode``, every intent flows through
> ``PolicyGate``, and ``emit_intent`` is wired as a real MCP tool inside
> ``ClaudeBackend`` (JSON-in-text fallback). Codex roles post a single
> ``validated_json_output`` envelope per turn (parser accepts ``json`` /
> ``validated_json_output`` fences). The ``ActionRegistry`` loads 22
> actions from ``actions/_meta/*.yaml``; the ``SubAgentRunner`` first
> tries the matching ``ActionExecutor`` (Python wrapper around the
> bundled ``scripts/*.sh``), falling back to the LLM path when env vars
> are missing. Five executors live today: ``baseline``, ``bench_runner``,
> ``profile``, ``param_sweep_run``, ``kernel_opt`` — they shell out to
> ``run_baseline.sh`` / ``run_profile.sh`` / ``run_sweep.sh`` /
> ``geak_ray_submit.py`` / ``oob_ray_submit.py``. The Conductor
> auto-derives ``cumulative_gain`` from any ``baseline_tput`` /
> ``current_tput`` update. See ``IMPLEMENTATION-CHECKLIST.md``.
>
> **Auto-bootstrap**: When ``--backend claude`` is selected, the CLI
> probes for Node.js (>=18) and the ``claude`` CLI binary on PATH. With
> ``--auto-install`` (or ``INFERENCE_OPTIMIZER_AUTO_INSTALL=1``) it
> downloads a portable Node + ``npm install -g @anthropic-ai/claude-code``
> into ``~/.cache/inference-optimizer/`` — **no sudo, no system pollution**.

Adaptive, autonomous LLM-inference performance optimization on AMD MI355X
GPU sandboxes. Single entrypoint that picks one of three execution modes
based on `MAX_HOURS`:

| `MAX_HOURS` | Mode                  | Roles active                      | Token budget |
| ----------: | --------------------- | --------------------------------- | -----------: |
| `< 2`       | `quick_param_sweep`   | Executor only                     | ~0.5M        |
| `2..6`      | `guided_kernel_opt`   | Executor + ephemeral RCA-Critic   | ~3M          |
| `> 6`       | `marathon_multi_agent`| Executor + Critic + Sage + Watchdog | ~11.5M    |

## Required env

| var                 | description                                              |
| ------------------- | -------------------------------------------------------- |
| `MODEL_PATH`        | path to model weights (or HF repo)                       |
| `MAX_HOURS`         | wall-clock budget (float)                                |

## Optional objective (pick at most one)

| var                       | mode mapping                                             |
| ------------------------- | -------------------------------------------------------- |
| `TARGET_GAIN_PCT`         | TargetGainObjective (% gain over baseline)               |
| `TARGET_TPUT_PER_GPU`     | TargetTputObjective (absolute tok/s/GPU)                 |
| `TARGET_DIR`              | TargetBaselineObjective (compare against another run)    |
| (none of the above)       | TimeOnlyObjective (maximize gain in time budget)         |

## Optional infra envs

| var                                    | description                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `INFERENCE_OPTIMIZER_SESSION_ROOT`     | dev override; default `/hyperloom/inference-optimizer-sessions` |
| `INFERENCE_OPTIMIZER_DB_PATH`          | local-disk path for SQLite (path A backup deploy)        |
| `KERNEL_OPT_BACKENDS`                  | default `geak,codex`                                     |
| `KERNEL_OPT_IMAGE`                     | container image for kernel-opt sub-agents                |
| `INFERENCE_OPTIMIZER_VENV_BIN`         | default `/opt/venv/bin`                                  |
| `INFERENCE_OPTIMIZER_AUTO_INSTALL`     | `1`/`true`/`yes` → bootstrap installs Node + claude CLI  |
| `ANTHROPIC_API_KEY`                    | required for `--backend claude` (or use Bedrock/Vertex)  |
| `ANTHROPIC_AUTH_TOKEN`                 | Bearer-auth alternative to `ANTHROPIC_API_KEY`; required when proxy expects `Authorization: Bearer ...` instead of `x-api-key` |
| `ANTHROPIC_BASE_URL`                   | OpenAI-compat / Anthropic-compat proxy root (e.g. corp gateway). Claude SDK appends `/v1/messages`. |
| `OPENAI_API_KEY`                       | required for `--backend codex` (or use Azure / proxy)    |
| `OPENAI_BASE_URL`                      | OpenAI-compatible proxy root (Azure / Foundry / corp). The SDK appends `/v1/chat/completions`. CLI flag `--codex-base-url` overrides. |
| `OPENAI_MODEL`                         | default model id for `--backend codex` (overridden by `--codex-model`) |
| `INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL`| `0`/`false`/`no`/`off` → skip TLS verification on the OpenAI proxy (self-signed corp certs). Default verify on. |
| `NODE_TLS_REJECT_UNAUTHORIZED`         | claude CLI / Node child-proc equivalent of the above; set `0` to skip TLS verify when `ANTHROPIC_BASE_URL` points at a self-signed proxy |

---

## Launch (Cursor agent — read this first)

When the user invokes this skill, do exactly this:

1. **Read** the user's required envs (`MODEL_PATH`, `MAX_HOURS`) and any
   single optional `TARGET_*`.
2. **Pick a session root**:
   - production: leave `INFERENCE_OPTIMIZER_SESSION_ROOT` unset
     (defaults to `/hyperloom/inference-optimizer-sessions`).
   - dev/Windows/no NFS: set `INFERENCE_OPTIMIZER_SESSION_ROOT` to a
     local writable path (e.g. `$TEMP/io-sessions`).
3. **Install Python deps** (once per environment):

```bash
pip install -r src/inference_optimizer/requirements.txt
```

4. **Run the CLI** from `src/`:

```bash
cd src
python -m inference_optimizer \
    --model "$MODEL_PATH" \
    --max-hours "$MAX_HOURS" \
    [--target-gain-pct "$TARGET_GAIN_PCT" |
     --target-tput-per-gpu "$TARGET_TPUT_PER_GPU" |
     --target-dir "$TARGET_DIR"] \
    [--backend mock|claude|codex] \
    [--codex-model gpt-5.4] \
    [--codex-base-url https://api.example.com/v1] \
    [--auto-install] \
    [--session-id <id-to-resume>]
```

   For PowerShell:

```powershell
cd src
python -m inference_optimizer `
    --model $env:MODEL_PATH `
    --max-hours $env:MAX_HOURS `
    --backend claude `
    --auto-install
```

5. **Print the session_dir** the CLI emits on stderr. The user needs it
   to inspect state and the SQLite DB.

## Run on local GPU (minimum)

For a real run that boots sglang/vllm and produces an actual
`cumulative_gain` number, the CLI exposes the GPU/framework knobs that
the bundled `scripts/*.sh` shell scripts need. Auto-probe fills in the
defaults the operator didn't specify (GPU count via `rocm-smi` /
`amd-smi` / `nvidia-smi`, framework via Python import).

### Pre-flight requirements

| Requirement | Check |
|---|---|
| sglang or vllm in current Python env | `python -c "import sglang"` or `import vllm` |
| ROCm or NVIDIA SMI on PATH | `rocm-smi -h` or `nvidia-smi -h` |
| InferenceX checkout | `ls $INFERENCEX_PATH/benchmarks/benchmark_lib.sh` |
| OPENAI_API_KEY (codex) or ANTHROPIC_API_KEY (claude) | `env \| grep -E 'OPENAI\|ANTHROPIC'_API_KEY` |

### Run

```bash
export INFERENCEX_PATH=/opt/InferenceX
export OPENAI_API_KEY=sk-...                   # or ANTHROPIC_API_KEY
export INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-local

cd src
python -m inference_optimizer \
    --model Qwen/Qwen3-8B \
    --max-hours 1 \
    --target-gain-pct 10 \
    --inferencex-path "$INFERENCEX_PATH" \
    --backend codex --codex-model gpt-5.4 \
    --reactor-tick-s 5.0 --clock-tick-s 10.0 \
    --log-level INFO
```

Optional pinned overrides (auto-probe fills these otherwise):

```bash
    --tp 4 --conc 32 --isl 1024 --osl 256 \
    --port 8888 --framework sglang
```

The CLI banner now prints what it picked up:

```
[inference-optimizer] starting
  ...
  actions     : 22 loaded
  gpu         : count=4 type=gfx950
  framework   : sglang (version=0.5.10rc0...)
  config      : tp=4 conc=32 isl=1024 osl=256 port=8888
  inferencex  : /opt/InferenceX
```

### What happens after launch

1. The executor reactor wakes up, sees the `Available actions`
   catalogue + a `First action hint (live)` block (because
   `baseline_tput=0`).
2. It emits `delegate(action_name="baseline")`.
3. The dispatcher loop picks the queued task; `BaselineExecutor` shells
   out to `scripts/run_baseline.sh` which launches sglang, runs the
   benchmark, and writes `results/<task_id>/baseline_*.json`.
4. The executor parses the JSON, emits
   `update_state(baseline_tput=X)`, the Conductor records it, and
   `cumulative_gain` is auto-derived.
5. Subsequent rounds (`bench_runner`, `param_sweep_run`, optionally
   `kernel_opt`) push `current_tput` up; once `cumulative_gain ≥
   target_gain_pct`, the early-stop signal `target_reached` fires and
   the run ends gracefully.

### Monitor

```bash
bash src/inference_optimizer/scripts/monitor.sh --watch 5
# or:
python -m inference_optimizer.scripts.monitor --watch 5 --per-agent
```

### v0.7 backends

| backend         | what it does                                                                                    | status |
| --------------- | ----------------------------------------------------------------------------------------------- | ------ |
| `mock`          | scripted heartbeats, no LLM calls, runs locally                                                 | OK     |
| `claude`        | real Claude via `claude-agent-sdk`; in-process MCP server exposes `emit_intent` as a real tool, JSON-in-text envelope is the automatic fallback | OK |
| `codex`         | real Codex/GPT via the `openai` SDK; no-tools roles (Critic/Sage) post one `validated_json_output` envelope per turn, with a 1-shot repair pass on parse failure | OK |

All backends route through the same `Conductor`, but the Conductor now
spawns one reactor per role returned by `roles_for_mode(execution_mode)`,
and **every intent** is validated through `PolicyGate` before any
side-effect runs. Quick mode runs just the executor; guided mode adds an
ephemeral RCA-Critic; marathon mode adds Sage + Watchdog.

A `--backend claude` run today exercises:

- the multi-role LLM loop with role-specific system prompts and intent
  allow-lists,
- the in-process MCP server that publishes ``emit_intent`` (no separate
  process, no extra ports),
- the message bus, the SQLite SoT (events / tasks / cursors / leases),
  the clock, and graceful stopping,
- the ``SubAgentRunner`` skeleton that drains queued ``delegate`` tasks
  using the ``ActionRegistry`` (it currently reuses the Conductor's
  backend; real OOB sub-agent processes land in Phase 7).

It does **not yet** trigger real benchmarks or kernel-opt rebuilds —
those are Phase 8.

### Bootstrap flow when `--backend claude`

The CLI runs `ensure_claude_cli(auto_install=...)` before constructing
the backend. Decision matrix:

| node>=18? | `claude` on PATH? | `--auto-install` | result                                        |
| :-------: | :---------------: | :--------------: | --------------------------------------------- |
| yes       | yes               | any              | proceed (probe-only)                          |
| yes       | no                | no               | exit 2 with `npm install -g @anthropic-ai/claude-code` instructions |
| yes       | no                | **yes**          | run `npm install -g --prefix=~/.cache/inference-optimizer/npm-prefix` |
| no/old    | no                | no               | exit 2 with copy-pasteable Node + claude install snippets |
| no/old    | no                | **yes**          | download portable Node from nodejs.org → install claude → prepend PATH |

`~/.cache/inference-optimizer/` ends up with:

```
node-v20.18.0/                  ← portable Node distribution
  bin/node, bin/npm             ← (Linux/Mac); `node.exe`/`npm.cmd` on Windows
npm-prefix/
  bin/claude                    ← @anthropic-ai/claude-code entry point
  lib/node_modules/...
```

The CLI mutates `os.environ['PATH']` so the SDK's child processes find
the new binaries. Nothing leaks outside the cache dir; `rm -rf
~/.cache/inference-optimizer` undoes everything.

### Quick local dry-run (Windows PowerShell)

```powershell
cd src
$env:INFERENCE_OPTIMIZER_SESSION_ROOT = "$env:TEMP\io-sessions"
python -m inference_optimizer --model fake/model --max-hours 0.001 `
    --reactor-tick-s 0.3 --clock-tick-s 0.5 --log-level INFO
```

`max-hours 0.001` ≈ 3.6 s of wall time. The run will end with
`reason=time_exhausted`.

### `--backend codex` (OpenAI-compatible)

`CodexBackend` talks to anything OpenAI-compatible — direct
`api.openai.com`, Azure OpenAI, AMD primus-safe LLM proxy, Foundry, etc.
No bootstrap step is needed (it's pure-pip via `openai>=1.50`).

Pick the endpoint via env or CLI:

```bash
export OPENAI_API_KEY=ak-...
export OPENAI_BASE_URL=https://your.proxy.example.com/api/v1/llm-proxy/v1
# Skip TLS verify when the proxy uses a self-signed cert:
export INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL=0

python -m inference_optimizer \
    --model /hyperloom/models/Qwen3-30B-A3B \
    --max-hours 0.05 \
    --backend codex \
    --codex-model gpt-5.4
```

For an Anthropic-compat proxy used with `--backend claude` that needs
`Authorization: Bearer` (instead of `x-api-key`):

```bash
export ANTHROPIC_AUTH_TOKEN=ak-...                                        # Bearer
export ANTHROPIC_API_KEY=$ANTHROPIC_AUTH_TOKEN                            # silence ClaudeBackend warn
export ANTHROPIC_BASE_URL=https://your.proxy.example.com/api/v1/llm-proxy
export NODE_TLS_REJECT_UNAUTHORIZED=0                                     # claude CLI runs under Node
python -m inference_optimizer --model X --max-hours 0.05 \
    --backend claude --claude-model claude-opus-4-7
```

The Codex endpoint is auto-derived: `${OPENAI_BASE_URL}/chat/completions`
(the SDK adds the `/v1` only if missing). Don't double-append `/v1`.

---

## Monitor (during a run)

Everything observable lives under `<session_dir>/`:

```
<session_dir>/
  state.json                 ← latest SharedState snapshot (refreshed every clock tick)
  storage/conductor.db       ← SQLite SoT (events, tasks, cursors, leases)
  personas/                  ← agent persona notes (Marathon)
  checkpoints/               ← (later) named state snapshots
  kb/                        ← Knowledge Base (cross-run lessons)
  results/                   ← benchmark / kernel-opt artefacts
  findings/                  ← RCA reports
```

### Quick health check

```bash
python scripts/inspect_session.py [<session_dir>]
```

Picks the newest session under
`$INFERENCE_OPTIMIZER_SESSION_ROOT` if no dir is given. Prints event
counts, cursor positions, last 5 events.

### Manual SQL probes

```bash
sqlite3 <session_dir>/storage/conductor.db <<'SQL'
SELECT topic, COUNT(*) FROM events GROUP BY topic ORDER BY 2 DESC;
SELECT * FROM cursors;
SELECT * FROM tasks ORDER BY started_at DESC LIMIT 5;
SELECT * FROM leases;
SQL
```

### Watch the snapshot file

```bash
# Linux/macOS
watch -n 2 cat <session_dir>/state.json

# PowerShell
while ($true) { Clear-Host; Get-Content "<session_dir>\state.json"; Start-Sleep 2 }
```

---

## What you should expect to see in Cursor

When the agent invokes the CLI, **stderr** will print a banner like:

```
[inference-optimizer] starting
  session_dir : .../<session_id>
  db          : .../<session_id>/storage/conductor.db
  ...
```

Then logging from the conductor:

```
INFO ... conductor: bootstrapped session=<id> mode=quick_param_sweep max_minutes=0.10
INFO ... conductor: stopped reason=time_exhausted elapsed=0.07m
```

And a final summary:

```
[inference-optimizer] stopped
  reason         : time_exhausted
  elapsed_min    : 0.07
  cumulative_gain: 0.00%
  session_dir    : .../<session_id>
```

The `cumulative_gain` will stay at 0% until real benchmarking is wired
in (Phase 8 — `benchmark`, `objective` updates).

---

## Trigger examples

```text
@inference-optimizer
MODEL_PATH=/data/gpt-oss-20b
MAX_HOURS=1
TARGET_GAIN_PCT=10
```

```text
@inference-optimizer
MODEL_PATH=/data/deepseek-v3
MAX_HOURS=8
TARGET_TPUT_PER_GPU=6000
```

## Entry-point file

`src/inference_optimizer/cli.py::main()` — argparse + asyncio runner.

`src/inference_optimizer/orchestrator/conductor.py::Conductor.run()` —
the async main loop. The Conductor wires every subsystem, replays SQLite
state on resume, and orchestrates reactors.

## Skill asset layout (all under `.cursor/skills/inference-optimizer/`)

```
SKILL.md                    ← this file (entrypoint Cursor reads)
README.md                   ← architecture overview
KNOWLEDGE-BASE.md           ← cross-run lessons schema
IMPLEMENTATION-CHECKLIST.md ← per-phase progress tracker

actions/                    ← active catalogue (22 actions)
  <name>.md                 ← prompt body for the LLM sub-agent
  _meta/<name>.yaml         ← machine-readable metadata
                              (cost, lanes, allowed_modes, side_effects)
actions_old/                ← legacy single-skill descriptions, kept for
                              cross-referencing while we drain TODOs
kernel-opt/                 ← per-backend prompt templates
  geak.md / claude.md / codex.md / llm.md
scripts/                    ← real shell + Python tools (GPU-side)
  run_baseline.sh           ← launch sglang/vllm + benchmark + profile
  run_profile.sh            ← profile-only against running server
  run_sweep.sh              ← CONC × ISL/OSL grid sweep
  eval_accuracy.sh          ← lm-evaluation-harness GSM8K
  bootstrap.sh              ← BYOI: GEAK / intellikit / Ray / OOB / npm
  common.sh                 ← kill_server / wait_for_health / filter_trace
  executor.sh               ← local / claw / fully-local dispatch
  geak_ray_submit.py        ← Ray-scheduled GEAK CLI
  oob_ray_submit.py         ← Ray-scheduled OOB (Claude/Codex) CLI
  ray_submit.py             ← generic Ray claw submit
  patch_inductor.py         ← apply GEAK output to Inductor cache
  trace_action.py           ← per-component start/end timestamps
system_prompts/             ← per-agent role markdown bodies
```

**Python ↔ shell bridge** (Phase 8a, see ``IMPLEMENTATION-CHECKLIST.md``):
the `SubAgentRunner` first looks up an `ActionExecutor` in
`src/inference_optimizer/orchestrator/action_executors/` for the
queued delegate task. If found, the executor shells out to the
matching `scripts/*.sh` and parses the result file (`metrics.json` /
`results.tsv` / `eval_summary_*.json`). It then emits `update_state`
intents that flow through the same `PolicyGate` as LLM intents.
When required env vars are missing the executor raises
`ExecutorEnvError` and the runner falls back to the LLM-driven path
(useful for dev boxes without GPU + InferenceX).

## Where to read more

- [`inference-optimizer-DESIGN-modified.md`](./inference-optimizer-DESIGN-modified.md) — full v0.5 design
- [`IMPLEMENTATION-CHECKLIST.md`](./IMPLEMENTATION-CHECKLIST.md) — granular task list (this is what you tick off)
- [`README.md`](./README.md) — short architecture summary
- [`KNOWLEDGE-BASE.md`](./KNOWLEDGE-BASE.md) — historical lessons (sprint+marathon merged)

## TODO (next phases)

- [x] Phase 6.1 ClaudeBackend + bootstrap (Node + claude CLI auto-install)
- [x] Phase 6.2 CodexBackend (`validated_json_output` parser + 22 tests +
      shared `build_repair_prompt` helper, OpenAI-compat proxy support)
- [x] Phase 4 PolicyGate wiring (intent permissions + quick allowlist)
- [x] F1 multi-reactor Conductor (one reactor per role, role-aware prompts)
- [x] F2 full intent dispatcher (10 intent types, idempotent task queueing)
- [x] F3a `ActionRegistry` (load `actions/_meta/*.yaml` + system prompts)
- [x] F3b `SubAgentRunner` skeleton (lane acquire → backend → metrics)
- [x] F4 MCP custom-tool ``emit_intent`` registered inside `ClaudeBackend`
- [x] 11.4 Codex no-tools + `validated_json_output` stability suite
      (22 unit tests + 2 e2e proxy sessions)
- [x] 13.6 Codex Critic / Sage repair-prompt fallback (1-shot retry via
      `build_repair_prompt`)
- [x] Phase 8a Python ↔ shell `ActionExecutor` bridge (5 executors:
      `baseline` / `bench_runner` / `profile` / `param_sweep_run` /
      `kernel_opt`). They shell out to `scripts/run_baseline.sh` /
      `run_profile.sh` / `run_sweep.sh` / `geak_ray_submit.py` /
      `oob_ray_submit.py` and emit `update_state` intents through the
      same PolicyGate as LLM-emitted ones. Conductor auto-derives
      `cumulative_gain` from any `(baseline_tput, current_tput)` pair.
      CLI loads `ActionRegistry` by default → dispatcher loop runs.
- [ ] Phase 7 OOB sub-agent dispatch (per-task per-role backend factory)
- [ ] Phase 8b real GPU integration (sandbox bootstrap.sh + InferenceX)
- [ ] Phase 9 BudgetAwareScheduler wired to Conductor prompt
- [ ] Marathon cadences (persona distill, KB synthesis)
- [ ] Cursor skill manifest entry + demo recordings
