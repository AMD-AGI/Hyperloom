# Inference Optimizer (v0.6 — multi-reactor + emit_intent MCP tool)

> **STATUS**: ``MockBackend`` (dry-run) and ``ClaudeBackend`` (real Claude
> via ``claude-agent-sdk``) both work. The Conductor now spawns one
> reactor per role (executor + critic + sage + watchdog) according to
> ``ExecutionMode``, every intent flows through ``PolicyGate``, and
> ``emit_intent`` is wired as a real MCP tool inside ``ClaudeBackend``
> (with JSON-in-text envelope as automatic fallback). The
> ``ActionRegistry`` loads action specs from ``actions/_meta/*.yaml`` and
> the ``SubAgentRunner`` skeleton drains queued ``delegate`` tasks. Real
> OOB sub-agent processes + benchmark/kernel-opt actions land in later
> phases. See ``IMPLEMENTATION-CHECKLIST.md``.
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
    [--backend mock|claude] \
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

### v0.6 backends

| backend         | what it does                                                                                    | status |
| --------------- | ----------------------------------------------------------------------------------------------- | ------ |
| `mock`          | scripted heartbeats, no LLM calls, runs locally                                                 | OK     |
| `claude`        | real Claude via `claude-agent-sdk`; in-process MCP server exposes `emit_intent` as a real tool, JSON-in-text envelope is the automatic fallback | OK |
| `codex`         | real Codex SDK, `validated_json_output`                                                         | TODO (Phase 6.x) |

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

## Where to read more

- [`inference-optimizer-DESIGN-modified.md`](./inference-optimizer-DESIGN-modified.md) — full v0.5 design
- [`IMPLEMENTATION-CHECKLIST.md`](./IMPLEMENTATION-CHECKLIST.md) — granular task list (this is what you tick off)
- [`README.md`](./README.md) — short architecture summary
- [`KNOWLEDGE-BASE.md`](./KNOWLEDGE-BASE.md) — historical lessons (sprint+marathon merged)

## TODO (next phases)

- [x] Phase 6.1 ClaudeBackend + bootstrap (Node + claude CLI auto-install)
- [x] Phase 4 PolicyGate wiring (intent permissions + quick allowlist)
- [x] F1 multi-reactor Conductor (one reactor per role, role-aware prompts)
- [x] F2 full intent dispatcher (10 intent types, idempotent task queueing)
- [x] F3a `ActionRegistry` (load `actions/_meta/*.yaml` + system prompts)
- [x] F3b `SubAgentRunner` skeleton (lane acquire → backend → metrics)
- [x] F4 MCP custom-tool ``emit_intent`` registered inside `ClaudeBackend`
- [ ] Phase 6.2 CodexBackend (`validated_json_output` parser + tests)
- [ ] Phase 7 OOB sub-agent dispatch (separate process, per-task backend)
- [ ] Phase 8 real benchmark / kernel-opt actions wired to MI355X sandboxes
- [ ] Phase 9 BudgetAwareScheduler + accuracy gate
- [ ] Marathon cadences (persona distill, KB synthesis)
- [ ] Cursor skill manifest entry + demo recordings
