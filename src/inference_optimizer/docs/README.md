# Inference Optimizer (skill)

Adaptive, effect-driven, multi-agent loop that drives an LLM serving
stack on AMD MI355X toward a higher `tok/s/GPU` while staying within
an `accuracy_drop ≤ 1%` budget.

## Architecture (3-pane summary)

```
                 ┌────────────────────────────────┐
                 │ Conductor (single owner)       │
                 │  ┌──────────┐  ┌────────────┐  │
                 │  │ reactors │  │ scheduler  │  │
                 │  │ (1‑4)    │  │ + actions  │  │
                 │  └──────────┘  └────────────┘  │
                 │  PolicyGate · TaskRegistry     │
                 │  CursorStore · ResourceLocks   │
                 │  SqliteConnection (WAL)        │
                 └──────────────────┬─────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       ┌────────────┐        ┌────────────┐         ┌────────────┐
       │ Backend    │        │ KB / Sage  │         │ SubAgent   │
       │ (claude /  │        │ (jsonl +   │         │ Runner +   │
       │  codex /   │        │  Codex)    │         │ Actions    │
       │  mock)     │        │            │         │            │
       └────────────┘        └────────────┘         └─────┬──────┘
                                                          │
                                              spawn OOB ▼
                                       ┌─────────────────────────┐
                                       │ scripts/run_baseline.sh │
                                       │ scripts/eval_accuracy   │
                                       │ scripts/patch_inductor  │
                                       └─────────────────────────┘
```

## Three modes (auto-selected by `MAX_HOURS`)

| Mode                  | `MAX_HOURS` | Roles                          | Token budget |
|-----------------------|-------------|--------------------------------|--------------|
| `quick_param_sweep`   | < 2         | executor                       | ~0.5M        |
| `guided_kernel_opt`   | 2 – 6       | executor + critic              | ~3M          |
| `marathon_multi_agent`| > 6         | executor + critic + watchdog + sage | ~11.5M  |

Quick mode skips kernel-opt and the marathon-only cadences (persona
distill / strategic review / cross-run synth). All modes get
`PolicyGate`, `ActionRegistry`, the `SubAgentRunner` dispatcher, and
the SQLite-backed event/cursor/lease/task store.

## Required env

| Var                            | Required | Notes                                  |
|--------------------------------|----------|----------------------------------------|
| `MODEL_PATH`                   | ✅       | HF hub id or local path                |
| `MAX_HOURS`                    | ✅       | float; selects mode                    |
| `TARGET_GAIN_PCT`              |          | numeric goal (0–100)                   |
| `TARGET_TPUT_PER_GPU`          |          | numeric goal (`tok/s/GPU`)             |
| `TARGET_DIR`                   |          | dir with prior runs / explicit targets |
| `INFERENCE_OPTIMIZER_SESSION_ROOT` |      | override session root                  |
| `INFERENCE_OPTIMIZER_DB_PATH`  |          | override DB location                   |

At most one of `TARGET_GAIN_PCT` / `TARGET_TPUT_PER_GPU` / `TARGET_DIR`
should be set.

## Launch

PowerShell:

```pwsh
$env:MODEL_PATH = "openai/gpt-oss-20b"
$env:MAX_HOURS  = "1.0"
python -m inference_optimizer --backend mock --auto-install --log-level INFO
```

Bash + Claude (auto-install Node + claude CLI on first use):

```bash
MODEL_PATH=openai/gpt-oss-20b MAX_HOURS=1.0 \
  python -m inference_optimizer --backend claude --auto-install
```

Bash + Codex (no Node bootstrap needed; pure-pip via `openai>=1.50`):

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or your proxy / Azure / Foundry
# self-signed cert proxy? add: export INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL=0

MODEL_PATH=openai/gpt-oss-20b MAX_HOURS=1.0 \
  python -m inference_optimizer --backend codex --codex-model gpt-5.4
```

For an Anthropic-compat proxy that wants `Authorization: Bearer ...`
(instead of the standard `x-api-key`), set both `ANTHROPIC_AUTH_TOKEN`
and `ANTHROPIC_BASE_URL`; if the cert is self-signed also set
`NODE_TLS_REJECT_UNAUTHORIZED=0` so the bundled claude CLI's Node
process accepts it.

## Monitoring

```bash
# one-shot snapshot
bash src/inference_optimizer/scripts/monitor.sh

# tail every 5 s
bash src/inference_optimizer/scripts/monitor.sh --watch 5

# detailed views
bash src/inference_optimizer/scripts/monitor.sh --per-agent --per-lane --top-events 20
```

Or the cross-platform Python port:

```bash
python -m inference_optimizer.scripts.monitor --watch 5 --per-agent
```

Exits 0 when `cursors lag ≤ 10` and no zombie tasks; non-zero otherwise
(useful as a pre-commit / CI smoke gate).

## Status / Known limitations

- All three backends (`mock`, `claude`, `codex`) are wired and tested;
  the Conductor still uses a single backend instance for every reactor,
  so per-role backend mixing (e.g. Claude Executor + Codex Critic in
  the same run) is the next step (Phase 7 — per-task backend factory).
- `dispatch_pending_delegates` runs in-process for v0.7; OOB process
  isolation lands with Phase 7.
- Real `run_baseline.sh` / `eval_accuracy.sh` need the sprint sandbox;
  set `DRY_RUN_MOCK=1` to use the deterministic test fixtures.

## More

- `inference-optimizer-DESIGN-modified.md` — full design spec
- `IMPLEMENTATION-CHECKLIST.md` — granular progress tracker
- `KNOWLEDGE-BASE.md` — cross-run lessons schema
