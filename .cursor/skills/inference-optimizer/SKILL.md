# Inference Optimizer

Thin Cursor entrypoint for the standalone `inference_optimizer` Python package.

Runtime assets now live in `src/inference_optimizer/` so the optimizer can be packaged and run outside Cursor on cloud-native platforms. The `.cursor/skills/inference-optimizer` directory should only contain Cursor-facing instructions and examples.

Long-form design notes live in `docs/inference_optimizer/`.

## When Invoked

Use defaults. Do not ask for backend, framework, TP, CONC, or target unless the user explicitly requests overrides.

Required inputs:

| var | description |
| --- | --- |
| `MODEL_PATH` | path to model weights or HF repo |
| `MAX_HOURS` | wall-clock budget; selects quick/guided/marathon mode |

Backend selection:

| env present | backend |
| --- | --- |
| `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | `claude` |
| `OPENAI_API_KEY` | `codex` |
| neither | stop and report missing backend credentials |

Default runtime env:

```bash
export INFERENCE_OPTIMIZER_SESSION_ROOT="${INFERENCE_OPTIMIZER_SESSION_ROOT:-/tmp/io-sessions}"
export INFERENCE_OPTIMIZER_AUTO_INSTALL="${INFERENCE_OPTIMIZER_AUTO_INSTALL:-1}"
```

For Anthropic-compatible corporate proxies, also set:

```bash
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$ANTHROPIC_API_KEY}"
export NODE_TLS_REJECT_UNAUTHORIZED="${NODE_TLS_REJECT_UNAUTHORIZED:-0}"
```

For OpenAI-compatible corporate proxies with self-signed TLS, set:

```bash
export INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL="${INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL:-0}"
```

## Launch Procedure

Always run from the repository root when using the source checkout.

1. Install Python dependencies:

```bash
pip install -q -r src/inference_optimizer/requirements.txt
```

2. Run preflight and stop on failure:

```bash
MODEL_PATH="$MODEL_PATH" \
INFERENCEX_PATH="${INFERENCEX_PATH:-/hyperloom/InferenceX}" \
INFERENCE_OPTIMIZER_SESSION_ROOT="${INFERENCE_OPTIMIZER_SESSION_ROOT:-/tmp/io-sessions}" \
bash src/inference_optimizer/scripts/preflight.sh
```

3. Launch the optimizer:

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --inferencex-path "${INFERENCEX_PATH:-/hyperloom/InferenceX}" \
  --backend "${BACKEND:-claude}" \
  --reactor-tick-s 5.0 \
  --clock-tick-s 10.0 \
  --log-level INFO
```

Notes:

- The CLI applies `TARGET_GAIN_PCT=100` by default when no explicit target is passed.
- Auto-install is on by default for the Claude CLI; pass `--no-auto-install` only when the user asks.
- The CLI prints `session_dir : ...`; return that path and a monitor command to the user.
- **`--transport` defaults to `multi-cli`** — every active agent (executor / critic / watchdog / sage) runs as its own `claude --print --continue` (or `codex` with explicit `conversation.jsonl`) restart-loop CLI. The Conductor only owns the Router + PolicyGate + SQLite bus. This is what avoids the 8-12h context exhaustion seen on long single-process runs.

## Transport mode (`--transport`)

| flag | when to use | what happens |
| --- | --- | --- |
| **`multi-cli` (default, RECOMMENDED)** | every production run; required for marathon >6h | every active agent role becomes its own `claude --print --continue` (or `codex` with explicit `conversation.jsonl`) restart-loop CLI; the Conductor only runs the Router + PolicyGate + SQLite bus |
| `single-proc` | dev / CI / fast quick<2h smoke runs | one Conductor process owns every reactor as an asyncio task; cheaper to start, easier to `pdb` |
| `hybrid` | transitional debugging | only agents listed in `--cli-agents executor,critic` become CLIs; the rest stay in-process |

To force the legacy in-process model (e.g. for a unit test or a fast quick smoke), pass `--transport single-proc` explicitly.

In `multi-cli` mode the Conductor still owns:
- the SQLite `events` bus (durable, replayable, `monitor.sh` keeps working)
- PolicyGate (every cross-process intent goes through the same gate)
- the delegate dispatcher + ActionExecutor subprocess pool

Per-agent assets live at `src/inference_optimizer/agents/<role>/`:
`agent_card.yaml` declares capabilities + restart policy, and
`system_prompt.md` carries the inbox/outbox workflow contract on top of
the canonical role prompt under `orchestrator/system_prompts/`.

## Asset Layout

Canonical runtime assets are package-local:

```text
src/inference_optimizer/actions/                 action prompts + YAML metadata
src/inference_optimizer/agents/<role>/           agent_card.yaml + system_prompt.md (multi-cli)
src/inference_optimizer/scripts/                 GPU and benchmark scripts
src/inference_optimizer/kernel_opt/              kernel optimization prompts
src/inference_optimizer/orchestrator/system_prompts/ role system prompts (legacy + canonical)
src/inference_optimizer/orchestrator/multi_cli/  Router, launcher, envelope, codex_continuity
```

Override the package asset root only for tests or vendored deployments:

```bash
export INFERENCE_OPTIMIZER_ASSET_ROOT=/path/to/inference_optimizer
```

`INFERENCE_OPTIMIZER_SKILL_ROOT` remains a legacy alias but should not be used for new deployments.
