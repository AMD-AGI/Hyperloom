#!/usr/bin/env bash
# Multi-CLI smoke test — proves the --transport multi-cli + auto-launch
# pipeline works end-to-end on a real machine (with or without GPU).
#
# Two flavours:
#
#   ./multi_cli_smoke.sh mock        — uses backend=mock-cli (Python
#                                      mock_agent), no Claude/Codex/GPU
#                                      needed. Runs in <60s. Use this in
#                                      CI + on a developer laptop to
#                                      verify the wiring after every
#                                      pull.
#
#   ./multi_cli_smoke.sh claude      — uses real claude --print --continue
#                                      under --transport hybrid with only
#                                      executor as a CLI. Requires
#                                      ANTHROPIC_API_KEY. Runs for
#                                      MAX_HOURS (default 0.25 = 15min).
#                                      Use this on a real GPU sandbox to
#                                      validate the inbox/outbox protocol
#                                      against an actual Claude session.
#
#   ./multi_cli_smoke.sh marathon    — full marathon: all 4 agents as CLIs,
#                                      tmux-managed for human attach.
#                                      Requires both ANTHROPIC + OPENAI
#                                      keys + tmux + InferenceX. Use this
#                                      to validate the production layout.
#
# Required env (ALL flavours):
#   MODEL_PATH          — model weights / hub id
#   INFERENCEX_PATH     — only for claude / marathon
#   ANTHROPIC_API_KEY   — only for claude / marathon
#   OPENAI_API_KEY      — only for marathon (Critic / Sage)
#
# Optional:
#   MAX_HOURS           — wall budget, defaults per flavour
#   SESSION_ROOT        — overrides INFERENCE_OPTIMIZER_SESSION_ROOT
#                         (default /tmp/io-multicli-smoke)

set -euo pipefail

FLAVOUR="${1:-mock}"

cd "$(dirname "$(realpath "$0")")/../../.."   # repo root
REPO_ROOT="$(pwd)"

: "${SESSION_ROOT:=/tmp/io-multicli-smoke}"
mkdir -p "$SESSION_ROOT"
export INFERENCE_OPTIMIZER_SESSION_ROOT="$SESSION_ROOT"

case "$FLAVOUR" in
    mock)
        : "${MAX_HOURS:=0.005}"   # ~18s
        : "${MODEL_PATH:=/tmp/fake-model-for-mock-smoke}"
        echo "[multi_cli_smoke] flavour=mock  session_root=$SESSION_ROOT  max_hours=$MAX_HOURS"
        # The mock-cli backend doesn't ship as a discoverable card; we
        # synthesise an override via a tiny Python harness that calls
        # Conductor directly with launcher_overrides. This proves the
        # full stack without needing a per-flavour agent_card.yaml.
        PYTHONPATH=src python - <<'PY'
import asyncio
import os
from pathlib import Path

from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import Conductor, TransportMode
from inference_optimizer.orchestrator.multi_cli.agent_card import (
    AgentCard, RestartPolicy,
)
from inference_optimizer.paths import make_session_dir, db_path_for

session_dir = make_session_dir()
print(f"[multi_cli_smoke] session_dir = {session_dir}")
print(f"[multi_cli_smoke] db          = {db_path_for(session_dir)}")

card = AgentCard(
    name="executor", role="executor", backend="mock-cli",
    card_path=Path("/dev/null"), card_dir=Path("/dev/null"),
    enabled=True,
    restart_policy=RestartPolicy(max_restarts=1, backoff_seconds=0,
                                 continue_flag=False),
    extra={"mock_cli_args": [
        "--poll-s", "0.1", "--max-iterations", "200",
        "--baseline-on-start",
        "--emit-on-event", "event=send_message",
    ]},
)
env = {
    "MODEL_PATH": os.environ.get("MODEL_PATH", "/tmp/fake-model"),
    "MAX_HOURS": os.environ.get("MAX_HOURS", "0.005"),
}
pythonpath = str(Path("src").resolve())
conductor = Conductor(
    session_dir, backend=MockBackend(), env=env,
    transport_mode=TransportMode.MULTI_CLI,
    cli_agents=("executor",),
    agents_root=session_dir / "no-such-dir",   # force overrides
    router_tick_s=0.05, clock_tick_s=0.5,
    launch_cli_agents="subprocess",
    cli_shutdown_grace_s=10.0,
    launcher_env={"PYTHONPATH": pythonpath, **env},
    launcher_overrides={"executor": card},
)
asyncio.run(conductor.run())

print("\n[multi_cli_smoke] === post-run summary ===")
import sqlite3
conn = sqlite3.connect(db_path_for(session_dir))
conn.row_factory = sqlite3.Row
print("Bus events by topic:")
for row in conn.execute("SELECT topic, COUNT(*) AS n FROM events GROUP BY topic ORDER BY n DESC"):
    print(f"  {row['topic']:20s} {row['n']}")
print("\nBus events by from_agent:")
for row in conn.execute("SELECT from_agent, COUNT(*) AS n FROM events GROUP BY from_agent ORDER BY n DESC"):
    print(f"  {row['from_agent']:20s} {row['n']}")
print("\nTasks by kind/state:")
for row in conn.execute("SELECT kind, state, COUNT(*) AS n FROM tasks GROUP BY kind, state"):
    print(f"  {row['kind']:12s} {row['state']:15s} {row['n']}")
conn.close()

outbox = session_dir / "agents" / "executor" / "outbox.jsonl"
print(f"\nExecutor outbox lines: {sum(1 for _ in open(outbox))}" if outbox.exists() else "(no outbox)")
print(f"Session dir: {session_dir}")
PY
        ;;

    claude)
        : "${MAX_HOURS:=0.25}"
        : "${MODEL_PATH:?MODEL_PATH is required for the claude smoke}"
        : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required for the claude smoke}"
        : "${INFERENCEX_PATH:=/hyperloom/InferenceX}"
        echo "[multi_cli_smoke] flavour=claude  model=$MODEL_PATH  max_hours=$MAX_HOURS"
        PYTHONPATH=src python -m inference_optimizer \
            --model "$MODEL_PATH" \
            --max-hours "$MAX_HOURS" \
            --inferencex-path "$INFERENCEX_PATH" \
            --backend claude \
            --transport hybrid \
            --cli-agents executor \
            --launch-cli-agents subprocess \
            --reactor-tick-s 5.0 \
            --clock-tick-s 10.0 \
            --router-tick-s 0.5 \
            --log-level INFO
        ;;

    marathon)
        : "${MAX_HOURS:=7}"
        : "${MODEL_PATH:?MODEL_PATH is required}"
        : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
        : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for Codex roles}"
        : "${INFERENCEX_PATH:=/hyperloom/InferenceX}"
        if ! command -v tmux >/dev/null 2>&1; then
            echo "ERROR: tmux is required for the marathon flavour" >&2
            echo "       (use launch_cli_agents=subprocess if you only need headless)" >&2
            exit 1
        fi
        echo "[multi_cli_smoke] flavour=marathon  model=$MODEL_PATH  max_hours=$MAX_HOURS"
        PYTHONPATH=src python -m inference_optimizer \
            --model "$MODEL_PATH" \
            --max-hours "$MAX_HOURS" \
            --inferencex-path "$INFERENCEX_PATH" \
            --backend claude \
            --transport multi-cli \
            --launch-cli-agents tmux \
            --reactor-tick-s 5.0 \
            --clock-tick-s 10.0 \
            --router-tick-s 0.5 \
            --log-level INFO &
        CONDUCTOR_PID=$!
        echo "[multi_cli_smoke] conductor PID = $CONDUCTOR_PID"
        echo "[multi_cli_smoke] tmux session = io-multicli (attach: tmux attach -t io-multicli)"
        wait $CONDUCTOR_PID
        ;;

    *)
        echo "Usage: $0 {mock|claude|marathon}" >&2
        exit 64
        ;;
esac

echo "[multi_cli_smoke] done."
