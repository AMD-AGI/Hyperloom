#!/usr/bin/env bash
# quantization-agent installer.
#
# Sets up the runtime quantization-agent needs:
#   - claude-agent-sdk  (already pinned in inference_optimizer's pyproject.toml,
#                        but we install it explicitly so the agent is usable
#                        standalone)
#   - amd-quark         (editable install from $QUARK_ROOT/Quark so the agent's
#                        Step 1 model-intake helper can resolve quark.*)
#   - PyYAML            (used by result_collector to parse run_manifest.yaml)
#
# Mirrors the contract of kernel-agent/scripts/install.sh but is deliberately
# small — quantization-agent is a thin SDK wrapper, not a build/test rig.

set -euo pipefail

QUANTIZATION_AGENT_ROOT="${QUANTIZATION_AGENT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# QUARK_ROOT defaults to the canonical checkout location on Hyperloom hosts.
# Operators should override when their checkout lives elsewhere — the runner's
# resolve_quark_root() honours $QUARK_ROOT as the first fallback after the
# explicit kwarg.
QUARK_ROOT="${QUARK_ROOT:-/scratch/kewang/workspace/Quark}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '[quantization-agent install] %s\n' "$*"; }

# 1. Sanity-check the Quark checkout.
if [ ! -d "$QUARK_ROOT" ]; then
  log "ERROR: QUARK_ROOT=$QUARK_ROOT does not exist."
  log "       Either clone amd-quark there, or export QUARK_ROOT=<path-to-Quark> before re-running."
  exit 1
fi
SKILL_ENTRY="$QUARK_ROOT/.claude/skills/quark-torch-ptq/SKILL.md"
if [ ! -f "$SKILL_ENTRY" ]; then
  log "ERROR: $SKILL_ENTRY not found — does this Quark checkout include the .claude/skills tree?"
  exit 1
fi
log "Quark repo: $QUARK_ROOT"
log "PTQ skill entry: $SKILL_ENTRY"

# 2. claude-agent-sdk — try to import first to keep idempotent reruns cheap.
if ! "$PYTHON_BIN" -c "import claude_agent_sdk" >/dev/null 2>&1; then
  log "installing claude-agent-sdk via pip"
  "$PYTHON_BIN" -m pip install --quiet 'claude-agent-sdk>=0.1.65'
else
  log "claude-agent-sdk already importable"
fi

# 3. amd-quark — install editable from $QUARK_ROOT if the repo has a setup.py /
# pyproject.toml. Otherwise fall back to pip's amd-quark wheel (Quark publishes
# wheels for cuda + rocm builds; pip picks the right one).
if ! "$PYTHON_BIN" -c "import quark" >/dev/null 2>&1; then
  if [ -f "$QUARK_ROOT/pyproject.toml" ] || [ -f "$QUARK_ROOT/setup.py" ]; then
    log "installing amd-quark editable from $QUARK_ROOT"
    "$PYTHON_BIN" -m pip install --quiet -e "$QUARK_ROOT"
  else
    log "installing amd-quark wheel via pip (no setup.py/pyproject.toml at $QUARK_ROOT)"
    "$PYTHON_BIN" -m pip install --quiet amd-quark
  fi
else
  log "quark already importable"
fi

# 4. PyYAML — required by result_collector for run_manifest.yaml parsing.
if ! "$PYTHON_BIN" -c "import yaml" >/dev/null 2>&1; then
  log "installing PyYAML"
  "$PYTHON_BIN" -m pip install --quiet 'PyYAML>=6.0'
else
  log "PyYAML already importable"
fi

# 5. Self-test: import the agent's own modules to surface syntax errors early.
log "self-test: importing quantization-agent modules"
PYTHONPATH="$QUANTIZATION_AGENT_ROOT/tools:${PYTHONPATH:-}" \
  "$PYTHON_BIN" -c "
import intent, result_collector, runner
i = intent.normalize_intent({'global_scheme': 'fp8'})
assert i.global_scheme == 'fp8'
print(f'intent_digest={intent.intent_hash(i)}')
print('OK')
"

# 6. Optional: persist QUARK_ROOT to a small env file so other shells pick it
# up. Skipped if HYPERLOOM_RUNTIME_DIR is unset to avoid polluting random
# /tmp paths.
if [ -n "${HYPERLOOM_RUNTIME_DIR:-}" ]; then
  ENV_FILE="${HYPERLOOM_RUNTIME_DIR}/quantization-agent.env.sh"
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<EOF
export QUARK_ROOT="$QUARK_ROOT"
export QUANTIZATION_AGENT_ROOT="$QUANTIZATION_AGENT_ROOT"
EOF
  log "wrote $ENV_FILE"
fi

log "install complete"
