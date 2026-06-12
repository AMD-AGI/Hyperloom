#!/usr/bin/env bash
#
# One-time OOB backend install on a Dynamo GPU pod (idempotent), driven over
# SSH by `inference_optimizer.multi_node install-oob`. Mirrors the sandbox
# install.sh ensure_node + ensure_oob so claude/codex/cursor kernel-agent
# backends work on the Dynamo backend (no Ray).
#
# Inputs via env (piped over SSH stdin, never argv): OOB_SRC (shared NFS path),
# OOB_BASE_URL, OOB_API_KEY (creds). The OOB python package installs from the
# shared OOB_SRC checkout; claude/codex/@cursor/sdk are npm globals.
#
# Emits one JSON line. Exit 0 on installed/skipped (best-effort: missing tools
# are reported, not fatal, so the kernel phase surfaces a clear error later).
#
set -uo pipefail

OOB_SRC="${OOB_SRC:-}"
PIP="/opt/venv/bin/pip"
command -v "$PIP" >/dev/null 2>&1 || PIP="python3 -m pip"
OOB_INSTALL_SRC=""
if [ -n "$OOB_SRC" ] && [ -d "$OOB_SRC" ]; then
  if [ -f "$OOB_SRC/pyproject.toml" ]; then
    OOB_INSTALL_SRC="$OOB_SRC"
  fi
fi

# 1. Node.js / npm (NodeSource apt) — needed for claude/codex/@cursor CLIs.
if ! command -v node >/dev/null 2>&1 || ! npm --version >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if curl -fsSL https://deb.nodesource.com/setup_20.x -o /tmp/mn_ns20.sh 2>/dev/null; then
      bash /tmp/mn_ns20.sh >/dev/null 2>&1 || true
      apt-get -y install nodejs >/dev/null 2>&1 || true
    fi
  fi
fi

# 2. oob python CLI from the shared OOB_SRC checkout (no copy: NFS is shared).
if ! command -v oob >/dev/null 2>&1; then
  if [ -n "$OOB_INSTALL_SRC" ]; then
    [ -f "$OOB_INSTALL_SRC/requirements.txt" ] && \
      $PIP install -q --no-cache-dir --break-system-packages -r "$OOB_INSTALL_SRC/requirements.txt" >/tmp/mn_oob_pip.log 2>&1 || true
    $PIP install -q --no-cache-dir --break-system-packages "$OOB_INSTALL_SRC" >>/tmp/mn_oob_pip.log 2>&1 || true
  fi
fi

# 3. claude / codex / @cursor-sdk npm globals.
if command -v npm >/dev/null 2>&1; then
  npm config set prefix /usr/local >/dev/null 2>&1 || true
  command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 || true
  command -v codex  >/dev/null 2>&1 || npm install -g @openai/codex@0.100.0 >/dev/null 2>&1 || true
  if ! NODE_PATH="$(npm root -g 2>/dev/null || true)" node -e "require.resolve('@cursor/sdk')" >/dev/null 2>&1; then
    npm install -g @cursor/sdk >/dev/null 2>&1 || true
  fi
fi

# 4. Auth files (verbatim from ensure_oob). Anthropic SDK appends /v1 itself.
mkdir -p /root/.claude /root/.codex
ANTH_URL="${OOB_BASE_URL:-}"; ANTH_URL="${ANTH_URL%/}"; ANTH_URL="${ANTH_URL%/v1}"
KEY="${OOB_API_KEY:-${OPENAI_API_KEY:-}}"
cat > /root/.claude/config.json <<EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${KEY}",
  "customApiUrl": "${ANTH_URL}"
}
EOF
chmod 600 /root/.claude/config.json
cat > /root/.codex/auth.json <<EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${KEY}"
}
EOF
chmod 600 /root/.codex/auth.json

# 5. Report.
MISSING=""
for t in oob claude codex; do
  command -v "$t" >/dev/null 2>&1 || MISSING="${MISSING} ${t}"
done
if [ -z "$MISSING" ]; then
  echo '{"status":"installed"}'
else
  echo "{\"status\":\"partial\",\"missing\":\"${MISSING# }\"}"
fi
