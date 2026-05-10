#!/usr/bin/env bash
# Marathon launcher — single blocking entrypoint for the skill.
#
# The skill's top-level SKILL.md tells the agent to:
#   export MODEL_NAME=... BASE_DIR=... [MAX_HOURS=...] ...
#   bash $SKILL_ROOT/scripts/launcher/run.sh
#
# Auto-detects two modes:
#   * sandbox — remote GPU sandbox (claude CLI, tmux, jq pre-installed via /app/*)
#   * local   — any GPU host with an SGLang/vLLM image; missing deps installed on the fly
#
# =====================================================================
# Required env (caller sets):
#   MODEL_NAME                — e.g. DeepSeek-R1-0528
#   BASE_DIR                  — baseline / handoff / prior session root
#
# Auth (sandbox injects automatically; local caller must export):
#   ANTHROPIC_AUTH_TOKEN      — LLM auth (copied to ANTHROPIC_API_KEY)
#   ANTHROPIC_BASE_URL        — OCI LLM gateway
#   SAFE_API_KEY              — MCP auth (falls back to ANTHROPIC_AUTH_TOKEN)
#
# Optional tuning (defaults shown):
#   MAX_HOURS=24              — wall-clock budget
#   FRAMEWORK=sglang          — sglang | vllm
#   MODEL_CLASS=moe_mla       — dense | moe_mla | moe_swa | moe_mla_nsa
#   GPU_COUNT=8  GPU_TYPE=MI355X  TP=8  EP=1  PRECISION=fp8
#   CONC=64  ISL=1024  OSL=1024
#   MODEL_PATH                — absolute path to weights (optional)
#   IMAGE                     — GEAK container image (if empty, geak backend skipped)
#   KERNEL_OPT_WORKSPACE=control-plane-sandbox
#   KERNEL_OPT_BACKENDS=geak,claude,codex
#   DRY_RUN=0                 — 1 = preflight only, no tmux
#   REPORT_INTERVAL_S=60      — monitor cadence
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The actual Marathon protocol (SKILL.md / actions / kernel-manager / watchdog /
# modes / kb / scripts) is the source of truth maintained by Neha in the main
# Hyperloom repo. We reference it, not copy it, so skill packages stay small
# (~800 lines) and Neha's upstream fixes flow through automatically.
: "${SPEC_ROOT:=/hyperloom/Hyperloom/marathon_optimization/marathon_harness/skills}"
# Fallback to sibling repo checkout when /hyperloom/Hyperloom isn't mounted
# (e.g. developer machines running straight off /shared_nfs/<user>/Hyperloom/).
if [[ ! -d "$SPEC_ROOT" ]]; then
    for cand in \
        "/shared_nfs/xiaofei/Hyperloom/marathon_optimization/marathon_harness/skills" \
        "$(cd "$SKILL_ROOT/../../../marathon_optimization/marathon_harness/skills" 2>/dev/null && pwd || true)"
    do
        if [[ -d "$cand" ]]; then SPEC_ROOT="$cand"; break; fi
    done
fi

log() { echo "[run.sh $(date +%H:%M:%S)] $*"; }

# ============================================================
# preflight_deps() — fail-fast dependency check + auto-install
# ============================================================
# Run before STRICT/defaults/mode detection so missing CLI dependencies show up
# as the first visible failure, not deep inside a backgrounded tmux launch.
run_preflight_step() {
    local label=$1
    shift
    local timeout_s="${PREFLIGHT_STEP_TIMEOUT_S:-300}"
    log "  ${label} (timeout=${timeout_s}s)"
    set +e
    timeout --signal=TERM --kill-after=30s "$timeout_s" "$@"
    local rc=$?
    set -e
    if [[ $rc -eq 124 || $rc -eq 137 || $rc -eq 143 ]]; then
        echo "ERROR: preflight step '$label' timed out after ${timeout_s}s" >&2
        exit 14
    fi
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: preflight step '$label' failed with exit $rc" >&2
        exit 14
    fi
}

ensure_node() {
    if command -v node >/dev/null 2>&1; then
        local ver
        ver="$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/')"
        if [[ -n "$ver" && "$ver" -ge 18 ]]; then
            log "  node: $(node --version) (>=18, OK)"
            return 0
        fi
        log "  node $(node --version) is <18, installing Node 20 binary"
    else
        log "  node not on PATH, installing Node 20 binary"
    fi
    local NODE_VER=v20.18.0 arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64)  arch=x64 ;;
        aarch64) arch=arm64 ;;
        *) echo "ERROR: unsupported arch '$arch' for Node binary install" >&2; return 1 ;;
    esac
    local url="https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-$arch.tar.xz"
    run_preflight_step "install Node $NODE_VER" \
        bash -o pipefail -c 'curl -fsSL "$1" | tar -xJ -C /usr/local --strip-components=1 --exclude="*.md" --exclude="LICENSE"' _ "$url"
    hash -r
    log "  installed: node $(node --version)  npm $(npm --version)"
}

preflight_deps() {
    log "preflight_deps: checking required CLIs (tmux jq curl claude)"
    log "  hostname=$(hostname)  arch=$(uname -m)"

    command -v timeout >/dev/null 2>&1 \
        || { echo "ERROR: required CLI 'timeout' is missing (coreutils); cannot enforce preflight timeouts" >&2; exit 9; }

    local need_apt=()
    for tool in tmux jq curl; do
        command -v "$tool" >/dev/null 2>&1 || need_apt+=("$tool")
    done

    if [[ ${#need_apt[@]} -gt 0 ]]; then
        log "  missing apt pkgs: ${need_apt[*]} — installing"
        if ! command -v apt-get >/dev/null 2>&1; then
            echo "ERROR: need ${need_apt[*]} but apt-get unavailable; install manually then re-run" >&2
            exit 10
        fi
        run_preflight_step "apt-get update" \
            bash -o pipefail -c 'apt-get update -qq 2>&1 | tail -2'
        run_preflight_step "apt-get install ${need_apt[*]}" \
            bash -o pipefail -c 'apt-get install -y -qq "$@" 2>&1 | tail -2' _ "${need_apt[@]}"
        for t in "${need_apt[@]}"; do
            command -v "$t" >/dev/null 2>&1 \
                || { echo "ERROR: $t still missing after apt-get install — check apt sources / network" >&2; exit 11; }
        done
    fi

    if ! command -v claude >/dev/null 2>&1 && [[ "${STAGE_ONLY:-0}" != "1" ]]; then
        log "  claude CLI missing — installing via npm"
        if ! command -v npm >/dev/null 2>&1; then
            ensure_node || { echo "ERROR: Node.js install failed (prereq for claude CLI)" >&2; exit 12; }
        fi
        run_preflight_step "npm install @anthropic-ai/claude-code" \
            bash -o pipefail -c 'npm install -g @anthropic-ai/claude-code 2>&1 | tail -3'
        command -v claude >/dev/null 2>&1 \
            || { echo "ERROR: claude CLI unavailable after 'npm install -g @anthropic-ai/claude-code'" >&2; exit 13; }
    fi

    log "preflight_deps OK:"
    if command -v tmux   >/dev/null 2>&1; then log "  tmux:   $(tmux -V 2>&1)"; fi
    if command -v jq     >/dev/null 2>&1; then log "  jq:     $(jq --version 2>&1)"; fi
    if command -v curl   >/dev/null 2>&1; then log "  curl:   $(curl --version 2>&1 | head -1)"; fi
    if command -v claude >/dev/null 2>&1; then log "  claude: $(command -v claude) ($(claude --version 2>&1 | head -1))"; fi
}

preflight_deps

# ---------- STRICT mode (MUST run BEFORE defaults) ----------
# When STRICT=1, every parameter that has a default must ALSO be explicitly passed
# by the caller. Catches: "agent only exported MODEL_NAME/BASE_DIR and silently
# fell back to defaults".
if [[ "${STRICT:-0}" == "1" ]]; then
    declare -a strict_required=(MODEL_NAME BASE_DIR MAX_HOURS FRAMEWORK MODEL_CLASS \
                                GPU_COUNT GPU_TYPE TP EP PRECISION CONC ISL OSL)
    strict_missing=()
    for v in "${strict_required[@]}"; do
        [[ -n "${!v:-}" ]] || strict_missing+=("$v")
    done
    if [[ ${#strict_missing[@]} -gt 0 ]]; then
        echo "ERROR: STRICT=1 requires every parameter to be explicitly exported." >&2
        echo "       Missing: ${strict_missing[*]}" >&2
        echo "       Either set them all, or unset STRICT to use defaults silently." >&2
        exit 20
    fi
    log "STRICT=1: all ${#strict_required[@]} critical params are explicitly set"
fi

# ---------- defaults ----------
: "${MAX_HOURS:=24}"
: "${FRAMEWORK:=sglang}"
: "${MODEL_CLASS:=moe_mla}"
: "${GPU_COUNT:=8}"
: "${GPU_TYPE:=MI355X}"
: "${TP:=8}"
: "${EP:=1}"
: "${PRECISION:=fp8}"
: "${CONC:=64}"
: "${ISL:=1024}"
: "${OSL:=1024}"
: "${MODEL_PATH:=}"
: "${IMAGE:=}"
: "${KERNEL_OPT_WORKSPACE:=control-plane-sandbox}"
: "${KERNEL_OPT_BACKENDS:=geak,claude,codex}"
: "${INFERENCEX_PATH:=/hyperloom/InferenceX}"
: "${DRY_RUN:=0}"
: "${REPORT_INTERVAL_S:=60}"

# ---------- required user env ----------
for v in MODEL_NAME BASE_DIR; do
    [[ -n "${!v:-}" ]] || { echo "ERROR: required env var $v is not set" >&2; exit 2; }
done

# ---------- auth (sandbox injects; local caller must export) ----------
[[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]] && export ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN"
: "${SAFE_API_KEY:=${ANTHROPIC_AUTH_TOKEN:-}}"
export SAFE_API_KEY

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY not set" >&2; exit 6
fi
if [[ -z "${ANTHROPIC_BASE_URL:-}" ]]; then
    echo "ERROR: ANTHROPIC_BASE_URL not set" >&2; exit 6
fi

# ---------- mode detection ----------
if [[ -n "${ENGINE_TYPE:-}" && "$ENGINE_TYPE" == "claude" && -d /app ]]; then
    MODE="sandbox"
else
    MODE="local"
fi

# ---------- session dir ----------
TS="$(date +%Y%m%d-%H%M%S)"
if [[ "$MODE" == "sandbox" ]]; then
    # /workspace/hyperloom/ auto-syncs to S3 per template-gpu.json (S3_BUCKET=claw)
    SESSION_ROOT="${WORKSPACE_ROOT:-/workspace/hyperloom/marathon-sessions}"
    WORK_DIR="/workspace/marathon"
else
    SESSION_ROOT="${WORKSPACE_ROOT:-/shared_nfs/xiaofei/marathon-sessions}"
    WORK_DIR="${MARATHON_WORK_DIR:-/tmp/marathon}"
fi
SESSION_DIR="$SESSION_ROOT/$TS"

# ========== STEP 1 — PREFLIGHT ==========
log "Mode: $MODE"
log "  MODEL=$MODEL_NAME  BASE_DIR=$BASE_DIR  MAX_HOURS=$MAX_HOURS"
log "  FRAMEWORK=$FRAMEWORK  CLASS=$MODEL_CLASS  GPU=${GPU_COUNT}×${GPU_TYPE}  TP=$TP  PRECISION=$PRECISION"
log "  Workload: CONC=$CONC  ISL=$ISL  OSL=$OSL"
log "  MODEL_PATH=${MODEL_PATH:-(none)}"
log "  KERNEL_OPT: IMAGE=${IMAGE:-(none; geak skipped)}  WORKSPACE=$KERNEL_OPT_WORKSPACE  BACKENDS=$KERNEL_OPT_BACKENDS"
log "  SKILL_ROOT=$SKILL_ROOT   (launcher)"
log "  SPEC_ROOT=$SPEC_ROOT     (protocol source of truth)"
log "  INFERENCEX_PATH=$INFERENCEX_PATH"
log "  SESSION_DIR=$SESSION_DIR"

if [[ ! -d "$BASE_DIR" ]]; then
    if mkdir -p "$BASE_DIR" 2>/dev/null; then
        log "BASE_DIR did not exist, created: $BASE_DIR"
    else
        echo "ERROR: BASE_DIR does not exist and cannot be created: $BASE_DIR" >&2; exit 3
    fi
fi

for f in "$SPEC_ROOT/SKILL.md" "$SPEC_ROOT/kernel-manager/SKILL.md" "$SPEC_ROOT/watchdog/SKILL.md"; do
    [[ -f "$f" ]] || { echo "ERROR: spec file missing: $f (set SPEC_ROOT env to Neha's marathon_harness/skills path)" >&2; exit 4; }
done

for f in "$SCRIPT_DIR/pane_orchestrator.md" "$SCRIPT_DIR/pane_kernel_mgr.md" "$SCRIPT_DIR/pane_watchdog.md"; do
    [[ -f "$f" ]] || { echo "ERROR: pane prompt missing: $f" >&2; exit 5; }
done
log "Preflight OK"

# ========== STEP 1.5 — bootstrap baseline launch script (cold→Mode B) ==========
# Give the marathon agent a concrete Mode B starting point so warm-start can skip
# the cold find/grep scan across /hyperloom (wastes 5–10 min per session).
#
# Strategy (in order):
#   1. If BASE_DIR/scripts already has *.sh → skip (user/agent provided one).
#   2. Try to copy a production-validated launch script from $INFERENCEX_PATH
#      whose name matches the model. This is the SAFEST path — those scripts
#      are tuned by the framework team.
#   3. Fall back to a TRULY MINIMAL template (model-path, host, port, tp,
#      trust-remote-code, "$@" only). NO opinionated perf flags here —
#      mem-fraction / cuda-graph-max-bs / disable-radix-cache / aiter / etc
#      belong in the marathon DFS action stack, not in the baseline.
#      A wrong default at the baseline either deceives KEEP/REVERT measurements
#      (false +/- gain) or crashes the server (e.g. mem-fraction=0.85 on a
#      MoE model whose expert weights need more headroom).
#
# Triggers only when ALL of the following hold:
#   * BASE_DIR/scripts/ contains no *.sh
#   * MODEL_PATH non-empty AND is an existing directory
#   * BASE_DIR is writable by current user
LAUNCH_SH="$BASE_DIR/scripts/launch_server.sh"
if compgen -G "$BASE_DIR/scripts/*.sh" > /dev/null 2>&1; then
    log "launch.sh auto-generate skipped (reason: BASE_DIR already has scripts/*.sh)"
elif [[ -z "${MODEL_PATH:-}" ]]; then
    log "launch.sh auto-generate skipped (reason: MODEL_PATH empty)"
elif [[ ! -d "$MODEL_PATH" ]]; then
    log "launch.sh auto-generate skipped (reason: MODEL_PATH not a directory: $MODEL_PATH)"
elif [[ ! -w "$BASE_DIR" ]]; then
    log "launch.sh auto-generate skipped (reason: BASE_DIR not writable)"
else
    mkdir -p "$BASE_DIR/scripts"

    # --- Step A: try to reuse an InferenceX production launch script ---
    INFX_SCRIPT=""
    if [[ -d "$INFERENCEX_PATH" ]]; then
        # Match on lowercased model basename (e.g. Qwen-Qwen3-8B → qwen3-8b, gpt-oss-120b).
        # `|| true` on each segment because we run with `set -euo pipefail` and a non-match
        # in grep / empty find result would otherwise abort the whole launcher.
        MODEL_KEY="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | sed 's/^qwen-//' || true)"
        MODEL_KEY_NODASH="$(echo "$MODEL_KEY" | tr -d - || true)"
        # head -1 is intentional: take the first match, deterministic enough for warm-start.
        INFX_SCRIPT="$( { find "$INFERENCEX_PATH" -type f -name '*.sh' 2>/dev/null || true; } \
                       | { grep -i "/${FRAMEWORK}/" || true; } \
                       | { grep -iE "(${MODEL_KEY}|${MODEL_KEY_NODASH})" || true; } \
                       | head -1 || true)"
    fi

    if [[ -n "$INFX_SCRIPT" && -f "$INFX_SCRIPT" ]]; then
        cp "$INFX_SCRIPT" "$LAUNCH_SH"
        chmod +x "$LAUNCH_SH"
        log "baseline launch script: copied InferenceX production script"
        log "  source: $INFX_SCRIPT"
        log "  dest:   $LAUNCH_SH"
        log "  (agent will use this as Mode B warm-start; tune via DFS)"
    else
        # --- Step B: fall back to MINIMAL template ---
        # Heredoc is unquoted so $MODEL_PATH / $TP / $MODEL_NAME / $(date) expand at
        # write time; runtime placeholders ("$@") are escaped with \$.
        case "$FRAMEWORK" in
            sglang)
                cat > "$LAUNCH_SH" <<LAUNCHEOF
#!/usr/bin/env bash
# AUTO-GENERATED MINIMAL TEMPLATE — do NOT add perf flags here.
# Created by run.sh on $(date -Iseconds) for $MODEL_NAME.
#
# This template is intentionally minimal. All perf flags
# (--mem-fraction-static, --cuda-graph-max-bs, --disable-radix-cache,
#  attention backends, --enable-mtp, env vars like SGLANG_USE_AITER) are
# OPTIMIZATION ACTIONS for the marathon agent to discover via the DFS loop
# and KB. Adding defaults here gives the agent a misleading baseline:
#   - flags that help may show "no gain" because they're already on
#   - flags that hurt this model become the silent baseline
# Either way KEEP/REVERT measurements lose meaning.
set -euo pipefail
python3 -m sglang.launch_server \\
    --model-path "$MODEL_PATH" \\
    --host=0.0.0.0 --port 8888 \\
    --tensor-parallel-size $TP \\
    --trust-remote-code \\
    "\$@"
LAUNCHEOF
                ;;
            vllm)
                cat > "$LAUNCH_SH" <<LAUNCHEOF
#!/usr/bin/env bash
# AUTO-GENERATED MINIMAL TEMPLATE — do NOT add perf flags here.
# Created by run.sh on $(date -Iseconds) for $MODEL_NAME.
#
# All perf flags (--gpu-memory-utilization, --enable-prefix-caching,
# --max-num-seqs, --kv-cache-dtype, etc) are OPTIMIZATION ACTIONS for the
# marathon agent. Adding defaults here biases KEEP/REVERT measurements.
set -euo pipefail
python3 -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_PATH" \\
    --tensor-parallel-size $TP \\
    --host 0.0.0.0 --port 8888 \\
    --trust-remote-code \\
    "\$@"
LAUNCHEOF
                ;;
            *)
                log "FRAMEWORK=$FRAMEWORK: no minimal template available, agent will cold-start"
                LAUNCH_SH=""  # signal that nothing was generated
                ;;
        esac
        if [[ -n "$LAUNCH_SH" && -f "$LAUNCH_SH" ]]; then
            chmod +x "$LAUNCH_SH"
            log "baseline launch script: minimal template (no InferenceX match for '$MODEL_KEY')"
            log "  dest: $LAUNCH_SH"
            log "  (agent will add perf flags via DFS actions; do NOT preset them here)"
        fi
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN complete"
    exit 0
fi

# ========== STEP 2 — DEP VERIFY (install was done up-front in preflight_deps) ==========
for t in tmux jq; do
    command -v "$t" >/dev/null 2>&1 \
        || { echo "ERROR: '$t' disappeared after preflight_deps — PATH was mutated" >&2; exit 16; }
done
if [[ "${STAGE_ONLY:-0}" != "1" ]]; then
    command -v claude >/dev/null 2>&1 \
        || { echo "ERROR: 'claude' disappeared after preflight_deps — PATH was mutated" >&2; exit 16; }
fi

# ========== STEP 3 — BOOTSTRAP (session dir + mcp.json + env file + pane scripts) ==========
mkdir -p "$SESSION_DIR/logs" \
         "$SESSION_DIR/kernel_manager/merge_ready" \
         "$SESSION_DIR/kernel_manager/rca_reports"
touch    "$SESSION_DIR/kernel_manager/work_queue.jsonl" \
         "$SESSION_DIR/kernel_manager/results.jsonl" \
         "$SESSION_DIR/kernel_manager/event_log.jsonl" \
         "$SESSION_DIR/kernel_manager/findings.jsonl"

mkdir -p "$WORK_DIR"
ln -snf "$SPEC_ROOT"  "$WORK_DIR/spec"
ln -snf "$SKILL_ROOT" "$WORK_DIR/launcher"

# mcp.json — claude CLI native schema ({"type":"sse"} / {"type":"http"}); no jq rewrite.
# Env vars expand directly via bash heredoc; no <placeholder> pattern.
MCP_CONFIG="$WORK_DIR/mcp.json"
cat > "$MCP_CONFIG" <<JSON
{
  "mcpServers": {
    "oci-geak-agent": {
      "type": "sse",
      "url": "https://oci-slc.primus-safe.amd.com/control-plane/control-plane-dev/geak-agent-wvsbv/mcp/sse",
      "headers": { "Authorization": "Bearer ${SAFE_API_KEY}" }
    },
    "oci-oob-agent": {
      "type": "sse",
      "url": "https://oci-slc.primus-safe.amd.com/control-plane/control-plane-dev/agent-mcp-server-gpu-62tcr/sse",
      "headers": { "Authorization": "Bearer ${SAFE_API_KEY}" }
    },
    "oci-traceLens-agent": {
      "type": "http",
      "url": "https://oci-slc.primus-safe.amd.com/control-plane/control-plane-dev/trace-lens-agent-qqpfv/mcp"
    }
  }
}
JSON
chmod 600 "$MCP_CONFIG"
log "mcp.json → $MCP_CONFIG"
jq -r '.mcpServers | to_entries[] | "  \(.key): \(.value.type) \(.value.url)"' "$MCP_CONFIG"

# env file consumed by pane launcher scripts
ENV_FILE="$WORK_DIR/.env"
cat > "$ENV_FILE" <<ENVEOF
SESSION_DIR=$SESSION_DIR
BASE_DIR=$BASE_DIR
SKILL_ROOT=$SKILL_ROOT
SPEC_ROOT=$SPEC_ROOT
MCP_CONFIG=$MCP_CONFIG
MODEL_NAME=$MODEL_NAME
MODEL_CLASS=$MODEL_CLASS
FRAMEWORK=$FRAMEWORK
GPU_COUNT=$GPU_COUNT
GPU_TYPE=$GPU_TYPE
TP=$TP
EP=$EP
PRECISION=$PRECISION
CONC=$CONC
ISL=$ISL
OSL=$OSL
MODEL_PATH=$MODEL_PATH
IMAGE=$IMAGE
KERNEL_OPT_WORKSPACE=$KERNEL_OPT_WORKSPACE
KERNEL_OPT_BACKENDS=$KERNEL_OPT_BACKENDS
INFERENCEX_PATH=$INFERENCEX_PATH
MAX_HOURS=$MAX_HOURS
PANE_CLAUDE_TIMEOUT_S=${PANE_CLAUDE_TIMEOUT_S:-1800}
PANE_MAX_RESTARTS=${PANE_MAX_RESTARTS:-200}
PANE_CLAUDE_DEBUG=${PANE_CLAUDE_DEBUG:-0}
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL
ENVEOF
chmod 600 "$ENV_FILE"

MODEL_ARG="${ANTHROPIC_DEFAULT_SONNET_MODEL:-claude-sonnet-4-6}"
ALLOWED_TOOLS="Bash Read Write Edit MultiEdit Glob Grep TodoWrite Task WebSearch WebFetch mcp__oci-geak-agent mcp__oci-oob-agent mcp__oci-traceLens-agent"

# Compose a restart-loop pane launcher. The inner `claude --print` is one-shot;
# outer `while` with `--continue` resumes prior conversation context indefinitely.
# `--output-format stream-json --verbose` flushes tool_use/result immediately so
# the log is observable (prevents "dead process / 0-byte log" misreads).
write_pane_script() {
    local name=$1 prompt_file=$2 user_msg=$3 log_file=$4
    local script_path="$WORK_DIR/run_pane_${name}.sh"
    local sp_b64; sp_b64="$(base64 -w0 < "$prompt_file")"
    local um_q;   um_q="$(printf '%q' "$user_msg")"

    cat > "$script_path" <<PANE
#!/usr/bin/env bash
# Auto-generated pane launcher for '${name}'.
set -a; source "$ENV_FILE"; set +a
cd "$WORK_DIR"

SYSTEM_PROMPT=\$(base64 -d <<'B64'
${sp_b64}
B64
)

LOG="$log_file"
STOP_FILE="$SESSION_DIR/STOP_PANE_${name}"
# PID file for the currently-running \`claude --print\` invocation. run.sh's
# monitor uses this to soft-restart a wedged kernel-mgr pane when the
# orchestrator sets state.json:km_requested_restart=true.
PID_FILE="$SESSION_DIR/.pane_${name}.pid"
MAX_RESTARTS=\${PANE_MAX_RESTARTS:-200}
CLAUDE_TIMEOUT_S=\${PANE_CLAUDE_TIMEOUT_S:-1800}
CLAUDE_DEBUG=\${PANE_CLAUDE_DEBUG:-0}
DIAG_DIR="\$SESSION_DIR/diagnostics/${name}"
mkdir -p "\$DIAG_DIR" 2>/dev/null || true
ATTEMPT=0
CONTINUE_FLAG=""
USER_MSG=${um_q}

echo "[\$(date -Iseconds)] [pane:${name}] launcher starting (max_restarts=\$MAX_RESTARTS claude_timeout=\${CLAUDE_TIMEOUT_S}s debug=\$CLAUDE_DEBUG)" >> "\$LOG"
while [ ! -f "\$STOP_FILE" ] && [ \$ATTEMPT -lt \$MAX_RESTARTS ]; do
    ATTEMPT=\$((ATTEMPT + 1))
    echo "[\$(date -Iseconds)] [pane:${name}] attempt=\$ATTEMPT continue=\$CONTINUE_FLAG" >> "\$LOG"
    CLAUDE_START=\$(date +%s)
    DEBUG_ENV=""
    if [ "\$CLAUDE_DEBUG" = "1" ]; then
        DEBUG_ENV="ANTHROPIC_LOG=debug"
    fi
    env \$DEBUG_ENV timeout --signal=TERM --kill-after=30s "\$CLAUDE_TIMEOUT_S" claude --print \\
        --output-format stream-json --verbose \\
        \$CONTINUE_FLAG \\
        --model "$MODEL_ARG" \\
        --mcp-config "$MCP_CONFIG" \\
        --permission-mode dontAsk \\
        --add-dir "$SPEC_ROOT" \\
        --add-dir "$SKILL_ROOT" \\
        --add-dir "$BASE_DIR" \\
        --add-dir "$SESSION_DIR" \\
        --add-dir "$INFERENCEX_PATH" \\
        --allowedTools "$ALLOWED_TOOLS" \\
        --system-prompt "\$SYSTEM_PROMPT" \\
        \$USER_MSG \\
        >> "\$LOG" 2>&1 < /dev/null &
    INNER_PID=\$!
    echo \$INNER_PID > "\$PID_FILE"
    wait \$INNER_PID
    EXIT=\$?
    rm -f "\$PID_FILE" 2>/dev/null || true
    CLAUDE_DURATION=\$(( \$(date +%s) - CLAUDE_START ))
    if [ "\$EXIT" = "124" ] || [ "\$EXIT" = "137" ]; then
        echo "[\$(date -Iseconds)] [pane:${name}] WARN: claude --print exceeded \${CLAUDE_TIMEOUT_S}s wallclock (exit=\$EXIT, ran for \${CLAUDE_DURATION}s); restarting with --continue" >> "\$LOG"
        DIAG_FILE="\$DIAG_DIR/hang_attempt\${ATTEMPT}_\$(date +%Y%m%dT%H%M%S).txt"
        {
            echo "=== Hang post-mortem for [pane:${name}] attempt=\$ATTEMPT ==="
            echo "timestamp: \$(date -Iseconds)"
            echo "duration_seconds: \$CLAUDE_DURATION"
            echo "timeout_setting: \$CLAUDE_TIMEOUT_S"
            echo "exit_code: \$EXIT"
            echo
            echo "--- claude/node processes (post-timeout, may already be reaped) ---"
            ps -eo pid,ppid,etime,stat,cmd 2>/dev/null | grep -E '\\b(claude|node)\\b' | grep -v grep | head -20 || echo "(none)"
            echo
            echo "--- ESTABLISHED TCP to candidate hang targets ---"
            (ss -tnp 2>/dev/null || netstat -tnp 2>/dev/null) | grep -E 'ESTAB|ESTABLISHED' | grep -E 'anthropic|claude|amd\\.com|primus|oci-' | head -20 || echo "(none)"
            echo
            echo "--- log mtime ---"
            stat -c '%y %n' "\$LOG" 2>/dev/null || echo "(stat failed)"
            echo "now: \$(date -Iseconds)"
            echo
            echo "--- last 20 lines of pane log ---"
            tail -20 "\$LOG" 2>/dev/null
            echo "=== end ==="
        } > "\$DIAG_FILE" 2>&1
        echo "[\$(date -Iseconds)] [pane:${name}] hang diagnostic snapshot -> \$DIAG_FILE" >> "\$LOG"
    else
        echo "[\$(date -Iseconds)] [pane:${name}] claude exit=\$EXIT (ran for \${CLAUDE_DURATION}s)" >> "\$LOG"
    fi
    [ -f "\$STOP_FILE" ] && break
    sleep 15
    CONTINUE_FLAG="--continue"
    USER_MSG="Continue. Read \$SESSION_DIR/state.json to resume; then proceed with the next protocol step."
done
echo "[\$(date -Iseconds)] [pane:${name}] launcher exiting (attempt=\$ATTEMPT stop_file=\$([ -f \"\$STOP_FILE\" ] && echo yes || echo no))" >> "\$LOG"
PANE
    chmod +x "$script_path"
    echo "$script_path"
}

SCRIPT_W=$(write_pane_script "watchdog"     "$SCRIPT_DIR/pane_watchdog.md"     'Begin watchdog monitoring.'                               "$SESSION_DIR/logs/watchdog.log")
SCRIPT_O=$(write_pane_script "orchestrator" "$SCRIPT_DIR/pane_orchestrator.md" 'Begin the Marathon protocol from Step 0 WARM-START now.'  "$SESSION_DIR/logs/orchestrator.log")
SCRIPT_K=$(write_pane_script "kernel-mgr"   "$SCRIPT_DIR/pane_kernel_mgr.md"   'Begin polling the kernel-manager work queue.'             "$SESSION_DIR/logs/kernel-mgr.log")

# ========== STEP 4 — START TMUX ==========
if [[ "${STAGE_ONLY:-0}" == "1" ]]; then
    log "STAGE_ONLY=1 — stopping before tmux start"
    log "Generated files:"
    log "  $MCP_CONFIG"
    log "  $ENV_FILE"
    log "  $SCRIPT_W"
    log "  $SCRIPT_O"
    log "  $SCRIPT_K"
    echo "SESSION_DIR=$SESSION_DIR"
    echo "WORK_DIR=$WORK_DIR"
    exit 0
fi
tmux kill-session -t marathon 2>/dev/null || true
tmux new-session  -d -s marathon -n watchdog     -c "$WORK_DIR"
tmux new-window   -t marathon   -n orchestrator  -c "$WORK_DIR"
tmux new-window   -t marathon   -n kernel-mgr    -c "$WORK_DIR"
tmux send-keys -t marathon:watchdog     "bash $SCRIPT_W; exit" C-m
tmux send-keys -t marathon:orchestrator "bash $SCRIPT_O; exit" C-m
tmux send-keys -t marathon:kernel-mgr   "bash $SCRIPT_K; exit" C-m
log "tmux session 'marathon' started — windows: $(tmux list-windows -t marathon -F '#W' | paste -sd ',')"
echo ""
echo "SESSION_DIR=$SESSION_DIR"
echo "WORK_DIR=$WORK_DIR"

# Best-effort SESSION_REPORT.md writer used when the orchestrator pane fails to
# finalize before cleanup runs out of patience. Reads $SESSION_DIR/state.json
# and emits a minimal but useful report so a long-running session never ends
# with "process killed mid-tool-call → zero report → 3 hours of work invisible".
# This is a safety net, NOT a replacement for the pane's own report writer.
generate_fallback_report() {
    local SR="$SESSION_DIR/SESSION_REPORT.md"
    local STATE="$SESSION_DIR/state.json"
    [[ -f "$SR" ]] && return 0          # pane already wrote one — don't overwrite
    [[ -f "$STATE" ]] || { log "  fallback report skipped: no state.json"; return 1; }
    log "  writing fallback SESSION_REPORT.md from state.json"
    {
        echo "# Marathon Session Report (FALLBACK)"
        echo
        echo "_This report was generated by run.sh cleanup, NOT by the orchestrator pane._"
        echo "_Reason: pane did not write SESSION_REPORT.md within the grace window;"
        echo "likely killed mid-tool-call (long bench/compile). Data is read directly_"
        echo "_from state.json and may be partial._"
        echo
        echo "**Session:** $(basename "$SESSION_DIR")"
        echo "**Generated:** $(date -Iseconds) (fallback)"
        echo
        echo "## Snapshot"
        jq -r '
          "**Phase:** \(.phase // "?")  ",
          "**Baseline:** \(.baseline_tput_per_gpu // 0) tok/s/GPU  ",
          "**Current:** \(.current_tput_per_gpu // 0) tok/s/GPU  ",
          "**Cumulative gain:** \(.cumulative_gain_pct // 0)%  ",
          "**Crashes:** \(.crash_count // 0)  ",
          "**Dreams:** \(.dream_count // 0)  ",
          "**KM merges kept:** \(.kernel_manager_merges_kept // 0) / pushed \(.kernel_manager_targets_pushed // 0)"
        ' "$STATE" 2>/dev/null
        echo
        echo "## Completed Actions"
        jq -r '(.completed_actions // []) | if length==0 then "_(none)_" else
          map("- [\(.score // "?")] \(.action // "?"): tput=\(.result.tput_per_gpu // "-") gain=\(.result.gain_pct // "-")% status=\(.status // "-")") | join("\n")
          end' "$STATE" 2>/dev/null
        echo
        echo "## Remaining Action Stack"
        jq -r '(.action_stack // []) | if length==0 then "_(empty)_" else
          map("- [\(.score // "?")] \(.action // "?"): \((.description // "")[0:100])") | join("\n")
          end' "$STATE" 2>/dev/null
        echo
        echo "## Discovered Constraints"
        jq -r '(.architecture_constraints // []) | if length==0 then "_(none)_" else
          map("- " + .) | join("\n")
          end' "$STATE" 2>/dev/null
        echo
        echo "---"
        echo "_End of fallback report. Inspect \`$SESSION_DIR/state.json\` and \`$SESSION_DIR/logs/\` for full detail._"
    } > "$SR" 2>/dev/null \
      && log "  fallback report written: $SR ($(wc -c < "$SR" 2>/dev/null) bytes)" \
      || log "  WARN: failed to write fallback report"
}

cleanup() {
    log "cleanup: signalling panes + killing tmux + killing inference server"
    for name in watchdog orchestrator kernel-mgr; do
        touch "$SESSION_DIR/STOP_PANE_$name" 2>/dev/null || true
    done
    # SESSION_REPORT.md is produced in FINALIZE before `[run.sh] Done`; the
    # EXIT trap may race sandbox teardown, so only attempt a last-ditch fallback here.
    if [[ ! -f "$SESSION_DIR/SESSION_REPORT.md" ]]; then
        log "  cleanup: SESSION_REPORT.md still missing — last-ditch fallback attempt"
        generate_fallback_report
    fi
    # Kill pane claude processes BEFORE tmux kill, so they don't reparent to PID 1 as zombies.
    # Match by the unique signature --mcp-config <WORK_DIR>/mcp.json (every pane's claude has this).
    for p in $(pgrep -f "claude .*--mcp-config $WORK_DIR/mcp.json" 2>/dev/null || true); do
        log "  killing pane claude pid=$p (TERM)"
        kill -TERM "$p" 2>/dev/null || true
    done
    sleep 3
    for p in $(pgrep -f "claude .*--mcp-config $WORK_DIR/mcp.json" 2>/dev/null || true); do
        log "  hard-killing pane claude pid=$p (KILL)"
        kill -KILL "$p" 2>/dev/null || true
    done
    tmux kill-session -t marathon 2>/dev/null || true
    # Kill the inference server the orchestrator launched (zombie protection).
    # Otherwise sglang/vllm server keeps loading + occupies GPU until external kill.
    local PIDFILE=/tmp/.marathon_server.pid
    if [[ -f "$PIDFILE" ]]; then
        local SPID
        SPID="$(cat "$PIDFILE" 2>/dev/null || true)"
        if [[ -n "$SPID" ]] && kill -0 "$SPID" 2>/dev/null; then
            log "  killing inference server pid=$SPID (from $PIDFILE)"
            kill "$SPID" 2>/dev/null || true
            sleep 5
            kill -0 "$SPID" 2>/dev/null && { log "  hard kill -9 $SPID"; kill -9 "$SPID" 2>/dev/null || true; }
        fi
        rm -f "$PIDFILE" 2>/dev/null || true
    fi
    # Also clean any stray sglang/vllm matching the model we launched
    for p in $(pgrep -f 'python.*-m (sglang.launch_server|vllm.entrypoints|vllm serve)' 2>/dev/null); do
        log "  killing stray inference process pid=$p"
        kill "$p" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ========== STEP 5 — MONITOR ==========
START=$(date +%s)
BUDGET=$(awk "BEGIN{printf \"%d\", $MAX_HOURS * 3600}")
LAST_REPORT=0
log "monitor: budget=${MAX_HOURS}h  report every ${REPORT_INTERVAL_S}s"

KM_LAST_RESTART=0
while tmux has-session -t marathon 2>/dev/null; do
    NOW=$(date +%s); ELAPSED=$(( NOW - START ))

    # Orchestrator sets state.json:km_requested_restart=true after kernel-mgr
    # silence while pending work exists. Soft-kill the current inner claude;
    # the pane launcher will relaunch with --continue.
    if [[ -f "$SESSION_DIR/state.json" ]] \
        && [[ $(( NOW - KM_LAST_RESTART )) -ge 300 ]] \
        && jq -e '.km_requested_restart == true' "$SESSION_DIR/state.json" >/dev/null 2>&1; then
        KM_PID_FILE="$SESSION_DIR/.pane_kernel-mgr.pid"
        if [[ -f "$KM_PID_FILE" ]]; then
            KM_PID=$(cat "$KM_PID_FILE" 2>/dev/null || true)
            if [[ -n "$KM_PID" ]] && kill -0 "$KM_PID" 2>/dev/null; then
                log "monitor: km_requested_restart=true -> SIGTERM inner claude pid=$KM_PID"
                kill -TERM "$KM_PID" 2>/dev/null || true
                for _ in $(seq 1 12); do
                    kill -0 "$KM_PID" 2>/dev/null || break
                    sleep 5
                done
                if kill -0 "$KM_PID" 2>/dev/null; then
                    log "monitor: km pid=$KM_PID still alive after 60s SIGTERM grace; escalating to SIGKILL"
                    kill -KILL "$KM_PID" 2>/dev/null || true
                fi
                KM_LAST_RESTART=$NOW
            else
                rm -f "$KM_PID_FILE" 2>/dev/null || true
            fi
        else
            log "monitor: km_requested_restart=true but no PID file at $KM_PID_FILE; skipping"
            KM_LAST_RESTART=$NOW
        fi
    fi

    if [[ $(( NOW - LAST_REPORT )) -ge $REPORT_INTERVAL_S ]]; then
        printf "[%s elapsed=%dmin]" "$(date +%H:%M:%S)" $(( ELAPSED / 60 ))
        if [[ -f "$SESSION_DIR/state.json" ]]; then
            jq -r ' "  phase=\(.phase // "?") tput=\(.current_tput_per_gpu // 0) gain=\(.cumulative_gain_pct // 0)% completed=\((.completed_actions // [])|length) crash=\(.crash_count // 0) dream=\(.dream_count // 0) km=\(.kernel_manager_merges_kept // 0)/\(.kernel_manager_targets_pushed // 0)" ' \
                "$SESSION_DIR/state.json" 2>/dev/null || echo "  (state.json not parseable)"
        else
            echo "  (state.json not written yet)"
        fi
        # Forward the latest activity from ALL THREE panes so the calling
        # agent / human watching /tmp/marathon.log sees full picture, not just orch.
        for pane in orchestrator kernel-mgr watchdog; do
            pane_log="$SESSION_DIR/logs/${pane}.log"
            [[ -s "$pane_log" ]] || continue
            echo "  [${pane}]"
            tail -300 "$pane_log" 2>/dev/null \
                | jq -rR 'fromjson? |
                    if .type=="assistant" then
                        (.message.content[]? |
                            if .type=="text"     then "  text: " + (.text[0:240])
                            elif .type=="tool_use" then "  tool: " + .name + " " + ((.input.command // .input.path // (.input|tostring))[0:200])
                            else empty end)
                    elif .type=="user" then
                        (.message.content[]? | select(.type=="tool_result") |
                            "  result: " + ((.content // "" | tostring)[0:160]))
                    elif .type=="system" then "  [system " + (.subtype // "") + "]"
                    else empty end' 2>/dev/null \
                | tail -3 | sed 's/^/    /'
        done
        LAST_REPORT=$NOW
    fi

    if [[ $ELAPSED -ge $BUDGET ]]; then
        log "time budget reached; graceful shutdown via STOP_PANE_* files"
        for name in watchdog orchestrator kernel-mgr; do
            touch "$SESSION_DIR/STOP_PANE_$name" 2>/dev/null || true
        done
        # NOTE: previously a `tmux send-keys` call followed here that tried to push
        # "Stop DFS; ... write SESSION_REPORT.md; exit." into the orchestrator pane.
        # That was a no-op: the keys land in the outer bash stdin, but the inner
        # `claude --print < /dev/null` never reads them. Removed to stop pretending
        # we have a second signaling channel. STOP_PANE_* files + cleanup() (which
        # has its own SESSION_REPORT.md wait + fallback writer) are the only paths.
        for _ in 1 2 3 4 5 6; do
            [[ -f "$SESSION_DIR/SESSION_REPORT.md" ]] && break
            tmux has-session -t marathon 2>/dev/null || break
            sleep 10
        done
        break
    fi
    sleep 10
done

# ========== STEP 6 — FINALIZE ==========
log "marathon complete"

# SESSION_REPORT.md must exist before `[run.sh] Done`. In the remote sandbox the
# polling agent stops as soon as it sees Done, often triggering container
# snapshot/teardown before the EXIT trap has time to write the fallback report.
if [[ ! -f "$SESSION_DIR/SESSION_REPORT.md" ]]; then
    log "finalize: 60s grace window for orchestrator pane to produce SESSION_REPORT.md"
    waited=0
    while [[ $waited -lt 60 ]]; do
        [[ -f "$SESSION_DIR/SESSION_REPORT.md" ]] && { log "  pane wrote SESSION_REPORT.md after ${waited}s"; break; }
        sleep 5; waited=$((waited + 5))
    done
fi
if [[ ! -f "$SESSION_DIR/SESSION_REPORT.md" ]]; then
    log "finalize: pane never produced SESSION_REPORT.md within grace; invoking fallback writer"
    generate_fallback_report
fi
sync 2>/dev/null || true
sleep 5

echo ""
echo "============================================"
echo "  MARATHON FINAL"
echo "  SESSION_DIR: $SESSION_DIR"
echo "============================================"
if [[ -f "$SESSION_DIR/SESSION_REPORT.md" ]]; then
    echo "=== SESSION_REPORT.md ==="
    cat "$SESSION_DIR/SESSION_REPORT.md"
    echo "=== END SESSION_REPORT ==="
else
    echo "=== NO SESSION_REPORT.md — final state.json ==="
    cat "$SESSION_DIR/state.json" 2>/dev/null || echo "(no state.json)"
    echo "=== END state.json ==="
fi
echo "[run.sh] Done"
