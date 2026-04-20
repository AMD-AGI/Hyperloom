#!/usr/bin/env bash
# Marathon launcher — single blocking entrypoint for the skill.
#
# The skill's top-level SKILL.md tells the agent to:
#   export MODEL_NAME=... BASE_DIR=... [MAX_HOURS=...] ...
#   bash $SKILL_ROOT/scripts/launcher/run.sh
#
# Auto-detects two modes:
#   * sandbox — Claw GPU sandbox (claude CLI, tmux, jq pre-installed via /app/*)
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

# ========== STEP 2 — INSTALL DEPS (local mode only) ==========
# Ubuntu apt nodejs is too old (v12) to run claude CLI (needs >=18). Pull the
# Node 20 binary tarball from nodejs.org when npm is missing or node is too old.
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
    log "  fetching $url"
    curl -fsSL "$url" \
      | tar -xJ -C /usr/local --strip-components=1 --exclude='*.md' --exclude='LICENSE' \
      || { echo "ERROR: failed to download/extract Node $NODE_VER" >&2; return 1; }
    hash -r
    log "  installed: node $(node --version)  npm $(npm --version)"
}

if [[ "$MODE" == "local" ]]; then
    need=()
    command -v tmux >/dev/null 2>&1 || need+=(tmux)
    command -v jq   >/dev/null 2>&1 || need+=(jq)
    command -v curl >/dev/null 2>&1 || need+=(curl)
    if [[ ${#need[@]} -gt 0 ]]; then
        log "Installing: ${need[*]}"
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq 2>&1 | tail -2
            apt-get install -y -qq "${need[@]}" 2>&1 | tail -2
        else
            echo "ERROR: need ${need[*]} but apt-get not available" >&2; exit 10
        fi
    fi
    if ! command -v claude >/dev/null 2>&1 && [[ "${STAGE_ONLY:-0}" != "1" ]]; then
        log "claude CLI not on PATH — installing via npm"
        if ! command -v npm >/dev/null 2>&1; then
            ensure_node || { echo "ERROR: failed to install Node.js (prerequisite for claude CLI)" >&2; exit 12; }
        fi
        npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
    fi
    if [[ "${STAGE_ONLY:-0}" != "1" ]]; then
        command -v claude >/dev/null 2>&1 || { echo "ERROR: claude CLI unavailable after install" >&2; exit 13; }
    fi
fi

if command -v claude >/dev/null 2>&1; then
    log "  claude: $(command -v claude) ($(claude --version 2>&1 | head -1))"
fi
command -v tmux >/dev/null 2>&1 && log "  tmux: $(tmux -V 2>&1)" || log "  tmux: (not installed; STAGE_ONLY)"

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
      "url": "https://oci-slc.example-internal-host.invalid/control-plane/control-plane-dev/geak-agent-wvsbv/mcp/sse",
      "headers": { "Authorization": "Bearer ${SAFE_API_KEY}" }
    },
    "oci-oob-agent": {
      "type": "sse",
      "url": "https://oci-slc.example-internal-host.invalid/control-plane/control-plane-dev/agent-mcp-server-gpu-62tcr/sse",
      "headers": { "Authorization": "Bearer ${SAFE_API_KEY}" }
    },
    "oci-traceLens-agent": {
      "type": "http",
      "url": "https://oci-slc.example-internal-host.invalid/control-plane/control-plane-dev/trace-lens-agent-qqpfv/mcp"
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
MAX_RESTARTS=50
ATTEMPT=0
CONTINUE_FLAG=""
USER_MSG=${um_q}

echo "[\$(date -Iseconds)] [pane:${name}] launcher starting" >> "\$LOG"
while [ ! -f "\$STOP_FILE" ] && [ \$ATTEMPT -lt \$MAX_RESTARTS ]; do
    ATTEMPT=\$((ATTEMPT + 1))
    echo "[\$(date -Iseconds)] [pane:${name}] attempt=\$ATTEMPT continue=\$CONTINUE_FLAG" >> "\$LOG"
    claude --print \\
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
        >> "\$LOG" 2>&1 < /dev/null
    EXIT=\$?
    echo "[\$(date -Iseconds)] [pane:${name}] claude exit=\$EXIT" >> "\$LOG"
    [ -f "\$STOP_FILE" ] && break
    sleep 15
    CONTINUE_FLAG="--continue"
    USER_MSG="Continue. Read \$SESSION_DIR/state.json to resume; then proceed with the next protocol step."
done
echo "[\$(date -Iseconds)] [pane:${name}] launcher exiting" >> "\$LOG"
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
    # Graded grace window for SESSION_REPORT.md:
    #   Phase 1 (0-180s):  let the orchestrator pane write its proper report
    #                      after seeing STOP_PANE_orchestrator (best path).
    #   Phase 2 (>180s):   pane is stuck in a long tool call (compile/bench).
    #                      run.sh writes a fallback SESSION_REPORT.md from
    #                      state.json so we never end with zero report.
    #   Then proceed to TERM/KILL of pane claude processes (Phase 3).
    local waited=0
    while [[ $waited -lt 180 ]]; do
        [[ -f "$SESSION_DIR/SESSION_REPORT.md" ]] && { log "  SESSION_REPORT.md present after ${waited}s (pane wrote it)"; break; }
        sleep 5; waited=$((waited + 5))
    done
    if [[ ! -f "$SESSION_DIR/SESSION_REPORT.md" ]]; then
        log "  pane did not write SESSION_REPORT.md within 180s; using fallback"
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

while tmux has-session -t marathon 2>/dev/null; do
    NOW=$(date +%s); ELAPSED=$(( NOW - START ))

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
