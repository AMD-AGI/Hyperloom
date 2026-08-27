#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Pre-release E2E pod bootstrap. Runs as the SaFE Authoring workload entrypoint. It
# reproduces the manual release path: install the packaged wheel, write a pod-local
# .env, install a pinned Claude CLI, then drive setup + demo skills through
# `claude --print`. The poll job (outside the pod) judges the session afterwards.
# See hyperloom-pre-release-e2e-ci-design.md §12.
#
# Two modes, selected by E2E_DOCKER_HOST:
#   * unset  -> a single baremetal/docker leg in THIS pod (LEG_ID given).
#   * "1"     -> the privileged 8-GPU host: fan out DOCKER_LEGS to nested containers
#                via docker-run-hyperloom.sh <gpu_index> <leg>, one GPU each.
#
# Inputs (env, injected by the dispatch script):
#   CI_VERSION NFS_ROOT
#   ANTHROPIC_API_KEY_B64          base64 key; decoded here, written only to pod-local .env
#   ANTHROPIC_BASE_URL (optional)  CLAUDE_MODEL  CLAUDE_CLI_VERSION  TARGET_GAIN
#   Baremetal leg:  LEG_ID  HYPERLOOM_RUN_MODE=baremetal  HYPERLOOM_BACKEND  HYPERLOOM_MODEL_PATH  DEMO_HOURS
#   Docker host:    E2E_DOCKER_HOST=1  DOCKER_LEGS  DOCKER_GPU_MAP(json)  MODEL_3H  MODEL_12H
set -euo pipefail

: "${CI_VERSION:?}"; : "${NFS_ROOT:?}"; : "${ANTHROPIC_API_KEY_B64:?}"
: "${CLAUDE_MODEL:?}"; : "${CLAUDE_CLI_VERSION:?}"
TARGET_GAIN="${TARGET_GAIN:-100}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_DIR="${NFS_ROOT%/}/bootstrap/${CI_VERSION}/prompts/pre-release"
WHEEL_DIR="${NFS_ROOT%/}/wheels/${CI_VERSION}"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

install_claude_cli() {
  if command -v claude >/dev/null 2>&1; then log "claude CLI present: $(claude --version 2>/dev/null || true)"; return; fi
  log "installing Claude CLI @ ${CLAUDE_CLI_VERSION}"
  # Pinned install. The exact channel is environment-specific; keep the version in one
  # place (CLAUDE_CLI_VERSION) so the pin is auditable.
  npm install -g "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}" >/dev/null 2>&1 \
    || { log "ERROR: claude CLI install failed"; return 1; }
}

# Run ONE leg to completion inside the current filesystem (baremetal pod, or already
# inside a nested docker container). Args: leg backend model_path hours run_mode
run_leg() {
  local leg="$1" backend="$2" model_path="$3" hours="$4" run_mode="$5"
  local root="${NFS_ROOT%/}/runs/${CI_VERSION}/${leg}"
  local session="${root}/session"
  mkdir -p "$root" "$session"
  log "leg=$leg mode=$run_mode backend=$backend hours=$hours model=$model_path"

  # 1. install the wheel into the leg root (produces importable tree + bundled skills)
  local wheels=("$WHEEL_DIR"/hyperloom_inference_optimizer-*.whl)
  [ -e "${wheels[0]}" ] || { log "ERROR: no wheel in $WHEEL_DIR"; return 1; }
  log "pip install ${wheels[0]} --target $root"
  pip install --no-input --target "$root" "${wheels[0]}" >/dev/null

  # 2. decode the key and write the pod-local .env (NEVER on stdout / NEVER to a
  #    location the poll reads). Restrict perms; scrub on exit.
  local envf="${root}/.env"
  ( umask 077
    {
      echo "ANTHROPIC_API_KEY=$(printf '%s' "$ANTHROPIC_API_KEY_B64" | base64 -d)"
      [ -n "${ANTHROPIC_BASE_URL:-}" ] && echo "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
      echo "CLAUDE_MODEL=${CLAUDE_MODEL}"
      echo "USER_DATA_PATH=${session}"
      echo "HYPERLOOM_RUN_MODE=${run_mode}"
      echo "FRAMEWORK=${backend}"
      echo "MODEL_PATH=${model_path}"
      echo "TARGET_GAIN=${TARGET_GAIN}"
      echo "DEMO_HOURS=${hours}"
    } > "$envf"
  )
  trap 'sed -i "/^ANTHROPIC_API_KEY=/d" "'"$envf"'" 2>/dev/null || true' EXIT

  # 3. pin the session dir so the poll finds it without guessing by timestamp (design §9)
  export INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR="${session}"
  echo "${session}" > "${session}/.session_dir"

  # 4. source env + drive setup, then demo, through the Agent CLI
  set -a
  # shellcheck disable=SC1090  # envf path is dynamic (per-leg)
  . "$envf"
  set +a
  export PYTHONPATH="${root}:${PYTHONPATH:-}"

  local setup_prompt="${PROMPTS_DIR}/setup-${run_mode}-${backend}.md"
  local demo_prompt;   demo_prompt="${PROMPTS_DIR}/demo-${hours}h.md"
  [ -f "$setup_prompt" ] || { log "ERROR: missing $setup_prompt"; return 1; }
  [ -f "$demo_prompt" ]  || { log "ERROR: missing $demo_prompt"; return 1; }

  log "claude --print (setup)"
  claude --print < "$setup_prompt"
  log "claude --print (demo ${hours}h)"
  claude --print < "$demo_prompt"
  log "leg $leg agent turns complete; poll will judge $session/reports/final.json"
}

# ---- docker host: fan out to nested containers -----------------------------
run_docker_host() {
  : "${DOCKER_LEGS:?}"; : "${DOCKER_GPU_MAP:?}"; : "${MODEL_3H:?}"; : "${MODEL_12H:?}"
  local runner="${SELF_DIR}/docker-run-hyperloom.sh"
  [ -x "$runner" ] || chmod +x "$runner" 2>/dev/null || true
  log "docker host: legs='${DOCKER_LEGS}'"
  local pids=()
  for leg in $DOCKER_LEGS; do
    local idx; idx="$(printf '%s' "$DOCKER_GPU_MAP" | jq -r --arg l "$leg" '.[$l]')"
    log "launch nested container: gpu=$idx leg=$leg"
    # Each nested container binds one card and runs THIS bootstrap inside, in
    # single-leg mode. docker-run-hyperloom.sh enforces GPU + cpu/mem quota (§8).
    "$runner" "$idx" "$leg" &
    pids+=("$!")
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return "$rc"
}

# ---- entry -----------------------------------------------------------------
install_claude_cli || exit 1

if [ "${E2E_DOCKER_HOST:-}" = "1" ]; then
  run_docker_host
else
  : "${LEG_ID:?}"; : "${HYPERLOOM_BACKEND:?}"; : "${HYPERLOOM_MODEL_PATH:?}"; : "${DEMO_HOURS:?}"
  run_leg "$LEG_ID" "$HYPERLOOM_BACKEND" "$HYPERLOOM_MODEL_PATH" "$DEMO_HOURS" "${HYPERLOOM_RUN_MODE:-baremetal}"
fi
