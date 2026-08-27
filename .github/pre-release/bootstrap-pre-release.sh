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

# The SaFE rocm/pytorch Authoring image ships NO node/npm (verified 2026-08-27 on a
# live pod), so the Claude CLI's `npm install -g` fails with "npm: command not found".
# Install a pinned Node.js LTS from the official binary tarball (both nodejs.org and the
# npm registry are reachable from the pod) into /opt/node and expose it on PATH. Idempotent.
NODE_VERSION="${NODE_VERSION:-v20.18.0}"
NODE_PREFIX="${NODE_PREFIX:-/opt/node}"
ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    log "node present: $(node --version 2>/dev/null) npm $(npm --version 2>/dev/null)"; return 0
  fi
  log "installing Node ${NODE_VERSION} (no node/npm in the Authoring base image)"
  mkdir -p "$NODE_PREFIX"
  curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ -C "$NODE_PREFIX" --strip-components=1 \
    || { log "ERROR: Node download/extract failed"; return 1; }
  export PATH="${NODE_PREFIX}/bin:${PATH}"
  ln -sf "${NODE_PREFIX}/bin/node" /usr/local/bin/node 2>/dev/null || true
  ln -sf "${NODE_PREFIX}/bin/npm"  /usr/local/bin/npm  2>/dev/null || true
  command -v node >/dev/null 2>&1 || { log "ERROR: node still missing after install"; return 1; }
  log "node installed: $(node --version) npm $(npm --version)"
}

install_claude_cli() {
  if command -v claude >/dev/null 2>&1; then log "claude CLI present: $(claude --version 2>/dev/null || true)"; return; fi
  ensure_node || return 1
  log "installing Claude CLI @ ${CLAUDE_CLI_VERSION}"
  # Pinned install. The exact channel is environment-specific; keep the version in one
  # place (CLAUDE_CLI_VERSION) so the pin is auditable. Surface npm's error (don't
  # swallow) so a failure is diagnosable in bootstrap.log.
  npm install -g "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}" >/tmp/npm-claude.log 2>&1 \
    || { log "ERROR: claude CLI install failed"; tail -20 /tmp/npm-claude.log; return 1; }
  # npm's global bin may be under the Node prefix, not on the default PATH.
  ln -sf "${NODE_PREFIX}/bin/claude" /usr/local/bin/claude 2>/dev/null || true
  command -v claude >/dev/null 2>&1 || { log "ERROR: claude not on PATH after install"; return 1; }
  log "claude CLI installed: $(claude --version 2>/dev/null || true)"
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
      # Gateways like AMD's APIM (llm-api.amd.com) reject the bearer key alone with
      # "401 Access denied due to missing subscription key" -- they need an
      # Ocp-Apim-Subscription-Key header. The Claude CLI reads ANTHROPIC_CUSTOM_HEADERS
      # (newline-delimited "Name: value") and sends it on every request; ${VAR} is
      # expanded so the one key covers both the bearer and the subscription header.
      # Double-quote in .env (space+colon in an unquoted value is parsed as a command
      # on source -> exit 127). See docs/reference/authentication.md "extra headers".
      [ -n "${ANTHROPIC_CUSTOM_HEADERS:-}" ] && echo "ANTHROPIC_CUSTOM_HEADERS=\"${ANTHROPIC_CUSTOM_HEADERS}\""
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

  # Run the agent FROM the leg root: the setup prompt refers to "the current
  # workspace (REPO_ROOT)" for .env and the installed tree. Bootstrap otherwise runs
  # from '/', where claude finds no .env and refuses (it sees ~29 unrelated
  # workspaces under /shared_nfs and blocks rather than guess). cd fixes the cwd.
  cd "$root"
  # --dangerously-skip-permissions: `claude --print` is non-interactive, so the default
  # permission mode is fail-closed -- setup's pip/shell steps have no approver and die.
  # This is an isolated per-leg CI pod (same posture as the per-PR CI's _incontainer.sh),
  # so bypassing the approval gate is acceptable and required for unattended setup/demo.
  # IS_SANDBOX=1: the SaFE pod runs as root, and claude refuses to skip permissions under
  # root unless IS_SANDBOX=1 (SWSPLAT-42390) -- Hyperloom's own kernel-agent sets the same.
  export IS_SANDBOX=1
  log "claude --print (setup)"
  claude --print --dangerously-skip-permissions < "$setup_prompt"
  log "claude --print (demo ${hours}h)"
  claude --print --dangerously-skip-permissions < "$demo_prompt"
  log "leg $leg agent turns complete; poll will judge $session/reports/final.json"
}

# Install docker + start a pod-local dockerd. VERIFIED on a real privileged MI355X pod
# (2026-08-27): the Authoring base image ships NO docker/dockerd/docker.sock, but the
# pod has full capabilities (CapEff=0x1ffffffffff), so a self-hosted dockerd works.
# Two non-obvious requirements, both confirmed by probing:
#   * --storage-driver=vfs  -- overlayfs-on-overlayfs fails inside the container rootfs.
#   * no systemd in the pod -- start dockerd detached via setsid, then poll the socket.
ensure_dockerd() {
  if docker info >/dev/null 2>&1; then log "dockerd already up"; return 0; fi
  if ! command -v dockerd >/dev/null 2>&1; then
    log "installing docker.io (no docker in the Authoring base image)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq  >/tmp/apt-docker.log 2>&1 || { log "ERROR: apt-get update failed"; tail -20 /tmp/apt-docker.log; return 1; }
    apt-get install -y -qq docker.io >>/tmp/apt-docker.log 2>&1 \
      || { log "ERROR: docker.io install failed"; tail -30 /tmp/apt-docker.log; return 1; }
    log "docker installed: $(docker --version 2>&1)"
  fi
  log "starting pod-local dockerd (vfs, detached via setsid)"
  setsid bash -c 'dockerd --host=unix:///var/run/docker.sock --storage-driver=vfs >/var/log/dockerd.log 2>&1' \
    </dev/null >/dev/null 2>&1 &
  local i
  for i in $(seq 1 60); do
    docker info >/dev/null 2>&1 && { log "dockerd up after ${i}s"; return 0; }
    sleep 1
  done
  log "ERROR: dockerd did not become ready"; tail -40 /var/log/dockerd.log 2>/dev/null || true
  return 1
}

# ---- docker host: fan out to nested containers -----------------------------
run_docker_host() {
  : "${DOCKER_LEGS:?}"; : "${DOCKER_GPU_MAP:?}"; : "${MODEL_3H:?}"; : "${MODEL_12H:?}"
  local runner="${SELF_DIR}/docker-run-hyperloom.sh"
  [ -x "$runner" ] || chmod +x "$runner" 2>/dev/null || true
  ensure_dockerd || { log "ERROR: cannot provide docker on the host pod"; return 1; }
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

# The SaFE rocm/pytorch Authoring image is minimal: besides node/npm (see ensure_node)
# it also lacks `jq`, which run_docker_host uses to read DOCKER_GPU_MAP. Install it up
# front (apt works in the pod) so both the host pod and the nested single-leg containers
# -- which re-run THIS script -- have it. Idempotent; cheap when already present.
ensure_base_tools() {
  command -v jq >/dev/null 2>&1 && return 0
  log "installing jq (not in the Authoring base image)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/tmp/apt-jq.log 2>&1 && apt-get install -y -qq jq >>/tmp/apt-jq.log 2>&1 \
    || { log "ERROR: jq install failed"; tail -20 /tmp/apt-jq.log; return 1; }
  log "jq installed: $(jq --version 2>&1)"
}

# ---- entry -----------------------------------------------------------------
ensure_base_tools || exit 1
install_claude_cli || exit 1

if [ "${E2E_DOCKER_HOST:-}" = "1" ]; then
  run_docker_host
else
  : "${LEG_ID:?}"; : "${HYPERLOOM_BACKEND:?}"; : "${HYPERLOOM_MODEL_PATH:?}"; : "${DEMO_HOURS:?}"
  run_leg "$LEG_ID" "$HYPERLOOM_BACKEND" "$HYPERLOOM_MODEL_PATH" "$DEMO_HOURS" "${HYPERLOOM_RUN_MODE:-baremetal}"
fi
