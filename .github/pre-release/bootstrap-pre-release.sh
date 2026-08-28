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
#   * unset  -> a single baremetal leg in THIS pod (LEG_ID given).
#   * "1"     -> the privileged 8-GPU host: run each of DOCKER_LEGS as a backgrounded
#                run_leg (docker mode) ON THE HOST POD. Each leg's agent follows the
#                demo skill to `docker run` its OWN single-GPU container (renderD =
#                128 + gpu_index*8), so the skill owns the container lifecycle. The
#                per-leg GPU-isolation values are computed here and injected via .env.
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
  #
  # For a docker leg the agent (not the harness) starts the single-GPU container per the
  # demo skill. The demo skill's literal `docker run` cannot express our per-card
  # isolation (single renderD node, NUMERIC device GIDs, cpu/mem caps, seccomp), so we
  # compute those HERE -- on the host pod, where /dev/kfd + /dev/dri/renderD* exist -- and
  # inject them as E2E_* .env values the setup prompt tells the agent to copy verbatim.
  # renderD = 128 + GPU_INDEX*8 (stride 8, VERIFIED on a real privileged MI355X x8 pod).
  # The pod /etc/group has no `video`/`render` NAMES, so numeric GIDs are required.
  local dk_rd="" dk_kfd_gid="" dk_dri_gid="" dk_nfs_mount=""
  if [ "$run_mode" = docker ]; then
    : "${GPU_INDEX:?docker leg needs GPU_INDEX}"
    dk_rd=$(( 128 + GPU_INDEX * 8 ))
    dk_kfd_gid="$(stat -c %g /dev/kfd 2>/dev/null || echo 0)"
    dk_dri_gid="$(stat -c %g "/dev/dri/renderD${dk_rd}" 2>/dev/null || stat -c %g /dev/dri 2>/dev/null || echo 0)"
    # NOTE: we deliberately do NOT inject HYPERLOOM_IMAGE. The demo skill owns the image
    # list (examples/*/SKILL.md "Suggested Docker images") -- duplicating those tags here
    # is what caused the wrong sglang tag. Instead the setup-docker-*.md prompt tells the
    # agent to pick from the skill's list by backend, and for sglang to detect the GPU
    # arch (gfx950 -> mi35x, gfx942 -> mi30x) via rocminfo and choose the matching tag.
    # The model path (e.g. /shared_nfs/models/<name>) is a SYMLINK into a DIFFERENT
    # /shared_nfs subtree (/shared_nfs/huggingface_models/...). Mounting only NFS_ROOT
    # (the CI subdir) or the model's parent leaves the symlink TARGET unmounted -> broken
    # symlink in the container -> optimize can't boot. Mount the whole shared-NFS root
    # (the top-level dir, e.g. /shared_nfs) so both the model and its symlink target
    # resolve at the same absolute path inside the container. Derive it as the first path
    # component of NFS_ROOT (override with E2E_SHARED_NFS_ROOT if the layout differs).
    dk_nfs_mount="${E2E_SHARED_NFS_ROOT:-/$(printf '%s' "${NFS_ROOT#/}" | cut -d/ -f1)}"
  fi
  local envf="${root}/.env"
  ( umask 077
    {
      echo "ANTHROPIC_API_KEY=$(printf '%s' "$ANTHROPIC_API_KEY_B64" | base64 -d)"
      [ -n "${ANTHROPIC_BASE_URL:-}" ] && echo "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
      # Gateways like AMD's APIM (llm-api.amd.com) reject the bearer key alone with
      # "401 Access denied due to missing subscription key" -- they need an
      # Ocp-Apim-Subscription-Key header whose value is the SAME key. The Claude CLI
      # reads ANTHROPIC_CUSTOM_HEADERS (newline-delimited "Name: value") and sends it
      # on every request; ${ANTHROPIC_API_KEY} is expanded in-pod so the one key
      # covers both bearer + subscription. The header NAME is fixed and non-secret and
      # its value carries no NEW secret (the key already arrives via *_B64), so we
      # default it here rather than requiring a repo variable -- one less "forgot to
      # configure it -> 401" trap. This matches Hyperloom's own llm_config.py contract
      # (parse_custom_headers expands ${VAR}). An externally-supplied
      # ANTHROPIC_CUSTOM_HEADERS still wins, for a gateway with a different convention.
      # Only emitted when a gateway base URL is set (direct api.anthropic.com needs no
      # subscription header). Double-quote in .env (space+colon unquoted -> exit 127).
      if [ -n "${ANTHROPIC_CUSTOM_HEADERS:-}" ]; then
        echo "ANTHROPIC_CUSTOM_HEADERS=\"${ANTHROPIC_CUSTOM_HEADERS}\""
      elif [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
        echo 'ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}"'
      fi
      echo "CLAUDE_MODEL=${CLAUDE_MODEL}"
      echo "USER_DATA_PATH=${session}"
      echo "HYPERLOOM_RUN_MODE=${run_mode}"
      echo "FRAMEWORK=${backend}"
      echo "MODEL_PATH=${model_path}"
      echo "TARGET_GAIN=${TARGET_GAIN}"
      echo "DEMO_HOURS=${hours}"
      # docker leg: hand the container lifecycle to the agent (demo skill) and carry the
      # CI hard constraints it must apply to its `docker run` (see setup-docker-*.md).
      if [ "$run_mode" = docker ]; then
        # HYPERLOOM_IMAGE intentionally NOT set: the agent selects it from the skill's
        # image list (by backend + detected GPU arch). See setup-docker-*.md.
        echo "HYPERLOOM_CONTAINER_NAME=hyperloom-${leg}"   # unique per leg (shared host dockerd)
        echo "HYPERLOOM_SHM_SIZE=${LEG_SHM:-64g}"
        echo "E2E_GPU_INDEX=${GPU_INDEX}"
        echo "E2E_RENDERD=${dk_rd}"
        echo "E2E_KFD_GID=${dk_kfd_gid}"
        echo "E2E_DRI_GID=${dk_dri_gid}"
        echo "E2E_LEG_CPUS=${LEG_CPUS:-32}"
        echo "E2E_LEG_MEM=${LEG_MEM:-128g}"
        echo "E2E_NFS_MOUNT=${dk_nfs_mount}"
      fi
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
  log "leg $leg demo turn returned; waiting for the background optimize to finish"

  # ---- wait for the backgrounded `optimize` to reach a terminal state --------
  # `claude --print` is ONE non-interactive turn: it returns right after the demo
  # skill backgrounds `optimize` (setsid nohup). If we returned now, run.sh would
  # exit 0 and SaFE would mark a FALSE "Succeeded" while the benchmark is still
  # running. Block here until the run writes reports/final.json (or a deadline).
  #
  # The real artifacts do NOT live under $session directly: make_session_dir()
  # creates a NESTED per-run dir  $session/<sanitized_model>/<UTC_ts>-<rand8>/
  # and re-pins INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to it -- but that re-pin
  # happens in the CLI's own process and never reaches us. So we discover the real
  # dir by globbing the newest one under $session that contains a state.json, and
  # re-point the poll's pin (.session_dir) at it.
  local wait_interval="${LEG_WAIT_INTERVAL_S:-45}"
  # Liveness, NOT a wall-clock startup budget. The prior fixed "optimize must produce a
  # state.json within N seconds" judged LIVE legs dead: baremetal SGLang builds from
  # source (gfx950, py3.12), starts a Ray head, then loads the server -- >15m before the
  # first state.json, and `optimize` isn't even a process yet during the build, so a pid
  # check can't tell "not launched yet" from "died". Instead we ask "how long since ANY
  # file under $session was last written?": USER_DATA_PATH=$session, so the build/runtime
  # logs, Ray output, and session artifacts all land under this tree and keep its mtime
  # fresh while anything is making progress. A leg is dead only if the tree has been
  # completely IDLE for stall_grace AND no state.json exists yet -- this waits out slow
  # builds (as long as they keep writing) but reaps a truly hung/exited launch quickly.
  local stall_grace="${LEG_STALL_GRACE_S:-600}"       # 10m of NO file writes -> dead
  local final_grace="${LEG_FINAL_GRACE_S:-120}"       # state stop_reason -> final.json
  local deadline_s=$(( hours * 3600 + 3600 ))         # demo budget + 1h margin (hard cap)
  local start_ts; start_ts="$(date +%s)"
  local real_sdir="" final_json="" state_json=""

  while :; do
    local now elapsed; now="$(date +%s)"; elapsed=$(( now - start_ts ))

    if [ -z "$real_sdir" ]; then
      # newest dir under $session that actually contains a state.json
      real_sdir="$(find "$session" -mindepth 2 -type f -name state.json -printf '%T@ %h\n' 2>/dev/null \
                   | sort -rn | head -n1 | cut -d' ' -f2- || true)"
      if [ -n "$real_sdir" ]; then
        final_json="${real_sdir}/reports/final.json"
        state_json="${real_sdir}/state.json"
        log "leg $leg real session dir: $real_sdir"
        # Re-pin so the poll (leg_session_dir -> head -n1 .session_dir) finds the report.
        echo "$real_sdir" > "${session}/.session_dir"
      else
        # No state.json yet -> judge liveness by how long since ANY file under $session
        # was written. newest mtime across the tree; if the tree is empty, fall back to
        # $session's own mtime so a brand-new leg isn't reaped on its first iteration.
        local last_write idle
        last_write="$(find "$session" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -n1)"
        [ -n "$last_write" ] || last_write="$(stat -c %Y "$session" 2>/dev/null || echo "$now")"
        idle=$(( now - ${last_write%.*} ))
        if [ "$idle" -ge "$stall_grace" ]; then
          log "ERROR: leg $leg -- no state.json and no file written under $session for ${idle}s (>= ${stall_grace}s stall; build/optimize hung or exited)"
          return 1
        fi
      fi
    fi

    if [ -n "$real_sdir" ]; then
      if [ -f "$final_json" ]; then
        log "leg $leg final.json present after ${elapsed}s; demo ran to completion"
        return 0
      fi
      local stop=""
      [ -f "$state_json" ] && stop="$(jq -r '.stop_reason // ""' "$state_json" 2>/dev/null || echo "")"
      if [ -n "$stop" ]; then
        log "leg $leg state.json stop_reason='$stop'; waiting up to ${final_grace}s for final.json"
        local g0; g0="$(date +%s)"
        while [ ! -f "$final_json" ] && [ $(( $(date +%s) - g0 )) -lt "$final_grace" ]; do sleep 5; done
        if [ -f "$final_json" ]; then
          log "leg $leg final.json present (stop_reason='$stop'); demo complete"
          return 0
        fi
        log "ERROR: leg $leg state stop_reason='$stop' but final.json never appeared within ${final_grace}s"
        return 1
      fi
    fi

    if [ "$elapsed" -ge "$deadline_s" ]; then
      log "ERROR: leg $leg deadline ${deadline_s}s reached without reports/final.json (real_sdir='${real_sdir:-<none>}')"
      return 1
    fi
    sleep "$wait_interval"
  done
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

# ---- docker host: run each leg (docker mode) ON THIS host pod ----------------
# The privileged 8-GPU host runs a dockerd, then drives each docker leg as a backgrounded
# run_leg in docker mode. Each leg's agent follows the demo skill to `docker run` its OWN
# single-GPU container (renderD = 128 + gpu_index*8), applying the CI isolation flags that
# run_leg injected into the leg .env. No nested bootstrap: session artifacts land under
# $session on this pod's NFS exactly as for baremetal, so the wait-for-final.json loop in
# run_leg works unchanged. Each `run_leg &` is its own subshell, so their per-leg EXIT
# traps (the .env key scrub) don't clobber each other.
run_docker_host() {
  : "${DOCKER_LEGS:?}"; : "${DOCKER_GPU_MAP:?}"; : "${MODEL_3H:?}"; : "${MODEL_12H:?}"
  ensure_dockerd || { log "ERROR: cannot provide docker on the host pod"; return 1; }
  log "docker host: legs='${DOCKER_LEGS}'"
  local pids=() leg idx backend hours model_path
  for leg in $DOCKER_LEGS; do
    idx="$(printf '%s' "$DOCKER_GPU_MAP" | jq -r --arg l "$leg" '.[$l]')"
    case "$leg" in
      *-vllm-*)   backend=vllm ;;
      *-sglang-*) backend=sglang ;;
      *) log "ERROR: cannot infer backend from leg '$leg'"; return 1 ;;
    esac
    case "$leg" in
      *-3h)  hours=3;  model_path="$MODEL_3H" ;;
      *-12h) hours=12; model_path="$MODEL_12H" ;;
      *) log "ERROR: cannot infer duration from leg '$leg'"; return 1 ;;
    esac
    log "launch docker leg: gpu=$idx renderD$(( 128 + idx * 8 )) leg=$leg backend=$backend hours=$hours"
    ( GPU_INDEX="$idx" run_leg "$leg" "$backend" "$model_path" "$hours" docker ) &
    pids+=("$!")
  done
  local rc=0 p
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
