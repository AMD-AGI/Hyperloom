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

# Seconds a leg has been idle: time since the newest write anywhere under the leg root,
# floored at the wait loop's start so the preceding `claude --print` turns (which can
# leave the tree untouched for longer than the stall grace) are never charged to the leg.
# Args: leg_root loop_start_epoch now_epoch
leg_idle_s() {
  local leg_root="$1" loop_start="$2" now="$3" last since
  last="$(find "$leg_root" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -n1)"
  since="${last%.*}"
  [ -n "$since" ] && [ "$since" -ge "$loop_start" ] 2>/dev/null || since="$loop_start"
  echo $(( now - since ))
}

# 0 once the leg has a live run: `optimize` creates a NESTED per-run dir under the
# session and writes state.json into it. Scoped to this leg's own session tree, so it
# stays correct on the shared docker host where four legs run side by side.
# Args: session_dir
leg_run_started() {
  [ -n "$(find "$1" -mindepth 2 -type f -name state.json -print -quit 2>/dev/null)" ]
}

# A stable per-leg conversation id, so a follow-up turn can --resume the SAME session
# instead of starting from scratch. `--session-id` wants a UUID, so shape a sha1 of
# leg+CI_VERSION into one (version nibble 4, variant nibble 8).
# Args: leg ci_version
leg_session_uuid() {
  local h; h="$(printf '%s|%s' "$1" "$2" | sha1sum | cut -c1-32)"
  printf '%s-%s-4%s-8%s-%s' "${h:0:8}" "${h:8:4}" "${h:12:3}" "${h:15:3}" "${h:18:12}"
}

# One agent turn, mirrored to the leg's NFS transcript so it survives pod deletion.
# Args: agent_log [claude flags...]   (the prompt/nudge arrives on stdin)
agent_turn() {
  local alog="$1"; shift
  claude --print --dangerously-skip-permissions "$@" 2>&1 | tee -a "$alog"
}

# Follow-up nudges for a conversation that ended before finishing. Deliberately short:
# the resumed session still holds the context (what was launched, which log is being
# watched), so restating the whole prompt would only add noise -- and re-feeding it as a
# FRESH turn is worse, because the agent then has to rediscover all of that.
SETUP_RESUME_NUDGE='Your previous turn ended before setup finished. Do NOT restart anything that is already running. Check on the install you launched: if it is still in progress, keep monitoring it and only answer once it has finished. Once it has finished successfully, complete any remaining setup steps and then reply with exactly the completion line the setup instructions asked for. If it has failed, report the failure.'

DEMO_RESUME_NUDGE='Your previous turn ended without leaving a running optimize behind, and nothing has been written under the workspace since, so the work is not progressing. Do NOT fabricate a result. Finish the launch in THIS turn: complete the install if it is still needed, start optimize detached with setsid nohup so it survives the end of this turn, then confirm the nested session run dir and its state.json exist and report their paths.'

# Clean terminal stop_reason values (hyperloom.inference_optimizer.cli._SUCCESS_STOP_REASONS).
is_clean_stop_reason() {
  case "$1" in
    target_reached|global_converged|time_exhausted|max_ticks|sweep_done|conc_sweep_done)
      return 0 ;;
    *) return 1 ;;
  esac
}

# Optimize writes state.json root-only; the poll runner reads as ubuntu.
# Pin the session dir for the poll, stamped with this run's tag (dispatch passes RUN_TAG).
# On a reused CI_VERSION the previous run's pin is still on disk pointing at a finished
# session; the tag is how the poll tells that leftover from ours. Args: dir session
pin_session_dir() {
  { printf '%s\n' "$1"; printf '%s\n' "${RUN_TAG:-}"; } > "${2}/.session_dir"
}

publish_state_for_poll() {
  local state_json="$1" sdir="$2" session="$3"
  [ -f "$state_json" ] || return 0
  chmod a+r "$state_json" 2>/dev/null || true
  chmod a+X "$sdir" "$session" 2>/dev/null || true
}

# Run ONE leg to completion inside the current filesystem (baremetal pod, or already
# inside a nested docker container). Args: leg backend model_path hours run_mode
run_leg() {
  local leg="$1" backend="$2" model_path="$3" hours="$4" run_mode="$5"
  local root="${NFS_ROOT%/}/runs/${CI_VERSION}/${leg}"
  local session="${root}/session"
  mkdir -p "$root" "$session"
  # A reused CI_VERSION (workflow_dispatch reuse_ci_version, or a job re-run) lands on the
  # same paths, so the previous run's artifacts are still here. Nothing older than leg_t0
  # belongs to this run; the artifacts are kept for post-mortem, never trusted as ours.
  local leg_t0; leg_t0="$(date +%s)"
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
        # image list (by backend + detected GPU arch). See setup-docker-*.md. To pin the
        # tag to the SKILL's exact string (no freelancing), the setup prompt greps it out
        # of the demo skill's SKILL.md -- so hand the agent that file's absolute path.
        # `pip install --target "$root"` materializes the example demo skills (declared as
        # [tool.setuptools.data-files] in pyproject.toml) at
        #   $root/.claude/skills/<demo-skill-name>/SKILL.md
        # (empirically verified). The demo skill is chosen by leg duration:
        #   *-3h  -> hyperloom-qwen3-8b-3h ; *-12h -> hyperloom-qwen3-14b-fp8-12h.
        local demo_skill=""
        case "$leg" in
          *-3h)  demo_skill="hyperloom-qwen3-8b-3h" ;;
          *-12h) demo_skill="hyperloom-qwen3-14b-fp8-12h" ;;
        esac
        echo "HYPERLOOM_SKILL_PATH=${root}/.claude/skills/${demo_skill}/SKILL.md"
        echo "HYPERLOOM_CONTAINER_NAME=hyperloom-${leg}"   # unique per leg (shared host dockerd)
        local leg_mem leg_shm
        case "$leg" in
          *-3h)
            leg_mem="${DOCKER_LEG_MEM_3H:-256g}"
            leg_shm="${DOCKER_LEG_SHM_3H:-64g}"
            ;;
          *-12h)
            leg_mem="${DOCKER_LEG_MEM_12H:-512g}"
            leg_shm="${DOCKER_LEG_SHM_12H:-64g}"
            ;;
          *)
            leg_mem="${LEG_MEM:-256g}"
            leg_shm="${LEG_SHM:-64g}"
            ;;
        esac
        echo "HYPERLOOM_SHM_SIZE=${leg_shm}"
        echo "E2E_GPU_INDEX=${GPU_INDEX}"
        echo "E2E_RENDERD=${dk_rd}"
        echo "E2E_KFD_GID=${dk_kfd_gid}"
        echo "E2E_DRI_GID=${dk_dri_gid}"
        echo "E2E_LEG_CPUS=${LEG_CPUS:-32}"
        echo "E2E_LEG_MEM=${leg_mem}"
        echo "E2E_NFS_MOUNT=${dk_nfs_mount}"
      fi
    } > "$envf"
  )
  trap 'sed -i "/^ANTHROPIC_API_KEY=/d" "'"$envf"'" 2>/dev/null || true' EXIT

  # 3. pin the session dir so the poll finds it without guessing by timestamp (design §9)
  export INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR="${session}"
  pin_session_dir "${session}" "${session}"

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
  # Mirror both turns onto NFS. SaFE deletes the PyTorchJob as soon as a leg fails, taking
  # the pod's stdout with it, so without this the agent's own account of the failure is
  # unrecoverable and a post-mortem is left reconstructing events from file mtimes.
  local agent_log="${session}/agent-${leg}.log"
  # Rotate rather than append: the setup loop below decides it is done by grepping this
  # file for the completion marker, and a previous run's marker would satisfy it on turn 1
  # with nothing installed. Rotating keeps the old transcript for post-mortem.
  if [ -f "$agent_log" ]; then
    mv "$agent_log" "${agent_log%.log}.prev-$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null || true
  fi
  local uuid; uuid="$(leg_session_uuid "$leg" "$CI_VERSION")"
  # `claude --print` is "print response and exit": ONE answer per invocation. Setup here
  # means installing a framework layer -- a vLLM ROCm wheel into an isolated venv, or
  # SGLang compiled from source for gfx950 -- which takes 10-30min, far longer than a
  # single turn can hold. Every leg's agent therefore backgrounds the install and polls
  # its log, and eventually answers with a progress note ("Phase 2 is installing the
  # isolated vLLM env. Still running.") which ENDS the turn with setup incomplete.
  #
  # The install itself is detached and keeps running (verified: setup_vllm.log grew from
  # 71KB to 77KB across a turn boundary), so the fix is to give the agent another turn --
  # RESUMING the same conversation, so it still knows what it started and where the log
  # is. Budget it in TIME, not turns: an earlier count-based cap of 3 turns amounted to
  # ~4min of wall clock and killed two legs whose installs were demonstrably still
  # progressing. A leg only fails here if the tree goes quiet (nothing installing) or the
  # whole setup deadline elapses.
  local setup_deadline_s="${LEG_SETUP_DEADLINE_S:-2700}"   # 45m of setup, then give up
  local setup_stall_s="${LEG_SETUP_STALL_S:-600}"          # 10m with no writes -> dead
  local setup_max_turns="${LEG_SETUP_MAX_TURNS:-30}"       # token-spend backstop
  local setup_t0; setup_t0="$(date +%s)"
  local turn=0 snow sidle
  while :; do
    turn=$(( turn + 1 ))
    if [ "$turn" = 1 ]; then
      log "claude --print (setup, turn 1, session $uuid); agent transcript -> $agent_log"
      agent_turn "$agent_log" --session-id "$uuid" < "$setup_prompt"
    else
      log "claude --print (setup, turn $turn, resuming session $uuid)"
      if ! printf '%s\n' "$SETUP_RESUME_NUDGE" | agent_turn "$agent_log" --resume "$uuid"; then
        log "WARN: leg $leg -- could not resume session $uuid; re-feeding the full setup prompt"
        agent_turn "$agent_log" < "$setup_prompt" || true
      fi
    fi
    if grep -qiE "setup complete: ${run_mode}/${backend}" "$agent_log"; then
      log "leg $leg setup reported complete on turn $turn"
      break
    fi
    snow="$(date +%s)"
    sidle="$(leg_idle_s "$root" "$setup_t0" "$snow")"
    if [ "$sidle" -ge "$setup_stall_s" ]; then
      log "ERROR: leg $leg -- setup turn $turn ended early and nothing was written under $root for ${sidle}s; the install is not progressing"
      return 1
    fi
    if [ $(( snow - setup_t0 )) -ge "$setup_deadline_s" ]; then
      log "ERROR: leg $leg -- setup never reported 'setup complete: ${run_mode}/${backend}' within ${setup_deadline_s}s ($turn turns)"
      return 1
    fi
    if [ "$turn" -ge "$setup_max_turns" ]; then
      log "ERROR: leg $leg -- setup still incomplete after $setup_max_turns turns; giving up"
      return 1
    fi
    log "WARN: leg $leg -- setup turn $turn ended early (install still progressing, idle ${sidle}s); resuming the conversation"
    sleep "${LEG_TURN_GAP_S:-30}"
  done
  log "claude --print (demo ${hours}h, resuming session $uuid)"
  # Same conversation as setup: the agent already knows this workspace, which framework
  # got installed and where its logs are, exactly like a human continuing the same chat.
  if ! agent_turn "$agent_log" --resume "$uuid" < "$demo_prompt"; then
    log "WARN: leg $leg -- could not resume session $uuid for the demo; running it standalone"
    agent_turn "$agent_log" < "$demo_prompt"
  fi
  log "leg $leg demo turn returned; waiting for the background optimize to finish"

  # ---- wait for the backgrounded `optimize` to reach a terminal state --------
  # `claude --print` is ONE non-interactive turn: it returns right after the demo
  # skill backgrounds `optimize` (setsid nohup). If we returned now, run.sh would
  # exit 0 and SaFE would mark a FALSE "Succeeded" while the benchmark is still
  # running. Block here until state.json carries a terminal stop_reason (or a deadline).
  #
  # The real artifacts do NOT live under $session directly: make_session_dir()
  # creates a NESTED per-run dir  $session/<sanitized_model>/<UTC_ts>-<rand8>/
  # and re-pins INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to it -- but that re-pin
  # happens in the CLI's own process and never reaches us. So we discover the real
  # dir by globbing the newest one under $session that contains a state.json, and
  # re-point the poll's pin (.session_dir) at it.
  local wait_interval="${LEG_WAIT_INTERVAL_S:-45}"
  # Liveness, NOT a wall-clock startup budget: a leg is dead only once NOTHING under the
  # leg root has been written for stall_grace and no state.json exists yet. This waits out
  # arbitrarily slow builds (baremetal SGLang compiles from source and starts a Ray head,
  # >15m before the first state.json) while still reaping a hung launch within ~10m.
  # Two properties matter, both learned from legs that were killed while demonstrably
  # alive (2026-08-28):
  #   * scope is $root, not $session -- the agent's launcher and its setup/install logs
  #     land next to the workspace, and install.sh writes $root/.cache. Watching only
  #     $session missed all of it and reaped a leg 26s after it last wrote a file.
  #   * idleness is measured from the LATER of (last write, loop start) -- the two
  #     `claude --print` turns can leave the tree untouched for longer than stall_grace,
  #     and that pre-loop gap must not be charged to the leg, or the very first
  #     iteration condemns it.
  local stall_grace="${LEG_STALL_GRACE_S:-600}"       # 10m of NO file writes -> dead
  local deadline_s=$(( hours * 3600 + 3600 ))         # demo budget + 1h margin (hard cap)
  # Two clocks: start_ts backs the hard deadline and is NEVER reset (resetting it would
  # let the leg outlive the SaFE pod timeout and lose the clean failure path); grace_ts
  # backs the stall window and restarts after each re-driven turn.
  local start_ts; start_ts="$(date +%s)"
  local grace_ts="$start_ts"
  local real_sdir="" final_json="" state_json=""
  # An idle tree with no state.json means the demo turn ended without leaving a running
  # `optimize` behind. That is recoverable -- ask the agent to finish the job rather than
  # failing the leg -- but only a bounded number of times. Deliberately driven off the
  # stall signal instead of an "is it launched yet" probe right after the turn: the demo
  # skill's launcher runs install.sh BEFORE backgrounding optimize, so state.json can
  # legitimately be 10+ minutes away, and re-driving on a short grace would double-launch.
  local demo_redrives=0 max_demo_redrives="${LEG_DEMO_REDRIVES:-5}"

  while :; do
    local now elapsed; now="$(date +%s)"; elapsed=$(( now - start_ts ))

    if [ -z "$real_sdir" ]; then
      # newest dir under $session that actually contains a state.json
      # -newermt leg_t0 excludes a previous run's nested dirs, which sit under the same
      # $session and would otherwise win "newest" and end this wait with their state.json.
      real_sdir="$(find "$session" -mindepth 2 -type f -name state.json -newermt "@$leg_t0" -printf '%T@ %h\n' 2>/dev/null \
                   | sort -rn | head -n1 | cut -d' ' -f2- || true)"
      if [ -n "$real_sdir" ]; then
        final_json="${real_sdir}/reports/final.json"
        state_json="${real_sdir}/state.json"
        log "leg $leg real session dir: $real_sdir"
        # Re-pin so the poll (leg_session_dir -> head -n1 .session_dir) finds the report.
        pin_session_dir "$real_sdir" "$session"
        publish_state_for_poll "$state_json" "$real_sdir" "$session"
      else
        local idle
        idle="$(leg_idle_s "$root" "$grace_ts" "$now")"
        if [ "$idle" -ge "$stall_grace" ]; then
          if [ "$demo_redrives" -lt "$max_demo_redrives" ]; then
            demo_redrives=$(( demo_redrives + 1 ))
            log "WARN: leg $leg -- idle ${idle}s with no state.json; the demo turn left nothing running. Resuming the conversation to finish the launch ($demo_redrives/$max_demo_redrives)"
            if ! printf '%s\n' "$DEMO_RESUME_NUDGE" | agent_turn "$agent_log" --resume "$uuid"; then
              log "WARN: leg $leg -- could not resume session $uuid; re-feeding the demo prompt"
              agent_turn "$agent_log" < "$demo_prompt" || true
            fi
            log "leg $leg demo re-drive $demo_redrives returned"
            grace_ts="$(date +%s)"   # fresh stall grace for the new turn; deadline unchanged
            continue
          fi
          log "ERROR: leg $leg -- no state.json and no file written under $root for ${idle}s after $demo_redrives demo re-drive(s); giving up"
          return 1
        fi
      fi
    fi

    if [ -n "$real_sdir" ]; then
      if [ -f "$state_json" ]; then
        publish_state_for_poll "$state_json" "$real_sdir" "$session"
      fi
      if [ -f "$final_json" ]; then
        log "leg $leg final.json present after ${elapsed}s; demo complete"
        return 0
      fi
      local stop=""
      [ -f "$state_json" ] && stop="$(jq -r '.stop_reason // ""' "$state_json" 2>/dev/null || echo "")"
      if [ -n "$stop" ]; then
        if is_clean_stop_reason "$stop"; then
          log "leg $leg state.json stop_reason='$stop' after ${elapsed}s; demo complete"
          return 0
        fi
        log "ERROR: leg $leg state stop_reason='$stop' (not a clean terminal exit)"
        return 1
      fi
    fi

    if [ "$elapsed" -ge "$deadline_s" ]; then
      log "ERROR: leg $leg deadline ${deadline_s}s reached without a terminal stop_reason (real_sdir='${real_sdir:-<none>}')"
      return 1
    fi
    sleep "$wait_interval"
  done
}

# Layer-deduplicating storage drivers, tried in order. vfs is deliberately NOT a
# fallback: it copies every image layer and every container in full, which grew the
# host pod past its ephemeralStorage limit and got it EVICTED within minutes
# ("ephemeral local storage usage exceeds the total limit of containers 1792Gi",
# observed live 2026-08-28). Running on vfs is a guaranteed eviction, so a leg that
# cannot get a deduplicating driver must fail loudly instead.
DOCKER_DRIVERS="${DOCKER_DRIVERS:-overlay2 fuse-overlayfs}"

# Start a detached dockerd on one driver; 0 when the socket answers. Cleans up the
# failed daemon so the next driver starts from a clean socket.
start_dockerd_with_driver() {
  local driver="$1" data_root="$2" dlog="/var/log/dockerd-${1}.log" i
  mkdir -p "$data_root" || return 1
  log "starting pod-local dockerd (driver=$driver, data-root=$data_root)"
  setsid bash -c "dockerd --host=unix:///var/run/docker.sock --storage-driver='$driver' --data-root='$data_root' >'$dlog' 2>&1" \
    </dev/null >/dev/null 2>&1 &
  for i in $(seq 1 60); do
    docker info >/dev/null 2>&1 && { log "dockerd up after ${i}s (driver=$driver)"; return 0; }
    sleep 1
  done
  log "WARN: dockerd did not become ready with driver=$driver"
  tail -20 "$dlog" 2>/dev/null || true
  pkill -f 'dockerd --host=unix:///var/run/docker.sock' 2>/dev/null || true
  sleep 3
  rm -f /var/run/docker.sock 2>/dev/null || true
  return 1
}

# fuse-overlayfs is the userspace fallback when the kernel refuses overlay2 on the
# data-root filesystem. Needs the binary plus /dev/fuse.
ensure_fuse_overlayfs() {
  command -v fuse-overlayfs >/dev/null 2>&1 || {
    log "installing fuse-overlayfs"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >>/tmp/apt-docker.log 2>&1 || true
    apt-get install -y -qq fuse-overlayfs >>/tmp/apt-docker.log 2>&1 \
      || { log "WARN: fuse-overlayfs install failed"; return 1; }
  }
  [ -c /dev/fuse ] || { log "WARN: /dev/fuse missing; fuse-overlayfs unusable"; return 1; }
}

# Install docker + start a pod-local dockerd. VERIFIED on a real privileged MI355X pod:
# the Authoring base image ships NO docker/dockerd/docker.sock, but the pod has full
# capabilities (CapEff=0x1ffffffffff), so a self-hosted dockerd works. There is no
# systemd in the pod, hence setsid + socket polling rather than a service start.
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
  # --- data-root must sit on a filesystem that can back a deduplicating driver ---
  # The container rootfs (/) is itself an overlay, and overlayfs cannot use an overlay
  # upperdir -- that is why this used to run on vfs. /shared-data is a plain xfs mount
  # (/dev/mapper/nvme_vg-nvme_lv) and DOES accept an overlay upperdir (probe-verified:
  # `mount -t overlay` with upperdir/workdir under /shared-data succeeds, so xfs ftype=1
  # holds). It is an emptyDir, so it still counts toward the pod's ephemeralStorage quota
  # and dies with the pod -- the quota headroom comes from the driver's layer dedup, not
  # from the location.
  local docker_data_root="/var/lib/docker"
  local dr_candidate="/shared-data/e2e-docker/${CI_VERSION:-current}"
  if mkdir -p "$dr_candidate" 2>/dev/null && touch "$dr_candidate/.wtest" 2>/dev/null; then
    rm -f "$dr_candidate/.wtest" 2>/dev/null || true
    docker_data_root="$dr_candidate"
  else
    log "WARN: /shared-data not writable; dockerd falls back to $docker_data_root (overlay-on-overlay, overlay2 will likely be refused)"
  fi
  # Each driver gets its own subdir: dockerd refuses a data-root that already holds a
  # different driver's tree.
  local driver
  for driver in $DOCKER_DRIVERS; do
    if [ "$driver" = fuse-overlayfs ]; then ensure_fuse_overlayfs || continue; fi
    if start_dockerd_with_driver "$driver" "${docker_data_root}/${driver}"; then
      log "docker storage driver in use: $(docker info -f '{{.Driver}}' 2>/dev/null)"
      return 0
    fi
  done
  log "ERROR: no layer-deduplicating docker storage driver available (tried: $DOCKER_DRIVERS)."
  log "ERROR: refusing to fall back to vfs -- it has no layer dedup and evicts the pod on ephemeralStorage."
  return 1
}

# ---- docker host: run each leg (docker mode) ON THIS host pod ----------------
# The privileged 8-GPU host runs a dockerd, then drives each docker leg as a backgrounded
# run_leg in docker mode. Each leg's agent follows the demo skill to `docker run` its OWN
# single-GPU container (renderD = 128 + gpu_index*8), applying the CI isolation flags that
# run_leg injected into the leg .env. No nested bootstrap: session artifacts land under
# $session on this pod's NFS exactly as for baremetal, so the wait-for-stop_reason loop in
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
