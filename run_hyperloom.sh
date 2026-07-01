#!/usr/bin/env bash
# run_hyperloom.sh — start / exec / stop the Hyperloom vLLM VL container.
#
# Usage:
#   ./run_hyperloom.sh                         # start (or exec if already running)
#   ./run_hyperloom.sh --start                 # same as default
#   ./run_hyperloom.sh --stop                  # stop and remove the container
#   ./run_hyperloom.sh --mount ~/workspace/ml-perf   # extra read-only mount
#   ./run_hyperloom.sh --api-port 12345        # override auto-detected API port
#
# AMD LLM API port:
#   Auto-detected from $ANTHROPIC_BASE_URL on the host (e.g. http://127.0.0.1:41829/Anthropic).
#   Falls back to 9443 only if ANTHROPIC_BASE_URL is not set.
#   Override with --api-port if needed.
#   The container uses --network host so it shares the host's 127.0.0.1 directly.
#
# Run from the worktree root:
#   cd ~/workspace/Hyperloom-feat-vl-model-support && ./run_hyperloom.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="hyperloom-vl-vllm-local-${USER}"
CONTAINER="hyperloom-vl-vllm-local-${USER}"
EXTRA_MOUNTS=()   # populated by --mount flags
# Auto-detect AMD API port from ANTHROPIC_BASE_URL on the host (e.g. http://127.0.0.1:41829/Anthropic).
# Falls back to 9443 (standard SSH RemoteForward). Override with --api-port.
_detect_api_port() {
  if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
    local port; port="$(echo "$ANTHROPIC_BASE_URL" | grep -oP '(?<=:)\d+(?=/)' | head -1)"
    [[ -n "$port" ]] && echo "$port" && return
  fi
  echo "9443"
}
AMD_API_PORT="$(_detect_api_port)"

# ── Credentials ──────────────────────────────────────────────────────────────
# Load from .env in the repo root if not already exported.
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    line="${line#export }"
    key="${line%%=*}"
    val="${line#*=}"
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    [[ -z "${!key:-}" ]] && export "$key=$val"
  done < "${SCRIPT_DIR}/.env"
fi

SAFE_API_KEY="${SAFE_API_KEY:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1}"
ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { echo "[run_hyperloom] ERROR: $*" >&2; exit 1; }
log()  { echo "[run_hyperloom] $*"; }

container_running() { docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; }
container_exists()  { docker inspect "$CONTAINER" &>/dev/null; }
image_exists()      { docker image inspect "$IMAGE" &>/dev/null; }

exec_into() {
  log "exec into $CONTAINER ..."
  # Ensure llm-api.amd.com resolves to 127.0.0.1 so claude reaches the host tunnel.
  docker exec "$CONTAINER" bash -c \
    "grep -q 'llm-api.amd.com' /etc/hosts || echo '127.0.0.1 llm-api.amd.com' >> /etc/hosts" \
    2>/dev/null || true
  docker exec -it "$CONTAINER" bash
}

build_image() {
  log "image $IMAGE not found — building ..."
  [[ -n "${SSH_AUTH_SOCK:-}" ]] || die "SSH_AUTH_SOCK is not set. Run: ssh-add \$HOME/.ssh/id_amd"
  DOCKER_BUILDKIT=1 docker build --ssh default \
    -t "$IMAGE" \
    "$SCRIPT_DIR"
}

start_container() {
  [[ -n "$SAFE_API_KEY" ]]            || die "SAFE_API_KEY is not set. Export it or add it to .env"
  [[ -n "${SSH_AUTH_SOCK:-}" ]]       || die "SSH_AUTH_SOCK is not set. Run: ssh-add \$HOME/.ssh/id_amd"

  # Verify the AMD API tunnel is reachable on the requested port before starting.
  if ss -tlnp 2>/dev/null | grep -q ":${AMD_API_PORT}"; then
    log "AMD LLM API tunnel detected on port ${AMD_API_PORT}"
  else
    log "WARNING: nothing listening on port ${AMD_API_PORT} — claude may not reach the AMD LLM API"
    log "         Start the tunnel: ssh RemoteForward ${AMD_API_PORT} llm-api.amd.com:443"
    log "         Or for amd-llm-proxy: systemctl --user start amd-llm-proxy"
  fi

  # Use host ANTHROPIC_BASE_URL verbatim if set (already has the right port/scheme).
  # Otherwise fall back to the standard llm-api.amd.com form on the detected port.
  local anthropic_base_url="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${AMD_API_PORT}/Anthropic}"
  log "ANTHROPIC_BASE_URL -> ${anthropic_base_url} (AMD API port: ${AMD_API_PORT})"
  log "starting container $CONTAINER ..."

  docker run -d \
    --name "$CONTAINER" \
    --group-add video \
    --group-add render \
    --network host \
    --shm-size 64g \
    --device /dev/kfd \
    --device /dev/dri \
    -v "${SCRIPT_DIR}:/workspace/Hyperloom" \
    -v "/data2/hf_hub_cache:/models" \
    -v "${HOME}/.claude:/root/.claude" \
    -v "${HOME}/.claude.json:/root/.claude.json" \
    -v "${SSH_AUTH_SOCK}:/ssh-agent" \
    "${EXTRA_MOUNTS[@]}" \
    -e SSH_AUTH_SOCK=/ssh-agent \
    -e USER_DATA_PATH=/workspace/hyperloom \
    -e SAFE_API_KEY="${SAFE_API_KEY}" \
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    -e ANTHROPIC_BASE_URL="${anthropic_base_url}" \
    -e ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS}" \
    -e AMD_API_PORT="${AMD_API_PORT}" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "tail -f /dev/null"
}

do_start() {
  if container_running; then
    log "container $CONTAINER is already running"
    exec_into
    return
  fi

  if container_exists; then
    log "container $CONTAINER exists but is stopped — restarting ..."
    if docker start "$CONTAINER" 2>/dev/null; then
      exec_into
      return
    fi
    log "restart failed (stale record) — removing and creating fresh ..."
    docker rm -f "$CONTAINER" 2>/dev/null || true
  fi

  image_exists || build_image
  start_container
  log "container started"
  exec_into
}

do_stop() {
  if container_exists; then
    log "stopping and removing $CONTAINER ..."
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm   "$CONTAINER" 2>/dev/null || true
    log "done"
  else
    log "container $CONTAINER does not exist — nothing to stop"
  fi
}

# ── Argument parsing ──────────────────────────────────────────────────────────
ACTION="start"
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --start) ACTION="start" ;;
    --stop)  ACTION="stop"  ;;
    --api-port)
      i=$(( i + 1 ))
      [[ $i -lt ${#args[@]} ]] || die "--api-port requires a PORT argument"
      AMD_API_PORT="${args[$i]}"
      [[ "$AMD_API_PORT" =~ ^[0-9]+$ ]] || die "--api-port must be a number, got: ${AMD_API_PORT}"
      log "AMD API port set to ${AMD_API_PORT}"
      ;;
    --mount)
      i=$(( i + 1 ))
      [[ $i -lt ${#args[@]} ]] || die "--mount requires a PATH argument"
      host_path="${args[$i]}"
      host_path="${host_path%/}"   # strip trailing slash
      [[ -e "$host_path" ]] || die "--mount path does not exist: $host_path"
      host_path="$(realpath "$host_path")"
      mount_name="$(basename "$host_path")"
      EXTRA_MOUNTS+=(-v "${host_path}:/mnt/${mount_name}:ro")
      log "will mount ${host_path} -> /mnt/${mount_name} (read-only)"
      ;;
    -h|--help)
      echo "Usage: $0 [--start|--stop] [--api-port PORT] [--mount PATH] ..."
      echo ""
      echo "  --start            (default) Start container and exec into bash"
      echo "  --stop             Stop and remove the container"
      echo "  --api-port PORT    AMD LLM API tunnel port on the host (default: 9443)"
      echo "                     Use 9444 for amd-llm-proxy (Shadeform / no-sudo nodes)"
      echo "  --mount PATH       Mount PATH read-only at /mnt/<basename> inside the container"
      echo "                     Can be repeated for multiple repos"
      echo ""
      echo "Examples:"
      echo "  $0                                          # default port 9443"
      echo "  $0 --api-port 9444                         # amd-llm-proxy port"
      echo "  $0 --api-port 9444 --mount ~/workspace/ml-perf"
      echo "  $0 --stop"
      exit 0 ;;
    "") ;;
    *) die "unknown argument: $arg" ;;
  esac
  i=$(( i + 1 ))
done

case "$ACTION" in
  start) do_start ;;
  stop)  do_stop  ;;
esac
