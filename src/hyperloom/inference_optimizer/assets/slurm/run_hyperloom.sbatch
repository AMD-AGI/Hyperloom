#!/usr/bin/env bash
# Hyperloom optimization job for a single model, run inside a vllm/sglang ROCm
# container via enroot/pyxis. Submit through submit.sh (do not sbatch directly
# unless HL_CONF_DIR is exported).
#
# Args: $1 = model key (row in models.tsv), $2 = backend (python|claude)
#
#SBATCH --job-name=hyperloom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=hyperloom-%x-%j.out

set -euo pipefail

MODEL_KEY="${1:?usage: run_hyperloom.sbatch <model_key> [python|claude]}"
BACKEND="${2:-python}"

# Config dir carrying models.tsv + proxy.env (exported by submit.sh).
CONF_DIR="${HL_CONF_DIR:?HL_CONF_DIR not set (submit via submit.sh)}"

# --- resolve the model row -------------------------------------------------
ROW="$(awk -F'\t' -v k="$MODEL_KEY" '$1==k{print;exit}' "$CONF_DIR/models.tsv")"
[ -n "$ROW" ] || { echo "unknown model key: $MODEL_KEY (see $CONF_DIR/models.tsv)" >&2; exit 2; }
IFS=$'\t' read -r KEY REPO_ID FRAMEWORK GPU_TYPE IMAGE TP PREC ISL OSL CONC MODEL_CLASS MAX_HOURS TARGET_GAIN <<<"$ROW"
# Real hardware may differ from the model's tabled target (e.g. an MI355X-only
# cluster running an MI300X-tagged row). HL_GPU_TYPE_OVERRIDE wins when set.
GPU_TYPE="${HL_GPU_TYPE_OVERRIDE:-$GPU_TYPE}"

# Resolve the model source path. Precedence:
#   1. REPO_ID is already an existing local dir (absolute path in models.tsv);
#   2. HL_MODEL_BASE is set and <HL_MODEL_BASE>/<basename REPO_ID> exists;
#   3. fall back to REPO_ID as-is (an HF repo id downloaded at launch).
# REPO_ID stays the logical/HF identity; MODEL_PATH is what the optimizer loads.
MODEL_PATH="$REPO_ID"
if [ -d "$REPO_ID" ]; then
  MODEL_PATH="$REPO_ID"
elif [ -n "${HL_MODEL_BASE:-}" ] && [ -d "${HL_MODEL_BASE%/}/${REPO_ID##*/}" ]; then
  MODEL_PATH="${HL_MODEL_BASE%/}/${REPO_ID##*/}"
fi
echo "[hl] model resolve: repo_id=$REPO_ID base=${HL_MODEL_BASE:-<none>} -> $MODEL_PATH"
export KEY REPO_ID MODEL_PATH FRAMEWORK GPU_TYPE IMAGE TP PREC ISL OSL CONC MODEL_CLASS MAX_HOURS TARGET_GAIN BACKEND

# --- identity: end-to-end join key the optimizer reads from env ------------
export CLAW_SESSION_ID="${CLAW_SESSION_ID:-$(cat /proc/sys/kernel/random/uuid)}"

# --- shared filesystem (e.g. weka, /mnt/vast, ...) --------------------------
# HL_SHARED_MOUNT is bind-mounted into the container and hosts all artifacts.
export HL_SHARED_MOUNT="${HL_SHARED_MOUNT:-/path}"
export USER_DATA_PATH="${HL_DATA_ROOT:-${HL_SHARED_MOUNT}/hyperloom-slurm}/${KEY}/${CLAW_SESSION_ID}"
mkdir -p "$USER_DATA_PATH"

# --- proxy profile (optional) ----------------------------------------------
PROXY_MOUNTS=""
if [ -f "$CONF_DIR/proxy.env" ]; then
  set -a; . "$CONF_DIR/proxy.env"; set +a
fi
if [ "${HL_PROXY_ENABLED:-0}" = "1" ]; then
  : "${OPENAI_BASE_URL:?proxy enabled but OPENAI_BASE_URL unset}"
  : "${OPENAI_API_KEY:?proxy enabled but OPENAI_API_KEY unset}"
  if [ -n "${HL_CA_BUNDLE_HOST:-}" ] && [ -f "$HL_CA_BUNDLE_HOST" ]; then
    PROXY_MOUNTS=",${HL_CA_BUNDLE_HOST}:${HL_CA_BUNDLE_CONTAINER}:ro"
    export REQUESTS_CA_BUNDLE="$HL_CA_BUNDLE_CONTAINER"
    export SSL_CERT_FILE="$HL_CA_BUNDLE_CONTAINER"
  fi
fi
# Hard requirements regardless of proxy toggle.
: "${OPENAI_API_KEY:?export OPENAI_API_KEY (or set it in proxy.env)}"
: "${OPENAI_BASE_URL:?export OPENAI_BASE_URL (or set it in proxy.env)}"
export HF_TOKEN="${HF_TOKEN:-}"

# --- copy the in-container runner to the shared mount ----------------------
# run_hyperloom.sbatch runs _incontainer.sh.in inside the container.
# Kernel-opt rewrite ladder (geak|forge); forwarded into the container.
export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"
INCONTAINER="$USER_DATA_PATH/_incontainer.sh"
cp "$CONF_DIR/_incontainer.sh.in" "$INCONTAINER"
chmod +x "$INCONTAINER"

# --- preflight: reclaim GPUs from orphaned hyperloom containers -------------
# A cancelled or NODE_FAIL'd job kills the host-side `docker run --rm` client but
# leaves its container running, still holding GPU memory. The next job on that
# node then either trips benchmark_lib.sh's wait_for_amd_gpu_clean (it maxes
# VRAM% over every GPU on the box and hard-fails after 15min) or is refused by
# vLLM's startup free-memory check. Observed on two nodes at once: a GLM sglang
# container from a job cancelled three hours earlier still held ~168 GiB on each
# of GPU 0-3, which is exactly where the next job's tp=4 server wanted to land.
#
# A container counts as ours-and-orphaned only when its CLAW_SESSION_ID has no
# live `docker run` client, so a co-tenant job on the same node is never touched.
#
# Every branch below is spelled as `if`, not `cmd && ...`: this file runs under
# `set -e`, where a failing AND-list aborts the job.
hl_vram_max() {
  rocm-smi --showmemuse 2>/dev/null \
    | grep -oE 'GPU Memory Allocated \(VRAM%\): [0-9]+' \
    | awk '{if ($NF > m) m = $NF} END {print m+0}'
}

hl_kill_container_cgroup() {
  local full="$1" p
  for p in $(cat /sys/fs/cgroup/system.slice/docker-"$full".scope/cgroup.procs \
                 /sys/fs/cgroup/docker/"$full"/cgroup.procs 2>/dev/null); do
    if ! sudo -n kill -9 "$p" 2>/dev/null; then
      kill -9 "$p" 2>/dev/null || true
    fi
  done
  return 0
}

hl_reclaim_orphan_gpus() {
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  local c full sid img found=0 i vram
  echo "[hl] preflight: vram%max=$(hl_vram_max) before reclaim"
  for c in $(docker ps -q 2>/dev/null); do
    sid="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null \
           | sed -n 's/^CLAW_SESSION_ID=//p' | head -1)"
    if [ -z "$sid" ]; then
      continue
    fi
    if [ "$sid" = "${CLAW_SESSION_ID:-}" ]; then
      continue
    fi
    if pgrep -f "docker run.*CLAW_SESSION_ID=$sid" >/dev/null 2>&1; then
      continue
    fi
    full="$(docker inspect -f '{{.Id}}' "$c" 2>/dev/null || true)"
    img="$(docker inspect -f '{{.Config.Image}}' "$c" 2>/dev/null || true)"
    echo "[hl] preflight: orphan container $c session=$sid image=$img -> reclaiming"
    found=1
    docker kill "$c" >/dev/null 2>&1 || true
    # A zombie container init refuses `docker kill`; SIGKILL its cgroup instead.
    if [ -n "$full" ]; then
      hl_kill_container_cgroup "$full"
    fi
    docker rm -f "$c" >/dev/null 2>&1 || true
  done
  if [ "$found" != "1" ]; then
    echo "[hl] preflight: no orphan containers"
    return 0
  fi
  for i in $(seq 1 18); do
    vram="$(hl_vram_max)"
    if [ "${vram:-0}" -le 10 ]; then
      echo "[hl] preflight: GPUs reclaimed (vram%max=$vram after $((i * 10))s)"
      return 0
    fi
    sleep 10
  done
  echo "[hl] preflight: WARNING vram%max=$(hl_vram_max) still high after 180s" >&2
  return 0
}
hl_reclaim_orphan_gpus || true

# --- launch the container --------------------------------------------------
# Runtime auto-detect: pyxis/enroot or docker.
# Override with HL_CONTAINER_RUNTIME=pyxis|docker.
RUNTIME="${HL_CONTAINER_RUNTIME:-auto}"
if [ "$RUNTIME" = "auto" ]; then
  if command -v enroot >/dev/null 2>&1 || srun --help 2>&1 | grep -q -- '--container-image'; then
    RUNTIME=pyxis
  elif command -v docker >/dev/null 2>&1; then
    RUNTIME=docker
  else
    echo "no container runtime found (need pyxis/enroot or docker)" >&2; exit 4
  fi
fi
echo "[hl] container runtime: $RUNTIME  image: $IMAGE"

case "$RUNTIME" in
  pyxis)
    # Private registry image needs enroot creds (~/.config/enroot/.credentials).
    srun --container-image="docker://${IMAGE}" \
         --container-mounts="${HL_SHARED_MOUNT}:${HL_SHARED_MOUNT}${PROXY_MOUNTS}" \
         --container-writable \
         --no-container-remap-root \
         --export=ALL \
         bash "$INCONTAINER"
    ;;
  docker)
    # Private registry image needs `docker login <registry>` first (or a
    # locally-cached image, in which case no login/pull is needed).
    DOCKER_ENV=()
    # All three Anthropic credential forms are forwarded, plus each side's
    # gateway headers: a header-authenticated gateway rejects the call without
    # them, and forwarding only some forms silently downgrades the container to
    # a credential the operator did not choose.
    for v in CLAW_SESSION_ID USER_DATA_PATH HL_SHARED_MOUNT \
             OPENAI_BASE_URL OPENAI_API_KEY OPENAI_CUSTOM_HEADERS \
             ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN \
             ANTHROPIC_CUSTOM_HEADERS CLAUDE_CODE_OAUTH_TOKEN \
             REQUESTS_CA_BUNDLE SSL_CERT_FILE HF_TOKEN HYPERLOOM_SOURCE_DIR \
             HYPERLOOM_SOURCE_REPO HYPERLOOM_SOURCE_REF \
             KEY REPO_ID MODEL_PATH FRAMEWORK GPU_TYPE TP PREC ISL OSL CONC MODEL_CLASS \
             MAX_HOURS TARGET_GAIN BACKEND KERNEL_OPT_BACKEND_ORDER HL_HOST_ALIAS_IP HL_HOST_ALIAS_NAME \
             CURSOR_DEFAULT_MODEL GEAK_CLAUDE_MODEL CLAUDE_MODEL CODEX_MODEL LLM_MODEL \
             KNOWLEDGE_STORE_MODE KNOWLEDGE_LOCAL_ROOT \
             GBRAIN_BASE_URL GBRAIN_TOKEN \
             HYPERLOOM_LANGFUSE_ENABLE LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
      [ -n "${!v:-}" ] && DOCKER_ENV+=(-e "$v=${!v}")
    done
    ADD_HOST=()
    [ -n "${HL_HOST_ALIAS_NAME:-}" ] && [ -n "${HL_HOST_ALIAS_IP:-}" ] \
      && ADD_HOST=(--add-host "${HL_HOST_ALIAS_NAME}:${HL_HOST_ALIAS_IP}")
    CA_MOUNT=()
    [ -n "${HL_CA_BUNDLE_HOST:-}" ] && [ -f "${HL_CA_BUNDLE_HOST}" ] \
      && CA_MOUNT=(-v "${HL_CA_BUNDLE_HOST}:${HL_CA_BUNDLE_CONTAINER}:ro")
    # --ulimit lifts the container's open-files hard cap so the Ray raylet's fd
    # preflight can raise soft nofile high enough to stay up (issue #433).
    docker run --rm --init \
      --name "hl-${KEY}-${SLURM_JOB_ID:-$$}" \
      --ulimit "nofile=${HL_NOFILE:-1048576}:${HL_NOFILE:-1048576}" \
      --device=/dev/kfd --device=/dev/dri \
      --security-opt seccomp=unconfined --group-add video --ipc=host \
      --shm-size="${HL_SHM_SIZE:-64g}" \
      -v "${HL_SHARED_MOUNT}:${HL_SHARED_MOUNT}" \
      "${CA_MOUNT[@]}" "${ADD_HOST[@]}" "${DOCKER_ENV[@]}" \
      "${IMAGE}" bash "$INCONTAINER"
    ;;
  *)
    echo "[hl] unknown HL_CONTAINER_RUNTIME: $RUNTIME" >&2; exit 4 ;;
esac
