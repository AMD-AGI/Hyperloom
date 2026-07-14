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

# --- shared filesystem (weka on core42, /mnt/vast on vultr, ...) ------------
# HL_SHARED_MOUNT is bind-mounted into the container and hosts all artifacts.
export HL_SHARED_MOUNT="${HL_SHARED_MOUNT:-/wekafs}"
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
  export SAFE_API_KEY="${SAFE_API_KEY:-$OPENAI_API_KEY}"
  if [ -n "${HL_CA_BUNDLE_HOST:-}" ] && [ -f "$HL_CA_BUNDLE_HOST" ]; then
    PROXY_MOUNTS=",${HL_CA_BUNDLE_HOST}:${HL_CA_BUNDLE_CONTAINER}:ro"
    export REQUESTS_CA_BUNDLE="$HL_CA_BUNDLE_CONTAINER"
    export SSL_CERT_FILE="$HL_CA_BUNDLE_CONTAINER"
  fi
fi
# Hard requirements regardless of proxy toggle.
: "${SAFE_API_KEY:?export SAFE_API_KEY (or set it in proxy.env)}"
: "${OPENAI_BASE_URL:?export OPENAI_BASE_URL (or set it in proxy.env)}"
export HF_TOKEN="${HF_TOKEN:-}"

# --- copy the shared in-container runner to the shared mount ----------------
# run_hyperloom.sbatch and run_hyperloom-spur.sbatch execute the SAME
# _incontainer.sh.in inside the container; keep it single-sourced to avoid
# drift. HL_KERNEL_BACKEND selects the python carrier's kernel-agent backend
# (stock Slurm -> claude); it is forwarded into the container.
export HL_KERNEL_BACKEND="${HL_KERNEL_BACKEND:-claude}"
INCONTAINER="$USER_DATA_PATH/_incontainer.sh"
cp "$CONF_DIR/_incontainer.sh.in" "$INCONTAINER"
chmod +x "$INCONTAINER"

# --- launch the container --------------------------------------------------
# Runtime auto-detect: pyxis/enroot (core42-style) or docker (vultr-style).
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
    for v in CLAW_SESSION_ID USER_DATA_PATH HL_SHARED_MOUNT SAFE_API_KEY \
             OPENAI_BASE_URL OPENAI_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL \
             REQUESTS_CA_BUNDLE SSL_CERT_FILE HF_TOKEN HYPERLOOM_SOURCE_DIR \
             HYPERLOOM_SOURCE_REPO HYPERLOOM_SOURCE_REF \
             KEY REPO_ID MODEL_PATH FRAMEWORK GPU_TYPE TP PREC ISL OSL CONC MODEL_CLASS \
             MAX_HOURS TARGET_GAIN BACKEND HL_KERNEL_BACKEND HL_HOST_ALIAS_IP HL_HOST_ALIAS_NAME \
             CURSOR_DEFAULT_MODEL GEAK_CLAUDE_MODEL CLAUDE_MODEL CODEX_MODEL LLM_MODEL \
             RECIPE_KB_REMOTE GBRAIN_BASE_URL GBRAIN_TOKEN GBRAIN_KG_GUIDED \
             GBRAIN_KG_NATIVE GBRAIN_RECIPE_SCAN_BUDGET_SEC \
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
    docker run --rm \
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
