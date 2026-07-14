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

# --- write the in-container runner to weka (visible inside the container) ---
INCONTAINER="$USER_DATA_PATH/_incontainer.sh"
cat > "$INCONTAINER" <<'INCONTAINER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[hl] session=$CLAW_SESSION_ID model=$REPO_ID fw=$FRAMEWORK gpu=$GPU_TYPE tp=$TP backend=$BACKEND"

# 1) host alias so the gateway domain resolves to the jump host (SNI).
if [ -n "${HL_HOST_ALIAS_IP:-}" ] && [ -n "${HL_HOST_ALIAS_NAME:-}" ]; then
  if ! grep -q "$HL_HOST_ALIAS_NAME" /etc/hosts 2>/dev/null; then
    echo "${HL_HOST_ALIAS_IP} ${HL_HOST_ALIAS_NAME}" >> /etc/hosts \
      || echo "[hl][warn] cannot write /etc/hosts; ensure --container-writable" >&2
  fi
fi

# 2) gateway reachability preflight (non-fatal; surfaces misconfig early).
curl -sS -o /dev/null -w "[hl] gateway /models HTTP=%{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-$SAFE_API_KEY}" \
  "${OPENAI_BASE_URL}/models" || echo "[hl][warn] gateway preflight failed" >&2

# 3) Hyperloom source. Precedence:
#    - if HYPERLOOM_SOURCE_DIR already holds a checkout (pyproject.toml or .git),
#      use it as-is (e.g. a bind-mounted /mnt/vast/hyperloom) -> no network clone;
#    - otherwise clone from GitHub into the default temp dir.
export HYPERLOOM_SOURCE_DIR="${HYPERLOOM_SOURCE_DIR:-/tmp/Hyperloom-main}"
if [ -f "$HYPERLOOM_SOURCE_DIR/pyproject.toml" ] || [ -d "$HYPERLOOM_SOURCE_DIR/.git" ]; then
  echo "[hl] using existing Hyperloom source at $HYPERLOOM_SOURCE_DIR (no clone)"
else
  echo "[hl] cloning Hyperloom into $HYPERLOOM_SOURCE_DIR"
  git clone --depth 1 --branch "${HYPERLOOM_SOURCE_REF:-main}" \
    "${HYPERLOOM_SOURCE_REPO:-https://github.com/AMD-AGI/Hyperloom.git}" "$HYPERLOOM_SOURCE_DIR"
fi
export REPO_ROOT="$HYPERLOOM_SOURCE_DIR"
cd "$REPO_ROOT"

# 4) install optimizer + components on top of the image's /opt/venv ROCm stack.
#    NOTE: post tree-reform (src-layout) the installer lives under assets/.
bash src/hyperloom/inference_optimizer/assets/install.sh
. "${USER_DATA_PATH}/runtime/kernel-agent.env.sh"

RUN_DIR="$USER_DATA_PATH/optimizer_runs"; mkdir -p "$RUN_DIR"
LAUNCH_INFO_FILE="$RUN_DIR/launch_${CLAW_SESSION_ID}.json"

case "$BACKEND" in
  python)
    # Workload envs the Coordinator reuses for baseline -> params -> sweep.
    export TP EP=1 CONC ISL OSL PRECISION="$PREC" NODES=1
    exec inference_optimizer --verbose optimize \
      --model "$MODEL_PATH" \
      --framework "$FRAMEWORK" \
      --gpu-type "$GPU_TYPE" \
      --model-class "$MODEL_CLASS" \
      --target-gain "$TARGET_GAIN" \
      --max-hours "$MAX_HOURS" \
      --tick-interval-sec 30 \
      --kernel-claude \
      --launch-info-file "$LAUNCH_INFO_FILE"
    ;;
  claude)
    command -v claude >/dev/null 2>&1 || { echo "[hl] claude CLI not in image" >&2; exit 3; }
    PROMPT="$(python3 - <<'PY'
import os
tpl = open("ci/prompt_template.md").read()
tpl = tpl.replace("/wekafs/HyperloomV2/src/hyperloom/inference_optimizer/SKILL.md",
                  os.environ["REPO_ROOT"] + "/src/hyperloom/inference_optimizer/SKILL.md")
vals = dict(
    model_hf=os.environ["REPO_ID"], model_path=os.environ.get("MODEL_PATH", os.environ["REPO_ID"]),
    gpu_type=os.environ["GPU_TYPE"], gpu_type_lc=os.environ["GPU_TYPE"].lower(),
    framework=os.environ["FRAMEWORK"], precision=os.environ["PREC"],
    tp=os.environ["TP"], ep="1", conc=os.environ["CONC"],
    isl=os.environ["ISL"], osl=os.environ["OSL"], nodes="1",
    random_range_ratio="1.0", target_gain=os.environ["TARGET_GAIN"],
    max_hours=os.environ["MAX_HOURS"], kernel_agent_build_geak_rag_index="1",
    oob_path="", tracelens_root="", multinode_section="", inferenceX_data="(none)",
)
print(tpl.format(**vals))
PY
)"
    exec claude --dangerously-skip-permissions -p "$PROMPT"
    ;;
  *)
    echo "[hl] unknown backend: $BACKEND" >&2; exit 2 ;;
esac
INCONTAINER_EOF
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
             MAX_HOURS TARGET_GAIN BACKEND HL_HOST_ALIAS_IP HL_HOST_ALIAS_NAME \
             CURSOR_DEFAULT_MODEL GEAK_MODEL_NAME CLAUDE_MODEL CODEX_MODEL LLM_MODEL \
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
    docker run --rm \
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
