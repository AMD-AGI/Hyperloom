#!/usr/bin/env bash
###############################################################################
# Launch a gpt-oss-120b vLLM server for the fused-RMSNorm demo.
#
#   MODE=baseline ./serve.sh      # stock vLLM (unfused fp32 residual add)
#   MODE=fused    ./serve.sh      # + fused add+RMSNorm Triton kernel (this demo)
#
# Leaves the container running and detached. Poll $PORT/v1/models for readiness.
# Everything is overridable via env (see the block below).
###############################################################################
set -eu

MODE="${MODE:-baseline}"                 # baseline | fused
NAME="${NAME:-gptoss-demo-$MODE}"
PORT="${PORT:-8001}"
CARD="${CARD:-1}"                        # physical GPU (ROCR index) to lease
IMAGE="${IMAGE:-registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2}"
MODEL_HOST="${MODEL_HOST:-/home/yanyuqin/models}"          # host dir containing gpt-oss-120b/
MXFP4_PATCH="${MXFP4_PATCH:-/home/yanyuqin/tk-patch/mxfp4_utils.py}"  # required gfx1250 MoE fix
FUSED_PATCH="${FUSED_PATCH:-$(cd "$(dirname "$0")" && pwd)/layernorm_fused.py}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

VLLM_MXFP4_DST="/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py"
VLLM_LN_DST="/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py"

# ROCR masks to the physical card; HIP indexes densely within it (Ray needs it).
extra_mounts=(-v "$MXFP4_PATCH:$VLLM_MXFP4_DST:ro")
if [ "$MODE" = "fused" ]; then
  extra_mounts+=(-v "$FUSED_PATCH:$VLLM_LN_DST:ro")
  echo ">> MODE=fused: bind-mounting the fused add+RMSNorm kernel"
else
  echo ">> MODE=baseline: stock vLLM RMSNorm (unfused fp32 residual add)"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$MODEL_HOST:/models" "${extra_mounts[@]}" \
  -e ROCR_VISIBLE_DEVICES="$CARD" -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_SKINNY_GEMM=0 -e HF_HUB_OFFLINE=1 -e VLLM_PLUGINS="" \
  -e HSA_USE_SVM=0 -e HSA_XNACK=0 -e HIP_FORCE_DEV_KERNARG=1 \
  "$IMAGE" -lc "
    cd /root
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model /models/gpt-oss-120b \
      --served-model-name gpt-oss-120b /models/gpt-oss-120b \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-model-len $MAX_MODEL_LEN \
      --port $PORT --enforce-eager \
      --compilation-config '{\"custom_ops\":[\"none\",\"-rms_norm\",\"-fused_add_rms_norm\",\"-silu_and_mul\",\"-gelu_and_mul\",\"-gelu_new\",\"-gelu_fast\",\"-rotary_embedding\",\"-quick_gelu\"]}'
  " >/dev/null

echo ">> launched container '$NAME' on card $CARD, port $PORT"
echo ">> waiting for model load (warm weights ~1-2 min; cold ~15 min)..."
for i in $(seq 1 240); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo ">> READY after ~$((i*5))s"; exit 0; }
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || {
    echo ">> CONTAINER DIED — last log:"; docker logs --tail 20 "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 5
done
echo ">> TIMEOUT waiting for readiness"; exit 1
