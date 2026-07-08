#!/usr/bin/env bash
# Runs INSIDE the container. Launches gpt-oss-120b vLLM (baseline serving config)
# with whatever PYTORCH_TUNABLEOP_* env is already exported (record / read / off).
# Kills any prior server and waits for the port to free before relaunching, so
# callers never poll a dying instance (the "false-ready" trap).
set -u
PORT="${PORT:-8001}"
LOG="${LOG:-/models/forge_demo_out/vllm.log}"

pkill -f vllm.entrypoints 2>/dev/null || true
# wait until the port is actually free (old server gone)
for _ in $(seq 1 30); do
  curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || break
  sleep 1
done

mkdir -p "$(dirname "$LOG")"
cd /root
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /models/gpt-oss-120b \
  --served-model-name gpt-oss-120b /models/gpt-oss-120b \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --port "$PORT" --enforce-eager \
  --compilation-config '{"custom_ops":["none","-rms_norm","-fused_add_rms_norm","-silu_and_mul","-gelu_and_mul","-gelu_new","-gelu_fast","-rotary_embedding","-quick_gelu"]}' \
  > "$LOG" 2>&1 &
echo "vllm pid=$! log=$LOG"
