#!/usr/bin/env bash
###############################################################################
# REAL Hyperloom demo — deterministic GEMM tuning with forge (forge_gemm_tune).
#
# This exercises an actual Hyperloom kernel-optimization backend (`forge`), NOT
# a hand-written patch. forge_gemm_tune is LLM-free ("exhaustive search via
# aiter CK tuners / PyTorch TunableOp"), so it runs with ZERO credentials — no
# LLM gateway, no ANTHROPIC key. That's why it works where GEAK / the full
# `inference_optimizer optimize` coordinator can't on a credential-less box.
#
# Flow (the canonical forge dense-GEMM pipeline):
#   1. RECORD  — serve gpt-oss-120b with PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 and
#                drive traffic to capture the real dense GEMM shapes.
#   2. TUNE    — `forge-gemm-tune run --tuner vllm_dense_tunableop` runs PyTorch
#                TunableOp exhaustive search over those shapes -> tuned CSV.
#   3. VALIDATE— serve with the tuned CSV vs baseline, back-to-back A/B.
#
# Honest expected outcome on pre-silicon gfx1250: forge finds faster non-Default
# hipBLASLt/rocBLAS kernels for many shapes but none clears its 3% bar, and the
# end-to-end A/B shows ~no gain — the dense GEMMs are already near-optimal here.
# The point of the demo is the *workflow* (running a Hyperloom backend), and that
# a deterministic tuner correctly reports "no headroom" rather than inventing one.
#
#   ./run_forge_demo.sh
###############################################################################
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- config (all overridable) ----
NAME="${NAME:-hl-forge-demo}"
CARD="${CARD:-1}"; PORT="${PORT:-8001}"
IMAGE="${IMAGE:-registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2}"
MODEL_HOST="${MODEL_HOST:-/home/yanyuqin/models}"
MXFP4_PATCH="${MXFP4_PATCH:-/home/yanyuqin/tk-patch/mxfp4_utils.py}"
FORGE_SRC="${FORGE_SRC:-/home/yanyuqin/hyperloom-run/deps/KernelForge/src/forge_gemm_tune}"
OUT_HOST="${OUT_HOST:-$MODEL_HOST/forge_demo_out}"     # artifacts land here (mounts at /models/forge_demo_out)
CONC="${CONC:-32}"; NREQ="${NREQ:-64}"; ISL="${ISL:-1024}"; OSL="${OSL:-128}"
WARMUP="${WARMUP:-3}"; RUNS="${RUNS:-4}"
PY=python3
VLLM_MXFP4_DST="/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py"

say() { echo "" ; echo "==== $* ====" ; }
poll_ready() {  # poll host until the server answers, or die
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)" = "200" ] \
      && { echo ">> ready ~$((i*5))s"; return 0; }
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { echo ">> container died"; docker logs --tail 20 "$NAME"; return 1; }
    sleep 5
  done; echo ">> timeout"; return 1
}
median() { "$PY" -c "import sys,statistics as s; v=[float(x) for x in sys.argv[1:]]; print(f'{s.median(v):.1f}')" "$@"; }
measure() {  # $1 label -> echoes median tok/s (also warms first)
  for _ in $(seq 1 "$WARMUP"); do "$PY" "$HERE/bench.py" "$PORT" w "$CONC" "$NREQ" "$ISL" "$OSL" >/dev/null 2>&1; done
  local vals=()
  for _ in $(seq 1 "$RUNS"); do
    local t; t="$("$PY" "$HERE/bench.py" "$PORT" "$1" "$CONC" "$NREQ" "$ISL" "$OSL" 2>/dev/null | tail -1 \
      | "$PY" -c "import sys,json;print(json.loads(sys.stdin.read())['out_tok_s'])" 2>/dev/null)"
    echo "     $1: $t tok/s" >&2; vals+=("$t")
  done
  median "${vals[@]}"
}

mkdir -p "$OUT_HOST"

say "0. control container ($NAME) on card $CARD + install forge"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$MODEL_HOST:/models" -v "$MXFP4_PATCH:$VLLM_MXFP4_DST:ro" -v "$FORGE_SRC:/forge:ro" \
  -e ROCR_VISIBLE_DEVICES="$CARD" -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_SKINNY_GEMM=0 -e HF_HUB_OFFLINE=1 -e VLLM_PLUGINS="" \
  -e HSA_USE_SVM=0 -e HSA_XNACK=0 -e HIP_FORCE_DEV_KERNARG=1 \
  "$IMAGE" -lc "sleep infinity" >/dev/null
sleep 3
docker cp "$HERE/_run_vllm.sh" "$NAME:/root/_run_vllm.sh" >/dev/null
docker exec "$NAME" bash -lc 'cp -r /forge /tmp/forge_gemm_tune && cd /tmp/forge_gemm_tune && pip install -e . -q 2>&1 | tail -1; forge-gemm-tune --version'

say "1. RECORD dense GEMM shapes (PYTORCH_TUNABLEOP_RECORD_UNTUNED=1)"
docker exec -d "$NAME" bash -lc "
  export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
  export PYTORCH_TUNABLEOP_UNTUNED_FILENAME=/models/forge_demo_out/untuned_%d.csv
  PORT=$PORT LOG=/models/forge_demo_out/vllm_record.log bash /root/_run_vllm.sh"
poll_ready || exit 1
"$PY" "$HERE/bench.py" "$PORT" record 8 16 "$ISL" "$OSL" >/dev/null 2>&1   # drive traffic to record shapes
docker exec "$NAME" bash -lc 'pkill -f vllm.entrypoints; sleep 4'
UNTUNED=$(docker exec "$NAME" bash -lc 'ls /models/forge_demo_out/untuned_*.csv 2>/dev/null | head -1' | tr -d "\r")
echo ">> recorded shapes: $UNTUNED ($(docker exec "$NAME" bash -lc "grep -cE '^Gemm' $UNTUNED" 2>/dev/null | tr -d '\r') GEMM shapes)"

say "2. TUNE with forge (vllm_dense_tunableop -> PyTorch TunableOp)"
docker exec "$NAME" bash -lc "cd /tmp && forge-gemm-tune run \
  --model-path /models/gpt-oss-120b --framework vllm --precision bf16 --quant-type mxfp4 \
  --gpu-type auto --tuner vllm_dense_tunableop --tunableop-input '$UNTUNED' \
  --output-dir /models/forge_demo_out/tune --skip-gpu-check --global-timeout 1800 2>&1 | grep -iE 'tuner|shapes|status|improved' | tail -8"
TUNED=$(docker exec "$NAME" bash -lc 'ls /models/forge_demo_out/tune/tuners/vllm_dense_tunableop/tunableop_results.csv 2>/dev/null' | tr -d "\r")
echo ">> tuned CSV: $TUNED"
docker exec "$NAME" bash -lc "echo '   total shapes:' \$(grep -cE '^Gemm' '$TUNED'); echo '   non-Default kernels:' \$(grep -E '^Gemm' '$TUNED' | grep -vc ',Default,')"

say "3a. serve TUNED + A/B measure"
docker exec -d "$NAME" bash -lc "
  export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME='$TUNED'
  PORT=$PORT LOG=/models/forge_demo_out/vllm_tuned.log bash /root/_run_vllm.sh"
poll_ready || exit 1
TUNED_TP=$(measure forge-tuned)

say "3b. serve BASELINE (TunableOp off) + A/B measure"
docker exec -d "$NAME" bash -lc "export PYTORCH_TUNABLEOP_ENABLED=0; PORT=$PORT LOG=/models/forge_demo_out/vllm_base.log bash /root/_run_vllm.sh"
poll_ready || exit 1
BASE_TP=$(measure baseline)

DELTA=$("$PY" -c "b=$BASE_TP;f=$TUNED_TP;print(f'{100*(f-b)/b:+.1f}%')")
echo ""
echo "================= FORGE GEMM-TUNE RESULT (median out tok/s @ conc=$CONC) ================="
printf "  %-28s %8s\n" "baseline (TunableOp off)" "$BASE_TP"
printf "  %-28s %8s\n" "forge-tuned dense GEMMs"  "$TUNED_TP"
printf "  %-28s %8s\n" "throughput delta"         "$DELTA"
echo "========================================================================================"
echo "artifacts: $OUT_HOST/tune/result.json   |   cleanup: docker rm -f $NAME"
