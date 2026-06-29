#!/usr/bin/env bash
# MiniMax-M3 MXFP8 on MI355X (gfx950) — NON-MTP baseline + concurrency sweep.
#
# Mirrors the InferenceX leaderboard recipe
#   InferenceX/benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi355x.sh
# (vllm serve: MXFP8 experts, mandatory --block-size 128, --language-model-only,
#  FP8 KV cache, --attention-backend TRITON_ATTN) and the launch/bookkeeping
# structure of Hyperloom/scripts/run_dsv4_optimize.sh, but instead of tearing the
# server down per concurrency it launches ONCE and sweeps the benchmark across
# CONC_LIST so we can reproduce the InferenceX 1k/1k curve point-by-point.
#
# RUN THIS INSIDE the `fanxingran_minimax_m3_raw` container (vllm is installed
# there, not on the host):
#   docker exec -it fanxingran_minimax_m3_raw bash
#   bash /home/xingran.fan@amd.com/Hyperloom/scripts/run_minimaxm3_mi355x.sh
#
# Cards 0,1,2,3 are the free GPUs on this node (4-7 are busy) -> TP=4, which is
# exactly the InferenceX search-space row `{ tp: 4, conc-start: 1, conc-end: 64 }`
# for minimaxm3-fp8-mi355x-vllm. Concurrency sweep defaults to 1,4,8,16,32.
#
# Follow:   tail -f run-logs/minimaxm3-nonmtp-latest/server.log
# Results:  run-logs/minimaxm3-nonmtp-<ts>/bmk_conc*.json  (InferenceX schema)
#
# Profiling pass (operator-level torch traces for the optimization phase):
#   PROFILE=1 bash scripts/run_minimaxm3_mi355x.sh
#
# Override anything via env, e.g.:
#   TP=4 CONC_LIST="1 4 8 16 32" MM3_PORT=8893 bash scripts/run_minimaxm3_mi355x.sh
#
# NOTE: use MM3_PORT (not PORT) to pick the server port — the vllm image presets
# PORT=8888, where an unrelated MiniMax server is already running on this node.
# NOTE: no `set -u` here: InferenceX/benchmark_lib.sh reads optional vars
# (EVAL_ONLY, etc.) unguarded, which nounset would turn into hard errors.
set -eo pipefail
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"

# ── P0 GUARD: pin single-arch build for the custom MXFP8 HIP kernels ─────────────
# The container default PYTORCH_ROCM_ARCH is multi-arch and INCLUDES gfx1100
# (RDNA3), which lacks the fp8-conversion insts the smallm MXFP8 kernels emit
# (__builtin_amdgcn_cvt_pk_f32_fp8). If the kernels JIT-build at runtime against
# that list they fail on gfx1100 -> silent fallback to Triton, and (worse) ranks
# can diverge (some HIP, some Triton) -> the whole TP group stalls on the slowest
# rank. Pin to gfx950 so every rank takes the HIP path. Override only if you know
# the target arch differs.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
if [ "$PYTORCH_ROCM_ARCH" != "gfx950" ]; then
    echo "[P0-WARN] PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH (not gfx950); smallm MXFP8 HIP kernels may JIT-fail on non-gfx950 archs -> Triton fallback."
fi
# Drop any stale multi-arch JIT artifacts so the pinned arch rebuilds cleanly.
rm -rf "${HOME}"/.cache/torch_extensions/*smallm* 2>/dev/null || true
echo "[P0-GUARD] PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH (smallm HIP kernels pinned single-arch)"

# ── Repo + InferenceX paths ─────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
INFERENCEX_PATH="${INFERENCEX_PATH:-/home/xingran.fan@amd.com/InferenceX}"
# benchmark_lib.sh gives us wait_for_server_ready + run_benchmark_serving +
# start/stop_gpu_monitor, identical to what the InferenceX recipe uses.
source "$INFERENCEX_PATH/benchmarks/benchmark_lib.sh"

# ── Workload shape (matches the InferenceX FP8 1k/1k reference) ─────────────────
MODEL="${MODEL:-/it-share-4/MiniMax-M3-FP8}"      # local MXFP8 checkpoint (quant_method=mxfp8)
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
# generate_sweep_configs.py: max-model-len = isl + osl + 256
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"
# Seq-len label for result filenames (1024->1k, 8192->8k, else raw tokens).
_isl_lbl=$([ $((ISL % 1024)) -eq 0 ] && echo "$((ISL/1024))k" || echo "${ISL}")
_osl_lbl=$([ $((OSL % 1024)) -eq 0 ] && echo "$((OSL/1024))k" || echo "${OSL}")
SEQ_LABEL="${_isl_lbl}${_osl_lbl}"

# ── Parallelism (cards 0-3 free => TP=4, the InferenceX tp:4 row) ───────────────
TP="${TP:-4}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${MM3_PORT:-8893}"
# InferenceX 1k/1k tp:4 row spans conc 1..64; user wants 1,4,8,16,32.
CONC_LIST="${CONC_LIST:-1 4 8 16 32}"
# Concurrencies listed in HALVE_CONCS use half the sample count (rounded up) to speed
# up the long high-conc points, e.g. HALVE_CONCS="64 128".
HALVE_CONCS="${HALVE_CONCS:-}"

# Pin to the free GPUs. The container sees all 8. Use a SINGLE mask only:
# setting both ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES composes them
# (ROCR remaps physical->logical, then HIP selects within that), so e.g.
# GPUS=4,5,6,7 would double-mask to nothing ("No HIP GPUs available").
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$GPUS}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# ── Run bookkeeping ─────────────────────────────────────────────────────────────
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/run-logs/minimaxm3-nonmtp-${SEQ_LABEL}-$RUN_TS}"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$REPO_ROOT/run-logs/minimaxm3-nonmtp-${SEQ_LABEL}-latest"
ln -sfn "$RUN_DIR" "$REPO_ROOT/run-logs/minimaxm3-nonmtp-latest"
SERVER_LOG="$RUN_DIR/server.log"
export GPU_METRICS_CSV="$RUN_DIR/gpu_metrics.csv"

# ── Profiling (operator-level torch traces; off by default) ─────────────────────
PROFILE="${PROFILE:-0}"
PROFILE_SERVE_ARGS=()
if [ "$PROFILE" = "1" ]; then
    # vLLM 0.22+ dropped the VLLM_TORCH_PROFILER_DIR env var in favour of the
    # nested profiler-config CLI flags; keep the env for older builds too.
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-$RUN_DIR/profiling}"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    PROFILE_SERVE_ARGS=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$VLLM_TORCH_PROFILER_DIR\"}")
    export PROFILE=1   # benchmark_lib adds --profile + caps num_prompts when PROFILE=1
fi

# ── vLLM serve recipe knobs (copied 1:1 from the InferenceX recipe) ─────────────
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# ── Shared-expert fusion (vLLM PR #46545; off by default = upstream default) ────
# MiniMax-M3 runs its always-on shared expert as a SEPARATE dense MLP every MoE
# layer (gate_up GEMM + act + down GEMM, x60 layers). Folding it into the routed
# grouped GEMM removes those per-layer launches — the dominant cost at low/medium
# concurrency, where decode is launch-bound. Numerically equivalent to the
# separate-MLP path (gsm8k unchanged); measured +30% tok/s @ conc=1 … +5.6% @
# conc=128 on MM3 MXFP8 / TP4 / TRITON_ATTN. Backend-neutral (triton/flydsl mxfp8
# MoE), so it does NOT need the aiter master switch. Full writeup + caveats:
#   Hyperloom/hyperloom/kb/fusion/empirical_kb.md
# Constraints: ROCm only; needs n_shared_experts + gated activation; must NOT run
# under expert parallelism (the appended shared slot isn't handled by the EP
# expert_map path). Enable with FUSE_SHARED_EXPERTS=1.
FUSE_SHARED_EXPERTS="${FUSE_SHARED_EXPERTS:-0}"
if [ "$FUSE_SHARED_EXPERTS" = "1" ]; then
    if [ "$EP_SIZE" -gt 1 ] || [ "$DP_ATTENTION" = "true" ]; then
        echo "[FSE-WARN] FUSE_SHARED_EXPERTS=1 ignored: shared-expert fusion is unsupported under expert/data-parallel attention (EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION). Keep EP off to use it."
    else
        export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
        echo "[FSE] shared-expert fusion ON (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1) — see hyperloom/kb/fusion/empirical_kb.md"
    fi
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP" --enable-expert-parallel)
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

echo "============================================================"
echo " MiniMax-M3 MXFP8 MI355X — NON-MTP baseline sweep"
echo "   Model:        $MODEL"
echo "   GPUs:         $ROCR_VISIBLE_DEVICES  (TP=$TP EP=$EP_SIZE DPATTN=$DP_ATTENTION)"
echo "   Workload:     ISL=$ISL OSL=$OSL RRR=$RANDOM_RANGE_RATIO MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "   Concurrency:  $CONC_LIST"
echo "   Port:         $PORT      Profile: $PROFILE"
echo "   Run dir:      $RUN_DIR"
echo "============================================================"

start_gpu_monitor

# ── Optional knob overrides (env-driven; default reproduces the baseline) ───────
KNOB_ARGS=()
if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    KNOB_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
# Build a single --compilation-config JSON from the cudagraph knobs (can only be
# passed once). CUDAGRAPH_MODE e.g. FULL / FULL_AND_PIECEWISE / PIECEWISE.
_cc_fields=()
[ -n "${CUDAGRAPH_MODE:-}" ]         && _cc_fields+=("\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"")
[ -n "${CUDAGRAPH_CAPTURE_SIZE:-}" ] && _cc_fields+=("\"max_cudagraph_capture_size\": $CUDAGRAPH_CAPTURE_SIZE")
# pass_config sub-object (e.g. sequence parallelism / allreduce-rms fusion).
_pc_fields=()
[ -n "${ENABLE_SP:-}" ]        && _pc_fields+=("\"enable_sp\": $ENABLE_SP")
# M3 hidden_size(6144) 低于 SP 阈值启发式 -> enable_sp 会被自动关掉;显式给
# sp_min_token_num 强制开启(任何 >= 该 token 数的 batch 走 SP)。
[ -n "${SP_MIN_TOKEN_NUM:-}" ] && _pc_fields+=("\"sp_min_token_num\": $SP_MIN_TOKEN_NUM")
[ -n "${FUSE_ALLREDUCE_RMS:-}" ] && _pc_fields+=("\"fuse_allreduce_rms\": $FUSE_ALLREDUCE_RMS")
if [ "${#_pc_fields[@]}" -gt 0 ]; then
    _cc_fields+=("\"pass_config\": {$(IFS=,; echo "${_pc_fields[*]}")}")
fi
if [ "${#_cc_fields[@]}" -gt 0 ]; then
    _cc_json="{$(IFS=,; echo "${_cc_fields[*]}")}"
    KNOB_ARGS+=(--compilation-config "$_cc_json")
fi
echo "[knobs] KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-default}  CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-default}  CUDAGRAPH_CAPTURE_SIZE=${CUDAGRAPH_CAPTURE_SIZE:-default}  ENABLE_SP=${ENABLE_SP:-default}  QR=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-default}  ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}  AITER=${VLLM_ROCM_USE_AITER:-0}/MHA=${VLLM_ROCM_USE_AITER_MHA:-default}  FUSE_SHARED_EXPERTS=${FUSE_SHARED_EXPERTS:-0}(FSE=${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-0})"

# ── Launch the server ONCE (background), wait for /health ───────────────────────
set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
    --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    "${KNOB_ARGS[@]}" \
    "${PROFILE_SERVE_ARGS[@]}" \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

cleanup() {
    echo "[cleanup] stopping server PID $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    stop_gpu_monitor 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ── Concurrency sweep — one benchmark_serving run per conc, server stays up ─────
SUMMARY="$RUN_DIR/summary.csv"
echo "conc,output_tok_per_s,total_tok_per_s,req_per_s,mean_ttft_ms,mean_tpot_ms,p99_tpot_ms" > "$SUMMARY"

for CONC in $CONC_LIST; do
    export CONC
    # num-prompts = CONC * NPROMPT_MULT (default 10), floored at 10 and optionally
    # capped at NPROMPT_CAP — lets a quick sweep cut sample counts at high conc.
    NPROMPTS=$(( CONC * ${NPROMPT_MULT:-10} ))
    # Per-conc halving for the configured high-conc points (e.g. 64/128).
    for _h in $HALVE_CONCS; do [ "$_h" = "$CONC" ] && NPROMPTS=$(( (NPROMPTS + 1) / 2 )); done
    if [ "$NPROMPTS" -lt 10 ]; then NPROMPTS=10; fi
    if [ -n "${NPROMPT_CAP:-}" ] && [ "$NPROMPTS" -gt "$NPROMPT_CAP" ]; then NPROMPTS="$NPROMPT_CAP"; fi
    RESULT_NAME="bmk_minimaxm3_${SEQ_LABEL}_fp8_vllm_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_nonmtp_conc${CONC}"
    echo ">>> [conc=$CONC] benchmarking ($NPROMPTS prompts) -> $RESULT_NAME.json"
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NPROMPTS" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_NAME" \
        --result-dir "$RUN_DIR" \
        --bench-serving-dir "$INFERENCEX_PATH" \
        --server-pid "$SERVER_PID" \
        --trust-remote-code || echo "!!! conc=$CONC benchmark failed (continuing)"

    # Pull the key metrics into a one-line-per-conc summary for quick alignment.
    python3 - "$RUN_DIR/$RESULT_NAME.json" "$CONC" "$SUMMARY" <<'PYEOF' || true
import json, sys
path, conc, summary = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"  (no result json: {e})"); sys.exit(0)
row = [conc,
       round(d.get("output_throughput", 0), 2),
       round(d.get("total_token_throughput", 0), 2),
       round(d.get("request_throughput", 0), 4),
       round(d.get("mean_ttft_ms", 0), 2),
       round(d.get("mean_tpot_ms", 0), 3),
       round(d.get("p99_tpot_ms", 0), 3)]
open(summary, "a").write(",".join(str(x) for x in row) + "\n")
print(f"  conc={conc}: output={row[1]} tok/s  total={row[2]} tok/s  "
      f"ttft={row[4]}ms  tpot={row[5]}ms")
PYEOF
done

echo
echo "==================== NON-MTP SWEEP SUMMARY ===================="
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "=============================================================="
echo "Per-conc JSON + gpu_metrics.csv in: $RUN_DIR"
