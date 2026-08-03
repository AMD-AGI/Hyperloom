#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

if ! command -v hipcc &>/dev/null && [ -x /opt/rocm/bin/hipcc ]; then
    export PATH="/opt/rocm/bin:${PATH}"
fi

if [ "${WORLDPLAY_USE_FP8_GEMMS:-0}" != "0" ] || [ "${WORLDPLAY_USE_FP8_GEMM:-0}" != "0" ]; then
    echo "[worldplay][lock] FP8 GEMM is ignored because WorldPlay is BF16 locked."
fi
export WORLDPLAY_USE_FP8_GEMMS=0
export WORLDPLAY_USE_FP8_GEMM=0
export WORLDPLAY_USE_FP4_GEMMS=0

unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    n=$(echo "${ROCR_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1)))
fi
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set by Magpie}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
OUTPUT_FILE="${RESULT_DIR}/${RESULT_FILENAME}.json"
mkdir -p "${RESULT_DIR}"

MODEL_PATH="${MODEL:?MODEL must be set}"
ACTION_CKPT="${WORLDPLAY_ACTION_CKPT:-}"
WORLDPLAY_DIR="${WORLDPLAY_DIR:?WORLDPLAY_DIR must be set}"
BENCH_PY="${WORLDPLAY_BENCH:?WORLDPLAY_BENCH must be set}"
TORCHRUN="${WORLDPLAY_TORCHRUN:-${WORLDPLAY_DIR}/.venv/bin/torchrun}"
[ -x "${TORCHRUN}" ] || TORCHRUN="$(command -v torchrun)"

TP_DEG="${TP:-8}"
HEIGHT="${WORLDPLAY_HEIGHT:-480}"
WIDTH="${WORLDPLAY_WIDTH:-832}"
NUM_FRAMES="${WORLDPLAY_NUM_FRAMES:-125}"
NUM_STEPS="${WORLDPLAY_NUM_STEPS:-50}"
WARMUP="${WORLDPLAY_WARMUP_CHUNKS:-1}"
REPEATS="${WORLDPLAY_REPEATS:-3}"
MODEL_TYPE="${WORLDPLAY_MODEL_TYPE:-ar}"
OFFLOADING="${WORLDPLAY_OFFLOADING:-1}"
RESIDENT="${WORLDPLAY_TRANSFORMER_RESIDENT:-0}"
GROUP_OFFLOAD="${WORLDPLAY_GROUP_OFFLOADING:-}"

EXTRA_FLAGS=(--offloading "${OFFLOADING}" --transformer_resident_ar_rollout "${RESIDENT}")
[ -n "${MODEL_PATH}" ] && EXTRA_FLAGS+=(--model_path "${MODEL_PATH}")
[ -n "${ACTION_CKPT}" ] && EXTRA_FLAGS+=(--action_ckpt "${ACTION_CKPT}")
[ "${WORLDPLAY_USE_TORCH_COMPILE:-0}" = "1" ] && EXTRA_FLAGS+=(--enable_torch_compile)
[ "${WORLDPLAY_FEW_STEP:-0}" = "1" ] && EXTRA_FLAGS+=(--few_step)
[ -n "${GROUP_OFFLOAD}" ] && EXTRA_FLAGS+=(--group_offloading "${GROUP_OFFLOAD}")

QUALITY_FLAGS=()
[ -n "${XDIT_QUALITY_REF:-}" ] && QUALITY_FLAGS+=(--quality-ref "${XDIT_QUALITY_REF}")
[ -n "${XDIT_QUALITY_REF_WRITE:-}" ] && QUALITY_FLAGS+=(--quality-ref-write "${XDIT_QUALITY_REF_WRITE}")
[ -n "${WORLDPLAY_QUALITY_SSIM_MIN:-}" ] && QUALITY_FLAGS+=(--quality-ssim-min "${WORLDPLAY_QUALITY_SSIM_MIN}")
[ -n "${WORLDPLAY_QUALITY_LPIPS_MAX:-}" ] && QUALITY_FLAGS+=(--quality-lpips-max "${WORLDPLAY_QUALITY_LPIPS_MAX}")
[ -n "${WORLDPLAY_QUALITY_MSE_MAX:-}" ] && QUALITY_FLAGS+=(--quality-mse-max "${WORLDPLAY_QUALITY_MSE_MAX}")
[ -n "${WORLDPLAY_QUALITY_FRAMES:-}" ] && QUALITY_FLAGS+=(--quality-frames "${WORLDPLAY_QUALITY_FRAMES}")

POSE="${WORLDPLAY_POSE:-w-$(( (NUM_FRAMES - 1) / 4 ))}"
BENCH_OUT="${RESULT_DIR}/worldplay_bench.json"

PROFILE_FLAGS=()
if [ "${PROFILE:-0}" = "1" ] && [ "${WORLDPLAY_SUPPORTS_PROFILER:-1}" != "0" ]; then
    TORCH_TRACE_DIR="${RESULT_DIR}/torch_trace"
    mkdir -p "${TORCH_TRACE_DIR}"
    PROFILE_FLAGS+=(--torch_profiler_dir "${TORCH_TRACE_DIR}")
    REPEATS="${WORLDPLAY_PROFILE_REPEATS:-1}"
fi

echo "=== Magpie WorldPlay bench (BF16, runner=${RUNNER_TYPE:-unknown}) ==="
echo "  model=${MODEL_PATH} res=${WIDTH}x${HEIGHT} frames=${NUM_FRAMES} steps=${NUM_STEPS} sp=${TP_DEG}"
echo "  offloading=${OFFLOADING} resident=${RESIDENT} group_offload=${GROUP_OFFLOAD:-<pipe-default>} compile=${WORLDPLAY_USE_TORCH_COMPILE:-0}"
echo "  torchrun=${TORCHRUN} bench=${BENCH_PY}"

export PYTHONPATH="${WORLDPLAY_DIR}:${PYTHONPATH:-}"

START_NS=$(date +%s%N)
# shellcheck disable=SC2086
"${TORCHRUN}" --nproc_per_node="${TP_DEG}" --master_port="${WORLDPLAY_MASTER_PORT:-29533}" \
    "${BENCH_PY}" \
    --worldplay-dir "${WORLDPLAY_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --video_length "${NUM_FRAMES}" \
    --num_inference_steps "${NUM_STEPS}" \
    --pose "${POSE}" \
    --height "${HEIGHT}" --width "${WIDTH}" \
    --warmup "${WARMUP}" --repeats "${REPEATS}" \
    --tag "worldplay_${RUNNER_TYPE:-mi355x}_sp${TP_DEG}" \
    --out "${BENCH_OUT}" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    ${QUALITY_FLAGS[@]+"${QUALITY_FLAGS[@]}"} \
    ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"} \
    ${EXTRA_WORLDPLAY_ARGS:-}
END_NS=$(date +%s%N)
WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

REPORT_FILE="${RESULT_DIR}/benchmark_report.json"
python3 - "${BENCH_OUT}" "${OUTPUT_FILE}" "${WALL_MS}" "${MODEL_PATH}" "${REPORT_FILE}" <<'PYEOF'
import json
import sys

bench_out, output_path, wall_ms_str, model_path, report_path = sys.argv[1:6]
wall_ms = int(wall_ms_str)
try:
    with open(bench_out, encoding="utf-8") as fh:
        data = json.load(fh)
except Exception as exc:
    data = {}
    print(f"[worldplay][warn] could not read bench JSON {bench_out}: {exc}", file=sys.stderr)

summary = data.get("summary") or {}
steady = summary.get("steadystate_fps") or {}
overall = summary.get("overall_fps") or {}
fps = steady.get("mean")
if fps is None:
    fps = overall.get("mean")
fps = float(fps) if fps is not None else 0.0
cfg = data.get("config") or {}
completed = int(data.get("completed_repeats", 0) or 0)
planned = int(data.get("planned_repeats", completed) or completed)
per_frame_ms = (1000.0 / fps) if fps > 0 else None

quality_gate = data.get("quality_gate")
if not isinstance(quality_gate, dict):
    quality_gate = {"skipped": True, "reason": "worldplay_quality_gate_not_emitted"}

result = {
    "framework": "worldplay",
    "model": model_path,
    "workload_kind": "scriptable",
    "throughput_unit": "fps",
    "output_throughput": round(fps, 6),
    "request_throughput": round(fps, 6),
    "completed": completed,
    "num_prompts": planned,
    "duration": round(wall_ms / 1000.0, 3),
    "frames_per_run": cfg.get("video_length"),
    "mean_e2el_ms": (round(per_frame_ms, 3) if per_frame_ms else None),
    "precision_locked": "bf16",
    "quality_gate": quality_gate,
    "bench_summary": summary,
    "bench_config": cfg,
}
with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)

report = {
    "framework": "worldplay",
    "workload_kind": "scriptable",
    "throughput_unit": "fps",
    "output_throughput": round(fps, 6),
    "quality_gate": quality_gate,
}
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)

qsummary = quality_gate.get("reason") if quality_gate.get("skipped") else (
    f"passed={quality_gate.get('passed')} ssim={quality_gate.get('ssim')} "
    f"mse={quality_gate.get('mse')} lpips={quality_gate.get('lpips')}"
)
print(f"[worldplay] steadystate={fps:.4f} fps ({completed}/{planned} repeats) "
      f"quality[{qsummary}] -> {output_path}")
PYEOF

echo "=== Magpie WorldPlay bench complete ==="
