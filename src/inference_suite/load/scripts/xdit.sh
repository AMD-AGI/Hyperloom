#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# xDiT diffusion benchmark (server-less): runs the xdit CLI once, parses the per-image latency, scores the last image
# against a BF16 reference when one is given, and writes the result JSON to ${RESULT_DIR}/${RESULT_FILENAME}.json.
# Its quality_gate carries the scores raw, or says the reference was established; whoever reads the result judges it.
#
# Set by inference_suite.load.scriptable: RESULT_DIR, RESULT_FILENAME, MODEL, TP, ROCR_VISIBLE_DEVICES and, when the
# workload names one, QUALITY_REFERENCE. The workload's params set the knobs below.
#   XDIT_ATTENTION_BACKEND (aiter) XDIT_USE_TORCH_COMPILE (1) XDIT_HEIGHT (1024) XDIT_WIDTH (1024)
#   XDIT_NUM_STEPS (28) XDIT_NUM_ITERATIONS (25) XDIT_WARMUP_CALLS (5) XDIT_GUIDANCE_SCALE (4.0) XDIT_SEED (42)
#   XDIT_MAX_SEQ_LEN (512) XDIT_ULYSSES_DEGREE (${TP}) XDIT_MODEL_ARG (name|path) XDIT_INPUT_IMAGE XDIT_PROMPT
#   XDIT_QUALITY_REF_WRITE (save the last image as the reference when no reference is given) EXTRA_XDIT_ARGS
# Precision is BF16: XDIT_USE_FP4_GEMMS / XDIT_USE_FP8_GEMMS are ignored, since another precision is another model.
set -euo pipefail

if ! command -v hipcc &>/dev/null && [ -x /opt/rocm/bin/hipcc ]; then
    export PATH="/opt/rocm/bin:${PATH}"
fi
for locked in XDIT_USE_FP4_GEMMS XDIT_USE_FP8_GEMMS; do
    if [ "${!locked:-0}" != "0" ]; then
        echo "[xdit][lock] ${locked}=${!locked} ignored: precision is BF16."
    fi
done

# ROCR_VISIBLE_DEVICES re-indexes the visible GPUs to 0..N-1, so HIP takes the logical range.
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    n=$(echo "${ROCR_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1)))
fi
export HSA_NO_SCRATCH_RECLAIM=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set}"
OUTPUT_FILE="${RESULT_DIR}/${RESULT_FILENAME:?RESULT_FILENAME must be set}.json"
mkdir -p "${RESULT_DIR}"

MODEL_PATH="${MODEL:?MODEL must be set}"
case "${XDIT_MODEL_ARG:-name}" in
    path) XDIT_MODEL_NAME="${MODEL_PATH}" ;;
    *)    XDIT_MODEL_NAME="${XDIT_MODEL_NAME:-$(basename "${MODEL_PATH}")}" ;;
esac

ATTENTION_BACKEND="${XDIT_ATTENTION_BACKEND:-aiter}"
ULYSSES="${XDIT_ULYSSES_DEGREE:-${TP:?TP must be set}}"
NUM_ITERATIONS="${XDIT_NUM_ITERATIONS:-25}"
QUALITY_REF="${QUALITY_REFERENCE:-}"

EXTRA_FLAGS=()
if [ "${XDIT_USE_TORCH_COMPILE:-1}" = "1" ]; then
    EXTRA_FLAGS+=(--use_torch_compile)
fi
case "${ATTENTION_BACKEND}" in
    aiter_fp8|aiter_sage|aiter_sage_v2|aiter_sparse_sage|aiter_sparse_sage_v2)
        echo "[xdit][warn] ${ATTENTION_BACKEND} is quantized attention, a known regression on FLUX; prefer aiter."
        ;;
esac

XDIT_INPUT_IMAGE="${XDIT_INPUT_IMAGE:-/app/data/flux_cat.png}"
if ! command -v xdit &>/dev/null; then
    echo "[xdit][fatal] the xdit binary is not on PATH" >&2
    exit 1
fi
if [ ! -f "${XDIT_INPUT_IMAGE}" ]; then
    echo "[xdit][warn] input image not found: ${XDIT_INPUT_IMAGE}" >&2
fi

echo "[xdit] model=${MODEL_PATH} attn=${ATTENTION_BACKEND} ulysses=${ULYSSES} iters=${NUM_ITERATIONS}"
echo "[xdit] quality_ref=${QUALITY_REF:-<none>} extra_xdit_args=${EXTRA_XDIT_ARGS:-<none>}"

XDIT_RUN_DIR="$(mktemp -d "${RESULT_DIR}/xdit_run.XXXXXX")"
XDIT_LOG="${XDIT_RUN_DIR}/xdit_stdout.log"
START_NS=$(date +%s%N)

# shellcheck disable=SC2086  # EXTRA_XDIT_ARGS is word-split on purpose.
xdit \
    --model "${XDIT_MODEL_NAME}" \
    --seed "${XDIT_SEED:-42}" \
    --prompt "${XDIT_PROMPT:-Add a cool hat to the cat}" \
    --height "${XDIT_HEIGHT:-1024}" \
    --width "${XDIT_WIDTH:-1024}" \
    --num_inference_steps "${XDIT_NUM_STEPS:-28}" \
    --max_sequence_length "${XDIT_MAX_SEQ_LEN:-512}" \
    --warmup_calls "${XDIT_WARMUP_CALLS:-5}" \
    --ulysses_degree "${ULYSSES}" \
    --guidance_scale "${XDIT_GUIDANCE_SCALE:-4.0}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --attention_backend "${ATTENTION_BACKEND}" \
    --input_images "${XDIT_INPUT_IMAGE}" \
    --output_directory "${XDIT_RUN_DIR}" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    ${EXTRA_XDIT_ARGS:-} \
    2>&1 | tee "${XDIT_LOG}"

WALL_MS=$(( ($(date +%s%N) - START_NS) / 1000000 ))

python3 - "${XDIT_LOG}" "${OUTPUT_FILE}" "${WALL_MS}" "${NUM_ITERATIONS}" "${XDIT_RUN_DIR}" \
          "${QUALITY_REF}" "${XDIT_QUALITY_REF_WRITE:-}" "${MODEL_PATH}" "${ATTENTION_BACKEND}" <<'PYEOF'
import glob
import json
import os
import re
import sys

log_path, output_path, wall_ms, num_iterations, run_dir, reference, reference_write, model, backend = sys.argv[1:10]
wall_s = int(wall_ms) / 1000.0
num_iterations = int(num_iterations)

with open(log_path, encoding="utf-8", errors="replace") as handle:
    log_text = handle.read()

LATENCY_PATTERNS = (
    r"Average time over \d+ runs:\s*([\d.]+)s",
    r"Iteration \d+ completed in ([\d.]+)s",
    r"100%\|[^|]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]",
    r"([\d.]+)s/it\]",
    r"(?:average|mean|avg)\s+(?:e2e|end.to.end|latency|time)\s*[:\s=]+\s*([\d.]+)\s*(?:s|sec)",
    r"(?:e2e|latency|inference)\s*[:\s=]+\s*([\d.]+)\s*s",
)
latency_s = None
latency_source = "wall_clock"
for pattern in LATENCY_PATTERNS:
    matches = re.findall(pattern, log_text, re.IGNORECASE)
    if matches:
        latency_s = float(matches[-1])
        latency_source = "log"
        break
if latency_s is None:
    latency_s = wall_s / max(num_iterations, 1)
throughput = 1.0 / latency_s if latency_s > 0 else 0.0

generated = sorted(glob.glob(os.path.join(run_dir, "*.png")) + glob.glob(os.path.join(run_dir, "*.jpg")))


def _scores(reference_path, image_path):
    """LPIPS, SSIM and MSE of the image against the reference; a score its library cannot compute is None."""
    import numpy as np
    from PIL import Image

    ref = np.array(Image.open(reference_path).convert("RGB")).astype("float32") / 255.0
    image = Image.open(image_path).convert("RGB")
    if (image.height, image.width) != ref.shape[:2]:
        image = image.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
    gen = np.array(image).astype("float32") / 255.0
    scores = {"mse": round(float(np.mean((ref - gen) ** 2)), 6), "ssim": None, "lpips": None}
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        print(f"[xdit][warn] no SSIM: {exc}", file=sys.stderr)
    else:
        scores["ssim"] = round(float(structural_similarity(ref, gen, channel_axis=2, data_range=1.0)), 6)
    try:
        import lpips
        import torch
    except ImportError as exc:
        print(f"[xdit][warn] no LPIPS: {exc}", file=sys.stderr)
    else:
        ref_t, gen_t = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2 - 1 for array in (ref, gen))
        with torch.no_grad():
            distance = lpips.LPIPS(net="alex", verbose=False)(ref_t, gen_t)
        scores["lpips"] = round(float(distance.item()), 6)
    return scores


quality_gate = None
if reference and generated:
    quality_gate = {"reference": reference, "image": os.path.basename(generated[-1]), **_scores(reference, generated[-1])}
elif reference_write and generated:
    from PIL import Image

    os.makedirs(os.path.dirname(reference_write) or ".", exist_ok=True)
    Image.open(generated[-1]).convert("RGB").save(reference_write)
    print(f"[xdit] wrote the quality reference {reference_write}", file=sys.stderr)
    quality_gate = {"skipped": True, "reason": "reference_established", "reference": reference_write}

result = {
    "framework": "xdit",
    "model": model,
    "workload_kind": "scriptable",
    "throughput_unit": "img/s",
    "output_throughput": round(throughput, 6),
    "request_throughput": round(throughput, 6),
    "completed": len(generated),
    "num_prompts": num_iterations,
    "duration": round(wall_s, 3),
    "latency_s": round(latency_s, 4),
    "latency_source": latency_source,
    "mean_e2el_ms": round(latency_s * 1000.0, 3),
    "attention_backend": backend,
    "precision_locked": "bf16",
    "quality_gate": quality_gate,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(f"[xdit] throughput={throughput:.4f} img/s latency={latency_s:.3f}s completed={len(generated)} -> {output_path}")
PYEOF
