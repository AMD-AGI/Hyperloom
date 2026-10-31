#!/usr/bin/env bash
# HyperloomHPC adapter — minimap2 (genomics long-read alignment) on AMD GPUs.
#
# Workload source: https://github.com/AMD-Ecosystem/minimap2  branch `rocm-enablement`
#   cmake -S . -B build -DGPU=AMD -DGPUARCH=gfx950 -DROCM_PATH=$(rocm-sdk path --root)
#   cmake --build build -j
#
# What the loop sweeps here are minimap2's GPU knobs, with --gpu-chain always on.
# This is NOT a CPU-vs-GPU comparison -- the baseline is the GPU path at its
# default configuration, and the objective is to configure it better.
#
# Env the adapter reads (all optional, all overridable by a lever variant):
#   MINIMAP2_BIN            path to the built binary
#   MINIMAP2_REF            reference FASTA
#   MINIMAP2_READS          reads FASTQ
#   MINIMAP2_PRESET         -x preset (default map-ont)
#   MINIMAP2_THREADS        host threads (default 32)
#   MINIMAP2_ITERS          timed iterations (default 3)
#   MINIMAP2_GPU_CHAIN_WORKERS / _FLUSH_THRESHOLD / _BATCH_MAX_ALIGN / _BATCH_MAX_MEM
#   MINIMAP2_EXTRA_ARGS     free-form extra argv
#   MINIMAP2_REF_PAF        baseline PAF for the quality gate
#   MINIMAP2_QUALITY_MIN    min alignment recovery (default 0.98)
set -euo pipefail

BIN="${MINIMAP2_BIN:?set MINIMAP2_BIN to the built minimap2}"
REF="${MINIMAP2_REF:?set MINIMAP2_REF}"
READS="${MINIMAP2_READS:?set MINIMAP2_READS}"
PRESET="${MINIMAP2_PRESET:-map-ont}"
THREADS="${MINIMAP2_THREADS:-32}"
ITERS="${MINIMAP2_ITERS:-3}"
QMIN="${MINIMAP2_QUALITY_MIN:-0.98}"
RESULT_DIR="${RESULT_DIR:-/tmp/hlhpc-minimap2}"; mkdir -p "$RESULT_DIR"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
OUT="$RESULT_DIR/out.paf"; ERRLOG="$RESULT_DIR/minimap2.err"

GPU=(--gpu-chain)
[[ -n "${MINIMAP2_GPU_CHAIN_WORKERS:-}"  ]] && GPU+=(--gpu-chain-workers    "$MINIMAP2_GPU_CHAIN_WORKERS")
[[ -n "${MINIMAP2_GPU_FLUSH_THRESHOLD:-}" ]] && GPU+=(--gpu-flush-threshold "$MINIMAP2_GPU_FLUSH_THRESHOLD")
[[ -n "${MINIMAP2_GPU_BATCH_MAX_ALIGN:-}" ]] && GPU+=(--gpu-batch-max-align "$MINIMAP2_GPU_BATCH_MAX_ALIGN")
[[ -n "${MINIMAP2_GPU_BATCH_MAX_MEM:-}"   ]] && GPU+=(--gpu-batch-max-mem   "$MINIMAP2_GPU_BATCH_MAX_MEM")
read -r -a EXTRA <<< "${MINIMAP2_EXTRA_ARGS:-}"

run_once() {  # -> elapsed ms on stdout
    local s e
    s=$(date +%s%N)
    "$BIN" -t "$THREADS" "${GPU[@]}" "${EXTRA[@]}" -x "$PRESET" "$REF" "$READS" > "$1" 2>"$ERRLOG"
    e=$(date +%s%N)
    echo $(( (e - s) / 1000000 ))
}

echo "[minimap2] warmup (untimed: pays index build + GPU init once)"
run_once /dev/null >/dev/null

total_ms=0; total_aln=0
for i in $(seq 1 "$ITERS"); do
    ms=$(run_once "$OUT")
    n=$(grep -vc '^@' "$OUT" || true); n=${n:-0}
    [[ "$n" -gt 0 ]] || { echo "[minimap2] ERROR: iter $i produced 0 alignments" >&2; tail -5 "$ERRLOG" >&2; exit 1; }
    total_ms=$(( total_ms + ms )); total_aln=$(( total_aln + n ))
    echo "[minimap2] iter $i/$ITERS: $n alignments in ${ms}ms"
done
mean_ms=$(( total_ms / ITERS )); mean_aln=$(( total_aln / ITERS ))
tput=$(python3 -c "print(f'{$mean_aln*1000/$mean_ms:.2f}')")

# --- quality gate: alignment recovery against a baseline PAF ------------------
passed=true; score=1.0
if [[ -n "${MINIMAP2_REF_PAF:-}" && -f "${MINIMAP2_REF_PAF}" ]]; then
    ref_n=$(grep -vc '^@' "$MINIMAP2_REF_PAF" || true)
    read -r score passed < <(python3 -c "
m=$mean_aln; r=$ref_n; s=(min(m,r)/r) if r else 0.0
print(f'{s:.4f}', 'true' if s >= $QMIN else 'false')")
    echo "[minimap2] gate: $mean_aln vs $ref_n recovered ($score), threshold $QMIN -> passed=$passed"
else
    echo "[minimap2] gate: no MINIMAP2_REF_PAF -- recording recovery 1.0 against this run itself"
fi

cat > "$RESULT_DIR/$RESULT_FILENAME.json" <<JSON
{
  "output_throughput": $tput,
  "throughput_unit": "alignments/s (minimap2 $PRESET, GPU chaining)",
  "mean_latency_ms": $mean_ms,
  "quality_gate": {
    "passed": $passed,
    "alignment_recovery": $score,
    "metric": "PAF alignment recovery vs baseline, threshold $QMIN"
  },
  "workload": {
    "name": "minimap2",
    "runner_type": "${RUNNER_TYPE:-}",
    "config": {
      "preset": "$PRESET", "threads": $THREADS, "iters": $ITERS,
      "gpu_chain_workers": "${MINIMAP2_GPU_CHAIN_WORKERS:-auto}",
      "extra_args": "${MINIMAP2_EXTRA_ARGS:-}"
    }
  }
}
JSON
echo "[minimap2] $tput aln/s, gate passed=$passed -> $RESULT_DIR/$RESULT_FILENAME.json"
