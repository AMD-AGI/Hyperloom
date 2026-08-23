#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The-Monk
# SPDX-License-Identifier: MIT
#
# Hyperloom scriptable benchmark runner for AMD RDNA4 (gfx1201) via llama.cpp.
#
# WHY llama.cpp AND NOT vLLM/SGLANG: below ~32GB the serving stacks' VRAM and
# concurrency assumptions do not transfer, and llama.cpp is what this class of
# card is actually served with. This is a server-less "scriptable" workload:
# no OpenAI endpoint, no HTTP client -- run a benchmark, write one JSON.
#
# CONTRACT (from bypass_scriptable.py / benchmark_result.py):
#   in : $MODEL $RESULT_DIR $RESULT_FILENAME $RUNNER_TYPE  (Hyperloom sets these)
#   out: $RESULT_DIR/$RESULT_FILENAME.json, InferenceX-shaped. To be selectable,
#        output_throughput MUST be > 0 and, if a quality_gate is present, it
#        must pass. A gate that is absent is non-blocking; a gate that FAILS
#        makes the run unselectable no matter how fast it was -- which is the
#        point of shipping one.
#
# INSTALL: point Hyperloom at this file's directory --
#   export HYPERLOOM_BYPASS_SCRIPTS_DIR=/path/to/examples/rdna
#
# KNOBS (env): LLAMA_CPP_DIR|LLAMA_BENCH, PP, TG, REPS, NGL, EXTRA_BENCH_ARGS,
#              PPL_FILE, PPL_MAX, LLAMA_PERPLEXITY
set -euo pipefail

MODEL="${MODEL:?MODEL is required (path to a .gguf)}"
RESULT_DIR="${RESULT_DIR:-$PWD}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
RUNNER_TYPE="${RUNNER_TYPE:-r9700}"
PP="${PP:-1024}"        # prefill tokens  -> ttft / prompt throughput
TG="${TG:-128}"         # decode tokens   -> output_throughput (the headline)
REPS="${REPS:-3}"       # llama-bench repetitions; it reports avg + stddev
NGL="${NGL:-999}"       # offload everything to the GPU by default

# --- resolve llama-bench. FAIL LOUDLY: a missing binary must not read as a
# --- zero-throughput result, which is indistinguishable from "the GPU is slow".
LLAMA_BENCH="${LLAMA_BENCH:-}"
if [ -z "$LLAMA_BENCH" ]; then
    for c in "${LLAMA_CPP_DIR:-}/build/bin/llama-bench" \
             "${LLAMA_CPP_DIR:-}/llama-bench" \
             "$(command -v llama-bench 2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { LLAMA_BENCH="$c"; break; }
    done
fi
[ -n "$LLAMA_BENCH" ] && [ -x "$LLAMA_BENCH" ] || {
    echo "ERROR: llama-bench not found. Set LLAMA_BENCH=/path/to/llama-bench" >&2
    echo "       or LLAMA_CPP_DIR=/path/to/llama.cpp (expects build/bin/llama-bench)." >&2
    exit 3; }
[ -f "$MODEL" ] || { echo "ERROR: MODEL not found: $MODEL" >&2; exit 3; }

mkdir -p "$RESULT_DIR"
OUT="$RESULT_DIR/$RESULT_FILENAME.json"
BENCH_JSON="$RESULT_DIR/llama_bench.json"

# --- provenance. A number without the stack that produced it is not comparable
# --- across sessions, which is the whole reason Hyperloom keeps a KB.
# rocminfo is not always present (or not on PATH) even on a working ROCm box --
# it ships in rocminfo/rocm-core, which a minimal runtime install omits. Fall
# back to amd-smi, then to llama.cpp's own device line, so provenance degrades
# to "unknown" only when nothing on the system can name the arch.
GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
[ -z "$GFX" ] && GFX="$(amd-smi static -g 0 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
DRIVER="$(cat /sys/module/amdgpu/version 2>/dev/null || true)"
HIP_VER="$(hipconfig --version 2>/dev/null || true)"

echo ">> llama-bench: $LLAMA_BENCH"
echo ">> model      : $MODEL"
echo ">> config     : pp=$PP tg=$TG reps=$REPS ngl=$NGL arch=${GFX:-unknown}"

START=$(date +%s)
# shellcheck disable=SC2086  # EXTRA_BENCH_ARGS is a deliberate word-split knob
"$LLAMA_BENCH" -m "$MODEL" -p "$PP" -n "$TG" -ngl "$NGL" -r "$REPS" \
    ${EXTRA_BENCH_ARGS:-} -o json > "$BENCH_JSON"
END=$(date +%s)

# --- optional correctness gate. Speed without a correctness gate is how a
# --- wrong-but-fast kernel wins a benchmark; PPL_MAX makes that unselectable.
PPL_JSON=null
if [ -n "${PPL_FILE:-}" ]; then
    LLAMA_PPL="${LLAMA_PERPLEXITY:-$(dirname "$LLAMA_BENCH")/llama-perplexity}"
    if [ -x "$LLAMA_PPL" ]; then
        echo ">> perplexity gate: $PPL_FILE (max ${PPL_MAX:-unset})"
        PPL_LOG="$RESULT_DIR/perplexity.log"
        "$LLAMA_PPL" -m "$MODEL" -f "$PPL_FILE" -ngl "$NGL" > "$PPL_LOG" 2>&1 || true
        PPL_VAL="$(grep -oE 'Final estimate: PPL = [0-9.]+' "$PPL_LOG" | grep -oE '[0-9.]+$' | tail -1 || true)"
        PPL_JSON="{\"metric\":\"perplexity\",\"value\":${PPL_VAL:-null},\"threshold\":${PPL_MAX:-null}}"
    else
        echo "WARN: PPL_FILE set but llama-perplexity not found; no gate emitted." >&2
    fi
fi

RESULT_DIR="$RESULT_DIR" OUT="$OUT" BENCH_JSON="$BENCH_JSON" MODEL="$MODEL" \
RUNNER_TYPE="$RUNNER_TYPE" PP="$PP" TG="$TG" REPS="$REPS" NGL="$NGL" \
GFX="$GFX" DRIVER="$DRIVER" HIP_VER="$HIP_VER" PPL_JSON="$PPL_JSON" \
DURATION="$((END - START))" python3 - <<'PY'
import json, os

rows = json.load(open(os.environ["BENCH_JSON"]))
def pick(kind):
    """avg_ts for the pp (n_prompt>0) or tg (n_gen>0) row."""
    for r in rows:
        if kind == "pp" and int(r.get("n_prompt") or 0) > 0 and int(r.get("n_gen") or 0) == 0:
            return r
        if kind == "tg" and int(r.get("n_gen") or 0) > 0 and int(r.get("n_prompt") or 0) == 0:
            return r
    return None

pp, tg = pick("pp"), pick("tg")
pp_ts = float(pp["avg_ts"]) if pp else None   # prompt tokens/s
tg_ts = float(tg["avg_ts"]) if tg else None   # decode tokens/s == the headline

if not tg_ts or tg_ts <= 0:
    # Refuse to emit a "valid" result with no decode throughput: Hyperloom would
    # treat 0 as a legitimate slow variant rather than as a broken run.
    raise SystemExit("FATAL: no decode (tg) throughput parsed from llama-bench output")

n_pp, n_tg = int(os.environ["PP"]), int(os.environ["TG"])
ppl = os.environ.get("PPL_JSON", "null")
gate = json.loads(ppl) if ppl and ppl != "null" else None
if gate is not None:
    v, thr = gate.get("value"), gate.get("threshold")
    gate["passed"] = bool(v is not None and thr is not None and float(v) <= float(thr))
    if thr is None:
        # Measured but no threshold to judge against: record it, do not claim a verdict.
        gate.pop("passed", None)
        gate["skipped"] = True
        gate["reason"] = "no PPL_MAX threshold supplied"

report = {
    "success": True,
    "framework": "llamacpp",
    "model": os.environ["MODEL"],
    "workload_kind": "scriptable",
    "throughput_unit": "tokens/s",
    "throughput": {
        "output_throughput": tg_ts,
        "total_token_throughput": tg_ts,
        "completed_requests": int(os.environ["REPS"]),
        "duration_seconds": float(os.environ["DURATION"]),
    },
    "latency": {
        # llama-bench reports rates, not per-request latency. TPOT is the exact
        # reciprocal of decode t/s; TTFT is DERIVED from prefill rate (prompt
        # tokens / prompt t/s) and is therefore a model of first-token time, not
        # a measurement of one. Labelled so nobody reads it as measured.
        "tpot": {"mean_ms": 1000.0 / tg_ts},
        "ttft": {"mean_ms": (n_pp / pp_ts * 1000.0) if pp_ts else None},
    },
    "quality_gate": gate,
    "rdna": {
        "runner_type": os.environ["RUNNER_TYPE"],
        "gfx_arch": os.environ.get("GFX") or None,
        "amdgpu_driver": os.environ.get("DRIVER") or None,
        "hip_version": os.environ.get("HIP_VER") or None,
        "prompt_tokens": n_pp,
        "gen_tokens": n_tg,
        "n_gpu_layers": int(os.environ["NGL"]),
        "prompt_throughput_tps": pp_ts,
        "ttft_ms_is_derived": True,
        "llama_bench_rows": rows,
    },
}
json.dump(report, open(os.environ["OUT"], "w"), indent=2)
print(f">> wrote {os.environ['OUT']}")
print(f">> prefill {pp_ts:.2f} t/s | decode {tg_ts:.2f} t/s" if pp_ts else f">> decode {tg_ts:.2f} t/s")
PY
