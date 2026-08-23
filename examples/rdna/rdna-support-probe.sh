#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The-Monk
# SPDX-License-Identifier: MIT
#
# RDNA support probe — produces the evidence needed to CONFIRM support on this
# machine, or to show exactly what is missing to ADD it.
#
# Prints a paste-ready markdown report and exits with the verdict:
#   0 = CONFIRMED    correctness PASSED *and* a measurement passed Hyperloom's validator
#   1 = PARTIAL      board resolves, but one of those two halves is missing
#   2 = UNSUPPORTED  arch is not mapped (see AGENTS.md §2)
#
#   ./rdna-support-probe.sh              identity + resolution only
#   ./rdna-support-probe.sh --bench      correctness gate + benchmark + harness verdict
#
# CORRECTNESS IS NOT OPTIONAL. Throughput alone cannot confirm support: a kernel
# that is fast and WRONG will produce a beautiful number. --bench runs
# llama.cpp's test-backend-ops (every op compared against the CPU reference on
# this arch) and refuses CONFIRMED without it.
#
# Env for --bench: MODEL=/path/*.gguf  plus LLAMA_BENCH= or LLAMA_CPP_DIR=
# Optional: PY=<python>  GPU=<index, default 0>  TEST_BACKEND_OPS=<path>
#           CORRECTNESS_OPS=MUL_MAT,MUL_MAT_ID   (comma-separated, or ALL)
#           PPL_FILE=<corpus> PPL_MAX=<float>    (adds a model-level gate too)
set -uo pipefail

BENCH=0; [ "${1:-}" = "--bench" ] && BENCH=1
GPU="${GPU:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
u(){ echo "${1:-unknown}"; }

# --- identity. amd-smi first (present on modern ROCm), then rocm-smi, then
# --- rocminfo; each can be absent on a perfectly working box.
SMI_STATIC="$(amd-smi static -g "$GPU" 2>/dev/null || true)"
PRODUCT="$(sed -n 's/.*MARKET_NAME:[[:space:]]*//p' <<<"$SMI_STATIC" | head -1)"
[ -z "$PRODUCT" ] && PRODUCT="$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card Series:[[:space:]]*//p' | head -1)"
CUS="$(sed -n 's/.*NUM_COMPUTE_UNITS:[[:space:]]*//p' <<<"$SMI_STATIC" | head -1)"
GFX="$(grep -oE 'gfx[0-9a-f]+' <<<"$SMI_STATIC" | head -1)"
[ -z "$GFX" ] && GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
# VRAM size is nested under a "VRAM:" block, not a flat VRAM_SIZE key.
VRAM="$(awk '/^[[:space:]]*VRAM:/{f=1;next} f&&/SIZE:/{print $2" "$3;exit}' <<<"$SMI_STATIC")"
# Count amdgpu cards ONLY. A server board's BMC (ASPEED etc.) and any
# simple-framebuffer appear under /sys/class/drm and would inflate this --
# reporting a 2-GPU host as 3-GPU is exactly the kind of wrong detail that
# makes a support report untrustworthy.
NGPU="$(for d in /sys/class/drm/card*/device/driver; do
            [ -e "$d" ] && basename "$(readlink -f "$d")"; done 2>/dev/null | grep -c '^amdgpu$')"
[ "${NGPU:-0}" = "0" ] && NGPU="$(amd-smi list 2>/dev/null | grep -c '^GPU:')"
[ "${NGPU:-0}" = "0" ] && NGPU="unknown"

# --- stack
KERNEL="$(uname -r)"
DRIVER="$(cat /sys/module/amdgpu/version 2>/dev/null || true)"
ROCM_V="$(amd-smi version 2>/dev/null | sed -n 's/.*ROCm version:[[:space:]]*\([^ |]*\).*/\1/p' | head -1)"
[ -z "$ROCM_V" ] && ROCM_V="$(cat /opt/rocm/.info/version 2>/dev/null || hipconfig --version 2>/dev/null || true)"

# --- pick a python that can import the package
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in python3 "$REPO/.venv/bin/python" /usr/bin/python3; do
        command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
        PYTHONPATH="$REPO/src" "$c" -c "import hyperloom.common.gpu_identity" >/dev/null 2>&1 \
            && { PY="$c"; break; }
    done
fi

RESOLVED="-"; IDENTITY="-"; RUNNER="-"
if [ -n "$PY" ]; then
    read -r RESOLVED IDENTITY RUNNER <<<"$(GFX="$GFX" PYTHONPATH="$REPO/src" "$PY" - <<'PYRESOLVE' 2>/dev/null
import os
from hyperloom.inference_optimizer.gpu_types import (
    _GFX_TO_RUNNER, _resolve_amd_gpu_type, amd_gpu_dispatch_identity)
ident = amd_gpu_dispatch_identity()
print(_resolve_amd_gpu_type() or "-",
      f"{ident[0]}/{ident[1]}CU" if ident else "-",
      _GFX_TO_RUNNER.get(os.environ.get("GFX", ""), "-"))
PYRESOLVE
)"
    : "${RESOLVED:=-}" "${IDENTITY:=-}" "${RUNNER:=-}"
else
    echo "WARN: no python could import hyperloom from $REPO/src -- resolution unknown." >&2
    echo "      Set PY=/path/to/python (the one with the repo deps installed)." >&2
fi

# --- CORRECTNESS FIRST. A wrong-but-fast kernel is worth zero, so this gates
# --- the verdict; the benchmark below is only meaningful once this passes.
CORRECT="not run"; CORRECT_OK=0
if [ "$BENCH" = 1 ]; then
    TBO="${TEST_BACKEND_OPS:-}"
    if [ -z "$TBO" ]; then
        for c in "$(dirname "${LLAMA_BENCH:-/nonexistent}")/test-backend-ops" \
                 "${LLAMA_CPP_DIR:-}/build/bin/test-backend-ops" \
                 "$(command -v test-backend-ops 2>/dev/null || true)"; do
            [ -n "$c" ] && [ -x "$c" ] && { TBO="$c"; break; }
        done
    fi
    if [ -z "$TBO" ] || [ ! -x "$TBO" ]; then
        CORRECT="NOT RUN — test-backend-ops not found (build it, or set TEST_BACKEND_OPS)"
    else
        # ~90s on gfx1201 for the default pair. A one-time cost for a support
        # claim; narrow it with CORRECTNESS_OPS while iterating.
        OPS="${CORRECTNESS_OPS:-MUL_MAT,MUL_MAT_ID}"
        CLOG="$(mktemp)"; CORRECT_OK=1; PASSED_TOTAL=""
        if [ "$OPS" = "ALL" ]; then OPLIST=""; else OPLIST="$(tr ',' ' ' <<<"$OPS")"; fi
        for op in ${OPLIST:-__all__}; do
            if [ "$op" = "__all__" ]; then ARGS=(); else ARGS=(-o "$op"); fi
            if HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$GPU}" "$TBO" "${ARGS[@]}" >"$CLOG" 2>&1; then
                n="$(grep -oE '[0-9]+/[0-9]+ tests passed' "$CLOG" | tail -1)"
                PASSED_TOTAL="$PASSED_TOTAL ${op}:${n:-ok}"
            else
                CORRECT_OK=0
                CORRECT="FAILED on ${op} — $(grep -iE 'FAIL|error' "$CLOG" | head -2 | tr '\n' ' ')"
                break
            fi
        done
        [ "$CORRECT_OK" = 1 ] && CORRECT="PASSED vs CPU reference (${OPS}):${PASSED_TOTAL}"
        rm -f "$CLOG"
    fi
fi

# --- then measure, and let Hyperloom judge the measurement
BENCH_MD=""; VALID="not run"; PP_TS=""; TG_TS=""
if [ "$BENCH" = 1 ]; then
    SCRIPT_PATH="$REPO/examples/rdna/custom_${RUNNER}.sh"
    if [ "$RUNNER" = "-" ] || [ ! -x "$SCRIPT_PATH" ]; then
        VALID="no runner script for this arch ($SCRIPT_PATH)"
    elif [ -z "${MODEL:-}" ]; then
        VALID="skipped: set MODEL=/path/to/model.gguf"
    else
        OUTDIR="$(mktemp -d)"
        if RESULT_DIR="$OUTDIR" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$GPU}" \
             "$SCRIPT_PATH" >"$OUTDIR/run.log" 2>&1; then
            RES="$OUTDIR/inferencex_result.json"
            TG_TS="$($PY -c "import json;print(json.load(open('$RES'))['throughput']['output_throughput'])" 2>/dev/null)"
            PP_TS="$($PY -c "import json;print(json.load(open('$RES'))['rdna']['prompt_throughput_tps'])" 2>/dev/null)"
            # THE VERDICT IS THE HARNESS'S, NOT OURS.
            VALID="$(PYTHONPATH="$REPO/src" "$PY" - "$RES" <<'PY' 2>/dev/null || echo "validator unavailable"
import json,sys
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement, is_valid_measurement)
m = extract_benchmark_measurement(json.load(open(sys.argv[1])))
print("ACCEPTED by is_valid_measurement()" if is_valid_measurement(m) else "REJECTED by is_valid_measurement()")
PY
)"
            BENCH_MD="prefill ${PP_TS:-?} t/s / decode ${TG_TS:-?} t/s"
            QGATE="$($PY -c "
import json;g=json.load(open('$RES')).get('quality_gate')
print('none supplied' if not g else ('PASSED' if g.get('passed') else ('SKIPPED: '+str(g.get('reason'))) if g.get('skipped') else 'FAILED')+' '+str({k:v for k,v in g.items() if k in ('metric','value','threshold')}))" 2>/dev/null)"
        else
            VALID="runner FAILED — see log tail below"
            BENCH_MD="$(tail -5 "$OUTDIR/run.log" 2>/dev/null)"
        fi
    fi
fi

# --- verdict
if [ "$RUNNER" = "-" ] || [ -z "$RESOLVED" ] || [ "$RESOLVED" = "-" ]; then
    VERDICT="UNSUPPORTED"; RC=2
elif [[ "$VALID" == ACCEPTED* ]] && [ "$CORRECT_OK" = 1 ]; then
    VERDICT="CONFIRMED"; RC=0
elif [[ "$VALID" == ACCEPTED* ]]; then
    # Fast is not the same as right. Say so in the verdict, not in a footnote.
    VERDICT="PARTIAL (throughput only — CORRECTNESS UNVERIFIED)"; RC=1
elif [ "$CORRECT_OK" = 1 ]; then
    VERDICT="PARTIAL (correctness passed, no accepted measurement)"; RC=1
else
    VERDICT="PARTIAL"; RC=1
fi

cat <<MD

--------------------------- paste everything below ---------------------------
### RDNA hardware report — **$VERDICT**

| field | value |
|---|---|
| product | $(u "$PRODUCT") |
| gfx arch | $(u "$GFX") |
| compute units | $(u "$CUS") |
| VRAM | $(u "$VRAM") |
| GPUs in host | $(u "$NGPU") |
| kernel | $KERNEL |
| amdgpu driver | $(u "$DRIVER") |
| ROCm | $(u "$ROCM_V") |
| hyperloom resolves as | $(u "$RESOLVED") |
| dispatch identity | $(u "$IDENTITY") |
| runner for this arch | $(u "$RUNNER") |
| **correctness** | $CORRECT |
| benchmark | ${BENCH_MD:-not run} |
| harness verdict | $VALID |
| model-level gate | ${QGATE:-none supplied} |

MD
if [ "$RC" = 2 ]; then cat <<MD
**Not mapped.** To add it (see AGENTS.md §2), the minimum is:

- \`src/hyperloom/common/gpu_identity.py\`: \`"<board>": ("${GFX:-<gfx>}", ${CUS:-<CUs>})\`
  — board key must be a substring of \`$(u "$PRODUCT")\` for autodetect to match
- \`src/hyperloom/inference_optimizer/gpu_types.py\`: \`_GFX_TO_RUNNER["${GFX:-<gfx>}"] = "<board>"\`
- a runner \`examples/rdna/custom_<board>.sh\` (copy \`custom_r9700.sh\`)
- tests mirroring \`test_rdna4_r9700_support.py\`, negatives included
- a row in the \`docs/rdna-port/README.md\` matrix

Map only the arch you actually ran. Do not add sibling chips you have not booted.
MD
elif [ "$RC" = 1 ]; then cat <<MD
**Recognised, not yet proven.** CONFIRMED needs BOTH halves:

1. **correctness** — \`test-backend-ops\` passing against the CPU reference on this arch
2. **a measurement** Hyperloom's own validator accepts

\`MODEL=/path/model.gguf LLAMA_CPP_DIR=/path/llama.cpp ./rdna-support-probe.sh --bench\`

A throughput number with no correctness gate is not support: a kernel that is
fast and wrong reports beautifully. Add \`PPL_FILE=\`/\`PPL_MAX=\` for a
model-level gate on top. Support-matrix entries move to MEASURED only on CONFIRMED.
MD
else cat <<MD
**Confirmed on this machine.** Say what you did NOT test — GPU count, other
models, multi-GPU — so the matrix does not overstate it.
MD
fi
echo "------------------------------------------------------------------------------"
exit "$RC"
