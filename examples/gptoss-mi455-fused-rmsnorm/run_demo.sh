#!/usr/bin/env bash
###############################################################################
# End-to-end demo: reproduce the fused add+RMSNorm speedup on gpt-oss-120b
# (MI455 / gfx1250). For each of {baseline, fused} it launches the server,
# checks correctness, warms (crucial — cold Triton autotune is ~2.5x slower),
# and measures throughput. Then prints the A/B comparison.
#
#   ./run_demo.sh                 # full A/B (two model loads; ~10-15 min warm)
#   MICRO=1 ./run_demo.sh         # ALSO run the fast isolated kernel microbench first
#
# Tunables: CARD, PORT, CONC, NREQ, ISL, OSL, WARMUP, RUNS (see below).
###############################################################################
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CARD="${CARD:-1}"; PORT="${PORT:-8001}"
CONC="${CONC:-32}"; NREQ="${NREQ:-64}"; ISL="${ISL:-1024}"; OSL="${OSL:-128}"
WARMUP="${WARMUP:-3}"; RUNS="${RUNS:-4}"
PY="${PY:-python3}"

med() { "$PY" -c "import sys,statistics as s; v=[float(x) for x in sys.argv[1:]]; print(f'{s.median(v):.1f}')" "$@"; }

run_mode() {  # $1 = baseline|fused ; echoes median out tok/s
  local mode="$1"
  echo "" >&2
  echo "################ MODE=$mode ################" >&2
  MODE="$mode" CARD="$CARD" PORT="$PORT" NAME="gptoss-demo-$mode" bash "$HERE/serve.sh" >&2 || return 1
  PORT="$PORT" bash "$HERE/correctness.sh" >&2
  echo ">> warming ($WARMUP runs, discarded)..." >&2
  for _ in $(seq 1 "$WARMUP"); do "$PY" "$HERE/bench.py" "$PORT" w "$CONC" "$NREQ" "$ISL" "$OSL" >/dev/null 2>&1; done
  echo ">> measuring ($RUNS runs)..." >&2
  local vals=()
  for _ in $(seq 1 "$RUNS"); do
    local j; j="$("$PY" "$HERE/bench.py" "$PORT" "$mode" "$CONC" "$NREQ" "$ISL" "$OSL" 2>/dev/null | tail -1)"
    local t; t="$(echo "$j" | "$PY" -c "import sys,json;print(json.loads(sys.stdin.read())['out_tok_s'])" 2>/dev/null)"
    echo "     $mode run: $t tok/s" >&2; vals+=("$t")
  done
  med "${vals[@]}"
}

if [ "${MICRO:-0}" = "1" ]; then
  echo "===== Isolated kernel microbench (correctness + op speedup, no full server) ====="
  # Needs torch+triton+GPU; run it inside a throwaway gptoss container.
  docker rm -f gptoss-demo-micro >/dev/null 2>&1 || true
  docker run -d --name gptoss-demo-micro --network host --ipc=host --privileged \
    --device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES="$CARD" -e HIP_VISIBLE_DEVICES=0 \
    "${IMAGE:-registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2}" \
    -lc "sleep infinity" >/dev/null
  docker cp "$HERE/microbench.py" gptoss-demo-micro:/root/microbench.py >/dev/null 2>&1
  # Run from /root: the image's default cwd has a triton *source* tree that
  # shadows the installed package, breaking `import triton`.
  docker exec gptoss-demo-micro bash -lc 'cd /root && python3 microbench.py' 2>&1 | grep -viE "hotswap|rocm_sdk|warn"
  docker rm -f gptoss-demo-micro >/dev/null 2>&1 || true
fi

BASE_MED="$(run_mode baseline)"
FUSED_MED="$(run_mode fused)"

SPEEDUP="$("$PY" -c "b=$BASE_MED; f=$FUSED_MED; print(f'{100*(f-b)/b:+.1f}%')")"
echo ""
echo "==================== RESULT (median out tok/s @ conc=$CONC, ISL=$ISL, OSL=$OSL) ===================="
printf "  %-28s %8s\n" "baseline (stock RMSNorm)" "$BASE_MED"
printf "  %-28s %8s\n" "fused add+RMSNorm (demo)" "$FUSED_MED"
printf "  %-28s %8s\n" "throughput delta" "$SPEEDUP"
echo "===================================================================================================="
echo "Note: pre-silicon gfx1250 drifts across a long session; run baseline+fused back-to-back (this script does)."
echo "Cleanup: docker rm -f gptoss-demo-baseline gptoss-demo-fused"
