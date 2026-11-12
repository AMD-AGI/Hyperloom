#!/usr/bin/env bash
# Run the Qwen3 seam monokernel task (TP4, gfx942) through forge-loop, end to end:
# a fresh git workspace from task/, a self-check of the pristine launch, the campaign,
# and the best launch exported beside the workspace.
#
# Usage:
#   ./run_example.sh [--workspace DIR] [--max-hours H] [--model NAME]
#
# The four GPUs must be otherwise idle: every CTA of the launch has to be resident,
# and work from another process on the same GPU can keep some from starting.
#
# Prerequisites: 4 x MI300X (gfx942); torch, FlyDSL, aiter, SGLang and sgl-amd-mono
# (b10f3b7) importable by python3; the agent provider's credentials. kernelforge runs
# from the checkout this script sits in.
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$EXAMPLE_DIR/../../.." && pwd)"
WS="$PWD/mono_qwen3_seam_ws"
MAX_HOURS=2
MODEL_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WS="$(realpath -m "$2")"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --model) MODEL_ARGS=(--model "$2"); shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -e "$WS" ] && [ -n "$(ls -A "$WS" 2>/dev/null)" ]; then
  echo "error: $WS exists and is not empty; pass a new --workspace" >&2
  exit 2
fi
WS_PARENT="$(dirname "$WS")"
WS_NAME="$(basename "$WS")"
CHECKS="$WS_PARENT/$WS_NAME.selfcheck"
DELIVERED="$WS_PARENT/$WS_NAME.delivered"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# required by the claude CLI under root
export IS_SANDBOX="${IS_SANDBOX:-1}"
# compiled launches of this campaign only, beside the workspace
export FLYDSL_RUNTIME_CACHE_DIR="$WS_PARENT/$WS_NAME.flydsl-cache"
mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR" "$CHECKS"

echo "==> Preflight"
python3 - <<'EOF'
import sys

import torch

n = torch.cuda.device_count()
archs = sorted({torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(n)})
if n < 4 or archs != ["gfx942"]:
    sys.exit(f"error: the task needs 4 gfx942 GPUs; visible: {n} ({', '.join(archs) or 'none'})")
import aiter  # noqa: F401
import flydsl  # noqa: F401
import kernelforge  # noqa: F401
from sgl_amd_mono.common import guard  # noqa: F401
from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear  # noqa: F401
EOF

echo "==> Workspace: $WS"
mkdir -p "$WS"
cp -r "$EXAMPLE_DIR/task/." "$WS/"
# the measurement harness this checkout's forge-loop ships, beside the driver that imports it
cp "$SRC_DIR/kernelforge/workflows/forge_loop/preparation/dist_harness.py" "$WS/benchmark/forge_mono/"
cd "$WS"
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
forge_experiments/
EOF
git init -q
git config user.email "forge-example@local"
git config user.name "forge-example"
git add -A
git commit -q -m "mono-qwen3-seam: pristine task"
python3 -c 'import sys; sys.path.insert(0, "benchmark/forge_mono"); from contract import check_contract; check_contract()'

echo "==> Self-check of the pristine launch (logs: $CHECKS)"
python3 benchmark/forge_mono/driver.py 2>&1 | tee "$CHECKS/correctness.log"
grep -qx "allclose: True" "$CHECKS/correctness.log" || { echo "error: the pristine launch fails its judge" >&2; exit 1; }
python3 benchmark/forge_mono/driver.py --bench-mode --warmup 3 --iters 10 2>&1 | tee "$CHECKS/bench.log"
for case in s1 s4 s8 s16; do
  grep -q "^case_ms: $case " "$CHECKS/bench.log" || { echo "error: no case_ms for $case" >&2; exit 1; }
done
python3 benchmark/forge_mono/driver.py --profile-run >"$CHECKS/profile_run.log" 2>&1 \
  || { echo "error: --profile-run failed (see $CHECKS/profile_run.log)" >&2; exit 1; }
python3 - "$CHECKS" <<'EOF'
import subprocess
import sys

import yaml

suite = yaml.safe_load(open("config.yaml"))
for step in ("compile", "correctness"):
    for command in suite[f"{step}_command"]:
        print(f"--> {command}", flush=True)
        with open(f"{sys.argv[1]}/{step}_suite.log", "a") as log:
            done = subprocess.run(command, shell=True, stdout=log, stderr=subprocess.STDOUT, timeout=suite[f"{step}_timeout"])
        if done.returncode:
            sys.exit(f"error: {command!r} exited {done.returncode} (see {sys.argv[1]}/{step}_suite.log)")
EOF
git status --porcelain | grep -q . && { echo "error: the self-check left the workspace dirty" >&2; git status --short >&2; exit 1; }

echo "==> forge-loop (max_hours=$MAX_HOURS)"
python3 -m kernelforge.cli forge-loop "${MODEL_ARGS[@]}" \
  --workspace "$WS" \
  --kernel "$WS/qwen3_seam/seam.py" \
  --driver "$WS/benchmark/forge_mono/driver.py" \
  --program-md-file "$EXAMPLE_DIR/program.md" \
  --kernel-backend mono \
  --task-type repository \
  --source-files "$WS/qwen3_seam/seam.py" \
  --target-functions "run_seam,stage_push,stage_reduce,stage_norm" \
  --framework sglang \
  --operator-name qwen3_attention_seam \
  --nproc-per-node 4 \
  --lanes 1 \
  --gpu-target gfx942 \
  --gpu-type mi300x \
  --max-hours "$MAX_HOURS" \
  --git-branch campaign \
  --experiments-dir "$WS/forge_experiments"

BEST="$WS/forge_experiments/best_result.json"
if [ ! -f "$BEST" ]; then
  echo "==> No KEEP: the pristine launch stands ($WS/qwen3_seam)"
  exit 0
fi
COMMIT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit_hash"])' "$BEST")"
PRISTINE="$(git rev-list --max-parents=0 HEAD)"
OUT="$DELIVERED/$COMMIT"
mkdir -p "$OUT"
git archive "$COMMIT" qwen3_seam | tar -x -C "$OUT"
git diff "$PRISTINE" "$COMMIT" -- qwen3_seam >"$OUT/qwen3_seam.diff"
cp "$BEST" "$OUT/"
echo "==> Delivered: $OUT/qwen3_seam (diff against the pristine launch: $OUT/qwen3_seam.diff)"
