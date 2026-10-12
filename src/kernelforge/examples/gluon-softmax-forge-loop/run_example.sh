#!/usr/bin/env bash
# Drive this forge-loop task (Gluon softmax) end to end.
#
# forge-loop git-inits its workspace and edits the kernel IN PLACE, so this
# script copies the task out of the packaged example tree into a scratch workspace
# first (keeping the repo tree clean) — the isolate-then-run pattern any
# forge-loop caller should follow.
#
# Usage:
#   ./run_example.sh [WORKSPACE_DIR]
#
# Environment overrides (all optional):
#   GPU_TARGET   gfx arch (default: autodetect via rocminfo, else gfx950)
#   MAX_HOURS    wall-clock budget in hours            (default: 2.0 — see below)
#   LANES        concurrent Implementer lanes per round (default: 1 — see below)
#   FORGE_MODEL  model name served by your gateway     (default: forge default)
#
# Prerequisites:
#   * Hyperloom installed so `kernelforge` is on PATH (pip install -e .)
#   * A gfx950 (CDNA4) GPU with torch + Triton; Gluon must import:
#     python -c "from triton.experimental import gluon"
#   * Claude auth configured: ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, or
#     CLAUDE_CODE_OAUTH_TOKEN for subscription billing
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/forge_loop_gluon_softmax_$(date +%s)}"

detect_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true
  fi
}
GPU_TARGET="${GPU_TARGET:-$(detect_arch)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
# 2.0, not the 1.0 minimum. The loop holds a 30-minute finalize reserve back, so
# a 1.0h budget leaves a 30-minute iteration window -- and a round is admitted
# only when what remains also covers planning plus one session plus the
# measurement. Measured on this task at the default width: planning alone took
# 28.6 min and the round was then refused with 21 min left against 22 needed, so
# the campaign ended having run zero iterations. 2.0 leaves a 90-minute window
# and still keeps Analysis static-only and the Plan Critic off, both of which
# switch on above 2.0.
MAX_HOURS="${MAX_HOURS:-2.0}"
# One lane, not the default 3. A round's planning cost scales with its width --
# partitioning plus one synthesis per lane -- and this task has a single obvious
# axis, so the extra lanes buy width the evidence does not support while
# spending most of a short budget on plans. Production campaigns on a real
# kernel want the default; raise it here with LANES if you want to watch the
# fan-out instead.
LANES="${LANES:-1}"

if ! command -v kernelforge >/dev/null 2>&1; then
  echo "error: 'kernelforge' not found on PATH. Install Hyperloom first:" >&2
  echo "         pip install -e /path/to/Hyperloom" >&2
  exit 1
fi

# Gluon preflight. It lives under triton.experimental, is not a stabilized API,
# and its AMD surface differs by generation -- async copy to LDS and scaled MFMA
# are in the cdna4 namespace and absent from cdna3. Fail here rather than three
# quarters of an hour into a campaign. This is the same probe the knowledge base
# tells the agent to run before its first edit.
echo "==> Gluon preflight"
python3 - <<'PY' || { echo "error: Gluon is unavailable on this interpreter." >&2; exit 1; }
import sys
import triton
print(f"    triton {triton.__version__}")
try:
    from triton.experimental import gluon
except Exception as exc:
    print(f"    GLUON UNAVAILABLE: {type(exc).__name__}: {exc}")
    sys.exit(1)
print(f"    gluon exports: {sorted(getattr(gluon, '__all__', []))}")
for gen in ("cdna3", "cdna4"):
    try:
        mod = __import__(
            f"triton.experimental.gluon.language.amd.{gen}", fromlist=[gen]
        )
    except Exception as exc:
        print(f"    {gen}: unavailable ({exc})")
        continue
    have = [n for n in ("buffer_load", "async_copy", "mfma", "mfma_scaled")
            if hasattr(mod, n)]
    print(f"    {gen}: {', '.join(have) or '(none of the expected ops)'}")
PY

echo "==> Preparing scratch workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cp "$EXAMPLE_DIR/softmax_kernel.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/driver.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/graph_harness.py" "$WORKSPACE/"   # measurement harness (protected)
cp "$EXAMPLE_DIR/program.md" "$WORKSPACE/"

# forge-loop's keep/revert relies on git; give it a repo with an initial commit.
# Build artifacts and the loop's own outputs stay untracked so a revert never
# fails on a dirtied tree.
cd "$WORKSPACE"
if [ ! -d .git ]; then
  git init -q
  git config user.email "forge-example@local"
  git config user.name "forge-example"
fi
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.log
build/
forge_experiments/
EOF
git add -A
git commit -q -m "forge example: initial workspace" || true

# Two genuinely environmental vars (not forge config): IS_SANDBOX is required by
# the claude CLI under root; PYTHONUNBUFFERED makes the stream flush promptly.
export IS_SANDBOX="${IS_SANDBOX:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS+=(--model "$FORGE_MODEL")
fi

echo "==> Launching forge-loop (gpu=mi355x/$GPU_TARGET, max_hours=$MAX_HOURS)"
kernelforge forge-loop \
  --kernel "$WORKSPACE/softmax_kernel.py" \
  --driver "$WORKSPACE/driver.py" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_result.json" \
  --program-md-file "$WORKSPACE/program.md" \
  --kernel-backend gluon \
  --gpu-target "$GPU_TARGET" \
  --snr-threshold 30.0 \
  --max-hours "$MAX_HOURS" \
  --lanes "$LANES" \
  --git-branch forge-optimize \
  --target-functions "softmax,_softmax_kernel" \
  "${MODEL_ARGS[@]}"

echo "==> Done. Best kernel is checked out in: $WORKSPACE/softmax_kernel.py"
echo "    Iteration archive + profiles: $WORKSPACE/forge_experiments/"
echo "    Machine-readable result:      $WORKSPACE/forge_experiments/forge_result.json"
