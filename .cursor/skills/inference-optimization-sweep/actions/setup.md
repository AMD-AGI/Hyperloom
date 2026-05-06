# Action: Environment Setup (Sweep-Only)

## Inputs (must be provided by the caller)

| Variable | Required | Description |
|----------|:--------:|-------------|
| `MODEL` | yes | Model path or HuggingFace id |
| `TP` | yes | Tensor parallel size (must equal visible GPU count for sweep) |
| `FRAMEWORK` | yes | `sglang` or `vllm` |
| `EXTRA_SGLANG_ARGS` / `EXTRA_VLLM_ARGS` | yes | Server flags for the framework you picked |
| `RUNNER_TYPE` | no | Auto-detected from `rocm-smi --showproductname` when absent |
| `INFERENCEX_PATH` | no | Auto-detected via `/shared_nfs/*/InferenceX` when absent |
| `MAGPIE_PATH` | no | Resolved by `_resolve_magpie_path` (see below) |

The skill aborts early if any required variable is empty.

## Procedure

### Step 0: Set PATH for venv

```bash
export PATH="/opt/venv/bin:$PATH"
```

`/usr/bin/python3` does NOT have sglang / vllm / numpy installed. Every bash
command in this skill MUST start with the venv on `PATH`.

### Step 1: Validate caller-provided inputs

```bash
fail=0
for v in MODEL TP FRAMEWORK; do
    if [ -z "${!v}" ]; then echo "ERROR: $v is required" >&2; fail=1; fi
done
EXTRA_KEY="EXTRA_$(echo "${FRAMEWORK:-sglang}" | tr '[:lower:]' '[:upper:]')_ARGS"
if [ -z "${!EXTRA_KEY:-}" ]; then
    echo "ERROR: $EXTRA_KEY is required (e.g. --attention-backend aiter --kv-cache-dtype fp8_e4m3 --disable-radix-cache)" >&2
    fail=1
fi
[ "$fail" -eq 0 ] || exit 1
```

### Step 2: Detect environment

```bash
GPU_COUNT=$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)
GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")
case "$GPU_ARCH" in
    gfx942) RUNNER_TYPE=${RUNNER_TYPE:-mi300x} ;;
    gfx950) RUNNER_TYPE=${RUNNER_TYPE:-mi355x} ;;
    *)      RUNNER_TYPE=${RUNNER_TYPE:-mi355x} ;;
esac
INFERENCEX_PATH=${INFERENCEX_PATH:-$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)}
[ -d "$INFERENCEX_PATH" ] || { echo "ERROR: InferenceX not found at $INFERENCEX_PATH" >&2; exit 1; }
```

### Step 3: Mode detection + executor

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization-sweep}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

if [ "${GEAK_LOCAL:-true}" = "true" ]; then
    MODE="local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/inference-optimization-sweep}"
else
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization-sweep}"
fi
source "$SCRIPTS_DIR/executor.sh"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
RESULT_DIR="${WORKSPACE_ROOT}/results/${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

export MODE MODEL TP FRAMEWORK RUNNER_TYPE INFERENCEX_PATH RESULT_DIR
```

### Step 4: Resolve Magpie path and ensure CLI is installed

```bash
_mirror_if_readonly() {
    local src="$1" dst="$2"
    [ -z "$src" ] && return 1
    if touch "$src/.rw_probe" 2>/dev/null; then
        rm -f "$src/.rw_probe"
        echo "$src"
    else
        if [ ! -d "$dst" ]; then
            echo "Mirroring $src -> $dst (source is read-only)" >&2
            cp -a "$src" "$dst"
        fi
        echo "$dst"
    fi
}

_resolve_magpie_path() {
    local candidate
    if [ -n "${MAGPIE_PATH:-}" ] && [ -d "$MAGPIE_PATH" ]; then
        echo "$MAGPIE_PATH"; return 0
    fi
    candidate="/hyperloom/users/8cf535bc3ad11fa15e48157cf3b3f726/Magpie"
    [ -d "$candidate" ] && { echo "$candidate"; return 0; }
    candidate=$(ls -d /shared_nfs/*/Magpie 2>/dev/null | head -1)
    [ -n "$candidate" ] && [ -d "$candidate" ] && { echo "$candidate"; return 0; }
    echo "/shared_nfs/Magpie"
}

INFERENCEX_PATH=$(_mirror_if_readonly "$INFERENCEX_PATH" "/tmp/InferenceX")
export INFERENCEX_PATH

MAGPIE_PATH=$(_resolve_magpie_path)
MAGPIE_PATH=$(_mirror_if_readonly "$MAGPIE_PATH" "/tmp/Magpie")
export MAGPIE_PATH

if ! command -v magpie >/dev/null 2>&1; then
    pip install -e "$MAGPIE_PATH" 2>&1 | tail -3
fi

# Sweep capability check: actions/sweep.md needs SweepMatrix support.
python3 -c "from Magpie.modes.benchmark.config import SweepMatrix" 2>/dev/null || {
    echo "ERROR: Installed Magpie at $MAGPIE_PATH lacks sweep_matrix support." >&2
    echo "Install a Magpie revision with the sweep_matrix feature (e.g. branch feature/zhanglei/magpie-sweep)." >&2
    exit 1
}
```

### Step 5: Apply ROCm/HIP rotary embedding patches (SGLang only, idempotent)

```bash
ROPE_FILE="/sgl-workspace/sglang/python/sglang/srt/layers/rotary_embedding.py"
if [ "$FRAMEWORK" = "sglang" ] && [ -f "$ROPE_FILE" ]; then
    python3 - "$ROPE_FILE" <<'PY'
import sys
path = sys.argv[1]
patches = [
    (
        "if _is_cuda:\n"
        "    from sglang.jit_kernel.rope import (\n"
        "        FusedSetKVBufferArg,\n"
        "        apply_rope_with_cos_sin_cache_inplace,\n"
        "    )\n"
        "else:\n"
        "    FusedSetKVBufferArg = None",
        "if _is_cuda:\n"
        "    from sglang.jit_kernel.rope import (\n"
        "        FusedSetKVBufferArg,\n"
        "        apply_rope_with_cos_sin_cache_inplace,\n"
        "    )\n"
        "elif _is_hip:\n"
        "    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace\n"
        "    FusedSetKVBufferArg = None\n"
        "else:\n"
        "    FusedSetKVBufferArg = None",
    ),
    (
        "if not _is_cuda:\n"
        "            cache = cache.to(dtype)",
        "if not _is_cuda and not _is_hip:\n"
        "            cache = cache.to(dtype)",
    ),
]
with open(path) as f:
    txt = f.read()
for old, new in patches:
    if old in txt:
        txt = txt.replace(old, new)
with open(path, "w") as f:
    f.write(txt)
print("ROCm rotary embedding patches applied (idempotent).")
PY
fi
```

vLLM does not need this patch.

## Outputs

After this action completes the following are exported and validated:

- `MODE`, `MODEL`, `TP`, `FRAMEWORK`, `RUNNER_TYPE`, `INFERENCEX_PATH`,
  `RESULT_DIR`, `MAGPIE_PATH`
- `EXTRA_SGLANG_ARGS` or `EXTRA_VLLM_ARGS` (passthrough from caller)
- `magpie` is callable on `PATH`
- `Magpie.modes.benchmark.config.SweepMatrix` import probe passed

## Failure Handling

- Required input missing → exit 1 with the offending variable name.
- InferenceX path not found → exit 1; user must mount or set `INFERENCEX_PATH`.
- Magpie missing `SweepMatrix` → exit 1; user must upgrade Magpie.
- ROCm patch step is best-effort; failure is logged but does not abort.
