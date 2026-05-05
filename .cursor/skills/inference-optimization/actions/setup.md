# Action: Environment Setup

## Inputs
- User-specified MODEL, TP, CONC, ISL, OSL, FRAMEWORK (optional — auto-detect if not provided)

## Procedure

### Step 0 (CRITICAL): Set PATH for venv

**ALWAYS prepend `/opt/venv/bin` to PATH before any python3 command.** The system
`/usr/bin/python3` does NOT have sglang/vllm/numpy installed. Every bash command
in this skill MUST start with:

```bash
export PATH="/opt/venv/bin:$PATH"
```

### Step 1: Auto-detect environment

```bash
export PATH="/opt/venv/bin:$PATH"

MODEL=$(ls -d /shared_nfs/*/models/*/ 2>/dev/null | head -1)
GPU_COUNT=$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)
GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")
INFERENCEX_PATH=$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)

# Map GPU arch to runner type (used by Magpie --benchmark-script)
case "$GPU_ARCH" in
    gfx942) RUNNER_TYPE=mi300x ;;
    gfx950) RUNNER_TYPE=mi355x ;;
    *)      RUNNER_TYPE=mi355x ;;
esac

FRAMEWORK="${FRAMEWORK:-sglang}"
if [ "$FRAMEWORK" = "vllm" ]; then
    FRAMEWORK_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
else
    FRAMEWORK_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)
fi

TP=$GPU_COUNT
if [ "$TP" -le 1 ]; then CONC=4; elif [ "$TP" -le 4 ]; then CONC=32; else CONC=64; fi
```

User-specified values override auto-detected ones. **NEVER override user-specified TP**
— if the prompt says TP=8, use TP=8 even if GPU_COUNT differs.

### Step 2: Set paths and env vars

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

# Mode detection
if [ "${GEAK_LOCAL:-true}" = "true" ]; then
    MODE="local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/inference-optimization}"
else
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
fi

# Source executor backend (enables exec_on_gpu for local/claw dispatch)
source "$SCRIPTS_DIR/executor.sh"

export MODEL="$MODEL"
export TP="$TP"
export CONC="$CONC"
export ISL="${ISL:-1024}"
export OSL="${OSL:-256}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export RUNNER_TYPE="$RUNNER_TYPE"

# ── Read-only path mirroring ──────────────────────────────────────────────
# InferenceX and Magpie may reside on a read-only mount (NFS / container
# overlay). Copy them to /tmp when not writable so that benchmark scripts
# can create temp files, logs, and cleanup markers inside the tree.
_mirror_if_readonly() {
    local src="$1" dst="$2"
    [ -z "$src" ] && return 1
    if touch "$src/.rw_probe" 2>/dev/null; then
        rm -f "$src/.rw_probe"
        echo "$src"
    else
        if [ ! -d "$dst" ]; then
            echo "Mirroring $src → $dst (source is read-only)" >&2
            cp -a "$src" "$dst"
        fi
        echo "$dst"
    fi
}

INFERENCEX_PATH=$(_mirror_if_readonly "$INFERENCEX_PATH" "/tmp/InferenceX")
export INFERENCEX_PATH

# Resolve Magpie path: caller override > current workspace > NFS user dir > legacy default.
# The current workspace is hard-coded as first candidate so the sweep_matrix-aware
# Magpie checkout that lives next to Hyperloom is picked up automatically.
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

MAGPIE_PATH=$(_resolve_magpie_path)
MAGPIE_PATH=$(_mirror_if_readonly "$MAGPIE_PATH" "/tmp/Magpie")
export MAGPIE_PATH

# Install Magpie CLI
if ! command -v magpie &>/dev/null; then
    pip install -e "$MAGPIE_PATH" 2>&1 | tail -3
fi

# Sweep capability check: actions/sweep.md needs SweepMatrix support in Magpie
python3 -c "from Magpie.modes.benchmark.config import SweepMatrix" 2>/dev/null || \
    echo "WARNING: Installed Magpie at $MAGPIE_PATH lacks sweep_matrix support; actions/sweep.md will fail." >&2
```

Environment variables are set in the YAML config `envs` section and passed via `magpie benchmark --benchmark-config`. Each benchmark call generates its own YAML config file.

### Step 3: Apply ROCm/HIP compatibility patches (MANDATORY on AMD GPUs)

SGLang's JIT rotary embedding kernel requires `nvcc` (NVIDIA) which does not exist on ROCm.
The server will crash during CUDA graph capture without these patches. Apply them **once**
at setup time — they persist for the entire optimization run.

```python
import subprocess, sys

# Patch 1: rotary_embedding.py — skip JIT fallback on HIP, use sgl_kernel instead
ROPE_FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/rotary_embedding.py"

patches = [
    # 1a: Import apply_rope from sgl_kernel on HIP (not flashinfer-dependent jit_kernel.rope)
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
    # 1b: Keep cos_sin_cache in FP32 on HIP (sgl_kernel requires it)
    (
        "if not _is_cuda:\n"
        "            cache = cache.to(dtype)",
        "if not _is_cuda and not _is_hip:\n"
        "            cache = cache.to(dtype)",
    ),
    # 1c: Skip JIT fallback on HIP — use sgl_kernel's non-fallback path
    (
        "            (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])\n"
        "            and not (_is_cpu)\n"
        "            and not (_is_xpu)\n"
        "            and not (_is_npu)\n"
        "            and not (_is_musa)\n"
        "        ):",
        "            (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])\n"
        "            and not (_is_cpu)\n"
        "            and not (_is_xpu)\n"
        "            and not (_is_npu)\n"
        "            and not (_is_musa)\n"
        "            and not (_is_hip)\n"
        "        ):",
    ),
]

with open(ROPE_FILE) as f:
    content = f.read()

for old, new in patches:
    if old in content:
        content = content.replace(old, new)
        print(f"Applied patch: {old[:60]}...")
    else:
        print(f"Patch target not found (may already be applied): {old[:60]}...")

with open(ROPE_FILE, "w") as f:
    f.write(content)

print("ROCm/HIP rotary embedding patches applied.")
```

**Why:** On HIP, `sglang.jit_kernel.pos_enc` uses `tvm_ffi` which calls `nvcc` (not
available on ROCm). The `sgl_kernel` package has a pre-compiled HIP-compatible
`apply_rope_with_cos_sin_cache_inplace`. The `cos_sin_cache` must stay FP32 for
`sgl_kernel` (CUDA keeps it FP32 too, but the HIP path converts it to model dtype).

### Mamba/Hybrid models on ROCm

Models with Mamba layers (e.g., Qwen3-Next) require `--mamba-scheduler-strategy no_buffer`
on ROCm. The default `extra_buffer` strategy uses FLA backend which is CUDA-only.

**Always include this in `EXTRA_SGLANG_ARGS` for Mamba/hybrid models on ROCm:**
```bash
EXTRA_SGLANG_ARGS="$EXTRA_SGLANG_ARGS --mamba-scheduler-strategy no_buffer"
```

Without this, the server crashes immediately with:
`Mamba extra_buffer is only supported on CUDA devices with FLA backend`

## Outputs
- All environment variables set
- `$SKILL_ROOT`, `$SCRIPTS_DIR` paths validated
- `$RESULT_DIR` created

**Claw mode:** After environment setup, create the RayJob before proceeding. See
[`../modes/CLAW.md`](../modes/CLAW.md) "RayJob Lifecycle" for the full `workload_create`
payloads, wait logic, and claw execution environment setup.

## Failure Handling
- If no model found: ask user for MODEL path
- If no GPUs detected: check `amd-smi` / `rocm-smi` installation
- If InferenceX not found: check `/shared_nfs/*/InferenceX/`
