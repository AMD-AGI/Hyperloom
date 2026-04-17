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

# InferenceX path — prefer the one bundled inside the current Hyperloom workspace
# (has the newest patches, including TraceLens PR integration). Fall back to any
# /shared_nfs/*/InferenceX only if no workspace-local copy exists. User can override
# by pre-exporting INFERENCEX_PATH.
if [ -z "${INFERENCEX_PATH:-}" ]; then
    INFERENCEX_PATH=$(ls -d /shared_nfs/*/Hyperloom/inference_optimization/InferenceX 2>/dev/null | head -1)
    if [ -z "$INFERENCEX_PATH" ]; then
        INFERENCEX_PATH=$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)
    fi
fi
# Sanity: the selected InferenceX must contain utils/apply_tracelens_patches.sh if
# profiling is planned (see Step 4). If not, bail loudly rather than silently using
# a stale sibling clone.
if [ -n "${INFERENCEX_PATH:-}" ] && [ ! -f "$INFERENCEX_PATH/utils/apply_tracelens_patches.sh" ]; then
    echo "WARN: $INFERENCEX_PATH lacks utils/apply_tracelens_patches.sh" >&2
    echo "      Profile action will fail. Set INFERENCEX_PATH to the Hyperloom-bundled copy." >&2
fi

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

# RESULT_DIR: unique per-run output directory. All actions (baseline/profile/sweep)
# write into this tree. Falls back to /tmp if WORKSPACE_ROOT is not writable
# (common when running inside a container with a read-only /workspace overlay).
if [ -z "${RESULT_DIR:-}" ]; then
    _ts="$(date +%Y%m%d_%H%M%S)"
    RESULT_DIR="$WORKSPACE_ROOT/results/$_ts"
    if ! mkdir -p "$RESULT_DIR" 2>/dev/null; then
        RESULT_DIR="/tmp/inference-optimization/results/$_ts"
        mkdir -p "$RESULT_DIR"
        echo "WARN: $WORKSPACE_ROOT not writable, RESULT_DIR → $RESULT_DIR" >&2
    fi
fi
export RESULT_DIR

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

MAGPIE_PATH="${MAGPIE_PATH:-/shared_nfs/Magpie}"
MAGPIE_PATH=$(_mirror_if_readonly "$MAGPIE_PATH" "/tmp/Magpie")
export MAGPIE_PATH

# Install Magpie CLI
if ! command -v magpie &>/dev/null; then
    pip install -e "$MAGPIE_PATH" 2>&1 | tail -3
fi
```

Environment variables are set in the YAML config `envs` section and passed via `magpie benchmark --benchmark-config`. Each benchmark call generates its own YAML config file.

### Step 3: Apply ROCm/HIP compatibility patches (CONDITIONAL — try without first)

**Do NOT apply unconditionally. Some MI300X/MI355X container images already
have a working HIP rotary path and do not need this patch. Applying it on
those images actively breaks the server.**

Decision procedure:

1. **First try launching the server without this patch** (proceed directly to
   Step 4). The skill's profile/baseline actions will attempt a server start.
2. **Only apply this patch if the server crashes** with one of these specific
   symptoms in the log (`$RESULT_DIR/.../server.log`):
   - `FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'` coming
     from `sglang/jit_kernel/pos_enc.py` or `sglang/jit_kernel/rope.py`
   - `tvm_ffi` import failure on a HIP device during rotary embedding
   - Crash stack containing `sglang.jit_kernel.pos_enc` on an `_is_hip=True` path

3. **Do NOT apply this patch if the server crashes with any of these** —
   applying it will make things worse:
   - `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
     'apply_rope_pos_ids_cos_sin_cache'` (the patched path calls an op that
     the container's compiled `sgl_kernel` doesn't register)
   - Any crash from inside `sgl_kernel/elementwise.py`
   - Any crash where rotary_embedding works fine and the error is elsewhere

**If you applied this patch and hit the sgl_kernel op mismatch, revert it:**
```bash
cd /sgl-workspace/sglang/python
git checkout -- sglang/srt/layers/rotary_embedding.py
```
`git checkout` is safe: none of the 10 TraceLens sglang patches touch
`rotary_embedding.py`, so this only reverts Step 3.

**Rationale for the downgrade.** Earlier versions of this skill marked Step 3
`MANDATORY`, but the fix only works against a specific `sglang` + `sgl_kernel`
pairing. On images where sglang's original HIP code path works (e.g. current
AMD MI355X dev images with `sgl_kernel 0.3.21+`), applying this patch
introduces the op-mismatch crash above. The safe default is "apply only when
the documented symptoms appear".

**If you determine the patch IS needed, apply it as below:**

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

### Step 4: Apply TraceLens patches (REQUIRED when profiling)

**Only needed when you will run `actions/profile.md` (i.e., `--enable-profile-cuda-graph`
or `--profiler-config.capture_torch_profiler_dir` will be passed to the server).**
Without these patches the server rejects those flags with `unrecognized arguments` and
no `capture_traces/` folder is produced.

These patches are **separate** from the ROCm rotary patch in Step 3 — they add graph
capture tracing and `sglang_profiler::*` roofline annotations to sglang/vLLM source.
See `InferenceX/AGENTS.md` "TraceLens Patch Integration" for details.

**Two execution paths — use whichever matches your run mode:**

1. **Docker/Slurm launcher mode** (`runners/launch_mi300x-amd.sh`,
   `runners/launch_mi355x-amds.sh`): `benchmark_lib.sh` auto-applies the patches
   inside the container when sourced. You only need to export the env var **before**
   invoking the launcher:
   ```bash
   export TRACELENS_PATCHES=1
   export TRACELENS_REPO="${TRACELENS_REPO:-/shared_nfs/*/TraceLens-internal}"  # optional; avoids cloning
   ```
   No manual patch command required.

2. **Local/direct mode** (`magpie -e local benchmark`, or any flow that does NOT go
   through one of the updated launchers): the auto-apply hook in `benchmark_lib.sh`
   still fires when the script is sourced, **but only if `TRACELENS_PATCHES=1` is in
   the environment seen by that shell**. Magpie's YAML `envs:` section does not
   propagate to `apply_tracelens_patches()` reliably, so apply the patches explicitly
   **before** the first `magpie benchmark` call. Use this idempotent sequence that
   tolerates the known PR bug in `apply_tracelens_patches.sh` (see Failure Handling
   below):

   ```bash
   export TRACELENS_REPO="${TRACELENS_REPO:-$(ls -d /shared_nfs/*/TraceLens-internal 2>/dev/null | head -1)}"
   MARKER=/tmp/.tracelens_patches_applied

   _tracelens_already_applied() {
       # Probe: if all sglang patches reverse-apply cleanly, they are all in place.
       local pdir="$TRACELENS_REPO/examples/custom_workflows/inference_analysis/sglang_roofline_patches"
       [[ -d "$pdir" ]] || return 1
       ( cd /sgl-workspace/sglang/python && \
         for p in "$pdir"/*.patch; do git apply --check -R "$p" >/dev/null 2>&1 || return 1; done )
   }

   if [[ -f "$MARKER" ]]; then
       echo "[TraceLens] Marker present, skipping patch apply"
   elif _tracelens_already_applied; then
       # PR bug workaround: patches are on disk but apply_tracelens_patches.sh never
       # wrote the marker. Mark them as applied and skip the re-apply (which would fail).
       echo "[TraceLens] Patches already on disk (reverse-apply probe passed); setting marker"
       touch "$MARKER"
   else
       bash "$INFERENCEX_PATH/utils/apply_tracelens_patches.sh" \
           --framework "$FRAMEWORK" --tracelens "$TRACELENS_REPO" --no-install
       touch "$MARKER"
   fi
   ```

**Verify patches took effect** (sglang example):
```bash
python3 -c "from sglang.srt.server_args import ServerArgs; import inspect; \
  assert 'enable_shape_discovery_for_cuda_graph_profile' in inspect.getsource(ServerArgs), \
  'TraceLens patches NOT applied'; print('TraceLens sglang patches: OK')"
```

**Skip this step entirely if `PROFILE` is not set and no tracelens analysis is planned**
— the patches are no-ops for pure throughput benchmarking and add unnecessary risk of
source mutation.

**Known PR bug in `apply_tracelens_patches.sh` (report, don't silently patch).**
The standalone script runs `git apply <patch>` unconditionally and never writes
`/tmp/.tracelens_patches_applied`. The marker is only written by the wrapper
`apply_tracelens_patches()` in `benchmark_lib.sh`, *after* the standalone returns 0.
So when patches are already on disk but the marker is missing (typical across
container restarts, or after any prior manual apply), a second invocation fails
with `error: patch does not apply`, and `set -euo pipefail` in `benchmark_lib.sh`
kills the benchmark at server launch. The reverse-apply probe above is a
workaround, not a fix. The real fix must be made in
`InferenceX/utils/apply_tracelens_patches.sh` by the PR author (add `git apply
--check` + reverse-apply fallback, then `touch "$MARKER"` at the end). When that
fix lands upstream, the `_tracelens_already_applied` probe above becomes a no-op
and can be removed.

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
- All environment variables set: `MODEL`, `TP`, `CONC`, `ISL`, `OSL`, `FRAMEWORK`,
  `RUNNER_TYPE`, `WORKSPACE_ROOT`, `RESULT_DIR`, `INFERENCEX_PATH`, `MAGPIE_PATH`
- `$SKILL_ROOT`, `$SCRIPTS_DIR` paths validated
- `$RESULT_DIR` created (`$WORKSPACE_ROOT/results/<ts>` or `/tmp/...` fallback)
- If profile action will run: TraceLens patches applied and verified (Step 4)

**Claw mode:** After environment setup, create the RayJob before proceeding. See
[`../modes/CLAW.md`](../modes/CLAW.md) "RayJob Lifecycle" for the full `workload_create`
payloads, wait logic, and claw execution environment setup.

## Failure Handling
- If no model found: ask user for MODEL path
- If no GPUs detected: check `amd-smi` / `rocm-smi` installation
- If InferenceX not found: check `/shared_nfs/*/InferenceX/`
