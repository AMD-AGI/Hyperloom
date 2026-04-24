# Action: Deep Kernel Analysis

Perform per-kernel deep analysis: trace dispatch paths, discover implementation variants,
verify configuration, and assess build-system requirements for optimization. This action
transforms opaque kernel names from profiler output into actionable optimization targets.

## Inputs
- Profile trace from `actions/profile.md`
- List of top-N kernels by GPU time % (from `state.kernel_candidates`)
- `state.kernel_dispatch_map` (initially empty — this action populates it)

## Mandatory Scope (CRITICAL)

You MUST analyze at least the **top 10 kernels by GPU%** from the profile.
MIN_GPU_PCT = 2.0% — every kernel above this threshold gets full Steps 1-5.
Do NOT stop after finding the first opportunity. The goal is a COMPREHENSIVE
action stack, not a single fix.

After analyzing individual kernels, you MUST also run Step 0 (Env Var Scan)
to find framework-level dispatch flags that control entire kernel families.

## Procedure

### Step 0: Environment Variable Dispatch Scan (MANDATORY)

Before individual kernel analysis, scan for framework env vars that control
major kernel dispatch paths. These are the highest-impact, lowest-effort
optimizations and are often missed by per-kernel analysis.

```bash
# a) Find ALL env vars that control kernel dispatch in the framework
grep -rn "os.environ.get\|os.getenv\|envs\." \
    $FRAMEWORK_ROOT/model_executor/layers/ \
    $FRAMEWORK_ROOT/worker/ \
    $FRAMEWORK_ROOT/compilation/ \
    2>/dev/null | grep -i "rocm\|aiter\|triton\|gemm\|attn\|mfma\|fp4\|fp8\|buffer\|rope"

# b) For EACH env var found, check:
#    - Is it currently SET in the serve script?
#    - What is its DEFAULT value?
#    - What kernel path does it enable/disable?
#    - What is the expected performance impact?

# c) Common high-impact ROCm/vLLM env vars to check:
#    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM   — enables ASM FP4 GEMM (replaces Triton)
#    VLLM_ROCM_FP8_MFMA_PAGE_ATTN       — enables MFMA page attention
#    AMDGCN_USE_BUFFER_OPS              — enables buffer load/store ops
#    AITER_CONFIG_FMOE                  — path to tuned MoE GEMM CSV
#    AITER_CONFIG_GEMM_BF16             — path to tuned BF16 GEMM CSV
#    VLLM_ROCM_USE_AITER_TRITON_ROPE    — Triton vs CK rope kernel
#    CU_NUM                             — forces CU count for config lookup
```

For each unset or suboptimal env var, create a candidate action:
```python
{
    "name": f"Enable {env_var}={recommended_value}",
    "type": "env-var-dispatch",
    "classification": "self-fix",
    "score": gpu_pct_affected * 10,  # high score — low effort, high impact
    "description": f"Set {env_var} in serve script to activate {path_description}",
}
```

### Steps 1-5: Per-Kernel Analysis

For **each** kernel consuming >MIN_GPU_PCT (2.0%) of GPU time, run Steps 1-5 below.
This is the most detailed analysis in the Marathon — invest time here.

### Step 1: Dispatch Path Trace (per kernel)

Trace the full call chain from Python API to GPU kernel launch:

```python
# a) Identify the Python entry point
#    Search the framework source for where the kernel is invoked.
#    Follow the dispatch logic to understand platform branching.

# b) Document the dispatch chain:
# Example (generic):
#
#   Application layer:
#     model.forward() → layer.attention() → rotary_embedding()
#   Framework layer:
#     RotaryEmbedding.forward() → _dispatch_to_backend()
#   Dispatch branch:
#     if platform_is_X: from compiled_extension import kernel  # Path A
#     elif platform_is_Y: fallback_implementation()            # Path B ← are we here?
#     else: vendor_library_call()                              # Path C
#   Extension layer:
#     compiled_extension.so → launch_kernel<<< >>>
#
# Key question: IS THE BEST PATH ACTIVE? Dispatch bugs are common.

# c) Verify which path is actually executing:
import dis, inspect
# Use dis.dis() on the dispatching function to confirm the active branch
# Or add temporary tracing: print(f"Using kernel path: {kernel_fn.__module__}.{kernel_fn.__name__}")
```

Record findings in `state.kernel_dispatch_map[kernel_name]`:
```python
{
    "dispatch_chain": [
        "model.forward()",
        "layer.op_name()",
        "framework.dispatch()",
        "compiled_ext.kernel()"
    ],
    "active_path": "A",    # which branch is actually executing
    "optimal_path": "A",   # which branch SHOULD be executing (might differ!)
    "dispatch_bug": false,  # true if active != optimal
    "library": "sgl-kernel | aiter | triton | hipblaslt | custom",
    "source_type": "triton | cpp_cuda | cpp_hip | jit | vendor_binary",
    "source_file": "/path/to/kernel/source.py or .cu",
    "build_system": "pip_install | cmake | jit_compile | none_binary",
}
```

### Step 2: Variant Discovery

Search for alternative implementations of the same operation:

```bash
# a) Search the framework for alternative implementations
# Look for:
#   - Platform-specific branches (CUDA vs HIP vs CPU)
#   - Backend-specific implementations (triton vs cpp vs vendor)
#   - Compiled but unused optimized paths
#   - Conditional imports that might be taking the wrong branch

# b) Search operator libraries for tuned versions
# For each major library in the framework ecosystem:
rg "def $KERNEL_FUNCTION_NAME" --type py $FRAMEWORK_ROOT/
rg "$KERNEL_CLASS_NAME" --type py $OPERATOR_LIBS/

# c) Check for compiled extensions that provide the same op
python3 -c "
import torch
# List all registered custom ops that match the operation name
ops = [name for name in dir(torch.ops) if hasattr(getattr(torch.ops, name, None), '$OP_NAME')]
print(f'Registered ops matching $OP_NAME: {ops}')
"

# d) Check vendor library support
# Does the vendor GPU library have a fused/optimized version?
```

Record all discovered variants:
```python
{
    "variants": [
        {
            "name": "JIT compiled (default)",
            "source": "/path/to/jit/source.cuh",
            "active": true,
            "performance_rank": 3,  # worst=3 in this example
            "notes": "Compiles at runtime, not optimized for target GPU"
        },
        {
            "name": "sgl-kernel compiled",
            "source": "/path/to/compiled/source.cu",
            "active": false,   # NOT ACTIVE — potential dispatch bug!
            "performance_rank": 1,
            "notes": "Pre-compiled with hipcc -O3, platform-tuned"
        },
        {
            "name": "Vendor library",
            "source": "binary",
            "active": false,
            "performance_rank": 2,
            "notes": "Available but not dispatched to"
        },
    ]
}
```

### Step 3: Configuration Verification

Check whether shape-specific tuning exists for this kernel:

```bash
# a) Find the kernel's config loading path
rg "get_config|load_config|tuned_config" --type py "$KERNEL_SOURCE_DIR/"

# b) Check for shape-specific configs
# Many kernels use M×N×K shapes to select tiling/block parameters.
# Check if configs exist for the EXACT shapes this model uses:
python3 -c "
import json, glob

# Find all config files
config_files = glob.glob('$OPERATOR_LIB_ROOT/**/configs/**/*.json', recursive=True)
config_files += glob.glob('$OPERATOR_LIB_ROOT/**/configs/**/*.csv', recursive=True)

print(f'Found {len(config_files)} config files')

# Check for model-specific shapes
# (get shapes from profiler or model config)
for f in sorted(config_files):
    print(f)
"

# c) Determine if generic defaults or shape-specific configs are in use
# Generic default = optimization opportunity for operator-tuning action
```

Record config status:
```python
{
    "config_status": "shape-specific" | "generic-default" | "no-config" | "hardcoded",
    "config_file": "/path/to/config.json" | null,
    "model_shapes_covered": ["7168x4608", "7168x2048"],
    "model_shapes_missing": ["7168x1024"],  # these fall back to generic
}
```

### Step 4: Build System Analysis

Determine what's needed to apply optimizations:

```bash
# a) How is the kernel built?
#    - pip install -e . → editable install, changes take effect on restart
#    - cmake → need to rebuild the C++ extension
#    - JIT → changes to source .cuh/.cu take effect automatically
#    - vendor binary → cannot modify, only configure

# b) Can we patch-in-place?
#    Some changes only need a Python file edit + server restart.
#    Others require a full library rebuild.

# c) What's the rebuild cost?
#    - Python-only change: ~0 sec
#    - Triton kernel change: ~0 sec (JIT compiled on first call)
#    - C++ extension rebuild: 5-15 min
#    - Full framework rebuild: 10-30 min
#    - Docker image rebuild: 30-60 min (avoid if possible)
```

Record:
```python
{
    "patch_type": "python-dispatch" | "triton-source" | "cpp-rebuild" | "config-only" | "vendor-binary",
    "rebuild_required": false | true,
    "rebuild_command": "cd /path && pip install -e ." | null,
    "rebuild_time_estimate_min": 5,
    "rollback_strategy": "git checkout -- file.py" | "pip install ." | "restore backup",
}
```

### Step 5: Opportunity Report — Classification and Routing

Synthesize findings into actionable items. The key decision: **self-fix** (orchestrator
applies directly) vs **oob-rewrite** (Kernel Manager handles via work queue).

#### Classification Rules

| Signal | Classification | Route |
|--------|---------------|-------|
| `dispatch_bug: true` | **Self-fix** | Orchestrator fast path |
| `config_status: "generic-default"` | **Self-fix** (config) or operator-tuning | Orchestrator or operator-tuning action |
| Inactive variant with better `performance_rank` and Python-only activation | **Self-fix** | Orchestrator fast path |
| Inactive variant requiring C++/HIP rebuild to activate | **OOB-rewrite** | Kernel Manager work queue |
| `source_type` in `(triton, cpp_cuda, cpp_hip)` and no dispatch bug | **OOB-rewrite** | Kernel Manager work queue |
| Multi-file framework scheduling change | **OOB-rewrite** | Kernel Manager work queue |

```python
import json, os, datetime

for kernel_name, analysis in state.kernel_dispatch_map.items():
    opportunities = []

    # ──────────────────────────────────────────────────
    # SELF-FIX targets → push directly onto DFS stack
    # ──────────────────────────────────────────────────

    # Dispatch bug → immediate fix, highest priority (orchestrator fast path)
    if analysis['dispatch_bug']:
        opportunities.append({
            'type': 'dispatch-fix',
            'classification': 'self-fix',
            'score_boost': 10,
            'action': 'Fix dispatch routing in framework',
            'cost_min': analysis.get('rebuild_time_estimate_min', 5),
        })

    # Generic config → operator tuning opportunity
    if analysis['config_status'] == 'generic-default':
        opportunities.append({
            'type': 'operator-tuning',
            'classification': 'self-fix',
            'score_boost': 5,
            'action': 'Run autotuner for model-specific shapes',
            'cost_min': 30,
        })

    # Unused optimized variant (Python-only activation) → self-fix
    for variant in analysis.get('variants', []):
        if (not variant['active']
            and variant['performance_rank'] < analysis.get('active_variant_rank', 99)
            and analysis.get('patch_type') == 'python-dispatch'):
            opportunities.append({
                'type': 'dispatch-fix',
                'classification': 'self-fix',
                'score_boost': 8,
                'action': f'Activate {variant["name"]} (better than current)',
                'cost_min': analysis.get('rebuild_time_estimate_min', 5),
            })

    # ──────────────────────────────────────────────────
    # OOB-REWRITE targets → write to Kernel Manager work queue
    # ──────────────────────────────────────────────────

    # Triton/custom kernel → Kernel Manager dispatches to OOB agents
    if analysis['source_type'] in ('triton', 'cpp_cuda', 'cpp_hip') and not analysis['dispatch_bug']:
        oob_target = {
            'type': 'deep-kernel-opt',
            'classification': 'oob-rewrite',
            'score_boost': 3,
            'action': f'Submit {kernel_name} to Kernel Manager for OOB optimization',
            'cost_min': 15,
        }
        opportunities.append(oob_target)

        # Also write a structured work queue entry for the Kernel Manager
        work_queue_entry = {
            "id": f"{kernel_name}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "kernel_name": kernel_name,
            "gpu_pct": analysis.get('gpu_pct', 0),
            "source_file": analysis.get('source_file', ''),
            "source_type": analysis.get('source_type', ''),
            "dispatch_analysis": {
                "active_path": analysis.get('active_path', ''),
                "optimal_path": analysis.get('optimal_path', ''),
                "dispatch_bug": False,
            },
            "trace_shapes": analysis.get('trace_shapes', {}),
            "constraints": analysis.get('constraints', {}),
            "strategy": _classify_strategy(analysis),
            "backends": _select_backends(analysis),
            "priority": analysis.get('gpu_pct', 0),
        }

        queue_path = os.path.join(
            os.environ.get("RESULT_DIR", "/tmp"),
            "kernel_manager", "work_queue.jsonl"
        )
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        work_queue_entry["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        work_queue_entry["status"] = "pending"
        work_queue_entry["attempts"] = 0
        with open(queue_path, "a") as f:
            f.write(json.dumps(work_queue_entry) + "\n")

        state['kernel_manager_targets_pushed'] = state.get('kernel_manager_targets_pushed', 0) + 1

    # Unused optimized variant requiring rebuild → Kernel Manager
    for variant in analysis.get('variants', []):
        if (not variant['active']
            and variant['performance_rank'] < analysis.get('active_variant_rank', 99)
            and analysis.get('patch_type') != 'python-dispatch'):
            opportunities.append({
                'type': 'framework-rebuild',
                'classification': 'oob-rewrite',
                'score_boost': 8,
                'action': f'Activate {variant["name"]} (requires rebuild)',
                'cost_min': analysis.get('rebuild_time_estimate_min', 10),
            })

    # Push self-fix opportunities directly to DFS action stack
    for opp in opportunities:
        if opp['classification'] == 'self-fix':
            state['action_stack'].append((opp['score_boost'], opp['type'], {
                'kernel': kernel_name,
                **opp,
            }))
        else:
            # OOB-rewrite targets are on the work queue; push a low-priority
            # placeholder on the DFS stack so the orchestrator knows to poll
            state['action_stack'].append((1, 'kernel-manager-poll', {
                'kernel': kernel_name,
                'note': 'Kernel Manager processing — poll results.jsonl',
            }))


def _classify_strategy(analysis):
    """Map analysis findings to a Kernel Manager strategy string."""
    if analysis.get('dispatch_bug'):
        return 'dispatch-fix'
    if analysis.get('config_status') == 'generic-default':
        return 'config-only'
    if analysis.get('source_type') == 'triton':
        if analysis.get('constraints', {}).get('register_constrained'):
            return 'oob-rewrite-register-constrained'
        return 'triton-rewrite'
    if analysis.get('source_type') in ('cpp_cuda', 'cpp_hip'):
        return 'hip-kernel'
    return 'oob-rewrite'


def _select_backends(analysis):
    """Choose which OOB backends to dispatch to based on kernel type."""
    if analysis.get('source_type') in ('cpp_cuda', 'cpp_hip'):
        return ['geak', 'claude']
    if analysis.get('source_type') == 'triton':
        return ['geak', 'codex', 'claude', 'llm']
    return ['codex', 'claude']
```

## Outputs
- `state.kernel_dispatch_map` — fully populated for all top-N kernels (at least 10)
- `state.action_stack` — enriched with AT LEAST 5-10 candidates:
  - **Env-var-dispatch** actions from Step 0 (highest priority — low effort, high impact)
  - **Self-fix** dispatch-fix actions (high priority, orchestrator handles directly)
  - **Config-only** actions (tuned CSV loading, shape-specific configs)
  - **Kernel-manager-poll** placeholders (low priority, remind orchestrator to check results)
- `state.dispatch_bugs_found` — list of dispatch routing issues found
- `state.untuned_shapes` — list of kernel shapes using generic configs
- `$RESULT_DIR/kernel_manager/work_queue.jsonl` — kernel targets for the Kernel Manager
- `state.kernel_manager_targets_pushed` — count of targets pushed to work queue
- Per-kernel analysis logged to `$RESULT_DIR/kernel_analysis/`

## Validation Gate
If action_stack has fewer than 5 candidates after deep analysis, the analysis
is INCOMPLETE. Go back and analyze more kernels, check more env vars, look at
the serve script for missing optimization flags. The profiler data should
yield at least 5-10 actionable items for any non-trivially-optimized model.

## Heuristic Update

- If dispatch bugs found: boost deep-kernel-analysis for remaining kernels to 10
- If >50% kernels using generic configs: boost operator-tuning globally
- If no bugs and all configs shape-specific: reduce deep-kernel-analysis to 2
- If work queue targets pushed > 0: ensure merge-op polling is active in DFS loop

## Why This Matters

This action catches the class of bugs we found with the RoPE kernel: a framework routing
9.5× slower path when a compiled-optimized path was available. Without dispatch tracing,
the kernel appears "slow" and gets submitted to OOB agents for rewriting — wasting time
on the wrong problem. Deep analysis finds the root cause.

The classification system ensures the orchestrator handles simple fixes immediately (fast
path) while complex rewrites go to the Kernel Manager for asynchronous processing with
full local testing and patch generation.
