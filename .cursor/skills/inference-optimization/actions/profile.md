# Action: Profile & Analyze

Combines profiling (Phase 3), TraceLens analysis (Phase 4), and candidate identification (Phase 5).

## Inputs
- Running server from `baseline.md`
- `$RESULT_DIR` with server log and baseline results

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME profiling TraceLens" --top-k 3 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category pitfall --tags TraceLens --compact
```

## Procedure

**Claw mode:** All profiling and filesystem commands must run via `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Profile" section for wrapper syntax.

### Step 1: Profile with Magpie (torch.profiler + TraceLens + Gap Analysis in one shot)

Magpie handles the full profiling pipeline: server launch → profiling → trace collection → TraceLens analysis → gap analysis → cleanup. No manual profiler HTTP endpoints needed.

#### Step 1a: Compute steady-state profiling window

Instead of profiling the entire run (which produces oversized traces), profile only a
representative steady-state window controlled by `DELAY_ITERS` and `MAX_ITERS`.

```bash
EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"

# Steady-state window: profile only a representative slice of execution
_raw_max=$((16 * OSL / CONC))
MAX_ITERS=$(( _raw_max < 256 ? 256 : (_raw_max > 1024 ? 1024 : _raw_max) ))

PROFILE_NUM_PROMPTS=$((CONC * 10))
# R = PROFILE_NUM_PROMPTS / CONC (request multiplier)
_R=$((PROFILE_NUM_PROMPTS / CONC))
DELAY_ITERS=$(( (((_R + 1) / 2) * 5 * OSL) - (MAX_ITERS / 2) ))
if [[ $DELAY_ITERS -lt 0 ]]; then DELAY_ITERS=0; fi

TRACE_DIR="$RESULT_DIR/profile_traces"
mkdir -p "$TRACE_DIR"
```

#### Step 1b: Build framework-specific profiler args

**For SGLang:**
```bash
if [[ "$FRAMEWORK" == "sglang" ]]; then
    PROFILER_SERVER_ARGS="--enable-profile-cuda-graph --enable-shape-discovery-for-cuda-graph-profile"
    PROFILER_EXTRA_ARGS="$BASELINE_SERVER_ARGS $PROFILER_SERVER_ARGS"

    PROFILER_EXTRA_BODY=$(python3 -c "import json; print(json.dumps({
        'shape_discovery': True,
        'roofline_annotations': True,
        'start_step': $DELAY_ITERS,
        'num_steps': $MAX_ITERS,
        'merge_profiles': False,
        'profile_by_stage': False
    }))")

cat > "$RESULT_DIR/profile_config.yaml" <<EOF
benchmark:
  framework: sglang
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: sglang_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    PROFILE_NUM_PROMPTS: $PROFILE_NUM_PROMPTS
    EXTRA_SGLANG_ARGS: "$PROFILER_EXTRA_ARGS"
    SGLANG_PROFILE_WITH_STACK: "true"
    SGLANG_PROFILE_RECORD_SHAPE: "true"
    PROFILE_EXTRA_BODY: '$PROFILER_EXTRA_BODY'
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
  gap_analysis:
    enabled: true
    top_k: 30
  timeout_seconds: 1800
EOF
fi
```

**For vLLM:**
```bash
if [[ "$FRAMEWORK" == "vllm" ]]; then
    PROFILER_VLLM_ARGS="$BASELINE_SERVER_ARGS"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --enforce-eager"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --profiler-config.capture_torch_profiler_dir ${TRACE_DIR}/capture_traces"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --profiler-config.detailed_trace_annotation True"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --profiler-config.delay_iterations $DELAY_ITERS"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --profiler-config.max_iterations $MAX_ITERS"
    PROFILER_VLLM_ARGS="$PROFILER_VLLM_ARGS --profiler-config.ignore_frontend True"

cat > "$RESULT_DIR/profile_config.yaml" <<EOF
benchmark:
  framework: vllm
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: vllm_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    PROFILE_NUM_PROMPTS: $PROFILE_NUM_PROMPTS
    EXTRA_VLLM_ARGS: "$PROFILER_VLLM_ARGS"
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: 1200
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
  gap_analysis:
    enabled: true
    top_k: 30
  timeout_seconds: 1800
EOF
fi
```

#### Step 1c: Run the profiling benchmark

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/profile_config.yaml -o $RESULT_DIR/profile 2>&1", run_in_background=true)`
- poll: every 120s (profiling is slower than vanilla benchmark) call `bash_output(shell_id)` until DONE_REGEX (`Benchmark Result|benchmark_report\.json|✅`) or ERROR_REGEX (`Traceback|exit [1-9]|signal=SIG|OOM`) matches
- collect: `bash(command="cat $RESULT_DIR/profile/benchmark_*/benchmark_report.json")`

**NOTE:** The profiling run uses `CONC * 10` prompts (via `PROFILE_NUM_PROMPTS`) with
steady-state windowing (`DELAY_ITERS`/`MAX_ITERS`) to capture a representative slice
without oversized traces. Throughput numbers include profiling overhead — use clean
baseline numbers from `baseline.md` for performance tracking.

**Key profiler settings enabled for TraceLens:**
- **Eager mode** (vLLM): `--enforce-eager` disables CUDA graphs so trace shows actual kernels instead of opaque `hipGraphLaunch` calls
- **Shape & callstack profiling** (SGLang): `SGLANG_PROFILE_WITH_STACK`, `SGLANG_PROFILE_RECORD_SHAPE`
- **Graph capture tracing**: SGLang `--enable-profile-cuda-graph`, vLLM `--profiler-config.capture_torch_profiler_dir`
- **Detailed annotations**: SGLang `extra_body.roofline_annotations`, vLLM `--profiler-config.detailed_trace_annotation`
- **Steady-state window**: SGLang `extra_body.start_step/num_steps`, vLLM `--profiler-config.delay_iterations/max_iterations`

**ALWAYS `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling** — leaked env vars cause 30x slowdown.

### Step 2: TraceLens analysis (CLI)

Use **filtered** TP-0 trace (generated by `run_baseline.sh`).

**Ensure TraceLens CLI is installed:**
```bash
TraceLens_generate_perf_report_pytorch --help >/dev/null 2>&1 || \
  (cp -r /hyperloom/TraceLens-internal /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal)
```

**Validate trace file:**
```bash
python3 -c "
import gzip, json, sys
path = '$TRACE_DIR/filtered-TP-0.trace.json.gz'
try:
    with gzip.open(path) as f: data = json.load(f)
    print(f'Trace OK: {len(data.get(\"traceEvents\", []))} events')
except Exception as e:
    print(f'Trace validation failed: {e}', file=sys.stderr); sys.exit(1)
"
```

**Generate performance report:**
```bash
mkdir -p "$TRACE_DIR/tracelens_output"
TraceLens_generate_perf_report_pytorch \
  --profile_json_path "$TRACE_DIR/filtered-TP-0.trace.json.gz" \
  --output_xlsx_path "$TRACE_DIR/tracelens_output/perf_report.xlsx" \
  --output_csvs_dir "$TRACE_DIR/tracelens_output/perf_report_csvs" \
  --gpu_arch_json_path /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch/$GPU_TYPE.json \
  --enable_pseudo_ops \
  --group_by_num_kernels
```

**Prepare category data (GPU utilization, top ops, tree data, category filtering):**
```bash
python3 /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path "$TRACE_DIR/filtered-TP-0.trace.json.gz" \
  --platform $GPU_TYPE \
  --output-dir "$TRACE_DIR/tracelens_output"
```

**Run standalone analysis subagents:**

Read the skill file `/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md` and follow Steps 6–10 (system-level analysis, compute kernel analysis, validation, aggregation, and report generation) using:
- Output directory: `$TRACE_DIR/tracelens_output`
- Platform: `$GPU_TYPE`
- Analysis mode: `default` (for training/eager inference) or `inference` (for vLLM/SGLang)

The final standalone analysis report will be at `$TRACE_DIR/tracelens_output/standalone_analysis.md`.

**TraceLens `other` category trap:** Triton kernels launched via `hipModuleLaunchKernel` appear in the `other` category, NOT in `triton`. Check `other_metrics.json` for individual `hipModuleLaunchKernel::*` entries — these may be significant GEAK candidates hidden under a misleading category name. Always drill into `other` before dismissing it.

### Step 3: Identify GEAK candidates

**From TraceLens output or direct trace parsing:**

```python
import gzip, json, time, os

filtered = os.path.join(TRACE_DIR, 'filtered-TP-0.trace.json.gz')
with gzip.open(filtered) as f:
    trace = json.load(f)

kernels = {}
for e in trace.get('traceEvents', []):
    if e.get('cat') == 'kernel':
        name = e.get('name', '')
        kernels.setdefault(name, {'count': 0, 'total_us': 0})
        kernels[name]['count'] += 1
        kernels[name]['total_us'] += e.get('dur', 0)

total = sum(v['total_us'] for v in kernels.values())
candidates = []
for name, v in sorted(kernels.items(), key=lambda x: -x[1]['total_us']):
    pct = v['total_us'] / total * 100
    if pct < 3: break
    is_vendor = any(x in name for x in ['Cijk_', 'aiter::', 'hipModule', 'ck::kernel'])
    if not is_vendor:
        candidates.append((name, pct, v['count']))
        print(f"GEAK candidate: {name} ({pct:.1f}%)")
```

**Decision table for candidate types:**

| Kernel pattern | GEAK? | Why |
|----------------|-------|-----|
| `Cijk_*` (hipBLASLt GEMM) | No | Vendor BLAS |
| `aiter::fmha_v3_*` | No | Vendor attention |
| `triton_*` / `_permute_kernel` | **Yes** | Triton with Python source |
| `topkGatingSoftmax` | **Yes** | MoE routing kernel |
| Custom scheduling/routing | **Yes** | Token dispatch, KV cache ops |

**Architecture-based fallback** (when profiling/TraceLens fails):
```python
import json
config = json.load(open(f'{MODEL}/config.json'))
text_cfg = config.get('text_config', config)
has_moe = text_cfg.get('n_routed_experts', 0) > 0
has_mla = text_cfg.get('kv_lora_rank', 0) > 0

if has_moe:
    print("Estimated: MoE GEMM ~50-60%, Attention ~10-15%, Elementwise ~10-20%, Comm ~10-15%")
else:
    print("Estimated: GEMM ~60-70%, Attention ~10-15%, RMSNorm/Act ~5-10%, Comm ~5-10%")
```

**Exhaustive search checklist before declaring "no candidates":**
- [ ] Checked TraceLens categories for non-vendor kernels
- [ ] Searched Inductor cache: `find /tmp/torchinductor_root -name "*.py" | xargs grep -l "@triton"`
- [ ] Searched framework source: `find /opt/venv -path "*/sglang/*" -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Searched aiter source: `find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Verified ALL kernels >3% GPU time are vendor C++
- [ ] Searched for FUSED kernels that could replace multi-step pipelines

## Accuracy Validation
N/A — profiling is read-only, no changes to validate.

## Outputs
- `gpu_utilization_pct`: computation vs idle vs communication
- `kernel_breakdown`: list of (kernel_name, gpu_pct, is_vendor, is_geak_candidate)
- `geak_candidates`: ranked list of candidates with gpu_pct and source location
- `trace_path`: path to filtered trace for comparative analysis later

## Heuristic Update
For each GEAK candidate found:
- Score = `gpu_pct * expected_speedup_for_type / cost_minutes * (1 - accuracy_risk)`
- Reduction kernels (RMSNorm): expected_speedup=0.5, cost=15min, accuracy_risk=0.15
- Pointwise kernels: expected_speedup=0.3, cost=15min, accuracy_risk=0.1
- Template/GEMM kernels: expected_speedup=0.1, cost=15min, accuracy_risk=0.05, crash_risk=0.5

If >50% GPU time in vendor kernels: boost backend exploration scores significantly.

## Failure Handling
- TraceLens CLI not installed: copy to `/tmp` and install (`cp -r /hyperloom/TraceLens-internal /tmp/ && pip install -e /tmp/TraceLens-internal`)
- TraceLens CLI fails: fall back to direct trace parsing with error recovery (Step 3)
- Trace too large: use filtered trace; if still too large, use architecture-based estimation
- vLLM profiling empty: use `/start_profile` HTTP endpoints or skip to architecture-based
