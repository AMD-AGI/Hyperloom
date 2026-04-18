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

### Step 0 (REQUIRED): Confirm TraceLens patches are applied

All the profiler args used below — SGLang's `--enable-profile-cuda-graph` /
`--enable-shape-discovery-for-cuda-graph-profile`, vLLM's
`--profiler-config.capture_torch_profiler_dir` / `.detailed_trace_annotation` — are
**added by the TraceLens patches, not by upstream**. Without the patches the server
will die at startup with `unrecognized arguments`, and no `capture_traces/` folder
will be produced regardless of how the YAML is written.

Execute `actions/setup.md` **Step 4 (Apply TraceLens patches)** before continuing. If
you are unsure whether patches are already applied, run the one-liner verifier in that
step — it's idempotent.

**Quick gate** (fail fast if setup was skipped):
```bash
if [[ "$FRAMEWORK" == "sglang" ]]; then
  python3 -c "from sglang.srt.server_args import ServerArgs; import inspect; \
    assert 'enable_shape_discovery_for_cuda_graph_profile' in inspect.getsource(ServerArgs)" \
    || { echo "ERROR: TraceLens sglang patches missing; run setup.md Step 4"; exit 1; }
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  python3 -c "from vllm.config.profiler import ProfilerConfig; import inspect; \
    assert 'capture_torch_profiler_dir' in inspect.getsource(ProfilerConfig)" \
    || { echo "ERROR: TraceLens vLLM patches missing; run setup.md Step 4"; exit 1; }
fi
```

### Step 1: Profile with Magpie (torch.profiler + TraceLens + Gap Analysis in one shot)

Magpie handles the full profiling pipeline: server launch → profiling → trace collection → TraceLens analysis → gap analysis → cleanup. No manual profiler HTTP endpoints needed.

#### Step 1a: Compute steady-state profiling window

Instead of profiling the entire run (which produces oversized traces), profile only a
representative steady-state window controlled by `DELAY_ITERS` and `MAX_ITERS`.

```bash
EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"

# Steady-state window: profile only a representative slice of execution.
# MAX_ITERS = active profiler steps; clamped to [256, 1024] to bound trace size.
_raw_max=$((16 * OSL / CONC))
MAX_ITERS=$(( _raw_max < 256 ? 256 : (_raw_max > 1024 ? 1024 : _raw_max) ))

PROFILE_NUM_PROMPTS=$((CONC * 10))
# R = PROFILE_NUM_PROMPTS / CONC (request multiplier).
# Total decode steps across the run ≈ R * OSL. We aim to start profiling near
# the middle of the run, i.e. at ~ (R * OSL) / 2, and then capture MAX_ITERS
# steps centered on that midpoint. Expressed as bash integer arithmetic:
#   midpoint = (_R * OSL) / 2
#   DELAY_ITERS = max(0, midpoint - MAX_ITERS/2)
# Clamped to a safe margin from the end so the profiler actually fires even
# after the run is shortened by early-EOS or short OSL.
_total_decode_iters=$(( _R * OSL ))
_midpoint=$(( _total_decode_iters / 2 ))
DELAY_ITERS=$(( _midpoint - (MAX_ITERS / 2) ))
if [[ $DELAY_ITERS -lt 0 ]]; then DELAY_ITERS=0; fi
# Safety: if window would end after total decode, shrink MAX_ITERS to fit.
_window_end=$(( DELAY_ITERS + MAX_ITERS ))
if [[ $_window_end -gt $_total_decode_iters ]]; then
    MAX_ITERS=$(( _total_decode_iters - DELAY_ITERS ))
    [[ $MAX_ITERS -lt 32 ]] && { DELAY_ITERS=0; MAX_ITERS=$(( _total_decode_iters < 256 ? _total_decode_iters : 256 )); }
fi

# For very short validation runs (e.g., CONC≤8 + OSL≤128) the midpoint formula
# can still push the window past the end. You may override with small values:
#   DELAY_ITERS=100 MAX_ITERS=200
# bash profile.md  # for TraceLens patch integration validation only
echo "Profile window: start_step=$DELAY_ITERS num_steps=$MAX_ITERS (total≈$_total_decode_iters)"

TRACE_DIR="$RESULT_DIR/profile_traces"
mkdir -p "$TRACE_DIR"
```

**Formula history:** older versions of this skill had
`DELAY_ITERS = (((R+1)/2) * 5 * OSL) - (MAX_ITERS/2)` — the `* 5` was a bug that
pushed the start step far past the actual run duration on any realistic workload
(e.g., OSL=128/CONC=8/R=10 → DELAY_ITERS=3072 vs total ≈ 1280). The midpoint
formula above replaces it.

#### Step 1b: Build framework-specific profiler args

**`BASELINE_SERVER_ARGS`** is the server-argument string carried over from
`actions/baseline.md`. If you skipped baseline (e.g., pure profile validation run),
set it empty: `BASELINE_SERVER_ARGS="${BASELINE_SERVER_ARGS:-}"`. The default values
baked into the Magpie benchmark script (`--mem-fraction-static=0.8
--disable-radix-cache`) are sufficient for trace collection.

**For SGLang:**
```bash
BASELINE_SERVER_ARGS="${BASELINE_SERVER_ARGS:-}"
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
BASELINE_SERVER_ARGS="${BASELINE_SERVER_ARGS:-}"
if [[ "$FRAMEWORK" == "vllm" ]]; then
    PROFILER_VLLM_ARGS="$BASELINE_SERVER_ARGS"
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
```bash
magpie benchmark --benchmark-config "$RESULT_DIR/profile_config.yaml" -o "$RESULT_DIR/profile"
```

**NOTE:** The profiling run uses `CONC * 10` prompts (via `PROFILE_NUM_PROMPTS`) with
steady-state windowing (`DELAY_ITERS`/`MAX_ITERS`) to capture a representative slice
without oversized traces. Throughput numbers include profiling overhead — use clean
baseline numbers from `baseline.md` for performance tracking.

**Key profiler settings enabled for TraceLens:**
- **Shape & callstack profiling** (SGLang): `SGLANG_PROFILE_WITH_STACK`, `SGLANG_PROFILE_RECORD_SHAPE`
- **Graph capture tracing**: SGLang `--enable-profile-cuda-graph`, vLLM `--profiler-config.capture_torch_profiler_dir`
- **Detailed annotations**: SGLang `extra_body.roofline_annotations`, vLLM `--profiler-config.detailed_trace_annotation`
- **Steady-state window**: SGLang `extra_body.start_step/num_steps`, vLLM `--profiler-config.delay_iterations/max_iterations`

**ALWAYS `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling** — leaked env vars cause 30x slowdown.

### Step 2: TraceLens analysis (CLI)

**CRITICAL — pick the right CLI.** TraceLens ships multiple `TraceLens_*` entry
points. Only one of them can stitch the runtime trace with the per-BS capture
traces produced by the patched sglang/vLLM; the others treat `hipGraphLaunch` as
a black box and will report ~88% GPU time in one opaque bucket — useless for
kernel-level analysis.

| CLI | `--capture_folder` | Use when |
|-----|:---:|----------|
| `TraceLens_generate_perf_report_pytorch_inference` | ✅ | **Always prefer for sglang/vLLM with graph capture** (this action) |
| `TraceLens_generate_perf_report_pytorch` | ❌ | Only for traces **without** cuda graph capture (training, eager inference) |

**Ensure TraceLens CLI is installed:**
```bash
TraceLens_generate_perf_report_pytorch_inference --help >/dev/null 2>&1 || \
  (cp -r /shared_nfs/*/TraceLens-internal /tmp/TraceLens-internal 2>/dev/null; \
   pip install -e /tmp/TraceLens-internal)
```

**Locate the trace files produced by Magpie.** Magpie writes to
`$RESULT_DIR/profile/benchmark_<fw>_<ts>/torch_trace/` (not `$TRACE_DIR` directly),
so resolve the actual paths:
```bash
WORKSPACE=$(ls -td "$RESULT_DIR"/profile/benchmark_* | head -1)
RUNTIME_TRACE=$(ls "$WORKSPACE"/torch_trace/*-TP-0.trace.json.gz 2>/dev/null | head -1)
CAPTURE_FOLDER="$WORKSPACE/torch_trace/capture_traces"

# Sanity: both must exist, otherwise the patch chain didn't work
[[ -f "$RUNTIME_TRACE" ]] || { echo "ERROR: runtime trace missing — sglang profiler never fired"; exit 1; }
[[ -d "$CAPTURE_FOLDER" ]] && [[ $(ls "$CAPTURE_FOLDER"/bs_*.json.gz 2>/dev/null | wc -l) -gt 0 ]] \
    || { echo "ERROR: capture folder empty — --enable-profile-cuda-graph was not honored (patches not applied?)"; exit 1; }
echo "Runtime trace: $RUNTIME_TRACE ($(du -h "$RUNTIME_TRACE" | awk '{print $1}'))"
echo "Capture files: $(ls "$CAPTURE_FOLDER"/bs_*.json.gz | wc -l)"
```

**Validate runtime trace has roofline annotations** (sglang only — vLLM uses a
different annotation schema):
```bash
python3 -c "
import gzip, json, sys
with gzip.open('$RUNTIME_TRACE') as f: data = json.load(f)
evts = data.get('traceEvents', [])
rp = sum(1 for e in evts if 'sglang_profiler::' in e.get('name',''))
print(f'Runtime trace: {len(evts)} events, {rp} sglang_profiler:: annotations')
if rp == 0 and '$FRAMEWORK' == 'sglang':
    print('ERROR: no roofline annotations — extra_body.roofline_annotations was not honored', file=sys.stderr)
    sys.exit(1)
"
```

**Resolve GPU arch JSON path** (TraceLens-internal is the canonical source):
```bash
GPU_ARCH_DIR=$(ls -d /shared_nfs/*/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch 2>/dev/null | head -1)
case "$RUNNER_TYPE" in
    mi300x) GPU_ARCH_JSON="$GPU_ARCH_DIR/MI300X.json" ;;
    mi325x) GPU_ARCH_JSON="$GPU_ARCH_DIR/MI325X.json" ;;
    mi355x) GPU_ARCH_JSON="$GPU_ARCH_DIR/MI355X.json" ;;
    *)      GPU_ARCH_JSON="$GPU_ARCH_DIR/MI355X.json" ;;
esac
```

**Generate performance report (INFERENCE-AWARE, with capture folder):**
```bash
mkdir -p "$TRACE_DIR/tracelens_output"
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path "$RUNTIME_TRACE" \
  --capture_folder "$CAPTURE_FOLDER" \
  --output_xlsx_path "$TRACE_DIR/tracelens_output/perf_report.xlsx" \
  --output_csvs_dir "$TRACE_DIR/tracelens_output/perf_report_csvs" \
  --gpu_arch_json_path "$GPU_ARCH_JSON" \
  --enable_pseudo_ops \
  --enable_kernel_summary \
  --group_by_num_kernels
```

**Sanity-check the stitching worked.** If the inference CLI silently falls back
to "no capture folder" mode (e.g. capture folder empty or schema mismatch), the
output will be dominated by `hipGraphLaunch`. Detect this explicitly:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('$TRACE_DIR/tracelens_output/perf_report_csvs/ops_summary_by_category.csv')
pct_other = df[df['op category']=='other']['Percentage (%)'].sum()
pct_gemm  = df[df['op category']=='GEMM']['Percentage (%)'].sum()
print(df.to_string(index=False))
if pct_other > 80 or pct_gemm < 20:
    print(f'ERROR: kernels not stitched (other={pct_other:.1f}%, GEMM={pct_gemm:.1f}%).', file=__import__('sys').stderr)
    print('       Runtime trace may be missing capture annotations, or wrong CLI was used.', file=__import__('sys').stderr)
    __import__('sys').exit(1)
print(f'Stitching OK: GEMM={pct_gemm:.1f}%, other={pct_other:.1f}%')
"
```

**Prepare category data (GPU utilization, top ops, tree data, category filtering):**
```bash
TRACELENS_ROOT=$(ls -d /shared_nfs/*/TraceLens-internal 2>/dev/null | head -1)
GPU_TYPE="${RUNNER_TYPE^^}"  # mi355x → MI355X
python3 "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py" \
  --trace-path "$RUNTIME_TRACE" \
  --platform "$GPU_TYPE" \
  --output-dir "$TRACE_DIR/tracelens_output"
```

**Run standalone analysis subagents:**

Read the skill file `$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md` and follow Steps 6–10 (system-level analysis, compute kernel analysis, validation, aggregation, and report generation) using:
- Output directory: `$TRACE_DIR/tracelens_output`
- Platform: `$GPU_TYPE`
- Analysis mode: **`inference`** (for vLLM/SGLang — this is the default for this action since Step 2 uses the inference CLI)

**Hard requirements when executing the skill (do not relax, even if a category looks trivial or LLM-dominated):**
1. **Strictly follow the step order** in the skill file — do NOT skip any step and do NOT merge steps. This applies to LLM-heavy categories (e.g. kernel fusion, elementwise) which are the most commonly skipped.
2. **In Step 6 and Step 7, each category MUST be executed by an independent Task subagent** (`subagent_type: generalPurpose`) to ensure context isolation. Launch them in parallel exactly as the skill specifies; never analyze multiple categories inside a single subagent turn or inline in the orchestrator.
3. **Each subagent MUST write out its findings file following the sub-agent template** defined in the skill:
   - Step 6 (system-level) → `$TRACE_DIR/tracelens_output/system_findings/<name>_findings.md`
   - Step 7 (compute kernel) → `$TRACE_DIR/tracelens_output/category_findings/<category>_findings.md`
4. **The final aggregated report MUST follow the report template** in the skill and be written to `$TRACE_DIR/tracelens_output/standalone_analysis.md`.

These requirements apply identically whether the skill is invoked in TraceLens Standalone mode or as part of the Hyperloom E2E flow.

The final standalone analysis report will be at `$TRACE_DIR/tracelens_output/standalone_analysis.md`.

**TraceLens `other` category trap:** Triton kernels launched via `hipModuleLaunchKernel` appear in the `other` category, NOT in `triton`. Check `other_metrics.json` for individual `hipModuleLaunchKernel::*` entries — these may be significant GEAK candidates hidden under a misleading category name. Always drill into `other` before dismissing it.

### Step 3: Identify GEAK candidates

**From TraceLens output or direct trace parsing:**

```python
import gzip, json, time, os

# RUNTIME_TRACE is set in Step 2. For pure DFS-loop usage (no profile run yet)
# fall back to any *-TP-0.trace.json.gz under $RESULT_DIR/profile.
import glob
runtime_trace = os.environ.get('RUNTIME_TRACE') or \
    (sorted(glob.glob(f"{os.environ['RESULT_DIR']}/profile/**/torch_trace/*-TP-0.trace.json.gz", recursive=True))[-1])
with gzip.open(runtime_trace) as f:
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
- **Server startup fails with `unrecognized arguments: --enable-profile-cuda-graph` (sglang) or `--profiler-config.capture_torch_profiler_dir` (vLLM):** TraceLens patches were not applied. Go back to `setup.md` Step 4 and apply them.
- **`capture_traces/` folder empty but server started fine:** `--enable-profile-cuda-graph` flag wasn't passed through. Verify `EXTRA_SGLANG_ARGS` contains it in the Magpie YAML and that Magpie's benchmark script doesn't strip it.
- **Runtime trace has 0 `sglang_profiler::*` annotations:** `extra_body.roofline_annotations=true` wasn't honored by the `/start_profile` endpoint. Check `SGLANG_PROFILE_RECORD_SHAPE` / `SGLANG_PROFILE_WITH_STACK` are exported, and that `scheduler_profiler_mixin.patch` was applied.
- **Step 2 sanity check fails (`other` > 80%, GEMM < 20%):** You're running the wrong TraceLens CLI, or the capture folder wasn't linked. Confirm `TraceLens_generate_perf_report_pytorch_inference` (not the non-inference variant) and `--capture_folder` points to a non-empty directory.
- **TraceLens CLI not installed:** `pip install -e $(ls -d /shared_nfs/*/TraceLens-internal | head -1)` (copy to `/tmp` first if source is read-only).
- **TraceLens CLI fails with trace parse errors:** fall back to direct trace parsing in Step 3.
- **Trace too large (>1GB):** reduce `MAX_ITERS` (Step 1a) or `PROFILE_NUM_PROMPTS`.
