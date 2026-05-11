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

**Remote mode:** All profiling and filesystem commands must be submitted to the RayJob via Ray Dashboard REST (`POST /api/jobs/`). See [`../modes/REMOTE.md`](../modes/REMOTE.md) "Profile" section for wrapper syntax.

### PD / MoRI profiling topology

For PD disaggregation / MoRI, keep the profiling topology aligned:

- Send benchmark traffic to the public serving endpoint, usually the router.
- Send `/start_profile` and `/stop_profile` to the SGLang backend that owns the stage being profiled, usually the prefill server.
- `/start_profile` and `/stop_profile` MUST be the built-in HTTP endpoints of
  the already-running SGLang server process. Do not create, patch, monkeypatch,
  wrap, or simulate a custom `start_profile` / `stop_profile` implementation,
  and do not add a sidecar service with those routes.
- Do not profile the router unless intentionally measuring router overhead.
- Do not benchmark prefill/decode backends directly when evaluating the full PD path.
- Record the actual prefill/decode/router ports from the launch commands or run context; do not hardcode ports such as `30000` or `8888`.
- In remote mode, the trace output path used by the profiled RayJob server MUST
  be a RayJob-writable `/wekafs/...` path. Set `TRACE_DIR` and
  `SGLANG_TORCH_PROFILER_DIR` before launching the profiled backend, and if the
  implementation calls `/start_profile` directly, pass the same path as
  `output_dir` in the request body. Do not use sandbox-only
  `/workspace/hyperloom/...` paths for RayJob trace output.
- For PD/MoRI, the default `/start_profile` request MUST start profiling
  immediately. Do not pass `start_step` / `num_steps` by default; short profile
  traffic can miss the requested window and make `/stop_profile` report
  `Profiling is not in progress`.
- Sandbox-side TraceLens summaries, manifests, and the final report must be
  written under `/workspace/hyperloom/` after reading the RayJob-produced
  `/wekafs/...` trace/result paths.

A profiling attempt is valid for TraceLens if TraceLens can read the raw trace and produce GPU timeline / kernel summary outputs. Prefer `/stop_profile` returning `200 OK`, but do not reject the trace solely because `/stop_profile` returned 500: SGLang can still write a complete trace after reporting an internal error. If `/stop_profile` times out, the gzip trace is invalid, or TraceLens reports `No GPU events found in the trace`, treat the profiling attempt as invalid. Do not conclude that the workload has no GPU events from an invalid trace.

For PD/MoRI profiling, use this default `/start_profile` payload:

```json
{
  "output_dir": "$TRACE_DIR",
  "activities": ["CPU", "GPU"],
  "with_stack": true,
  "record_shapes": true,
  "profile_prefix": "prefill"
}
```

Only use `start_step` / `num_steps` after a successful immediate profile proves
the trace is too large, and only when profile traffic is long enough to cover
the requested engine-step window. Otherwise reduce output length, prompt count,
and concurrency before increasing timeout. Use a larger `/stop_profile` timeout
for distributed serving as a fallback (for example, `6000s`), while keeping the
benchmark endpoint and profile endpoint aligned with the topology above.

After `/stop_profile`, do not decide based on a single fixed sleep. Poll
`$TRACE_DIR` until trace count and total size are stable. If `$TRACE_DIR` is
empty, inspect the profiled server log for `Traces are saved to:` before
declaring failure. If the trace was written to an unexpected path, record that
path and move/copy the summary artifacts into the final `/workspace/hyperloom/`
bundle.

Do not manually parse large raw trace files in the normal path. The normal path is:

```text
trace -> TraceLens -> GPU timeline / kernel summary / report
```

Only use lightweight trace diagnostics when TraceLens fails, and only for debugging the profiling setup.

### Step 1: Profile with torch.profiler

**NOTE:** `run_baseline.sh` already handles profiling in one run. It pre-sets `SGLANG_TORCH_PROFILER_DIR` at server launch, runs the clean baseline, then activates profiling via `/start_profile` HTTP endpoint.

If running separately:
```bash
RUN_CONTEXT_FILE="$RESULT_DIR/run_context.env" bash "$SCRIPTS_DIR/run_profile.sh"
```

**PD/MoRI exception:** If the official scripts cannot express the required
prefill/decode/router topology, use a documented Ray Dashboard REST Python
driver instead. The driver may launch prefill/decode/router processes and call
the built-in SGLang `/start_profile` and `/stop_profile` endpoints, but it must
follow the topology and path rules above, set `SGLANG_TORCH_PROFILER_DIR` before
launching the profiled backend, pass the `output_dir` payload to
`/start_profile`, validate that the produced trace has GPU/kernel events, invoke
TraceLens on the produced trace, and save final summaries under
`/workspace/hyperloom/`. This exception does not allow custom profiling
endpoints or a sidecar `stop_profile` service.

**vLLM V1 caveat:** `multiprocessing.spawn` workers — main process profiler gets empty traces. Use `/start_profile` + `/stop_profile` HTTP endpoints instead.

**vLLM v0.17+ `--profiler-config` format:**

vLLM v0.17 changed the profiling interface. The `/start_profile` endpoint requires `--profiler-config` at server launch time. The config MUST be JSON format:

```bash
# CORRECT — JSON format (vLLM v0.17+)
python3 -m vllm.entrypoints.openai.api_server \
    --profiler-config '{"profiler": "torch", "trace_dir": "/wekafs/inference-optimization/traces/<run_id>"}' \
    ...

# WRONG — key=value format (will fail with "invalid JSON")
python3 -m vllm.entrypoints.openai.api_server \
    --profiler-config 'profiler=torch,trace_dir=/wekafs/inference-optimization/traces/<run_id>' \
    ...
```

**ALWAYS `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling** — leaked env vars cause 30x slowdown.

### Step 2: TraceLens analysis (CLI)

Record tracing start before calling TraceLens:
```bash
python3 $SCRIPTS_DIR/trace_action.py --component tracelens --action start
```

Use **filtered** TP-0 trace (generated by `run_baseline.sh`, `run_profile.sh`,
or the documented PD/MoRI profile driver).

**Ensure TraceLens CLI is installed:**
```bash
TL_DIR="/hyperloom/TraceLens-internal"
[ -d "/opt/TraceLens" ] && TL_DIR="/opt/TraceLens"
TraceLens_generate_perf_report_pytorch_inference --help >/dev/null 2>&1 || \
  (cp -r $TL_DIR /tmp/TraceLens-internal && pip install -e /tmp/TraceLens-internal)
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
TL_DIR="/hyperloom/TraceLens-internal"
[ -d "/opt/TraceLens" ] && TL_DIR="/opt/TraceLens"

mkdir -p "$TRACE_DIR/tracelens_output"
mkdir -p "$TRACE_DIR/tracelens_output/perf_report_csvs"
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path "$TRACE_DIR/filtered-TP-0.trace.json.gz" \
  --output_xlsx_path "$TRACE_DIR/tracelens_output/perf_report.xlsx" \
  --output_csvs_dir "$TRACE_DIR/tracelens_output/perf_report_csvs" \
  --gpu_arch_json_path "$TL_DIR/TraceLens/AgenticMode/Standalone/utils/arch/$GPU_TYPE.json" \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --enable_kernel_summary
```

**Prepare category data (GPU utilization, top ops, tree data, category filtering):**
```bash
TL_DIR="/hyperloom/TraceLens-internal"
[ -d "/opt/TraceLens" ] && TL_DIR="/opt/TraceLens"

if [ -f "$TRACE_DIR/tracelens_output/perf_report_csvs/ops_summary.csv" ]; then
  python3 "$TL_DIR/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py" \
    --trace-path "$TRACE_DIR/filtered-TP-0.trace.json.gz" \
    --platform $GPU_TYPE \
    --output-dir "$TRACE_DIR/tracelens_output"
else
  echo "TraceLens produced GPU-only output; use kernel_summary.csv as fallback"
fi
```

**Run standalone analysis subagents:**

Read the skill file `/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md` and follow Steps 6–10 (system-level analysis, compute kernel analysis, validation, aggregation, and report generation) using:
- Output directory: `$TRACE_DIR/tracelens_output`
- Platform: `$GPU_TYPE`
- Analysis mode: `default` (for training/eager inference) or `inference` (for vLLM/SGLang)

The final standalone analysis report will be at `$TRACE_DIR/tracelens_output/standalone_analysis.md`.

Record tracing end after TraceLens completes:
```bash
python3 $SCRIPTS_DIR/trace_action.py --component tracelens --action end
```
**TraceLens `other` category trap:** Triton kernels launched via `hipModuleLaunchKernel` appear in the `other` category, NOT in `triton`. Check `other_metrics.json` for individual `hipModuleLaunchKernel::*` entries — these may be significant OOB candidates hidden under a misleading category name. Always drill into `other` before dismissing it.

### Step 3: Identify kernel optimization candidates

These candidates will be submitted to active OOB backends in `KERNEL_OPT_BACKENDS`
(default `codex,claude`) simultaneously when capacity allows. Codex and Claude race as
equals — the best verified result wins regardless of which OOB backend produced it.

**From the trace kernel breakdown:**

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

# 5-tier kernel classification (replaces binary is_vendor filter)
def classify_kernel(name):
    """Classify kernel into optimization tier. ALL tiers are actionable."""
    if any(x in name for x in ['triton_', '_permute_kernel', 'triton_poi_', 'triton_red_', 'triton_tem_']):
        return 'T1_TRITON'
    if any(x in name for x in ['aiter::', 'fmha_v3', 'mha_fwd', 'fused_moe', 'moe_ck',
                                 'topkGating', '_gemm_a8w8', '_fused_rms']):
        return 'T2_AITER_CK'
    if any(x in name for x in ['launch_server', 'schedule', 'batch', 'token_dispatch',
                                 'kv_cache', 'radix', 'prefix_match']):
        return 'T3_FRAMEWORK'
    if any(x in name for x in ['nccl', 'rccl', 'allreduce', 'AllReduce', 'broadcast',
                                 'all_gather', 'reduce_scatter']):
        return 'T4_COMM'
    if any(x in name for x in ['Cijk_', 'hipModule', 'ck::kernel']):
        return 'T5_COMPILED'
    if 'vectorized_elementwise_kernel' in name:
        return 'T2_AITER_CK'  # C++ dispatch — Strategy C can convert to T1
    return 'T2_AITER_CK'  # default to aiter/CK tier for unknown GPU kernels

candidates = []
for name, v in sorted(kernels.items(), key=lambda x: -x[1]['total_us']):
    pct = v['total_us'] / total * 100
    if pct < 1: break  # lowered from 3% to 1% — deep optimization needs wider net
    tier = classify_kernel(name)
    candidates.append((name, pct, v['count'], tier))
    print(f"[{tier}] {name} ({pct:.1f}%, {v['count']}x)")
```

**5-Tier Classification — Decision Table:**

ALL tiers are optimization targets. The optimization *method* varies by tier, not whether
it's attempted. OOB can rewrite Triton/Python/framework source and can work on HIP/C++
source when the user provides source paths (e.g., `/opt/aiter/csrc/`). Do NOT skip
source-backed kernels as "vendor" merely because their runtime names look compiled.

| Tier | Kernel pattern | Optimization method | Backend |
|------|----------------|--------------------:|---------|
| **T1: Triton/Inductor** | `triton_*`, `_permute_kernel`, Inductor cache | Direct source rewrite, block tuning | OOB (Codex/Claude) |
| **T2: aiter/CK dispatch** | `aiter::*`, `topkGating*`, `_gemm_a8w8*`, `vectorized_elementwise*` | Source rewrite (if source provided), Python dispatch rewrite, config tuning, Strategy C selective compile, env flags | OOB (Codex/Claude) |
| **T3: Framework scheduling** | Token dispatch, KV cache ops, batch scheduler, prefill/decode overlap | Source edits in SGLang/vLLM scheduling code, `pip install -e .` | OOB (Claude/Codex) |
| **T4: Communication** | NCCL/RCCL, AllReduce, custom allreduce | Topology tuning, quantized collectives, overlap scheduling, NCCL env vars | OOB (Claude/Codex) |
| **T5: Compiled binaries** | `Cijk_*` (hipBLASLt), `ck::kernel` | GEMM CSV tuning, shape-specific config, NOT source rewrite | vendor-kernel-config action |

**T2 sub-strategies (aiter/CK kernels are NOT "unoptimizable"):**
- **Source rewrite (OOB)**: When user provides source paths (e.g., `/opt/aiter/csrc/`), map trace kernel names back to `.cu`/`.hip` files using `rg` in the provided repo, then submit full source to OOB. OOB can rewrite source-backed HIP/C++ kernels.
- **Dispatch rewrite**: aiter Python wrappers (`aiter/ops/`) choose kernel variants at runtime. Rewriting dispatch logic for model-specific shapes can select faster paths.
- **Config tuning**: GEMM CSV files, fused MoE configs, FP8 bypass removal (see GLM-5-FP8 case).
- **Strategy C**: Selective `torch.compile` on norm/activation submodules → converts `vectorized_elementwise_kernel` to Inductor Triton → T1 optimization applies.
- **Env flags**: `SGLANG_ROCM_FUSED_DECODE_MLA`, `ROCM_QUICK_REDUCE_QUANTIZATION`, `AITER_ENABLE_VSKIP`.

**Architecture-based fallback** (only after live profiling is explicitly unavailable):
```python
import json
config = json.load(open(f'{MODEL}/config.json'))
text_cfg = config.get('text_config', config)
has_moe = text_cfg.get('n_routed_experts', 0) > 0
has_mla = text_cfg.get('kv_lora_rank', 0) > 0

if has_moe:
    print("Estimated: T5 vendor GEMM ~30-40%, T2 aiter/CK ~20-30%, T4 comm ~10-20%, T1 Triton ~5-15%, T3 scheduling ~5-10%")
else:
    print("Estimated: T5 vendor GEMM ~40-50%, T1 Triton ~15-25%, T4 comm ~5-10%, T3 scheduling ~5-10%")
```

**Exhaustive search checklist before declaring "no candidates":**
- [ ] Classified ALL kernels >1% GPU time into tiers T1-T5
- [ ] Searched Inductor cache: `find /tmp/torchinductor_root -name "*.py" | xargs grep -l "@triton"`
- [ ] Searched framework source: `find /opt/venv -path "*/sglang/*" -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Searched aiter source: `find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Searched aiter dispatch wrappers: `find /sgl-workspace/aiter/aiter/ops -name "*.py"`
- [ ] Searched framework scheduling code: `find /opt/venv -path "*/sglang/srt/managers/*" -name "*.py"`
- [ ] Checked for NCCL/RCCL communication hotspots in T4
- [ ] Searched for FUSED kernels that could replace multi-step pipelines
- [ ] If `vectorized_elementwise_kernel` >5%: tried Strategy C selective compile
- [ ] If T2 aiter kernels >30%: checked dispatch logic for model-specific tuning
- [ ] If T4 comm >15%: checked collective config, overlap opportunities
- [ ] If user provided kernel source paths: verified those kernels are included as candidates regardless of name

## Accuracy Validation
N/A — profiling is read-only, no changes to validate.

## Outputs
- `gpu_utilization_pct`: computation vs idle vs communication
- `kernel_breakdown`: list of (kernel_name, gpu_pct, tier, optimization_method)
- `tier_summary`: dict of tier → total gpu_pct (T1 through T5)
- `kernel_opt_candidates`: ranked list of ALL candidates >1% gpu_pct with tier and source location
- `trace_path`: path to filtered trace for comparative analysis later

## Heuristic Update
For each candidate found, score by tier:

| Tier | expected_speedup | cost_minutes | accuracy_risk | crash_risk |
|------|:----------------:|:------------:|:-------------:|:----------:|
| T1: Triton/Inductor | 0.5 | 15 | 0.15 | 0.1 |
| T2: aiter/CK dispatch | 0.2 | 30 | 0.10 | 0.2 |
| T3: Framework scheduling | 0.15 | 45 | 0.05 | 0.3 |
| T4: Communication | 0.10 | 30 | 0.0 | 0.1 |
| T5: Compiled binaries | 0.05 | 20 | 0.05 | 0.1 |

- Score = `gpu_pct * expected_speedup / cost_minutes * (1 - accuracy_risk) * (1 - crash_risk)`
- If T2+T5 >60% GPU time: boost `call-stack-opt` and `vendor-kernel-config` scores
- If T4 >15% GPU time: boost communication tuning scores
- If T3 scheduling idle >20%: boost framework scheduling optimization scores

## Failure Handling
- TraceLens CLI not installed: copy to `/tmp` and install (`cp -r /hyperloom/TraceLens-internal /tmp/ && pip install -e /tmp/TraceLens-internal`)
- TraceLens CLI fails: fall back to direct trace parsing with error recovery (Step 3)
- No trace after `/stop_profile`: poll `$TRACE_DIR` until stable, inspect server logs for the actual `Traces are saved to:` path, then rerun profiling with the canonical immediate `output_dir` payload if no valid trace exists
- Trace too large: use filtered trace; if still too large, reduce profile traffic or use `kernel_summary.csv` / lightweight candidate extraction before using architecture-based estimation
- vLLM profiling empty: use `/start_profile` HTTP endpoints or skip to architecture-based
