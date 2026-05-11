Use /wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization-multi-node skill to optimize {model_hf} inference performance.
mode: remote

Configuration:
Model path: {model_path}
Framework: {framework}
Precision: {precision}
Inference params: ISL={isl}, OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO=0.8
MoRI topology: prefill TP=8 EP=1, decode TP=8 EP=1
GPU type: {gpu_type}
InferenceX path: {inferencex_path}

Environment:
The current runtime (Claw client) cannot access /wekafs directly.
RayJob / TraceLens can all access /wekafs.
Use SaFE MCP for RayJob lifecycle and Ray Dashboard REST for commands inside the RayJob.

Task submission:
RayJob submit to the core42-sandbox workspace.
RayJob image: {rayjob_image}
One head, one worker:
  head:   CPU=96, GPU=8, memory=1024Gi, ephemeralStorage=400Gi
  worker: CPU=96, GPU=8, memory=1024Gi, ephemeralStorage=400Gi

env (all must be included in workload_create.env):
  - RAY_JOB_ENTRYPOINT=base64("tail -f /dev/null")
  - SAFE_API_KEY={safe_api_key}
  - OOB_API_KEY={safe_api_key}
  - ANTHROPIC_API_KEY={safe_api_key}
  - OPENAI_API_KEY={safe_api_key}
  - AMD_LLM_API_KEY={safe_api_key}
  - LLM_API_KEY={safe_api_key}
  - OOB_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
  - PATH_TO_BNXT_TAR_PACKAGE=/wekafs/primus/data/libbnxt/libbnxt_re-234.0.154.0.tar.gz
  - REBUILD_BNXT=0

entry_points (per resource, base64-encoded init scripts; same content for head and worker so bootstrap runs on BOTH nodes):
  - head:   base64(
      'export PATH=/opt/venv/bin:$PATH'
      ' && export SKILL_ROOT=/wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization-multi-node'
      ' && export HYPERLOOM_BUNDLE=/wekafs/fully-local'
      ' && export FRAMEWORK={framework}'
      ' && bash $SKILL_ROOT/scripts/bootstrap.sh'
    )
  - worker: base64(
      'export PATH=/opt/venv/bin:$PATH'
      ' && export SKILL_ROOT=/wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization-multi-node'
      ' && export HYPERLOOM_BUNDLE=/wekafs/fully-local'
      ' && export FRAMEWORK={framework}'
      ' && bash $SKILL_ROOT/scripts/bootstrap.sh'
    )

After the RayJob is Running, verify with one short Ray Dashboard REST job:
`source /etc/profile.d/hyperloom-env.sh && which oob && which claude && which codex && which ray && oob --help | head -5`.

Serving:
- Use SGLang MoRI 1P1D
- prefill: num-worker=1, TP=8, EP=1, DP-attn=false, GPUs=8 (head node)
- decode:  num-worker=1, TP=8, EP=1, DP-attn=false, GPUs=8 (worker node)
- transfer backend: mori

Kernel Optimization:
KERNEL_OPT_BACKENDS: {kernel_opt_backends}
KERNEL_OPT_CLAUDE_MODEL: claude-opus-4-7
KERNEL_OPT_IMAGE: {kernel_opt_image}
KERNEL_OPT_WORKSPACE: core42-sandbox
Use OOB only via `oob_ray_submit.py run -a claude`.
Submit the top 5 valid OOB optimization candidates if available. If fewer than 5 valid candidates exist, explain why in the report.
Must optimize at least {min_kernels} kernels

Requirements:
Save all results and the optimization report to /workspace/hyperloom/
You MUST execute the full skill pipeline (Phase 0-10), including parameter sweep.
Even if the baseline already exceeds the InferenceX target, still run the sweep and write the report.
After writing optimization_report.md, also write /workspace/hyperloom/ci_metrics.json:
```json
{{"baseline_throughput": 8053.90, "optimized_throughput": 8850.12, "gain_pct": 9.88, "tok_per_gpu_baseline": 4026.95, "tok_per_gpu_optimized": 4425.06, "actions_taken": ["params_max_num_seqs_512", "kernel_fused_moe_kept"]}}
```
Rules:
- baseline_throughput / optimized_throughput = total output tok/s (all GPUs combined)
- tok_per_gpu_baseline = baseline_throughput / {tp}
- tok_per_gpu_optimized = optimized_throughput / {tp}
- gain_pct = (optimized - baseline) / baseline * 100. Use 0.0 if no improvement.
- All six field names are MANDATORY.

CRITICAL — Result Directory:
export NFS_ROOT={nfs_root}
export RESULT_DIR=/workspace/hyperloom/

Use this MCP to create RayJob:
```json
{{
  "mcpServers": {{
    "primus-safe-core42": {{
      "url": "{safe_base_url}/api/v1/safe-mcp/mcp",
      "headers": {{
        "Authorization": "Bearer {safe_api_key}"
      }}
    }}
  }}
}}
```

If any pipeline step is not finished correctly, DO IT AGAIN until it returns correct results.
DO NOT create PyTorchJob. Create RayJob only.
Do NOT use existing RayJobs. Create a new RayJob.

NEVER execute find / or find on network mount points (e.g., /wekafs, /mnt/weka) without depth limits.
If you must use find, you MUST include -maxdepth 4 (e.g., find / -maxdepth 4 -name "...").
Prioritize using which, ls, or checking $PATH / $PYTHONPATH over searching the filesystem.

Avoid Long Timeouts: NEVER set a single bash timeout to 30 minutes. Use short timeouts (60-120 seconds) per execution.
Stateful Polling: Perform a single check or short loop (2-3 iterations with 30s intervals), then exit and report status.
External Iteration: If the desired state (e.g., Running) is not reached, report status and initiate a new short-lived bash task.
Fail Fast: Do not stay inside a single blocking bash script for long durations.

InferenceX Baseline:
Target GPU: {target_gpu}
Raw performance values:
{inferenceX_data}

Baseline Benchmark Script:
{benchmark_script_section}
