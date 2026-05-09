Use the inference-optimization skill in REMOTE mode to optimize {model_hf} inference performance on SaFE cluster.
mode: remote

## Skill Reference

Read and follow ALL instructions from the skill directory on /wekafs:
  /wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization/SKILL.md
  /wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization/modes/REMOTE.md
  /wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization/addendum.md

The REMOTE.md file is the authoritative reference for remote mode execution. Follow it precisely.

## MCP Server

Connect to the SaFE MCP server before submitting any workloads:
  Server name: primus-safe-core42
  URL: {safe_base_url}/api/v1/safe-mcp/mcp
  API key: {safe_api_key}

## Model Configuration

Model path: {model_path}
Framework: {framework}
Precision: {precision}
Inference params: ISL={isl}, OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO=0.8
TP={tp}, EP={ep}
GPU type: {gpu_type}
Target GPU: {target_gpu}

## RayJob Configuration

Submit a RayJob via SaFE MCP with the following spec:

Image: {rayjob_image}
Head node: CPU=96, GPU=0, memory=256Gi, ephemeralStorage=200Gi
Worker nodes: {rayjob_workers} × (CPU=96, GPU=8, memory=1024Gi, ephemeralStorage=400Gi)
RAY_JOB_ENTRYPOINT: base64("tail -f /dev/null")  — idle entrypoint, DO NOT run model server here

Environment variables to inject into the RayJob:
  NFS_ROOT={nfs_root}
  RESULT_DIR={result_dir}
  MODEL={model_path}
  TP={tp}
  CONC={conc}
  ISL={isl}
  OSL={osl}

## Serving Mode: MoRI 1P1D (Multi-node Disaggregated Prefill-Decode)

{serving_instructions}

## Baseline Benchmark Script

{benchmark_script_section}

## Kernel Optimization

KERNEL_OPT_BACKENDS: {kernel_opt_backends}
KERNEL_OPT_IMAGE: {kernel_opt_image}
KERNEL_OPT_WORKSPACE: {kernel_opt_workspace}
GEAK step_limit: {geak_step_limit}
Must optimize at least {min_kernels} kernels

## Results

Save all results and the optimization report to {result_dir}
Execute the full skill pipeline: baseline → parameter sweep → kernel optimization → report.
Even if the baseline already exceeds the InferenceX target, you MUST still run the sweep and write the report.
After writing optimization_report.md, also write {result_dir}/ci_metrics.json with this exact structure:
```json
{{"baseline_throughput": 8053.90, "optimized_throughput": 8850.12, "gain_pct": 9.88, "tok_per_gpu_baseline": 4026.95, "tok_per_gpu_optimized": 4425.06, "actions_taken": ["params_max_num_seqs_512", "kernel_fused_moe_kept"]}}
```
Rules:
- baseline_throughput / optimized_throughput = total output tok/s (all GPUs combined)
- tok_per_gpu_baseline = baseline_throughput / {tp}. tok_per_gpu_optimized = optimized_throughput / {tp}
- gain_pct = (optimized - baseline) / baseline * 100. Use 0.0 if no improvement.
- All six field names are MANDATORY.

## Remote Mode Critical Rules (from addendum.md)

ADDENDUM-01: Never run blocking bash commands that wait for a long-running process. All model server launches and benchmark runs must be submitted as RayJob tasks and polled via Ray Dashboard REST API or SaFE MCP.

ADDENDUM-02: Never use Ray client (ray.init / ray.get) from inside the sandbox. The sandbox has no GPU and no direct Ray cluster access. Use Ray Dashboard REST API or exec_on_gpu via SaFE MCP.

ADDENDUM-03: The RayJob entrypoint MUST be idle (tail -f /dev/null or equivalent). Do NOT launch the model server in the entrypoint. Launch it via exec_on_gpu after the RayJob is running.

ADDENDUM-04: For multi-node TP, pin each process to a distinct node using Ray placement groups or node labels. Verify that worker processes are spread across the correct number of nodes before benchmarking.

ADDENDUM-05: SSE parsing — when reading Ray Dashboard or SaFE MCP streaming responses, handle partial chunks and reconnect on disconnect. Do not assume a single read returns the full response.

ADDENDUM-06: /wekafs is read-only from the sandbox. Write all results to {result_dir} (which must be on a writable path). The RayJob workers can write to /wekafs directly if needed.

## InferenceX Baseline

Target GPU: {target_gpu}
Raw performance values:
{inferenceX_data}

Optimize and push ahead of {target_gpu}. Use InferenceX data as the starting point for the baseline.

IMPORTANT benchmark parameters (must match InferenceX):
- Use --random-range-ratio 0.8 for ALL benchmarks (baseline, DFS, sweep). Do NOT use 1.0.
- Use --num-prompts equal to CONC * 10 (e.g. CONC=512 → 5120 prompts). Do NOT use CONC * 3.

CRITICAL: Read the InferenceX benchmark script listed above and replicate its server configuration exactly (attention backends, env vars, memory settings, EP flags). Do NOT guess server params — extract them from the script. Do NOT run the script directly. Instead: cat the script, identify the server launch command and env vars, then reproduce them via exec_on_gpu on the RayJob worker.

Do NOT call exit_plan_mode or enter_plan_mode — these tools don't exist. Execute directly.
