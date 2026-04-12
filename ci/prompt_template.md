Use the inference-optimization skill to optimize {model_hf} inference performance.
mode: {mode}

Configuration:
Model path: {model_path}
Framework: {framework}
Precision: {precision}
Inference params: ISL={isl}, OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO=0.8
TP={tp}, EP={ep}
GPU type: {gpu_type}
InferenceX path: {inferencex_path}

SandboxImage: {sandbox_image}

Baseline Benchmark Script:
{benchmark_script_section}

Kernel Optimization:
KERNEL_OPT_BACKENDS: {kernel_opt_backends}
KERNEL_OPT_IMAGE: {kernel_opt_image}
KERNEL_OPT_WORKSPACE: {kernel_opt_workspace}
GEAK step_limit: {geak_step_limit}
Must optimize at least {min_kernels} kernels

Requirements:
Save all results and the optimization report to {result_dir}
Execute the full skill pipeline, including parameter sweep.
Even if the baseline already exceeds the InferenceX target, you MUST still run the sweep phase and write the report.
After writing optimization_report.md, also write {result_dir}/ci_metrics.json.
Copy-paste this template and fill in the numbers. Do NOT rename fields or nest objects:
```json
{{
  "baseline_throughput": REPLACE_WITH_NUMBER,
  "optimized_throughput": REPLACE_WITH_NUMBER,
  "gain_pct": REPLACE_WITH_NUMBER,
  "tok_per_gpu_baseline": REPLACE_WITH_NUMBER,
  "tok_per_gpu_optimized": REPLACE_WITH_NUMBER,
  "actions_taken": ["action1", "action2"]
}}
```
Rules:
- baseline_throughput / optimized_throughput = total output tok/s (all GPUs combined)
- tok_per_gpu_baseline / tok_per_gpu_optimized = output tok/s divided by TP (={tp})
- gain_pct = (optimized - baseline) / baseline * 100. Use 0.0 if no improvement.
- All six fields above are MANDATORY with these exact names. The CI will show N/A if any are missing.
- Do NOT nest into sub-objects. Do NOT rename fields.

InferenceX Baseline:
Target GPU: {target_gpu}
Raw performance values:
{inferenceX_data}

Optimize and push ahead of {target_gpu}. Use InferenceX data from Hyperloom as starting point for sglang {runner} baseline.

IMPORTANT benchmark parameters (must match InferenceX):
- Use --random-range-ratio 0.8 for ALL benchmarks (baseline, DFS, sweep). Do NOT use 1.0.
- Use --num-prompts equal to CONC * 10 (e.g. CONC=512 → 5120 prompts). Do NOT use CONC * 3.

CRITICAL: For the baseline, you MUST read the InferenceX benchmark script listed above and replicate its server configuration exactly. The script contains platform-specific params (attention backends, env vars, memory settings, expert parallelism flags) essential for accurate baselines. Do NOT guess server params — extract them from the script. Do NOT run the script directly (it has external dependencies like benchmark_lib.sh). Instead: cat the script, identify the server launch command and export env vars, then reproduce them in your own launch.
