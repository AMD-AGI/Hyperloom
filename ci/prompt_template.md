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
Execute the full skill pipeline (Phase 0-10), including parameter sweep.
After writing optimization_report.md, also write {result_dir}/ci_metrics.json with EXACTLY this schema:
{{"baseline_throughput": <total output tok/s>, "optimized_throughput": <total output tok/s>, "gain_pct": <float>, "tok_per_gpu_baseline": <float>, "tok_per_gpu_optimized": <float>, "actions_taken": ["action1", "action2"]}}

InferenceX Baseline:
Target GPU: {target_gpu}
Raw performance values:
{inferenceX_data}

Optimize and push ahead of {target_gpu}. Use InferenceX data from Hyperloom as starting point for sglang {runner} baseline.

IMPORTANT benchmark parameters (must match InferenceX):
- Use --random-range-ratio 0.8 for ALL benchmarks (baseline, DFS, sweep). Do NOT use 1.0.
- Use --num-prompts equal to CONC * 10 (e.g. CONC=512 → 5120 prompts). Do NOT use CONC * 3.

CRITICAL: For the baseline, you MUST read the InferenceX benchmark script listed above and replicate its server configuration exactly. The script contains platform-specific params (attention backends, env vars, memory settings, expert parallelism flags) essential for accurate baselines. Do NOT guess server params — extract them from the script. Do NOT run the script directly (it has external dependencies like benchmark_lib.sh). Instead: cat the script, identify the server launch command and export env vars, then reproduce them in your own launch.
