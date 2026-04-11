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
GEAK image: {geak_image}
GEAK_WORKSPACE: {geak_workspace}
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

IMPORTANT: Use --random-range-ratio 0.8 for ALL benchmarks (baseline, DFS, sweep). This matches InferenceX's official benchmark parameters. Do NOT use 1.0.

CRITICAL: For the baseline benchmark, you MUST use the InferenceX benchmark script listed above (if available). It contains platform-specific server parameters (attention backends, env vars, memory settings, expert parallelism flags) that are essential for accurate baseline numbers. Do NOT construct vLLM/SGLang server launch commands manually — the InferenceX scripts already have the optimal configuration. Read the script, extract the server launch command and env vars, and use them directly.
