Use the mlperf-optimization skill to optimize GPT-OSS-20B MLPerf training performance.
mode: local

Configuration:
Benchmark: gpt-oss-20b
Quality target: eval log perplexity ≤ 3.34
Precision: FP8 hybrid (E4M3 activations/weights, E5M2 gradients)
MLPERF_DIR: /root/Hyperloom-plus-mlperf/training_optimization/mlperf
Baseline Config script: config_MI355X_1x8x1_fp8.sh
Training YAML: conf/gpt_oss_20B-pretrain-fp8.yaml
Framework: primus (Megatron-based)
GPU: 8× MI355X 

Known prior results:
Reference TTT: ~206 min (must re-verify via Tier 4 baseline)

Kernel Optimization:
KERNEL_OPT_BACKENDS: geak,oob-claude,oob-codex
All backends run in parallel per candidate kernel. Best verified speedup wins.
GEAK: oci-geak-agent MCP
  image: harbor.oci-slc.primus-safe.amd.com/sync/tasimage/primus:gpt-oss-20b_training_6.0_2026-04-07-19-47-24_dev
  workspace: control-plane-sandbox, step_limit: 100
OOB-Claude: oob-optimizer-dev MCP (agent="claude")
  max_turns: 30, poll_interval: 15s, poll_timeout: 10min
OOB-Codex: oob-optimizer-dev MCP (agent="codex")
  max_turns: 20, poll_interval: 10s, poll_timeout: 5min
Iterative refinement: 3 rounds per OOB backend
Must optimize at least 5 kernels

Requirements:
Save results to /root/mlperf_results/
Baseline and final run MUST be Tier 4 full convergence — never Tier 1/2/3.
Follow all review checkpoints.
