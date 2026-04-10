Use the mlperf-optimization skill to optimize GPT-OSS-20B MLPerf training performance.
mode: local
Configuration:
Benchmark: gpt-oss-20b (MLPerf Training 5.1.0)
Quality target: validation log perplexity = 3.34
Precision: BF16/FP8 hybrid
MLPERF_DIR: /root/Hyperloom-plus-mlperf/training_optimization/mlperf
Config script: config_MI355X_1x8x1_fp8.sh
Training YAML: conf/gpt_oss_20B-pretrain-fp8.yaml
Framework: primus (Megatron-based)
GPU type: MI355X
GPUs per node: 8
Nodes: 1
Training params:
GBS=32, MBS=2, LR=4.0e-4, seq_length=8192
TP=1, PP=1, EP=1

Requirements:
Save all results and the optimization report to /root/mlperf_results/
Execute the full skill pipeline, including parameter sweep.
Use run_mlperf_trial for ALL training runs — never call torchrun or run_and_time.sh directly.
Follow all review checkpoints.