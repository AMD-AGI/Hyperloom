Use the inference_optimizer skill at @/wekafs/HyperloomV2/inference_optimizer/SKILL.md to optimize {model_hf} inference performance on {gpu_type}.

The skill is the Python `inference_optimizer` package on WekaFS; this sandbox image mounts /wekafs read-only. Read SKILL.md first, then follow Step 1 ("Install") and Step 2 ("Launch a New Optimization"). Do NOT use any marketplace skill download or `download_skill` tool — the runtime IS on the mount.

Run config — pass as CLI flags to `inference_optimizer optimize`:
  --model {model_path}
  --framework {framework}
  --gpu-type {gpu_type_lc}
  --target-gain 10
  --max-hours 2
  --tick-interval-sec 30
  --kernel-claude

Workload envs — export before launch (Coordinator reuses these for baseline → params → sweep):
  TP={tp} EP={ep} CONC={conc} ISL={isl} OSL={osl} PRECISION={precision}

Session dir: /workspace/hyperloom (SKILL.md default — do NOT override).

ci_metrics.json contract — written automatically by inference_optimizer/manifest.py to /workspace/hyperloom/ci_metrics.json:
  baseline_throughput    total output tok/s across all GPUs
  optimized_throughput   same, after sweep
  gain_pct               (optimized - baseline)/baseline * 100, 0.0 if no gain
  tok_per_gpu_baseline   baseline_throughput / {tp}
  tok_per_gpu_optimized  optimized_throughput / {tp}
  actions_taken          array of "phaseNN_<name>: ..." strings

If manifest.py fails to emit a complete file, fall back: read /workspace/hyperloom/state.json (`baseline_tput`, `current_best`, `cumulative_gain`) and write ci_metrics.json yourself before exiting. All six field names are MANDATORY — the CI shows N/A if missing or renamed.

baseline floor (target reference — already published in the benchmark report):
{inferenceX_data}

Same {gpu_type}/{framework}/{precision}/TP={tp}/CONC={conc} workload. Your baseline should land within 5% of these numbers. Much lower likely = cold aiter JIT (SKILL.md "Cold-start Discipline" auto-bumps timeout to 3600s).

Auth (already re-exported by kernel-agent.env.sh; for reference only):
  SAFE_API_KEY={safe_api_key}
  SAFE_API_BASE={safe_base_url}
