Use the inference_optimizer skill at @/wekafs/HyperloomV2/inference_optimizer/SKILL.md to optimize {model_hf} inference performance on {gpu_type}.

The skill is the Python `inference_optimizer` package on WekaFS; this sandbox image mounts /wekafs read-only. Read SKILL.md first, then follow Step 1 ("Install") and Step 2 ("Launch a New Optimization"). Do NOT use any marketplace skill download or `download_skill` tool — the runtime IS on the mount.

Run config — pass as CLI flags to `inference_optimizer optimize`:
  --model {model_path}
  --framework {framework}
  --gpu-type {gpu_type_lc}
  --target-gain {target_gain}
  --max-hours {max_hours}
  --tick-interval-sec 30
  --kernel-claude

- Run in background: setsid nohup

Workload envs — export before launch (Coordinator reuses these for baseline → params → sweep):
  TP={tp} EP={ep} CONC={conc} ISL={isl} OSL={osl} RANDOM_RANGE_RATIO={random_range_ratio} PRECISION={precision} NODES={nodes}
  KERNEL_AGENT_BUILD_GEAK_RAG_INDEX={kernel_agent_build_geak_rag_index}

Runtime paths (live on the shared mount; the agent does not need to re-clone):
  OOB_PATH={oob_path}
  InferenceX_PATH={inferencex_path}
  TRACELENS_ROOT={tracelens_root}
{multinode_section}
Session dir: /workspace/hyperloom (SKILL.md default — do NOT override).

ci_metrics.json contract — written automatically by inference_optimizer/manifest.py to /workspace/hyperloom/ci_metrics.json:
  baseline_throughput    total output tok/s across all GPUs
  optimized_throughput   same, after sweep
  gain_pct               (optimized - baseline)/baseline * 100, 0.0 if no gain
  tok_per_gpu_baseline   baseline_throughput / {tp}
  tok_per_gpu_optimized  optimized_throughput / {tp}
  actions_taken          array of "phaseNN_<name>: ..." strings

If manifest.py fails to emit a complete file, fall back: read /workspace/hyperloom/state.json (`baseline_tput`, `current_best`, `cumulative_gain`) and write ci_metrics.json yourself before exiting. All six field names are MANDATORY — the CI shows N/A if missing or renamed.

baseline floor (target reference — already published on Hyperloom):
{inferenceX_data}

Same {gpu_type}/{framework}/{precision}/TP={tp}/CONC={conc} workload. Your baseline should land within 5% of these numbers. Much lower likely = cold aiter JIT (SKILL.md "Cold-start Discipline" auto-bumps timeout to 3600s).

Requirements:
1. Save files to writable folder, session_dir: /workspace/hyperloom is writable.
2. Report the session ID, log path, PID, and initial health check result.
3. Then monitor the process every 300s, until work is done.

Auth: SAFE_API_KEY and SAFE_API_BASE are already exported into the sandbox env
by kernel-agent.env.sh — read them via `os.environ.get(...)` or `$SAFE_API_KEY`
from bash. Do NOT echo the key into logs, chat messages, manifests, or any
file that gets uploaded to artifacts.
