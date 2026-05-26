You are running on a SaFE CI sandbox to validate **PR-head commit `{git_ref}`** for `{model_hf}` on AMD MI300X.

The deployed inference_optimizer at `/wekafs/HyperloomV2` is the **production / scheduled-CI** baseline; for THIS run we want the *unmerged PR head* — so do NOT cd into `/wekafs/HyperloomV2`. Clone PR head into the sandbox first, install from there, then drive the inference_optimizer skill from the cloned repo.

## Step 0 — Prepare PR-head workspace (do this FIRST)

```bash
cd /tmp
git clone --filter=tree:0 --no-checkout \
    https://x-access-token:{gh_token}@github.com/AMD-AGI/Hyperloom.git Hyperloom-pr
cd Hyperloom-pr
git fetch --depth=50 origin {git_ref}
git checkout {git_ref}
export REPO_ROOT="$(pwd)"
echo "[pr-ci] REPO_ROOT=$REPO_ROOT"
echo "[pr-ci] HEAD=$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "{git_ref}" || {{ echo "[pr-ci] ERROR: HEAD != {git_ref}"; exit 2; }}
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
```

If step 0 fails (network, token, install.sh broken on PR head, …) **stop and exit non-zero** — do NOT silently fall back to `/wekafs/HyperloomV2`, the comparison would be meaningless.

## Step 1 — Run inference_optimizer skill from the cloned PR-head repo

Use the skill at `$REPO_ROOT/inference_optimizer/SKILL.md` (NOT the wekafs copy). Read SKILL.md, then follow Step 2 ("Launch a New Optimization") with these CLI flags:

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

ci_metrics.json contract — written automatically by `inference_optimizer/manifest.py` to `/workspace/hyperloom/ci_metrics.json`:
  baseline_throughput    total output tok/s across all GPUs
  optimized_throughput   same, after sweep
  gain_pct               (optimized - baseline)/baseline * 100, 0.0 if no gain
  tok_per_gpu_baseline   baseline_throughput / {tp}
  tok_per_gpu_optimized  optimized_throughput / {tp}
  actions_taken          array of "phaseNN_<name>: ..." strings

If manifest.py fails to emit a complete file, fall back: read `/workspace/hyperloom/state.json` (`baseline_tput`, `current_best`, `cumulative_gain`) and write ci_metrics.json yourself before exiting. All six field names are MANDATORY — the CI shows N/A if missing or renamed.

InferenceX floor (target to beat — already published on Hyperloom):
{inferenceX_data}

Same {gpu_type}/{framework}/{precision}/TP={tp}/CONC={conc} workload. Your baseline should land within 5% of these numbers. Much lower likely = cold aiter JIT (SKILL.md "Cold-start Discipline" auto-bumps timeout to 3600s).

Auth: SAFE_API_KEY and SAFE_API_BASE are already exported into the sandbox env
by kernel-agent.env.sh — read them via `os.environ.get(...)` or `$SAFE_API_KEY`
from bash. Do NOT echo the key into logs, chat messages, manifests, or any
file that gets uploaded to artifacts.
