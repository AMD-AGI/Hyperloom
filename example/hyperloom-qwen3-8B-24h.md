@./src/hyperloom/inference_optimizer
Use the skill at @./src/hyperloom/inference_optimizer/SKILL.md to optimize `Qwen3-8B`.

Environment:
- MODEL_PATH=<optional; if unset, download `Qwen/Qwen3-8B` from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>
- FRAMEWORK=<provided by the existing environment; do not set or override it>
- GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>
- TP=1, CONC=64, ISL=1024, OSL=1024
- PRECISION=fp8
- --target-gain 30
- --max-hours 24

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it (for example LLM API keys/base URLs, FRAMEWORK, HF_TOKEN). Do not copy secret values into the prompt, terminal output, reports, or logs.

If MODEL_PATH is unset, do not assume the Hugging Face CLI exists. Download the model with Python:

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path.cwd() / ".cache" / "hyperloom-models" / "Qwen3-8B"
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
print(target.resolve())
PY
export MODEL_PATH="$(pwd)/.cache/hyperloom-models/Qwen3-8B"
```

- Run in background: setsid nohup

Requirements:
1. Install packages and save artifacts to writable folder.
2. Report the session ID, log path, PID, and initial health check result.
3. Then monitor the process every 300s, until work is done.
4. To recover an unexpected crash, ONLY DO `optimize --resume` (same session dir). Which means, after the first launch, NEVER start a new `optimize` — that spawns a new <UTC_ts> session and is forbidden. If a `stop_reason` in current session state.json is final: stop and exit.
5. Modification of USER_DATA_PATH environment variable is not allowed.
