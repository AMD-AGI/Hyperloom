---
name: hyperloom-qwen3-8b-3h
description: Run a 3-hour Hyperloom Qwen3-8B optimization session without the Kernel Agent. Use when the user wants a short, no-kernel Hyperloom demo on the local AMD ROCm environment.
---

# Hyperloom Qwen3-8B 3h No-Kernel Run

Read and follow `@../../inference_optimizer/SKILL.md` first. This skill provides the concrete workload and launch constraints for a short Qwen3-8B demo.

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-8B from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
- `TP=1`
- `CONC=64`
- `ISL=1024`
- `OSL=1024`
- `PRECISION=fp8`
- `--target-gain 30`
- `--max-hours 3`
- `--no-kernel`

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

If `MODEL_PATH` is unset, do not assume the Hugging Face CLI exists. Download the model with Python:

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

## Launch Requirements

1. Install packages and save artifacts to a writable folder.
2. Run in background with `setsid nohup`.
3. Include `--no-kernel` in the `inference_optimizer optimize` command so the Kernel Agent phase is skipped.
4. Report the session ID, log path, PID, and initial health check result.
5. Monitor the process every 300 seconds until work is done.
6. To recover an unexpected crash, only run `optimize --resume` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
7. If `stop_reason` in the current session `state.json` is final, stop and exit.
