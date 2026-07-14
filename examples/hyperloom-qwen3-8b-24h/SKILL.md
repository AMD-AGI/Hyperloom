---
name: hyperloom-qwen3-8b-24h
description: Run a long-horizon Hyperloom Qwen3-8B optimization session. Use when the user wants the cyclic macro-cycle behavior for a roughly 24-hour demo.
---

# Hyperloom Qwen3-8B Long-Horizon Run

Read and follow `@../../inference_optimizer/SKILL.md` first. This skill provides the concrete workload and launch constraints for a long-horizon Qwen3-8B demo.

## Long-Horizon Gate

Current Hyperloom treats `--max-hours 24` as a long-horizon run. Long-horizon cyclic macro-cycles require one of:

1. `--max-hours >= 24`
2. an unbounded run (`max_minutes == 0`)

`INFERENCE_OPTIMIZER_CYCLIC_PHASES` is enabled by default, but a falsy value (`0`, `false`, `no`, or `off`) disables cyclic macro-cycles. For this skill, ensure it is unset or truthy.

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-8B from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
- `TP=1`
- `CONC=64`
- `ISL=1024`
- `OSL=1024`
- `PRECISION=fp8`
- `INFERENCE_OPTIMIZER_CYCLIC_PHASES=1`
- `--target-gain 30`
- `--max-hours 24`

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
3. Report the session ID, log path, PID, and initial health check result.
4. Monitor the process every 300 seconds until work is done.
5. To recover an unexpected crash, only run `optimize --resume` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
6. If `stop_reason` in the current session `state.json` is final, stop and exit.
