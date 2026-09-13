---
name: hyperloom-qwen3-14b-fp8-12h-forge
description: Run a 12-hour Hyperloom Qwen3-14B-FP8 optimization session with the per-kernel KernelForge backend instead of GEAK. Use when the user wants the medium-length Hyperloom demo and has asked for the forge kernel backend.
---

# Hyperloom Qwen3-14B-FP8 12h Run (Forge Kernel Backend)

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@hyperloom/inference_optimizer/SKILL.md` (wheel install) or `@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill provides the concrete workload and launch constraints for a 12-hour Qwen3-14B-FP8 demo.

This is the [`hyperloom-qwen3-14b-fp8-12h`](../hyperloom-qwen3-14b-fp8-12h/SKILL.md)
demo with **one** difference: the KERNEL_AGENT phase runs the per-kernel
KernelForge backend instead of GEAK. The workload, budget, and phase split are
identical on purpose, so the two runs stay directly comparable.

## Kernel Backend

Set `KERNEL_OPT_BACKEND_ORDER=forge` in the environment that launches
`optimize`. This is the only switch: the opt-in is an **exact** match on
`forge`, and every other value (including unset) leaves GEAK owning the whole
kernel phase.

```bash
export KERNEL_OPT_BACKEND_ORDER=forge
```

Nothing else has to be installed or configured for this:

- KernelForge is vendored into Hyperloom. There is no repository to clone and
  no `FORGE_PATH` to point anywhere.
- The runtime installer already installs the `claude` CLI that the forge
  backend drives, unconditionally — the backend is chosen per session, later.
- Forge reuses the LLM credentials setup already wrote. It reads
  `CLAUDE_MODEL` / `CODEX_MODEL`, the same pair every other Hyperloom
  component reads, so no separate key or model id is needed.
- `rocprof-compute` profiling deps are installed unconditionally too.

Do **not** set the other `FORGE_*` variables. They are internal tuning knobs
with working defaults; overriding them is not part of this demo.

Write the value into `.env` as well when the user wants it to persist across
runs, so a `--resume-from` relaunch keeps the same backend:

```bash
KERNEL_OPT_BACKEND_ORDER=forge
```

Before launch, confirm the value is actually set in the launching shell and
report it. Launching without it silently runs GEAK and produces a run that
looks like this demo but is not.

## Run Mode

Resolve the run mode before launching Hyperloom:

1. If `HYPERLOOM_RUN_MODE=baremetal` or it is unset, run this demo directly on the host.
2. If `HYPERLOOM_RUN_MODE=docker`, this skill owns the Docker setup. Ask the user whether they want a `vllm` or `sglang` Docker image unless `HYPERLOOM_IMAGE` is already set. Use a ROCm image that already contains the selected framework; do not install the framework inside Docker.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Read `HYPERLOOM_DOCKER_TARGET_HOST` from `.env` when present. If it names a
  host different from `$(hostname)`, first SSH to that host and continue this
  Docker setup there; do not start Docker on the login/current host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container (ROCm/framework comes from
  the image). Do **not** use `--skip-base-check` — let Phase 1 preflight validate
  the container environment.
- Do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host.
- `KERNEL_OPT_BACKEND_ORDER=forge` must be set **inside the container**, in the
  same `docker exec` that launches `optimize`. Exporting it only on the host
  does not reach the optimizer.

### Prior workload cleanup (required)

Before any replacement launch after a failed or abandoned demo run (`docker run`,
`install.sh`, or a new/fresh `optimize`), follow **IR-1 — Prior workload cleanup
gate** in `@${HYPERLOOM_SKILL_PATH}`. Run all probes on the **docker host**; never
skip the user-approval step (#1314).

Suggested Docker images:

- `vllm`: `docker.io/vllm/vllm-openai-rocm:v0.27.1`
- `sglang` MI300X: `docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x-20260825`
- `sglang` MI355X: `docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260825`

In Docker mode, start a long-running container on `HYPERLOOM_DOCKER_TARGET_HOST`
(or the current host when it is unset) before running setup or optimize:

```bash
export REPO_ROOT="$(pwd -P)"
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --entrypoint tail \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$REPO_ROOT:$REPO_ROOT" \
  "$HYPERLOOM_IMAGE" \
  -f /dev/null
```

Mount the Hyperloom workspace at the same absolute path (`-v "$REPO_ROOT:$REPO_ROOT"`) so paths in `.env`, logs, and session artifacts stay valid. If `USER_DATA_PATH` or a pre-downloaded model directory is outside the workspace, add matching `-v host_path:host_path` mounts before starting the container.

Then run the setup backend inside the container:

```bash
docker exec -w "$REPO_ROOT" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
  'REPO_ROOT="$(pwd -P)"; PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
```

After that, run all remaining commands for this demo inside the same container with `docker exec -w "$REPO_ROOT" ...`; do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host in Docker mode. When the demo is finished, ask the user whether to stop the container. If they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-14B-FP8 from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
- `KERNEL_OPT_BACKEND_ORDER=forge` (required by this demo; see [Kernel Backend](#kernel-backend))

Required optimize CLI flags:

- `--tp 1`
- `--conc 64`
- `--isl 1024`
- `--osl 1024`
- `--precision fp8`
- `--target-gain 50`
- `--max-hours 12`
- `--max-minutes-framework-pct 0.43`
- `--max-minutes-kernel-pct 0.42`

There is no CLI flag for the kernel backend — it is selected by the environment
variable only. Do not invent one.

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

Before resolving or downloading any model, always ask the user which model path to use. Present the currently resolved option when `MODEL_PATH` is already set, and always offer a custom local path plus the demo default. Do not continue until the user chooses one.

Use this decision flow:

- If the user chooses the existing `MODEL_PATH`, inspect that path and use it only when it contains `config.json`; otherwise ask again for a valid path or the demo default.
- If the user provides a custom local path, export `MODEL_PATH` to that path and require `config.json` before launch.
- If the user chooses the demo default, set `MODEL_PATH=${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8` and download `Qwen/Qwen3-14B-FP8` there when `config.json` is not already present.

Do not assume the Hugging Face CLI exists; resolve or download the selected model with Python:

```bash
python -m pip install -U huggingface_hub
export REPO_ROOT="$(pwd -P)"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8}"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
if (target / "config.json").is_file():
    print(f"Using existing model at {target.resolve()}")
else:
    snapshot_download(
        repo_id="Qwen/Qwen3-14B-FP8",
        local_dir=str(target),
    )
print(target.resolve())
PY
```

## Pre-launch Runtime Install

Before the first `optimize` launch, run the full runtime installer in the same
environment that will launch the optimizer. Preflight loads `kernel-agent.env.sh`
before it can reach the later Ray/Magpie/InferenceX auto-install checks, so this
step must happen before launching.

For Docker mode, run this inside the container. For bare-metal mode, run it on
the host:

```bash
export REPO_ROOT="$(pwd -P)"
# .env fills gaps only: re-exporting the non-empty pre-source snapshot keeps every
# value the caller exported. Wider than install.sh, which guards a fixed list.
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"
unset _dotenv_prev
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
bash "$INSTALL_SH"
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

If `hyperloom/inference_optimizer/assets/install.sh` is not present (source
checkout layout), use `src/hyperloom/inference_optimizer/assets/install.sh`.

Sourcing `.env` in the block above sets `KERNEL_OPT_BACKEND_ORDER` to whatever
the file carries, and `eval "$_dotenv_prev"` then replays the caller's
pre-existing exports on top of it. Either value can win, so export `forge`
**after** this block, and verify it right before launching:

```bash
export KERNEL_OPT_BACKEND_ORDER=forge
echo "kernel backend: ${KERNEL_OPT_BACKEND_ORDER}"
```

## User-visible Progress

Keep the user informed with concise status updates throughout the demo. Do not
dump full debug logs into chat; report the important values and paths so the user
can tell that work is progressing.

Before launch, report the launch plan:

- model path and whether it is an existing local model or a downloaded default;
- run mode (`baremetal` or `docker`) and target host/container when applicable;
- framework, TP, concurrency, ISL, OSL, precision, max hours, and required demo
  flags;
- the resolved kernel backend (`KERNEL_OPT_BACKEND_ORDER`);
- `USER_DATA_PATH` and where runtime artifacts will be written.

After the runtime install, report whether it succeeded and the path to
`kernel-agent.env.sh`. After starting the optimizer, report:

- optimizer PID;
- run log path;
- launch-info JSON path;
- resolved session directory;
- `state.json` path;
- initial health check result.

Confirm the backend actually took effect rather than assuming it did. The
optimizer records the resolved choice as `kernel_optimizer` in the session
`state.json`, which is `geak` unless the env var opted in:

```bash
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
```

Check this right after launch and report the value. If it is `geak`, stop and
tell the user the environment variable did not reach the optimizer, instead of
letting a 12-hour run continue as an unlabelled GEAK run.

During monitoring, print a short summary at each 300-second check:

- process alive/stopped;
- phase and `stop_reason`;
- baseline throughput, current best throughput, and cumulative gain when present;
- latest benchmark result or candidate decision when available;
- the most relevant recent log lines, excluding secrets.

When the run finishes, report the final status, final report path, best result,
and the stop reason. Never print API keys, tokens, or custom header values.

## Launch Requirements

1. Run the pre-launch runtime install above and source
   `$USER_DATA_PATH/runtime/kernel-agent.env.sh` before launching.
2. Export `KERNEL_OPT_BACKEND_ORDER=forge` in the launching shell, after
   sourcing `kernel-agent.env.sh`, and confirm the value before launch. In
   docker mode, set it inside the same `docker exec` that runs `optimize`.
3. Keep `PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"` in the launch shell so robustness
   and critic subprocesses can import `hyperloom.agents` after changing cwd.
4. Run in background with `setsid nohup`.
5. Pass all required optimize CLI flags in the `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
6. Include `--max-minutes-framework-pct 0.43` and `--max-minutes-kernel-pct 0.42`
   in the optimize command. Do **not** pass `--no-framework-agent` or `--no-kernel` —
   this demo runs the full OPTIMIZE phase (FRAMEWORK_AGENT + KERNEL_AGENT), and
   `--no-kernel` would skip the very phase this demo exists to exercise.
7. Report the session ID, log path, PID, and initial health check result.
8. Monitor the process every 300 seconds until work is done.
9. To recover an unexpected crash, only run `optimize --resume-from "$SESSION_DIR"` against the same session dir, with `KERNEL_OPT_BACKEND_ORDER=forge` still set. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
10. If `stop_reason` in the current session `state.json` is final, stop and exit.
