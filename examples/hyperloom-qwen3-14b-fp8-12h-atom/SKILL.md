---
name: hyperloom-qwen3-14b-fp8-12h-atom
description: Run a 12-hour Hyperloom Qwen3-14B-FP8 optimization session on the ATOM serving framework with the per-kernel KernelForge backend. Use when the user wants the medium-length Hyperloom demo on ATOM rather than SGLang or vLLM.
---

# Hyperloom Qwen3-14B-FP8 12h Run (ATOM Framework)

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@hyperloom/inference_optimizer/SKILL.md` (wheel install) or `@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill provides the concrete workload and launch constraints for a 12-hour Qwen3-14B-FP8 demo on ATOM.

This is the [`hyperloom-qwen3-14b-fp8-12h`](../hyperloom-qwen3-14b-fp8-12h/SKILL.md)
demo moved onto a different **serving framework**. The workload, budget, and
phase split are identical on purpose, so a run here stays directly comparable
with the SGLang/vLLM variants of the same demo.

Two things are pinned by this demo rather than inherited from `.env`:

- `FRAMEWORK=atom` — the whole point of this variant. The other 12h demos read
  whatever framework `.env` carries; this one does not.
- `KERNEL_OPT_BACKEND_ORDER` — left unset. On ATOM the CLI defaults it to
  `forge`, because GEAK's seam resolution is unproven on this backend.

## Framework

ATOM (AiTer Optimized Model) is an AMD out-of-tree serving engine. Hyperloom
launches it as `python3 -m atom.entrypoints.openai_server` and talks to it over
the same OpenAI-compatible surface it uses for SGLang and vLLM, so the phase
structure, benchmarking, and reporting are unchanged.

Set the framework in the environment that launches `optimize`, and confirm it
before launch:

```bash
export FRAMEWORK=atom
```

**ATOM cannot be installed by the bare-metal installer.** `install_baremetal.sh`
lists `atom` in its default `--frameworks` verification list, so Phase 1
preflight passes on a host that already has ATOM. But `--install-framework`
accepts only `none`, `sglang`, and `vllm` — there is no code path that installs
ATOM. The ATOM layer must already be present, which in practice means running
this demo from an ATOM container image (see [Run Mode](#run-mode)).

### Server arguments

ATOM needs no special launch flags for this model on MI300/MI355-class hardware.
The following argument set was verified to boot Qwen3-14B-FP8 and serve
completions at `-tp 1`:

```
--model $MODEL_PATH -tp 1 --server-port <port> --max-model-len 8192
```

`--max-model-len 8192` leaves ample headroom over this demo's 1024 + 1024
token budget. Do **not** copy the `--level 0 --block-size 64 --kv_cache_dtype bf16`
flag set from ATOM's published `Qwen3-8B-FP8` recipe: those constraints belong to
a small-VRAM gfx1201 consumer card and are not required here.

If Hyperloom needs the arguments passed explicitly, supply them through the
`--server-args` CLI flag. Do not export `EXTRA_ATOM_ARGS` in the launching shell
and expect it to be read: that variable is the transport, not the knob — the
optimizer writes `--server-args` into it inside each materialized Magpie YAML,
so a value exported by hand is overwritten rather than merged.

## Kernel Backend

On ATOM the optimizer **defaults** `KERNEL_OPT_BACKEND_ORDER` to `forge`, so
this demo needs no action here. Leave the variable unset and the CLI reports the
choice at launch:

```
framework=atom: KERNEL_OPT_BACKEND_ORDER defaulted to 'forge'
  (on atom GEAK must resolve a live rewrite seam; forge needs none)
```

This is not a preference. GEAK's own extraction rules forbid *guessing* a
rewrite seam on a quantized, non-vLLM backend: it must grep the live server for
the actual quant-apply or backend forward and use that verbatim. That path is
sound in principle but unproven on ATOM, and the default phase split gives the
kernel phase half the session — a poor place to find out. Forge works per kernel
and needs no seam discovery at all, so it is the safer default here.

Only a value you set yourself is kept. Setting anything other than `forge`
(the opt-in is an **exact** match) hands the phase back to GEAK, and the CLI
warns that its seam resolution is unproven here. Do not set it for this demo.

Nothing else has to be installed or configured for the forge backend:

- KernelForge is vendored into Hyperloom. There is no repository to clone and
  no `FORGE_PATH` to point anywhere.
- The runtime installer ensures the `claude_agent_sdk` Python package the forge
  backend imports. It does **not** install the `claude` CLI binary that the SDK
  drives. On an image that ships neither Node nor that binary, the SDK call
  hangs until the caller's timeout rather than failing loudly, so check for it
  before launching and install it if missing:

  ```bash
  command -v claude || npm install -g @anthropic-ai/claude-code
  ```

  Measured on `rocm/atom-dev:v0.1.7-rc0`: neither `node` nor `claude` is
  present, and `install.sh` leaves it that way.
- Forge reuses the LLM credentials setup already wrote. It reads
  `CLAUDE_MODEL` / `CODEX_MODEL`, the same pair every other Hyperloom
  component reads, so no separate key or model id is needed.

Do **not** set the other `FORGE_*` variables. They are internal tuning knobs
with working defaults; overriding them is not part of this demo.

Write the framework into `.env` as well when the user wants it to persist across
runs, so a `--resume-from` relaunch stays on ATOM:

```bash
FRAMEWORK=atom
```

Do **not** write `KERNEL_OPT_BACKEND_ORDER` into `.env` for this demo. Leaving
it out is what lets the ATOM default apply; a value written there is treated as
your choice and is kept, including on a `--resume-from` relaunch.

Before launch, confirm `FRAMEWORK` is actually set in the launching shell and
report it.

## Run Mode

Resolve the run mode before launching Hyperloom:

1. If `HYPERLOOM_RUN_MODE=docker` or it is unset, run this demo in Docker. This
   is the normal path for ATOM, because the installer cannot add the ATOM layer
   to a bare-metal host.
2. If `HYPERLOOM_RUN_MODE=baremetal`, only continue when ATOM is already
   importable on the host. Verify it before launch and stop if it is not:

   ```bash
   python3 -c "import atom, os; print(os.path.dirname(atom.__file__))"
   ```

   Some ATOM images ship the engine inside a virtualenv rather than the system
   interpreter, so check the interpreter that will actually launch the server.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Read `HYPERLOOM_DOCKER_TARGET_HOST` from `.env` when present. If it names a
  host different from `$(hostname)`, first SSH to that host and continue this
  Docker setup there; do not start Docker on the login/current host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container. ATOM comes from the
  image and cannot be installed by the script in any case. Do **not** use
  `--skip-base-check` — let Phase 1 preflight validate the container
  environment; `atom` is already in the default `--frameworks` list, so the
  check passes on a correct image.
- Do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host.
- `FRAMEWORK=atom` must be set **inside the container**, in the same
  `docker exec` that launches `optimize`. Exporting it only on the host does not
  reach the optimizer; the kernel-backend default is applied by the CLI itself,
  so it needs no such handling.

### Prior workload cleanup (required)

Before any replacement launch after a failed or abandoned demo run (`docker run`,
`install.sh`, or a new/fresh `optimize`), follow **IR-1 — Prior workload cleanup
gate** in `@${HYPERLOOM_SKILL_PATH}`. Run all probes on the **docker host**; never
skip the user-approval step (#1314).

This matters more on ATOM than on the other frameworks. The engine spawns
multiprocessing workers that do not share the entrypoint in their command line,
so a `pkill -f openai_server` reaches the leader and leaves the workers alive
still holding VRAM — and the next launch then fails on a card that looks full.
Signal the whole process group instead, and confirm the card is actually free
before relaunching:

```bash
rocm-smi --showmemuse
pgrep -af "openai_server|spawn_main"
```

Suggested Docker image:

- ATOM: `docker.io/rocm/atom-dev:v0.1.7-rc0`

`rocm/atom-dev` is a public repository, so this tag pulls anonymously. Prefer
this pinned tag over `latest`: `latest` tracks the newest nightly build and
moves, which makes a run unreproducible.

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
- `FRAMEWORK=atom` (required by this demo; see [Framework](#framework))
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
- `KERNEL_OPT_BACKEND_ORDER=<leave unset; the CLI defaults it to forge on ATOM>` (see [Kernel Backend](#kernel-backend))

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

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`. `FRAMEWORK` is the one value this demo overrides rather than inherits.

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

Sourcing `.env` in the block above sets `FRAMEWORK` and
`KERNEL_OPT_BACKEND_ORDER` to whatever the file carries, and
`eval "$_dotenv_prev"` then replays the caller's pre-existing exports on top of
it. Either value can win, so export both **after** this block, and verify them
right before launching:

```bash
export FRAMEWORK=atom
unset KERNEL_OPT_BACKEND_ORDER
echo "framework: ${FRAMEWORK}  kernel backend: <defaulted to forge by the CLI>"
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

Confirm both the framework and the backend actually took effect rather than
assuming they did. The optimizer records the resolved choices in the session
`state.json`; `kernel_optimizer` is `geak` unless the environment variable
opted in:

```bash
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
grep -o '"framework": *"[^"]*"' "$SESSION_DIR/state.json"
```

Check these right after launch and report the values. If `kernel_optimizer` is
`geak`, or the framework is not `atom`, stop and tell the user the environment
did not reach the optimizer, instead of letting a 12-hour run continue
mislabelled.

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
2. Export `FRAMEWORK=atom` in the launching shell, after sourcing
   `kernel-agent.env.sh`, and confirm it before launch. In docker mode, set it
   inside the same `docker exec` that runs `optimize`. Leave
   `KERNEL_OPT_BACKEND_ORDER` unset so the ATOM default applies.
3. Keep `PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"` in the launch shell so critic
   subprocesses can import `hyperloom.agents` after changing cwd.
4. Run it detached the way the harness understands: if `$CLAW_SESSION_ID` is set and your bash tool takes a `run_in_background` parameter, hand the command to it with `run_in_background=true`; otherwise use `setsid nohup ... &`. See the Launch section of the packaged `hyperloom/inference_optimizer/SKILL.md` for why — a hand-detached run is invisible to Claw and its sandbox is reclaimed about fifteen minutes after the turn ends.
5. Pass all required optimize CLI flags in the `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
6. Include `--max-minutes-framework-pct 0.43` and `--max-minutes-kernel-pct 0.42`
   in the optimize command. Do **not** pass `--no-framework-agent` or `--no-kernel` —
   this demo runs the full OPTIMIZE phase (FRAMEWORK_AGENT + KERNEL_AGENT), and
   `--no-kernel` would skip the very phase this demo exists to exercise.
7. Report the session ID, log path, PID, and initial health check result.
8. Monitor the process every 300 seconds until work is done.
9. Unexpected crashes are not automatically resumed. After explicit operator approval, only run `optimize --resume-from "$SESSION_DIR"` against the same session dir, with `FRAMEWORK=atom` still set. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
10. If an approved relaunch is needed after a crash, clear any surviving ATOM workers
    first (see [Prior workload cleanup](#prior-workload-cleanup-required)); an
    orphaned worker still holding VRAM makes the replacement launch fail on a
    card that looks full.
11. If `stop_reason` in the current session `state.json` is final, stop and exit.
