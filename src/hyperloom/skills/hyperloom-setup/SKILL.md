---
name: hyperloom-setup
description: Configure Hyperloom in the current agent workspace after pip install --target . by collecting LLM settings, choosing direct baremetal or Docker execution, writing .env, and running the setup backend directly only in baremetal mode.
---

# Hyperloom Setup

Use this skill after the user prepares a dedicated workspace, opens that
directory in the agent, and installs Hyperloom into the current directory:

```bash
pip install your_package.whl --target .
```

The current directory is the Hyperloom workspace and install target. It is normal
for this directory to contain many Python package folders; users do not need to
inspect them. Do not use an existing project directory unless the user accepts
that setup may create or update `.env` in that directory.

This skill should normally run once per workspace. It resolves a run mode into
`HYPERLOOM_RUN_MODE` (`baremetal` or `docker`) for this session. In `baremetal`
mode it runs the setup backend on the host; in `docker` mode it writes `.env`
only. It does **not** start any container.
Whether to generate or run a Docker container is decided later by the example
(workload) skill based on `HYPERLOOM_RUN_MODE`.

- `baremetal`: run directly in the current development environment, even if that
  environment is itself a container. It provides ROCm and ROCm torch; setup can
  reuse preinstalled SGLang, vLLM, or ATOM, or optionally install SGLang/vLLM.
  Do not start another Docker container or require a physical host OS.
- `docker`: writes `.env` and records the run mode; the example (workload) skill
  starts the container and runs setup inside it.

## Run Mode Resolution

Resolve the run mode in this order and skip the interactive question when a value
is already present:

1. `HYPERLOOM_RUN_MODE` in the shell environment (`baremetal` or `docker`).
2. Otherwise, ask the user (Step 2 question 5).

`HYPERLOOM_RUN_MODE` is a value resolved for this session from the shell
environment. Keep it for the current run; the example (workload) skill uses it to
decide whether to generate a container (and which image to use). This skill does
not start a container itself.

## Workflow

You must run this as an interactive onboarding flow. Do not stop after listing
required values. Ask the user each question, collect the answer, warn before
writing `.env`, write `.env`, read it back for validation, and continue to the
setup command. If setup already completed for this workspace and the user is not
changing provider, model, `USER_DATA_PATH`, run mode, Docker target host, or
bare-metal framework setup choice, do not run setup again; continue with the
demo skill using the existing `.env`.

## Step 1: Confirm Workspace

Confirm the current directory is the dedicated Hyperloom workspace selected by
the user and contains a `hyperloom/` Python package directory from `pip install
--target .`.

If `hyperloom/` is missing, do not search for or switch to another target
directory. Tell the user to open the intended dedicated workspace in the agent
and install Hyperloom into that current directory:

```bash
pip install hyperloom-inference-optimizer==1.1.2 --target .
```

Then stop and ask the user to rerun `/hyperloom-setup` from that workspace.

## Step 2: Ask Configuration Questions

Use the agent's structured question UI (option cards) for every question. It
requires at least two options per question, so for a free-form value (base URL,
custom model id) present two options — `Use default (<value>)` and `Custom` —
and only when the user picks `Custom` ask a plain-text follow-up for the exact
value.

1. Ask the Anthropic base URL with the structured UI.
   - Present exactly these three option labels in this order:
     - `Use default (https://api.anthropic.com)` — this remains the recommended
       default.
     - `Use AMD gateway (https://llm-api.amd.com/anthropic)`.
     - `Custom`.
   - If the user picks `Custom`, ask a plain-text follow-up for the URL.

2. Explain that secrets must be edited in `.env`, not pasted into chat.
   - Never ask the user to paste API keys into the conversation.
   - Create `.env` with placeholders for secret values.
   - Ask the user to edit `.env` directly and replace placeholders.
   - After the user confirms the file is edited, validate only whether secret keys are set; do not print secret values.

3. Collect non-secret values and write secret placeholders:

   Ask the base URL and model questions below with the structured UI using two
   options — `Use default (<value>)` and `Custom` — and only ask a plain-text
   follow-up for the exact value when the user picks `Custom`.

   - Write `ANTHROPIC_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask `ANTHROPIC_BASE_URL` with exactly these option labels in this order:
     `Use default (https://api.anthropic.com)` /
     `Use AMD gateway (https://llm-api.amd.com/anthropic)` / `Custom`.
   - Ask `CLAUDE_MODEL`: options `Use default (claude-opus-5)` / `Custom`.
4. Explain `USER_DATA_PATH`:
   - It is the writable root for Hyperloom runtime files, dependency checkouts, logs, optimizer runs, and generated env files.
   - Offer `<workspace>/session` (the current workspace directory plus a
     `session` subdirectory) as the default/recommended option, using its
     absolute path. Each optimizer run still creates its own UTC-stamped
     subdirectory under it.
   - If an existing `USER_DATA_PATH` is visible in the current shell or terminal context, offer that exact value as one option.
   - Always offer a custom path option.
   - Do not auto-select; write `USER_DATA_PATH` only after the user explicitly chooses (they may accept the default).

5. Ask where to run Hyperloom (sets `HYPERLOOM_RUN_MODE` for this session). Skip
   this question if `HYPERLOOM_RUN_MODE` is already set in the shell environment
   (see [Run Mode Resolution](#run-mode-resolution)); just confirm it and use it
   for this run.

   Present exactly these two option labels in this order:

   1. `docker (Recommended)`
   2. `baremetal`

   When asking, add a one-line reminder that Docker is recommended because it
   ships a validated ROCm + framework stack and keeps the host untouched, while
   baremetal is for advanced users and may cause environment-specific issues or
   modify the host environment.

   - `docker (Recommended)`: record `docker` as the run mode; the example
     (workload) skill generates a ROCm container later.
   - `baremetal`: run the setup backend directly in the current development
     Python environment, including when the development platform is a container.

6. Only when the user chose `docker`, resolve the Docker target host
   (`HYPERLOOM_DOCKER_TARGET_HOST`). This is where the example skill will run
   `docker run` / `docker exec` later.

   First inspect the current machine and any Slurm allocation:

   ```bash
   echo "current host: $(hostname)"
   if command -v squeue >/dev/null 2>&1; then
     echo "SLURM detected — current allocations:"
     squeue -u "$USER" 2>/dev/null || squeue 2>/dev/null
     echo "allocated nodes: $(squeue -u "$USER" -h -t RUNNING -o '%N' 2>/dev/null | paste -sd, -)"
   else
     echo "no SLURM (squeue not found) — use the current host"
   fi
   ```

   - If `squeue` is not available, set `HYPERLOOM_DOCKER_TARGET_HOST` to the
     current host (`$(hostname)`) without asking another question.
   - If Slurm is detected but there are no allocated nodes, set
     `HYPERLOOM_DOCKER_TARGET_HOST` to the current host and explain that no
     Slurm allocation was available to choose.
   - If Slurm is detected and the user has allocated nodes, ask which host should
     run the Docker container. Offer the current host, each allocated node, and a
     custom host option.
   - If the chosen host is not the current host, tell the user that the demo
     skill will first SSH to that host and run all Docker commands there. Do not
     start or restart any Slurm job from setup.

7. Only when the user chose `baremetal`, ask whether to install a serving
   framework (used as the `--install-framework` value in Step 4). Present exactly
   these three option labels in this order and do not reorder them by
   recommendation:
   1. `none`: use an already-installed vLLM/SGLang/ATOM stack in the selected
      Python environment. This skips framework installation, not setup's
      credential/runtime configuration or eligible ROCm hotfixes.
   2. `sglang`: install SGLang ROCm framework components (shared with the host torch).
   3. `vllm (isolated)`: install vLLM into a dedicated venv, leaving the host
      torch/SGLang stack untouched. On ROCm 7.2.x the ROCm wheel brings its own
      torch; on ROCm 10 vLLM is built from source in a venv that reuses the host
      ROCm torch.
   - Do not mark any option as recommended. Present the three options in the exact
     order above without a default selection.
   - ATOM must already be installed; there is no `--install-framework atom`.
     For an ATOM workload, use `none` and explicitly verify/select it with
     `--frameworks atom --require-frameworks`, especially when multiple serving
     frameworks are installed. Do not change global framework defaults or choose
     whichever other framework happens to import first.

8. Only when the user chose `baremetal` **and** `vllm (isolated)` in Step 7,
   briefly note that on ROCm 7.2.x the installer enforces the vLLM 0.28.0+
   wheel's glibc floor (glibc >= 2.39); the ROCm 10 source build has no such
   gate. If setup later fails with that error, explain it in plain
   language and point the user to Docker mode or a pre-0.28 override — do not
   implement a second version gate here.

## Step 3: Write `.env`

Create or update `.env` in the current directory.

Before writing, explicitly tell the user:

- `.env` will be created or updated in the current workspace.
- If `.env` already exists, unrelated keys are preserved, but Hyperloom setup
  keys selected in this run will be updated.
- A dedicated workspace is recommended to avoid modifying an existing project's
  `.env`.

- For every value the user chose in this run (base URL, model, run
  mode, `USER_DATA_PATH`, Docker target host), write exactly what the user
  selected. This wins over any pre-existing value in `.env` or the shell
  environment — e.g. if the user picked the Anthropic official URL, write
  `ANTHROPIC_BASE_URL=https://api.anthropic.com`
  even when a different `ANTHROPIC_BASE_URL` already exists.
- Preserve existing keys unrelated to this setup.
- Never print secret values back to the user.
- Do not overwrite an existing non-placeholder secret key.

Write the Anthropic keys plus the common keys:

- `Anthropic`: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_MODEL`.

Common keys:

- `USER_DATA_PATH`
- `HYPERLOOM_RUN_MODE` (`baremetal` or `docker`, the resolved run mode for this session)
- `HYPERLOOM_DOCKER_TARGET_HOST` (only when `HYPERLOOM_RUN_MODE=docker`; the host
  where the demo skill should run Docker)
- `HYPERLOOM_SKILL_PATH` — absolute path to the optimizer skill
  (`<workspace>/hyperloom/inference_optimizer/SKILL.md` for a wheel install,
  `<workspace>/src/hyperloom/inference_optimizer/SKILL.md` for a source checkout).
  Write it in both modes so the demo skill can resolve it even when `docker`
  mode skips the host setup backend.
- `KERNEL_OPT_BACKEND_ORDER` (write `forge` when explicitly selected in Step 7).
  Preserve an existing explicit value from `.env` or the shell. With no selection,
  do not add a key: the CLI defaults ATOM to `forge` and other frameworks to GEAK.
  Match after trimming whitespace and lowercasing: exact `forge` selects Forge;
  any other non-empty normalized value routes to GEAK. Keep the original value.

### AMD APIM subscription header

If the chosen base URL host is `llm-api.amd.com`, that
gateway requires the API key to also be sent as an
`Ocp-Apim-Subscription-Key` header. Write the custom-headers key as a reference
to the same API key, so the user only fills one secret:

```bash
ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}"
```

The value **must** be wrapped in double quotes: the setup backend and launch
scripts may load `.env` with a shell `source`, so an unquoted value containing a
space and a colon (`Ocp-Apim-Subscription-Key: ...`) is parsed as a command and
fails with exit 127.

Skip this key entirely when the selected Anthropic base URL host is not
`llm-api.amd.com`.

After writing `.env`, tell the user to edit the file directly and replace each `<PLEASE_FILL_IN>` placeholder. Wait for the user to confirm before running setup.

Then read `.env` back and confirm:
- non-secret values are correct;
- secret values are `set` or `missing`;
- no secret key still equals `<PLEASE_FILL_IN>`.

If any required secret is missing or still a placeholder, stop and ask the user to edit `.env` again.

## Step 4: Run Setup Backend

In `baremetal` mode, run the backend on the host. The `--install-framework` value
is the framework the user chose in Step 2 (`none` / `vllm` / `sglang`). In
`docker` mode, skip the backend on the host (see below).

### `baremetal`

Run the backend directly in the current directory on the host, passing the
framework the user chose in Step 2.

For `none` with a preinstalled SGLang/vLLM stack:

```bash
export REPO_ROOT="$(pwd -P)"
PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes
```

For **preinstalled ATOM**, use the environment-loading and **Selected Python**
checks in `hyperloom-qwen3-14b-fp8-12h-atom` before the commands below: real
`import atom` and non-empty `torch.version.hip`, with `PYTHON`, `python3` on
`PATH`, and any activated `VIRTUAL_ENV` pointing at the same existing environment.
Set `INFERENCE_OPTIMIZER_FORCE_PYTHON=1`; `/opt/venv` is not required. Do not
create a fresh venv assuming it inherits another venv's packages. Run `rocm-smi`
before GPU probes and report conflicts without stopping other workloads.
Stop on failed imports rather than bypassing verification.

Run the non-mutating ATOM verification from the selected workspace:

```bash
"$PYTHON" -m hyperloom.inference_optimizer.setup --check-only -- \
  --install-framework none --frameworks atom --require-frameworks \
  --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}"
```

For ATOM, explain any proposed environment changes and obtain approval before
actual setup in that same shell; `--yes` is an installer option, not consent.
`none` is not read-only: setup still configures `.env` and can apply eligible
ROCm hotfixes. Do not repeat completed setup just for the handoff or overwrite
the already selected `USER_DATA_PATH`.

```bash
"$PYTHON" -m hyperloom.inference_optimizer.setup -- \
  --install-framework none --frameworks atom --require-frameworks \
  --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}" --yes
```

Use the `USER_DATA_PATH` explicitly selected in Step 2, or the existing value
when reusing setup. If it differs between shell and `.env`, resolve that conflict
with the user first; do not copy stale ambient settings over a freshly confirmed
setup choice. These checks establish local importability, not validation of every
ATOM/ROCm/Python version combination.

For `vllm` (installs into an isolated venv; `--install-framework vllm` already
defaults to isolated, the flag below is explicit):

```bash
export REPO_ROOT="$(pwd -P)"
PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework vllm --framework-env isolated --yes
```

Downgrade path only when the user explicitly chooses a pre-0.28 vLLM on a host
that failed the glibc check:

```bash
export REPO_ROOT="$(pwd -P)"
export VLLM_VERSION=0.27.1
PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework vllm --framework-env isolated --yes
```

For `sglang`:

```bash
export REPO_ROOT="$(pwd -P)"
PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework sglang --yes
```

`none` reuses a preinstalled framework; it does not remove it. If no serving
framework is importable, report the missing prerequisite. For ATOM, select an
existing compatible ATOM environment or prepare one separately with approval;
do not bypass the check or install a different framework as a substitute.
`BENCHMARK_BASE_URL` and `HYPERLOOM_SKIP_FRAMEWORK_CHECK` remain available for
other existing remote/special workflows, not to bypass checks in this local ATOM example.

### `docker`

Do **not** run `hyperloom.inference_optimizer.setup` on the host.
The example (workload) skill will start the container and run setup inside it.

After writing `.env`, tell the user:
- setup on the host is skipped in docker mode;
- the demo skill will `docker run` + `docker exec` setup inside the ROCm container;
- `FRAMEWORK` being unset after this skill is expected.

This skill does not start a container. `HYPERLOOM_RUN_MODE` is recorded so the
example (workload) skill can decide whether to generate a Docker container when
it runs the optimization.

## Step 5: Confirm Detected Framework

In `baremetal` mode, after setup completes, the backend writes the detected
serving framework to `FRAMEWORK` in `.env` (`sglang`, `vllm`, or `atom`). For a
preinstalled ATOM stack, `--frameworks atom` restricts verification/detection to
ATOM even if SGLang or vLLM is also installed. In `docker` mode, skip this until
the demo skill runs setup inside the container. Read
`.env` back and check it:

- If `FRAMEWORK` is set, report it. Downstream demo skills read this value.
- If `HYPERLOOM_RUN_MODE` is `docker`, an unset `FRAMEWORK` is expected: the
  container has not run yet, so the host has no importable serving framework.
  Tell the user the example (workload) skill will detect or provide the
  framework when it starts the container. Do not offer to re-run setup with
  `--install-framework sglang` or `vllm`.
- If `HYPERLOOM_RUN_MODE` is `baremetal` and `FRAMEWORK` is missing or empty,
  setup did not detect a serving framework (for example, `none` without an
  importable SGLang/vLLM/ATOM stack). Check the selected interpreter and report
  the failure. Offer SGLang/vLLM installation only if that is the user's intended
  framework; ATOM users need a preinstalled compatible ATOM stack. Do not invent
  a `FRAMEWORK` value or silently switch frameworks.

## Step 6: Report Result

Report:
- The `.env` path.
- The run mode (`baremetal` or `docker`).
- The Docker target host when `HYPERLOOM_RUN_MODE=docker`.
- The setup command that was run (or that host setup was skipped in `docker` mode).
- Whether setup completed or failed (in `docker` mode, report that host setup was skipped).
- The detected `FRAMEWORK` value (or that it is unset).
- The last relevant error lines on failure.

Do not print secret values back to the user.

## Step 7: Hand Off to a Demo Skill

When setup completed in `baremetal` mode, or when `.env` is written in `docker`
mode, ask the user whether they want to run a demo optimization now, and if so
which option:

- `3h` — short, no-kernel run. Best for a first end-to-end check.
- `12h` — medium-length Qwen3-14B-FP8 run; use the ATOM variant when ATOM is
  detected or explicitly selected (including a Docker run awaiting detection).
- `custom advanced` — user-selected model, framework, workload, budget, phase
  toggles, and advanced CLI flags.

### Kernel backend follow-up (only for `12h`)

When — and only when — the user picks `12h`, ask one follow-up question with the
structured UI: which kernel optimization backend the KERNEL_AGENT phase should
use. Do not ask this for `3h` (it runs `--no-kernel`, so there is no kernel
phase to route) or for `custom advanced` (that skill collects its own flags).

When `FRAMEWORK=atom`, or the user selected the ATOM demo in Docker mode before
container-side detection, hand off to `hyperloom-qwen3-14b-fp8-12h-atom`.
GEAK's live rewrite-seam resolution is unproven on ATOM, so recommend the CLI's
`forge` default rather than offering a routine backend-choice question. If no
backend was selected, write nothing to `.env`. Preserve an explicit shell or
`.env` value; if it is non-empty after trimming whitespace and lowercasing and
not exactly `forge`, report the original value and ask whether to keep it or
explicitly switch before launch. Never silently unset or delete it.

Present exactly these two option labels in this order:

1. `geak` — the default backend, which owns the whole kernel phase.
2. `forge` — the per-kernel KernelForge backend.

Both options run the identical Qwen3-14B-FP8 workload and budget; only the
kernel backend differs, so the two runs stay directly comparable.

The choice selects which demo skill to load and sets
`KERNEL_OPT_BACKEND_ORDER`:

- `geak` → load `hyperloom-qwen3-14b-fp8-12h`. Leave `KERNEL_OPT_BACKEND_ORDER`
  unset, or write `geak`; after trimming whitespace and lowercasing, anything
  other than exact `forge` means GEAK for this SGLang/vLLM demo.
- `forge` → load `hyperloom-qwen3-14b-fp8-12h-forge` and write
  `KERNEL_OPT_BACKEND_ORDER=forge` to `.env` so a `--resume-from` relaunch keeps
  the same backend.

`KERNEL_OPT_BACKEND_ORDER` is the only switch, and the opt-in is an **exact**
match on `forge` after trimming whitespace and lowercasing. There is no CLI flag
for it; do not invent one. KernelForge is vendored into Hyperloom and reuses the
LLM credentials written above, so do not ask for any other `FORGE_*` value.
Leave CLI dependency preparation to the existing runtime installer for the
selected backend. Report actual installer or runtime failures before proposing
an approved repair; do not infer failure from `command -v claude` or add a new
API probe.

If `.env` was already written before this question, update it with the selected
value rather than re-running the whole setup backend.

If the user wants to run a custom model with a preset workload, keep using one
of the fixed demo presets. Ask the user for a local model path, confirm that the
directory exists and contains `config.json`, then export `MODEL_PATH=<that path>`
before loading the selected demo skill. Explain that the model path is replaced,
but the selected fixed demo still owns the workload preset: tensor parallelism,
concurrency, input/output lengths, precision, target gain, and run budget are
not retuned. If the user wants to choose those workload values explicitly, load
the `custom advanced` demo skill instead.

If the user declines, stop here. If `HYPERLOOM_RUN_MODE` is `baremetal` and
`FRAMEWORK` is unset, do not offer a demo; tell the user to install a serving
framework first (see Step 5).

When the user picks a length, load the matching demo skill and follow it — you
stop acting on this setup skill and run the demo skill's instructions instead:

The demo skills are installed under each agent's discovery dir (`.agents/skills/`,
`.claude/skills/`, `.cursor/skills/`); load the matching one by name:

- `3h` → `hyperloom-qwen3-8b-3h`
- `12h` + `geak` → `hyperloom-qwen3-14b-fp8-12h`
- `12h` + `forge` → `hyperloom-qwen3-14b-fp8-12h-forge`
- `12h` + `FRAMEWORK=atom` → `hyperloom-qwen3-14b-fp8-12h-atom` (no backend
  question; the CLI defaults atom to `forge`)
- `custom advanced` → `hyperloom-custom-advanced`

The demo skill reads the values already in `.env` (LLM keys/base URLs,
`FRAMEWORK`, `USER_DATA_PATH`). Fixed presets do not ask the user to re-enter
workload settings. The custom advanced skill reuses setup values, then asks for
workload and phase choices. It resolves the optimizer skill from `.env`
`HYPERLOOM_SKILL_PATH` (written by Step 3).
