---
name: hyperloom-setup
description: Configure Hyperloom after pip install --target by collecting LLM settings, choosing a bare-metal or Docker run mode, writing .env, and running the setup backend on baremetal hosts only.
---

# Hyperloom Setup

Use this skill after the user installs Hyperloom into the current workspace:

```bash
pip install your_package.whl --target .
```

The current directory should be the Hyperloom target directory, for example `~/hyperloom`. It is normal for this directory to contain many Python package folders; users do not need to inspect them.

This skill resolves a run mode into `HYPERLOOM_RUN_MODE` (`baremetal` or `docker`)
for this session. In `baremetal` mode it runs the setup backend on the host; in
`docker` mode it writes `.env` only. It does **not** start any container.
Whether to generate or run a Docker container is decided later by the example
(workload) skill based on `HYPERLOOM_RUN_MODE`.

- `baremetal`: the host provides ROCm, and setup can optionally install the SGLang or vLLM framework layer.
- `docker`: writes `.env` and records the run mode; the example (workload) skill
  starts the container and runs setup inside it.

## Run Mode Resolution

Resolve the run mode in this order and skip the interactive question when a value
is already present:

1. `HYPERLOOM_RUN_MODE` in the shell environment (`baremetal` or `docker`).
2. Otherwise, ask the user (Step 2 question 6).

`HYPERLOOM_RUN_MODE` is a value resolved for this session from the shell
environment. Keep it for the current run; the example (workload) skill uses it to
decide whether to generate a container (and which image to use). This skill does
not start a container itself.

## Workflow

You must run this as an interactive onboarding flow. Do not stop after listing required values. Ask the user each question, collect the answer, write `.env`, read it back for validation, and continue to the setup command.

## Step 1: Confirm Workspace

Confirm the current directory contains a `hyperloom/` Python package directory. If it does not, ask the user for the target directory and switch there.

## Step 2: Ask Configuration Questions

Use the agent's structured question UI (option cards) for every question. It
requires at least two options per question, so for a free-form value (base URL,
custom model id) present two options — `Use default (<value>)` and `Custom` —
and only when the user picks `Custom` ask a plain-text follow-up for the exact
value.

1. Choose one LLM mode (two options):
   - `Anthropic`
   - `DeepSeek`

   Present exactly those two option labels. Do not add parenthetical
   descriptions, vendor examples, or base URLs to this first question.

2. Ask the base URL as a structured follow-up after the mode is chosen.
   - For `Anthropic`, present exactly these three option labels in this order:
     - `Use default (https://api.anthropic.com)` — this remains the recommended
       default.
     - `Use AMD gateway (https://llm-api.amd.com/anthropic)`.
     - `Custom`.
   - For `DeepSeek`, present exactly these two option labels in this order:
     - `Use default (https://api.deepseek.com/anthropic)`.
     - `Custom`.
   - If the user picks `Custom`, ask a plain-text follow-up for the URL.

3. Explain that secrets must be edited in `.env`, not pasted into chat.
   - Never ask the user to paste API keys into the conversation.
   - Create `.env` with placeholders for secret values.
   - Ask the user to edit `.env` directly and replace placeholders.
   - After the user confirms the file is edited, validate only whether secret keys are set; do not print secret values.

4. Collect provider-specific non-secret values and write secret placeholders:

   Ask the base URL and model questions below with the structured UI using two
   options — `Use default (<value>)` and `Custom` — and only ask a plain-text
   follow-up for the exact value when the user picks `Custom`.

   For `Anthropic`:
   - Write `ANTHROPIC_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask `ANTHROPIC_BASE_URL` with exactly these option labels in this order:
     `Use default (https://api.anthropic.com)` /
     `Use AMD gateway (https://llm-api.amd.com/anthropic)` / `Custom`.
   - Ask `CLAUDE_MODEL`: options `Use default (claude-opus-4-8)` / `Custom`.

   For `DeepSeek`:
   - Write `DEEPSEEK_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask `DEEPSEEK_BASE_URL` with exactly these option labels in this order:
     `Use default (https://api.deepseek.com/anthropic)` / `Custom`.
   - Ask `DEEPSEEK_MODEL`: options `Use default (deepseek-chat)` / `Custom`.
5. Explain `USER_DATA_PATH`:
   - It is the writable root for Hyperloom runtime files, dependency checkouts, logs, optimizer runs, and generated env files.
   - Offer `<workspace>/session` (the current workspace directory plus a
     `session` subdirectory) as the default/recommended option, using its
     absolute path. Each optimizer run still creates its own UTC-stamped
     subdirectory under it.
   - If an existing `USER_DATA_PATH` is visible in the current shell or terminal context, offer that exact value as one option.
   - Always offer a custom path option.
   - Do not auto-select; write `USER_DATA_PATH` only after the user explicitly chooses (they may accept the default).

6. Ask where to run Hyperloom (sets `HYPERLOOM_RUN_MODE` for this session). Skip
   this question if `HYPERLOOM_RUN_MODE` is already set in the shell environment
   (see [Run Mode Resolution](#run-mode-resolution)); just confirm it and use it
   for this run.

   Present exactly these two option labels in this order:

   1. `docker`
   2. `baremetal`

   - `docker`: record docker as the run mode; the example (workload) skill will
     generate a ROCm container later. Choose this when the host does not have the
     framework installed but Docker with GPU access is available.
   - `baremetal`: run the setup backend directly on this host. Choose this when
     the host provides ROCm, or when Hyperloom is already installed inside a
     Docker container and setup should run directly in that current container
     environment.
   - If the user is unsure and is already inside a framework image or shell with
     a working framework, recommend `baremetal`; otherwise recommend `docker`.

7. Only when the user chose `docker`, resolve the Docker target host
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

8. Only when the user chose `baremetal`, ask whether to install a serving
   framework (used as the `--install-framework` value in Step 4):
   - `none`: use an already-installed SGLang/vLLM framework stack on the host.
   - `sglang`: install SGLang ROCm framework components (shared with the host torch).
   - `vllm (isolated)`: install vLLM into a dedicated venv. vLLM's ROCm wheel
     pins its own torch, so it runs in an isolated env and never touches the
     host torch/SGLang stack.
   - If the user is unsure, recommend `none` when a framework is already present;
     otherwise recommend `sglang`.

## Step 3: Write `.env`

Create or update `.env` in the current directory.

- For every value the user chose in this run (LLM mode, base URL, model, run
  mode, `USER_DATA_PATH`, Docker target host), write exactly what the user
  selected. This wins over any pre-existing value in `.env` or the shell
  environment — e.g. if the user picked the Anthropic official URL, write
  `ANTHROPIC_BASE_URL=https://api.anthropic.com`
  even when a different `ANTHROPIC_BASE_URL` already exists.
- Preserve existing keys unrelated to this setup.
- Never print secret values back to the user.
- Do not write `HYPERLOOM_INSTALL_SOURCE`.
- Do not overwrite an existing non-placeholder secret key.

Write only the keys for the selected LLM mode, plus the common keys. Do not
write keys that belong to a mode the user did not choose.

- `Anthropic`: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_MODEL`.
- `DeepSeek`: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`.

Common keys (all modes):

- `USER_DATA_PATH`
- `HYPERLOOM_RUN_MODE` (`baremetal` or `docker`, the resolved run mode for this session)
- `HYPERLOOM_DOCKER_TARGET_HOST` (only when `HYPERLOOM_RUN_MODE=docker`; the host
  where the demo skill should run Docker)
- `HYPERLOOM_SKILL_PATH` — absolute path to the optimizer skill
  (`<workspace>/hyperloom/inference_optimizer/SKILL.md` for a wheel install,
  `<workspace>/src/hyperloom/inference_optimizer/SKILL.md` for a source checkout).
  Write it in both modes so the demo skill can resolve it even when `docker`
  mode skips the host setup backend.

### AMD APIM subscription header

For `Anthropic` or `DeepSeek`, if the chosen base URL host is `llm-api.amd.com`,
that gateway requires the API key to also be sent as an
`Ocp-Apim-Subscription-Key` header. Add the custom-headers key, reusing the same
key value:

- `Anthropic`: `ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: <ANTHROPIC_API_KEY>"`
- `DeepSeek`: `ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: <DEEPSEEK_API_KEY>"`

The value **must** be wrapped in double quotes: the setup backend loads `.env`
with a shell `source`, so an unquoted value containing a space and a colon
(`Ocp-Apim-Subscription-Key: ...`) is parsed as a command and fails with exit
127. Any `.env` value containing spaces or `:` must be double-quoted.

Write the placeholder header with `<PLEASE_FILL_IN>` when the key itself is
still a placeholder, so the header value tracks the real key after the user
edits `.env`. Skip this entirely when the base URL host is not `llm-api.amd.com`.

After writing `.env`, tell the user to edit the file directly and replace each `<PLEASE_FILL_IN>` placeholder. Wait for the user to confirm before running setup.

Then read `.env` back and confirm:
- non-secret values are correct;
- secret values are `set` or `missing`;
- no secret key still equals `<PLEASE_FILL_IN>`.

If any required secret is missing or still a placeholder, stop and ask the user to edit `.env` again.

## Step 4: Run Setup Backend

In `baremetal` mode, run the backend on the host. The `--install-framework` value
is the framework the user chose in Step 2 (`none` / `sglang` / `vllm`). In
`docker` mode, skip the backend on the host (see below).

### `baremetal`

Run the backend directly in the current directory on the host, passing the
framework the user chose in Step 2.

For `none`:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none
```

For `sglang`:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework sglang --yes
```

For `vllm` (installs into an isolated venv; `--install-framework vllm` already
defaults to isolated, the flag below is explicit):

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework vllm --framework-env isolated --yes
```

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
serving framework to `FRAMEWORK` in `.env` (`sglang` or `vllm`). In `docker`
mode, skip this until the demo skill runs setup inside the container. Read
`.env` back and check it:

- If `FRAMEWORK` is set, report it. Downstream demo skills read this value.
- If `HYPERLOOM_RUN_MODE` is `docker`, an unset `FRAMEWORK` is expected: the
  container has not run yet, so the host has no importable serving framework.
  Tell the user the example (workload) skill will detect or provide the
  framework when it starts the container. Do not offer to re-run setup with
  `--install-framework sglang` or `vllm`.
- If `HYPERLOOM_RUN_MODE` is `baremetal` and `FRAMEWORK` is missing or empty,
  no serving framework was importable on the host (e.g. `--install-framework
  none` without SGLang/vLLM installed). Tell the user that demo skills needing
  a framework will not run until one is installed, and offer to re-run setup
  with `--install-framework sglang` or `vllm`. Do not invent a `FRAMEWORK`
  value.

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
which length:

- `3h` — short, no-kernel run. Best for a first end-to-end check.
- `8h` — medium-length run.
- `24h` — long-horizon cyclic run.

If the user declines, stop here. If `HYPERLOOM_RUN_MODE` is `baremetal` and
`FRAMEWORK` is unset, do not offer a demo; tell the user to install a serving
framework first (see Step 5).

When the user picks a length, load the matching demo skill and follow it — you
stop acting on this setup skill and run the demo skill's instructions instead:

The demo skills are installed under each agent's discovery dir (`.agents/skills/`,
`.claude/skills/`, `.cursor/skills/`); load the matching one by name:

- `3h` → `hyperloom-qwen3-8b-3h`
- `8h` → `hyperloom-qwen3-30b-a3b-8h`
- `24h` → `hyperloom-gpt-oss-120b-24h`

The demo skill reads the values already in `.env` (LLM keys/base URLs,
`FRAMEWORK`, `USER_DATA_PATH`), so the user re-enters nothing. It resolves the
optimizer skill from `.env` `HYPERLOOM_SKILL_PATH` (written by Step 3).
