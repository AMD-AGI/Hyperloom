---
name: hyperloom-setup
description: Configure Hyperloom after pip install --target by collecting LLM settings, choosing a bare-metal or Docker run mode, writing .env, and running the setup backend.
---

# Hyperloom Setup

Use this skill after the user installs Hyperloom into the current workspace:

```bash
pip install your_package.whl --target .
```

The current directory should be the Hyperloom target directory, for example `~/hyperloom`. It is normal for this directory to contain many Python package folders; users do not need to inspect them.

This skill supports two run modes, selected by `HYPERLOOM_RUN_MODE`:

- `baremetal`: run the setup backend directly on the host. The host provides ROCm, and setup can optionally install the SGLang or vLLM framework layer.
- `docker`: start a user-specified ROCm container that already ships the serving framework, then run the setup backend inside it. The framework comes from the image, so no framework install happens in this mode.

## Invocation

The run mode can be passed as an argument after the command, so the user does not
have to answer the run-mode question interactively:

```text
/hyperloom-setup                 # ask for the run mode interactively
/hyperloom-setup baremetal       # force bare-metal mode
/hyperloom-setup docker <image>  # force docker mode with a specific image
```

Resolve the run mode in this order and skip the matching interactive question
when a value is already present:

1. An explicit argument after the command (`baremetal` or `docker`).
2. `HYPERLOOM_RUN_MODE` in the shell environment.
3. Otherwise, ask the user (Step 2 question 6).

In `docker` mode an image is **required** — there is no default, and the skill
does not ask for it interactively. The user must supply it up front, either as a
second argument after `docker` (`/hyperloom-setup docker <image>`) or via
`HYPERLOOM_IMAGE` in the shell environment. If neither is present, stop and
tell the user to re-run with an image, suggesting one of these AMD reference
images (or the user can bring their own):

- SGLang MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- SGLang MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`
- vLLM MI300X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- vLLM MI355X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`

The image must be a ROCm image that already provides a working SGLang or vLLM
stack for the user's GPU; in `docker` mode this skill does not install the
framework.

The run mode, image, container name, and shared-memory size are agent
orchestration values resolved for this session from command arguments or the
shell environment. Keep them only for the current run and use them directly to
run the backend and, in `docker` mode, to start the container.

## Workflow

You must run this as an interactive onboarding flow. Do not stop after listing required values. Ask the user each question, collect the answer, write `.env`, read it back for validation, and continue to the setup command.

## Step 1: Confirm Workspace

Confirm the current directory contains a `hyperloom/` Python package directory. If it does not, ask the user for the target directory and switch there.

## Step 2: Ask Configuration Questions

Ask these questions using the agent's structured question UI when available.

1. Choose one LLM mode:
   - `Anthropic`
   - `OpenAI`
   - `LLM Gateway`

   Present exactly those three option labels. Do not add parenthetical
   descriptions, vendor examples, or base URLs to this first question.

2. Ask the base URL as a separate follow-up question after the mode is chosen.

3. Explain that secrets must be edited in `.env`, not pasted into chat.
   - Never ask the user to paste API keys into the conversation.
   - Create `.env` with placeholders for secret values.
   - Ask the user to edit `.env` directly and replace placeholders.
   - After the user confirms the file is edited, validate only whether secret keys are set; do not print secret values.

4. Collect provider-specific non-secret values and write secret placeholders:

   For `Anthropic`:
   - Write `ANTHROPIC_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask for `ANTHROPIC_BASE_URL`; if blank, use `https://api.anthropic.com`.
   - Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.

   For `OpenAI`:
   - Write `OPENAI_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask for `OPENAI_BASE_URL`; if blank, use `https://api.openai.com/v1`.
   - Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

   For `LLM Gateway`:
   - Write `SAFE_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask for `OPENAI_BASE_URL`; if blank, use `https://global.primus-safe.amd.com/api/v1/llm-proxy/v1`.
   - Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.
   - Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

5. Explain `USER_DATA_PATH`:
   - It is the writable root for Hyperloom runtime files, dependency checkouts, logs, optimizer runs, and generated env files.
   - If an existing `USER_DATA_PATH` is visible in the current shell or terminal context, offer that exact value as one option.
   - Always offer the current workspace directory as an option.
   - Always offer a custom path option.
   - Do not assume or auto-select any option; write `USER_DATA_PATH` only after the user explicitly chooses.
   - If the user selects the current workspace directory, write its absolute path.

6. Ask where to run Hyperloom (sets `HYPERLOOM_RUN_MODE` for this session). Skip
   this question if the run mode was already resolved from a command argument or
   an existing `HYPERLOOM_RUN_MODE` value (see [Invocation](#invocation)); just
   confirm it and use it for this run.
   - `baremetal`: run the setup backend directly on this host. Choose this when
     the host provides ROCm (the framework may already be present, or setup can
     install it).
   - `docker`: run the setup backend inside a ROCm container that already ships
     the serving framework. Choose this when the host does not have the
     framework installed but Docker with GPU access is available.
   - If the user is unsure and is already inside a framework image or shell with
     a working framework, recommend `baremetal`; otherwise recommend `docker`.

7. Only when the user chose `baremetal`, ask whether to install a serving
   framework (used as the `--install-framework` value in Step 4):
   - `none`: use an already-installed SGLang/vLLM framework stack on the host.
   - `sglang`: install SGLang ROCm framework components.
   - `vllm`: install vLLM ROCm framework components.
   - If the user is unsure, recommend `none` when a framework is already present;
     otherwise recommend `sglang`.

In `docker` mode, do not ask for the container image here — the user supplies
`HYPERLOOM_IMAGE` up front (see [Invocation](#invocation)); just validate it is
set and use it for this run.

## Step 3: Write `.env`

Create or update `.env` in the current directory.

- Preserve unrelated existing keys.
- Replace values collected by this setup.
- Never print secret values back to the user.
- Do not write `HYPERLOOM_INSTALL_SOURCE`.
- Do not overwrite an existing non-placeholder secret key.

Write only the Hyperloom configuration keys the setup backend consumes:

- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_MODEL`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CODEX_MODEL`
- `SAFE_API_KEY`, `OPENAI_BASE_URL`, `CLAUDE_MODEL`, `CODEX_MODEL`
- `USER_DATA_PATH`

After writing `.env`, tell the user to edit the file directly and replace each `<PLEASE_FILL_IN>` placeholder. Wait for the user to confirm before running setup.

Then read `.env` back and confirm:
- non-secret values are correct;
- secret values are `set` or `missing`;
- no secret key still equals `<PLEASE_FILL_IN>`.

If any required secret is missing or still a placeholder, stop and ask the user to edit `.env` again.

## Step 4: Run Setup Backend

Run the backend based on `HYPERLOOM_RUN_MODE`. The `--install-framework` value
differs by mode: in `baremetal` mode use the framework the user chose in Step 2
(`none` / `sglang` / `vllm`); in `docker` mode always use `none` because the
image already provides the framework.

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

For `vllm`:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework vllm --yes
```

### `docker`

Start a long-running container from `HYPERLOOM_IMAGE`, mounting the workspace at
the same path inside the container so paths stay consistent, then run the
backend with `--install-framework none` inside it.

The container name and shared-memory size are optional agent orchestration
values read from the shell environment (same as the run mode and image):

- `HYPERLOOM_CONTAINER_NAME` — container name (default `hyperloom-local`).
- `HYPERLOOM_SHM_SIZE` — `--shm-size` value (default `64g`).

1. Start the container (skip if a container named `$HYPERLOOM_CONTAINER_NAME` is
   already running the desired image):

   ```bash
   docker run -d \
     --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
     --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
     --device /dev/kfd \
     --device /dev/dri \
     --group-add video \
     -v "$PWD:$PWD" \
     "$HYPERLOOM_IMAGE" \
     tail -f /dev/null
   ```

   - If `USER_DATA_PATH` is outside the workspace, add another
     `-v "$USER_DATA_PATH:$USER_DATA_PATH"` mount so the container can write there.
   - To reuse a host model directory, add `-v /path/to/models:/models`.

2. Run the setup backend inside the container:

   ```bash
   docker exec -w "$PWD" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
     'PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
   ```

In `docker` mode, do not run `--install-framework sglang` or
`--install-framework vllm`; the serving framework must come from the chosen
image.

## Step 5: Report Result

Report:
- The `.env` path.
- The run mode (`baremetal` or `docker`), and in `docker` mode the image and
  container name used.
- The setup command that was run.
- Whether setup completed or failed.
- The last relevant error lines on failure.

Do not print secret values back to the user.
