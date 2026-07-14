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

This skill resolves a run mode into `HYPERLOOM_RUN_MODE` (`baremetal` or `docker`)
for this session and runs the setup backend. It does **not** start any container.
Whether to generate or run a Docker container is decided later by the example
(workload) skill based on `HYPERLOOM_RUN_MODE`.

- `baremetal`: the host provides ROCm, and setup can optionally install the SGLang or vLLM framework layer.
- `docker`: the container (and its serving framework) is provided later by the example when it runs the workload, so setup installs no framework in this mode.

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

2. Ask the base URL as a structured follow-up after the mode is chosen: two
   options `Use default (<provider default URL>)` and `Custom`; if `Custom`,
   ask a plain-text follow-up for the URL.

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
   - Ask `ANTHROPIC_BASE_URL`: options `Use default (https://api.anthropic.com)` / `Custom`.
   - Ask `CLAUDE_MODEL`: options `Use default (claude-opus-4-8)` / `Custom`.

   For `DeepSeek`:
   - Write `DEEPSEEK_API_KEY=<PLEASE_FILL_IN>` unless already set to a non-placeholder value.
   - Ask `DEEPSEEK_BASE_URL`: options `Use default (https://api.deepseek.com/anthropic)` / `Custom`.
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
   - `baremetal`: run the setup backend directly on this host. Choose this when
     the host provides ROCm (the framework may already be present, or setup can
     install it).
   - `docker`: record docker as the run mode; the example (workload) skill will
     generate a ROCm container later. Choose this when the host does not have the
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

## Step 3: Write `.env`

Create or update `.env` in the current directory.

- For every value the user chose in this run (LLM mode, base URL, model, run
  mode, `USER_DATA_PATH`), write exactly what the user selected. This wins over
  any pre-existing value in `.env` or the shell environment — e.g. if the user
  picked the Anthropic official URL, write `ANTHROPIC_BASE_URL=https://api.anthropic.com`
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

Run the backend based on `HYPERLOOM_RUN_MODE`. The `--install-framework` value
differs by mode: in `baremetal` mode use the framework the user chose in Step 2
(`none` / `sglang` / `vllm`); in `docker` mode always use `none` because the
container the example generates provides the framework.

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

Run the backend with `--install-framework none` and `--skip-base-check`. The
host does not need ROCm or a serving framework yet; the container the example
generates later provides both:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --skip-base-check --install-framework none --yes
```

This skill does not start a container. `HYPERLOOM_RUN_MODE` is recorded so the
example (workload) skill can decide whether to generate a Docker container when
it runs the optimization.

## Step 5: Confirm Detected Framework

After setup completes, the backend writes the detected serving framework to
`FRAMEWORK` in `.env` (`sglang` or `vllm`). Read `.env` back and check it:

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
- The setup command that was run.
- Whether setup completed or failed.
- The detected `FRAMEWORK` value (or that it is unset).
- The last relevant error lines on failure.

Do not print secret values back to the user.

## Step 7: Hand Off to a Demo Skill

Only when setup completed and `FRAMEWORK` is set, ask the user whether they want
to run a demo optimization now, and if so which length:

- `3h` — short, no-kernel run. Best for a first end-to-end check.
- `8h` — medium-length run.
- `24h` — long-horizon cyclic run.

If the user declines, stop here. If `FRAMEWORK` is unset, do not offer a demo;
tell the user to install a serving framework first (see Step 5).

When the user picks a length, load the matching demo skill and follow it — you
stop acting on this setup skill and run the demo skill's instructions instead:

- `3h` → `@.agents/skills/hyperloom-qwen3-8b-3h/SKILL.md`
- `8h` → `@.agents/skills/hyperloom-qwen3-8b-8h/SKILL.md`
- `24h` → `@.agents/skills/hyperloom-qwen3-8b-24h/SKILL.md`

The demo skill reads the values already in `.env` (LLM keys/base URLs,
`FRAMEWORK`, `USER_DATA_PATH`), so the user re-enters nothing. Where the demo
skill references `@../../inference_optimizer/SKILL.md`, that relative path does
not resolve in an installed workspace; use the absolute optimizer skill path
from `.env` `HYPERLOOM_SKILL_PATH` instead.
