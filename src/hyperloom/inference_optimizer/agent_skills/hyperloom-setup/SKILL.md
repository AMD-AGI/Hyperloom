---
name: hyperloom-setup
description: Configure Hyperloom after pip install --target by collecting LLM settings, writing .env, and running the setup backend.
---

# Hyperloom Setup

Use this skill after the user installs Hyperloom into the current workspace:

```bash
pip install your_package.whl --target .
```

The current directory should be the Hyperloom target directory, for example `~/hyperloom`. It is normal for this directory to contain many Python package folders; users do not need to inspect them.

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

2. Collect provider-specific values:

   For `Anthropic`:
   - Ask for `ANTHROPIC_API_KEY`.
   - Ask for `ANTHROPIC_BASE_URL`; if blank, use `https://api.anthropic.com`.
   - Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.

   For `OpenAI`:
   - Ask for `OPENAI_API_KEY`.
   - Ask for `OPENAI_BASE_URL`; if blank, use `https://api.openai.com/v1`.
   - Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

   For `LLM Gateway`:
   - Ask for `SAFE_API_KEY`.
   - Ask for `OPENAI_BASE_URL`; if blank, use `https://global.primus-safe.amd.com/api/v1/llm-proxy/v1`.
   - Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.
   - Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

3. Explain `USER_DATA_PATH`:
   - It is the writable root for Hyperloom runtime files, dependency checkouts, logs, optimizer runs, and generated env files.
   - Ask whether to use the current directory as the default or provide a custom path.
   - If the user chooses the default, write the absolute current directory path.

## Step 3: Write `.env`

Create or update `.env` in the current directory.

- Preserve unrelated existing keys.
- Replace values collected by this setup.
- Never print secret values back to the user.
- Do not write `HYPERLOOM_INSTALL_SOURCE`.

Use these keys:

- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_MODEL`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CODEX_MODEL`
- `SAFE_API_KEY`, `OPENAI_BASE_URL`, `CLAUDE_MODEL`, `CODEX_MODEL`
- `USER_DATA_PATH`

After writing, read `.env` back and confirm that all non-secret values are correct. Report secret keys as `set` only.

## Step 4: Run Setup Backend

Run setup from the same directory:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.cli.setup
```

If the user asks to install a serving framework, pass it after `--`, for example:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.cli.setup -- --install-framework sglang --yes
```

## Step 5: Report Result

Report:
- The `.env` path.
- The setup command that was run.
- Whether setup completed or failed.
- The last relevant error lines on failure.

Do not print secret values back to the user.
