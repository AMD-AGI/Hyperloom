---
name: hyperloom-setup
description: Configure Hyperloom after pip install --target by collecting LLM settings, writing .env, and running the setup backend.
---

# Hyperloom Setup

Use this skill when the user has installed Hyperloom into the current workspace with:

```bash
pip install your_package.whl --target .
```

The current working directory should be the Hyperloom target directory (for example `~/hyperloom`).

## Workflow

1. Confirm the current directory is the Hyperloom target directory.
   - It should contain a `hyperloom/` Python package directory.
   - If not, ask the user for the correct target directory and switch to it.

2. Ask the user to choose one LLM configuration mode:
   - `Anthropic`
   - `OpenAI`
   - `LLM Gateway`

3. Collect credentials and model names.

For `Anthropic`:
- Ask for `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`.
- Ask for `ANTHROPIC_BASE_URL`; if blank, use `https://api.anthropic.com`.
- Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.

For `OpenAI`:
- Ask for `OPENAI_API_KEY`.
- Ask for `OPENAI_BASE_URL`; if blank, use `https://api.openai.com/v1`.
- Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

For `LLM Gateway`:
- Ask for `SAFE_API_KEY` or `LLM_GATEWAY_KEY`.
- Ask for `OPENAI_BASE_URL`; if blank, use `https://global.primus-safe.amd.com/api/v1/llm-proxy/v1`.
- Tell the user the default `CLAUDE_MODEL` is `claude-opus-4-8`; ask whether to change it.
- Tell the user the default `CODEX_MODEL` is `gpt-4.1`; ask whether to change it.

4. Write or update `.env` in the current directory.
   - Preserve unrelated existing keys.
   - Replace values for keys collected in this setup.
   - Set `USER_DATA_PATH` to the current directory unless the user explicitly provides another path.

5. Run the setup backend from the same directory:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.cli.setup
```

If the user wants a framework installed, pass it after `--`, for example:

```bash
PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.cli.setup -- --install-framework sglang --yes
```

6. Report:
- The `.env` path.
- The setup command that was run.
- Any setup failures and the last relevant error lines.

Do not print secret values back to the user.
