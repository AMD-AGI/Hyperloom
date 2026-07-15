---
myst:
    html_meta:
        "description": "Authoritative reference for Hyperloom credentials and environment configuration. Covers single-gateway and split-entrypoint LLM setup, SAFE_API_KEY, path variables, and hosted mode."
        "keywords": "Hyperloom, authentication, credentials, SAFE_API_KEY, OPENAI_BASE_URL, ANTHROPIC_BASE_URL, environment variables, LLM gateway, ROCm, AMD GPU, API key, configuration"
---
# Hyperloom authentication and credentials

This is the single authoritative reference for credentials and
environment configuration in Hyperloom. If any other document
(`README.md`, `src/hyperloom/inference_optimizer/SKILL.md`,
`src/hyperloom/agents/kernel/SKILL.md`,
`src/hyperloom/agents/robustness/SKILL.md`) appears to contradict this
page, this page wins. Open an issue against the contradicting file.

Hyperloom needs at most two classes of configuration:

- **LLM gateway credentials** — at least one upstream base URL and one API
   key (see [LLM gateway credentials](#llm-gateway-credentials)). On the AMD
   network the usual pair is `SAFE_API_KEY` + `OPENAI_BASE_URL`; split
   Anthropic/OpenAI entrypoints and third-party gateways are also supported.
- **Path / workspace layout** — run bare-metal setup from the installed
   Hyperloom target directory. You normally only set `USER_DATA_PATH`
   (writable artifact root; default `/workspace/hyperloom`); setup writes the
   runtime env files and updates `.env`.

In the **single-gateway** setup, GEAK keys and Anthropic / OpenAI
aliases are **derived** from `SAFE_API_KEY` and
`OPENAI_BASE_URL` by `src/hyperloom/agents/kernel/scripts/install.sh` and
the inference optimizer CLI preflight. You normally do not set those
aliases by hand. Split-gateway and GEAK gateway endpoint overrides are the
exceptions (see below).

---

## Credential precedence

Hyperloom reads credentials from two places, in order:

| Source                                      | When used                                                 |
|---------------------------------------------|-----------------------------------------------------------|
| Process environment (`export FOO=...`)      | Always — wins unconditionally.                            |
| `$REPO_ROOT/.env`                           | Only for keys missing from the process environment.       |

Shell environment variables always win over `.env`.
Both the inference optimizer CLI dotenv loader and
`src/hyperloom/agents/kernel/scripts/install.sh` honor this rule. Do *not*
manually `source .env` from chat — it inverts the precedence and can
overwrite an exported key with a stale value from disk.

A key is considered "set" in `.env` if the line is uncommented and the
value is non-empty. Empty / commented lines are ignored; the
shell-exported value (if any) is kept.

If neither source supplies a usable LLM endpoint (at least one of
`OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and at least one of
`SAFE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`), the CLI fails fast at startup with a message
naming the missing pieces.

---

## LLM gateway credentials

Two deployment shapes are supported. Pick one; do not mix the blocks
unless you intend a split entrypoint.

### Single gateway (default — AMD / LiteLLM-style)

One OpenAI-compatible endpoint serves both Claude and GPT models
(LiteLLM gateway, AMD primus-safe, or any self-hosted `/v1` proxy).

| Variable           | Issuer                              | Where to obtain                                                                                       | Format              |
|--------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| `SAFE_API_KEY`     | AMD LiteLLM gateway                 | [LLM Gateway](https://global.primus-safe.amd.com/litellm-gateway)                                     | `ak-...`            |
| `OPENAI_BASE_URL`  | AMD LiteLLM gateway                 | `https://global.primus-safe.amd.com/api/v1/llm-proxy/v1` (default for the hosted SaFE setup)          | URL ending in `/v1` |

`SAFE_API_KEY` is the single AMD credential used by all downstream
tooling:

* GEAK and kernel tools inherit the resolved gateway credential from preflight.
* Orchestration LLM → `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (auto-aliased).
* Robustness-agent uses it for the optional LLM RCA engine.
* Critic-agent uses it for KB summary / synthesis calls.

You *never* need to copy `SAFE_API_KEY` into separate
provider-specific slots in `.env`.
If those variables are unset, install and preflight fill them from
`SAFE_API_KEY` at process-launch time. An explicitly set provider key is
never overwritten by `SAFE_API_KEY`.

The recommended setup, run once per shell:

```bash
cd "$REPO_ROOT"
cp .env.template .env
# Edit .env and set SAFE_API_KEY=ak-...
```

For one-off use without writing to disk:

```bash
export SAFE_API_KEY=ak-your-safe-apikey
export OPENAI_BASE_URL=https://global.primus-safe.amd.com/api/v1/llm-proxy/v1
```

### Split entrypoints (native Anthropic + OpenAI)

Use when Claude and GPT live on different upstream vendors or gateways.
Set each side explicitly; `SAFE_API_KEY` is optional.

| Variable              | Side      | Example                         |
|-----------------------|-----------|---------------------------------|
| `ANTHROPIC_BASE_URL`  | Claude    | `https://api.anthropic.com`     |
| `ANTHROPIC_API_KEY`   | Claude    | `sk-ant-...`                    |
| `OPENAI_BASE_URL`     | Codex/GPT | `https://api.openai.com/v1`     |
| `OPENAI_API_KEY`      | Codex/GPT | `sk-...`                        |

Preflight resolves `(anthropic_base_url, openai_base_url)` independently:
when both are set, each is kept as-is. Claude CLI auth uses
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in preference to any
shared key. GEAK inherits the OpenAI-side URL and key.

To pin models in split mode:

```bash
export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
export CLAUDE_MODEL=orchestration-model-id-on-the-anthropic-side
export CODEX_MODEL=model-id-on-the-openai-side
```

### Non-AMD / self-hosted gateway

Point `OPENAI_BASE_URL` and `SAFE_API_KEY` (or the split keys above) at
your own LiteLLM-compatible gateway (Vultr, TensorWave, on-prem, etc.).
Then opt out of the AMD-only orchestration model gate and pin models your
gateway actually serves:

```bash
export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
export CLAUDE_MODEL=your-gateway-orchestration-model
export CODEX_MODEL=your-gateway-kernel-model
```

With the opt-out set, preflight validates model IDs against your
gateway's `/models` catalog instead of hard-gating to AMD opus-4-7/4-6.

---

## Optional credentials

The following credentials are optional and only needed for specific backends.

### LLM RCA in robustness-agent

`robustness-agent`'s LLM root-cause-analysis engine activates when an
LLM base URL and API key are available (normally via the aliases above).
Set `ROBUSTNESS_LLM_RCA_DISABLED=1` to force-disable it even when
credentials are present.

## Path environment

These are *not* secrets. You normally do not hand-export
`INFERENCEX_PATH` or `TRACELENS_ROOT` — `install.sh` and its chained
kernel-agent installer clone and pin the open-source checkouts when missing.
Before launching optimization, source the generated env file:

```bash
export USER_DATA_PATH=/path/to/hyperloom-run   # optional; default /workspace/hyperloom
# The standard install.sh writes runtime/kernel-agent.env.sh; the bare-metal
# installer (install_baremetal.sh) writes a combined runtime/hyperloom.env.sh.
# Source whichever your install produced.
for _env in hyperloom.env.sh kernel-agent.env.sh; do
  [ -f "$USER_DATA_PATH/runtime/$_env" ] && source "$USER_DATA_PATH/runtime/$_env"
done
```

### Workspace variables

| Variable               | Set by operator? | Default                                          | Description                                                                                                      |
|------------------------|------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `USER_DATA_PATH`       | recommended      | `/workspace/hyperloom`                           | Writable root for session dirs, `runtime/`, `logs/`, optimizer artifacts. Replaces retired `WORKSPACE_PATH` / `INFERENCE_OPTIMIZER_SESSION_DIR`. |
| `REPO_ROOT`            | rarely           | auto-detected from script location               | This Hyperloom checkout. Locates `.env`, skills, scripts.                                                        |
| `KERNEL_AGENT_ENV`     | rarely           | `$USER_DATA_PATH/runtime/kernel-agent.env.sh`    | Output of `install.sh`; exports resolved paths and LLM aliases.                                                  |
| `HYPERLOOM_RUNTIME_DIR`| rarely           | `$USER_DATA_PATH/runtime`                        | Shared runtime tree (env files, GEAK config, Cortex KB bookkeeping).                                             |

### Dependency checkout variables

Open-source dependencies default under `HYPERLOOM_OPEN_SOURCE_ROOT`
(`/opt/hyperloom/open-source-repos`), decoupled from `USER_DATA_PATH` so
shared session storage does not collocate concurrent pods' clones.

Leave these variables unset unless you maintain your own checkouts. An
explicit path pointing at a missing directory fails preflight.

| Variable                     | Set by operator? | Default / auto-clone target                                | Description                                                                                                         |
|------------------------------|------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_OPEN_SOURCE_ROOT` | rarely           | `/opt/hyperloom/open-source-repos`                         | Pod-local root for open-source deps (TraceLens, InferenceX, Magpie, GEAK). Writable `/opt` required unless overridden. |
| `INFERENCEX_PATH`            | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/InferenceX`                | [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) for baseline and target analysis; `install.sh` clones it when unset. |
| `TRACELENS_ROOT`             | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/TraceLens`                 | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) for profiling and kernel detection; the kernel-agent installer clones and pins it when unset. |
| `TRACELENS_INTERNAL_ROOT`    | optional         | unset (open-source-only)                                   | Internal TraceLens extension (roofline gap, MI355+ MAF). Hyperloom never clones it — set only when you maintain a checkout. |
| `MAGPIE_PATH`                | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/Magpie`                    | Magpie benchmark wrappers; installed by `install.sh` when missing.                                                  |

```{note}
`INFERENCE_OPTIMIZER_SESSION_DIR` is no longer read. `WORKSPACE_PATH` is
legacy-only and still used in narrow fallbacks; prefer `USER_DATA_PATH`. See
[Upgrade Hyperloom version](upgrade.md).
```

---

## Direct upstream wiring

Claude, Codex, and GEAK talk to the configured upstream gateway directly.
The AMD primus-safe gateway accepts both header styles natively.

At preflight, the inference optimizer CLI:

- Resolves Anthropic and OpenAI base URLs.
- Writes `~/.claude/config.json` `customApiUrl` and `primaryApiKey`.
- Fills the provider credentials needed by child processes.

**401 recovery:**

1. Confirm `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and the matching
   API key are set and current.
2. Re-run preflight (any `inference_optimizer` CLI command) or
   `bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh" --check-only`.
3. Inspect `~/.claude/config.json` — `customApiUrl` must point at the
   configured upstream gateway.

GEAK uses the generated runtime configuration directly.

---

## Hosted mode (Primus-Claw)

When you launch through the [Hyperloom UI](https://crusoe.primus-safe.amd.com/hyperloom/),
you do not need to set any of the variables above by hand. The
sandbox initializer binds your LLM Gateway key as `SAFE_API_KEY`,
populates the path env from sandbox defaults, and runs install/preflight
so downstream tools inherit the gateway URL and aliases. See
[Quickstart — hosted UI](../install/quickstart.md).

---

## FAQ

These questions cover common credential configuration scenarios.

**Q: Do I have to put my key in `.env`?**

No. Exporting credentials in your shell is sufficient. `.env` is a
convenience for persistence between shells.

**Q: I exported `SAFE_API_KEY` but `.env` has a different value. Which wins?**

The exported (shell) value wins. `.env` only fills missing keys.

**Q: Can I run without `SAFE_API_KEY`?**

Yes, in split-entrypoint mode: set `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`
(and the matching base URLs) instead. See
[Split entrypoints](#split-entrypoints-native-anthropic--openai).

**Q: Where do provider-specific API keys come from?**

In single-gateway mode they are derived from `SAFE_API_KEY` by
`install.sh` and preflight. In split mode each side uses its own key.

**Q: My organization rotates the LLM gateway key weekly. How?**

Re-export the key(s) and re-run `install.sh` (idempotent). Preflight
and all aliases pick up the new value on the next CLI launch.

## More info

Use these resources for related configuration and reference information:

* [Environment variables](environment-variables.md) — Every environment variable read by the code, including non-credential tunables.
* [Troubleshooting Hyperloom](troubleshooting.md) — Common 401 / gateway / Ray-GPU symptoms.
