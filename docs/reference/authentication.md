---
myst:
    html_meta:
        "description": "Authoritative reference for Hyperloom credentials and environment configuration. Covers single-gateway and split-entrypoint LLM setup, OPENAI_API_KEY, path variables, and hosted mode."
        "keywords": "Hyperloom, authentication, credentials, OPENAI_API_KEY, OPENAI_BASE_URL, ANTHROPIC_BASE_URL, environment variables, LLM gateway, ROCm, AMD GPU, API key, configuration"
---
# Hyperloom authentication and credentials

This is the single authoritative reference for credentials and
environment configuration in Hyperloom. If any other document
(`README.md`, `src/hyperloom/inference_optimizer/SKILL.md`,
`src/hyperloom/agents/kernel/SKILL.md`,
`src/hyperloom/agents/robustness/SKILL.md`) appears to contradict this
page, this page wins. Open an issue against the contradicting file.

Hyperloom needs at most two classes of configuration:

- **LLM gateway credentials**: At least one provider side, configured with both
   its base URL and its own key (see
   [LLM gateway credentials](#llm-gateway-credentials)).
- **Path / workspace layout**: Run bare-metal setup from the installed
   Hyperloom target directory. You normally only set `USER_DATA_PATH`
   (writable artifact root; default `/workspace/hyperloom`); setup writes the
   runtime env files and updates `.env`.

Hyperloom never borrows one provider's key or endpoint for the other. A side is
configured by `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`, or by
`OPENAI_BASE_URL` + `OPENAI_API_KEY`, or both. A side you leave unset simply
disables the features that speak its protocol; a half-configured side fails
preflight rather than being silently completed from the other provider.

The only values still filled for you are the internal GEAK / LLM aliases
(`GEAK_API_KEY`, `LLM_API_KEY`, `AMD_LLM_API_KEY`, `GEAK_BASE_URL`,
`LLM_API_BASE`), which the inference optimizer CLI preflight copies from the
OpenAI side. You do not set those by hand.

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
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` /
`DEEPSEEK_API_KEY`), the CLI fails fast at startup with a message
naming the missing pieces.

Preflight also rejects a *mispaired* configuration: a base URL whose only key
belongs to the other provider, or a key whose only endpoint would come from the
other provider. Hyperloom would otherwise send that key to a foreign host. Give
each side its own `*_BASE_URL` and key, or drop the foreign key.

---

## LLM gateway credentials

A provider side is configured by **both** its base URL and its own key. Exactly
three shapes are accepted; anything else fails preflight.

| Shape | Set | Effect |
|-------|-----|--------|
| OpenAI side only | `OPENAI_BASE_URL` + `OPENAI_API_KEY` | Codex / GEAK run; Claude-side features are disabled |
| Anthropic side only | `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` | Claude runs; Codex / GEAK (OpenAI-protocol) are disabled |
| Both sides | all four, each side self-consistent | Everything runs |

One gateway serving both providers is the third shape: point both base URLs at
it and set both keys, even when the key value is the same.

| Variable             | Issuer                | Where to obtain                                                             | Format              |
|----------------------|-----------------------|-----------------------------------------------------------------------------|---------------------|
| `OPENAI_BASE_URL`    | Your LiteLLM gateway  | `https://<your-gateway-host>/api/v1/llm-proxy/v1` (adjust to your gateway)  | URL ending in `/v1` |
| `OPENAI_API_KEY`     | Your LiteLLM gateway  | Your gateway's LLM Gateway page                                             | `ak-...`            |
| `ANTHROPIC_BASE_URL` | Your gateway / Anthropic | Same gateway without the trailing `/v1`, or `https://api.anthropic.com`  | URL                 |
| `ANTHROPIC_API_KEY`  | Your gateway / Anthropic | Your gateway's page, or the Anthropic console                            | `ak-...` / `sk-ant-...` |

Downstream tooling reads the side it belongs to:

* GEAK and kernel tools inherit the OpenAI-side credential from preflight
  (`GEAK_API_KEY` / `LLM_API_KEY` / `AMD_LLM_API_KEY`).
* Orchestration Claude uses the Anthropic-side base URL + key, including the
  generated `~/.claude/config.json` primary key.
* Robustness-agent uses the OpenAI side for the optional LLM RCA engine.
* Critic-agent uses the OpenAI side for KB summary / synthesis calls.

You *never* need to copy a key into the internal GEAK / LLM slots in `.env`;
preflight fills those from the OpenAI-side key. Preflight does **not** cross-fill
the per-provider primary keys (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`), and an explicitly set provider key is never overwritten.

The recommended setup, run once per shell:

```bash
cd "$REPO_ROOT"
cp .env.template .env
# Edit .env: set the base URL + key for each side you want enabled.
```

For one-off use without writing to disk:

```bash
# One gateway serving both providers: configure both sides against it.
export OPENAI_BASE_URL=https://<your-gateway-host>/api/v1/llm-proxy/v1
export OPENAI_API_KEY=ak-your-gateway-apikey
export ANTHROPIC_BASE_URL=https://<your-gateway-host>/api/v1/llm-proxy
export ANTHROPIC_API_KEY=ak-your-gateway-apikey
```

Dropping the two `ANTHROPIC_*` lines is also valid — it just disables the
Claude-side features.

### Split entrypoints (native Anthropic + OpenAI)

Use when Claude and GPT live on different upstream vendors or gateways.
Set each side explicitly.

| Variable              | Side      | Example                         |
|-----------------------|-----------|---------------------------------|
| `ANTHROPIC_BASE_URL`  | Claude    | `https://api.anthropic.com`     |
| `ANTHROPIC_API_KEY`   | Claude    | `sk-ant-...`                    |
| `OPENAI_BASE_URL`     | Codex/GPT | `https://api.openai.com/v1`     |
| `OPENAI_API_KEY`      | Codex/GPT | `sk-...`                        |

Preflight resolves `(anthropic_base_url, openai_base_url)` independently: each
side is its own explicit base URL, or the official SDK endpoint implied by that
side's own key, or empty. Claude CLI auth uses `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`); GEAK uses the OpenAI-side URL and key. Neither side is
ever completed from the other.

To pin models in split mode:

```bash
export CLAUDE_MODEL=orchestration-model-id-on-the-anthropic-side
export CODEX_MODEL=model-id-on-the-openai-side
```

### Non-AMD / self-hosted gateway

Point `OPENAI_BASE_URL` and `OPENAI_API_KEY` (or the split keys above) at
your own LiteLLM-compatible gateway (Vultr, TensorWave, on-prem, etc.) and
pin models your gateway actually serves:

```bash
export CLAUDE_MODEL=your-gateway-orchestration-model
export CODEX_MODEL=your-gateway-kernel-model
```

Custom orchestration models are allowed by default: preflight validates the
chosen `CLAUDE_MODEL` against your gateway's `/models` catalog. To restore the
stricter AMD Claude allowlist (`claude-opus-4-8` preferred, `claude-opus-4-7` /
`claude-opus-4-6` fallback), set
`INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`.

---

## Optional credentials

The following credentials are optional and only needed for specific backends.

### LLM RCA in robustness-agent

`robustness-agent`'s LLM root-cause-analysis engine activates when an
LLM base URL and API key are available (normally through the aliases above).
Set `ROBUSTNESS_LLM_RCA_DISABLED=1` to force-disable it even when
credentials are present.

## Path environment

These are *not* secrets. You normally do not hand-export
`INFERENCEX_PATH` or `TRACELENS_ROOT` — `install.sh` and its chained
kernel-agent installer clone and pin the open-source checkouts when missing.
Runtime paths are persisted into `.env` by the installer; the CLI preflight
loads them and derives `PATH` / `LD_LIBRARY_PATH` from `ROCM_PATH` /
`VIRTUAL_ENV` / `VLLM_VENV_ROOT` at launch, so no separate file needs sourcing.

### Workspace variables

| Variable               | Set by operator? | Default                                          | Description                                                                                                      |
|------------------------|------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `USER_DATA_PATH`       | recommended      | `/workspace/hyperloom`                           | Writable root for session dirs, `runtime/`, `logs/`, optimizer artifacts. Replaces retired `WORKSPACE_PATH` / `INFERENCE_OPTIMIZER_SESSION_DIR`. |
| `REPO_ROOT`            | rarely           | auto-detected from script location               | This Hyperloom checkout. Locates `.env`, skills, scripts.                                                        |
| `KERNEL_AGENT_ENV`     | rarely           | `$USER_DATA_PATH/runtime/kernel-agent.env.sh`    | Output of `install.sh`; exports resolved paths and LLM aliases.                                                  |
| `HYPERLOOM_RUNTIME_DIR`| rarely           | `$USER_DATA_PATH/runtime`                        | Shared runtime tree (env files, GEAK config, Recipe KB bookkeeping).                                             |

### Dependency checkout variables

Open-source dependencies default under `HYPERLOOM_CACHE_DIR`
(`$REPO_ROOT/.cache`), cloned per revision as `<name>@<sha>`. The cache is
repo-local and writable, so open-source runs need no privileged `/opt` mount.

Leave these variables unset unless you maintain your own checkouts. An
explicit path pointing at a missing directory fails preflight.

| Variable                     | Set by operator? | Default / auto-clone target                                | Description                                                                                                         |
|------------------------------|------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_CACHE_DIR`        | rarely           | `$REPO_ROOT/.cache`                                        | Writable, repo-local root for open-source deps (TraceLens, InferenceX, GEAK), cloned per revision as `<name>@<sha>`. |
| `INFERENCEX_PATH`            | optional override | `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/InferenceX@<sha>` | [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) for baseline and target analysis; the inference_optimizer installer (`src/hyperloom/inference_optimizer/assets/install.sh`) clones it when unset. |
| `TRACELENS_ROOT`             | optional override | `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens@<sha>` | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) for profiling and kernel detection; the kernel-agent installer clones and pins it when unset. |
| `TRACELENS_INTERNAL_ROOT`    | optional         | unset (MAF measured on-device)                             | Optional internal TraceLens extension that backfills MAF without an on-device benchmark. When unset, Hyperloom measures MAF on an idle GPU (microbenchmark) — roofline gap / MI355+ MAF analysis is still produced, just measured locally. Hyperloom never clones it. |
| `MAGPIE_PATH`                | optional override | Resolved from installed `Magpie` package                  | Magpie package root for benchmark wrappers and patch inspection. `install.sh` pip-installs Magpie from `MAGPIE_PACKAGE_SPEC` when it is not importable. |

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
- Writes `~/.claude/config.json`: `customApiUrl` is set to the resolved
  Anthropic-side base URL, and `primaryApiKey` to the Anthropic-side key
  (explicit `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` wins, else
  `OPENAI_API_KEY`).
- Fills the internal GEAK / LLM aliases from the provider key for child
  processes; the per-provider primary keys are not cross-filled.

**401 recovery:**

1. Confirm `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and the matching
   API key are set and current.
2. Re-run preflight (any `python -m hyperloom.inference_optimizer.cli ...` command) or
   `bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh" --check-only`.
3. Inspect `~/.claude/config.json` — `customApiUrl` must point at the
   resolved Anthropic-side upstream gateway.

GEAK uses the generated runtime configuration directly.

---

## Hosted mode (Primus-Claw)

When you launch through the hosted Hyperloom UI,
you do not need to set any of the variables above by hand. The
sandbox initializer binds your LLM Gateway key as `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY`, populates the path env from sandbox defaults, and runs
install/preflight so downstream tools inherit the gateway URL and aliases.

---

## FAQ

These questions cover common credential configuration scenarios.

**Q: Do I have to put my key in `.env`?**

No. Exporting credentials in your shell is sufficient. `.env` is a
convenience for persistence between shells.

**Q: I exported `OPENAI_API_KEY` but `.env` has a different value. Which wins?**

The exported (shell) value wins. `.env` only fills missing keys.

**Q: Can I use separate provider keys instead of one shared gateway key?**

Yes, in split-entrypoint mode: set `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`
(and the matching base URLs) for each side. See
[Split entrypoints](#split-entrypoints-native-anthropic--openai).

**Q: Where do provider-specific API keys come from?**

From each side's own variable: the OpenAI side reads `OPENAI_API_KEY`, the
Anthropic side reads `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`). Neither is
derived from the other. One gateway serving both protocols still needs both set,
even when the value is identical.

**Q: My organization rotates the LLM gateway key weekly. How?**

Re-export the key(s) and re-run `install.sh` (idempotent). Preflight
and all aliases pick up the new value on the next CLI launch.

## More info

Use these resources for related configuration and reference information:

* [Environment variables](environment-variables.md): Every environment variable read by the code, including non-credential tunables.
* [Troubleshooting Hyperloom](troubleshooting.md): Common 401 / gateway / Ray-GPU symptoms.
