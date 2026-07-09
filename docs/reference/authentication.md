---
myst:
    html_meta:
        "description": "Authoritative reference for Hyperloom credentials and environment configuration. Covers single-gateway and split-entrypoint LLM setup, SAFE_API_KEY, CURSOR_API_KEY, path variables, and hosted mode."
        "keywords": "Hyperloom, authentication, credentials, SAFE_API_KEY, CURSOR_API_KEY, OPENAI_BASE_URL, ANTHROPIC_BASE_URL, environment variables, OOB, LLM gateway, ROCm, AMD GPU, API key, configuration"
---
# Hyperloom authentication and credentials

This is the single authoritative reference for credentials and
environment configuration in Hyperloom. If any other document
(`README.md`, `src/hyperloom/inference_optimizer/SKILL.md`,
`src/hyperloom/agents/kernel/SKILL.md`,
`src/hyperloom/agents/robustness/SKILL.md`) appears to contradict this
page, this page wins. Open an issue against the contradicting file.

Hyperloom needs at most three classes of configuration:

- **LLM gateway credentials** — at least one upstream base URL and one API
   key (see [LLM gateway credentials](#llm-gateway-credentials)). On the AMD
   network the usual pair is `SAFE_API_KEY` + `OPENAI_BASE_URL`; split
   Anthropic/OpenAI entrypoints and third-party gateways are also supported.
- The **Cursor SDK** key (`CURSOR_API_KEY`) — optional, only needed if
   you want the OOB `cursor` kernel-opt backend.
- **Path / workspace layout** — for local mode, run
   `src/hyperloom/inference_optimizer/assets/local_setup.sh` once (credentials and the
   Hyperloom checkout are enough). It clones missing dependency repos under
   `HYPERLOOM_OPEN_SOURCE_ROOT`, writes
   `$USER_DATA_PATH/runtime/local-setup.env.sh`, and exports `OOB_SRC`,
   `INFERENCEX_PATH`, `TRACELENS_ROOT`, and so on. You normally only set
   `USER_DATA_PATH` (writable artifact root; default `/workspace/hyperloom`).
   `REPO_ROOT` is auto-detected. Explicit overrides for dependency paths are
   optional (see [Path environment](#path-environment)).

In the **single-gateway** setup, GEAK keys, OOB Claude/Codex keys, and
Anthropic / OpenAI aliases are **derived** from `SAFE_API_KEY` and
`OPENAI_BASE_URL` by `src/hyperloom/agents/kernel/scripts/install.sh` and
the inference optimizer CLI preflight. You normally do not set those
aliases by hand. Split-gateway and GEAK/OOB endpoint overrides are the
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
| `SAFE_API_KEY`     | AMD LiteLLM gateway                 | [LLM Gateway](https://your-openai-compatible-gateway.example.com/litellm-gateway)                                     | `ak-...`            |
| `OPENAI_BASE_URL`  | AMD LiteLLM gateway                 | `https://your-openai-compatible-gateway.example.com/v1` (default for the hosted SaFE setup)          | URL ending in `/v1` |

`SAFE_API_KEY` is the single AMD credential used by all downstream
tooling:

* GEAK → `GEAK_API_KEY` / `GEAK_BASE_URL` (auto-aliased).
* OOB `claude` and `codex` → `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (auto-aliased).
* Robustness-agent uses it for the optional LLM RCA engine.
* Critic-agent uses it for KB summary / synthesis calls.

You *never* need to copy `SAFE_API_KEY` into separate
`GEAK_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` slots in `.env`.
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
export OPENAI_BASE_URL=https://your-openai-compatible-gateway.example.com/v1
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
shared key. GEAK and OOB Codex inherit the OpenAI-side URL and key.

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

### `CURSOR_API_KEY` — Cursor SDK kernel-opt backend

The OOB `cursor` backend talks to Cursor's gateway, not your LLM
gateway. It therefore requires a separate issuer key with prefix `crsr_...`:

| Variable                | Default           | Description                                                                                                                                                                          |
|-------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CURSOR_API_KEY`        | unset             | Cursor SDK key. Never inherited from `SAFE_API_KEY`. `cursor` is the tail of the default `forge,geak,claude,codex,cursor` ladder but is auto-dropped when this key is unset. |
| `CURSOR_DEFAULT_MODEL`  | `claude-opus-4-7-thinking-xhigh` | Override the default Cursor model id.                                                                                                                                                |

The selection notes carry `cursor_key_present: bool` for observability.
If you pass `--backends cursor` explicitly without a key set, the
attempt is still launched and surfaces as a single failed `cursor`
attempt with a 401, rather than a silent skip.

### LLM RCA in robustness-agent

`robustness-agent`'s LLM root-cause-analysis engine activates when an
LLM base URL and API key are available (normally via the aliases above).
Set `ROBUSTNESS_LLM_RCA_DISABLED=1` to force-disable it even when
credentials are present.

### GEAK / OOB endpoint overrides

GEAK Ray workers may run in a network namespace that cannot reach the
main gateway directly. By default `GEAK_BASE_URL` and `OOB_BASE_URL`
inherit the resolved OpenAI-compatible gateway URL. Set them explicitly
when you need a host-local reverse tunnel or another routable path:

```bash
export GEAK_BASE_URL=https://127.0.0.1:18444/api/v1/llm-proxy/v1
export OOB_BASE_URL=https://127.0.0.1:18444/api/v1/llm-proxy/v1
```

Preflight preserves intentional operator overrides. If `GEAK_BASE_URL` or
`OOB_BASE_URL` is set, Hyperloom treats it as deliberate and does not rewrite it.
Only set these variables when the target is reachable from the worker runtime.

---

## Path environment

These are *not* secrets. In local mode you normally do not hand-export
`OOB_SRC`, `INFERENCEX_PATH`, or `TRACELENS_ROOT` — `local_setup.sh`
clones them when missing and records the resolved paths in
`local-setup.env.sh`. Before launching optimization, source that file:

```bash
export USER_DATA_PATH=/path/to/hyperloom-run   # optional; default /workspace/hyperloom
bash src/hyperloom/inference_optimizer/assets/local_setup.sh
source "$USER_DATA_PATH/runtime/local-setup.env.sh"
```

`src/hyperloom/agents/kernel/scripts/install.sh` (invoked by preflight or manually) then
installs runtime dependencies (Ray, GEAK, OOB CLIs, Magpie, and so on) and
writes `$USER_DATA_PATH/runtime/kernel-agent.env.sh`. Source that too when
driving kernel-agent tools directly.

### Workspace variables

| Variable               | Set by operator? | Default                                          | Description                                                                                                      |
|------------------------|------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `USER_DATA_PATH`       | recommended      | `/workspace/hyperloom`                           | Writable root for session dirs, `runtime/`, `logs/`, optimizer artifacts. Replaces retired `WORKSPACE_PATH` / `INFERENCE_OPTIMIZER_SESSION_DIR`. |
| `REPO_ROOT`            | rarely           | auto-detected from script location               | This Hyperloom checkout. Locates `.env`, skills, scripts.                                                        |
| `LOCAL_SETUP_ENV`      | rarely           | `$USER_DATA_PATH/runtime/local-setup.env.sh`     | Output of `local_setup.sh`; source in every fresh shell before running the optimizer.                            |
| `KERNEL_AGENT_ENV`     | rarely           | `$USER_DATA_PATH/runtime/kernel-agent.env.sh`    | Output of `install.sh`; exports resolved paths and LLM aliases.                                                  |
| `HYPERLOOM_RUNTIME_DIR`| rarely           | `$USER_DATA_PATH/runtime`                        | Shared runtime tree (env files, GEAK config, Cortex KB bookkeeping).                                             |

### Dependency checkout variables

Open-source dependencies default under `HYPERLOOM_OPEN_SOURCE_ROOT`
(`/opt/hyperloom/open-source-repos`), decoupled from `USER_DATA_PATH` so
shared session storage does not collocate concurrent pods' clones.
`local_setup.sh` also accepts `HYPERLOOM_DEPS_ROOT` (or `--deps-root`) as
an alias and exports the resolved value as `HYPERLOOM_OPEN_SOURCE_ROOT`.

Leave these variables unset unless you maintain your own checkouts. An
explicit path pointing at a missing directory fails preflight.

| Variable                     | Set by operator? | Default / auto-clone target                                | Description                                                                                                         |
|------------------------------|------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_OPEN_SOURCE_ROOT` | rarely           | `/opt/hyperloom/open-source-repos`                         | Pod-local root for TraceLens, InferenceX, KernelForge, Magpie, GEAK, and OOB. Writable `/opt` required unless overridden. |
| `FORGE_PATH`                 | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/KernelForge`                | KernelForge checkout root for the `forge` backend. `local_setup.sh` also exports `KERNEL_FORGE_ROOT` with the same value. |
| `OOB_SRC`                    | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/KernelForge/OOB`           | OOB kernel-opt backends (claude / codex / cursor). Derived from the KernelForge checkout.                           |
| `INFERENCEX_PATH`            | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/InferenceX`                | [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) for baseline and target analysis.        |
| `TRACELENS_ROOT`             | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/TraceLens`                 | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) for profiling and kernel detection; pinned to a fixed SHA on auto-clone. |
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
- Fills unset alias env vars (`GEAK_*`, `OOB_*`, and so on).
- Preserves explicit `GEAK_BASE_URL` / `OOB_BASE_URL` overrides for separate
  routable endpoints.

**401 recovery:**

1. Confirm `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and the matching
   API key are set and current.
2. Re-run preflight (any `inference_optimizer` CLI command) or
   `bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh" --check-only`.
3. Inspect `~/.claude/config.json` — `customApiUrl` must point at the
   configured upstream gateway.

The `cursor` backend always bypasses your LLM gateway (Cursor's own
issuer). GEAK uses `GEAK_API_KEY` / `GEAK_BASE_URL` (or the generated
litellm config) directly.

---

## Hosted mode (Primus-Claw)

When you launch through the [Hyperloom UI](https://crusoe.example-internal-host.invalid/hyperloom/),
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

**Q: I don't have a Cursor account. Will optimization still work?**

Yes. The default kernel-opt ladder is `forge,geak,claude,codex,cursor`,
but `cursor` is auto-dropped when `CURSOR_API_KEY` is unset. The run
proceeds with the remaining backends.

**Q: Where do `GEAK_API_KEY` / `ANTHROPIC_API_KEY` come from?**

In single-gateway mode they are derived from `SAFE_API_KEY` by
`install.sh` and preflight. In split mode each side uses its own key.
Set `GEAK_BASE_URL` / `OOB_BASE_URL` explicitly only when you need a
separate routable endpoint.

**Q: My organization rotates the LLM gateway key weekly. How?**

Re-export the key(s) and re-run `install.sh` (idempotent). Preflight
and all aliases pick up the new value on the next CLI launch.

## More info

Use these resources for related configuration and reference information:

* [Environment variables](environment-variables.md) — Every environment variable read by the code, including non-credential tunables.
* [Troubleshooting Hyperloom](troubleshooting.md) — Common 401 / gateway / Ray-GPU symptoms.
