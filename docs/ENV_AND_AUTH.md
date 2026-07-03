# Authentication & Environment Guide

> **Scope.** This is the single authoritative reference for credentials and
> environment configuration in Hyperloom. If any other document
> (`README.md`, `inference_optimizer/SKILL.md`, `kernel-agent/SKILL.md`,
> `robustness-agent/SKILL.md`) appears to contradict this page, this page
> wins. Please open an issue against the contradicting file.
>
> 中文版：[ENV_AND_AUTH.zh-CN.md](ENV_AND_AUTH.zh-CN.md)

Hyperloom needs at most three classes of configuration:

1. **LLM gateway credentials** — at least one upstream base URL and one API
   key (see §2). On the AMD network the usual pair is `SAFE_API_KEY` +
   `OPENAI_BASE_URL`; split Anthropic/OpenAI entrypoints and third-party
   gateways are also supported.
2. The **Cursor SDK** key (`CURSOR_API_KEY`) — optional, only needed if
   you want the OOB `cursor` kernel-opt backend.
3. **Path / workspace layout** — for local mode, run
   `inference_optimizer/scripts/local_setup.sh` once (credentials + Hyperloom
   checkout are enough). It clones missing dependency repos under
   `HYPERLOOM_OPEN_SOURCE_ROOT` (default `/opt/hyperloom/open-source-repos`),
   writes `$USER_DATA_PATH/runtime/local-setup.env.sh`, and exports
   `OOB_SRC`, `INFERENCEX_PATH`, `TRACELENS_ROOT`, etc. You normally only
   set `USER_DATA_PATH` (writable artefact root; default
   `/workspace/hyperloom`). `REPO_ROOT` is auto-detected. Explicit overrides
   for dependency paths are optional (§4).

In the **single-gateway** setup, GEAK keys, OOB Claude/Codex keys, and
Anthropic / OpenAI aliases are **derived** from `SAFE_API_KEY` and
`OPENAI_BASE_URL` by `kernel-agent/scripts/install.sh` and
`inference_optimizer/cli.py` at preflight. You normally do not set those
aliases by hand. Split-gateway and GEAK/OOB endpoint overrides are the
exceptions (§2.3, §3.3).

---

## 1. Credential precedence

Hyperloom reads credentials from two places, in this order:

| Source                                      | When used                                                 |
|---------------------------------------------|-----------------------------------------------------------|
| Process environment (`export FOO=...`)      | Always — wins unconditionally.                            |
| `$REPO_ROOT/.env`                           | Only for keys **missing** from the process environment.   |

> **Hard rule.** Shell environment variables **always** win over `.env`.
> Both `inference_optimizer/cli.py` (`_load_dotenv_fallback`) and
> `kernel-agent/scripts/install.sh` honour this rule. Do **not**
> manually `source .env` from chat — it inverts the precedence and can
> overwrite an exported key with a stale value from disk.

A key is considered "set" in `.env` if the line is uncommented and the
value is non-empty. Empty / commented lines are ignored; the
shell-exported value (if any) is kept.

If neither source supplies a usable LLM endpoint (at least one of
`OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` **and** at least one of
`SAFE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`), the CLI fails fast at startup with a message
naming the missing pieces.

---

## 2. LLM gateway credentials

Two deployment shapes are supported. Pick one; do not mix the blocks
unless you intend a split entrypoint (mode B).

### 2.1 Single gateway (default — AMD / LiteLLM-style)

One OpenAI-compatible endpoint serves both Claude and GPT models
(LiteLLM gateway, AMD primus-safe, or any self-hosted `/v1` proxy).

| Variable           | Issuer                              | Where to obtain                                                                                       | Format              |
|--------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| `SAFE_API_KEY`     | AMD LiteLLM gateway (typical)         | [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway)                                     | `ak-...`            |
| `OPENAI_BASE_URL`  | Same gateway                        | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` (default for hosted SaFE)                  | URL ending in `/v1` |

Preflight derives `ANTHROPIC_BASE_URL` from `OPENAI_BASE_URL`, points
`~/.claude/config.json` `customApiUrl` at the upstream gateway, and
fills any **unset** alias keys from `SAFE_API_KEY`:

* GEAK → `GEAK_API_KEY` / `GEAK_BASE_URL`
* OOB `claude` / `codex` → `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
* Critic / Robustness LLM calls

You **never** need to copy `SAFE_API_KEY` into separate
`GEAK_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` slots in `.env`
for this mode. If those variables are unset, install + preflight fill
them at process-launch time. An explicitly set provider key is **never**
overwritten by `SAFE_API_KEY`.

**Recommended setup:**

```bash
cd "$REPO_ROOT"
cp .env.template .env
# Edit .env and set SAFE_API_KEY=ak-...
```

One-off (no disk write):

```bash
export SAFE_API_KEY=ak-your-safe-apikey
export OPENAI_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
```

### 2.2 Split entrypoints (native Anthropic + OpenAI)

Use when Claude and GPT live on **different** upstream vendors or
gateways. Set each side explicitly; `SAFE_API_KEY` is optional.

| Variable              | Side     | Example                                      |
|-----------------------|----------|----------------------------------------------|
| `ANTHROPIC_BASE_URL`  | Claude   | `https://api.anthropic.com`                  |
| `ANTHROPIC_API_KEY`   | Claude   | `sk-ant-...`                                 |
| `OPENAI_BASE_URL`     | Codex/GPT| `https://api.openai.com/v1`                    |
| `OPENAI_API_KEY`      | Codex/GPT| `sk-...`                                     |

Preflight resolves `(anthropic_base_url, openai_base_url)` independently:
when both are set, each is kept as-is. Claude CLI auth uses
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in preference to any
shared key. GEAK and OOB Codex inherit the OpenAI-side URL and key.

### 2.3 Non-AMD / self-hosted gateway (#340)

Point `OPENAI_BASE_URL` and `SAFE_API_KEY` (or the split keys above) at
your own LiteLLM-compatible gateway (Vultr, TensorWave, on-prem, etc.).
Then opt out of the AMD-only orchestration model gate and pin models your
gateway actually serves:

```bash
export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
export CLAUDE_MODEL=your-gateway-orchestration-model
export CODEX_MODEL=your-gateway-kernel-model
```

With the opt-out set, preflight validates model ids against your
gateway's `/models` catalog instead of hard-gating to AMD opus-4-7/4-6.

---

## 3. Optional credentials

### 3.1 `CURSOR_API_KEY` — Cursor SDK kernel-opt backend

The OOB `cursor` backend talks to **Cursor's** gateway, not your LLM
gateway. It requires a **separate** issuer key with prefix `crsr_...`:

| Variable                | Default                  | Description                                                                                                                                                                       |
|-------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CURSOR_API_KEY`        | unset                    | Cursor SDK key. **Never** inherited from `SAFE_API_KEY`. `cursor` is the tail of the default `forge,geak,claude,codex,cursor` ladder but is auto-dropped when this key is unset; set it to keep `cursor` in the ladder. |
| `CURSOR_DEFAULT_MODEL`  | `claude-opus-4-7`        | Override the default Cursor model id.                                                                                                                                              |

The selection notes carry `cursor_key_present: bool` for observability.
If you pass `--backends cursor` explicitly without a key set, the
attempt is still launched and surfaces as a single failed `cursor`
attempt with a 401, rather than a silent skip.

### 3.2 LLM RCA in robustness-agent

`robustness-agent`'s LLM root-cause-analysis engine activates when an
LLM base URL and API key are available (normally via the aliases above).
Set `ROBUSTNESS_LLM_RCA_DISABLED=1` to force-disable it even when
credentials are present.

### 3.3 GEAK / OOB endpoint override (#521)

GEAK Ray workers may run in a network namespace that cannot reach the
main gateway directly. By default `GEAK_BASE_URL` and `OOB_BASE_URL`
inherit the resolved OpenAI-compatible gateway URL. Set them explicitly
when you need a host-local reverse tunnel or another routable path:

```bash
export GEAK_BASE_URL=https://127.0.0.1:18444/api/v1/llm-proxy/v1
export OOB_BASE_URL=https://127.0.0.1:18444/api/v1/llm-proxy/v1
```

Preflight **preserves** intentional operator overrides. It only
force-rewrites URLs that still point at the removed legacy local proxy
(`127.0.0.1:4002`). GEAK reads its endpoint from the generated litellm
yaml (`$GEAK_CONFIG`); preflight syncs `base_url:` there when the
resolved gateway changes.

---

## 4. Path environment

These are *not* secrets. In local mode you normally **do not** hand-export
`OOB_SRC`, `INFERENCEX_PATH`, or `TRACELENS_ROOT` — `local_setup.sh`
clones them when missing and records the resolved paths in
`local-setup.env.sh`. Before launching optimization, source that file (and
set `USER_DATA_PATH` if you want a non-default workspace):

```bash
export USER_DATA_PATH=/path/to/hyperloom-run   # optional; default /workspace/hyperloom
bash inference_optimizer/scripts/local_setup.sh
source "$USER_DATA_PATH/runtime/local-setup.env.sh"
```

`kernel-agent/scripts/install.sh` (invoked by preflight or manually) then
installs runtime deps (Ray, GEAK, OOB CLIs, Magpie, etc.) and writes
`$USER_DATA_PATH/runtime/kernel-agent.env.sh`. Source that too when driving
kernel-agent tools directly.

### 4.1 Workspace (operator-facing)

| Variable                  | Set by operator? | Default                          | Description                                                                 |
|---------------------------|------------------|----------------------------------|-----------------------------------------------------------------------------|
| `USER_DATA_PATH`          | recommended      | `/workspace/hyperloom`           | Writable root for session dirs, `runtime/`, `logs/`, optimizer artefacts. Replaces retired `WORKSPACE_PATH` / `INFERENCE_OPTIMIZER_SESSION_DIR`. |
| `REPO_ROOT`               | rarely           | auto from script location        | This Hyperloom checkout (`.env`, skills, scripts).                          |
| `LOCAL_SETUP_ENV`         | rarely           | `$USER_DATA_PATH/runtime/local-setup.env.sh` | Output of `local_setup.sh`; source in every fresh shell.          |
| `KERNEL_AGENT_ENV`        | rarely           | `$USER_DATA_PATH/runtime/kernel-agent.env.sh` | Output of `install.sh`; exports resolved paths + LLM aliases.   |
| `HYPERLOOM_RUNTIME_DIR`   | rarely           | `$USER_DATA_PATH/runtime`        | Shared runtime tree (env files, GEAK config, Cortex KB bookkeeping).          |

### 4.2 Dependency checkouts (auto-provisioned)

Open-source dependencies default under **`HYPERLOOM_OPEN_SOURCE_ROOT`**
(`/opt/hyperloom/open-source-repos`), decoupled from `USER_DATA_PATH` so
shared WekaFS session storage does not collocate concurrent pods' clones.
`local_setup.sh` accepts the alias **`HYPERLOOM_DEPS_ROOT`** (or
`--deps-root`); it exports the resolved value as
`HYPERLOOM_OPEN_SOURCE_ROOT`.

| Variable                  | Set by operator? | Default / auto-clone target      | Description                                                                 |
|---------------------------|------------------|----------------------------------|-----------------------------------------------------------------------------|
| `HYPERLOOM_OPEN_SOURCE_ROOT` | rarely        | `/opt/hyperloom/open-source-repos` | Pod-local root for TraceLens, InferenceX, KernelForge, Magpie, GEAK, OOB. Writable `/opt` required unless overridden. |
| `OOB_SRC`                 | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/KernelForge/OOB` (via `local_setup.sh`) | OOB kernel-opt backends (claude / codex / cursor). Derived from [KernelForge](https://github.com/AMD-AGI/KernelForge) checkout. |
| `INFERENCEX_PATH`         | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/InferenceX` | [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) for baseline / target analysis. Preflight can also clone here when unset. |
| `TRACELENS_ROOT`          | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/TraceLens` | [TraceLens](https://github.com/AMD-AGI/TraceLens) for profile & kernel detection; pinned to a fixed SHA on auto-clone. |
| `TRACELENS_INTERNAL_ROOT` | optional         | unset (open-source-only)         | Internal TraceLens extension (roofline gap, MI355+ MAF). Hyperloom never clones it — set only when you maintain a checkout. |
| `MAGPIE_PATH`             | optional override | `${HYPERLOOM_OPEN_SOURCE_ROOT}/Magpie` | Magpie benchmark wrappers; installed by `install.sh` when missing.          |

Leave dependency vars **unset** unless you maintain your own checkouts.
An explicit `TRACELENS_ROOT` / `INFERENCEX_PATH` pointing at a missing
path fails preflight (issue #722 guard).

> **Migration note.** `WORKSPACE_PATH` and `INFERENCE_OPTIMIZER_SESSION_DIR`
> are **no longer read**. Rename launchers that still export them to
> `USER_DATA_PATH`. See [UPGRADING.md](UPGRADING.md).

---

## 5. Direct upstream wiring (no local auth-proxy)

Older Hyperloom builds ran a local auth-proxy on `127.0.0.1:4002` to
rewrite `x-api-key` into `Authorization: Bearer`. **That component has
been removed.** Claude, Codex, and GEAK now talk to the upstream gateway
directly. The AMD primus-safe gateway accepts both header styles natively.

At preflight, `inference_optimizer/cli.py`:

* Resolves Anthropic and OpenAI base URLs (§2).
* Writes `~/.claude/config.json` `customApiUrl` and `primaryApiKey`.
* Fills unset alias env vars (`GEAK_*`, `OOB_*`, etc.).
* Force-rewrites any leftover URL still pinned at `127.0.0.1:4002` to
  the real upstream gateway (stale installs only).

**401 recovery (current):**

1. Confirm `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and the matching
   API key are set and current.
2. Re-run preflight (any `inference_optimizer` CLI command) or
   `bash "$REPO_ROOT/kernel-agent/scripts/install.sh" --check-only`.
3. Inspect `~/.claude/config.json` — `customApiUrl` must point at the
   upstream gateway, not `127.0.0.1:4002`.

The `cursor` backend always bypasses your LLM gateway (Cursor's own
issuer). GEAK uses `GEAK_API_KEY` / `GEAK_BASE_URL` (or the generated
litellm config) directly.

---

## 6. Hosted mode (PrimusClaw)

When you launch through the [Hyperloom UI](https://core42.primus-safe.amd.com/hyperloom/),
**you do not need to set any of the variables above by hand**. The
sandbox initializer binds your LLM Gateway key as `SAFE_API_KEY`,
populates the path env from sandbox defaults, and runs install/preflight
so downstream tools inherit the gateway URL and aliases. See the root
README's "Quickstart — Hyperloom UI" section.

---

## 7. Quick FAQ

**Q: Do I have to put my key in `.env`?**
No. Exporting credentials in your shell is sufficient. `.env` is a
convenience for persistence between shells.

**Q: I exported a key but `.env` has a different value. Which wins?**
The exported (shell) value wins. `.env` only fills missing keys.

**Q: Can I run without `SAFE_API_KEY`?**
Yes, in split-entrypoint mode (§2.2): set `ANTHROPIC_API_KEY` +
`OPENAI_API_KEY` (and the matching base URLs) instead.

**Q: I don't have a Cursor account. Will optimization still work?**
Yes. The default kernel-opt ladder is `forge,geak,claude,codex,cursor`
(Forge-GEAK-OOB), but `cursor` is auto-dropped from it when `CURSOR_API_KEY` is
unset, so the run proceeds with `forge,geak,claude,codex`.

**Q: Where do `GEAK_API_KEY` / `ANTHROPIC_API_KEY` come from?**
In single-gateway mode they are derived from `SAFE_API_KEY` by
`install.sh` and preflight. In split mode each side uses its own key.
Set `GEAK_BASE_URL` / `OOB_BASE_URL` explicitly only when you need a
separate routable endpoint (#521).

**Q: My organization rotates the LLM gateway key weekly. How?**
Re-export the key(s) and re-run `install.sh` (idempotent). Preflight
and all aliases pick up the new value on the next CLI launch.

**Q: I still see `127.0.0.1:4002` in my env or Claude config.**
That is a stale legacy proxy URL. Run any optimize CLI command (preflight
rewrites it) or delete the stale `customApiUrl` from
`~/.claude/config.json` and re-run install.

---

## 8. See also

* [Configuration reference](CONFIGURATION_REFERENCE.md) — every
  environment variable read by the code, including non-credential
  tunables.
* [KB guide](KB_GUIDE.md) — local recipe KB and optional Cortex KB setup.
* [Troubleshooting](TROUBLESHOOTING.md) — common 401 / gateway /
  Ray-GPU symptoms.
