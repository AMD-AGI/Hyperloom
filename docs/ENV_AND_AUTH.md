# Authentication & Environment Guide

> **Scope.** This is the single authoritative reference for credentials and
> environment configuration in Hyperloom. If any other document
> (`README.md`, `inference_optimizer/SKILL.md`, `kernel-agent/SKILL.md`,
> `robustness-agent/SKILL.md`) appears to contradict this page, this page
> wins. Please open an issue against the contradicting file.

Hyperloom needs at most three classes of secrets:

1. The **AMD primus-safe LLM gateway** key (`SAFE_API_KEY`) — required.
2. The **Cursor SDK** key (`CURSOR_API_KEY`) — optional, only needed if
   you want the OOB `cursor` kernel-opt backend.
3. **Path environment** (`REPO_ROOT`, `OOB_SRC`, `INFERENCEX_PATH`,
   `USER_DATA_PATH`) — required for local mode. `TRACELENS_ROOT` is
   optional: leave it unset to let `install.sh` auto-clone the public
   repo into `$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens`; export
   it only as an explicit operator override.

Everything else (GEAK keys, OOB Claude/Codex keys, Anthropic / OpenAI
aliases, auth-proxy rewrites) is **derived** from `SAFE_API_KEY` and
`OPENAI_BASE_URL` by `kernel-agent/scripts/install.sh`. You should never
need to set them by hand.

---

## 1. Credential precedence

Hyperloom reads credentials from two places, in this order:

| Source                                      | When used                                                 |
|---------------------------------------------|-----------------------------------------------------------|
| Process environment (`export FOO=...`)      | Always — wins unconditionally.                            |
| `$REPO_ROOT/.env`                           | Only for keys **missing** from the process environment.   |

> **Hard rule.** Shell environment variables **always** win over `.env`.
> Both `inference_optimizer/cli.py` (`_maybe_load_env_file_into_environ`)
> and `kernel-agent/scripts/install.sh` honour this rule. Do **not**
> manually `source .env` from chat — it inverts the precedence and can
> overwrite an exported `SAFE_API_KEY` with a stale value from disk.

A key is considered "set" in `.env` if the line is uncommented and the
value is non-empty. Empty / commented lines are ignored; the
shell-exported value (if any) is kept.

If neither source supplies `SAFE_API_KEY`, the CLI fails fast at startup
with a message naming the missing variable.

---

## 2. Required credentials

| Variable           | Issuer                              | Where to obtain                                                                                       | Format              |
|--------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| `SAFE_API_KEY`     | AMD LiteLLM gateway                 | [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway)                                     | `ak-...`            |
| `OPENAI_BASE_URL`  | AMD LiteLLM gateway                 | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` (default for the hosted SaFE setup)          | URL                 |

`SAFE_API_KEY` is the single AMD credential used by **all** downstream
tooling:

* GEAK reads it as `GEAK_API_KEY` (auto-aliased).
* OOB `claude` and `codex` backends read it as `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` (auto-aliased via the auth-proxy).
* Robustness-agent uses it as `llm_api_key` for the optional LLM RCA
  engine.
* Critic-agent uses it for KB summary / synthesis calls.

You **never** need to copy `SAFE_API_KEY` into separate
`GEAK_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` slots in `.env`.
If those variables are unset, the install scripts fill them from
`SAFE_API_KEY` at process-launch time.

### Setting it

The recommended pattern, run once per shell:

```bash
cd "$REPO_ROOT"
cp .env.template .env
# Edit .env and set SAFE_API_KEY=ak-...
```

For one-off use without writing to disk:

```bash
export SAFE_API_KEY=ak-your-safe-apikey
export OPENAI_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
```

---

## 3. Optional credentials

### `CURSOR_API_KEY` — Cursor SDK kernel-opt backend

The OOB `cursor` backend talks to **Cursor's** gateway, not the AMD
primus-safe gateway. It therefore requires a **separate** issuer key
with prefix `crsr_...`:

| Variable                | Default                  | Description                                                                                                                                                                       |
|-------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CURSOR_API_KEY`        | unset                    | Cursor SDK key. **Never** inherited from `SAFE_API_KEY`. When unset, Hyperloom auto-drops `cursor` from the default kernel-opt ladder (`geak,claude,codex`) and continues without it. |
| `CURSOR_DEFAULT_MODEL`  | `claude-opus-4-7`        | Override the default Cursor model id.                                                                                                                                              |

The selection notes carry `cursor_key_present: bool` for observability.
If you pass `--backends cursor` explicitly without a key set, the
attempt is still launched and surfaces as a single failed `cursor`
attempt with a 401, rather than a silent skip.

### LLM RCA in robustness-agent

`robustness-agent`'s LLM root-cause-analysis engine activates **only**
when both `OPENAI_BASE_URL` and `SAFE_API_KEY` are set (they normally
are). Set `ROBUSTNESS_LLM_RCA_DISABLED=1` to force-disable it even when
credentials are present.

---

## 4. Path environment

These are *not* secrets, but they are required for local mode. The
installer and the agent use them to wire together the local stack.

| Variable           | Required for                                       | Default                | Description                                                                                                  |
|--------------------|----------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`        | Always                                             | `$(pwd)` if invoked from the repo root | This Hyperloom checkout. Locates `inference_optimizer/`, `kernel-agent/`, `.env`, skills, scripts.            |
| `OOB_SRC`          | OOB kernel-opt backends (claude / codex / cursor)  | none                   | Path to the `OOB/` subdirectory of the [Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) clone.            |
| `INFERENCEX_PATH`  | Baseline comparison, target analysis               | none                   | Path to the [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo.                    |
| `TRACELENS_ROOT`   | Profile & kernel detection                         | `$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens` (auto-clone) | When unset, `kernel-agent/scripts/install.sh` clones [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) here and pins it to a fixed SHA. Export it to point at an existing checkout (e.g. legacy `/wekafs/hyperloom/TraceLens-internal`) as an operator override — this skips both the clone and the SHA pin. |
| `TRACELENS_INTERNAL_ROOT` (optional) | Internal extension (roofline gap, MI355+ MAF) | none | Path to your own internal TraceLens extension checkout (internal users only; self-provided). Unset => open-source-only. Hyperloom never clones it. |
| `USER_DATA_PATH`   | Session artefacts (logs, runs, mirrors, breakdown) | `/workspace/hyperloom` | Writable directory. Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH` variables.    |

> **Migration note.** `WORKSPACE_PATH` and `INFERENCE_OPTIMIZER_SESSION_DIR`
> are **no longer read**. Rename launchers that still export them to
> `USER_DATA_PATH`. See [UPGRADING.md](UPGRADING.md).

---

## 5. The OOB auth-proxy (port 4002)

The `kernel-agent/scripts/install.sh` script starts a small local
auth-proxy on `127.0.0.1:4002`, supervised by
`scripts/ensure_auth_proxy.sh`. It rewrites the upstream `x-api-key`
header to `Authorization: Bearer <SAFE_API_KEY>` so the AMD
primus-safe gateway accepts requests from the `claude` and `codex`
npm CLIs. Without the proxy, every Claude/Codex CLI call returns
HTTP 401 / `Primus.00009 token not present`.

| Variable           | Default | Description                                                          |
|--------------------|---------|----------------------------------------------------------------------|
| `AUTH_PROXY_PORT`  | `4002`  | Bind port for the auth-proxy. Change only if 4002 is occupied.       |

The proxy is **not** used by:

* The `cursor` backend (talks to Cursor's own gateway directly).
* The `geak` backend (uses `GEAK_API_KEY` / `GEAK_BASE_URL` directly).

Recovery: if a tool fails with HTTP 401, re-run
`bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"`. The
supervisor is idempotent; it TCP-probes `:4002`, then HTTP-probes via
`curl`, and only restarts a stuck proxy.

---

## 6. Hosted mode (PrimusClaw)

When you launch through the [Hyperloom UI](https://core42.primus-safe.amd.com/hyperloom/),
**you do not need to set any of the variables above by hand**. The
sandbox initializer binds your LLM Gateway key as `SAFE_API_KEY`,
populates the path env from sandbox defaults, and starts the
auth-proxy automatically. See the root README's "Quickstart — Hyperloom
UI" section.

---

## 7. Quick FAQ

**Q: Do I have to put my key in `.env`?**
No. Exporting `SAFE_API_KEY` in your shell is sufficient. `.env` is a
convenience for persistence between shells.

**Q: I exported `SAFE_API_KEY` but `.env` has a different value. Which wins?**
The exported (shell) value wins. `.env` only fills missing keys.

**Q: I don't have a Cursor account. Will optimization still work?**
Yes. The default kernel-opt ladder silently drops `cursor` and races
`geak,claude,codex` only.

**Q: Where do `GEAK_API_KEY` / `ANTHROPIC_API_KEY` come from?**
They are derived from `SAFE_API_KEY` by `kernel-agent/scripts/install.sh`
and by `inference_optimizer/cli.py` at start-up. Do not set them in
`.env`.

**Q: My organization rotates the LLM gateway key weekly. How?**
Re-export `SAFE_API_KEY` and re-run `install.sh` (idempotent). The
auth-proxy and all aliases pick up the new value.

---

## 8. See also

* [Configuration reference](CONFIGURATION_REFERENCE.md) — every
  environment variable read by the code, including non-credential
  tunables.
* [KB guide](KB_GUIDE.md) — how to obtain or skip the knowledge-base
  tree referenced by `INFERENCE_OPTIMIZER_KB_ROOT`.
* [Troubleshooting](TROUBLESHOOTING.md) — common 401 / auth-proxy /
  Ray-GPU symptoms.
