---
myst:
    html_meta:
        "description": "Authoritative reference for Hyperloom credentials and environment configuration. Covers SAFE_API_KEY, CURSOR_API_KEY, path variables, the OOB auth-proxy, and hosted mode."
        "keywords": "Hyperloom, authentication, credentials, SAFE_API_KEY, CURSOR_API_KEY, auth-proxy, environment variables, OOB, LLM gateway, ROCm, AMD GPU, API key, configuration"
---
# Hyperloom authentication and credentials

This is the single authoritative reference for credentials and
environment configuration in Hyperloom. If any other document
(`README.md`, `inference_optimizer/SKILL.md`, `kernel-agent/SKILL.md`,
`robustness-agent/SKILL.md`) appears to contradict this page, this page wins. Please open an issue against the contradicting file.

Hyperloom needs at most three classes of secrets:

- The **AMD primus-safe LLM gateway** key (`SAFE_API_KEY`) — required.
- The **Cursor SDK** key (`CURSOR_API_KEY`) — optional, only needed if
   you want the OOB `cursor` kernel-opt backend.
- **Path environment** (`REPO_ROOT`, `OOB_SRC`, `INFERENCEX_PATH`,
   `USER_DATA_PATH`) — required at runtime for local mode. `local_setup.sh`
   and `install.sh` provision the dependency paths when they are unset.
   `TRACELENS_ROOT` is optional: leave it unset to let `install.sh`
   auto-clone the public repo into the pod-local open-source checkout
   root (`${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}`);
   export it only as an explicit operator override.

Everything else (GEAK keys, OOB Claude/Codex keys, Anthropic / OpenAI
aliases, auth-proxy rewrites) is **derived** from `SAFE_API_KEY` and
`OPENAI_BASE_URL` by `kernel-agent/scripts/install.sh`. You should never
need to set them by hand.

---

## Credential precedence

Hyperloom reads credentials from two places, in order:

| Source                                      | When used                                                 |
|---------------------------------------------|-----------------------------------------------------------|
| Process environment (`export FOO=...`)      | Always — wins unconditionally.                            |
| `$REPO_ROOT/.env`                           | Only for keys missing from the process environment.   |

Shell environment variables always win over `.env`.
Both `inference_optimizer/cli.py` (`_maybe_load_env_file_into_environ`)
and `kernel-agent/scripts/install.sh` honor this rule. Do **not**
manually `source .env` from chat — it inverts the precedence and can
overwrite an exported `SAFE_API_KEY` with a stale value from disk.

A key is considered "set" in `.env` if the line is uncommented and the
value is non-empty. Empty / commented lines are ignored; the
shell-exported value (if any) is kept.

If neither source supplies `SAFE_API_KEY`, the CLI fails fast at startup
with a message naming the missing variable.

---

## Required credentials

The following credentials are required for all Hyperloom runs.

| Variable           | Issuer                              | Where to obtain                                                                                       | Format              |
|--------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| `SAFE_API_KEY`     | AMD LiteLLM gateway                 | [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway)                                     | `ak-...`            |
| `OPENAI_BASE_URL`  | AMD LiteLLM gateway                 | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` (default for the hosted SaFE setup)          | URL                 |

`SAFE_API_KEY` is the single AMD credential used by all downstream
tooling:

* GEAK reads it as `GEAK_API_KEY` (auto-aliased).
* OOB `claude` and `codex` backends read it as `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` (auto-aliased using the auth-proxy).
* Robustness-agent uses it as `llm_api_key` for the optional LLM RCA
  engine.
* Critic-agent uses it for KB summary / synthesis calls.

You *never* need to copy `SAFE_API_KEY` into separate
`GEAK_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` slots in `.env`.
If those variables are unset, the install scripts fill them from
`SAFE_API_KEY` at process-launch time.

### Set the credentials

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

## Optional credentials

The following credentials are optional and only needed for specific backends.

### `CURSOR_API_KEY` — Cursor SDK kernel-opt backend

The OOB `cursor` backend talks to Cursor's gateway, not the AMD
primus-safe gateway. It therefore requires a separate issuer key
with prefix `crsr_...`:

| Variable                | Default                  | Description                                                                                                                                                                       |
|-------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CURSOR_API_KEY`        | unset                    | Cursor SDK key. Never inherited from `SAFE_API_KEY`. `cursor` is not in the default `forge,geak` ladder; include it explicitly in `KERNEL_OPT_BACKEND_ORDER` when you want to use it. |
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

## Path environment

These are *not* secrets. They must exist when the runtime starts, but the
standard local workflow provisions the dependency checkouts for you:
`local_setup.sh` resolves `OOB_SRC`, `INFERENCEX_PATH`, and `TRACELENS_ROOT`
and writes them into `local-setup.env.sh`; `install.sh` then installs and
verifies the runtime dependencies.

| Variable           | Required for                                       | Default                | Description                                                                                                  |
|--------------------|----------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`        | Always                                             | `$(pwd)` if invoked from the repo root | This Hyperloom checkout. Locates `inference_optimizer/`, `kernel-agent/`, `.env`, skills, scripts.            |
| `OOB_SRC`          | OOB kernel-opt backends (claude / codex / cursor)  | none                   | Path to the `OOB/` subdirectory of the [Primus-Claw](https://github.com/AMD-AGI/Primus-Claw) clone.            |
| `INFERENCEX_PATH`  | Baseline comparison, target analysis               | none                   | Path to the [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo.                    |
| `TRACELENS_ROOT`   | Profile & kernel detection                         | `${HYPERLOOM`<br>`_OPEN_SOU`<br>`RCE_ROOT`<br>`:-${TMP`<br>`DIR:-/tm`<br>`p}/hyperl`<br>`oom/open-so`<br>`urce-re`<br>`pos}/Tra`<br>`ceLens` (auto-clone) | When unset, `kernel-agent/scripts/install.sh` clones [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) into the pod-local open-source checkout root and pins it to a fixed SHA. Export it to point at a pre-existing checkout you maintain as an operator override — this skips both the clone and the SHA pin. |
| `TRACELENS_INTERNAL_ROOT` (optional) | Internal extension (roofline gap, MI355+ MAF) | none | Path to your own internal TraceLens extension checkout (internal users only; self-provided). Unset => open-source-only. Hyperloom never clones it. |
| `USER_DATA_PATH`   | Session artifacts (logs, runs, mirrors, breakdown) | `/workspace/hyperloom` | Writable directory. Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH` variables.    |

```{note}
`WORKSPACE_PATH` and `INFERENCE_OPTIMIZER_SESSION_DIR`
are no longer read. Rename launchers that still export them to
`USER_DATA_PATH`. See [Upgrade Hyperloom version](../reference/upgrade.md).
```

---

## The OOB auth-proxy (port 4002)

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

The proxy is not used by:

* The `cursor` backend (talks to Cursor's own gateway directly).
* The `geak` backend (uses `GEAK_API_KEY` / `GEAK_BASE_URL` directly).

Recovery: if a tool fails with HTTP 401, re-run
`bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"`. The
supervisor is idempotent; it TCP-probes `:4002`, then HTTP-probes using
`curl`, and only restarts a stuck proxy.

---

## Hosted mode (PrimusClaw)

When you launch through the [Hyperloom UI](https://core42.primus-safe.amd.com/hyperloom/),
you do not need to set any of the variables above by hand. The
sandbox initializer binds your LLM Gateway key as `SAFE_API_KEY`,
populates the path env from sandbox defaults, and starts the
auth-proxy automatically. See the root README's "Quickstart — Hyperloom
UI" section.

---

## FAQ

These questions cover common credential configuration scenarios.

**Q: Do I have to put my key in `.env`?**

No. Exporting `SAFE_API_KEY` in your shell is sufficient. `.env` is a
convenience for persistence between shells.

**Q: I exported `SAFE_API_KEY` but `.env` has a different value. Which wins?**

The exported (shell) value wins. `.env` only fills missing keys.

**Q: I don't have a Cursor account. Will optimization still work?**

Yes. The default kernel-opt ladder is `forge,geak`. `cursor` and other OOB
backends (`claude`, `codex`) are used only when you explicitly include them in
`KERNEL_OPT_BACKEND_ORDER`; `cursor` still requires `CURSOR_API_KEY`.

**Q: Where do `GEAK_API_KEY` / `ANTHROPIC_API_KEY` come from?**

They are derived from `SAFE_API_KEY` by `kernel-agent/scripts/install.sh`
and by `inference_optimizer/cli.py` at start-up. Do not set them in
`.env`.

**Q: My organization rotates the LLM gateway key weekly. How?**

Re-export `SAFE_API_KEY` and re-run `install.sh` (idempotent). The
auth-proxy and all aliases pick up the new value.

---

## More info

Use these resources for related configuration and reference information:

* [Environment variables](environment-variables.md) — Every environment variable read by the code, including non-credential tunables.
* [Integrate Recipe/Cortex knowledge base in Hyperloom](../reference/integrate-kb.md) — Local recipe KB and optional Cortex KB setup.
* [Troubleshooting Hyperloom](../troubleshooting.md) — Common 401 / auth-proxy / Ray-GPU symptoms.
