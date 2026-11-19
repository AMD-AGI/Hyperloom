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
`docs/reference/kernel-execution-path.md`) appears to contradict this
page, this page wins. Open an issue against the contradicting file.

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.
```

Hyperloom needs at most two classes of configuration:

- **LLM gateway credentials**: At least one provider side, configured with both
   its base URL and its own key (see
   [LLM gateway credentials](#llm-gateway-credentials)).
- **Path / workspace layout**: Run bare-metal setup from the installed
   Hyperloom target directory. You normally only set `USER_DATA_PATH`
   (writable artifact root; defaults to `/workspace/hyperloom` when `/workspace`
   is writable, otherwise to `session/` under the current directory); setup
   writes the runtime env files and updates `.env`.

Hyperloom never borrows one provider's key or endpoint for the other. A side is
configured by `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`, or by
`OPENAI_BASE_URL` + `OPENAI_API_KEY`, or both. A side you leave unset
disables the features that speak its protocol; a half-configured side fails
preflight rather than being silently completed from the other provider.

No key is derived for you under any other name. The one value the inference
optimizer CLI preflight still fills is `LLM_API_BASE`, which addresses an
OpenAI-protocol endpoint rather than authenticating to one, and which it takes
from the resolved OpenAI-side URL. `GEAK_API_KEY` / `GEAK_BASE_URL` are never
filled from either side: GEAK runs on the Anthropic side, so set them only to
point GEAK at something else.

---

## Credential precedence

Hyperloom reads credentials from two places, in order, and from a third on a
checkout another account can write:

| Source                                      | When used                                                 |
|---------------------------------------------|-----------------------------------------------------------|
| Process environment (`export FOO=...`)      | Always — wins unconditionally.                            |
| `$REPO_ROOT/.env`                           | Only for keys missing from the process environment.       |
| `$KERNEL_AGENT_ENV` (container-local)       | Only for the Anthropic-side credentials missing from the shell, when `.env` keeps none and lends none ([below](#a-checkout-another-account-can-write)). |

Shell environment variables always win over `.env`.
Both the inference optimizer CLI dotenv loader and
`src/hyperloom/inference_optimizer/assets/install.sh` honor this rule. Do *not*
manually `source .env` from chat — it inverts the precedence and can
overwrite an exported key with a stale value from disk.

A key is considered "set" in `.env` if the line is uncommented and the
value is non-empty. Empty / commented lines are ignored; the
shell-exported value (if any) is kept.

If neither source supplies a usable LLM endpoint (at least one of
`OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and at least one of
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` /
`DEEPSEEK_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN`), the CLI fails fast at startup
with a message naming the missing pieces.

Preflight also rejects a *mispaired* configuration: a base URL whose only key
belongs to the other provider, or a key whose only endpoint would come from the
other provider. Hyperloom would otherwise send that key to a foreign host. Give
each side its own `*_BASE_URL` and key, or drop the foreign key.

### A checkout another account can write

The Anthropic-side credentials `install.sh` persists (`ANTHROPIC_BASE_URL`,
`ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, and an `ANTHROPIC_CUSTOM_HEADERS`
you put in `.env`) stay out of the checkout's `.env` when another account can
write it: the checkout directory or `.env` is writable by a group or others,
belongs to another account, or `.env` is a link. A source directory bind-mounted
from shared storage, as the Slurm job's `--source-dir` is, is the usual case.
There another account could read or rewrite them, so `install.sh` writes them
into the container-local `kernel-agent.env.sh` (`KERNEL_AGENT_ENV`) alone. The
checkout is judged where it is, so one you reach through your own link (`~/hl ->
/scratch/you/hl`) is as private as the directory it leads to, and root (a
container started without `--user` on your bind-mounted checkout, as in
[Docker (source)](../install/install.md#docker-source)) takes the account that
owns the checkout as its own: a checkout only you can write keeps the
credentials in `.env` for root too. In a checkout another account can write,
`install.sh` also removes those an earlier install, you or another account left
in `.env`, with their retired spellings (`ANTHROPIC_AUTH_TOKEN`, `DEEPSEEK_*`,
`SAFE_API_KEY`, `AMD_LLM_API_KEY`, `LLM_GATEWAY_KEY`, `GEAK_API_KEY`), and every
other line such a `.env` does not lend (below): it rebuilds `.env` from the
assignments it lends alone, each written afresh as `NAME=VALUE`, and keeps no
comment, blank line or other byte of the old file. Its own write leaves a `.env`
of yours that the next reader may find private, so nothing another account
planted there may stay. The new `.env` is mode 0600 and belongs to the account
that ran `install.sh` (root, when root ran it, as every write of `.env` leaves
it); it is made beside `.env` and renamed over it, a link to a file is replaced
rather than written through, and a link to anything else is removed.

Such a `.env` lends no credential to any reader: a line there may be another
account's, and a gateway URL or headers of its own would carry your key to it.
`install.sh` and the CLI preflight skip every one of those keys in it, with a
warning that names the file and never a value, so a credential you left only in a
shared `.env` is neither used nor copied into `kernel-agent.env.sh`: export it.

Readers that took the credentials from `.env` take them from that file instead,
after the shell: `install.sh` itself, so a re-install in the same
container keeps them, and the CLI preflight, which counts them as `.env`'s when
it decides whether a single provider side is configured. The documented `source`
line and the Slurm job load that file anyway. A new container has no such file,
so export the credentials in the shell that runs `install.sh` there; to rotate a
key, export the new one and re-run `install.sh`.

Nor does such a `.env` lend anything that would let that account run code as you
or see your traffic. `install.sh` and the CLI preflight take from it only a few
choices among values Hyperloom ships, each with one of its listed values:
`FRAMEWORK` (`atom`, `custom`, `sglang`, `vllm`, `xdit`), `HF_HUB_OFFLINE` and
`HF_DATASETS_OFFLINE` (`0`, `1`), `HYPERLOOM_RUN_MODE` (`baremetal`, `docker`),
`HYPERLOOM_LLM_MODE` (`anthropic`, `deepseek`, `openai`),
`HYPERLOOM_FRAMEWORK_ENV` (`isolated`, `shared`) and `HYPERLOOM_DOTENV_CREDENTIALS`
(`kernel-agent-env`).
Every other line is skipped: the loader and interpreter variables (`LD_PRELOAD`,
`PYTHONPATH`, `NODE_OPTIONS`, `BASH_ENV`, ...), the proxies, what an install
fetches or runs (`GEAK_REPO`, `GEAK_REF`, `VLLM_ROCM_INDEX`, the `*_REPO` names,
the interpreter and `*_ROOT` paths), where your key goes (`OPENAI_BASE_URL`,
`OPENAI_CUSTOM_HEADERS`, `GEAK_BASE_URL`), the model names, the tuning knobs
and `USER_DATA_PATH`. Each skipped key is named in a warning, never its value;
export any you need in the shell. Such a `.env` never selects the workspace
either, even with a `HYPERLOOM_RUN_MODE` line: the `USER_DATA_PATH` you export
stands. The installers do not write skipped keys there: `install.sh` keeps them
in `kernel-agent.env.sh`, and `install_baremetal.sh` names each one it leaves
out. A `.env` only you can write is read as before. `install_baremetal.sh`
follows the same rules: on such a checkout it reads none of those keys, and
none of the credentials, from `.env`, and writes the Anthropic-side credentials
to `kernel-agent.env.sh` alone; an OpenAI key it does not write anywhere.

`install.sh` leaves `HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env` in such a
`.env`, which is no secret. Its own writes replace a `.env` another account could
write, in a directory that is yours, with one of yours, and the line tells the
readers that the credentials are still in the container-local file. The next
install of that now-private checkout persists them in `.env` as before and drops
the line. The OpenAI side, which `install.sh` does not write, stays where you put
it.

---

## LLM gateway credentials

A provider side is configured by **both** its base URL and its own key. Exactly
three shapes are accepted; anything else fails preflight.

| Shape | Set | Effect |
|-------|-----|--------|
| OpenAI side only | `OPENAI_BASE_URL` + `OPENAI_API_KEY` | Codex runs; Claude and GEAK are disabled |
| Anthropic side only | `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` | Claude and GEAK run; Codex (OpenAI-protocol) is disabled |
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

* GEAK runs Claude Code, so it uses the Anthropic-side base URL + key plus
  `GEAK_CLAUDE_MODEL`. An OpenAI-only deployment cannot start it.
* Kernel tools inherit the OpenAI-side credential under its own name,
  `OPENAI_API_KEY`, which is what crosses the Ray boundary to a worker.
* Orchestration Claude uses the Anthropic-side base URL + key, including the
  generated `~/.claude/config.json` primary key.
* Critic-agent uses the OpenAI side for KB summary / synthesis calls.

There are no internal key slots in `.env` to fill: each side's credential
travels under its own name. Preflight does **not** cross-fill the per-provider
primary keys (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`),
and an explicitly set provider key is never overwritten.

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
`ANTHROPIC_AUTH_TOKEN`); GEAK uses the same Anthropic-side URL and key. Neither
side is ever completed from the other.

To pin models in split mode:

```bash
export CLAUDE_MODEL=orchestration-model-id-on-the-anthropic-side
export CODEX_MODEL=model-id-on-the-openai-side
```

### Anthropic-side credential forms

The Anthropic side accepts three credentials. When more than one is set, the
Claude CLI picks the first of these it finds, so a lower one is unused:

| Variable                  | Billing                        | Obtained from        |
|---------------------------|--------------------------------|----------------------|
| `ANTHROPIC_API_KEY`       | API credits                    | Anthropic console / your gateway |
| `ANTHROPIC_AUTH_TOKEN`    | API credits (gateway bearer)   | Your gateway         |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Max/Pro subscription    | `claude setup-token` |

A subscription token alone is a complete Anthropic side: it implies
`https://api.anthropic.com`, so `ANTHROPIC_BASE_URL` can stay unset. Hyperloom
passes it through verbatim and never copies it into either API-key variable or
into the `~/.claude/config.json` `primaryApiKey` field, because the CLI would
then bill API credits instead of the subscription. For the same reason preflight
warns when a token and an API key are set together. A subscription token is also
not accepted by the gateway model-catalog probe, so preflight skips that probe
and trusts the model id you passed.

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
stricter AMD Claude allowlist (`claude-opus-5-5` preferred, `claude-opus-5` /
`claude-opus-4-8` / `claude-opus-4-7` / `claude-opus-4-6` fallback), set
`INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`.

A gateway that exposes no `/models` route cannot be probed; preflight warns and
proceeds instead of refusing to start, so no opt-in is needed for that case.

### Gateways that require extra headers

Some enterprise gateways authenticate on a header of their own — Azure API
Management's `Ocp-Apim-Subscription-Key`, a corporate proxy's tenant header — in
addition to, or instead of, the bearer key. AMD's own `llm-api.amd.com` gateway
is one of them: it requires the API key to also travel as an
`Ocp-Apim-Subscription-Key` header, which the guided setup writes for you when
you pick that gateway. Supply it per side with `ANTHROPIC_CUSTOM_HEADERS` or
`OPENAI_CUSTOM_HEADERS`. Both accept the Anthropic SDK's newline-delimited
`Name: value` form, and a JSON object as a convenience for launchers that already
store structured values.

`${VAR}` references are expanded by Hyperloom from the same environment, so a
single copy of the secret is enough. Quote the value so the space and the colon
survive: use single quotes when exporting in a shell, and double quotes in
`.env`, which setup and the launch scripts might load with `source`.

```bash
export ANTHROPIC_BASE_URL=https://<your-gateway-host>/api/v1/llm-proxy
export ANTHROPIC_API_KEY=ak-your-gateway-apikey
export ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}'
```

The Anthropic side reads only `ANTHROPIC_CUSTOM_HEADERS`. The OpenAI side reads
`OPENAI_CUSTOM_HEADERS` whenever you set `OPENAI_BASE_URL` explicitly — that
endpoint might be a different host, whose headers Hyperloom will not guess at. Only
when the OpenAI base URL was *derived* from `ANTHROPIC_BASE_URL` (one gateway, no
explicit OpenAI endpoint) does the OpenAI client fall back to
`ANTHROPIC_CUSTOM_HEADERS`, because then both protocols are the same host and the
subscription header was only ever written to the Anthropic variable. Headers are
operator-supplied either way and are never synthesized from the configured keys.

Both survive in `.env` across setup runs. An Anthropic-only deployment is the one
exception: setup scrubs the whole OpenAI side there, header included, because that
side is a second provider rather than part of the same gateway credential.

---

## Path environment

These are *not* secrets. You normally do not hand-export
`TRACELENS_ROOT` or `INFERENCE_SUITE_TOOL_CACHE` — `install.sh` and its chained
kernel-agent installer clone, stage and pin the open-source tools when missing.
Runtime paths are persisted into `.env` by the installer; the CLI preflight
loads them and derives `PATH` / `LD_LIBRARY_PATH` from `ROCM_PATH` /
`VIRTUAL_ENV` / `VLLM_VENV_ROOT` at launch, so no separate file needs sourcing.

### Workspace variables

| Variable               | Set by operator? | Default                                          | Description                                                                                                      |
|------------------------|------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `USER_DATA_PATH`       | recommended      | `/workspace/hyperloom` if `/workspace` is writable, else `<cwd>/session` | Writable root for session dirs, `runtime/`, `logs/`, optimizer artifacts. Bare-metal hosts without a writable `/workspace` take the second form, which is also what setup offers. |
| `REPO_ROOT`            | rarely           | auto-detected from script location               | This Hyperloom checkout. Locates `.env`, skills, scripts.                                                        |
| `KERNEL_AGENT_ENV`     | rarely           | `${TMPDIR:-/tmp}/hyperloom-<uid>-<key>/kernel-agent.env.sh` | Output of `install.sh`; exports resolved paths, LLM aliases and the Anthropic credentials. Kept on the container's own disk in a mode-`0700` directory, never under `USER_DATA_PATH`; see [Environment variables](environment-variables.md). |
| `HYPERLOOM_RUNTIME_DIR`| rarely           | `$USER_DATA_PATH/runtime`                        | Shared runtime tree (GEAK config, Recipe KB bookkeeping); no credentials.                                        |

### Dependency checkout variables

Open-source dependencies default under `HYPERLOOM_CACHE_DIR`
(`$REPO_ROOT/.cache`), cloned per revision as `<name>@<sha>`. The cache is
repo-local and writable, so open-source runs need no privileged `/opt` mount.

Leave these variables unset unless you maintain your own checkouts. An
explicit path pointing at a missing directory fails preflight.

| Variable                     | Set by operator? | Default / auto-clone target                                | Description                                                                                                         |
|------------------------------|------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_CACHE_DIR`        | rarely           | `$REPO_ROOT/.cache`                                        | Writable, repo-local root for open-source deps: TraceLens and GEAK, cloned per revision as `<name>@<sha>`, and the InferenceSuite tool cache. |
| `INFERENCEX_PATH`            | optional         | unset                                                      | An [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) checkout of your own, read for reference launch scripts. Benchmark rounds run the client staged in `INFERENCE_SUITE_TOOL_CACHE` instead; nothing clones it. |
| `TRACELENS_ROOT`             | optional override | `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens@<sha>` | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) for profiling and kernel detection; the kernel-agent installer clones and pins it when unset. |
| `TRACELENS_INTERNAL_ROOT`    | optional         | unset (MAF measured on-device)                             | Optional internal TraceLens extension that backfills MAF without an on-device benchmark. When unset, Hyperloom measures MAF on an idle GPU (microbenchmark) — roofline gap / MI355+ MAF analysis is still produced, just measured locally. In a multi-branch session the microbenchmark runs on a card leased from the toolbox, waiting for one and holding back new tests while the tests need every card; a toolbox with no card pool of its own measures nothing and the roofline reads vendor peaks. Hyperloom never clones it. |
| `INFERENCE_SUITE_TOOL_CACHE` | rarely            | `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/inference-suite-tools`| Tool cache holding the pinned InferenceX client, GSM8K task and AgentX harness that every benchmark round reads; `install.sh` stages it.                |

---

## Direct upstream wiring

Claude, Codex, and GEAK talk to the configured upstream gateway directly.
The AMD primus-safe gateway accepts both header styles natively.

At preflight, the inference optimizer CLI:

- Resolves Anthropic and OpenAI base URLs.
- Writes `~/.claude/config.json`: `customApiUrl` is set to the resolved
  Anthropic-side base URL, and `primaryApiKey` to the Anthropic-side key
  (explicit `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` wins, else
  `OPENAI_API_KEY`). A `CLAUDE_CODE_OAUTH_TOKEN` is never written here.
- Sets `LLM_API_BASE` to the resolved OpenAI-side URL. No key is filled: the
  per-provider primary keys and the GEAK aliases all stay as the operator left
  them.

**401 recovery:**

1. Confirm `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and the matching
   API key are set and current.
2. Re-run preflight (any `python -m hyperloom optimize ...` command) or
   `bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh" --check-only`.
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
convenience for persistence between shells. On a checkout another account can
write, `install.sh` keeps the Anthropic-side credentials out of `.env`; see
[A checkout another account can write](#a-checkout-another-account-can-write).

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
