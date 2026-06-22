---
myst:
    html_meta:
        "description": "Learn how to configure the optional knowledge base integration in Hyperloom. Covers local recipe KB setup, remote Cortex KB, resolver order, and degraded-mode behavior."
        "keywords": "Hyperloom, knowledge base, KB, recipe KB, Cortex KB, local KB, LLM inference, AMD GPU, ROCm, optimization, warm-start, session, configuration"
---
# Integrate Recipe/Cortex knowledge base in Hyperloom

This topic explains the optional Recipe/Cortex knowledge-base (KB)integration used by Hyperloom, how local and remote stores are selected, and
how the runtime behaves when KB sources are unavailable. The KB is optional;
Hyperloom can run in local-only or degraded mode.

Hyperloom uses a recipe-snapshot KB:

| KB path | Owner / process | Purpose |
|---------|-----------------|---------|
| Local recipe KB | `inference_optimizer` | Durable local writes for recipes, attempts, and session-derived optimization knowledge. |
| Remote Cortex KB (optional) | Cortex KB service | Optional read-side enrichment when `--cortex-kb-url` / `CORTEX_KB_URL` is configured. |

The old `INFERENCE_OPTIMIZER_KB_ROOT` JSONL store is retired. Current code doesn't read it; use `HYPERLOOM_LOCAL_KB_ROOT` or `--local-kb-root` instead.

---

## Start a run

You don't need to configure a remote KB to start. By default, Hyperloom writes a
local recipe KB under:

```text
$USER_DATA_PATH/kb
```

If `USER_DATA_PATH` is unset, the default workspace root is
`/workspace/hyperloom`, so the local KB falls back to `/workspace/hyperloom/kb`.

To choose an explicit local path:

```bash
export HYPERLOOM_LOCAL_KB_ROOT=/path/to/hyperloom-kb
# or pass:
inference_optimizer optimize --local-kb-root /path/to/hyperloom-kb ...
```

To force a run without KB hooks:

```bash
inference_optimizer optimize --degraded-kb ...
```

---

## Local recipe KB

The local store is always the write target. The resolver order is:

1. `--local-kb-root`
2. `HYPERLOOM_LOCAL_KB_ROOT`
3. `$USER_DATA_PATH/kb`
4. `/workspace/hyperloom/kb`

The store uses a nested on-disk layout keyed by recipe canonical-id components.
Treat the directory as Hyperloom-owned; use the CLI/runtime APIs rather than
editing files manually.

Recommended locations:

| Setup | Suggested path |
|-------|----------------|
| Single-user pod | `$USER_DATA_PATH/kb` |
| Shared persistent mount | `/wekafs/hyperloom/kb` |
| Hosted PrimusClaw sandbox | Platform-managed; don't override unless instructed. |

---

## Remote Cortex KB

Hyperloom never silently connects to a remote KB. Remote reads are enabled only
when you explicitly pass a URL:

```bash
inference_optimizer optimize --cortex-kb-url https://your-cortex-kb ...
```

or export:

```bash
export CORTEX_KB_URL=https://your-cortex-kb
```

Writes still go to the local recipe KB. If the configured remote is unreachable,
the dispatcher degrades to local-only behavior; use `--degraded-kb` to skip KB
hooks deliberately.

---

## Runtime behavior

When KB enrichment is unavailable, Hyperloom continues the optimization loop.
The warm-start context might be empty and cross-run priors might be weaker, but
baseline, profile/roofline, explore, kernel optimization, sweep, and report
still run.

The runtime records KB state in session artifacts so downstream consumers can
tell whether a run used local-only, remote-enriched, or degraded KB mode.

---

## FAQ

These questions cover common knowledge base configuration scenarios.

**Q: Should I set `INFERENCE_OPTIMIZER_KB_ROOT=skip`?**

No. That variable belongs to the retired JSONL KB path and isn't read by the
current runtime. Use `--degraded-kb` to skip KB hooks, or leave KB flags unset to
use the default local store.

**Q: Does a missing remote Cortex KB fail the run?**

No. An explicitly configured but unreachable remote degrades to local-only
operation. `--degraded-kb` skips the KB path intentionally.

**Q: Can I back up the KB?**

Yes. Back up the directory selected by `--local-kb-root` or
`HYPERLOOM_LOCAL_KB_ROOT`; otherwise back up `$USER_DATA_PATH/kb`.

**Q: Do I need the KB for a first run?**

No. First-run correctness is unaffected. You might see less reuse of historical
optimization knowledge until the local store accumulates data.

---

## Related guides

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md) — Credentials and path env.
* [Environment variables](environment-variables.md) — All environment
  variables, including the KB-related ones.
* [Hyperloom optimization loop](../conceptual/optimization-loop.md) —
  How warm-start context participates in the optimization loop.
