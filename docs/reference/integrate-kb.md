---
myst:
    html_meta:
        "description": "Learn how to configure the optional knowledge base integration in Hyperloom. Covers local recipe KB setup, remote Cortex KB, resolver order, and degraded-mode behavior."
        "keywords": "Hyperloom, knowledge base, KB, recipe KB, Cortex KB, local KB, LLM inference, AMD GPU, ROCm, optimization, warm-start, session, configuration"
---
# Integrate Recipe/Cortex knowledge base in Hyperloom

This topic explains the optional Recipe/Cortex knowledge-base (KB) integration used by Hyperloom, how local and remote stores are selected, and
how the runtime behaves when KB sources are unavailable. The KB is optional;
Hyperloom can run in local-only or degraded mode.

Hyperloom uses a recipe-snapshot KB. Three distinct stores are involved; they
serve different purposes and are configured independently:

| KB path | Owner / process | Purpose |
|---------|-----------------|---------|
| Local recipe KB | `inference_optimizer` | Always the write target for recipes, attempts, and session-derived optimization knowledge. Root resolved by `--local-kb-root` / `HYPERLOOM_LOCAL_KB_ROOT`. |
| Remote gbrain recipe KB (optional) | gbrain page store | Read side of the recipe KB. Consulted for warm-start reads when `GBRAIN_BASE_URL` / `GBRAIN_TOKEN` are set. Writes never go here directly (an out-of-band CronJob ingests the local store unless `RECIPE_KB_MIRROR_MODE=inline`). |
| Remote Cortex KB (optional) | Cortex KB service | Separate from the recipe KB. Used only by the Critic agent's per-proposal assess enrichment (`/v2/reasoning/assess`) when `--cortex-kb-url` / `CORTEX_KB_URL` is set. |

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
python3 -m hyperloom.inference_optimizer.cli optimize --local-kb-root /path/to/hyperloom-kb ...
```

To force a run without KB hooks:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --degraded-kb ...
```

---

## Local recipe KB

The local store is always the write target. The resolver order is:

1. `--local-kb-root` (or `HYPERLOOM_LOCAL_KB_ROOT`) when set
2. otherwise `workspace_root()/kb` — that is, `$USER_DATA_PATH/kb`, falling back to
   `/workspace/hyperloom/kb` when `USER_DATA_PATH` is unset

The store uses a nested on-disk layout keyed by recipe canonical-id components.
Treat the directory as Hyperloom-owned; use the CLI/runtime APIs rather than
editing files manually.

Recommended locations:

| Setup | Suggested path |
|-------|----------------|
| Single-user pod | `$USER_DATA_PATH/kb` |
| Shared persistent mount | `/wekafs/hyperloom/kb` |
| Hosted Primus-Claw sandbox | Platform-managed; don't override unless instructed. |

---

## Remote recipe KB reads (gbrain)

The read side of the recipe KB is the gbrain page store, not Cortex. It is
enabled only when gbrain is configured; writes always stay local.

```bash
export GBRAIN_BASE_URL=https://your-gbrain
export GBRAIN_TOKEN=...
```

When gbrain is unset or unreachable, the dispatcher degrades to local-only
reads. Use `--degraded-kb` to skip all KB hooks deliberately. By default
(`RECIPE_KB_MIRROR_MODE=external`) an out-of-band CronJob ingests the local
store into gbrain; set `RECIPE_KB_MIRROR_MODE=inline` to best-effort mirror each
local write in-process (the local write stays authoritative either way).

## Remote Cortex KB (Critic assess only)

The Cortex KB is a *separate* service from the recipe KB. It is consulted only
by the Critic agent's per-proposal assess enrichment (`/v2/reasoning/assess`),
never for recipe reads or writes. It is enabled only when you explicitly pass a
URL:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --cortex-kb-url https://your-cortex-kb ...
```

or export:

```bash
export CORTEX_KB_URL=https://your-cortex-kb
```

Leave it unset to skip Critic assess enrichment entirely. Recipe reads/writes
are unaffected by this setting.

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

**Q: Does a missing remote KB fail the run?**

No. An unset or unreachable gbrain recipe read side degrades to local-only
operation, and an unset Cortex KB skips Critic assess enrichment.
`--degraded-kb` skips the recipe KB path intentionally.

**Q: Can I back up the KB?**

Yes. Back up the directory selected by `--local-kb-root` or
`HYPERLOOM_LOCAL_KB_ROOT`; otherwise back up `$USER_DATA_PATH/kb`.

**Q: Do I need the KB for a first run?**

No. First-run correctness is unaffected. You might see less reuse of historical
optimization knowledge until the local store accumulates data.
