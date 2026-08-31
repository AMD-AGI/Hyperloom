---
myst:
    html_meta:
        "description": "Learn how to configure the optional knowledge base integration in Hyperloom. Covers local Recipe KB setup, KB Store remote writes, resolver order, and degraded-mode behavior."
        "keywords": "Hyperloom, knowledge base, KB, recipe KB, gbrain, local KB, LLM inference, AMD GPU, ROCm, optimization, warm-start, session, configuration"
---
# Integrate Recipe knowledge base in Hyperloom

This topic explains the optional recipe knowledge-base (KB) integration used by Hyperloom, how local and remote stores are selected, and
how the runtime behaves when KB sources are unavailable. The KB is optional;
Hyperloom can run in local-only or degraded mode.

Hyperloom uses a recipe-snapshot KB and selects exactly one store:

| KB path | Owner / process | Purpose |
|---------|-----------------|---------|
| Local recipe KB | `inference_optimizer` | Selected by local/default mode; reads and writes durable Recipe JSON, history, and attempts. |
| Remote Recipe KB | KB Store | Selected only by `KNOWLEDGE_STORE_MODE=remote`; reads the current Recipe View for warm replay and writes one final session at CLOSE. |

Ambient KB Store or GBrain credentials do not select remote mode.

---

## Start a run

You don't need to configure a remote KB to start. By default, Hyperloom writes a
local recipe KB under:

```text
$USER_DATA_PATH/knowledge
```

If `USER_DATA_PATH` is unset, the default is
`~/.cache/hyperloom/knowledge`.

To choose an explicit local path:

```bash
export KNOWLEDGE_LOCAL_ROOT=/path/to/hyperloom-knowledge
```

The older `HYPERLOOM_LOCAL_KB_ROOT` and `--local-kb-root` forms remain
deprecated compatibility inputs.

To force a run without KB hooks:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --degraded-kb ...
```

---

## Local recipe KB

Local/default mode uses this resolver order:

1. `KNOWLEDGE_LOCAL_ROOT` when set
2. deprecated `--local-kb-root` or `HYPERLOOM_LOCAL_KB_ROOT`
3. `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge`

When no root is explicit, the first upgraded startup performs a one-time copy
of legacy Recipes from `$USER_DATA_PATH/kb`, or `/workspace/hyperloom/kb` when
`USER_DATA_PATH` is unset. It does not copy lock/temp files and does not replace
existing destination Recipes. See the [upgrade guide](upgrade.md) for the
stop, backup, and rollout procedure.

The store uses a nested on-disk layout keyed by recipe canonical-id components.
Treat the directory as Hyperloom-owned; use the CLI/runtime APIs rather than
editing files manually.

Recommended locations:

| Setup | Suggested path |
|-------|----------------|
| Single-user pod | `$USER_DATA_PATH/knowledge` |
| Shared persistent mount | `/shared/hyperloom/knowledge` |
| Hosted Primus-Claw sandbox | Platform-managed; don't override unless instructed. |

---

## Remote Recipe KB (KB Store)

Remote mode writes the final session to KB Store:

```bash
export KNOWLEDGE_STORE_MODE=remote
export KB_STORE_URL=https://your-kb-store
export KB_STORE_TOKEN=...
```

Both credentials are required; missing credentials fail at startup. Remote mode
selects metadata through
`GET /v1/kb/{canonical_id}/views/hyperloom-recipe` with all five scope query
parameters: `kernel_optimizer` (`forge` or `geak`), `tp`, `conc`, `isl`, and
`osl`. For example:

```text
/v1/kb/{canonical_id}/views/hyperloom-recipe?kernel_optimizer=forge&tp=8&conc=64&isl=1024&osl=256
```

The same scope is required for scoped session rollup reads and is included in
Recipe writes and champion updates. Hyperloom derives it from the current
session state; if the optimizer is unsupported or a numeric dimension is not
positive, remote warm-start is skipped with `recipe_scope_invalid` instead of
failing the optimization run.

Remote mode uses `/v1/kb/search` for bounded seven-tuple fallback. It downloads
the selected session's exact file manifest and replays one combined Recipe:
the Config column, the ordered Patch column overlays, and the Kernel column's
GEMM/Fusion/Rewrite content. Remote mode does not construct the local Recipe
dispatcher or fall back to local Recipe data. Runtime amendments are skipped
and CLOSE performs one best-effort final write. Optional `GBRAIN_*` credentials
remain available for Framework PR capabilities.

Configuration replay requires an exact precision match. A bf16 run does not
select an fp16 record, or vice versa, during degraded warm-start search. If an
accepted owner patch disappears before staging, that owner section moves to
the durable dead letter and CLOSE still publishes the final config, other
owner sections, and Kernel knowledge.

Records written before the unified Recipe contract are not rewritten in place.
An incompatible record is skipped during View validation; a later successful
CLOSE publishes the current document and artifacts.

Graceful teardown and Ctrl-C retry an unfinished CLOSE write through the T4
fallback. No in-process hook can run after SIGKILL, container force-deletion,
host loss, or interpreter failure; preserving CLOSE-only knowledge across those
failures requires the platform to resume the durable session and finalize it.

---

## Runtime behavior

When KB enrichment is unavailable, Hyperloom continues the optimization loop.
The warm-start context might be empty and cross-run priors might be weaker, but
baseline, profile/roofline, explore, kernel optimization, sweep, and report
still run.

The runtime records KB state in session artifacts so downstream consumers can
tell whether a run used local, remote, or degraded KB mode.

---

## FAQ

These questions cover common knowledge base configuration scenarios.

**Q: Should I set `INFERENCE_OPTIMIZER_KB_ROOT=skip`?**

No. That variable belongs to the retired JSONL KB path and isn't read by the
current runtime. Use `--degraded-kb` to skip KB hooks, or leave KB flags unset to
use the default local store.

**Q: Does a missing remote KB fail the run?**

In local/default mode GBrain is not consulted. In remote mode missing
credentials fail configuration and write failures are surfaced; select local
mode or `--degraded-kb` rather than relying on an implicit fallback.

**Q: Can I back up the KB?**

Yes. Back up the directory selected by `--local-kb-root` or
`HYPERLOOM_LOCAL_KB_ROOT`; otherwise back up `KNOWLEDGE_LOCAL_ROOT` or the
default `$USER_DATA_PATH/knowledge`.

**Q: Do I need the KB for a first run?**

No. First-run correctness is unaffected. You might see less reuse of historical
optimization knowledge until the local store accumulates data.
