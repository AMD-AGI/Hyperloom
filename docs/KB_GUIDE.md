# Knowledge-Base Guide

> **Scope.** This page explains the optional recipe knowledge-base (KB)
> integration used by Hyperloom, how the local store and the gbrain read-side
> remote are selected, and how the runtime behaves when KB sources are
> unavailable. The KB is optional; Hyperloom can run in local-only or degraded
> mode.

Hyperloom uses a **recipe-snapshot KB**:

| KB path | Owner / process | Purpose |
|---------|-----------------|---------|
| Local recipe KB | `inference_optimizer` | Durable local writes for recipes, attempts, and session-derived optimization knowledge. |
| Remote gbrain KB (optional) | gbrain page store | Optional read-side enrichment when `GBRAIN_BASE_URL` / `GBRAIN_TOKEN` are configured. |

The old `INFERENCE_OPTIMIZER_KB_ROOT` JSONL store is retired. Current code does
not read it; use `HYPERLOOM_LOCAL_KB_ROOT` or `--local-kb-root` instead.

---

## 1. TL;DR — I just want to start a run

You do not need to configure a remote KB to start. By default, Hyperloom writes a
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

## 2. Local recipe KB

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
| Hosted PrimusClaw sandbox | Platform-managed; do not override unless instructed. |

---

## 3. Remote gbrain KB

Hyperloom never silently connects to a remote KB. Remote recipe reads are served
by the gbrain page store and are enabled only when you configure gbrain:

```bash
export GBRAIN_BASE_URL=https://your-gbrain
export GBRAIN_TOKEN=...
```

Writes still go to the local recipe KB. If gbrain is unreachable, the dispatcher
degrades to local-only behaviour; use `--degraded-kb` to skip KB hooks
deliberately.

> **Note.** `--cortex-kb-url` / `CORTEX_KB_URL` are NOT the recipe KB. They
> configure the Cortex KB used only by the Critic agent's per-proposal assess
> enrichment (`/v2/reasoning/assess`). Recipe reads always go to gbrain.

---

## 4. Runtime behaviour

When KB enrichment is unavailable, Hyperloom continues the optimization loop.
The warm-start context may be empty and cross-run priors may be weaker, but
baseline, profile/roofline, explore, kernel optimization, sweep, and report
still run.

The runtime records KB state in session artifacts so downstream consumers can
tell whether a run used local-only, remote-enriched, or degraded KB mode.

---

## 5. Quick FAQ

**Q: Should I set `INFERENCE_OPTIMIZER_KB_ROOT=skip`?**
No. That variable belongs to the retired JSONL KB path and is not read by the
current runtime. Use `--degraded-kb` to skip KB hooks, or leave KB flags unset to
use the default local store.

**Q: Does a missing remote gbrain KB fail the run?**
No. An explicitly configured but unreachable remote degrades to local-only
operation. `--degraded-kb` skips the KB path intentionally.

**Q: Can I back up the KB?**
Yes. Back up the directory selected by `--local-kb-root` or
`HYPERLOOM_LOCAL_KB_ROOT`; otherwise back up `$USER_DATA_PATH/kb`.

**Q: Do I need the KB for a first run?**
No. First-run correctness is unaffected. You may see less reuse of historical
optimization knowledge until the local store accumulates data.

---

## 6. See also

* [ENV_AND_AUTH.md](ENV_AND_AUTH.md) — credentials and path env.
* [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) — all environment
  variables, including the KB-related ones.
* [HOW_THE_OPTIMIZATION_LOOP_WORKS.md](HOW_THE_OPTIMIZATION_LOOP_WORKS.md) —
  how warm-start context participates in the optimization loop.
