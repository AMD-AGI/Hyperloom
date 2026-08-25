---
myst:
    html_meta:
        "robots": "noindex"
orphan: true
---

# KnowledgePlane phase 1

Phase 1 puts Recipe KB and the KernelForge knowledge control-plane behind one
Hyperloom `KnowledgePlane`. PR Monitor remains available through its existing
gating path, but is not moved into the new local/remote data plane. Framework
Doc, GEMM tuning, Read A, and Read C are unchanged.

## Configuration

| Variable | Contract |
| --- | --- |
| `KNOWLEDGE_STORE_MODE` | `local` or `remote`; unset defaults to `local` |
| `KNOWLEDGE_LOCAL_ROOT` | Local knowledge root. Defaults to `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` |
| `KB_STORE_URL` | Required only in `remote` mode |
| `KB_STORE_TOKEN` | Required only in `remote` mode |
| `GBRAIN_BASE_URL` / `GBRAIN_TOKEN` | Optional non-Recipe configuration for GBrain KG and Framework PR capabilities |

Unknown modes fail configuration. Remote mode fails before use if either
credential is missing. `--degraded-kb` remains a complete Recipe KB opt-out; it
does not select local mode.

This is a breaking mode cutover. Recipe remote mode now requires `KB_STORE_URL`
and `KB_STORE_TOKEN`. GBrain credentials do not satisfy or select Recipe remote
mode.

Local mode is local-only. Remote mode writes one final Recipe session to KB
Store at CLOSE; neither mode performs an inline mirror or dual write.

The older `--local-kb-root` and `HYPERLOOM_LOCAL_KB_ROOT` inputs remain a
deprecated compatibility path when `KNOWLEDGE_LOCAL_ROOT` is absent.

## One-time local Recipe migration

The default local root is always `$USER_DATA_PATH/knowledge`, or
`~/.cache/hyperloom/knowledge` when `USER_DATA_PATH` is unset. It never
permanently falls back to the old `$USER_DATA_PATH/kb` or
`/workspace/hyperloom/kb` roots.

On local Recipe KB startup only, when `KNOWLEDGE_LOCAL_ROOT`,
`--local-kb-root`, and `HYPERLOOM_LOCAL_KB_ROOT` are all unset, Hyperloom checks
the corresponding legacy root. If the new destination has neither Recipe data
nor the durable migration marker, complete Recipe directories (live
`recipe.json`, history, attempts, and safe metadata) are copied into the new
root. Lock and temporary files are excluded. Existing graph or KernelForge
data in the new root is retained. Existing destination Recipes or a completed
marker make the operation a no-op; explicit legacy roots continue in place and
are not migrated.

Migration failure while legacy Recipes exist aborts startup with a clear error
instead of silently cold-starting. Stop old writers and back up the legacy root
before upgrading; see [Upgrade Hyperloom version](reference/upgrade.md).

## Recipe backend behavior

Local mode uses `LocalRecipeStore`, including T0 warm replay, runtime
lesson/pitfall amendment, atomic JSON live rows, history, attempt logs, and
per-canonical-id file locks. Ambient KB Store credentials never trigger a
remote write.

Remote mode does not construct a `RecipeKB` dispatcher. It selects a current
Recipe View, performs bounded seven-tuple identity search when an exact record
is unavailable, and downloads the selected session's verified file manifest.
PRELUDE replays the combined config, ordered Explore/Framework overlays, and
Kernel section in one validation task. Runtime legacy amendments are no-ops,
and CLOSE calls the KB Store final-session writer once. No local `recipe.json`,
history, or attempt data is created. A KB Store transport failure is logged and
remains non-fatal; invalid or missing startup configuration fails before the
optimization run starts.

## Local knowledge graph

Local mode also constructs `LocalGraphStore`, an in-process filesystem
GraphStore backend adapter for `KGClient`. It does not start or connect to a
local MCP server and performs no HTTP, network, or subprocess work. Ambient
GBrain credentials are ignored.

Its durable root is `$KNOWLEDGE_LOCAL_ROOT/hyperloom/kg`:

```text
hyperloom/kg/
├── pages/<slug>.md
├── edges/outbound/<slug>.json
├── edges/inbound/<slug>.json
├── .lock
└── .edge-transaction.json  # present only while recovering an edge update
```

Slugs are validated before path construction, including every component of a
slash-separated slug; absolute paths, `.`/`..`, backslashes, empty components,
and unsupported characters are rejected. Pages and edge indexes use
fsync-plus-atomic-rename replacement. A module-level per-root thread lock and a
POSIX `fcntl` lock serialize all instances and processes. `add_link` updates
the outbound and inbound indexes under one exclusive lock, replacing context
for an existing `(from, to, type)` edge. A durable intent journal completes
both index replacements after interruption.

An empty first-run graph is available but returns no facts, so T0 and warm
replay continue with Recipe KB results. Framework fact emission materializes
entity pages and writes local edges through the same `KGClient` API.

## KernelForge bridge

Hyperloom's `KernelExperienceBridge` only:

1. validates and forwards the shared mode/root;
2. forwards validated KB Store credentials in Recipe remote mode and strips
   them in local mode;
3. keeps optional GBrain credentials in the Hyperloom parent for KG/Framework
   PR clients, but never forwards them to KernelForge children;
4. forces the legacy `KERNELFORGE_GBRAIN_ENABLED` Recipe-derived flag off; and
5. collects bounded capability/result provenance returned by KernelForge.

KernelForge continues to own local knowledge. Hyperloom does not implement
kernel-experience CRUD or ranking.

## Audit and secrets

Recipe read/write audit events include `mode`, `backend`, resolution, compact
result metadata, and failure class. Kernel experience passthrough records the
same mode/backend fields and bounded provenance. Tokens, credentials, complete
patches, and full KernelForge payloads are excluded.
