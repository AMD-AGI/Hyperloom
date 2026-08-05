# KnowledgePlane Phase 1

Phase 1 puts Recipe KB and the KernelForge knowledge control-plane behind one
Hyperloom `KnowledgePlane`. PR Monitor remains available through its existing
gating path, but is not moved into the new local/remote data plane. Framework
Doc, GEMM tuning, Read A, and Read C are unchanged.

## Configuration

| Variable | Contract |
| --- | --- |
| `KNOWLEDGE_STORE_MODE` | `local` or `remote`; unset defaults to `local` |
| `KNOWLEDGE_LOCAL_ROOT` | Local knowledge root; in remote mode its `.remote-locks/recipes` child stores lock metadata only. Defaults to `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` |
| `GBRAIN_BASE_URL` | Required only in `remote` mode |
| `GBRAIN_TOKEN` | Required only in `remote` mode |

Unknown modes fail configuration. Remote mode fails before use if either
credential is missing. `--degraded-kb` remains a complete Recipe KB opt-out; it
does not select local mode.

`RECIPE_KB_MIRROR_MODE` is deprecated and ignored by this plane. Local mode is
local-only. Remote mode reads and writes GBrain directly; neither mode performs
an inline mirror or external ingest.

The older `--local-kb-root` and `HYPERLOOM_LOCAL_KB_ROOT` inputs remain a
deprecated compatibility path when `KNOWLEDGE_LOCAL_ROOT` is absent.

## Recipe backend behavior

Local mode uses `LocalRecipeStore`, including its atomic JSON live row, history,
attempt log, and per-canonical-id file lock. Ambient GBrain credentials are not
used and no GBrain client is constructed.

Remote mode uses `GbrainRecipeStore`. A write updates the stable canonical
GBrain page directly and propagates transport/business failures to the caller.
New pages include a complete arbor Recipe JSON payload for CLOSE/T0/warm-replay
parity. Existing canonical pages without that payload continue through the
legacy page parser.

Remote writes serialize each canonical page with a SHA-256-named file under
`$KNOWLEDGE_LOCAL_ROOT/.remote-locks/recipes`. The lock spans the latest GBrain
read, store-side merge, and `put_page`, and combines a process-local thread lock
with POSIX `fcntl` locking. In remote mode this directory contains lock metadata
only: no `recipe.json`, history, or attempts are written locally.

All Hyperloom pods that can write remote Recipes must therefore mount the same
`KNOWLEDGE_LOCAL_ROOT` on a shared, read-write-many (RWX), POSIX-lock-capable
filesystem. Startup fails when `fcntl` is unavailable, and writes fail clearly
when the lock directory/file cannot be safely created or locked. A pod-local
volume does not provide cross-pod serialization.

Recipe identity, T0 matching tiers, champion selection, and warm-replay policy
are unchanged.

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
2. strips GBrain credentials from local-mode child environments;
3. forwards validated GBrain credentials in remote mode;
4. derives `KERNELFORGE_GBRAIN_ENABLED` from the selected mode, ignoring a user
   value; and
5. collects bounded capability/result provenance returned by KernelForge.

KernelForge continues to own local knowledge. Hyperloom does not implement
kernel-experience CRUD or ranking.

## Audit and secrets

Recipe read/write audit events include `mode`, `backend`, resolution, compact
result metadata, and failure class. Kernel experience passthrough records the
same mode/backend fields and bounded provenance. Tokens, credentials, complete
patches, and full KernelForge payloads are excluded.

## Remote concurrency limitation

GBrain's page API still has no compare-and-swap revision token. The shared file
lock prevents lost updates only when every remote Recipe writer uses this
Hyperloom store and the same working POSIX lock mount. Writers that bypass the
store, use different lock roots, or run where filesystem locks are not coherent
can still overwrite a canonical page.
