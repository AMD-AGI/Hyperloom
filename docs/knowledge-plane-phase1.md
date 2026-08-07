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

This is a breaking mode cutover. Deployments that previously relied on ambient
`GBRAIN_BASE_URL` / `GBRAIN_TOKEN` for remote-first Recipe reads must now set
`KNOWLEDGE_STORE_MODE=remote`. The default/local mode ignores those credentials
and remains local-only. `RECIPE_KB_MIRROR_MODE` is obsolete; remove it from
launchers, ConfigMaps, and secrets rather than using it to select behavior.

Local mode is local-only. Remote mode reads and writes GBrain directly; neither
mode performs an inline mirror or external ingest.

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

Recipe identity and T0 matching tiers are unchanged.

## Replay bundles and legacy Recipes

A replayable champion is one atomic value: the final effective server argv,
environment variables, source patches, measured throughput, and integrity
metadata all describe the same validated state. `best_config` remains a
compatibility projection of that bundle's config; a writer must not advance
`best_config` or `best_throughput` while carrying forward an older bundle.

Source modifications from every retained `source_patch` stack entry are
squashed per repository/base commit. Local Recipes keep those diffs inline;
remote Recipes externalize large diffs to content-addressed GBrain pages.
Both readers verify artifact hashes and the bundle digest before warm replay.
Environment values that point at a captured file are stored as
`{repo, path}` references and resolved only after that repository patch is
applied; uncaptured absolute paths make the bundle non-replayable.
Patch application is reversed patch-by-patch after benchmarking and never uses
`git reset --hard` or `git clean` on a shared framework checkout.
For replayed AITER C/C++/HIP sources, Hyperloom invokes KernelForge's
source-keyed JIT preparation before launch and fails closed when that integration
is unavailable; after reversing the patch it rekeys the cache to the restored
source state.

An authored-kernel overlay represented only by a host-local `PYTHONPATH`
directory is not portable. Until that overlay is flattened to a source patch
or resolved by a stable KernelForge artifact API, the Recipe is retained as
reference material with `replayable=false` and reason
`overlay_not_flattened_to_patch`.

Recipes written before replay bundles remain readable for history, lessons,
and anti-priors, but their unbound `best_config` is not executed
automatically. Operators will see `legacy_recipe_without_bundle` rather than a
generic empty-config reason. Backfilling such a row as executable requires an
explicit migration that can prove the config, patches, environment, and
throughput came from the same measured champion; copying only `best_config`
does not establish that guarantee.

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
