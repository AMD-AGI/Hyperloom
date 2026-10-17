---
myst:
    html_meta:
        "robots": "noindex"
orphan: true
---

# KnowledgePlane phase 1

Phase 1 puts Recipe KB and the KernelForge knowledge control-plane behind one
Hyperloom `KnowledgePlane`. PR Monitor is co-hosted by KB Store: one
`KB_STORE_URL` supplies the Recipe API, PR REST API, and specialist MCP
endpoint. Framework Doc, GEMM tuning, Read A, and Read C are unchanged.

## Configuration

| Variable | Contract |
| --- | --- |
| `KNOWLEDGE_STORE_MODE` | `local` or `remote`; unset defaults to `local` |
| `KNOWLEDGE_LOCAL_ROOT` | Local knowledge root. Defaults to `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` |
| `KB_STORE_URL` | KB Service endpoint. Local Recipe mode defaults to `https://global.primus-safe.amd.com/knowledge-base` for PR Monitor only. Remote Recipe mode requires an explicit value. PR REST and MCP are derived as `/pr-monitor/v1` and `/pr-monitor/mcp/`. |
| `KB_STORE_TOKEN` | Required only in `remote` mode |

Unknown modes fail configuration. Remote mode fails before use if either
credential is missing. `--degraded-kb` remains a complete Recipe KB opt-out; it
does not select local mode.

This is a breaking mode cutover. Recipe remote mode now requires `KB_STORE_URL`
and `KB_STORE_TOKEN`. GBrain credentials do not satisfy or select Recipe remote
mode.

Local Recipe mode is local-only. Remote Recipe mode writes one final Recipe
session to KB Store at CLOSE; neither mode performs an inline mirror or dual
write.

PR Monitor configuration is orthogonal to Recipe storage mode. Unless
`--degraded-pr` is set, Framework discovery, KernelForge PR priors, IR-3, and
specialist tools all use the co-hosted PR surface derived from `KB_STORE_URL`.
The former independent PR endpoint flags and Cortex endpoint variable are
unsupported. In local Recipe mode an unset URL uses the global KB Service
default. IR-3 disables PR Monitor when that endpoint is unreachable while
leaving local Recipe storage enabled; remote Recipe mode never defaults write
configuration.

## Recipe backend behavior

Local mode uses `LocalRecipeStore`, including T0 warm replay, runtime
lesson/pitfall amendment, atomic JSON live rows, history, and
per-canonical-id file locks. Ambient KB Store credentials never trigger a
remote write.

Remote mode does not construct a `RecipeKB` dispatcher. It selects a current
Recipe View, performs bounded seven-tuple identity search when an exact record
is unavailable, and downloads the selected session's verified file manifest.
PRELUDE replays the Config column, the ordered Patch column overlays, and the
Kernel column in one validation task. Runtime legacy amendments are no-ops,
and CLOSE calls the KB Store final-session writer once. No local `recipe.json`
or history is created. A KB Store transport failure is logged and
remains non-fatal; invalid or missing startup configuration fails before the
optimization run starts.

## Audit and secrets

Recipe read/write audit events include `mode`, `backend`, resolution, compact
result metadata, and failure class. Tokens, credentials, and complete patches
are excluded.
