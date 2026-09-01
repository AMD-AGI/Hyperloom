---
myst:
  html_meta:
    "description": "Configure KernelForge experience storage for durable local files or remote GBrain."
---

# Knowledge stores

KernelForge persists forge-loop experience pages and `optimizes` links through
one local/remote store contract. This store is separate from the packaged
`local_knowledge` prompt tree; `local_knowledge` stays read-only in its existing
location and is never copied into the experience store.

`kernelforge gemm-tune` has no knowledge base: every run tunes or
authors from scratch and writes only its own output directory.

## Environment contract

| Variable | Default | Meaning |
|:--|:--|:--|
| `KNOWLEDGE_STORE_MODE` | `local` | Exactly `local` or `remote`. Other values fail validation. |
| `KNOWLEDGE_LOCAL_ROOT` | See below | Shared root for local knowledge data. |
| `GBRAIN_BASE_URL` | none | GBrain base URL; required in `remote` mode. |
| `GBRAIN_TOKEN` | none | GBrain bearer token; required in `remote` mode. |

When `KNOWLEDGE_LOCAL_ROOT` is unset, its default is
`$USER_DATA_PATH/knowledge` if `USER_DATA_PATH` is present, otherwise
`~/.cache/hyperloom/knowledge`.

`local` mode never constructs a GBrain client and ignores ambient
`GBRAIN_BASE_URL` and `GBRAIN_TOKEN` values. In `remote` mode, both GBrain values
must be non-empty; validation happens before `forge-loop` starts.

## Local layout

KernelForge stores experiences below:

```text
$KNOWLEDGE_LOCAL_ROOT/
└── kernelforge/
    └── experiences/
        ├── .store.lock
        ├── pages/
        │   └── kernelforge-exp/
        │       ├── <operator>__<framework>__<backend>.md
        │       └── <operator>__<framework>__<backend>/
        │           └── <experience-id>.md
        └── links/
            └── backlinks/
                └── kernelforge-exp/<operator>__<framework>__<backend>.json
```

Page slugs and page content remain the existing
`kernelforge-exp/<op>...` format. Backlink JSON preserves the existing
`solution -> kernel` `optimizes` semantics. Writes use same-directory temporary
files, fsync, atomic replacement, and a process lock, so a persistent root is
safe to reuse across runs and processes.
