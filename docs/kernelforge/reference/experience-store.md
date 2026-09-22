---
myst:
  html_meta:
    "description": "Configure KernelForge experience storage for durable local files or remote KB Store."
---

# Knowledge stores

KernelForge persists forge-loop and rewrite candidates through one local/remote
store contract. This store is separate from the packaged
`local_knowledge` prompt tree; `local_knowledge` stays read-only in its existing
location and is never copied into the experience store.

`kernelforge gemm-tune` has no knowledge base: every run tunes or
authors from scratch and writes only its own output directory.

## Environment contract

| Variable | Default | Meaning |
|:--|:--|:--|
| `KNOWLEDGE_STORE_MODE` | `local` | Exactly `local` or `remote`. Other values fail validation. |
| `KNOWLEDGE_LOCAL_ROOT` | See below | Shared root for local knowledge data. |
| `KB_STORE_URL` | none | KB Store endpoint; required in `remote` mode. |
| `KB_STORE_TOKEN` | none | KB Store bearer token; required in `remote` mode. |

When `KNOWLEDGE_LOCAL_ROOT` is unset, its default is
`$USER_DATA_PATH/knowledge` if `USER_DATA_PATH` is present, otherwise
`~/.cache/hyperloom/knowledge`.

`local` mode never constructs a remote store client and ignores ambient
credentials. In `remote` mode, both KB Store values must be non-empty;
validation happens before `forge-loop` or `rewrite` starts.

GBrain credentials do not configure Forge storage. Runs that supplied only
`GBRAIN_BASE_URL` and `GBRAIN_TOKEN` previously continued without an experience
store; they now fail startup with the missing KB Store variable names. Set
`KB_STORE_URL` and `KB_STORE_TOKEN` for remote storage, or select `local` mode.
Python callers should pass `KnowledgeConfig` through `Config.knowledge_config`;
the obsolete `gbrain_url`, `gbrain_base_url`, `gbrain_token`, and `remote_backend`
arguments are no longer supported. Unknown `Config.from_env` override names
raise `TypeError`. Hyperloom's Framework PR GBrain client is independent.

## Local layout

KernelForge stores experiences below:

```text
$KNOWLEDGE_LOCAL_ROOT/
└── kernelforge/
    └── rewrite/
        └── <canonical identity segments>/
            ├── .lock
            ├── champion.json
            └── sessions/
                └── <session-id>/
                    ├── knowledge.json
                    └── files/
                        └── <candidate artifacts>
```

The canonical identity includes the producer, operator, framework, framework
version, backend, and GPU. Forge-loop records include `solution.patch` and a
human-readable summary; rewrite records include their kernel artifacts.
Writes use temporary files, fsync, atomic replacement, and a lock per identity,
so a persistent root is safe to reuse across runs and processes. This
configuration cleanup does not change the stored record layout or migrate
existing records.
