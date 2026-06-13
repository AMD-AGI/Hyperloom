# Paths & Session Layout

State lives under a **session directory** (per optimization run). The
**workspace root** is `$USER_DATA_PATH` (default `/workspace/hyperloom`) — it
holds shared `runtime/` and `logs/`.

## Layout (N17 default: `per_model_ts`)

```text
$USER_DATA_PATH/                          # workspace_root — set by operator / Claw / SaFE
├── runtime/                              # workspace-shared (install.sh, Magpie, kernel-agent.env.sh)
│   ├── kernel-agent.env.sh
│   ├── geak-config/local.yaml
│   ├── Magpie/
│   └── source-mirrors/{Primus-Claw,OOB,InferenceX,TraceLens[,TraceLens-internal]}/
│       # TraceLens public is required; TraceLens-internal is optional and only
│       # present when TRACELENS_INTERNAL_ROOT is set (open-source-only otherwise)
├── logs/                                 # workspace-shared launcher stdout
└── <model_basename>/                     # e.g. DeepSeek-R1-0528, deepseek-ai-DeepSeek-V3
    └── <UTC_YYYYMMDDTHHMMSSZ>/           # session_dir — manifest.json, state.json, runs/, …
        ├── manifest.json
        ├── state.json
        ├── storage/coordinator.db
        ├── agents/{orchestration,kernel,critic,robustness}/
        ├── runs/{baseline,profile,roofline,explore,sweep,...}/<task_id>/
        ├── kernel-agent/runs/<session_id>/
        ├── kernel-agent-workspace/<kernel_id>/
        ├── optimizer_runs/               # per-session launcher logs / PID / monitor
        ├── reports/
        └── …
```

**Claw / SaFE pods:** the launcher often sets `$USER_DATA_PATH` to a
run-scoped path *before* the optimizer starts, e.g.
`/hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/`. That outer
directory is **platform isolation** (one Claw job). The optimizer then creates
`<model_basename>/<UTC_ts>/` inside it. Full session path example:

    /hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/   ← USER_DATA_PATH (Claw)
        deepseek-ai-DeepSeek-V3/20260522T035359Z/                      ← session_dir (optimizer)

**Legacy flat layout:** set `INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat` so
`session_dir == workspace_root` (no `<model>/<ts>` subdirs).

## Path resolution (do not guess)

`paths.py` is the single authority for Hyperloom paths. The launching agent
does not need to recreate that logic in shell; it only needs to run
`install.sh`, source the generated `runtime/kernel-agent.env.sh`, and read the
session dir printed by the CLI.

| Concept | Env / helper | Meaning |
|---|---|---|
| Workspace root | `$USER_DATA_PATH` → `paths.workspace_root()` | Shared `runtime/` + `logs/` and parent of all sessions |
| Session dir | `$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` → `paths.session_dir()` | Per-run directory containing `manifest.json` / `state.json` / `storage/coordinator.db` |

**Launcher rule:** do not hand-build, create, delete, or repair paths under
`$USER_DATA_PATH/runtime/` (especially `source-mirrors/`). Those are
workspace-shared assets owned by `install.sh`, including Magpie, GEAK, OOB,
TraceLens mirrors, env files, and config. Manual edits there can corrupt
another run's checkout. If install state looks wrong, rerun `install.sh` or
follow `troubleshooting.md`; do not clone or clean the mirrors by hand.

**Session rule:** never treat `$USER_DATA_PATH` as the session dir when
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` is set. Read `manifest.json` /
`state.json` / `coordinator.db` from the **session dir**. For monitoring after
launch, learn the session dir from the **launch-info JSON** written by
`--launch-info-file` (`jq -r .session_dir <file>`) or, equivalently, from the
single `HYPERLOOM_LAUNCH key=value …` sentinel line the CLI prints to stdout
(`session_dir=…`). Those are the authoritative, machine-readable sources.
Never guess by walking `$USER_DATA_PATH/<model_basename>/` for the latest
`*T*Z/` timestamp dir — overlapping sessions on the same host make "latest"
pick the wrong run.

### Inputs that stay outside `$USER_DATA_PATH` by design

Read-only sources or warm-start caches, each overridable via its own env if you
want a fully self-contained session:

- **TraceLens** — `$TRACELENS_ROOT` (default
  `$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens`; when unset,
  `kernel-agent/scripts/install.sh` clones
  [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) there and pins it
  to a fixed SHA. Export `TRACELENS_ROOT=<path>` only as an operator override
  to point at a pre-existing checkout you maintain; this skips both the clone
  and the SHA pin). Optional internal extension at `$TRACELENS_INTERNAL_ROOT`
  (no default; internal users set it to their own checkout to opt in,
  otherwise open-source-only). The per-version
  `sglang_roofline_patches/sglang_<minor>_<patch>/` layout under TraceLens is
  required by `_server_patcher`.
- `$OOB_SRC` / `$HYPERLOOM_BUNDLE`
- `/sgl-workspace/{aiter,sglang,vllm}/`
- `~/.claude/config.json` + `~/.codex/auth.json`
- `~/.cache/amd-ai-devtool/semantic-index/` (GEAK RAG embedding cache)
- `/wekafs/hyperloom/geak-memory/memory.db` (GEAK cross-session memory)

Paths emitted by agents must resolve under the **session dir** — PolicyGate
enforces this (with a framework-source allowlist for `source_file`:
`/sgl-workspace/{aiter,sglang,vllm}/` plus any paths in
`$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` — colon-separated, unioned with
defaults; auto-probed by `inference_optimizer/scripts/install.sh`).

Always prefer `manifest.json` / `state.json` / `coordinator.db` under the
**session dir** over guessing from terminal logs.
