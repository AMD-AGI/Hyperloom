# Inference Optimizer / Kernel Agent: unify all writable artefacts under `$USER_DATA_PATH`

> Branch: `feature/zhenggong/hyperloomV2` → `main`
> Range: `origin/main..HEAD` (2 non-merge commits)
> Stats: **24 files changed, +515 / −168**

## Summary

This PR collapses the three legacy "writable root" envs (`WORKSPACE_PATH`, `WORKSPACE_ROOT`, `HYPERLOOM_ROOT`/`/opt/hyperloom`, `MAGPIE_DIR`/`/workspace/Magpie`, `$REPO_ROOT/optimizer_runs/`) into a single user-facing knob: **`$USER_DATA_PATH`** (default `/workspace/hyperloom`). After this change, a single env override relocates every writable per-pod / per-session product Hyperloom emits — Magpie clone, source mirrors, generated env / GEAK config, kernel-agent tool outputs, launcher stdout / PID / monitor logs, auth-proxy logs.

Two ride-along fixes inside the same commit pin `SERVER_LOG` / `GPU_METRICS_CSV` per task so Magpie's `InferenceX/benchmarks/single_node/*.sh` wrappers stop leaking server stdout / GPU telemetry to `/workspace/server.log` and `/workspace/gpu_metrics.csv`. A README cleanup commit (`185efcf`) removes a duplicate `Path configuration` table.

No new external dependencies. Behaviour change is the env-name migration (`WORKSPACE_PATH` / `WORKSPACE_ROOT` retired) and the writable-default relocation; backwards compatibility is preserved for resume / breakdown reads of pre-migration sessions.

---

## What changed, by theme

### 1. Single artefact root: everything writable now defaults under `$USER_DATA_PATH`

Before this PR, writable state was scattered across four roots that operators had to relocate independently:

| Legacy default                       | What lived there                             |
| ------------------------------------ | -------------------------------------------- |
| `/workspace/Magpie`                  | Magpie clone (env: `MAGPIE_DIR`)             |
| `/opt/hyperloom`                     | GEAK / OOB / TraceLens source mirrors (env: `HYPERLOOM_ROOT`) |
| `$REPO_ROOT/optimizer_runs/`         | launcher stdout / PID / robustness monitor  |
| `$WORKSPACE_PATH/kernel-agent/runs/` | per-tool TraceLens / GEAK / OOB outputs     |

After this PR, every one of those defaults lives under `$USER_DATA_PATH`:

```
$USER_DATA_PATH/                         (default /workspace/hyperloom)
├── kernel-agent/runs/<session_id>/      # kernel-agent CLI tool outputs (NEW)
├── kernel-agent-workspace/<kernel_id>/  # cross-task GEAK/OOB artefacts (unchanged)
├── optimizer_runs/                      # launcher stdout / PID / monitor logs (NEW)
├── runtime/
│   ├── kernel-agent.env.sh              # generated; sourced before any tool call
│   ├── geak-config/local.yaml           # generated GEAK litellm config
│   ├── Magpie/                          # MAGPIE_DIR default (was /workspace/Magpie)
│   └── source-mirrors/                  # HYPERLOOM_ROOT default (was /opt/hyperloom)
│       ├── geak/
│       ├── OOB/oob_cli/
│       └── TraceLens-internal/
├── logs/auth-proxy/                     # ensure_auth_proxy.sh stdout/stderr (NEW)
└── (existing storage/ personas/ runs/ reports/ logs/ ... unchanged)
```

Operator overrides still win — `MAGPIE_DIR`, `HYPERLOOM_ROOT`, `HYPERLOOM_RUNTIME_DIR`, `KERNEL_AGENT_ENV` continue to take precedence when explicitly set. The change is only to the **defaults** when those envs are unset.

### 2. New helpers in `inference_optimizer/paths.py` and `session_paths.py`

Single source of truth so call sites no longer string-concatenate `$WORKSPACE_ROOT/Magpie`:

| Helper                                        | Returns                                    |
| --------------------------------------------- | ------------------------------------------ |
| `paths.runtime_dir(sd)`                       | `<sd>/runtime/`                             |
| `paths.source_mirrors_dir(sd)`                | `<sd>/runtime/source-mirrors/`              |
| `paths.magpie_dir(sd)`                        | `<sd>/runtime/Magpie/`                      |
| `paths.kernel_agent_runs_root(sd)`            | `<sd>/kernel-agent/`                        |
| `paths.optimizer_runs_dir(sd)`                | `<sd>/optimizer_runs/`                      |
| `session_paths.kernel_agent_runs_dir(sd, sid)`| `<sd>/kernel-agent/runs/<sid>/`             |
| `session_paths.optimizer_run_log(sd, tag)`    | `<sd>/optimizer_runs/run_<tag>.log`         |
| `session_paths.optimizer_run_pidfile(sd, tag)`| `<sd>/optimizer_runs/run_<tag>.pid`         |

The session skeleton (`_SESSION_SKELETON`) gained four new directories — `kernel-agent/`, `optimizer_runs/`, `runtime/`, `runtime/source-mirrors/`, `runtime/geak-config/` — created lazily by `make_session_dir(parents=True, exist_ok=True)`.

The `kernel-agent/` and `kernel-agent-workspace/` trees are intentionally disjoint:

- `kernel-agent/runs/<session_id>/` is keyed by **tool-invocation session id** and holds per-call logs / status JSON / TraceLens output / `optimization_attempts.jsonl`.
- `kernel-agent-workspace/<kernel_id>/` is keyed by **kernel_id** and survives across multiple `kernel_optimization.py` invocations on the same kernel (Coordinator-owned, GEAK/OOB cross-task cache).

### 3. `inference_optimizer/cli.py` preflight: derive every default through `paths`

- `_ensure_oob_proxy_source` no longer hardcodes `/opt/hyperloom`; it falls back to `source_mirrors_dir(session_dir())` when `HYPERLOOM_ROOT` is unset.
- Magpie preflight: `MAGPIE_DIR` env wins, then `paths.magpie_dir(session_dir())`. The legacy `WORKSPACE_ROOT/Magpie` path is gone.
- InferenceX detection order rewritten to:
  1. `$MAGPIE_DIR/InferenceX` (canonical post-`install.sh` layout)
  2. `$USER_DATA_PATH/runtime/InferenceX` (standalone runtime checkout)
  3. `/wekafs/hyperloom/InferenceX` (host-level mount)
  4. `/opt/hyperloom/InferenceX` + `/wekafs/fully-local/inference_optimization/InferenceX` (legacy)
- `WORKSPACE_PATH` is still set for the **critic-agent runtime** because there it names the **skill-asset root** (the repo checkout the runtime resolves prompts against), not a writable artefact dir. A comment block in `cli.py` calls this out so it isn't refactored away by mistake.

### 4. `inference_optimizer/scripts/install.sh` + `kernel-agent/scripts/install.sh`

- Removed `WORKSPACE_ROOT` / `WORKSPACE_PATH` from the env-override list (top of file + `--help` text).
- New defaults:
  - `HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"`
  - `MAGPIE_DIR="${MAGPIE_DIR:-${HYPERLOOM_RUNTIME_DIR}/Magpie}"`
  - `KERNEL_AGENT_ROOT` and `REPO_ROOT` resolve from this script's own location instead of `$WORKSPACE_PATH/kernel-agent` (so the skill files always come from the repo checkout, not the writable mirror).
- Pre-create `${HYPERLOOM_RUNTIME_DIR}` and `${HYPERLOOM_ROOT}` before chaining so `ensure_magpie` / GEAK-config write / `kernel-agent.env.sh` write never race on missing parents.
- `kernel-agent/install.sh::main` now creates the runtime tree on `$USER_DATA_PATH` instead of `${KERNEL_AGENT_ROOT}/runs` (the source checkout is treated as read-only; outputs go under `$USER_DATA_PATH/kernel-agent/runs/` lazily).
- `write_env_file` no longer emits `WORKSPACE_ROOT` / `WORKSPACE_PATH` lines.

### 5. `kernel-agent/scripts/ensure_auth_proxy.sh`

- `HYPERLOOM_ROOT` default rewritten the same way (now `${HYPERLOOM_RUNTIME_DIR}/source-mirrors`).
- Auth-proxy stdout/stderr lands at `$USER_DATA_PATH/logs/auth-proxy/` (overridable via `AUTH_PROXY_LOG_DIR`) instead of `${HYPERLOOM_ROOT}/logs/`, so a single `$USER_DATA_PATH` tail covers it.
- Bootstrap header line corrected from `bash $WORKSPACE_PATH/kernel-agent/scripts/ensure_auth_proxy.sh` to `bash $REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh`.

### 6. `kernel-agent/tools/`

- `tracelens_analysis.py`: `--workspace-path` default switched from a 3-tier `USER_DATA_PATH` → `WORKSPACE_PATH` → `/workspace/hyperloom` fallback to a clean `os.environ.get("USER_DATA_PATH", "/workspace/hyperloom")`. The deprecated `_default_workspace_path()` helper (and its multi-paragraph fallback docstring) are deleted.
- `kernel_optimization.py`: `_kernel_agent_root()` now reads `$USER_DATA_PATH` (legacy `$WORKSPACE_PATH` removed). `--workspace-path` default also switched. Help text added.
- `parallel_e2e_runner.py`: `--workspace-path` default switched, and the env passed to subprocesses (`run_one_attempt`, `main`) now exports the resolved path as **`USER_DATA_PATH`** so nested `kernel_optimization.py` / `tracelens_analysis.py` calls inherit the same artefact root.
- `backends/ray_runtime.py::SAFE_ENV_KEYS`: `WORKSPACE_ROOT`, `WORKSPACE_PATH`, `AGENT_WORKSPACE_ROOT` removed from the propagate list so stale values can't leak from the driver into Ray workers.
- `kernel-agent/tests/test_kernel_agent{,_live}.py`: test harness exports `USER_DATA_PATH` instead of `WORKSPACE_PATH`.

### 7. `kernel_request_handlers.py`: pass the session root, not a sub-tree

`select_kernels_handler` and `_run_optimization_single` used to spawn `kernel_optimization.py --workspace-path $SD/kernel-agent-workspace`, which produced the awkward `<sd>/kernel-agent-workspace/kernel-agent/runs/...` double-nested layout. They now pass `--workspace-path $SD` so the tool writes its artefacts at the canonical `<sd>/kernel-agent/runs/<session_id>/` while still reading `<sd>/kernel-agent-workspace/<kernel_id>/` for the cross-task GEAK/OOB cache.

### 8. `breakdown/collectors.py`: tri-layer back-compat for kernel-agent invocations

The breakdown reporter has to render historical sessions, so `_kernel_agent_run_dirs()` and `_read_kernel_candidates()` now scan three layouts in order:

1. **New (post-migration)**: `<sd>/kernel-agent/runs/<sid>/`
2. **Legacy double-nested**: `<sd>/kernel-agent-workspace/kernel-agent/runs/<sid>/`
3. **Even older per-kernel**: `<sd>/kernel-agent-workspace/<kid>/kernel-agent/runs/<sid>/`

`_read_kernel_candidates()` also rewrites either the `kernel-agent` or the `kernel-agent-workspace` anchor when re-rooting `state.last_select_kernels.candidates_path` from a `/workspace/...` container path onto wekafs. `breakdown/SKILL.md` (and the troubleshooting table) updated to mention both new + legacy locations.

### 9. Magpie wrapper leak: `SERVER_LOG` / `GPU_METRICS_CSV` per-task

Bundled into the same commit because it shares the "stop leaking outside the session dir" theme:

`InferenceX/benchmarks/single_node/*.sh` wrappers default `SERVER_LOG=/workspace/server.log` and `GPU_METRICS_CSV=/workspace/gpu_metrics.csv`. Both honor env overrides — `_grid_runner._run_magpie` and `BaselineExecutor` now **always overwrite** (not `setdefault`) those env keys to `<output_dir>/server.log` and `<output_dir>/gpu_metrics.csv` so a stale parent-shell value can't redirect a variant's logs into a previous run's slot. `harvest_leaked_artifacts` continues to run as defense-in-depth for any wrapper that hardcodes the destination ignoring the env.

### 10. Doc / template surgery

- `README.md` (this PR — `185efcf`): removes the duplicated *Path configuration* table; the canonical version with descriptions stays. Also (in `9fa8b72`) adds an **Optional** env table documenting `USER_DATA_PATH`, `HYPERLOOM_ROOT`, `MAGPIE_DIR`, `INFERENCE_OPTIMIZER_RESCUE_PATHS`, plus a dedicated **Migration Notes** subsection at the bottom of the workload-knobs section.
- `inference_optimizer/SKILL.md`: session-layout block expanded to show `kernel-agent/runs/`, `optimizer_runs/`, `runtime/{kernel-agent.env.sh, geak-config, Magpie, source-mirrors}`. New "inputs that stay outside `$USER_DATA_PATH` by design" callout listing the read-only roots (`$TRACELENS_ROOT`, `$OOB_SRC`, `~/.claude/config.json`, GEAK RAG cache, GEAK memory db) and why. Launch / monitoring snippets moved from `$REPO_ROOT/optimizer_runs/` to `$USER_DATA_PATH/optimizer_runs/`.
- `kernel-agent/SKILL.md`: artefact-tree section split into the `kernel-agent/runs/<session_id>/` (per-session) vs `kernel-agent-workspace/<kernel_id>/` (cross-task) trees, with an explicit "the legacy `WORKSPACE_PATH` env was retired" callout. All `bash $WORKSPACE_PATH/...` invocations rewritten to `bash "$REPO_ROOT/..."` for skill-asset reads, and the post-install env-source line points at `${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}`. TraceLens analysis.md path callout: `$USER_DATA_PATH/kernel-agent/runs/<session_id>/tracelens/analysis.md`.
- `critic-agent/SKILL.md` + `AGENTS.md`: clarifies that `WORKSPACE_PATH` for critic-agent is the **skill-asset root** (Hyperloom sets it to `$REPO_ROOT`), not an artefact root; per-session writable outputs always live under `$USER_DATA_PATH/critic-{session-memory,workdir}/`.
- `inference_optimizer/scripts/setup_env.sh.example`: drops `WORKSPACE_PATH`, surfaces `USER_DATA_PATH` at the top, removes the legacy `INFERENCEX_PATH=/wekafs/InferenceX` default (now derived under `$USER_DATA_PATH/runtime/`).
- `inference_optimizer/scripts/event_counts.py`: stops re-implementing the "env > default" resolution rule and defers to `inference_optimizer.paths.session_dir()`.

---

## Files of note

| File                                                       | Δ        | What                                                                |
| ---------------------------------------------------------- | -------- | ------------------------------------------------------------------- |
| `inference_optimizer/paths.py`                             | **+81**  | `runtime_dir`, `source_mirrors_dir`, `magpie_dir`, `kernel_agent_runs_root`, `optimizer_runs_dir`; expanded `_SESSION_SKELETON` |
| `inference_optimizer/breakdown/collectors.py`              | +79 / −33| Tri-layer back-compat for `_kernel_agent_run_dirs` + `_read_kernel_candidates` |
| `inference_optimizer/cli.py`                               | +49 / −8 | `_ensure_oob_proxy_source` + Magpie/InferenceX preflight via `paths` |
| `inference_optimizer/SKILL.md`                             | +44 / −12| Layout / inputs-outside-`$USER_DATA_PATH` / launch-monitor refresh |
| `inference_optimizer/session_paths.py`                     | **+57**  | `kernel_agent_runs_dir`, `optimizer_run_log`, `optimizer_run_pidfile` |
| `kernel-agent/SKILL.md`                                    | +37 / −13| Artefact-tree split, install/source/path snippet rewrite             |
| `kernel-agent/scripts/install.sh`                          | +28 / −11| Drop `WORKSPACE_*`, default `MAGPIE_DIR`/`HYPERLOOM_ROOT` under runtime, autoroot resolution |
| `kernel-agent/tools/tracelens_analysis.py`                 | +12 / −36| Delete `_default_workspace_path()` 3-tier fallback                  |
| `kernel-agent/tools/kernel_optimization.py`                | +18 / −2 | `_kernel_agent_root()` + `--workspace-path` default rewrite          |
| `kernel-agent/tools/parallel_e2e_runner.py`                | +18 / −4 | Subprocess env now exports `USER_DATA_PATH`; default rewrite         |
| `kernel-agent/scripts/ensure_auth_proxy.sh`                | +14 / −2 | Auth-proxy log dir under `$USER_DATA_PATH/logs/auth-proxy/`           |
| `inference_optimizer/scripts/install.sh`                   | +25 / −8 | Drop `WORKSPACE_ROOT`, pre-create runtime tree, log new defaults     |
| `inference_optimizer/orchestrator/kernel_request_handlers.py` | +14 / −9 | `--workspace-path` ⇒ session root (not subtree)                     |
| `inference_optimizer/orchestrator/action_executors/_grid_runner.py` | **+13** | Pin `SERVER_LOG` / `GPU_METRICS_CSV` per task                       |
| `inference_optimizer/orchestrator/action_executors/baseline.py` | **+9**  | Same pin in `BaselineExecutor`                                       |
| `kernel-agent/tools/backends/ray_runtime.py`               | +7 / −2  | Drop `WORKSPACE_*`, `AGENT_WORKSPACE_ROOT` from `SAFE_ENV_KEYS`      |
| `README.md`                                                | +21 / −8 | New optional env table, migration notes, dedup of `Path configuration` |

---

## Migration notes for users / launchers

1. **Rename `$WORKSPACE_PATH` / `$WORKSPACE_ROOT` to `$USER_DATA_PATH`.** Both legacy envs are no longer read by the installers, the CLI preflight, or the kernel-agent tools. (The critic-agent runtime still uses `WORKSPACE_PATH` as its skill-asset root; Hyperloom sets it automatically — leave it alone there.)
2. **Magpie clone moved.** Default is now `$USER_DATA_PATH/runtime/Magpie/` (was `/workspace/Magpie/`). If you maintain a shared Magpie checkout across sessions, keep exporting `MAGPIE_DIR=...` — operator overrides still win.
3. **Source mirrors moved.** Default `HYPERLOOM_ROOT` is now `$USER_DATA_PATH/runtime/source-mirrors/` (was `/opt/hyperloom`). GEAK / OOB / TraceLens mirrors land there.
4. **Launcher artefacts moved.** `setsid nohup ... > optimizer_runs/run_<tag>.log` now writes under `$USER_DATA_PATH/optimizer_runs/` instead of `$REPO_ROOT/optimizer_runs/`. Update tail / log-collection scripts accordingly. The `setup_env.sh.example` and the SKILL launch snippet show the new shape.
5. **Pre-migration sessions still read.** The breakdown collector scans `<sd>/kernel-agent/runs/`, `<sd>/kernel-agent-workspace/kernel-agent/runs/`, and the per-kernel double-nested form, so historical sessions render in `breakdown` reports without a one-off fixup.
6. **Server-log / GPU-metrics relocation.** Magpie's wrapper-script defaults at `/workspace/server.log` and `/workspace/gpu_metrics.csv` are now per-task pinned to `<task_output_dir>/{server.log,gpu_metrics.csv}` whenever the optimizer launches Magpie. If you have external scrapers reading those `/workspace/*` paths, point them at the per-task workspace instead.

---

## Risk / rollout

- **Behaviour change**: Defaults for Magpie, source mirrors, optimizer-launcher logs, kernel-agent tool outputs, and auth-proxy logs all moved under `$USER_DATA_PATH`. Operators who relied on the old defaults without setting any env (rare in production, common in scratch pods) will find their artefacts in the new location.
- **Env retirement**: `WORKSPACE_PATH` / `WORKSPACE_ROOT` / `AGENT_WORKSPACE_ROOT` are no longer propagated through Ray's `SAFE_ENV_KEYS`. Any kernel-agent tool or backend reading those names will get an empty value — grep private launchers + custom backends for the legacy names before merging.
- **Layout change**: Coordinator's `kernel_request_handlers` now passes the session root (not `<sd>/kernel-agent-workspace`) as `--workspace-path` to `kernel_optimization.py`. New runs land at `<sd>/kernel-agent/runs/...`. Breakdown back-compat covers historical paths.
- **Wrapper env pin**: `_grid_runner._run_magpie` and `BaselineExecutor` always overwrite `SERVER_LOG` / `GPU_METRICS_CSV`, even when the parent shell already exported a value. This is intentional (see the inline comment on stale-redirect prevention).
- **No SDK / API surface change**, no new external dependencies, no CLI flag added or removed.

---

## Commit log (2 non-merge)

```
185efcf minor fix
9fa8b72 feat(inference-optimizer): migrate to unified USER_DATA_PATH for session artifacts
```
