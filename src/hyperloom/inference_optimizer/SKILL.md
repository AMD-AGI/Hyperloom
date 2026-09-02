---
name: inference_optimizer
description: |
  Launches and monitors Hyperloom's multi-agent inference optimizer for LLM
  serving on AMD GPUs. Use when the user asks to optimize an inference model,
  run Magpie benchmarks/profiles, resume an inference_optimizer session, tune
  SGLang/vLLM serving parameters, run TraceLens/kernel-agent, or validate
  end-to-end throughput gains in a new inference environment.
globs:
  - "**/inference*optim*"
  - "**/inference_optimizer*"
  # The Coordinator/orchestrator lives under src/hyperloom/orchestrator/;
  # keep this skill triggering on it since it still owns the launcher's runtime story.
  - "**/hyperloom/orchestrator/**"
---

# Inference Optimizer Skill

You are the launcher and monitor. The optimizer itself is the Python
`inference_optimizer` runtime under this repository. Do not manually optimize
inside chat unless debugging; launch the CLI, poll persisted state, and report
objective progress.

## What This Skill Runs

The CLI starts a Python Coordinator that coordinates:

- Orchestration: decides next actions (`baseline`, `explore`, `specialist`, `integrate_patch`, `sweep`, Kernel requests, `report`).
- Kernel (programmatic, not LLM): the Coordinator dispatches `trace_analyze`, `run_gemm_tuning`, `run_optimization`, `integrate`, and related request kinds directly to Python handlers without an LLM turn. The `run_fusion` and `run_collective` lanes share that handler table but are Coordinator-owned: they run at KERNEL entry behind their own gate and PolicyGate rejects an agent request for either.
- Critic: proposal review (default `--critic-agent`; see
  [Critic Backend Selection](#critic-backend-selection) for modes).
- Robustness: default `--robustness-agent` — drives the
  `hyperloom.agents.robustness` subprocess runtime for health monitoring, RCA, and scheduling-police
  intents. `--robustness-mock` for offline / smoke tests.
  - **Multi-node auto-downgrade (`--nodes >= 2`)**: the agent backend's
    `LocalProbeSource` targets sandbox-local resources only (ray status,
    inference server, GPU, FD, disk, shm). On multi-node every
    such resource lives in a separate pod (head / worker / RayJob), so each
    probe surfaces as a HIGH false positive that floods the bus. The CLI
    auto-downgrades to `--robustness-mock` (heartbeat only) and prints a
    WARNING; pass `--robustness-mock` explicitly to suppress it. See
    `src/hyperloom/inference_optimizer/multi_node/SKILL.md` (Robustness limitation in multi-node mode).

State lives under a **session directory** (per optimization run).
The **workspace root** is ``$USER_DATA_PATH`` (default
``/workspace/hyperloom``) — it holds shared ``runtime/`` and ``logs/``.

### Layout (N17 default: ``per_model_ts``)

```text
$USER_DATA_PATH/                          # workspace_root — set by operator / Claw / SaFE
├── runtime/                              # workspace-shared (install.sh, Magpie, kernel-agent.env.sh)
│   ├── kernel-agent.env.sh
│   ├── Magpie/
│   └── source-mirrors/{InferenceX,TraceLens[,TraceLens-internal]}/
│       # Open-source deps are installed by install.sh.
├── logs/                                 # workspace-shared launcher stdout
└── <model_basename>/                     # e.g. DeepSeek-R1-0528, deepseek-ai-DeepSeek-V3
    └── <UTC_YYYYMMDDTHHMMSSZ>-<rand8>/   # session_dir — manifest.json, state.json, runs/, …
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

**Claw / SaFE pods:** the launcher often sets ``$USER_DATA_PATH`` to a
run-scoped path *before* the optimizer starts, e.g.
``/hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/``.
That outer directory is **platform isolation** (one Claw job). The
optimizer then creates ``<model_basename>/<UTC_ts>-<rand8>/`` inside it. Full
session path example::

    /hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/   ← USER_DATA_PATH (Claw)
        deepseek-ai-DeepSeek-V3/20260522T035359Z-9f3c1a04/            ← session_dir (optimizer)

### Path resolution (do not guess)

`session/paths.py` is the single authority for Hyperloom paths. The launching
agent does not need to recreate that logic in shell; it only needs to run
`install.sh`, source the generated `runtime/kernel-agent.env.sh`, and read
the session dir printed by the CLI.

| Concept | Env / helper | Meaning |
|---|---|---|
| Workspace root | ``$USER_DATA_PATH`` → ``session.paths.workspace_root()`` | Shared ``runtime/`` + ``logs/`` and parent of all sessions |
| Session dir | ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` → ``session.paths.session_dir()`` | Per-run directory containing ``manifest.json`` / ``state.json`` / ``storage/coordinator.db`` |

**Launcher rule:** do not hand-build, create, delete, or repair paths
under ``$USER_DATA_PATH/runtime/`` (especially ``source-mirrors/``).
Those are workspace-shared assets owned by `install.sh`, including
Magpie, InferenceX, GEAK, TraceLens mirrors, env files, and config.
Manual edits there can corrupt another run's checkout. If install state looks wrong,
rerun `install.sh` or follow the Recovery section; do not clone or clean
the mirrors by hand.

**Session rule:** never treat ``$USER_DATA_PATH`` as the session dir when
``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` is set. Read
``manifest.json`` / ``state.json`` / ``coordinator.db`` from the
**session dir**. For monitoring after launch, learn the session dir from
the **launch-info JSON** written by ``--launch-info-file`` (``jq -r
.session_dir <file>``) or, equivalently, from the single
``HYPERLOOM_LAUNCH key=value …`` sentinel line the CLI prints to stdout
(``session_dir=…``). Those are the authoritative, machine-readable
sources. Never guess by walking ``$USER_DATA_PATH/<model_basename>/`` for
the latest ``*T*Z/`` timestamp dir — overlapping sessions on the same
host make "latest" pick the wrong run.

Inputs that stay outside `$USER_DATA_PATH` by design (read-only sources
or warm-start caches): **TraceLens** — `$TRACELENS_ROOT` (default
`${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens`; when unset,
`src/hyperloom/agents/kernel/scripts/install.sh` clones
[AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) there and pins
it to a fixed SHA. A pre-existing checkout you maintain is only used as
an explicit operator override — export `TRACELENS_ROOT=<path>` to opt
in, which skips both the clone and the SHA pin) with an **optional**
internal
extension at `$TRACELENS_INTERNAL_ROOT` (no default; internal users set
it to their own existing checkout to opt in,
otherwise open-source-only; rehydration module — Hyperloom keeps no internal
URL/path). The per-version
`sglang_roofline_patches/sglang_<minor>_<patch>/` layout under
TraceLens is required by `_server_patcher`),
`/sgl-workspace/{aiter,sglang,vllm}/`,
`~/.cache/amd-ai-devtool/semantic-index/`
(GEAK RAG embedding cache), `/shared/hyperloom/geak-memory/memory.db`
(GEAK cross-session memory). Each is overridable via its own env if
you want a fully self-contained session.

Paths emitted by agents must resolve under the **session dir** — PolicyGate
enforces this (with a framework-source allowlist for `source_file`:
`/sgl-workspace/{aiter,sglang,vllm}/` plus any paths in
`$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` — colon-separated, unioned
with defaults; auto-probed by `src/hyperloom/inference_optimizer/assets/install.sh`).

Always prefer `manifest.json` / `state.json` / `coordinator.db` under the
**session dir** over guessing from terminal logs.

## Iron Rules

SKILL-level constraints the launcher MUST satisfy before `Coordinator`
is allowed to boot. These IronRULEs are the gate
that runs **before** `python -m hyperloom.inference_optimizer.cli optimize` is even spawned.

### IR-1 — GPU MUST be unoccupied before every launch

Before every `python -m hyperloom.inference_optimizer.cli optimize` invocation (fresh start OR
`--resume-from`), verify that every visible GPU on this pod has **zero
foreign serving PIDs and VRAM usage below 1% of each card's total capacity**. A leftover
`sglang.launch_server` / `vllm.entrypoints` / `Magpie` from a previous
run silently degrades the next `baseline` by 5–30 % (shares VRAM +
schedules on the same XCD); `current_best` cannot detect this
pollution after the fact.
> Inside a running session, the equivalent guard is enforced in
> `orchestrator/kernel/request_handlers.py` via
> `_multi_node_server_lifecycle.py::restart_server_for_round`, which kills
> stale servers before every restart. IR-1 above is the *outer* gate that
> fires before the optimizer process exists.

#### Prior workload cleanup gate (error recovery)

**Trigger (MUST):** whenever a run fails, is abandoned, or you are about to
start a **replacement** workload after any error — credential failures,
optimizer crash, install/preflight failure, user retry/restart, or any recovery
where you would run `docker run`, `install.sh`, or a new/fresh `optimize`.
Applies in bare-metal and docker mode, **whether reusing the existing container
or starting a new one.** Leftover optimizer/serving processes or occupied VRAM
are the usual cause of misleading 0% validated gain (#1314).

**Exception:** `--resume-from "$SESSION_DIR"` against the **same** session
immediately after a clean crash (no credential change, user explicitly wants
resume) may skip — but if the probe finds live or ambiguous leftover workload,
run it anyway and ask the user.

**Probe on the docker host** (bare-metal: current host). **Never** rely on
`docker exec` for process or VRAM checks — container PID namespaces hide
processes in *other* containers; the host namespace is the superset (#1314).

```bash
export REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
# .env fills gaps only — same pattern as the launch block below.
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
eval "$_dotenv_prev"
unset _dotenv_prev
export USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"

# Prior launch handles from canonical artifacts (no last_launch.env — never written)
LATEST_PID_FILE="$(ls -t "$RUN_DIR"/run_*.pid 2>/dev/null | head -1 || true)"
LATEST_LAUNCH_INFO="$(ls -t "$RUN_DIR"/launch_*.json 2>/dev/null | head -1 || true)"
PRIOR_SESSION=""
if [ -n "$LATEST_LAUNCH_INFO" ]; then
  PRIOR_SESSION="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("session_dir") or "")' "$LATEST_LAUNCH_INFO" 2>/dev/null || true)"
fi
PRIOR_PID=""
PRIOR_PID_LIVE=false
if [ -n "$LATEST_PID_FILE" ] && [ -f "$LATEST_PID_FILE" ]; then
  PRIOR_PID="$(tr -d '[:space:]' < "$LATEST_PID_FILE" 2>/dev/null || true)"
  if [ -n "$PRIOR_PID" ] && kill -0 "$PRIOR_PID" 2>/dev/null; then
    PRIOR_PID_LIVE=true
  fi
fi
echo "prior_pid_file=${LATEST_PID_FILE:-none}"
echo "prior_pid=${PRIOR_PID:-none}"
echo "prior_pid_live=${PRIOR_PID_LIVE}"
echo "prior_launch_info=${LATEST_LAUNCH_INFO:-none}"
echo "prior_session=${PRIOR_SESSION:-none}"

# Foreign processes — host-level pgrep (patterns match preflight_optimizer.py)
pgrep -af 'hyperloom\.inference_optimizer\.cli.*optimize' || true
pgrep -af 'sglang\.launch_server|vllm\.entrypoints|Magpie' || true

# VRAM — stdlib-only rocm-smi parse (must run on docker host; no hyperloom import)
python3 - <<'PY'
import json, shutil, subprocess

def unreadable():
    print("gpu_vram=unreadable")

if not shutil.which("rocm-smi"):
    unreadable()
    raise SystemExit(0)
try:
    proc = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True,
        text=True,
        timeout=20,
    )
except (OSError, subprocess.SubprocessError):
    unreadable()
    raise SystemExit(0)
if proc.returncode != 0:
    unreadable()
    raise SystemExit(0)
try:
    data = json.loads(proc.stdout)
except (json.JSONDecodeError, ValueError):
    unreadable()
    raise SystemExit(0)
if not isinstance(data, dict):
    unreadable()
    raise SystemExit(0)

rows: list[tuple[float, float]] = []
for fields in data.values():
    if not isinstance(fields, dict):
        continue
    raw: dict[str, float] = {}
    for key, val in fields.items():
        kl = key.lower()
        if "vram" not in kl:
            continue
        if "used" in kl:
            raw["used"] = float(val)
        elif "total" in kl:
            raw["total"] = float(val)
    if not raw:
        continue
    try:
        used_mib = raw["used"] / 1024**2
        total_mib = raw["total"] / 1024**2
    except (KeyError, TypeError, ValueError):
        unreadable()
        raise SystemExit(0)
    if total_mib <= 0.0:
        unreadable()
        raise SystemExit(0)
    rows.append((used_mib, total_mib))

if not rows:
    unreadable()
    raise SystemExit(0)

for i, (used_mib, total_mib) in enumerate(rows):
    pct = used_mib / total_mib
    busy = pct > 0.01
    print(
        f"gpu{i}_vram_used={used_mib:.1f}/{total_mib:.1f} MiB "
        f"({pct:.2%}) {'BUSY' if busy else 'idle'}"
    )
PY

# Other hyperloom-named containers — exclude the one we will reuse
CURRENT_CONTAINER="${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
docker ps --filter "name=hyperloom" --format '{{.Names}}' 2>/dev/null \
  | grep -v "^${CURRENT_CONTAINER}$" || true
```

**If anything is found** (`prior_pid_live=true`, foreign process from pgrep,
any GPU line marked `BUSY` or `gpu_vram=unreadable`, or another
hyperloom-named running container), stop and ask the user explicitly — e.g.
*"The previous run may still be active (session …, PID …, GPU VRAM …). Stop
it before we continue?"* Wait for yes/no.

- **Yes:** stop in this order: serving PIDs from pgrep, then the optimizer
  (`kill "$PRIOR_PID"` when `prior_pid_live=true`, else the pgrep match), then
  **only** other hyperloom-named containers from the filtered list above
  (`docker stop <name>`). Only then continue with `--resume-from "$PRIOR_SESSION"`
  or a fresh launch as appropriate.
- **No:** do **not** treat results as clean. Continue only if the user insists;
  **MUST** add this caveat verbatim to the final report / session summary:

  > **Contaminated baseline warning (#1314):** Prior GPU workload was not
  > stopped per user choice. Validated gain and benchmark numbers may be
  > unreliable.

Never kill processes or stop containers without explicit user approval.

### IR-2 — install.sh MUST succeed before every launch

Run `bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"` and
source the regenerated
`${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}`
in the **same shell** that will spawn `python -m hyperloom.inference_optimizer.cli optimize`.
Skipping install strikes silently *after* `baseline` succeeds: missing
TraceLens/GEAK → `trace_analyze` / `kernel_opt` fail; no live
Ray head → `kernel_opt` tasks hang; missing `kernel-agent.env.sh` →
first kernel-opt gateway call returns `401`. `install.sh --check-only` is a
*diagnostic*, never a substitute.

**Resume carve-out.** `... optimize --resume-from` may skip install only when
ALL hold: (1) `install.sh` exited 0 earlier in the *same shell*; (2)
`kernel-agent.env.sh` is still sourced; (3) `manifest.json` exists under the
session dir passed to `--resume-from`.
Any failure → treat as fresh launch and re-run `install.sh`.

> The in-loop equivalent is `_preflight()` steps 1–12 (drift repair, not
> a substitute for this outer gate).

### IR-3 — PR Monitor reachability (in-loop, soft degrade)

The PR Monitor endpoint is co-hosted by KB Store. Hyperloom derives
`${KB_STORE_URL}/pr-monitor/v1/healthz` for this probe and
`${KB_STORE_URL}/pr-monitor/mcp/` for specialist tools; there are no separate
PR endpoint flags or Cortex endpoint variable.

`_preflight()` invokes:

```
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/preflight_kb.sh"
```

Exit codes (soft degrade — IR-3 never aborts launch):

- `0` → PR Monitor reachable (or probe skipped). `pr_monitor_enabled` stays `True`.
- `1` → PR Monitor unreachable. The cli auto-enables `--degraded-pr` and
  continues; `manifest.json` records `pr_degraded_reason=ir3_auto`.

Recipe KB enablement is independent: `--degraded-kb` sets `recipe_kb_enabled=False`
(T0/T2/T3/T4 no-ops) without affecting PR Monitor.

Operator opt-out: pass `--degraded-pr` to skip the PR Monitor probe (one
round-trip saved); `manifest.json` then records `reason=explicit_flag`.
Pass `--degraded-kb` and `--degraded-pr` together to short-circuit the entire
IR-3 step.

### IR-4 — OPTIMIZE phase contracts (Coordinator-internal)

These govern the optimizer's OPTIMIZE phase, not the launcher; the full
contract lives in `src/hyperloom/orchestrator/prompts/orchestration.md`. In
brief:

- **IR-4 — OPTIMIZE is specialist-informed**: prefer specialist- or
  research-backed variants when available, but `llm_direct`,
  `default_grid`, `specialist:<domain-or-tag>`, and `dynamic` provenance
  values are all accepted audit labels when phase and sequence gates pass.
  Specialist- and dynamic-sourced variants are not grid-size capped;
  per-round breadth is bounded by the `research_lane` / GPU pool leases
  (the `research_lane` scales with the `2 × visible GPU count` ceiling).
  Specialists author patches into an isolated worktree; `integrate_patch`
  does the actual `git apply` + throughput/accuracy gate after Critic
  review.
  GPU specialists are **on by default at whole-machine capacity** (WS2):
  `--gpu-specialist-capacity` defaults to the visible GPU count on the launch
  host (`_default_gpu_specialist_capacity()`), so Orchestration may dispatch
  `delegate{action_name='specialist', params={needs_gpu: true, gpu_count: ...}}`
  without any extra flag. Pass `--gpu-specialist-capacity N` to clamp the pool,
  and `--gpu-specialist-capacity 0` to disable GPU specialists entirely. The
  legacy `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY` env is ignored by the CLI
  default resolver; use the explicit flag for operator control. When enabled, GPU
  specialists serialize against serving through `gpu_research_lane` and
  exclusively own their leased cards: they may start/stop their own servers
  (any port that is not the production serving port 8888), profile, autotune,
  and run real benchmark loops. The one invariant is that they must not touch
  the production serving process, its cards, or port 8888.
- **Plateau**: both arms' signals and KERNEL_AGENT's are computed every tick
  and rendered in the orchestration prompt. One arm dry is advisory — the
  phase stays open on the other lever. **Both arms dry advances the phase**
  via `optimize_no_more_leverage`. A KERNEL_AGENT plateau stays advisory. The
  LLM may also emit
  `escalate_strategy_change{hint='skip_to_kernel'/'skip_to_sweep'/'skip_to_close'}`
  when it judges further effort unproductive.

### FRAMEWORK_AGENT phase — the optimisation phase

One phase, two arms (`--no-framework-agent` skips it entirely).

The **configuration arm** runs server-arg / env grids, sourced by specialist
fan-out. The **source arm** lands upstream PRs and specialist-authored
patches. They are worked in parallel and rotate on their own plateau
judgement, not on a wall-clock split.

The source arm's supply is the `candidate_discovery_specialist`: it surveys
the allowlisted repos, ranks what it finds against the stack and the tried
ledger, and judges each entry — already present, not applicable, or worth a
bench and by which route. The pump takes that batch in the order given; it
does not re-rank or re-audit. When discovery comes back empty its full retry
budget, the local-exploration arm authors against profile evidence instead;
with that off too, the source arm reports itself dry.

Every diff lands through `integrate_patch`, with `patch_source` naming where
it came from. KEEP commits to the live tree so the next candidate stacks on
top; REVERT does `git reset --hard`. Resume skips completed candidates by
idempotency key.

The phase exits when both arms are dry, when its budget is spent, or at the
absolute per-phase cap.

### IR-8 — `--framework atom` is single-node only

`--framework atom` (Magpie `atom_mi*x.sh` against
`atom.entrypoints.openai_server`) reaches full parity with sglang/vllm
EXCEPT multi-node: `_apply_atom_auto_tighten` in `cli.py` rejects
`--nodes >= 2` with `SystemExit(2)` (atom upstream has no multi-node TP
wiring). No other flag is auto-flipped — kernel-agent, framework-agent,
profile / roofline / TraceLens all run on atom. The atom-specific
behaviors (configs, cold-start seed grid, source roots) are summarized
under **Framework Selection** below.

## Retired modules and rules (do not re-introduce)

The live runtime uses `protocol/action_surfaces.ACTION_CATALOGUE`,
`_grid_runner.py`, and the unified specialist-informed `explore` flow. Do not
recreate the retired `backends` / `params` / `validate_stack` / scoring
modules, nor the `actions/_meta/*.yaml` catalogue and its ActionRegistry
loader, nor the `vendor_kernel_config` / `operator_tuning` /
`deep_kernel_analysis` actions (they never had an implementation).

Rules that look reasonable but break the current flow:

- **No "source lever before configuration" rule** in
  `prompts/orchestration.md` — the two are arms of one phase,
  worked in parallel and ranked by what the bottleneck calls for.
  Upstream diffs land through `integrate_patch` with
  `patch_source='upstream_pr'`; there is no separate
  `framework_agent` action for the LLM to propose or be denied.
  Use `--no-framework-agent` to skip the phase entirely.
- **`kernel_opt` sequencing** is no longer gated by an
  explore-minimum check (the
  `explore_attempts_minimum_before_kernel_opt` rule was retired
  in loosen_plan P1_06). KERNEL_AGENT phase may propose `kernel_opt`
  directly; the `trace_analyze → run_optimization` data
  dependency (P2_11 handler-level check) and the reusable
  `kernel_id` validation still keep the inputs valid.

## Setup

Two commands: Step 1 implements **IR-2** (install gate), Step 2 launches.
Both are idempotent; do not replicate them inside chat.

### Credentials

The common single-gateway setup uses `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
Split-gateway deployments may provide provider-specific `ANTHROPIC_*` /
`OPENAI_*` credentials instead. Shell-exported values win; `$REPO_ROOT/.env`
is loaded only to fill missing values. `install.sh` and the CLI preflight
enforce this internally; the launch recipes below enforce it by re-exporting a
snapshot of the caller's environment after sourcing `.env`, so a path variable
such as `USER_DATA_PATH` left in `.env` can never redirect a run to another
workspace. Never plain `set -a; . .env` — that inverts the precedence.
After Step 1, source the generated `kernel-agent.env.sh` in the same shell.


### Step 1 — Install (one-time per pod / venv rebuild)

```bash
export REPO_ROOT="$(pwd -P)"   # repo root containing src/hyperloom/ + .env
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"   # pod-local runtime env
```

`src/hyperloom/inference_optimizer/assets/install.sh` is the only install entrypoint for
full inference optimization. It installs the optimizer / Magpie / InferenceX
first, then chains to `src/hyperloom/agents/kernel/scripts/install.sh` for the kernel
optimization environment. `src/hyperloom/agents/kernel/scripts/install.sh` remains valid for
standalone kernel-agent debugging, but should not be the main entrypoint for a
full inference optimizer session.

The install phase always initializes the full Hyperloom runtime. Even if the
user later passes `--no-kernel` at runtime, the installer still prepares
kernel-agent / TraceLens / GEAK; `--no-kernel` only means
that this `optimize` run skips the kernel optimization phase.

`install.sh` installs everything in one shot (no `--with-*` flags to
remember). Direct steps in `src/hyperloom/inference_optimizer/assets/install.sh`:

| Component | Provided by |
|---|---|
| `inference_optimizer` pkg + `claude_agent_sdk` extras (`pip install -e .[test]`) | `ensure_inference_optimizer` |
| **Magpie** (`pip install "$MAGPIE_PACKAGE_SPEC"`; default spec pins `magpie-eval` to `$MAGPIE_REF`) | `ensure_magpie` |
| `INFERENCEX_PATH` resolution (honours a pre-existing `$INFERENCEX_PATH`, else clones `$INFERENCEX_REPO` pinned to `$INFERENCEX_REF` into `$INFERENCEX_DEFAULT_DIR` = `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/InferenceX@<sha>`, reusing an existing checkout there on re-runs) | `ensure_inferencex` |
| `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` appended to `kernel-agent.env.sh` | `_probe_framework_source_roots` |

Chained from `src/hyperloom/agents/kernel/scripts/install.sh` (single chain at the end
of `src/hyperloom/inference_optimizer/assets/install.sh`):

| Component | Provided by |
|---|---|
| `ray==2.44.1` + `click<8.3.0` | pip |
| TraceLens public (editable install) | `ensure_tracelens` (`pip install -e` at `$TRACELENS_ROOT`; skills, patches, CLI, analysis orchestrator) |
| TraceLens-internal (editable install, **optional**) | `ensure_tracelens` (`pip install -e` at `$TRACELENS_INTERNAL_ROOT` only when set; mirrors read-only checkout to `${HYPERLOOM_ROOT}/TraceLens-internal`; rehydration module). Unset => open-source-only. |
| GEAKv4 Claude Code workflow checkout + SDK deps | `ensure_geak` |

`${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}` is
regenerated by `install.sh` and contains gateway URLs, auth aliases,
GEAK runtime variables, and InferenceX path. Source it (don't try to derive these by
hand). Generated env/config state is written to the pod-local runtime directory,
not back into a shared WekaFS source checkout.

### Tool source fields (prompt → env, sandbox-only)

Prompt fields naming read-only source trees consumed by sandbox-side
`install.sh` / launcher. `export <K>="<v>"` in the launcher shell before
`install.sh`. These are **sandbox-only** — never ask the platform to bake them
into multi-node pod env; those pods have their own paths and do not consume
these.

| Prompt field | Env name | Consumer |
|---|---|---|
| `INFERENCEX_PATH: <path>` | `$INFERENCEX_PATH` | `src/hyperloom/inference_optimizer/assets/install.sh:ensure_inferencex` |
| `TRACELENS_ROOT: <path>` | `$TRACELENS_ROOT` | `src/hyperloom/agents/kernel/scripts/install.sh:ensure_tracelens` (public) |
| `TRACELENS_INTERNAL_ROOT: <path>` (optional) | `$TRACELENS_INTERNAL_ROOT` | `src/hyperloom/agents/kernel/scripts/install.sh:ensure_tracelens` (internal; only when set) |

**Multi-node escape hatch**: if `$TRACELENS_ROOT` / `$TRACELENS_INTERNAL_ROOT` / `$GEAK_ROOT` /
`$WORKSPACE_ROOT/Magpie` / `$INFERENCEX_PATH` may move or differ across nodes,
`rsync -a` them into `$SESSION_DIR/vendor/<name>/` and override the matching
env vars BEFORE running `install.sh`. Single-node WekaFS-mount setups (the
production default) need none of this — `ensure_tracelens`
already handles the read-only-source case.

### Step 1.5 — Write the advisory `model_arch` profile (best-effort)

After the CLI creates the session directory, produce an **advisory**
architecture profile so the orchestration + specialist prompts carry richer
model context than the coarse `--model-class` tag. This is **best-effort
and non-fatal**: a missing / invalid file simply causes Hyperloom to omit
the section — it never blocks launch, never replaces `--model-class`
(still required), and is always **subordinate to live TraceLens evidence**
at runtime (it drives no deterministic gating — atom seed grid, framework
gap token, recipe key, and prompt label all stay on `model_class`).

Steps for the launching agent:

1. **Gallery lookup** — fetch the LLM Architecture Gallery
   (`https://sebastianraschka.com/llm-architecture-gallery/`) and locate
   the card for the model being launched. Extract the schema fields below.
2. **Fallback classify** — if the model is not in the gallery, do a
   lightweight classify from the model's local `config.json` (decoder
   type, attention variant, expert counts, MTP, SWA window) and set
   `"source": "config_classify"`.
3. **Write the profile BEFORE launch and point `$HYPERLOOM_MODEL_ARCH_FILE`
   at it.** The CLI creates `session_dir` and reads the profile in the *same
   process*, so a file written to `<session_dir>/model_arch.json` after the
   session dir appears always loses the race — the run seeds
   `state.model_arch={}` and the profile never reaches any prompt. Write the
   JSON to a launcher-owned path instead and export
   `HYPERLOOM_MODEL_ARCH_FILE=<that path>` in the shell that spawns
   `optimize`; the CLI copies it into `<session_dir>/model_arch.json` for
   provenance. Use a **session-unique** filename — `$USER_DATA_PATH` is shared
   by concurrent sessions on WekaFS, so a fixed name races other launches.
   Include `model_name` (required for the stale-file guard). Set it to the
   **clean model name** (e.g. `Qwen2.5-7B-Instruct`);
   the guard normalizes launch forms — flat dirs, HF repo ids, and HF hub
   cache `models--org--repo/snapshots/<hash>` paths — so do NOT use the
   snapshot commit hash. All other fields are optional; renderers drop
   empty fields.

```json
{
  "model_name": "DeepSeek-R1-0528",
  "source": "gallery",
  "decoder_type": "Sparse MoE",
  "attention": "MLA",
  "layer_mix": "61 MLA",
  "kv_cache_per_token": "68.6 KiB",
  "active_params": "37B active / 671B total",
  "num_experts": 256,
  "experts_per_tok": 8,
  "mtp": true,
  "swa_window": null,
  "norm": "RMSNorm",
  "notes": "DeepSeek V3-style: dense prefix + shared expert + MTP-1 path"
}
```

If you cannot determine the architecture, skip this step — do not write a
placeholder file. Hyperloom degrades silently (WARNING in its own logs)
when the file is absent, invalid, or stale.

### Step 2 — Launch

**Multi-node (`nodes >= 2`):** [`multi_node/SKILL.md`](multi_node/SKILL.md).

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \           # sglang (default) / vllm / atom / xdit / custom
  --gpu-type MI300X \          # or omit for rocm-smi auto-detect
  --model-class moe_mla \      # dense / moe_mla / moe_swa / moe_mla_nsa; categorical key for atom seed grid + framework gap token + recipe key + prompt label
  --isl 512 --osl 512 \        # workload shape — pass whatever the prompt states; omitting them uses defaults ISL=1024/OSL=1024
  --conc 64 \                  # client concurrency — pass the prompt's value; default 64
  --tp 1 --ep 1 \              # parallelism — pass the prompt's TP/EP; defaults 1/1
  --precision bf16 \           # match the checkpoint (bf16 default); use fp8 for an FP8 checkpoint
  --max-hours 2 \
  --compare-against-gpu B200   # optional — when set, fetches real InferenceX reference; when unset, target_analysis still runs and writes a 'no_target_gpu_configured' marker JSON
```

**Caller responsibility (post-classify-removal)**: the in-loop `setup` /
`classify` actions were deleted; the SKILL caller is now expected to
supply session metadata directly via CLI flags. **Any workload value the
operator states in the prompt (ISL, OSL, CONC, TP, EP, precision, budget, and
every `--extra-env`) MUST be forwarded as the matching CLI flag** — these flags
are the only source of truth; an omitted flag silently falls back to its default
and the operator's stated value is lost:

| Surface | CLI flag | Notes |
|---|---|---|
| Model path | `--model` | required |
| Framework | `--framework` | `sglang` (default) / `vllm` / `atom` / `xdit` / `custom` — atom is single-node-only; xdit is scriptable diffusion (`img/s`, no serving server); `custom` is an operator-supplied workload and **additionally requires `--framework-path` and `--benchmark-scripts-dir`** (see below) |
| Custom source tree | `--framework-path` | **Required for `--framework custom`.** The workload's own checkout; patches are authored against it. |
| Custom bench scripts | `--benchmark-scripts-dir` | **Required for `--framework custom`.** Holds the entrypoint, looked up as `custom_<gpu-type>.sh`. Every knob it reads must be forwarded as `--extra-env`; the throughput unit is whatever its report declares. |
| GPU type | `--gpu-type` | rocm-smi auto-detect when unset |
| Model class | `--model-class` | categorical key for the deterministic consumers (atom seed grid, framework-agent gap search token, recipe key, prompt label); when unset, Coordinator boot infers and persists it from model metadata or model-path family keywords. For richer advisory model context see Step 1.5 (`model_arch.json`) |
| Input seq length | `--isl` | Pass the prompt's ISL. Default `1024` when omitted. |
| Output seq length | `--osl` | Pass the prompt's OSL. Default `1024` when omitted. |
| Concurrency | `--conc` | Pass the prompt's CONC (max in-flight requests). Default `64`. SWEEP measures a ladder around it; `--conc-sweep-concs` overrides the workload's default ladder. |
| Tensor parallel | `--tp` | Pass the prompt's TP. Default `1`. |
| Expert parallel | `--ep` | Pass the prompt's EP for MoE. Default `1`. |
| Precision | `--precision` | Match the checkpoint (`bf16` default / `fp8` / ...). Keep consistent with `--quantize`. |
| Budget | `--max-hours` | Pass the prompt's time budget. Default `2.0`. |
| Max model len | `--max-model-len` | Optional; auto-derived from ISL+OSL+headroom when omitted. |
| External reference GPU | `--compare-against-gpu` | Coordinator *always* hard-gates `target_analysis` to run first so `$SESSION_DIR/target_analysis/target_baseline.json` exists before `baseline` runs. When this flag is set the JSON carries the InferenceX reference (`reason="ok"`); when unset the JSON carries a structured `reason="no_target_gpu_configured"` marker. The report renders the "External baseline" section from this JSON in both cases (heading switches to "(not requested)" for the marker variant) |
| Quantization prelude | `--quantize` | Optional. Natural-language quantization request. Runs the quantization-agent once before the loop and rewrites `--model` to the quantized model. See Step 2b. Never runs on a resume. |
| Env pins | `--extra-env NAME=VALUE` | Repeatable; forward **every** one verbatim as its own flag (do not drop any or fold into the `Environment:` block). The CLI persists them in `state.json` and serializes them into `$INFERENCE_OPTIMIZER_EXTRA_ENV`; a dropped pin is lost silently — e.g. a missing `SGLANG_USE_AITER=0` leaves the explore aiter-MoE filter blind. A `--resume-from` re-exports the persisted set, so re-pass them only to change the set. |

### Step 2b — Optional quantization prelude (`--quantize`)

When the user asks to **quantize the model before optimizing** (e.g. "quantize
to FP8 then optimize", "run this in MX-FP4"), pass `--quantize "<scheme prompt>"`
to the same `optimize` command. This runs the **quantization-agent once as a
prelude**, before any baseline/session work: it drives AMD Quark PTQ from the
prompt, then rewrites `--model` to the exported quantized model so the entire
optimization loop runs on the quantized model.

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --quantize "fp8 global scheme, fp8 kv_cache, exclude lm_head; accept up to 5% relative eval gap" \
  --max-hours 2
```

- The `--quantize` text is the quantization request only (scheme / kv-cache /
  excluded layers / acceptable eval gap). **Do not** repeat the model path or
  export dir — the adapter folds `--model` + a per-model export dir under the
  workspace root (`<workspace_root>/quantization/<model>/quantized`) into the
  prompt automatically.
- **Structured path for UI/backends**: instead of free text, pass
  `--quantize-scheme <enum>` (one of `none` / `fp8` / `ptpc_fp8` / `mxfp4` /
  `mxfp4_fp8`); `mxfp4` / `mxfp4_fp8` are **MI355X-only**. It resolves to a
  curated prompt internally (`src/hyperloom/orchestrator/phases/quantization_schemes.py`). `none` or
  omit = no quantization. Free-text `--quantize` takes priority when both given.
- **Keep `--precision` consistent with the quantization.** When a quantization
  scheme is requested, also set `--precision`/`PRECISION` to that scheme (e.g.
  `--quantize-scheme fp8` → `--precision fp8`). Otherwise the
  benchmark configs, display names, and the optimization report carry the stale
  operator-supplied precision label (e.g. `fp8`/`bf16`) and **mislabel** an
  actually-quantized model. Never leave a conflicting precision when quantizing.
- Behavior: one-shot, **never runs on a resume**. On a failed/unusable
  quantization the run **hard-stops (`SystemExit(3)`)** — it never silently
  optimizes the un-quantized source after an explicit `--quantize`.
  The one exception is a **pre-flight scheme/GPU mismatch** via
  `--quantize-scheme` (e.g. `mxfp4` on a non-MI355X target): this is **skipped**
  (not a hard stop) and continues on the un-quantized model, emitting a
  `QUANTIZATION_SKIPPED:` line on stdout and setting
  `$HYPERLOOM_QUANTIZATION_SKIPPED` so the caller can detect it.
- Prerequisites (in addition to the normal Setup): `$QUARK_ROOT` must point at
  a Quark checkout containing `.claude/skills/quark-torch-*`, and the installed
  `amd-quark` package version must match that checkout (install editable from
  `$QUARK_ROOT` to keep them consistent). Claude SDK auth is the same
  `ANTHROPIC_*` env the rest of the loop uses.
- After it finishes, the `Quantization prelude: model -> <dir>` line on stdout
  shows the quantized model path that the rest of the run will use; include it
  in status reports.

A user request to optimize a model is approval to run Step 1 on a fresh
node; do not stop for an extra confirmation. After IR-2, smoke-test the
CLI:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/src/hyperloom/agents/kernel"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
# TRACELENS_ROOT: leave unset to let install.sh clone AMD-AGI/TraceLens
# to ${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens@<sha> and pin it
# to a fixed SHA. Only export it as an operator override to point at a
# pre-existing checkout you maintain; this skips both the clone and the
# SHA pin.
# export TRACELENS_ROOT=/path/to/your/TraceLens
# Optional TraceLens-internal checkout; export only to enable it (open-source-only if unset):
# export TRACELENS_INTERNAL_ROOT=/workspace/TraceLens-internal

export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
"$PYTHON" -m hyperloom.inference_optimizer.cli --help
```

Quirks: with `set -u`, assign dependent vars on separate lines (chained
`export A=... B=$A` can fail with `unbound variable`). The installer
leaves a live Ray head; `ray status` must succeed because `trace_analyze`
submits tasks with `num_gpus>=1` — never restart Ray with `--num-gpus=0`.

`_preflight()` runs every launch as the in-loop counterpart of IR-2 and
**owns** the things the launcher must NOT do by hand: re-export auth
aliases (LLM) from `OPENAI_API_KEY`,
auto-`pip install` the SDKs / `ray` / `Magpie` /
`InferenceX`, ROCm hygiene, `--gpu-type` auto-detect, and it emits the
canonical `Preflight diagnostics:` block (paste verbatim into status
reports). Two checks **abort** the run on failure: the model gate
(probed against `<OPENAI_BASE_URL>/models`; the allowlist
{`claude-opus-5` preferred, `claude-opus-4-8`, `claude-opus-4-7`,
`claude-opus-4-6` fallback}
binds only under `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0` — otherwise
the catalog probe is the sole gate; see
`## Failure Handling`) and, when `--critic-agent` is active, the
critic-agent runtime probe (`## Critic Backend Selection`).

Don't manually pip-install SDKs, start Ray,
or `curl /v1/models` — `_preflight()` owns these. See `docs/reference/kernel-execution-path.md`
for the kernel dispatch and artifact layout.

### Recovery

If the CLI exits with `Claude SDK exit code 1` or `Primus.00009 token not present`,
the gateway rejected the request. Check that `OPENAI_BASE_URL` / `OPENAI_API_KEY`
are set in `.env` (or the calling shell) and that the gateway is reachable:

```bash
curl -sS -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" | head
```

If `_preflight()` itself fails, run install in `--check-only` mode to see
which piece is missing, then re-run full install:

```bash
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh" --check-only
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
```

If install repeatedly fails while building GEAK / `mini-swe-agent` with
missing files such as `src/minisweagent/...`, the workspace-shared GEAK
mirror may be half-created (`.git` exists but `src/` is incomplete) or
the filesystem may be showing stale metadata. Do not manually clone GEAK,
delete only `build/`, or edit `source-mirrors/` in place. Stop any other
installer using the same `$USER_DATA_PATH`, remove the entire
`${HYPERLOOM_ROOT:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/source-mirrors}/geak`
directory, then rerun the full install so `install.sh` owns the fresh
clone. Multiple concurrent installs sharing one `$USER_DATA_PATH` also
share `source-mirrors/`; avoid running them at the same time.

In sandboxes where `/workspace/hyperloom` is unwritable, override the
**workspace root** with `USER_DATA_PATH` (not the per-session subdir):

```bash
export USER_DATA_PATH="/shared/hyperloom-sessions"   # workspace root
mkdir -p "$USER_DATA_PATH"
```

The CLI calls `make_session_dir(model_name=…)` once at startup; that
creates `$USER_DATA_PATH/<model_basename>/<UTC_ts>/` and pins
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`.

## Portable Preflight

Implements **IR-1**. Run order is always **IR-2 → IR-1 → launch**:
IR-2 must complete first so `torch` is available. Run the preflight tool
and abort on any non-zero exit before spawning the optimizer:

```bash
"$PYTHON" "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/preflight_optimizer.py" "$MODEL_PATH"
```

A non-zero exit means the GPU state is unknown or a violation was detected;
do not continue to `python -m hyperloom.inference_optimizer.cli optimize`.

## Benchmark Config

Default configs live here:

```bash
src/hyperloom/inference_optimizer/assets/configs/baseline_sglang.yaml
src/hyperloom/inference_optimizer/assets/configs/baseline_vllm.yaml
src/hyperloom/inference_optimizer/assets/configs/profile_sglang.yaml
src/hyperloom/inference_optimizer/assets/configs/profile_vllm.yaml
```

Two fields in each YAML are **fallback only** — the optimizer overrides
them at runtime:

- `benchmark.model` <- `--model` / `$MODEL_PATH`
- `benchmark.runner_type` <- `--gpu-type` / `$GPU_TYPE` / rocm-smi auto-detect

`benchmark.benchmark_script` is deliberately NOT set in the shipped
YAMLs. At materialize time Hyperloom pins it to
`{framework}_{runner_type}.sh` (e.g. `sglang_mi300x.sh` /
`sglang_mi355x.sh`) so Magpie's resolver hits priority 1 (explicit
user override) and uses the generic script — which respects
`RESULT_DIR` and `EXTRA_*_ARGS`. Each shipped YAML has a commented
`# benchmark_script: ...` template right under `framework:` for manual
debug overrides; Orchestration can also route per-task via
`params.benchmark_script` (sanitized).

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.
- `benchmark.envs.PATH`: must lead with the launcher Python's bin dir
  (`$(dirname "$PYTHON")`).

### Magpie leak-path salvage (`INFERENCE_OPTIMIZER_RESCUE_PATHS`)

In-loop, defense-in-depth — the launcher does not touch this. Magpie
shell wrappers hardcode artifacts under `/workspace/`
(`inferencex_result.json`, `server.log`, `gpu_metrics.csv`,
`profile_*.trace.json.gz`). When a task's in-workspace search finds no
usable measurement, the executors run an mtime-gated salvage pass over
`$INFERENCE_OPTIMIZER_RESCUE_PATHS` (unset = no salvage) and copy
fresh matches into the task workspace, tagged in `nonfatal_warnings`
(`rescued_from_leaked_path:` / `harvested_leaked_artifact:`). Extend the
scan roots via `$INFERENCE_OPTIMIZER_LEAK_ROOTS` if a script leaks
elsewhere; the default `{framework}_{runner_type}.sh` already respects
`$RESULT_DIR` so salvage normally never fires.

Operators only interact through two `task.params` knobs:
`params.benchmark_script` (bare
sanitized `*.sh` name; overrides the gpu_type auto-pick) and
`params.result_dir` (forwarded as `$RESULT_DIR`). A baseline retry after a
failure MUST change at least one of `params.benchmark_script` /
`params.result_dir` / `params.extra_server_args` / `params.extra_envs`
(prompt RULE F1 — LLM-side judgement, not a PolicyGate deny); a proposal
repeating a recent failing params fingerprint is dropped as a duplicate.
Three consecutive baseline failures with no enablement engaged stop the run
with `stop_reason='baseline_failed'` and route PRELUDE to CLOSE.

Operator server flags have one supported CLI entry point:
`optimize --server-args "<framework serve flags>"`. The CLI exports this as
`INFERENCE_OPTIMIZER_SERVER_ARGS`, and YAML materialization routes it into
`EXTRA_VLLM_ARGS` / `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS` for baseline,
profile, explore, and sweep. Explicit `--max-model-len` / `$MAX_MODEL_LEN`
wins over auto `ISL+OSL+headroom`. A comma `$CONC` value such as
`4,16,128` is accepted for compatibility; baseline uses the first value.
Use `--conc-sweep-concs` to override the ladder SWEEP measures (`256,128,64,32,16,8,4,2` synthetic, `1,4,8,10,14,20,28` under AgentX).

Operator server flags are the workload baseline, but they are not sacred. When
the configuration arm has evidence or an operator hint that a pinned flag may
be harmful, it
may test an ablation variant with `remove_args` (or `unset_envs` for inherited
environment variables). Do not simulate deletion by adding an unrelated
counter-flag: emit an explicit explore grid entry such as
`{"name": "remove_cuda_graph_max_bs", "remove_args": ["--cuda-graph-max-bs"]}`.
The executor removes those inherited args before appending the variant's
`extra_args`, then records the removal fields in `explore_search` for dedup and
audit.

### Workload-contract reuse (baseline → explore/sweep)

`baseline` materializes its YAML once with the operator's process env
(`CONC` / `ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL`
/ `ROCR_VISIBLE_DEVICES` + adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it
as `baseline_config.with_envs.yaml`, and forwards the path as
`task.params["config_path"]` to every `explore` / `sweep` task — so variants
benchmark the **same workload baseline ran** (without it they'd fall back to
the YAML's fallback defaults `CONC=64`/`ISL=1024`/`OSL=1024`). Per-variant
`extra_envs` still win (applied last).

## Critic Backend Selection

The Critic role has two backend modes. Default is `--critic-agent` (no
flag needed).

| Flag | Backend class | Behaviour |
|---|---|---|
| (none) / `--critic-agent` | `CriticAgentBackend` | Drives the `hyperloom.agents.critic` skill runtime via `python -m hyperloom.agents.critic.runtime.cli prepare-review` → Codex chat completion → `python -m hyperloom.agents.critic.runtime.cli commit-review`. Adds KB priors lookup (with circuit-breaker for unreachable services), per-session memory + idempotent `reviewed_msg_ids` (no double-verdict), `judge_bundle.review_constraints` injected into the LLM prompt, and `needs_review` / `critic_unavailable` source when context is missing. |
| `--critic-mock` | `MockCriticBackend` | Always-approve adapter. Use for offline / smoke tests when Codex creds aren't available. |

Default is overridable per pod via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` (one of `mock` / `agent`).

### Review transport: `--critic-protocol {auto,openai,anthropic}`

Both transports run the full critic-agent runtime; only the review reasoning
call differs. There is no degraded critic, so a transport whose credentials are
missing fails at startup rather than silently changing the review quality.

`auto` (the default) picks `anthropic` for an Anthropic-only config and `openai`
otherwise. Force `anthropic` when both sides are configured but the Critic
should run on a Claude subscription; `auto` would otherwise choose the OpenAI
transport.

Per-value behaviour and the exact credential each one accepts are in
[references/critic.md](references/critic.md#review-transport---critic-protocol-autoopenaianthropic).
That table lived here in duplicate and had already drifted from the code once,
so it is kept in one place.

### Required env when `--critic-agent` is active

| Var | Purpose | Default |
|---|---|---|
| `CRITIC_AGENT_ROOT` | Path to the directory containing `runtime/cli.py`. | in-tree `$REPO_ROOT/src/hyperloom/agents/critic/` |
| `CRITIC_KB_CLIENT_MODE` | `inmemory` keeps KB writes / reads off the wire. `live` requires `KB_BASE_URL`. | `inmemory` |
| `KB_BASE_URL` | KB service URL when `CRITIC_KB_CLIENT_MODE=live`. | unset (live mode aborts at start if absent) |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | Forwarded to the runtime; see `src/hyperloom/agents/critic/README.md`. | runtime defaults |
| `CRITIC_SESSION_MEMORY_DIR` | Where the runtime persists per-session decisions / reviewed_msg_ids. | `$SESSION_DIR/critic-session-memory` (auto-set by the optimizer; co-located with the Coordinator session and cleaned up alongside it). |
| `WORKSPACE_PATH` | Skill root the critic-agent runtime resolves prompt assets against. | `$REPO_ROOT` (auto-set). |

CLI startup checks that `CRITIC_AGENT_ROOT` resolves to a real directory
with `runtime/cli.py`, then runs `python -m hyperloom.agents.critic.runtime.cli --help`
(90s timeout, override via `CRITIC_AGENT_PROBE_TIMEOUT_SEC`) before the
Coordinator boots. Missing or broken runtime aborts
the run with a clear error pointing at `--critic-mock` as the offline
bypass.

### Per-turn artefacts (audit trail)

Each Critic turn writes a 6-digit workdir under
`$SESSION_DIR/critic-workdir/<turn_idx>/` (`request.json` /
`judge_bundle.json` / `review.json` / `emit.json`) plus session memory
under `$SESSION_DIR/critic-session-memory/<session_id>/`. The backend
prunes to the latest 50 turn workdirs each tick. Inspect these when
debugging critic verdicts (see `## Failure Handling`).


## Framework Selection

A session is single-framework. Pick `sglang` (default), `vllm`, or
`atom` via `--framework` or `$FRAMEWORK`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
python3 -m hyperloom.inference_optimizer.cli optimize --framework atom --model "$MODEL_PATH" --max-hours 2  # IR-8 single-node only
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to —
  `baseline_{sglang,vllm,atom}.yaml` and
  `profile_{sglang,vllm,atom}.yaml`. The per-framework resolver
  `_default_profile_config()` in `src/hyperloom/orchestrator/actions/executors/profile.py` picks
  the right file from `$FRAMEWORK`.
- Which framework-specific seed grid the `explore` action falls
  back to when no `params.grid` is supplied. atom is the only
  framework with a programmatic seed today
  (`_default_grid_for_framework("atom", ...)` in
  `src/hyperloom/orchestrator/actions/executors/explore.py`, populated by
  `_atom_default_grid()`); sglang and vllm continue to rely on
  the orchestration LLM emitting `provenance='default_grid'`
  variants and will fail with `error_class="empty_grid"` on a
  cold-start with no LLM input.
- Which extra-args env name `_grid_runner` writes
  (`EXTRA_VLLM_ARGS` / `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS`)
- Which KB partition orchestration reads for hints

Mixing frameworks in a single session is not supported; the CLI
locks `$FRAMEWORK` for the run. Resume re-reads `$FRAMEWORK` from the
shell — set it when you resume a non-default session.

**`--framework atom` specifics (IR-8):** single-node only
(`--nodes>=2` fails fast). Shipped configs `baseline_atom.yaml` /
`profile_atom.yaml`; the Magpie atom wrapper bridges `PROFILE=1` to
atom's `--torch-profiler-dir`, and TraceLens consumes the resulting
`*.pt.trace.json.gz` unchanged. atom source roots (`/app/ATOM/atom/`)
are in PolicyGate's allowlist + `_REUSABLE_SOURCE_ROOTS`, and the repo
URL `https://github.com/ROCm/ATOM.git` is in `hyperloom.agents.framework.repo_map`.
Unlike sglang/vllm, atom is the only framework with a programmatic
cold-start seed grid (`_atom_default_grid`: `atom_level_{2,3}`,
`atom_prefix_cache`, `atom_kv_fp8` on FP8, model-class-gated `atom_ep` /
`atom_dp_attn` / `atom_mtp_{1,3}`, `atom_cudagraph_bracket`) — sglang/vllm
fail `error_class="empty_grid"` on a cold start with no LLM variants.

## Enablement Targeted Builds

When a run needs a compiled component that only exists in source (AITER
FP4/MLA/NSA kernels, `sgl-kernel`, or vLLM-from-source), the enablement
subsystem acquires it off-loop. The launcher does not drive this; the notes
below describe the runtime behavior operators observe.

- **Off-loop build lane.** Compiled-component builds run on a dedicated
  single-slot build lane. Each build is spawned as a detached process group
  and polled (reaped) across coordinator ticks against a wall-clock budget, so
  a multi-hour compile never blocks the tick loop. An in-flight build is tracked
  by a durable sentinel (`pending_targeted_build`) so a crash/resume can recover
  or terminate it.
- **Runnable gate.** A verified build does not earn KEEP by
  artifact-verification alone — it must actually launch the model through a
  launch probe (the enablement runnable-decision gate) before it is kept.
- **Interpreter switch.** A from-source build records the venv interpreter it
  was compiled against (`runtime_python_exe`) and emits it as the
  `HYPERLOOM_FRAMEWORK_PYTHON` env into the per-variant YAML benchmark envs
  (`benchmark.envs`). Both backends export that mapping to the server env, and
  bypass additionally launches the server via `python -m` with that
  interpreter. This guarantees the server loads the exact build.
- **`build_budget_sec`.** Per-build-action wall-clock timeout knob; `0` selects
  the per-component default.

See `docs/reference/environment-variables.md` "Targeted builds (Rung 5)" for the
full env-var set, and `docs/reference/session-breakdown.md` for the emitted
`enablement` / `build_attempts[]` fields.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without
either, the optimizer auto-detects via `rocm-smi --showproductname`
(falling back to `torch.cuda.get_device_properties(0).gcnArchName`).

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi308x`, `mi325x`, `mi355x`. **`mi308x` and
`mi325x` map to `runner_type=mi300x`** with a warning, since the GPUs share the
same runner family and Magpie has not shipped `sglang_mi308x.sh` /
`sglang_mi325x.sh` / `vllm_mi308x.sh` / `vllm_mi325x.sh` yet. If you
need a true MI308X/MI325X-specific script, uncomment the `benchmark_script:`
template in the relevant YAML and point it at your script under
`InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.

## SGLang Parameter Search

Serving-parameter search runs through the `explore` action (the legacy
`params` / `backends` actions were merged into it); candidates are
written via `EXTRA_SGLANG_ARGS` / `benchmark.envs`. This is internal to
the optimizer — the launcher does not drive it. Useful InferenceX-derived
candidate families a specialist may surface: `--disable-radix-cache`,
`--max-running-requests`, `--tokenizer-worker-num`, `--stream-interval`,
and ROCm/TileLang envs (`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`,
`SGLANG_HACK_FLASHMLA_BACKEND=tilelang`). Speculative decoding
(`SGLANG_ENABLE_SPEC_V2` / `--speculative-*`) is model-specific — only
where a draft/MTP path exists, benchmarked with chat-formatted prompts.

### Per-Run Asset Override (advanced)

To override shipped configs without editing them, materialize a per-run asset
root and `export INFERENCE_OPTIMIZER_ASSET_ROOT="$ASSET_ROOT"` (env var, not a
CLI flag; `asset_root()` raises `AssetRootNotFound` if the dir is missing).
`mkdir -p "$ASSET_ROOT/assets/configs"`, `ln -sfn` `actions/` from
`$REPO_ROOT/src/hyperloom/inference_optimizer/` and `orchestrator/` from
`$REPO_ROOT/src/hyperloom/`, then
copy + edit the relevant `baseline_*.yaml` / `profile_*.yaml`. Reach for this
only when `_workload_envs.materialize_config_with_envs` defaults don't fit
(e.g. per-yaml `profiler.torch_profiler.enabled`); otherwise `--model` /
`--gpu-type` overrides are enough.

## Launch a New Optimization

Assumes Step 1 (install) already ran. Set `$USER_DATA_PATH` to the **workspace
root** (parent of per-session dirs). The CLI creates
`$USER_DATA_PATH/<model_basename>/<UTC_ts>/` via `make_session_dir`.
Launcher stdout / PID files go under that session's `optimizer_runs/`.
For sandboxes that don't persist `export`s across shell calls (Cursor agents),
copy `src/hyperloom/inference_optimizer/assets/setup_env.sh.example` to a
**session-scoped** path:
`$USER_DATA_PATH/optimizer_runs/setup_env_${CLAW_SESSION_ID:-$(date +%s)}.sh`,
fill in the workload block, and `.` it each call.

**IMPORTANT**: never use a shared filename like `setup_env.sh` — concurrent
sessions on different pods share `$USER_DATA_PATH` via WekaFS; a single file
causes MODEL_PATH race conditions where sessions launch the wrong model.
After `setsid nohup ... &`, locate the optimizer via
`pgrep -af 'hyperloom.inference_optimizer.*optimize'` — `$!` may be a wrapper PID.

```bash
cd "$REPO_ROOT"
# .env fills gaps only: re-exporting the non-empty pre-source snapshot keeps every
# value the caller exported. Wider than install.sh, which guards a fixed list.
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
eval "$_dotenv_prev"
unset _dotenv_prev
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
# RUN_LOG/PID/launch-info live under the workspace until the session_dir
# is known; move or re-tail from $session_dir/optimizer_runs/ after reading
# session_dir from the launch-info JSON below.
# /workspace/hyperloom is only the fallback when $USER_DATA_PATH is unset.
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
mkdir -p "$RUN_DIR"

setsid nohup python3 -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --launch-info-file "$RUN_DIR/launch_${RUN_TAG}.json" \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

`setsid nohup ... &` is required for runs > 5 min — Cursor's background
shell can die on SSH disconnect.

Critic defaults to `--critic-agent`; Robustness defaults to `--robustness-agent`.
See [Critic Backend Selection](#critic-backend-selection) for `--critic-mock`;
pod-level overrides via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` /
`INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND`.

After launching, do a short health check:

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"
# Authoritative session dir from the launch-info JSON (--launch-info-file).
# Never guess by timestamp: overlapping sessions break any "latest dir" pick.
launch_info="$RUN_DIR/launch_${RUN_TAG}.json"
session_dir="$(jq -r '.session_dir // empty' "$launch_info" 2>/dev/null)"
if [ -z "$session_dir" ]; then
  echo "ERROR: no .session_dir in $launch_info (launch-info JSON missing or" \
       "malformed). The optimizer likely died before emitting launch info;" \
       "inspect the HYPERLOOM_LAUNCH line and errors in $RUN_LOG." \
       "Refusing to guess the session dir from timestamps." >&2
  return 1 2>/dev/null || exit 1
fi
test -f "$session_dir/manifest.json" && echo "manifest_present=true session_dir=$session_dir"
test -f "$session_dir/state.json" && echo "state_exists=true" \
  && python3 -c "import json; print(json.load(open('$session_dir/state.json')).get('stop_reason'))"
```

Healthy = optimizer process alive + `manifest.json` + `state.json`
exist + no early `stop_reason`.

## Resume Existing Session

`--resume-from "$SESSION_DIR"` is the only way to resume; there is no
flag that lets the CLI choose a session for you. Get `$SESSION_DIR` from the
authoritative sources listed under the **Session rule** above — never by
walking `$USER_DATA_PATH/<model>/<UTC_ts>/` for the newest dir.
`$USER_DATA_PATH` must stay at the **workspace root** so
`runtime/kernel-agent.env.sh` resolves. The CLI refuses to start if
`manifest.json` or `state.json` is missing in that session dir.

Reuse the Launch template above with these diffs: drop `--model`, add
`--resume-from "$SESSION_DIR"`, set `RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`.
Resume preserves baseline, current best, params-search state, event history,
and kernel-agent artifacts; the CLI clears stale `stop_reason` and
`crash_count` before retrying.

**Most of the launch shape does not need re-passing.** `state.json` is the
authority for it, so a bare `--resume` keeps `--server-args`, every
`--extra-env` pin, the robustness flags (including
`--robustness-disable-server-probe`) and the warm-replay gates from the original
launch — the CLI re-exports the derived env from the persisted state and prints
each one it restored. Re-pass a flag only to *change* it: an explicit flag on
the resume wins and is persisted as the new value for later resumes. This is
what lets `robustness_monitor.sh` auto-resume a crashed run without knowing the
original command line.

Three exceptions:

- **`--nodes` must be re-passed for a multi-node resume**, together with
  `--mn-backend` / `--gpus-per-node` if they were set and the
  `HYPERLOOM_MN_EXT_*` hand-off. The persisted count only feeds the robustness
  defaults and the IR-8 check; the cluster hand-off is resolved from argv before
  `state.json` is read, so a bare `--resume` of a `--nodes >= 2` session
  benchmarks against the wrong endpoint.
- `--reference-script` is fixed at launch — the recipe is parsed into
  `state.json` once, so re-passing it on a resume does nothing. Start a fresh
  session to change the reference.
- `--server-args` and `--extra-env` are restored as a **set**: re-passing either
  replaces the whole thing, so changing one pin means re-passing them all.
  Robustness flags layer per-key instead.

## Robustness Monitor for Long Runs

For runs > 5 min, start a monitor in its own `setsid nohup` process. It polls
`state.json` every 5 min, exits without resuming when the session is terminal
(any `stop_reason` in `STOP_REASON_VOCAB`, `phase=CLOSE`, or
`reports/final.md` present — including failure sentinels like
`baseline_failed`), and resumes via `--resume-from` only when the optimizer
dies without those markers (unexpected crash).

```bash
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
mkdir -p "$RUN_DIR"
# Point the monitor at the authoritative session dir: it reads
# $INFERENCE_OPTIMIZER_SESSION_DIR first, else .session_dir from the
# launch-info JSON in $LAUNCH_INFO_FILE (written by --launch-info-file).
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
cp "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/robustness_monitor.sh.example" \
   "$RUN_DIR/robustness_monitor.sh"
chmod +x "$RUN_DIR/robustness_monitor.sh"
setsid nohup bash "$RUN_DIR/robustness_monitor.sh" \
  > "$RUN_DIR/robustness_monitor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 < /dev/null &
```

Reads `$PID_FILE` plus (optional) `$INFERENCE_OPTIMIZER_SESSION_DIR` /
`$LAUNCH_INFO_FILE` / `$MAX_HOURS` / `$TARGET_GAIN`. The session dir comes
from `$INFERENCE_OPTIMIZER_SESSION_DIR` when set, else from `.session_dir`
in the launch-info JSON at `$LAUNCH_INFO_FILE` (never from a timestamp
guess). Edit the example before copying if defaults need to change.
`stop_reason` interpretation matches the `## Monitoring` reader.

## Monitoring

Each poll is one fast read of persisted state, and you poll only when you are
next invoked. **Never block to reach the next poll**: no `sleep`, no wait loop,
no `tail -f`. The only sanctioned wait is the one-shot `sleep 30` health check
right after launch. Recurring 5-minute polling is the Robustness Monitor's job
and already runs in its own `setsid nohup` process (see above) — do not
reimplement it here.

**Why this is load-bearing, not style.** A blocking call holds the sandbox
connection open for its whole duration. Overlap it with a `roofline` trace flush
— 8 ranks serialising torch traces to NFS, which takes minutes on a large TP=8
MoE — and the connection can drop; the agent harness reads that as an
unreachable sandbox and rebuilds it, which kills the optimizer and the in-flight
profile with it. Session `Kimi-K3_20260818T062936Z_a62853e5` lost 3/3 rooflines,
its `analysis.md`, and its whole KERNEL phase to a `sleep 110; sleep 110;
sleep 80` progress loop. Launch, check once, hand off to the monitor, report.

Resolve `$SESSION` the same way the Robustness Monitor does — never from
`$USER_DATA_PATH`, which is the workspace root, not the session dir.

```bash
export SESSION="${INFERENCE_OPTIMIZER_SESSION_DIR:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_dir"])' "$LAUNCH_INFO_FILE")}"
python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/read_optimizer_state.py" "$SESSION"
```

It prints `stop_reason`, `baseline_tput`, `cumulative_gain_validated`, `current_best`,
`last_kernel_opt`, `last_trace_analyze`, `last_conc_sweep`, `explore_last_round`,
`phase`, plus the recent lifecycle events.

Recent action counts from SQLite (last 500 events grouped by category):

```bash
python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/event_counts.py" "$SESSION"
```

## Expected Flow

The optimizer should:

1. Establish or reuse `baseline_tput`.
2. **Coordinator** auto-enqueues an analysis task at the end of
  PRELUDE (after baseline) and at each validated-tput watermark
  (`current_tput / last_roofline_tput >= 1.10`; compound). Default is
  `roofline` (profile + trace_analyze + analysis.md); `--no-enable-roofline`
  switches to plain `profile`. The LLM cannot propose either —
  both names are Coordinator-managed and absent from
  `PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 returns
  `rule='phase_incompatible'`. Concurrent GPU work is
  serialised by the lane / GPU lease rather than a policy deny, so
  explore / kernel dispatches keep flowing while analysis refreshes.
  Each analysis also stamps a decode roofline ceiling
  (`src/hyperloom/orchestrator/kernel/roofline_ceiling.py`) for the report's
  `## Roofline Comparison` section.
3. Run `trace_analyze` once per trace/config and cache the result in
  `last_trace_analyze`.
4. Pick only `reusable_native_kernel_ids` for `run_optimization`.
5. Require compile + correctness + microbench/E2E evidence before KEEP.
6. Use `explore_search` to test parameters incrementally and remember
  rejected candidates across resume. The ledger keys entries by
  **content fingerprint** (a sha1 hash of sorted `extra_server_args` +
  sorted `extra_envs`), so renaming an already-tested variant does not
  bypass dedup — LLM-supplied `params.grid` is filtered through the same
  ledger as the default seed grid.
7. Use `optimization_stack` so backend + params + kernel changes do not
  overwrite each other.
8. Use `sweep` to understand workload-specific results beyond the smoke
  workload.

## Cache Topology & Cold-start Discipline

SGLang/vLLM on ROCm route hot fused kernels (RMSNorm / attention / MoE / GEMM /
RoPE) through `aiter`, which JIT-compiles per-shape variants on first sight
and caches `.so` on disk. First launch of a fresh (model, dtype, TP,
`max_model_len`, `max_num_seqs`, `gpu_memory_utilization`) signature can spend
30+ min in `hipcc` for 671B FP8 MoE; later launches reuse the cache in seconds.

### Cache locations

| Cache | Path | Clear |
|---|---|---|
| aiter JIT (primary cold-start cost) | `<aiter pkg root>/jit/` (resolved via `import aiter`; wheel installs hold ~80 pre-built `.so` here, plus runtime-JIT staging under `jit/build/<module>/build/`) | `rm -rf <aiter pkg root>/jit/build/` (clears JIT staging only; do NOT delete `jit/*.so` — those are wheel-bundled) |
| Triton | `~/.triton/cache/` (resolves via `$HOME`) | `rm -rf ~/.triton/cache` |
| torch.compile / Inductor | `/tmp/torchinductor_<user>/` (override `$TORCHINDUCTOR_CACHE_DIR`) | `rm -rf /tmp/torchinductor_root` |

`sgl_kernel` (`site-packages/sgl_kernel/common_ops.*.so`) is build-time only;
only `kernel_opt` / `integrate` may rebuild it.

### Cold-start triggers

First launch on this pod; change to `--max-model-len` / `--max-num-seqs` /
`--gpu-memory-utilization` / `--cuda-graph-max-bs` / `--quantization` /
`--enable-torch-compile`; pod rebuild; manual cache `rm`; aiter source patch.

### Auto-detection + timeout

The baseline executor counts aiter `.so` files (**< 20 = COLD**)
and picks a subprocess timeout accordingly: COLD → 9000s, WARM → 7800s
(`task.params['timeout_sec']` always wins). The profile executor inherits
the same probe with a 14400s warm default, so a COLD probe there lowers the
cap to 9000s. Each launch logs a
`baseline_executor: ...` marker and the cache state lands in the
`Preflight diagnostics:` block. If COLD_START repeats across retries the
JIT was killed mid-`hipcc` — bump
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC` above its 9000s default (e.g.
`=12000`; it replaces the cold cap, so a smaller value shortens it).
Override the probe dir via `INFERENCE_OPTIMIZER_AITER_JIT_DIR`.

## Kernel Apply Safety

Kernel optimization may modify `/sgl-workspace/aiter`, `/sgl-workspace/sglang`,
or compiled artifacts. Before applying a patch:

- Back up source files.
- Back up compiled `.so` / `.co` artifacts when available.
- On REVERT, restore compiled artifacts first, then source files, then restart
  the server. Avoid a rebuild on revert when the original compiled artifact was
  backed up.
- Only KEEP when correctness and E2E are acceptable.

If the user has not explicitly approved environment mutation, stop before real
apply/rebuild and ask. Dry-run and analysis are safe.

## Kernel E2E Retry Discipline

Microbench speedups are not enough. After `run_optimization` returns a candidate
kernel patch, `integrate` must validate the patch with E2E Magpie throughput and
record every attempt in `state.json`.

For the same `kernel_id + patch_path + EXTRA_SGLANG_ARGS`:

- `KEEP`: accept only when E2E gain clears the configured threshold.
- `REVERT`: reject that patch immediately and do not run it again.
- `NEEDS_REVIEW`: allow at most 3 E2E attempts. If none clears the KEEP
  threshold, reject that patch and move on to params search or a different
  reusable native kernel.

Do not repeatedly integrate the same patch because its microbench was strong.
If E2E results are unstable around zero gain, the correct action is to mark the
patch rejected, preserve the artifacts for human review, and spend the remaining
budget on untested params/backend candidates or the next kernel.

## Failure Handling

Auth / SDK drift (`Claude SDK exit code 1`, `Primus.00009 token not present`,
`ANTHROPIC_AUTH_TOKEN not set`, `BackendError: claude-agent-sdk not installed`,
`Fatal error in message reader`) is owned by `_preflight()`; see
`## Setup → Recovery` for the supervisor + install rerun loop. Manual SDK
fallback if frozen pip blocks `_ensure_python_sdks()`:
`python -m pip install 'claude-agent-sdk>=0.2.110' 'openai>=1.50' 'httpx>=0.27'`.
Transient SDK errors retry/resume up to the Coordinator emergency threshold.

### Model-gate errors (preflight #10)

Custom orchestration models are enabled by default and are validated against the
configured gateway catalog. Set `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`
only when you intentionally want the strict AMD Claude allowlist
(`claude-opus-5` / `claude-opus-4-8` / `claude-opus-4-7` / `claude-opus-4-6`).

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | You likely set `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`; unset it or set it to `1`, then ensure the model appears in the gateway `/models` catalog. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` for offline / smoke runs. See
`## Critic Backend Selection`.

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/src/hyperloom/agents/critic`, or check the `src/hyperloom/agents/critic/` install. |
| `hyperloom.agents.critic.runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `src/hyperloom/agents/critic/README.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `hyperloom.agents.critic.runtime.cli ... timed out after 30s` | KB stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory`. Reproducing in `inmemory` is a bug — that path must not block on I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `trace_analyze` with unchanged trace/config: bug — reuse `last_trace_analyze`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must contain explicit correctness evidence.
- `stop_reason=no_more_leverage`: stop and report; only resume if the user changes workload / search space / model / strategy.
- `stop_reason=policy_loop`: a legacy stop_reason kept in the vocabulary for resuming old sessions; nothing in the runtime sets it. Repeated `policy_denied` for the same (action, rule) pair is advisory only — there is no auto-prune at streak ≥5 and no `policy_loop` stop at streak ≥10. Inspect `SharedState.policy_denial_history` via the `why_denied` tool or the `=== Recent policy denials ===` block, then change something substantive (a new `params.grid` variant, a different `benchmark_script`, or a sibling action family). Do not hand-edit `state.json`.
- `stop_reason=time_exhausted`: resume same session (`--resume-from`); do not start fresh.

## Report Back To User

Report concise status:

- session id (from `manifest.json`) and log path
- `cumulative_gain_validated` and `current_best`
- explore accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
