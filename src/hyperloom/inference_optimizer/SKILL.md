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
- Kernel: responder path for `trace_analyze`, `run_optimization`, `integrate`.
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

**Claw / SaFE pods:** the launcher often sets ``$USER_DATA_PATH`` to a
run-scoped path *before* the optimizer starts, e.g.
``/hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/``.
That outer directory is **platform isolation** (one Claw job). The
optimizer then creates ``<model_basename>/<UTC_ts>/`` inside it. Full
session path example::

    /hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/   ← USER_DATA_PATH (Claw)
        deepseek-ai-DeepSeek-V3/20260522T035359Z/                      ← session_dir (optimizer)

**Legacy flat layout:** set ``INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat``
so ``session_dir == workspace_root`` (no ``<model>/<ts>`` subdirs).

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
`${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}/TraceLens`; when unset,
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
(GEAK RAG embedding cache), `/wekafs/hyperloom/geak-memory/memory.db`
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
`--resume`), verify that every visible GPU on this pod has **zero
foreign serving PIDs and ≲ 500 MiB VRAM in use**. A leftover
`sglang.launch_server` / `vllm.entrypoints` / `Magpie` from a previous
run silently degrades the next `baseline` by 5–30 % (shares VRAM +
schedules on the same XCD); `current_best` cannot detect this
pollution after the fact.
> Inside a running session, the equivalent guard is Kernel-agent IR-4
> (`kill_server` + `check_gpu_memory` before every server (re)start —
> see `src/hyperloom/orchestrator/prompts/kernel_agent.md`). IR-1 above is the
> *outer* gate that fires before the optimizer process exists.

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

**Resume carve-out.** `... optimize --resume` may skip install only when
ALL hold: (1) `install.sh` exited 0 earlier in the *same shell*; (2)
`kernel-agent.env.sh` is still sourced; (3) the session being resumed is
known (explicit `--resume-from`, `$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`,
or launch-info JSON) and its `manifest.json` exists under that session dir.
Any failure → treat as fresh launch and re-run `install.sh`.

> The in-loop equivalent is `_preflight()` steps 1–12 (drift repair, not
> a substitute for this outer gate).

### IR-3 — KB + PR Monitor reachability (in-loop, soft degrade)

`_preflight()` invokes:

```
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/preflight_kb.sh"
```

Exit codes (soft degrade — IR-3 never aborts launch):

- `0` → KB + PR Monitor both reachable. `cortex_enabled` / `pr_monitor_enabled` stay `True`.
- `1` → at least one branch unreachable. The cli automatically enables the
  matching `--degraded-*` and continues; `manifest.json` records
  `kb_degraded_reason=ir3_auto` (or `pr_degraded_reason=ir3_auto`).

Operator opt-out: pass `--degraded-kb` / `--degraded-pr` to skip the
corresponding probe (one round-trip saved); `manifest.json` then
records `reason=explicit_flag`. Both flags together short-circuit the
entire IR-3 step.

### IR-4 / IR-6 / IR-7 — EXPLORE phase contracts (Coordinator-internal)

These govern the optimizer's EXPLORE phase, not the launcher; the full
contract lives in `src/hyperloom/orchestrator/prompts/orchestration.md`. In
brief:

- **IR-4 — EXPLORE is specialist-informed**: prefer specialist- or
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
- **IR-6 HARD force-exit**: EXPLORE exits the moment wall-clock remaining
  < `--explore-force-exit-hours-remaining` (default 3.0 h) OR phase
  budget < `--explore-force-exit-budget-pct` (default 20%). Non-negotiable
  — leaves buffer for KERNEL_AGENT → SWEEP → CLOSE + report.
- **Plateau advisory**: EXPLORE / KERNEL_AGENT / FRAMEWORK plateau signals
  are computed every tick and rendered as advisory in the orchestration
  prompt. They do NOT drive phase advance — the LLM may emit
  `escalate_strategy_change{hint='skip_to_kernel'/'skip_to_sweep'/'skip_to_close'}`
  when it judges further effort unproductive. IR-6 force-exit and the
  per-phase budget remain the only hard advance gates.

### FRAMEWORK_AGENT phase (Coordinator-internal)

Inserted between PRELUDE and EXPLORE (`--no-framework-agent` opts out). The
Coordinator owns the loop end-to-end — the LLM never proposes the
`framework` action. It discovers a candidate batch **once** via
`fa phase-discover`; then each exploration processes exactly **one**
candidate, with the agent ranking the still-available candidates and
picking the one most likely to raise throughput (LLM ranker, with a
deterministic discovery-order fallback). The chosen candidate is
Critic-gated, then `git apply`d against the live framework_source_roots
and benchmarked; KEEP commits to the live tree (next candidate stacks on
top), REVERT does `git reset --hard`. Exits on low budget
(<0.6 × max_hours), **plateau (3 consecutive benchmarked candidate tests
with no KEEP**),
or an empty discovery batch. Resume skips completed candidates by
idempotency key. The launcher only chooses whether the phase runs
(`--no-framework-agent`).

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

The live runtime uses `actions/_meta/*.yaml`, `_grid_runner.py`, and the
unified specialist-informed `explore` flow. Do not recreate the retired
`backends` / `params` / `validate_stack` / scoring modules.

Rules that look reasonable but break the current flow:

- **No `framework first-explore priority` rule** in
  `system_prompts/orchestration.md` — conflicts with the EXPLORE
  specialist-informed flow.
  Framework-agent runs in the dedicated **FRAMEWORK** phase
  before EXPLORE; the LLM never proposes the `framework`
  action — it is Coordinator-managed and absent from
  `PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 denies any
  LLM-side propose / delegate with `rule='phase_incompatible'`.
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

The common single-gateway setup uses `SAFE_API_KEY` and `OPENAI_BASE_URL`.
Split-gateway deployments may provide provider-specific `ANTHROPIC_*` /
`OPENAI_*` credentials instead. Shell-exported values win; `$REPO_ROOT/.env`
is loaded only to fill missing values by `install.sh` and the CLI preflight.
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
| `INFERENCEX_PATH` resolution (scans `$MAGPIE_PATH/InferenceX` → `$HYPERLOOM_RUNTIME_DIR/InferenceX`, else clones a fresh writable checkout; read-only host mounts are no longer used) | `ensure_inferencex` |
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
`install.sh`. These are **sandbox-only** — do NOT forward them to the
RayJob via `--rayjob-extra-env`; the RayJob pod has its own paths and
does not consume these.

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

Before launching, produce an **advisory** architecture profile so the
orchestration + specialist prompts carry richer model context than the
coarse `--model-class` tag. This is **best-effort and non-fatal**: a
missing / invalid file simply causes Hyperloom to omit the section — it
never blocks launch, never replaces `--model-class` (still required), and
is always **subordinate to live TraceLens evidence** at runtime (it drives
no deterministic gating — atom seed grid, framework gap token, recipe key,
and prompt label all stay on `model_class`).

Steps for the launching agent:

1. **Gallery lookup** — fetch the LLM Architecture Gallery
   (`https://sebastianraschka.com/llm-architecture-gallery/`) and locate
   the card for the model being launched. Extract the schema fields below.
2. **Fallback classify** — if the model is not in the gallery, do a
   lightweight classify from the model's local `config.json` (decoder
   type, attention variant, expert counts, MTP, SWA window) and set
   `"source": "config_classify"`.
3. **Write the convention file** — write the profile to
   exactly `$USER_DATA_PATH/model_arch.json`: the file named
   `model_arch.json` at the workspace root. Do not create a subdirectory
   such as `model_arch_advisory/`; the CLI only reads the root-level
   convention file. Include `model_name` (required for the stale-file
   guard — its basename must match the launched `--model` basename or
   Hyperloom ignores the file). All other fields are optional; renderers
   drop empty fields.

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
  --framework vllm \           # sglang (default) / vllm / atom / xdit
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
operator states in the prompt (ISL, OSL, CONC, TP, EP, precision, budget) MUST
be forwarded as the matching CLI flag** — these flags are the only source of
truth; an omitted flag silently falls back to its default and the operator's
stated value is lost (issue #903):

| Surface | CLI flag | Notes |
|---|---|---|
| Model path | `--model` | required |
| Framework | `--framework` | `sglang` (default) / `vllm` / `atom` / `xdit` — atom is single-node-only; xdit is scriptable diffusion (`img/s`, no serving server) |
| GPU type | `--gpu-type` | rocm-smi auto-detect when unset |
| Model class | `--model-class` | categorical key for the deterministic consumers (atom seed grid, framework-agent gap search token, recipe key, prompt label); when unset, Coordinator boot infers and persists it from model metadata or model-path family keywords. For richer advisory model context see Step 1.5 (`model_arch.json`) |
| Input seq length | `--isl` | Pass the prompt's ISL. Default `1024` when omitted. |
| Output seq length | `--osl` | Pass the prompt's OSL. Default `1024` when omitted. |
| Concurrency | `--conc` | Pass the prompt's CONC (max in-flight requests). Default `64`. Use `--conc-sweep-concs` for a ladder. |
| Tensor parallel | `--tp` | Pass the prompt's TP. Default `1`. |
| Expert parallel | `--ep` | Pass the prompt's EP for MoE. Default `1`. |
| Precision | `--precision` | Match the checkpoint (`bf16` default / `fp8` / ...). Keep consistent with `--quantize`. |
| Budget | `--max-hours` | Pass the prompt's time budget. Default `2.0`. |
| Max model len | `--max-model-len` | Optional; auto-derived from ISL+OSL+headroom when omitted. |
| External reference GPU | `--compare-against-gpu` | Coordinator *always* hard-gates `target_analysis` as TODO 0 so `$SESSION_DIR/target_analysis/target_baseline.json` exists before `baseline` runs. When this flag is set the JSON carries the InferenceX reference (`reason="ok"`); when unset the JSON carries a structured `reason="no_target_gpu_configured"` marker. The report renders the "External baseline" section from this JSON in both cases (heading switches to "(not requested)" for the marker variant) |
| Quantization prelude | `--quantize` | Optional. Natural-language quantization request. Runs the quantization-agent once before the loop and rewrites `--model` to the quantized model. See Step 2b. Ignored on `--resume`. |

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
- Behavior: one-shot, **skipped on `--resume`**. On a failed/unusable
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
# to $HYPERLOOM_OPEN_SOURCE_ROOT/TraceLens and pin it to a
# fixed SHA. Only export it as an operator override to point at a
# pre-existing checkout you maintain; this skips both the clone and the
# SHA pin.
# export TRACELENS_ROOT=/path/to/your/TraceLens
# Optional internal extension; export only to enable it (open-source-only if unset):
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
aliases from `SAFE_API_KEY`, derive/override `ANTHROPIC_BASE_URL`,
auto-`pip install` the SDKs / `ray` / `Magpie` /
`InferenceX`, ROCm hygiene, `--gpu-type` auto-detect, and it emits the
canonical `Preflight diagnostics:` block (paste verbatim into status
reports). Two checks **abort** the run on failure: the hard model gate
(`--claude-model` ∈ {`claude-opus-4-7` preferred, `claude-opus-4-6`
fallback}, probed against `<OPENAI_BASE_URL>/models`; see
`## Failure Handling`) and, when `--critic-agent` is active, the
critic-agent runtime probe (`## Critic Backend Selection`).

Don't manually pip-install SDKs, start Ray,
or `curl /v1/models` — `_preflight()` owns these. See `src/hyperloom/agents/kernel/SKILL.md`
for the chained installer truth.

### Recovery

If the CLI exits with `Claude SDK exit code 1` or `Primus.00009 token not present`,
the gateway rejected the request. Check that `OPENAI_BASE_URL` / `SAFE_API_KEY`
are set in `.env` (or the calling shell) and that the gateway is reachable:

```bash
curl -sS -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" | head
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
export USER_DATA_PATH="/wekafs/xiaofei/sessions"   # workspace root
mkdir -p "$USER_DATA_PATH"
```

The CLI calls `make_session_dir(model_name=…)` once at startup; that
creates `$USER_DATA_PATH/<model_basename>/<UTC_ts>/` and pins
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`.

## Portable Preflight

Implements **IR-1**. Run order is always **IR-2 → IR-1 → launch**:
without IR-2 the script below has no `torch` to import. Verify the
model path, GPU visibility, and that no stale serving process holds
VRAM; exit non-zero on any violation so the calling shell aborts
before `python -m hyperloom.inference_optimizer.cli optimize` is spawned. Never print tokens.

```bash
export MODEL_PATH=/path/to/model
test -d "$MODEL_PATH"

"$PYTHON" - <<'PY'
import os
try:
    import torch
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])

patterns = ("hyperloom.inference_optimizer.cli", "Magpie", "sglang.launch_server")
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    text = cmd.replace(b"\0", b" ").decode("utf-8", "ignore")
    if text and any(p in text for p in patterns):
        print(f"existing_process {pid}: {text[:300]}")
PY
```

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
`$INFERENCE_OPTIMIZER_RESCUE_PATHS` (default `/workspace/`) and copy
fresh matches into the task workspace, tagged in `nonfatal_warnings`
(`rescued_from_leaked_path:` / `harvested_leaked_artifact:`). Extend the
scan roots via `$INFERENCE_OPTIMIZER_LEAK_ROOTS` if a script leaks
elsewhere; the default `{framework}_{runner_type}.sh` already respects
`$RESULT_DIR` so salvage normally never fires.

Operators only interact through two `task.params` knobs (full schema in
each `actions/_meta/<action>.yaml`): `params.benchmark_script` (bare
sanitized `*.sh` name; overrides the gpu_type auto-pick) and
`params.result_dir` (forwarded as `$RESULT_DIR`). The Coordinator's
`baseline_no_param_change` PolicyGate rule denies any baseline proposal
that changes params after a failure — the agent must retry with
identical params and the run terminates after 3 consecutive failures.

Operator server flags have one supported CLI entry point:
`optimize --server-args "<framework serve flags>"`. The CLI exports this as
`INFERENCE_OPTIMIZER_SERVER_ARGS`, and YAML materialization routes it into
`EXTRA_VLLM_ARGS` / `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS` for baseline,
profile, explore, and sweep. Explicit `--max-model-len` / `$MAX_MODEL_LEN`
wins over auto `ISL+OSL+headroom`. A comma `$CONC` value such as
`4,16,128` is accepted for compatibility; baseline uses the first value.
Use `--conc-sweep-concs` for the explicit sweep ladder.

Operator server flags are the workload baseline, but they are not sacred. When
EXPLORE has evidence or an operator hint that a pinned flag may be harmful, it
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

### Required env when `--critic-agent` is active

| Var | Purpose | Default |
|---|---|---|
| `CRITIC_AGENT_ROOT` | Path to the directory containing `runtime/cli.py`. | in-tree `$REPO_ROOT/src/hyperloom/agents/critic/` |
| `CRITIC_KB_CLIENT_MODE` | `inmemory` keeps KB writes / reads off the wire. `live` requires `KB_BASE_URL`. | `inmemory` |
| `KB_BASE_URL` | KB service URL when `CRITIC_KB_CLIENT_MODE=live`. | unset (live mode aborts at start if absent) |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | Forwarded to the runtime; see `src/hyperloom/agents/critic/AGENTS.md`. | runtime defaults |
| `CRITIC_SESSION_MEMORY_DIR` | Where the runtime persists per-session decisions / reviewed_msg_ids. | `$SESSION_DIR/critic-session-memory` (auto-set by the optimizer; co-located with the Coordinator session and cleaned up alongside it). |
| `WORKSPACE_PATH` | Skill root the critic-agent runtime resolves prompt assets against. | `$REPO_ROOT` (auto-set). |

`_preflight()` checks `CRITIC_AGENT_ROOT` resolves to a real directory
with `runtime/cli.py`, then runs `python -m hyperloom.agents.critic.runtime.cli --help`
(5s timeout) before the Coordinator boots. Missing or broken runtime aborts
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
root and pass `--asset-root`. `mkdir -p "$ASSET_ROOT/scripts/configs"`,
`ln -sfn` `actions/` from `$REPO_ROOT/src/hyperloom/inference_optimizer/` and
`orchestrator/` from `$REPO_ROOT/src/hyperloom/`, plus the two
`scripts/ab_torch_compile_*.py` from `$REPO_ROOT/src/hyperloom/inference_optimizer/`, then
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
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
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

`--resume` auto-picks the latest `$USER_DATA_PATH/<model>/<UTC_ts>/`
(without `--resume-from`) or an explicit path via `--resume-from`.
`$USER_DATA_PATH` must stay at the **workspace root** so
`runtime/kernel-agent.env.sh` resolves. The CLI refuses to start if
`manifest.json` or `state.json` is missing in the picked session dir.

Reuse the Launch template above with these diffs: drop `--model`, add
`--resume`, set `RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`. Resume preserves
baseline, current best, params-search state, event history, and kernel-agent
artifacts; the CLI clears stale `stop_reason` and `crash_count` before retrying.

## Robustness Monitor for Long Runs

For runs > 5 min, start a monitor in its own `setsid nohup` process. It polls
`state.json` every 5 min, exits without resuming when the session is terminal
(any `stop_reason` in `STOP_REASON_VOCAB`, `phase=CLOSE`, or
`reports/final.md` present — including failure sentinels like
`baseline_failed`), and resumes via `--resume` only when the optimizer dies
without those markers (unexpected crash).

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

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION="${USER_DATA_PATH:-/workspace/hyperloom}"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "last_trace_analyze", "last_sweep"):
    print(f"{k}: {s.get(k)}")
print("explore_last_round:", s.get("explore_search", {}).get("last_round"))
print("phase:", s.get("phase"))
PY
```

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

The baseline/profile executors count aiter `.so` files (**< 20 = COLD**)
and pick a subprocess timeout accordingly: COLD → 3600s, WARM → 2400s
(`task.params['timeout_sec']` always wins). Each launch logs a
`baseline_executor: ...` marker and the cache state lands in the
`Preflight diagnostics:` block. If COLD_START repeats across retries the
JIT was killed mid-`hipcc` — bump
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400` (don't just relaunch).
Override the probe dir via `INFERENCE_OPTIMIZER_AITER_JIT_DIR`.

## Pre-GEAK Unittest Harness (unittest skill)

Before `backend=geak` attempts, the main agent generates a GEAK-compatible
test harness by following `src/hyperloom/agents/kernel/skills/unittest/SKILL.md`. The skill
searches for existing tests, collects shapes/dtypes from TraceLens and
profiling data, and generates a 4-mode harness (`--correctness` / `--profile`
/ `--benchmark` / `--full-benchmark`) that matches GEAK's evaluation contract.

The resulting `test_command` is passed via `--test-command` to
`kernel_optimization.py`, which forwards it to GEAK. If the skill fails to
produce a valid harness (after up to 3 retries), `--test-command` is omitted
and GEAK falls back to its own test discovery cascade.

Validation uses `src/hyperloom/agents/kernel/skills/unittest/validate_harness.py` for both
static checks (argparse + 4 flags + GEAK output markers) and runtime
verification (run correctness + benchmark with reduced iterations).

The Coordinator does NOT need to drive this step — the main agent executes
the unittest skill before calling `kernel_optimization.py`. Observability
shows up as `test_command` in `optimization_attempts.jsonl[].backend_paths`.

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
(`claude-opus-4-8` / `claude-opus-4-7` / `claude-opus-4-6`).

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | You likely set `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`; unset it or set it to `1`, then ensure the model appears in the gateway `/models` catalog. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` for offline / smoke runs. See
`## Critic Backend Selection`.

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/src/hyperloom/agents/critic`, or check the `src/hyperloom/agents/critic/` install. |
| `hyperloom.agents.critic.runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `src/hyperloom/agents/critic/AGENTS.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `hyperloom.agents.critic.runtime.cli ... timed out after 30s` | KB stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory`. Reproducing in `inmemory` is a bug — that path must not block on I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `trace_analyze` with unchanged trace/config: bug — reuse `last_trace_analyze`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must contain explicit correctness evidence.
- `stop_reason=no_more_leverage`: stop and report; only resume if the user changes workload / search space / model / strategy.
- `stop_reason=policy_loop`: Coordinator hit ≥10 consecutive `policy_denied` events for the same action/rule pair; all top actions may be locked or pruned. Inspect `SharedState.policy_denial_history` and the per-tick `Policy denials` block. To recover: manually edit `state.json` to remove the action from `pruned_families`, clear `policy_denial_streak` / `stop_reason`, and re-propose with fresh `params.grid` content (omit stale `idempotency_key`).
- `stop_reason=time_exhausted`: resume same session (`--resume`); do not start fresh.

## Report Back To User

Report concise status:

- session id (from `manifest.json`) and log path
- `cumulative_gain` and `current_best`
- explore accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
