---
myst:
    html_meta:
        "description": "Understand how Hyperloom dispatches kernel optimization requests. Covers the request dispatch flow, registered request kinds, KERNEL phase entry, backend selection, and artifact layout."
        "keywords": "Hyperloom, kernel optimization, GEAK, Forge, request dispatch, kernel execution, AMD GPU, ROCm, KERNEL phase, TraceLens, rewrite controller, multi-node"
---

# Kernel optimization execution path

Kernel work in Hyperloom is not handled by an LLM agent. Every kernel
`REQUEST` emitted by orchestration is intercepted inline by the Coordinator
and routed to a registered Python handler. No LLM turn is consumed.

## Request dispatch

Orchestration emits a `request{target_agent: "kernel_agent", kind: "<kind>"}` intent.
`IntentRouter._handle_request` (`orchestrator/loop/intent_router.py`) intercepts it
before any agent backend runs:

1. `_sequence_denial_for_request` checks the baseline prerequisite — if
   `baseline_tput == 0` and the kind is not `trace_analyze`, the request is
   policy-denied immediately (no bus record).
2. Records the request on the message bus (`source: "orchestration"`).
3. Checks `shared_state.kernel_enabled`; auto-rejects with `agent_disabled` when
   `False` (that is, `--no-kernel`).
4. Looks up the handler in `KERNEL_REQUEST_HANDLERS`; auto-rejects with
   `unknown_kernel_kind` (and a `valid_kinds` list) when none is found.
5. Runs the handler inline: `result = await handler(payload, session_dir=...)`.
6. Posts a `response{source: "programmatic_handler"}` directly to the bus.
7. Appends any failure to `last_action_failures`.

The requester reads the response from its inbox on its next turn.

No PolicyGate path runs for the RESPONSE because it's written directly through
`bus.append_and_seq`, not emitted by an LLM.

## Registered request kinds

| Request kind | Handler | Entry point |
|---|---|---|
| `trace_analyze` | `trace_analyze_handler` | TraceLens `tracelens_analysis.py` |
| `run_gemm_tuning` | `run_gemm_tuning_handler` | GEAK or kernelforge gemm-tune |
| `run_optimization` | `run_optimization_handler` | GEAK or Forge per-kernel |
| `integrate` | `integrate_handler` | patch → re-baseline → KEEP/REVERT |
| `apply_patch` | `integrate_handler` (alias) | same as `integrate` |

Any kind outside this table, including the action-name `kernel_opt`, yields an
immediate `unknown_kernel_kind` rejection.

Registration is not permission. `run_gemm_tuning` is a Coordinator-owned lane:
PolicyGate rejects an orchestration-issued REQUEST for it
(`COORDINATOR_OWNED_KERNEL_REQUEST_KINDS` in
`inference_optimizer/protocol/action_surfaces.py`, raised as
`rule="phase_incompatible"`) because it is dispatched once at phase entry from
a lane budget, and a per-tick re-issue would spend time the allocation never
granted. PolicyGate validates the REQUEST payload from orchestration
(path-sandbox, phase-action gate) but never sees the RESPONSE.

`run_fusion_handler` is absent from the table on purpose: `KernelPhase` awaits
it directly, so no request ever carries that kind.

## KERNEL phase entry: Coordinator-direct calls

When the Coordinator enters the KERNEL phase (`phases/kernel.py::_on_enter_kernel`, dispatched by `phases/machine.py::_on_phase_entered`),
it calls the handlers directly in Python — not through the REQUEST bus. Which
calls it makes depends on the backend: the entry hook branches before any lane
runs.

```python
# 1. GEAK branch — the documented default. One whole-pipeline e2e run, then
#    the phase winds down to SWEEP. Nothing below this line executes.
if geak_enabled:                      # geak_selected(): order is not exactly `forge`
    await self._run_geak_kernel_phase(from_phase=from_phase)
    return

# 2. Forge branch — only with KERNEL_OPT_BACKEND_ORDER=forge. Two routes into
#    one shared tail, chosen by whether GEMM tuning is due.
if not self._gemm_tuning_required_before_kernel_opt():
    await self._finish_kernel_entry()
    return
result = await run_gemm_tuning_handler({...}, session_dir=session_dir)
...                                    # handle result, post the bus response
await self._finish_kernel_entry()
```

Both routes end in `_finish_kernel_entry()`, and that is where the rest of the
phase's own work happens:

```python
async def _finish_kernel_entry(self) -> None:
    await self._maybe_reprofile_for_kernel()
    await self._maybe_run_forge_fusion_before_kernel_opt()
    handoff_dir = write_forge_handoff(...)
    await self._run_kernel_rewrite_controller(handoff_dir, attempt_dir)
```

**The rewrite controller is not downstream of GEMM tuning.** Tuning GEMM shape
tables and rewriting kernel source are unrelated jobs, so each stage in the
shared tail consults only its own switch and each skip is a return inside its
own helper rather than out of the entry hook.
`INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING=1` therefore leaves the rewrite controller
alone.

The controller runs last and unconditionally — no trace, candidate count or
source-resolution verdict gates it, because choosing operators is its job rather
than Hyperloom's. Hyperloom writes it a Markdown handoff (`workload.md`,
`serving-context.md`, `trace-evidence.md`), runs
`python -m kernelforge.cli kernel-rewrite-controller` as a bounded subprocess,
and integrates whatever patches it publishes through
`integrate_controller_patches`. Communication operators ride this path like any
other operator; a task declares its rank count and the controller passes
`--nproc-per-node` down to forge-loop.

Results are synthesized as `kernel_agent → orchestration` response messages with
`source="kernel_entry_auto"` so orchestration's inbox looks the same as if the
request had come through the bus.

The fusion lane (`_maybe_run_forge_fusion_before_kernel_opt` →
`run_fusion_handler`, integrated by `_integrate_fusion`) gates on
`_fusion_required_before_kernel_opt()`: `HYPERLOOM_SKIP_FUSION` not truthy, a
framework in `{sglang, vllm, vllm-aiter}`, a `last_profile_trace` to discover
from, and no `last_fusion` whose status is already `ok` / `complete` / `kept`
(idempotent re-entry). It is forge-only — under the default `geak` backend
`_on_enter_kernel` returns before the lane is reached.

A fusion result is written to the `last_fusion` SharedState field and posted as
a `run_fusion_done` response with `source="kernel_entry_auto"`. A result that is
`kept` and `requires_e2e_validation` is handed to `integrate_handler`, which
applies the fused-kernel source patch, sets the fusion env flags on the
re-baseline server, and KEEPs only when measured e2e throughput clears the
threshold.

## Where the former Iron Rules are enforced

The seven rules from the retired `kernel_agent.md` live in executable Python:

| Former rule | Real enforcer |
|---|---|
| IR-1 submit all candidates in parallel | `_batch_kernel_candidates` + `_DEFAULT_KERNEL_BATCH_PARALLEL=8` in `request_handlers.py` |
| IR-2 never modify source before GEAK submission | `_is_runtime_generated_kernel` gate in `request_handlers.py` |
| IR-3 integration is mandatory after every KEEP | `phases/kernel_stack.py::KernelStackPhase._auto_enqueue_pending_integrations` (called by `intent_router.py`) |
| IR-4 kill stale servers before restart | `_multi_node_server_lifecycle.py::restart_server_for_round` |
| IR-5 safe process management | `orchestrator/actions/executors/_subprocess_kill.py` |
| IR-6 use apply_kernel_patch.py --target-file | `request_handlers.py::_maybe_apply_kernel_patch` → `agents/kernel/tools/apply_kernel_patch.py::apply_kernel_patch` |
| IR-7 never modify GEAK config | GEAK invocation wrappers in `request_handlers.py` / `geak_runner.py` |

## Backend selection

GEAK owns the KERNEL phase by default and decides kernel strategy internally.
The per-kernel Forge backend is an opt-in:

- **Default**: `geak`. It is the code default whenever
  `KERNEL_OPT_BACKEND_ORDER` is unset (`_DEFAULT_KERNEL_PHASE_BACKEND_ORDER` in
  `orchestrator/kernel/request_handlers.py`), so no launcher has to set it. The
  bare-metal installer additionally exports `${KERNEL_OPT_BACKEND_ORDER:-geak}`
  and persists it into `.env`, and the Slurm launchers export the same
  `:-geak` fallback into the job / container environment. `.env.template`
  ships the line commented out.
- **Forge (per-kernel)**: set `KERNEL_OPT_BACKEND_ORDER=forge` exactly. Any
  other value (including `--backends` CLI flags, payload `backends` hints, or
  `GEMM_TUNING_BACKEND`) doesn't enable Forge.

`run_gemm_tuning_handler` also defaults to GEAK unless
`KERNEL_OPT_BACKEND_ORDER=forge` is set. That default applies to an
LLM-issued `run_gemm_tuning` REQUEST, which is dispatched inline whatever the
backend. The KERNEL-**entry** GEMM tuning is a different matter: under the
default `geak` backend it never fires at all, because `_on_enter_kernel` hands
the phase to `_run_geak_kernel_phase` and returns before reaching it.

FlyDSL kernels (`source_type=flydsl`) are handled by Forge when it is enabled.

### Communication operators

Collectives are ordinary rewrite targets. TraceLens resolves a mangled NCCL/RCCL
kernel name back to the framework device source that issued it and publishes the
row in `kernel_candidates.json` with `candidate_source: nccl_summary`; vendor
RCCL/NCCL symbols never qualify, because they are opaque binaries with no
rewritable source. The controller's opportunity analysis reads those rows like
any other candidate and decides whether to publish a task for them.

What separates such a task is only its measurement harness: it declares
`world_size`, the controller forwards `--nproc-per-node` to forge-loop, and the
task preparer authors a driver that launches its own ranks, checks correctness
against the matching `torch.distributed` collective, and reduces latency and SNR
across ranks before reporting.

## Toolkit installation

Shell paths in this section follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.

The kernel tool scripts live under `hyperloom/agents/kernel/tools/` and are
resolved at runtime through the `HYPERLOOM_KERNEL_AGENT_ROOT` env var (set to
`<repo>/hyperloom/agents/kernel` by the CLI bootstrap). Install everything using:

```bash
export REPO_ROOT="$(pwd -P)"    # workspace holding the hyperloom package
# Pin the artifact root so the env file below has a known path. Left unset, the
# CLI picks /workspace/hyperloom when writable and session/ under $PWD otherwise.
export USER_DATA_PATH="${USER_DATA_PATH:-$REPO_ROOT/session}"
bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

`install.sh` is idempotent. It sets up TraceLens, GEAK, Ray, and writes the
env file. Re-run it after a venv rebuild or before each session.

Required env vars:

| Variable | Set by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | operator | Anthropic-side key; GEAK and TraceLens both run Claude Code |
| `ANTHROPIC_BASE_URL` | operator | Anthropic-side endpoint (point it at your gateway) |
| `TRACELENS_ROOT` | `install.sh` (operator can override) | TraceLens checkout; installer clones to `.cache/TraceLens` by default |
| `KERNEL_OPT_BACKEND_ORDER` | code default `geak` when unset; bare-metal installer and Slurm launchers export `${KERNEL_OPT_BACKEND_ORDER:-geak}` | Set to exactly `forge` to enable per-kernel Forge |

Forge needs **no path variable**. It ships inside the Hyperloom wheel, so the
`FORGE_PATH` that used to be required here is removed and nothing reads it. The
optional dev override is `KERNELFORGE_PROJECT_ROOT` (a writable root whose
resource subtrees take precedence over the packaged copies); see
[environment variables](environment-variables.md).

Optional:

| Variable | Purpose |
|---|---|
| `TRACELENS_INTERNAL_ROOT` | TraceLens internal extension; unset = open-source-only |
| `KERNEL_OPT_MAX_PARALLEL` | Override the 8-concurrent-kernel default |
| `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL` | Override partial-attempt retry cap (default 2) |
| `KERNEL_OPT_BACKEND_BUDGET_MIN` | Force the per-optimization wall-clock budget in minutes (default 90); wins over the LLM-authored payload value |

Fusion lane:

| Variable | Purpose |
|---|---|
| `HYPERLOOM_SKIP_FUSION` | Truthy disables the fusion lane before any other gate is evaluated |
| `FORGE_FUSION_TIMEOUT` | Wrapper timeout in seconds (default 7200 = 2h); a payload `timeout` / `timeout_sec` wins over it |
| `FORGE_FUSION_MAX_TURNS` | Agent turn cap for one fusion run (default 100); a payload `max_turns` wins over it |

Collective candidate extraction:

| Variable | Purpose |
|---|---|
| `HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES` | Truthy lets a source-resolved collective borrow shapes from the trace's sole all-reduce workload family |

## Artifact layout

All kernel tool output lands under
`$USER_DATA_PATH/kernel-agent/runs/<session_id>/`:

```
runs/<session_id>/
  session_state.json
  kernel_candidates.json
  tracelens/
    analysis.md                 # TraceLens canonical report (not copied by Hyperloom)
    tracelens_report.json
    system_findings/
    category_findings/
  optimization_attempts.jsonl
  prompts/<attempt_id>.md
  optimized/<attempt_id>_stdout.log
  verification/<kernel_id>.json
  results/<kernel_id>.json
  logs/<tool>/<run_id>.log
  status/<tool>/<run_id>.json
```

Cross-task GEAK artifacts keyed by `kernel_id` live at
`$USER_DATA_PATH/kernel-agent-workspace/<kernel_id>/`.

### Per-attempt stdout file naming

`run_attempt` in `kernel_optimization.py` writes one file per attempt under
`runs/<session_id>/optimized/`:

| Mode | Filename | Contents |
|---|---|---|
| Real backend run | `<attempt_id>_stdout.log` | Raw subprocess stdout (GEAK conversation log) |
| `--dry-run` | `<attempt_id>_optimized<source_suffix>` (e.g. `.cu`) | Synthetic placeholder for smoke tests |

**Backward compatibility**: Prior to 2026-05 the real-backend file shared the
`<attempt_id>_optimized<suffix>` name and contained subprocess stdout. That caused
`_source_text_looks_complete` to false-positive match generic English in transcript
lines and promote the log to `artifact_source = source_file`. The breakdown
collector uses `glob("<attempt_id>*")` so it discovers both naming schemes
transparently.

## Multi-node mode

When `--nodes >= 2`, the optimization sandbox has no GPU. Handlers adapt:

- **Applying patches**: `apply_kernel_patch.py` detects multi-node and fans the
  patch to every pod using `python3 -m hyperloom.inference_optimizer.multi_node apply-patch`.
  Revert uses `manifest.multinode.host_backup_map` to hit the same pods.
- **Compiling/benchmarking**: Forge/GEAK backends use
  `python3 -m hyperloom.inference_optimizer.multi_node kernel-bench` instead of
  local `hipcc`.
- **Integration**: `integrate_handler` forces a full server restart after a
  successful apply so the re-baseline measures the patched modules.
- **RayJob recreate**: `_replay_kernel_patches_for_multi_node` replays all
  applied kernel patches when a new RayJob pod starts.
