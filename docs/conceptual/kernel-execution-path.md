# Kernel optimization execution path

Kernel work in Hyperloom is **not handled by an LLM agent**. Every kernel
`REQUEST` emitted by Orchestration is intercepted inline by the Coordinator
and routed to a registered Python handler. No LLM turn is consumed.

## Request dispatch

Orchestration emits a `request{target_agent: "kernel_agent", kind: "<kind>"}` intent.
`IntentRouter._handle_request` (`orchestrator/loop/intent_router.py`) intercepts it
before any agent backend runs:

1. `_sequence_denial_for_request` checks the baseline prerequisite — if
   `baseline_tput == 0` and the kind is not `trace_analyze`, the request is
   policy-denied immediately (no bus record, no cursor advance).
2. Records the request on the message bus (`source: "orchestration"`).
3. Checks `shared_state.kernel_enabled`; auto-rejects with `agent_disabled` when
   `False` (i.e. `--no-kernel`).
4. Looks up the handler in `KERNEL_REQUEST_HANDLERS`; auto-rejects with
   `unknown_kernel_kind` (and a `valid_kinds` list) when none is found.
5. Runs the handler inline: `result = await handler(payload, session_dir=...)`.
6. Posts a `response{source: "programmatic_handler"}` directly to the bus.
7. Advances the kernel cursor past the request sequence number.

No PolicyGate path runs for the RESPONSE because it is written directly via
`bus.append_and_seq`, not emitted by an LLM.

## Registered request kinds

| Request kind | Handler | Entry point |
|---|---|---|
| `trace_analyze` | `trace_analyze_handler` | TraceLens `tracelens_analysis.py` |
| `run_gemm_tuning` | `run_gemm_tuning_handler` | GEAK or forge-gemm-tune |
| `run_fusion` | `run_fusion_handler` | forge-fusion |
| `run_optimization` | `run_optimization_handler` | GEAK or Forge per-kernel |
| `integrate` | `integrate_handler` | patch → re-baseline → KEEP/REVERT |
| `apply_patch` | `integrate_handler` (alias) | same as `integrate` |

Any other kind, including the action-name `kernel_opt`, yields an immediate
`unknown_kernel_kind` rejection. PolicyGate validates the REQUEST payload from
orchestration (path-sandbox, phase-action gate) but never sees the RESPONSE.

## KERNEL phase entry: Coordinator-direct calls

When the Coordinator enters the KERNEL phase (`phases/kernel.py::_on_enter_kernel`, dispatched by `phases/machine.py::_on_phase_entered`),
it calls the handlers directly in Python — not through the REQUEST bus:

```python
result = await run_gemm_tuning_handler({...}, session_dir=session_dir)
# then, if candidates remain:
result = await run_optimization_handler({...}, session_dir=session_dir)
```

Results are synthesized as `kernel_agent → orchestration` response messages with
`source="kernel_entry_auto"` so orchestration's inbox looks the same as if the
request had come through the bus.

## Where the former Iron Rules are enforced

The seven rules from the retired `kernel_agent.md` live in executable Python:

| Former rule | Real enforcer |
|---|---|
| IR-1 submit all candidates in parallel | `_batch_kernel_candidates` + `_DEFAULT_KERNEL_BATCH_PARALLEL=8` in `request_handlers.py` |
| IR-2 never modify source before GEAK submission | `_is_runtime_generated_kernel` gate in `request_handlers.py` |
| IR-3 integration is mandatory after every KEEP | `phases/kernel_stack.py::KernelStackPhase._auto_enqueue_pending_integrations` (called by `intent_router.py`) |
| IR-4 kill_server + check_gpu_memory before server restart | `apply_and_bench.py` subprocess tool; `_multi_node_server_lifecycle.py::restart_server_for_round` for multi-node |
| IR-5 safe process management | `orchestrator/actions/executors/_subprocess_kill.py` |
| IR-6 use apply_kernel_patch.py --target-file | `request_handlers.py::_maybe_apply_kernel_patch` → `agents/kernel/tools/apply_kernel_patch.py::apply_kernel_patch` |
| IR-7 never modify GEAK config | GEAK invocation wrappers in `request_handlers.py` / `geak_runner.py` |

## Backend selection

GEAK owns the KERNEL phase by default and decides kernel strategy internally.
The per-kernel Forge backend is an opt-in:

- **Default**: `KERNEL_OPT_BACKEND_ORDER=geak` (every launcher exports this default).
- **Forge (per-kernel)**: set `KERNEL_OPT_BACKEND_ORDER=forge` exactly. Any
  other value (including `--backends` CLI flags, payload `backends` hints, or
  `GEMM_TUNING_BACKEND`) does not enable Forge.

GEAK GEMM tuning uses `run_gemm_tuning_handler`, which also defaults to GEAK
unless `KERNEL_OPT_BACKEND_ORDER=forge` is set.

FlyDSL kernels (`source_type=flydsl`) and multi-GPU collective kernels
(`is_multigpu: True`) are handled by Forge when it is enabled.

## Toolkit installation

The kernel tool scripts live under
`src/hyperloom/agents/kernel/tools/` and are resolved at runtime via the
`HYPERLOOM_KERNEL_AGENT_ROOT` env var (set to `<repo>/src/hyperloom/agents/kernel`
by the CLI bootstrap). Install everything via:

```bash
export REPO_ROOT="$(pwd)"    # hyperloom repo root
bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
source "${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh"
```

`install.sh` is idempotent. It sets up TraceLens, GEAK, Ray, and writes the
env file. Re-run it after a venv rebuild or before each session.

Required env vars:

| Variable | Set by | Purpose |
|---|---|---|
| `SAFE_API_KEY` | operator | LLM gateway key (GEAK and TraceLens inherit this) |
| `OPENAI_BASE_URL` | operator | LLM gateway endpoint |
| `TRACELENS_ROOT` | `install.sh` (operator may override) | TraceLens checkout; installer clones to `.cache/TraceLens` by default |
| `KERNEL_OPT_BACKEND_ORDER` | launcher (default `geak`) | Set to `forge` to enable per-kernel Forge |

Optional:

| Variable | Purpose |
|---|---|
| `TRACELENS_INTERNAL_ROOT` | TraceLens internal extension; unset = open-source-only |
| `KERNEL_OPT_MAX_PARALLEL` | Override the 8-concurrent-kernel default |
| `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL` | Override partial-attempt retry cap (default 2) |

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

**Backward compatibility**: prior to 2026-05 the real-backend file shared the
`<attempt_id>_optimized<suffix>` name and contained subprocess stdout. That caused
`_source_text_looks_complete` to false-positive match generic English in transcript
lines and promote the log to `artifact_source = source_file`. The breakdown
collector uses `glob("<attempt_id>*")` so it discovers both naming schemes
transparently.

## Multi-node mode

When `--nodes >= 2`, the optimization sandbox has no GPU. Handlers adapt:

- **Applying patches**: `apply_kernel_patch.py` detects multi-node and fans the
  patch to every pod via `python3 -m hyperloom.inference_optimizer.multi_node apply-patch`.
  Revert uses `manifest.multinode.host_backup_map` to hit the same pods.
- **Compiling/benchmarking**: Forge/GEAK backends use
  `python3 -m hyperloom.inference_optimizer.multi_node kernel-bench` instead of
  local `hipcc`.
- **Integration**: `integrate_handler` forces a full server restart after a
  successful apply so the re-baseline measures the patched modules.
- **RayJob recreate**: `_replay_kernel_patches_for_multi_node` replays all
  applied kernel patches when a new RayJob pod starts.
