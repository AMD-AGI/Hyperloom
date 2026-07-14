# Hyperloom Env Governance Table

This table records the current env-var governance decision after the cleanup
branch. It separates "delete the env interface" from "move out of the main
runtime user-config table".

## Summary

| Category | Count | Decision |
|---|---:|---|
| Deleted env interfaces | 38 | Removed from runtime/tests/docs where they were only legacy knobs or ignored fallbacks. |
| External/operator envs | 30 | Keep documented as user/operator configuration. |
| Internal handoff envs | 36 | Keep in code, but do not present as primary user configuration. |
| Installer/CI/build envs | 25 | Keep in installer, bare-metal, Slurm, or CI documentation only. |
| Future refactor candidates | 22 | Do not delete directly; replace with structured config/state first. |

## Deleted Env Interfaces

These names should not reappear as runtime env fallbacks.

| Env / Pattern | Previous role | Replacement / current behavior |
|---|---|---|
| `INFERENCE_OPTIMIZER_CYCLIC_PHASES` | Phase cyclic switch | Cyclic behavior is code policy. |
| `INFERENCE_OPTIMIZER_PHASE_INTERLEAVE` | Phase interleave switch | Interleave remains disabled by policy. |
| `INFERENCE_OPTIMIZER_ENABLE_ROOFLINE` | Roofline/profile depth env | Use `--enable-roofline` / state. |
| `INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP` | Conc sweep default env | Use `--enable-conc-sweep` / `--no-enable-conc-sweep`. |
| `INFERENCE_OPTIMIZER_RESEARCH_SCOUT` | Research scout env | Use `--research-scout` / `--no-research-scout`. |
| `INFERENCE_OPTIMIZER_STATIC_RECON` | Static recon env | Use `--static-recon` / `--no-static-recon`. |
| `INFERENCE_OPTIMIZER_RECIPE_SEDIMENT` | Recipe sediment env | Use `--recipe-sediment` / `--no-recipe-sediment`. |
| `INFERENCE_OPTIMIZER_TARGET_ADVISORY` | Target advisory env | Use `--target-advisory` / `--no-target-advisory`. |
| `INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES` | Legacy state migration mode | Fixed code migration behavior. |
| `INFERENCE_OPTIMIZER_MIGRATION_MODE` | SharedState migration strictness | Fixed code migration behavior. |
| `INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS` | Breakdown transcript body env | Process-local setting / explicit tooling only. |
| `INFERENCE_OPTIMIZER_NO_FRAMEWORK` | Framework skip env | Use `--no-framework-agent`. |
| `INFERENCE_OPTIMIZER_FRAMEWORK_CONFIG_EXPLORATION` | Framework config lane env | Code/state controlled. |
| `INFERENCE_OPTIMIZER_SWEEP_SKIP_WHEN_NO_GAIN` | Historical sweep skip env | Removed; no current runtime interface. |
| `INFERENCE_OPTIMIZER_SATURATION_CONVERGENCE` | Saturation convergence env | Convergence behavior is code policy. |
| `INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK` | Framework plateau threshold env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_ENABLEMENT_MAX_STALL` | Enablement stall cap env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_DISPATCHER_POLL_SECONDS` | Dispatcher polling env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_CHECKPOINT_MIN_TICK_GAP` | Checkpoint tick gap env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_RESUME_REVERIFY_BEST` | Resume best reverify env | Removed env-only escape hatch. |
| `INFERENCE_OPTIMIZER_RESUME_DRIFT_FLOOR` | Resume drift floor env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_MIN_ENGAGED_GAIN_PCT` | Engaged gain threshold env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_MEASUREMENT_DIVERGENCE_WARN_PCT` | Measurement divergence warn env | Fixed code constant. |
| `HYPERLOOM_ROOFLINE_WATERMARK_RATIO` | Roofline watermark ratio env | Fixed code constant. |
| `WARM_REPLAY_ADVISORY_CONFIDENCE` | Warm replay advisory threshold env | Fixed code constant. |
| `INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS` | Old conc-sweep handoff env | Use `--conc-sweep-concs`. |
| `INFERENCE_OPTIMIZER_CONC_SWEEP_TIMEOUT_SEC` | Old conc-sweep timeout env | Use CLI/state conc-sweep timeout config. |
| `INFERENCE_OPTIMIZER_CONC_SWEEP_TOTAL_BUDGET_SEC` | Old conc-sweep budget env | Use CLI/state conc-sweep budget config. |
| `PR_FEED_WINDOW_DAYS` | PR feed look-back env | Use `--pr-feed-window-days`. |
| `INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY` | Research lane capacity env | Use `--research-lane-capacity`; default is GPU-derived. |
| `INFERENCE_OPTIMIZER_SPECIALIST_MODEL` | Specialist model env | Use `--specialist-model`. |
| `INFERENCE_OPTIMIZER_SPECIALIST_MAX_TURNS` | Specialist max turns env | Use `--specialist-max-turns`. |
| `INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS` | Specialist turn timeout env | Use `--specialist-per-turn-max-seconds`. |
| `INFERENCE_OPTIMIZER_SPECIALIST_DISPATCH_MODE` | Specialist dispatch env | Use `--specialist-dispatch-mode`. |
| `INFERENCE_OPTIMIZER_SPECIALIST_MCP_CONFIG` | Specialist MCP config env | Use `--specialist-mcp-config`. |
| `INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO` | Explore overtime ratio env | Use `--explore-overtime-kill-ratio`. |
| `INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC` | Explore variant timeout env | Use `--explore-variant-timeout-sec`. |
| `INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN` | Explore timeout safety margin env | Use `--explore-variant-timeout-safety-margin`. |

## External / Operator Env

These are acceptable to document as env configuration because an operator may
reasonably set them before launching Hyperloom.

| Env / Pattern | Scope | Notes |
|---|---|---|
| `SAFE_API_KEY` | LLM credential | Primary single-gateway secret; fans out to provider aliases. |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_CUSTOM_HEADERS` | OpenAI-compatible gateway | External gateway endpoint/secret/header configuration. |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS` | Claude / Anthropic gateway | Split-gateway and Claude CLI authentication. |
| `GEAK_API_KEY`, `GEAK_BASE_URL`, `GEAK_CLAUDE_MODEL` | GEAK advanced override | Keep as explicit GEAK backend overrides; do not use as general optimizer knobs. |
| `KB_BASE_URL`, `KB_SERVICE_TOKEN` | Critic live KB | Required only for live KB mode. |
| `GBRAIN_BASE_URL`, `GBRAIN_TOKEN` | Recipe KB read side | Service endpoint and secret. |
| `PRIMUS_CORTEX_PR_API` | Internal PR discovery | Canonical internal PR API endpoint. |
| `SAFE_API_URL` | SaFE / Claw API | Used by multi-node / Claw tooling. |
| `USER_DATA_PATH` | Session data root | Canonical user-facing session/log root. |
| `TRACELENS_ROOT`, `TRACELENS_INTERNAL_ROOT` | TraceLens source | Operator-maintained checkout override. |
| `MAGPIE_PATH`, `INFERENCEX_PATH` | Benchmark / analysis source | Operator-maintained checkout override. |
| `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` | GPU visibility | Platform/cluster injection. |
| `NCCL_IB_HCA`, `RAY_ADDRESS`, `LD_LIBRARY_PATH` | Platform runtime | External runtime/library environment. |
| `HYPERLOOM_LANGFUSE_ENABLE`, `LANGFUSE_*` | Observability | Live trace push / offline backfill credentials. |
| `TAVILY_API_KEY`, `SERPER_API_KEY`, `BRAVE_API_KEY` | Optional web search providers | Provider secrets, not core optimizer behavior. |
| `HYPERLOOM_RECOVER_ALLOW_GPU_RESET` | Recovery escape hatch | High-risk operator opt-in only. |

## Internal Handoff Env

These are still used by subprocesses, generated scripts, resume paths, or
agent runtimes. They should not be described as primary user configuration.

| Env / Pattern | Internal owner | Why it stays |
|---|---|---|
| `MODEL_PATH`, `MODEL_CLASS`, `FRAMEWORK`, `FRAMEWORK_VERSION` | CLI -> SharedState -> subprocesses | Workload identity handoff and resume fidelity. |
| `GPU_TYPE`, `TARGET_GPU_TYPE`, `PRECISION` | CLI / hardware detection / Magpie | Benchmark and prompt workload fidelity. |
| `TP`, `EP`, `CONC`, `ISL`, `OSL`, `PROFILE_OSL`, `MAX_MODEL_LEN` | Workload materialization | Magpie YAML and child process compatibility. |
| `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` | CLI session boot | Explicit current-session handoff. |
| `SESSION_DIR`, `HYPERLOOM_SESSION_DIR` | Agent compatibility | Legacy/session hints; prefer current-session dir. |
| `KERNEL_AGENT_ENV` | Kernel agent runtime | Env file generated by install scripts. |
| `HYPERLOOM_KERNEL_AGENT_ROOT`, `FRAMEWORK_AGENT_ROOT` | Agent runtime assets | Skill/tool root fallback. |
| `MULTI_NODE_STATE_FILE` | Multi-node runtime | Session-scoped state file handoff. |
| `NODES`, `INFERENCE_OPTIMIZER_NODES`, `INFERENCE_OPTIMIZER_GPUS_PER_NODE` | Multi-node runtime | Restart / executor / state reconstruction. |
| `INFERENCE_OPTIMIZER_MN_BACKEND`, `INFERENCE_OPTIMIZER_RAYJOB_IMAGE` | Multi-node CLI/runtime | RayJob/Dynamo backend and image fallback. |
| `PD_PREFILL_*`, `PD_DECODE_*`, `PD_TRANSFER_BACKEND`, `PD_IB_DEVICE` | PD disaggregated serving | Role-specific restart-server defaults. |
| `INFERENCE_OPTIMIZER_SERVER_ARGS` | CLI -> YAML materializer | Server flag handoff to framework-specific args. |
| `SKIP_VARIANTS` | Grid runner | Env fallback until skip rules are fully state/params only. |
| `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY`, `INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES` | GPU pool / policy | GPU specialist capacity and device pool fallback. |
| `HYPERLOOM_SPECIALIST_KB_MCP_URL`, `HYPERLOOM_LOCAL_KB_ROOT` | Specialist / local KB | Read-only KB tool endpoint and local recipe store root. |
| `ROBUSTNESS_SERVER_URL`, `ROBUSTNESS_LLM_RCA_DISABLED` | Robustness agent | Server discovery and LLM RCA kill switch. |
| `CRITIC_WEB_*`, `WEB_SEARCH_*`, `WEB_FETCH_*` | Critic runtime | Move to Critic web-tools docs; keep code. |
| `HYPERLOOM_REPORT_*` | Report tooling | Move to report tooling docs; keep code. |
| `CRITIC_KB_*`, `KB_TIMEOUT_MS`, `KB_RETRY_MAX`, `KB_DEAD_LETTER_DIR`, `CORTEX_KB_URL` | Critic KB | Move to Critic KB docs; keep code. |
| `HYPERLOOM_RUNTIME_DIR` | Installer / kernel agent runtime | Runtime tree for generated env files and GEAK config. |
| `HYPERLOOM_FRAMEWORK_SOURCE_ROOTS` | Kernel TraceLens resolver | Source-file resolver override; not a primary optimizer knob. |

## Installer / CI / Build Env

Keep these out of the main runtime user-config table. They belong in
installer, bare-metal, Slurm, or CI documentation.

| Env / Pattern | Owner |
|---|---|
| `MAGPIE_REPO`, `MAGPIE_REF` | Installer dependency pinning. |
| `INFERENCEX_REPO`, `INFERENCEX_REF` | Installer dependency pinning. |
| `GEAK_*` not listed above | GEAK installer/runtime backend settings. |
| `FORGE_*`, `KERNEL_FORGE_*` | Forge / KernelForge backend setup. |
| `SGLANG_*`, `AITER_*`, `VLLM_*` | Bare-metal framework install/build settings. |
| `ROCM_PATH`, `HIP_PATH`, `PYTORCH_ROCM_ARCH` | ROCm/HIP/PyTorch build environment. |
| `GITHUB_TOKEN`, `GH_TOKEN`, `HF_TOKEN` | CI/private checkout/model download credentials. |
| `SAFE_OPTIMIZE_*` | CI optimization-submit workflow controls. |

## Future Refactor Candidates

These can be removed only after replacing env handoff with structured
configuration or persisted state.

| Env / Pattern | Required replacement before deletion |
|---|---|
| Workload group: `MODEL_PATH`, `FRAMEWORK`, `TP`, `CONC`, `ISL`, `OSL`, etc. | `RunConfig` / `WorkloadConfig` passed through CLI, state, executors, and resume. |
| Multi-node / PD group: `INFERENCE_OPTIMIZER_NODES`, `PD_*`, `INFERENCE_OPTIMIZER_RAYJOB_IMAGE` | Persisted multi-node state and explicit restart-server parameters. |
| `INFERENCE_OPTIMIZER_SERVER_ARGS` | Structured server-args field in task params / state. |
| `SKIP_VARIANTS` | Params/state-only variant skip propagation. |
| GPU specialist envs | CLI/state-only GPU pool configuration. |
| Robustness envs | CLI/config-file replacement for server URL and RCA toggle. |
| Critic KB / web / report envs | Agent-specific config surfaces and docs. |
