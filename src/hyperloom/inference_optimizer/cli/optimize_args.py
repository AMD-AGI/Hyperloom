# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Argument groups of the ``optimize`` subcommand that need nothing from the parser module itself."""

from __future__ import annotations

import argparse

from hyperloom.common import predictor_config
from hyperloom.common.workload_defaults import DEFAULT_PRECISION
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS

from .. import framework_registry
from .backends import CRITIC_PROTOCOL_CHOICES
from .benchmark_config import add_benchmark_argument


def add_gpu_partition_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the target-GPU and compute-partition flags of ``optimize``."""
    opt.add_argument(
        "--gpu-type",
        type=str.lower,
        # Sorted for a stable --help listing, and derived so a board added to the board table is accepted here
        # without a second edit.
        choices=sorted(AMD_GPU_BOARDS),
        default=None,
        help="Hint for the real target GPU. The rocm-smi probe always "
        "wins when both are present and disagree; a WARN is "
        "emitted to stderr so the operator sees the typo. Used "
        "verbatim only when the probe fails (CPU sandbox / no "
        "rocm-smi). Magpie runner_type is derived separately; "
        "mi308x and mi325x currently run with mi300x runner scripts because "
        "Magpie does not yet ship MI308X/MI325X-specific SGLang/vLLM scripts.",
    )
    opt.add_argument(
        "--compute-partition-mode",
        type=str,
        default=None,
        metavar="MODE",
        help="Declare the compute-partition mode the GPU is already in: SPX "
        "(whole card), DPX (2), QPX (4) or CPX (8). This is an assertion, not "
        "a request -- nothing in the optimizer changes the mode, because doing "
        "so is privileged, evicts every process on the card, and renumbers its "
        "devices. Set the mode with amd-smi before launching this command: the "
        "shape is checked and recorded at launch, so a mode applied later is "
        "too late to be either. Passing this flag makes the session "
        "refuse to start if the card is in a different mode, which is what "
        "catches an external set that did not take effect before three hours "
        "are spent recording the numbers under the wrong topology. The mode is "
        "recorded in the platform fingerprint whether or not this flag is "
        "passed. See also --streams-per-partition.",
    )
    opt.add_argument(
        "--streams-per-partition",
        type=int,
        # None, not 2, so a resume can tell "not passed" from "passed 2" and let the persisted value stand.
        default=None,
        metavar="N",
        help="Concurrent streams to place on each partition when the card is "
        "partitioned. Defaults to 2, which is where every mode measured on "
        "MI355X peaked: one stream leaves each partition idle through the "
        "fixed per-pass cost, a second fills it, a third only adds queueing. "
        "Raise it only with evidence, and note that every stream on a "
        "partition holds its own copy of the weights -- so this multiplies "
        "the memory the workload has to fit, and the session refuses to start "
        "when it provably will not. A value below 1 is refused rather than "
        "quietly replaced by the default. Only a scriptable framework's "
        "benchmark places work per partition; passing this with a serving "
        "framework warns, because nothing would act on it.",
    )


def add_framework_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the framework selection and custom-workload flags of ``optimize``."""
    opt.add_argument(
        "--framework",
        choices=list(framework_registry.names()),
        default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
        "--framework > sglang (default). Selection is "
        "session-wide; mixing frameworks in a single session is not "
        "supported: --resume-from uses the framework the session was "
        "created with, and refuses a --framework that differs. NOTE: --framework atom is single-node-only "
        "(``--nodes>=2`` fails fast); profile / roofline, "
        "kernel-agent, and framework-agent are all enabled on atom. "
        "The auto-tighten guard only enforces ``--nodes 1``. "
        "--framework xdit is a server-less (scriptable) diffusion "
        "workload (xDiT): no serving server, throughput is img/s, and "
        "the accuracy gate is an image-quality gate (LPIPS/SSIM/MSE). "
        "--framework custom is your own workload: pass --framework-path and "
        "--benchmark-scripts-dir instead of shipping a framework definition.",
    )
    opt.add_argument(
        "--framework-path",
        default=None,
        metavar="DIR",
        help="Checkout of the framework to optimize. Sets FRAMEWORK_REPO_PATH, "
        "which is what puts the tree on PolicyGate's patch allowlist, so a "
        "specialist may only edit source you pointed at. Required for "
        "--framework custom; for the built-in frameworks it overrides the "
        "checkout they would otherwise clone.",
    )
    add_benchmark_argument(opt)
    opt.add_argument(
        "--benchmark-scripts-dir",
        default=None,
        metavar="DIR",
        help="Directory holding your benchmark entrypoint. Sets "
        "HYPERLOOM_BYPASS_SCRIPTS_DIR. The entrypoint is taken as "
        "custom_<gpu-type>.sh, or the single .sh in the directory. It must "
        "emit a quality_gate block in its report: for a server-less workload "
        "that gate is the only correctness signal, and a missing one scores "
        "zero, so every candidate is rejected.",
    )
    opt.add_argument(
        "--xdit-model-name",
        default=None,
        metavar="NAME",
        help="xDiT HuggingFace repo ID (e.g., 'black-forest-labs/FLUX.2-dev'). Required for xDiT runs.",
    )


def add_cluster_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the node-count and multi-node backend flags of ``optimize``."""
    opt.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Total number of GPU nodes for the inference cluster. "
        "1 (default) keeps the legacy single-pod path. "
        ">=2: `optimize` adopts the cluster the platform provisioned and handed "
        "over via HYPERLOOM_MN_EXT_* (see multi_node/SKILL.md), runs bootstrap "
        "once, and exports RAY_ADDRESS for kernel-agent. It never creates or "
        "releases the cluster; without a hand-off it exits 2. "
        "Default: 1.",
    )
    opt.add_argument(
        "--mn-backend",
        choices=("rayjob", "infera"),
        default=None,
        help="Multi-node backend when --nodes>=2: 'rayjob' (default, Ray "
        "head+workers) or 'infera' (idle InferaDeployment + SSH control "
        "plane). Defaults to rayjob when omitted. Single-node runs ignore "
        "this flag.",
    )
    opt.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="GPUs per multi-node pod (Infera worker/prefill/decode or RayJob "
        "head+workers). Defaults to 8 when omitted.",
    )
    # Platform-owned, inert here.
    opt.add_argument("--mn-image", default=None, help=argparse.SUPPRESS)
    opt.add_argument("--cpus-per-node", type=int, default=None, help=argparse.SUPPRESS)
    opt.add_argument("--mem-per-node", type=int, default=None, help=argparse.SUPPRESS)
    opt.add_argument("--extra-env", action="append", default=[], help=argparse.SUPPRESS)


def add_pd_topology_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the prefill-decode mode and group-shape flags of ``optimize``."""
    opt.add_argument(
        "--pd-mode",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="Prefill-Decode disaggregation mode. ALWAYS defaults to "
        "`aggregated` regardless of any inherited $PD_MODE env, so "
        "PD only turns on when the agent explicitly passes "
        "`--pd-mode disaggregated` (driven by the prompt's "
        "Environment block having a PD_MODE=disaggregated line). "
        "Stale env from a previous restart cannot accidentally "
        "re-enable PD.",
    )
    opt.add_argument(
        "--pd-prefill-nodes",
        type=int,
        default=0,
        help="Number of prefill nodes (disaggregated only); pn+dn=nodes",
    )
    opt.add_argument(
        "--pd-decode-nodes",
        type=int,
        default=0,
        help="Number of decode nodes (disaggregated only)",
    )
    opt.add_argument(
        "--pd-prefill-tp",
        type=int,
        default=0,
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-decode-tp",
        type=int,
        default=0,
        help="TP for decode group (disaggregated only); default = --tp",
    )


def add_pd_role_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the prefill-decode transport and per-role server flags of ``optimize``."""
    opt.add_argument(
        "--pd-transfer-backend",
        type=str,
        default="",
        help="sglang: mooncake|nixl ; vllm: NixlConnector|...; empty = default",
    )
    opt.add_argument(
        "--pd-ib-device",
        type=str,
        default="",
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). "
        "Empty = use $NCCL_IB_HCA from RayJob pod env at server-launch time.",
    )
    opt.add_argument(
        "--pd-prefill-ep",
        type=int,
        default=0,
        help="EP for the prefill group (disaggregated only); 0 = fall back to --ep. Multi-node PD only.",
    )
    opt.add_argument(
        "--pd-decode-ep",
        type=int,
        default=0,
        help="EP for the decode group (disaggregated only); 0 = fall back to --ep. Multi-node PD only.",
    )
    opt.add_argument(
        "--pd-prefill-extra-args",
        type=str,
        default="",
        help="Per-role sglang server args for the prefill group, appended after "
        "the shared server args (disaggregated only). Multi-node PD only.",
    )
    opt.add_argument(
        "--pd-decode-extra-args",
        type=str,
        default="",
        help="Per-role sglang server args for the decode group, appended after "
        "the shared server args (disaggregated only). Multi-node PD only.",
    )


def add_workload_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the profile OSL, reference-recipe, precision and framework-version flags of ``optimize``."""
    opt.add_argument(
        "--profile-osl",
        dest="profile_osl",
        type=int,
        default=None,
        help=(
            "Profiling-phase output sequence length. When set, it overrides "
            "--osl for the roofline/profile server ONLY, so its torch-profiler "
            "trace stays serializable; baseline/optimize phases still run at "
            "--osl. When unset the profile phase uses "
            "min(--osl, 1024) and is auto-lowered further if needed to keep the "
            "capture window within the serialization cap."
        ),
    )
    opt.add_argument(
        "--reference-script",
        dest="reference_script",
        type=str,
        default=None,
        help=(
            "Reference launch recipe (.sh path or http(s) URL). Its serve "
            "flags plus the exports the denylist allows seed the baseline "
            "server args at lowest priority (the config arm can override); shell-unsafe, "
            "credential-shaped and optimizer-owned workload variables are "
            "dropped. The recipe is applied as given — there is no model gate "
            "and no auto-discovery — so a path that cannot be read, or that "
            "yields neither a flag nor an export, exits 2 instead of falling "
            "back. Omit the flag to leave the baseline unchanged."
        ),
    )
    opt.add_argument("--precision", type=str, default=None, help=f"Model precision (default {DEFAULT_PRECISION})")
    opt.add_argument(
        "--framework-version",
        dest="framework_version",
        type=str,
        default=None,
        help=(
            "Framework version slug for the recipe-snapshot canonical id "
            "(scopes recipes to a specific framework release — sglang 0.4.5 "
            "and sglang 0.5.x have different scheduler defaults so they "
            "deserve separate KB rows). When omitted, auto-detected from "
            "the interpreter preflight resolved to serve the framework, "
            "else by importing the framework's top-level package and "
            "reading ``__version__`` (sglang/vllm/atom supported); auto-detect "
            "failure degrades to 'unknown_version'. Override with "
            "--framework-version=0.4.5 to pin a specific tag for the run."
        ),
    )


def add_target_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the stop-target flags of ``optimize``."""
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument(
        "--target-gain",
        type=float,
        default=None,
        help="Stop when cumulative_gain_validated >= N%% over baseline",
    )
    grp.add_argument(
        "--target-tput",
        type=float,
        default=None,
        help="Stop when current best reaches N, whole-server total (serving: tok/s; xDiT: img/s)",
    )
    grp.add_argument(
        "--target-baseline-dir", type=str, default=None, help="Stop when current best matches the baseline in DIR"
    )
    # Outside the group: it measures a different axis, so it ORs with whichever of the three above is set rather than
    # competing with them.
    opt.add_argument(
        "--target-roofline",
        type=float,
        default=None,
        help=(
            "Stop when the latest roofline snapshot reaches N%% of its modelled ceiling. Inert until a roofline "
            "has been measured or when the workload has no ceiling, and satisfied independently of "
            "--target-gain / --target-tput / --target-baseline-dir."
        ),
    )


def add_run_mode_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the multimodal-fallback, kernel-agent and launch-info flags of ``optimize``."""
    opt.add_argument(
        "--allow-mm-text-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a model carries a multimodal signal (e.g. vision_config) "
        "but exposes a text-generation path, run it on the TEXT path with "
        "a degraded-mode warning instead of fail-fasting. Image/audio "
        "inputs are ignored, so numbers reflect the text decoder alone. "
        "True VLMs with no text path (Llava / PaliGemma / Qwen-VL / "
        "Phi3V) still fail-fast regardless of this flag. Pass "
        "--no-allow-mm-text-fallback to fail-fast on text-coercible "
        "models too. Default: enabled.",
    )
    opt.add_argument(
        "--no-kernel",
        action="store_true",
        default=False,
        help="Disable the Kernel-agent entirely. The run will "
        "only do baseline + explore + sweep (pure "
        "parameter search). Useful when GEAK/GPU "
        "compile env is unavailable or you just want the "
        "quick-win parameter path. Default: kernel enabled.",
    )
    opt.add_argument(
        "--launch-info-file",
        type=str,
        default=None,
        help="Write a JSON file with the launched session's pid, "
        "session_dir, session_id, run_log, manifest path, gpu_type, "
        "framework and model. Launcher scripts can ``jq -r .pid`` / "
        "``jq -r .session_dir`` instead of grepping stdout or "
        "pgrep'ing. Always emitted alongside the ``HYPERLOOM_LAUNCH "
        "<key=value> ...`` single-line sentinel that is printed to "
        "stdout for stream-based parsers.",
    )


def add_framework_agent_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the framework-agent, enablement and local-explore flags of ``optimize``."""
    opt.add_argument(
        "--no-framework-agent",
        action="store_true",
        default=False,
        help="Skip the optimisation phase (PRELUDE → KERNEL_AGENT "
        "directly). The phase runs both levers: configuration grids, and "
        "upstream PRs plus specialist-authored source patches. Disable "
        "when the framework-agent toolchain is unavailable or you want a "
        "faster cold start. Default: enabled.",
    )
    opt.add_argument(
        "--enablement",
        dest="enablement",
        choices=["off", "launch", "eval", "all"],
        default="all",
        help="Admit the enablement self-heal lanes, which author framework "
        "patches when the baseline cannot be established. 'launch' handles a "
        "baseline that fails to boot; 'eval' handles one that boots but misses "
        "the accuracy floor; 'all' (default) handles both. Set 'off' to skip "
        "self-heal entirely: a baseline that keeps failing then terminates the "
        "run with stop_reason='baseline_failed' instead of opening an authoring "
        "loop, which is what you want for a quick triage run or when the model "
        "is already known to serve.",
    )
    opt.add_argument(
        "--no-framework-local-explore",
        dest="no_framework_local_explore",
        action="store_true",
        default=False,
        help="Disable the FRAMEWORK_AGENT local-exploration arm. By default, "
        "when PR discovery is empty/exhausted (or the ranker prefers it), the "
        "phase dispatches a write-capable specialist that authors a throughput "
        "patch from the live source + profiling evidence (and may web-search "
        "the latest upstream code) instead of skipping the phase. Disabling "
        "restores the historical behavior of exiting after "
        "DISCOVER_FAILURE_RETRY_LIMIT (3) empty/failed discoveries. Requires "
        "the authoring track (has no effect under diff-only mode).",
    )


def add_critic_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the Critic backend, protocol and prompt-override flags of ``optimize``."""
    # Critic backend selection; flags are aliases setting the same dest.
    opt.add_argument(
        "--critic-mock",
        dest="critic_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the always-approve mock Critic (offline / smoke tests).",
    )
    opt.add_argument(
        "--critic-agent",
        dest="critic_backend",
        action="store_const",
        const="agent",
        help="Force the critic-agent runtime backend (KB + session memory + "
        "review_constraints). Requires CRITIC_AGENT_ROOT or a sibling "
        "$REPO_ROOT/critic-agent/ directory.",
    )
    opt.add_argument(
        "--critic-protocol",
        dest="critic_protocol",
        choices=CRITIC_PROTOCOL_CHOICES,
        default="auto",
        help="Protocol for the Critic's review inference. 'openai' uses the "
        "OpenAI SDK; 'anthropic' uses the Messages API, or the Claude CLI when a "
        "CLAUDE_CODE_OAUTH_TOKEN subscription is the only credential. "
        "'auto' (default) reviews with the orchestration model over the protocol "
        "orchestration runs on; an explicit value reviews with that side's model "
        "(CLAUDE_MODEL or CODEX_MODEL) and fails at startup when that side has no "
        "credential. Preflight sends the review model one request and refuses to "
        "start when it does not answer; there is no fallback model. "
        "Ignored (with a warning) under --critic-mock, which runs no review "
        "inference.",
    )
    opt.add_argument(
        "--orch-prompt", type=str, default=None, help="Override Orchestration system prompt (file path or inline)"
    )
    opt.add_argument("--critic-prompt", type=str, default=None, help="Override Critic system prompt")


def add_kb_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the recipe KB and warm-replay flags of ``optimize``."""
    opt.add_argument(
        "--degraded-kb",
        dest="degraded_kb",
        action="store_true",
        default=False,
        help="Skip the recipe KB integration entirely (T0/T2/T3/T4 become "
        "no-ops). Also short-circuits any legacy IR-3 KB probe marker. "
        "Manifest records the reason as ``explicit_flag`` when set "
        "explicitly.",
    )
    # Warm-recipe replay: PRELUDE auto-applies KB best_config before optimising.
    opt.add_argument(
        "--no-warm-replay",
        dest="no_warm_replay",
        action="store_true",
        default=False,
        help="Disable the PRELUDE auto-replay of KB warm-start "
        "``best_config``. The warm_start_recipe is still rendered "
        "into the specialist prompt as priors, but the Coordinator "
        "will NOT auto-run the historical best_config. Use this "
        "for cold debugging / ablation runs.",
    )
    opt.add_argument(
        "--warm-replay-min-confidence",
        dest="warm_replay_min_confidence",
        type=float,
        default=None,
        help="Minimum ``warm_start_recipe.confidence`` required to "
        "trigger the auto-replay. Default 0.7 means an ``exact`` "
        "seven-tuple hit (conf 1.0) and a server-returned ``relative`` "
        "match (conf 0.7) both fire, while a ``miss`` (conf 0.0) "
        "does not. Raise it above 0.7 to require an exact hit "
        "before spending a verify on the warm config.",
    )


def add_service_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the PrimaTune predictor and PR Monitor flags of ``optimize``."""
    # PrimaTune predictor: off unless an endpoint is given, so a session that passes none behaves as before.
    opt.add_argument(
        "--primatune-endpoint",
        dest="primatune_endpoint",
        type=str,
        default=None,
        help="Base URL of a PrimaTune predictor service, asked once per FRAMEWORK_AGENT decision point. "
        "Its proposals join the untested-proposal queue beside the specialists' and are graded by the "
        "same KEEP threshold. Exported as $HYPERLOOM_PREDICTOR_ENDPOINT. "
        "See docs/reference/primatune-predictor.md.",
    )
    opt.add_argument(
        "--primatune-mode",
        dest="primatune_mode",
        type=str,
        choices=predictor_config.MODES,
        default=None,
        help="'shadow' (the default once an endpoint is set) asks and logs, queueing nothing; "
        "'active' queues the answer for benchmarking; 'off' disables. "
        "Exported as $HYPERLOOM_PREDICTOR_MODE.",
    )
    # PR Monitor REST + MCP are co-hosted by KB Store and derived from $KB_STORE_URL.
    opt.add_argument(
        "--degraded-pr",
        dest="degraded_pr",
        action="store_true",
        default=False,
        help="Disable the PR Monitor integration entirely. "
        "The specialist tool whitelist drops mcp__pr_monitor__* tools. "
        "Short-circuits the IR-3 PR Monitor probe; IR-3 sets this "
        "automatically when PR Monitor is unreachable (soft degrade).",
    )


def add_research_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the roofline and research-scout toggles of ``optimize``."""
    # Integration toggles.
    opt.add_argument(
        "--enable-roofline",
        dest="enable_roofline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select which analysis action the Coordinator enqueues at "
        "PRELUDE bootstrap and on every +10%% watermark crossing. "
        "Default on: ``roofline`` (composite profile + "
        "trace_analyze + analysis.md). Pass ``--no-enable-roofline`` "
        "to use plain ``profile`` instead (lighter — captures the "
        "trace only, skips trace_analyze). Behaviour is otherwise "
        "identical (same idempotency keys, same pending-task "
        "dispatch gate, same watermark anchor update).",
    )
    opt.add_argument(
        "--research-scout",
        dest="research_scout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-dispatch a read-only research scout at PRELUDE (and "
        "every --research-scout-interval optimisation rounds) that "
        "collects proven priors — reference launch scripts, model "
        "config.json architecture features, and cross-framework / "
        "NVIDIA research — into ``research_hints.md`` and seeds "
        "high-priority gaps. Default on; pass ``--no-research-scout`` "
        "to disable the whole feature.",
    )
    opt.add_argument(
        "--research-scout-interval",
        dest="research_scout_interval",
        type=int,
        default=3,
        help="Re-dispatch the research scout every N optimisation rounds with "
        "the current bottleneck context (append-only). Default 3. "
        "Ignored when ``--no-research-scout`` is set.",
    )


def add_recon_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the static-recon, recipe-sediment and target-advisory toggles of ``optimize``."""
    opt.add_argument(
        "--static-recon",
        dest="static_recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-dispatch a read-only static-recon specialist at PRELUDE, and "
        "again once per macro-cycle when the configuration arm plateaus, "
        "that greps the framework source for un-bridged capability "
        "switches (predicates that silently disable a faster path for "
        "this model/GPU/precision, e.g. a CUDA-only ``*_supported()`` "
        "on ROCm) and seeds bridge candidates as gaps[]. Default on; "
        "pass ``--no-static-recon`` to disable.",
    )
    opt.add_argument(
        "--recipe-sediment",
        dest="recipe_sediment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sediment KEEP/REVERT provenance into the persistent recipe: "
        "KEEP optimizations traceable to a research hint carry their "
        "source + measured gain into ``what_worked``; REVERTs land in "
        "``what_failed`` so the next warm-start avoids re-testing them. "
        "Default on; pass ``--no-recipe-sediment`` to keep the recipe "
        "purely ephemeral.",
    )
    opt.add_argument(
        "--target-advisory",
        dest="target_advisory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject an advisory 'External target gap' block (throughput / "
        "TPOT / interactivity gap vs the LLM-authored competitor "
        "target) into the orchestration and specialist prompts; when "
        "the TPOT ratio dominates it nudges toward latency-reducing "
        "directions. Advisory only — never gates Objective or scoring. "
        "Default on; pass ``--no-target-advisory`` to disable.",
    )


def add_conc_sweep_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the post-optimization concurrency-sweep flags of ``optimize``."""
    # Post-optimization concurrency sweep: a baseline-vs-optimized Magpie grid across CONC values (see
    # orchestrator/conc_sweep.py). Defaults to None so bootstrap can pick by benchmark mode.
    opt.add_argument(
        "--enable-conc-sweep",
        dest="enable_conc_sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run a post-optimization concurrency sweep (baseline vs "
        "current_best across CONC) and write "
        "reports/conc_sweep_summary.json + conc_sweep_raw.csv. "
        "On by default for synthetic workloads, off by default for MLPerf. "
        "Unavailable for native Magpie AgentX: start a separate session for each CONC.",
    )
    opt.add_argument(
        "--conc-sweep-concs",
        dest="conc_sweep_concs",
        type=str,
        default=None,
        help="Comma-separated CONC ladder for --enable-conc-sweep. Run with --conc first, then cheapest first "
        "(low-to-high for synthetic, high-to-low for MLPerf), so the order given does not matter. "
        "Defaults to the ladder for the workload: "
        "256,128,64,32,16,8,4,2 for synthetic, 1,4,8,10,14,20,28 for MLPerf. "
        "Unavailable for native Magpie AgentX.",
    )
    opt.add_argument(
        "--conc-sweep-total-budget-sec",
        dest="conc_sweep_total_budget_sec",
        type=int,
        default=9000,
        help="Total wall-clock budget (seconds) for the whole conc-sweep action, independent of the per-variant "
        "Magpie timeout. Past its first CONC the baseline arm may spend half of it; the optimized arm runs only the CONCs the "
        "baseline reached, so a budget that runs out drops whole pairs. Once exhausted, remaining variants are "
        "recorded as status=skipped / error_class=budget_exhausted and the JSON envelope carries "
        "budget_exhausted=true. Default 9000 (~2.5h); set to 0 to disable. Also bounded above by the main "
        "session wall-clock deadline since conc_sweep runs as a SWEEP-phase action.",
    )


def add_plateau_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the state-reset and config-arm plateau flags of ``optimize``."""
    # Per-variant explore overtime kill ratio (mirrored to SharedState.explore_overtime_kill_ratio). 0 disables.
    # Explore variant hard timeout — operator override for the auto-derived cap. 0 (default) keeps auto-derive;
    # mirrored to SharedState.explore_variant_timeout_sec_override.
    opt.add_argument(
        "--reset-state",
        dest="reset_state",
        action="store_true",
        default=False,
        help="Back up the existing ``state.json`` (if any) to "
        "``state.json.preReset.<unix_ts>`` and start the session "
        "from a blank SharedState. Recipe KB is NOT touched.",
    )
    # plateau threshold tuning: override defaults; locked at session start.
    opt.add_argument(
        "--plateau-explore-keep-gain",
        dest="plateau_explore_keep_gain",
        type=float,
        default=None,
        help="Config-arm plateau: max cumulative KEEP-gain (%%) across the "
        "lookback window below which the AND condition fires. "
        "Default 0.5.",
    )
    opt.add_argument(
        "--plateau-explore-empty-streak",
        dest="plateau_explore_empty_streak",
        type=int,
        default=None,
        help="Config-arm plateau: required count of *consecutive* specialist "
        "rounds with empty proposal_set before the AND condition "
        "fires. Default 5.",
    )
    opt.add_argument(
        "--plateau-explore-lookback",
        dest="plateau_explore_lookback",
        type=int,
        default=None,
        help="Config-arm plateau: number of trailing rounds the gain sum is computed over. Default 5.",
    )


def add_phase_budget_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the per-phase wall-clock budget flags of ``optimize``."""
    # phase budget percentages: each phase claims a fraction of the wall-clock budget (caps; may exit earlier).
    opt.add_argument(
        "--phase-budget-prelude-pct",
        dest="phase_budget_prelude_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for PRELUDE as a fraction of --max-hours. Default: 0.03. "
        "On --resume-from, any --*-pct flag resets every phase share to its default plus the given overrides; with none, the prior budget is kept.",
    )
    opt.add_argument(
        "--phase-budget-framework-pct",
        dest="phase_budget_framework_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for the OPTIMIZE (FRAMEWORK_AGENT) phase, which covers both configuration search "
        "and source landing. Default: 0.38. "
        "On --resume-from, any --*-pct flag resets every phase share to its default plus the given overrides; with none, the prior budget is kept.",
    )
    opt.add_argument(
        "--phase-budget-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL_AGENT. Default: 0.47. "
        "On --resume-from, any --*-pct flag resets every phase share to its default plus the given overrides; with none, the prior budget is kept.",
    )
    opt.add_argument(
        "--phase-budget-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.05. "
        "On --resume-from, any --*-pct flag resets every phase share to its default plus the given overrides; with none, the prior budget is kept.",
    )
    opt.add_argument(
        "--phase-budget-close-pct",
        dest="phase_budget_close_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for CLOSE. Default: 0.02. "
        "On --resume-from, any --*-pct flag resets every phase share to its default plus the given overrides; with none, the prior budget is kept.",
    )
