# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI argument parser — ``_build_parser`` and its purely-computational helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .. import framework_registry
from hyperloom.common.llm_config import DEFAULT_DEEPSEEK_MODEL
from hyperloom.orchestrator.roles.agent_role import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)
from hyperloom.orchestrator.scoring.proposal_scorer import DEFAULT_SCORER_MODELS

# Workload knob fallbacks applied when the operator passes neither the CLI flag
# nor an inherited value. Flags default to ``None`` so "omitted" is
# distinguishable from "typed the default"; the resolver in ``cli`` applies
# these constants only for genuinely-unset knobs (issue #903).
DEFAULT_ISL = 1024
DEFAULT_OSL = 1024
DEFAULT_CONC = 64
DEFAULT_TP = 1
DEFAULT_EP = 1
DEFAULT_PRECISION = "bf16"


class _RetiredFlag(argparse.Action):
    """Argparse action that hard-fails on a retired CLI flag with a migration hint (exits 2)."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        hint: str,
        **kwargs,
    ) -> None:
        """Register the retired flag as a zero-argument, hidden action.

        Args:
            option_strings (list[str]): The flag spellings this action handles.
            dest (str): The argparse destination name (unused; suppressed).
            hint (str): One-line migration hint shown in the error message when
                the retired flag is used.
            **kwargs: Passed through to :class:`argparse.Action`; ``nargs``,
                ``default``, and ``help`` are defaulted so the flag takes no
                value and stays out of ``--help``.
        """
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs.setdefault("help", argparse.SUPPRESS)
        self._hint = hint
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        """Abort parsing with a migration hint when the retired flag is seen.

        Args:
            parser (argparse.ArgumentParser): The parser invoking this action.
            namespace (argparse.Namespace): The in-progress parse namespace.
            values (Any): Parsed values for the flag (always empty; ``nargs=0``).
            option_string (str | None): The exact flag spelling that triggered this.

        Raises:
            SystemExit: Always — ``parser.error`` prints the message and exits 2.
        """
        parser.error(f"{option_string} was removed. {self._hint}")


def _positive_int_arg(value: str) -> int:
    """argparse type for positive integer knobs."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _default_claude_model_env() -> str:
    """Resolve the default Claude model from env."""
    explicit = (os.environ.get("CLAUDE_MODEL") or "").strip()
    if explicit:
        return explicit
    if (os.environ.get("DEEPSEEK_API_KEY") or "").strip() or (os.environ.get("DEEPSEEK_BASE_URL") or "").strip():
        return (os.environ.get("DEEPSEEK_MODEL") or "").strip() or DEFAULT_DEEPSEEK_MODEL
    if os.environ.get("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX") == "1":
        return (os.environ.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
    openai_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    anthropic_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    if openai_url and not anthropic_url:
        return (os.environ.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
    return DEFAULT_CLAUDE_MODEL


def _default_codex_model_env() -> str:
    """Resolve the default Codex-style model from env.

    When the operator only configured the Anthropic side, Codex-style JSON
    roles run through the unified gateway with the same Claude model as
    orchestration. This also overrides generated default ``CODEX_MODEL`` values
    from setup env files. Explicit ``CODEX_MODEL`` keeps the historical
    OpenAI-compatible behavior when an OpenAI base URL is configured.
    """
    anthropic_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    openai_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if (
        (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        or (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        or anthropic_url == "https://api.deepseek.com/anthropic"
    ) and not openai_url:
        return (os.environ.get("CLAUDE_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "").strip() or DEFAULT_DEEPSEEK_MODEL
    if anthropic_url and not openai_url:
        return (os.environ.get("CLAUDE_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL
    explicit = (os.environ.get("CODEX_MODEL") or "").strip()
    if explicit:
        return explicit
    return DEFAULT_CODEX_MODEL


def _default_research_lane_capacity() -> int:
    """Default ``--research-lane-capacity`` to the GPU ceiling (2×GPU).

    Returns:
        int: The policy GPU ceiling.
    """
    from hyperloom.orchestrator.policy.gate import research_lane_ceiling

    return research_lane_ceiling()


def _default_gpu_specialist_capacity() -> int:
    """Default ``--gpu-specialist-capacity`` to the whole visible machine.

    WS2 turns GPU specialists on by default at whole-machine capacity. When the
    operator needs a different value, pass ``--gpu-specialist-capacity``.

    Returns:
        int: The detected whole-machine GPU count, or ``0`` when nothing can be probed.
    """
    from hyperloom.orchestrator.policy.gate import detect_gpu_count

    return detect_gpu_count()


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser and subcommands.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    from hyperloom.orchestrator.specialists.domains import (
        DEFAULT_SPECIALIST_MAX_TURNS as _DEFAULT_SPECIALIST_MAX_TURNS,
    )

    p = argparse.ArgumentParser(
        prog="inference_optimizer",
        description="Inference Optimizer — multi-agent inference optimization (SGLang/vLLM/Atom/xDiT)",
    )
    p.add_argument("--verbose", "-v", action="count", default=0, help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Drive a multi-agent optimization run on a model")
    opt.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help="Model path (required for new runs; ignored when "
        "--resume is set — model is read from manifest.json/"
        "state.json)",
    )
    opt.add_argument(
        "--quantize",
        type=str,
        default=None,
        metavar="PROMPT",
        help="Optional natural-language quantization request. When set, the "
        "quantization-agent runs ONCE as a prelude before the "
        "optimization loop: it drives AMD Quark PTQ from this prompt, "
        "then rewrites --model to the exported quantized model so the "
        "rest of the run optimizes the quantized model. Ignored on "
        "--resume.",
    )
    from hyperloom.orchestrator.phases.quantization_schemes import QUANT_SCHEME_CHOICES

    opt.add_argument(
        "--quantize-scheme",
        choices=QUANT_SCHEME_CHOICES,
        default=None,
        metavar="SCHEME",
        help="Structured alternative to --quantize for UI/backends: pick a "
        "curated quantization scheme (resolved to a prompt internally). "
        f"Choices: {', '.join(QUANT_SCHEME_CHOICES)}. 'none' or omit = no "
        "quantization. Ignored if --quantize (free text) is also given.",
    )
    opt.add_argument(
        "--gpu-type",
        type=str.lower,
        choices=["mi300x", "mi308x", "mi325x", "mi355x"],
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
        "--framework",
        choices=list(framework_registry.names()),
        default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
        "--framework > sglang (default). Selection is "
        "session-wide; mixing frameworks in a single session is not "
        "supported. NOTE: --framework atom is single-node-only "
        "(``--nodes>=2`` fails fast); profile / roofline, "
        "kernel-agent, and framework-agent are all enabled on atom. "
        "The auto-tighten guard only enforces ``--nodes 1``. "
        "--framework xdit is a server-less (scriptable) diffusion "
        "workload (xDiT): no serving server, throughput is img/s, and "
        "the accuracy gate is an image-quality gate (LPIPS/SSIM/MSE).",
    )
    opt.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Total number of GPU nodes for the inference RayJob. "
        "1 (default) keeps the legacy single-pod path. "
        ">=2: `optimize` provisions the SaFE RayJob before preflight "
        "(unless already in /tmp/multi_node_state.json), runs bootstrap "
        "once, and exports RAY_ADDRESS for kernel-agent. Does not stop the "
        "RayJob on exit; run `python3 -m hyperloom.inference_optimizer.multi_node "
        "stop-multi-job` when you want to release it. Requires "
        "--mn-image or INFERENCE_OPTIMIZER_MN_IMAGE. "
        "Default: 1.",
    )
    opt.add_argument(
        "--mn-backend",
        choices=("rayjob", "infera"),
        default=None,
        help="Multi-node backend when --nodes>=2: 'rayjob' (default, Ray "
        "head+workers) or 'infera' (idle InferaDeployment + SSH control "
        "plane). Resolution: --mn-backend > $INFERENCE_OPTIMIZER_MN_BACKEND "
        "> rayjob. Single-node runs ignore this flag.",
    )
    opt.add_argument(
        "--mn-image",
        default=None,
        help="Container image for the multi-node pods (Infera worker/prefill/"
        "decode pods, or RayJob head+workers). Required when --nodes>=2 unless "
        "INFERENCE_OPTIMIZER_MN_IMAGE is set or state file "
        "last_create_request.image is present.",
    )
    opt.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="GPUs per multi-node pod (Infera worker/prefill/decode or RayJob "
        "head+workers). Resolution: flag > INFERENCE_OPTIMIZER_GPUS_PER_NODE > 8.",
    )
    opt.add_argument(
        "--cpus-per-node",
        type=int,
        default=None,
        help="CPU cores requested per multi-node pod (Infera or RayJob). "
        "Resolution: flag > 96 (default).",
    )
    opt.add_argument(
        "--mem-per-node",
        type=int,
        default=None,
        help="Memory (GiB) requested per multi-node pod (Infera or RayJob). "
        "Resolution: flag > 1024 (default).",
    )
    # --rayjob-extra-env is a prompt-driven pass-through forwarded verbatim to workload_spec.env; the CLI
    # invents no keys. Reserved RAY_JOB_ENTRYPOINT stripped downstream; credential keys auto-injected elsewhere.
    opt.add_argument(
        "--rayjob-extra-env",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra env entries to inject into the multi-node RayJob "
        "(repeatable). Agent maps each line of the user prompt's "
        "`env:` block into one --rayjob-extra-env K=V; the CLI "
        "does not own any default. Skip RAY_JOB_ENTRYPOINT "
        "(reserved by workload_spec). Only takes effect when "
        "--nodes>=2 and this run actually creates the RayJob; "
        "idempotent reuse of an existing rayjob_id keeps the env "
        "set at original create time.",
    )
    opt.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size. Pass `--tp N` directly from the prompt's "
        f"Environment block. Default: {DEFAULT_TP}.",
    )
    opt.add_argument(
        "--conc",
        type=_positive_int_arg,
        default=None,
        help="Magpie client concurrency cap (max in-flight requests). "
        "Pass `--conc N` directly from the prompt. Use "
        f"--conc-sweep-concs for a concurrency ladder. Default: {DEFAULT_CONC}.",
    )
    opt.add_argument(
        "--max-model-len",
        dest="max_model_len",
        type=_positive_int_arg,
        default=None,
        help="Explicit server-facing MAX_MODEL_LEN. Resolution: "
        "--max-model-len > auto(ISL+OSL+headroom, "
        "clamped to native context). Explicit values are preserved and "
        "exported into the materialized Magpie YAML.",
    )
    opt.add_argument(
        "--server-args",
        dest="server_args",
        type=str,
        default="",
        help="Framework server args to apply in every phase. Routed through "
        "the framework-specific EXTRA_*_ARGS env in Magpie YAMLs "
        "(EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS / EXTRA_ATOM_ARGS). "
        "Example: --server-args \"--kv-cache-dtype fp8_e4m3 "
        "--gpu-memory-utilization 0.85\".",
    )
    opt.add_argument(
        "--ep",
        type=int,
        default=None,
        help="Expert-parallel size for MoE inference. 1 (default) keeps "
        "experts sharded by TP (legacy behaviour). >=2 enables true "
        "expert parallelism: sglang adds `--expert-parallel-size N`, "
        "vllm adds `--enable-expert-parallel`. Typical: EP=TP for "
        "DSr1/DSv3 on multi-node. Default: 1. "
        "EP > TP is rejected at server-restart time.",
    )
    opt.add_argument(
        "--pd-mode",
        choices=("colocated", "disaggregated"),
        default="colocated",
        help="Prefill-Decode disaggregation mode. ALWAYS defaults to "
        "`colocated` regardless of any inherited $PD_MODE env, so "
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
        help="EP for the prefill group (disaggregated only); 0 = fall back to "
        "--ep. Multi-node PD only.",
    )
    opt.add_argument(
        "--pd-decode-ep",
        type=int,
        default=0,
        help="EP for the decode group (disaggregated only); 0 = fall back to "
        "--ep. Multi-node PD only.",
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
    opt.add_argument(
        "--skip-variants",
        type=str,
        default="",
        help="Comma/whitespace-separated list of variant names or fnmatch "
        "globs to drop from the backends/params grids before launch. "
        "Examples: `attn_aiter` (exact), `attn_aiter,sched_dfs` (two "
        "exacts), `attn_*,vllm_aiter_*` (globs). Exported into "
        "the internal SKIP_VARIANTS handoff so all executors and the multi-node orchestrator "
        "subprocess see the same value. Dropped variants surface in "
        "state.json under each action's `dropped_variants` field tagged "
        "`source=user_skip`.",
    )
    opt.add_argument("--max-hours", type=float, default=2.0, help="Wall-clock budget in hours (default 2.0)")
    opt.add_argument(
        "--closing-grace-sec",
        type=float,
        default=None,
        help=(
            "Extra seconds after the wall-clock deadline for Coordinator to "
            "flush a deterministic report task (no LLM). Default: "
            "min(120, max_hours * 60 * 0.02). Pass 0 to disable closing phase."
        ),
    )
    opt.add_argument("--isl", type=int, default=None, help=f"Input sequence length (default {DEFAULT_ISL})")
    opt.add_argument("--osl", type=int, default=None, help=f"Output sequence length (default {DEFAULT_OSL})")
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
            "flags + whitelisted exports seed the baseline server args at "
            "lowest priority (EXPLORE can override). Model-gated: ignored if "
            "the run's model differs from the recipe's. If the given path is "
            "unreadable / yields no flags, a matching InferenceX single-node "
            "recipe is auto-discovered (exact filename match only; fuzzy "
            "matches are logged, not used). When this flag is omitted, no "
            "discovery runs and the baseline is unchanged."
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
            "deserve separate KB rows). When omitted, auto-detected via "
            "importing the framework's top-level package and reading "
            "``__version__`` (sglang/vllm/atom supported); auto-detect "
            "failure degrades to 'unknown_version'. Override with "
            "--framework-version=0.4.5 to pin a specific tag for the run."
        ),
    )
    opt.add_argument(
        "--continue-kernel-after-gemm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After FP8 GEMM tuning succeeds, continue into source-level "
            "kernel_opt. Use --no-continue-kernel-after-gemm for a "
            "GEMM-only validation run."
        ),
    )
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument("--target-gain", type=float, default=None, help="Stop when cumulative_gain >= N%% over baseline")
    grp.add_argument(
        "--target-tput",
        type=float,
        default=None,
        help="Stop when current best reaches N (serving: tok/s/GPU; xDiT: img/s)",
    )
    grp.add_argument(
        "--target-baseline-dir", type=str, default=None, help="Stop when current best matches the baseline in DIR"
    )
    opt.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an existing session. Without --resume-from, "
        "auto-picks the latest per-session subdir under "
        "$USER_DATA_PATH/<model>/<UTC ts>/ (N17 layout) or "
        "falls back to $USER_DATA_PATH (legacy flat layout). "
        "USER_DATA_PATH MUST stay at workspace level "
        "(/path/.../sessions/, not the per-session subdir) "
        "so runtime/ resolution works. Skips the SharedState "
        "seed and lets the Coordinator replay the prior "
        "event log + state.json.",
    )
    opt.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Explicit per-session subdir to resume from. Use "
        "when multiple per-launch ts dirs exist under the "
        "same model and the latest is not what you want. "
        "Must be an absolute path under $USER_DATA_PATH "
        "(workspace_root). Implies --resume.",
    )
    opt.add_argument(
        "--force-resume",
        action="store_true",
        default=False,
        help=(
            "Allow ``--resume`` to push past a terminal "
            "``stop_reason='target_reached'``. "
            "Without this flag the resume aborts (Issue-G guard, per "
            "SKILL.md 'Run-time signals': that terminal requires an "
            "operator-side workload / strategy change before resuming). "
            "No-op outside ``--resume``."
        ),
    )
    opt.add_argument(
        "--model-class",
        type=str,
        default=None,
        help=(
            "Categorical model-class key. It is the deterministic key for "
            "several consumers: the atom explore seed grid "
            "(action_executors/explore.py), the framework-agent gap search "
            "token (action_executors/_framework_gap_composer.py), the recipe "
            "key, and the orchestration prompt label. Recognised values "
            "(case-insensitive, with -/+/space tolerated): dense / moe_mla / "
            "moe_swa / moe_mla_nsa. When unset, Coordinator boot infers and "
            "persists it from config.json (num_experts / architectures) or "
            "model-path family keywords, replacing the deleted ``classify`` "
            "action's lightweight state-write role. For richer *advisory* "
            "model context (attention variant, KV/token, experts, MTP, ...) "
            "the SKILL launcher writes $USER_DATA_PATH/model_arch.json, which "
            "is injected into prompts but drives no gating."
        ),
    )
    opt.add_argument("--target-summary", type=str, default=None, help="Free-text goal summary surfaced in prompts")
    opt.add_argument(
        "--compare-against-gpu",
        type=str,
        default=None,
        help=(
            "Reference GPU hardware key for external baseline comparison "
            "(e.g. b300 / mi355x / h200). target_analysis ALWAYS runs as "
            "TODO 0 and always writes "
            "$SESSION_DIR/target_analysis/target_baseline.json + a short "
            "MD report. When this flag is set, the JSON carries the "
            "matching InferenceX (https://inferencex.semianalysis.com) "
            "reference data point; when unset, the JSON carries a "
            "structured reason='no_target_gpu_configured' marker so the "
            "report still has a deterministic 'External baseline' "
            "section. The reference is API-measured and never influences "
            "the Objective, scoring, or any KEEP/REVERT gate; a matching "
            "row is surfaced to the EXPLORE gap advisory as direction only. "
            "Other dimensions (model / framework / precision / ISL / OSL) "
            "are derived from the corresponding CLI arguments."
        ),
    )
    opt.add_argument("--max-ticks", type=int, default=None, help="Hard tick cap (None = unlimited; mostly for tests)")
    opt.add_argument("--tick-interval-sec", type=float, default=0.0, help="Sleep between ticks (0 = no sleep)")
    opt.add_argument("--claude-model", type=str, default=_default_claude_model_env())
    opt.add_argument("--codex-model", type=str, default=_default_codex_model_env())
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
        "--no-explore",
        action="store_true",
        default=False,
        help="Skip the EXPLORE phase entirely. PRELUDE (and "
        "FRAMEWORK, if enabled) route straight to KERNEL "
        "— or to SWEEP when --no-kernel is also set. Useful "
        "for a baseline -> kernel-only run, or to validate "
        "the current recipe via SWEEP without a serving-"
        "param search. Default: explore enabled.",
    )
    opt.add_argument(
        "--enable-framework-config-exploration",
        action="store_true",
        default=False,
        help="(Stage-1, default OFF) Let the FRAMEWORK_AGENT phase run "
        "explore-style config-grid exploration (reusing the ExploreExecutor) "
        "before it advances, giving FRAMEWORK the EXPLORE config-search "
        "capability. The EXPLORE phase and overall phase flow are unchanged; "
        "results share the explore_search dedup ledger.",
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
    opt.add_argument(
        "--framework-discover-timeout-sec",
        type=float,
        default=0.0,
        help="Override the per-call timeout for "
        "``fa phase-discover``. 0 (the default) uses "
        "framework_agent_client.DEFAULT_FA_PHASE_TIMEOUT_SEC "
        "(180s). The Coordinator retries discover up to "
        "DISCOVER_FAILURE_RETRY_LIMIT (3) consecutive "
        "failures before marking FRAMEWORK done.",
    )
    opt.add_argument(
        "--no-framework-agent",
        action="store_true",
        default=False,
        help="Skip the FRAMEWORK_AGENT phase (PRELUDE → EXPLORE "
        "directly). The phase pre-scans upstream sglang/"
        "vllm PRs via framework-agent and lands KEPT "
        "patches before EXPLORE starts. Disable when "
        "the framework-agent toolchain is unavailable "
        "or you want a faster cold start. "
        "Default: framework phase enabled.",
    )
    opt.add_argument(
        "--kernel-codex",
        action="store_true",
        default=True,
        help=(
            "Use Codex for the Kernel-agent conversation backend (default — "
            "faster). This does not select the forge kernel rewrite "
            "ladder; use KERNEL_OPT_BACKEND_ORDER for that. Pass --kernel-claude "
            "to switch the conversation backend."
        ),
    )
    opt.add_argument(
        "--kernel-claude",
        action="store_false",
        dest="kernel_codex",
        help=(
            "Use Claude for the Kernel-agent conversation backend. This does not "
            "select the forge kernel rewrite ladder."
        ),
    )
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
    # Robustness backend selection (mirrors critic)
    opt.add_argument(
        "--robustness-mock",
        dest="robustness_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the heartbeat-only mock Robustness backend.",
    )
    opt.add_argument(
        "--robustness-agent",
        dest="robustness_backend",
        action="store_const",
        const="agent",
        help="Force the robustness-agent runtime backend (subprocess + JSON, "
        "mirrors critic-agent transport). Requires ROBUSTNESS_AGENT_ROOT "
        "or a sibling $REPO_ROOT/robustness-agent/ directory.",
    )
    opt.add_argument(
        "--robustness-server-url",
        dest="robustness_server_url",
        type=str,
        default=None,
        help="Override the robustness-server base URL forwarded into "
        "request.options. Honoured only when --robustness-agent is "
        "selected.",
    )
    opt.add_argument(
        "--robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_true",
        default=None,
        help="Forward llm_rca_enabled=true into request.options. The agent "
        "still falls back to NoopRcaEngine when LLM credentials aren't "
        "set in the runtime env.",
    )
    opt.add_argument(
        "--no-robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_false",
        help="Forward llm_rca_enabled=false into request.options.",
    )
    opt.add_argument(
        "--robustness-workload-uid",
        dest="robustness_workload_uid",
        type=str,
        default=None,
        help="Forward workload_uid into request.options. The robustness-server "
        "resolves it to every pod (head + workers) backing the RayJob via "
        "the cluster/workloads/{uid}/hierarchy endpoint. Falls back to "
        "$CLAW_WORKLOAD_UID / $WORKLOAD_UID / $RAY_JOB_ID when unset.",
    )
    opt.add_argument(
        "--robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_true",
        default=None,
        help="Force disable_local_probe=true. The robustness-agent silences "
        "its LocalProbe fallback so per-pod sandbox checks (ps, rocm-smi, "
        "local HTTP) cannot emit false-positive symptoms.",
    )
    opt.add_argument(
        "--no-robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_false",
        help="Force disable_local_probe=false (keep the LocalProbe fallback even in multi-node mode).",
    )
    opt.add_argument(
        "--robustness-disable-server-probe",
        dest="robustness_disable_server_probe",
        action="store_true",
        default=None,
        help="Force auto_probe_inference_server=false: stop the robustness-agent "
        "from auto-probing the local inference-server health endpoint "
        "(http://127.0.0.1:8888/health). Unlike --robustness-disable-local-probe "
        "this is surgical — the REST of LocalProbe (gpu-leak, gateway 401, "
        "coordinator-zombie, aiter-JIT, disk/fd) stays active. Use on "
        "single-node runs where the optimizer restarts the inference server "
        "between benchmarks: those restart windows otherwise trip "
        "false-positive local_server_unreachable symptoms (which can escalate "
        "to a premature skip_to_close / robustness_escalated stop). "
        "Auto-enabled in multi-node.",
    )
    opt.add_argument(
        "--no-robustness-disable-server-probe",
        dest="robustness_disable_server_probe",
        action="store_false",
        help="Force auto_probe_inference_server=true (keep the 127.0.0.1:8888 "
        "/health auto-probe even in multi-node mode).",
    )
    opt.add_argument(
        "--robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_true",
        default=None,
        help="Force enable_cluster_pod_metrics=true so the robustness-agent "
        "fans out per-pod metrics through robustness-server and feeds "
        "the local_health rules with cluster-decoded GPU snapshots.",
    )
    opt.add_argument(
        "--no-robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_false",
        help="Force enable_cluster_pod_metrics=false.",
    )
    opt.add_argument(
        "--robustness-pod-metrics-categories",
        dest="robustness_pod_metrics_categories",
        type=str,
        default=None,
        help="Comma-separated metric categories forwarded into "
        "pod_metrics_categories (e.g. 'gpu,memory'). Default 'gpu' is "
        "applied by the runtime when this flag is omitted.",
    )
    opt.add_argument(
        "--orch-prompt", type=str, default=None, help="Override Orchestration system prompt (file path or inline)"
    )
    opt.add_argument("--critic-prompt", type=str, default=None, help="Override Critic system prompt")
    opt.add_argument("--kernel-prompt", type=str, default=None, help="Override Kernel system prompt")
    # Cortex KB flag (Critic agent only).
    opt.add_argument(
        "--cortex-kb-url",
        dest="cortex_kb_url",
        type=str,
        default=None,
        help="Cortex KB base URL for this run, used only by the Critic "
        "agent's per-proposal assess enrichment (/v2/reasoning/assess). "
        "This flag is the single source of truth: the CLI no longer reads a "
        "$CORTEX_KB_URL env fallback (the flag value is forwarded to the "
        "critic subprocess as CORTEX_KB_URL). This is NOT the recipe KB — "
        "recipe reads are served by gbrain ($GBRAIN_*) and writes always "
        "go to --local-kb-root. Leave it UNSET to skip Critic assess "
        "enrichment entirely.",
    )
    opt.add_argument(
        "--specialist-kb-mcp-url",
        dest="specialist_kb_mcp_url",
        type=str,
        default=None,
        help="Read-only KB-graph (cortex_kb) MCP endpoint advertised to "
        "specialist subprocesses; also settable via "
        "$HYPERLOOM_SPECIALIST_KB_MCP_URL (bearer token from "
        "$HYPERLOOM_SPECIALIST_KB_MCP_TOKEN). When unset, falls back to "
        "the gbrain MCP derived from $GBRAIN_BASE_URL (+ /mcp) with a "
        "$GBRAIN_TOKEN bearer. Specialist KB writes always stay local.",
    )
    opt.add_argument(
        "--local-kb-root",
        dest="local_kb_root",
        type=str,
        default=None,
        help="Filesystem root for the local recipe-snapshot KB store. "
        "All writes (put_recipe / append_attempt / delete_recipe) go "
        "here regardless of --cortex-kb-url. Defaults to "
        "$HYPERLOOM_LOCAL_KB_ROOT, then $USER_DATA_PATH/kb, "
        "then /workspace/hyperloom/kb. Layout is a 5-level "
        "directory tree keyed by canonical_id components "
        "(model -> hardware -> framework -> framework_version -> "
        "precision); each leaf holds recipe.json + history/ + "
        "attempts.ndjson + .lock. See "
        "src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py for the "
        "on-disk contract.",
    )
    opt.add_argument(
        "--degraded-kb",
        dest="degraded_kb",
        action="store_true",
        default=False,
        help="Skip the Cortex KB integration entirely (T0/T2/T3/T4 become "
        "no-ops). Also short-circuits the IR-3 KB probe. IR-3 sets "
        "this automatically when kb-service is unreachable (soft "
        "degrade); manifest records the reason as ``explicit_flag`` "
        "vs ``ir3_auto``.",
    )
    opt.add_argument(
        "--cortex-strict-fingerprint",
        dest="cortex_strict_fingerprint",
        action="store_true",
        default=False,
        help="When set, T0 refuses warm_start_recipe rows whose "
        "stack_fingerprint does not match the current pod (recorded "
        "in manifest.json). Default: lenient (M1 records the flag "
        "in manifest only; consumed by M5 specialist assembly).",
    )
    # Warm-recipe replay: PRELUDE auto-applies KB best_config before EXPLORE.
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
        default=0.7,
        help="Minimum ``warm_start_recipe.confidence`` required to "
        "trigger the auto-replay. Default 0.7 means an ``exact`` "
        "5-tuple hit (conf 1.0) and a server-returned ``relative`` "
        "match (conf 0.7) both fire, while a ``miss`` (conf 0.0) "
        "does not. Raise it above 0.7 to require an exact hit "
        "before spending a verify on the warm config.",
    )
    opt.add_argument(
        "--warm-replay-min-reproduce-pct",
        dest="warm_replay_min_reproduce_pct",
        type=float,
        default=0.8,
        help="Minimum fraction of the recipe's recorded gain we need "
        "to reproduce to count as ``status=reproduced`` and push "
        "the warm config onto the optimization stack. Default "
        "0.8 — a recipe claiming +25%% counts if we measure "
        "+20%% or more. Below the threshold we record "
        "``status=drift`` and continue with the regular EXPLORE "
        "flow without inheriting the warm config.",
    )
    # PR Monitor REST + MCP.
    opt.add_argument(
        "--pr-monitor-url",
        dest="pr_monitor_url",
        type=str,
        default=(os.environ.get("PRIMUS_CORTEX_PR_API") or "").strip() or None,
        help="PR Monitor REST URL for this run (flag wins). Default: "
        "$PRIMUS_CORTEX_PR_API (the canonical PR API env), else unset. "
        "Set this flag / $PRIMUS_CORTEX_PR_API to a reachable HTTPS "
        "endpoint for your PR Monitor deployment. Pair "
        "with --pr-monitor-mcp-url when port-forwarding for local debug.",
    )
    opt.add_argument(
        "--pr-monitor-mcp-url",
        dest="pr_monitor_mcp_url",
        type=str,
        default=None,
        help="PR Monitor MCP URL handed to specialist LLM backends (flag "
        "wins). Default: the in-cluster MCP endpoint. The trailing slash "
        "is mandatory.",
    )
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
    opt.add_argument(
        "--pr-feed-window-days",
        dest="pr_feed_window_days",
        type=int,
        default=30,
        help="Look-back window for the PR feed warmup (days). Default: 30.",
    )
    # specialist research_lane capacity (locked at session start).
    opt.add_argument(
        "--research-lane-capacity",
        dest="research_lane_capacity",
        type=int,
        default=_default_research_lane_capacity(),
        help="Max concurrent LLM specialist sub-agents on the "
        "research_lane. 0 disables specialist "
        "dispatch entirely (degrades to LLM-direct grid). The "
        "default is the research-lane ceiling (2 x visible GPU "
        "count, falling back to a conservative value when no GPU "
        "is detected); values above the ceiling are silently "
        "clamped down. Locked at session start.",
    )
    opt.add_argument(
        "--gpu-specialist-capacity",
        dest="gpu_specialist_capacity",
        type=int,
        default=_default_gpu_specialist_capacity(),
        help="Number of GPUs available to specialists that request "
        "needs_gpu=true. Defaults to the whole machine (visible GPU "
        "count on the launch host); set "
        "--gpu-specialist-capacity 0 to disable GPU specialists. GPU specialists serialize against the "
        "serving lanes via gpu_research_lane. "
        "Set INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES to a "
        "comma-separated GPU id pool when the specialist pool should "
        "not use device ids 0..N-1. Locked at session start.",
    )
    # Advisory specialist-proposal scorer (ProposalScorer): scores each
    # proposal_set with gateway models as a reference for Orchestration; never gates.
    opt.add_argument(
        "--proposal-scorer-models",
        dest="proposal_scorer_models",
        type=str,
        default=",".join(DEFAULT_SCORER_MODELS),
        help="Comma-separated gateway model slugs that independently "
        "score each specialist proposal_set (advisory only; never "
        "gates; rater identities are anonymized in the orchestration "
        "prompt). Only takes effect when --proposal-scoring is also "
        "passed (scoring is OFF by default); this flag alone does not "
        "enable scoring. Default 'claude-opus-4-8,gpt-5.5,"
        "dvue-aoai-005-Kimi-K2.6,gemini/gemini-3.1-pro-preview'. "
        "Add a model by appending its slug. Empty list disables scoring "
        "even when enabled.",
    )
    opt.add_argument(
        "--proposal-scoring",
        dest="proposal_scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the advisory specialist-proposal scorer (disabled by "
        "default). Use --no-proposal-scoring to keep it off explicitly. "
        "Even when enabled it is skipped in Anthropic-only deployments "
        "(OpenAI-compatible only) or when the model list is empty. "
        "Advisory only; never gates.",
    )
    # Retired flag: hard-fail with a migration hint instead of an opaque error.
    opt.add_argument(
        "--enable-proposal-scoring",
        action=_RetiredFlag,
        hint="Use ``--proposal-scoring`` (default off) / ``--no-proposal-scoring`` instead.",
    )
    # specialist sub-agent backend selection: Claude (default), inherits orchestration model.
    opt.add_argument(
        "--specialist-model",
        dest="specialist_model",
        type=str,
        default=None,
        help="Claude model used for specialist sub-agents (defaults to "
        "the orchestration --claude-model). KB_design §3.5 §6.",
    )
    opt.add_argument(
        "--specialist-max-turns",
        dest="specialist_max_turns",
        type=int,
        default=_DEFAULT_SPECIALIST_MAX_TURNS,
        help="Hard cap on LLM turns per specialist task (KB_design "
        "§3.5 §6). On exhaustion the runner synthesises an empty "
        "specialist_done (Inv-5.3).",
    )
    opt.add_argument(
        "--specialist-per-turn-max-seconds",
        dest="specialist_per_turn_max_seconds",
        type=float,
        default=600.0,
        help="Wall-clock fallback ceiling per specialist task when no "
        "explicit wall_budget_sec is provided (legacy backstop, default "
        "600s; production dispatches use the WS1 wall-clock budget).",
    )
    # specialist dispatch shape
    opt.add_argument(
        "--specialist-dispatch-mode",
        dest="specialist_dispatch_mode",
        type=str,
        choices=("subprocess", "inprocess"),
        default="subprocess",
        help="Specialist execution shape. 'subprocess' (default) spawns "
        "a fresh `claude` CLI per task inside a per-task git worktree "
        "for isolation (PR-A2). 'inprocess' keeps the legacy M5 path "
        "(claude-agent-sdk in the orchestrator process) for tests / "
        "environments without the claude binary.",
    )
    opt.add_argument(
        "--specialist-mcp-config",
        dest="specialist_mcp_config",
        type=str,
        default=None,
        help="Optional path to an MCP config JSON forwarded to the "
        "subprocess claude (`--mcp-config`). Used to wire kb / pr "
        "MCP servers into specialists. Default: None.",
    )

    # Integration toggles. Roofline refresh is unconditional (fires at PRELUDE
    # and every 10% cumulative_gain_validated crossing).
    opt.add_argument(
        "--allow-empty-kernel-shape",
        dest="allow_empty_kernel_shape",
        action="store_true",
        default=False,
        help="Escape hatch (default off): allow kernel optimization to "
        "dispatch a candidate with no trace-anchored shape. Normally "
        "a shapeless candidate is rejected with a structured error so "
        "the run returns to ``trace_analyze`` instead of burning a "
        "kernel-optimization budget on an unanchored kernel.",
    )
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
        "every --research-scout-interval EXPLORE rounds) that "
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
        help="Re-dispatch the research scout every N EXPLORE rounds with "
        "the current bottleneck context (append-only). Default 3. "
        "Ignored when ``--no-research-scout`` is set.",
    )
    opt.add_argument(
        "--static-recon",
        dest="static_recon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-dispatch a read-only static-recon specialist at PRELUDE "
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
    # Post-optimization concurrency sweep (on by default): a baseline-vs-optimized
    # Magpie grid across CONC values (see orchestrator/conc_sweep.py).
    opt.add_argument(
        "--enable-conc-sweep",
        dest="enable_conc_sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a post-optimization concurrency sweep (baseline vs "
        "current_best across CONC) and write "
        "reports/conc_sweep_summary.json + conc_sweep_raw.csv. "
        "On by default; disable with --no-enable-conc-sweep.",
    )
    opt.add_argument(
        "--conc-sweep-concs",
        dest="conc_sweep_concs",
        type=str,
        default="256,128,64,32,16,8,4,2",
        help="Comma-separated CONC ladder for --enable-conc-sweep. Default 256,128,64,32,16,8,4,2 (high-to-low for single-server arm reuse).",
    )
    opt.add_argument(
        "--conc-sweep-timeout-sec",
        dest="conc_sweep_timeout_sec",
        type=int,
        default=1800,
        help="Per-variant timeout (seconds) for --enable-conc-sweep. "
        "Default 1800 (~30 min). Per-variant cap is also clamped "
        "by the remaining --conc-sweep-total-budget-sec.",
    )
    opt.add_argument(
        "--conc-sweep-total-budget-sec",
        dest="conc_sweep_total_budget_sec",
        type=int,
        default=9000,
        help="Total wall-clock budget (seconds) for the whole conc-sweep "
        "action, independent of the per-variant Magpie timeout. "
        "Once exhausted, remaining variants are recorded as "
        "status=skipped / error_class=budget_exhausted and the "
        "JSON envelope carries budget_exhausted=true. Default 9000 "
        "(~2.5h); set to 0 to disable. Also bounded above by the "
        "main session wall-clock deadline since conc_sweep runs as "
        "a SWEEP-phase action.",
    )
    # Retired flags: hard-fail at argparse with a migration hint, not a silent alias.
    _retired_hint = (
        "Use ``--enable-roofline`` (default on) / ``--no-enable-roofline`` "
        "instead. The PRELUDE-initial analysis task is unconditional and "
        "the composite/direct-profile bifurcation has been removed."
    )
    for _retired in (
        "--use-roofline-composite",
        "--no-use-roofline-composite",
        "--deny-direct-profile",
        "--no-deny-direct-profile",
        "--force-roofline-after-baseline",
        "--no-force-roofline-after-baseline",
    ):
        opt.add_argument(
            _retired,
            action=_RetiredFlag,
            hint=_retired_hint,
        )

    # Per-variant explore overtime kill ratio (mirrored to
    # SharedState.explore_overtime_kill_ratio). 0 disables.
    opt.add_argument(
        "--explore-overtime-kill-ratio",
        dest="explore_overtime_kill_ratio",
        type=float,
        default=2.0,
        help="Per-variant explore overtime kill: each single-variant "
        "Magpie run in the explore loop is reaped once its "
        "POST-READY (pure hot client) wall-clock exceeds "
        "``decision_anchor_sec * RATIO`` (the warm-decision anchor is "
        "``baseline_warm_runtime_sec``; pre-ready boot / weight load / "
        "first-request recompile is excluded — see "
        "INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY). The variant is "
        "recorded with outcome=KILLED_OVERTIME + runtime_sec + "
        "wall_clock_ratio_vs_baseline (no tput) so the LLM can "
        "distinguish it from a hard timeout / crash. Default 2.0 (kill "
        "at +100%% over the warm client anchor). Pass 0 to disable.",
    )
    # Explore variant hard timeout — operator override for the auto-derived cap.
    # 0 (default) keeps auto-derive; mirrored to
    # SharedState.explore_variant_timeout_sec_override.
    opt.add_argument(
        "--explore-variant-timeout-sec",
        dest="explore_variant_timeout_sec",
        type=int,
        default=0,
        help="Pin the per-variant hard timeout (seconds) inside the "
        "EXPLORE phase. ``0`` (default) auto-derives from "
        "``baseline_runtime_sec * (--explore-overtime-kill-ratio + "
        "--explore-variant-timeout-safety-margin)`` once baseline "
        "lands, with a 2400-14400 s range guard. Set to a positive "
        "integer to pin (CI smoke runs / debugging).",
    )
    opt.add_argument(
        "--explore-variant-timeout-safety-margin",
        dest="explore_variant_timeout_safety_margin",
        type=float,
        default=0.5,
        help="Headroom (as a fraction of baseline_runtime_sec) added on "
        "top of --explore-overtime-kill-ratio when the EXPLORE hard "
        "cap is auto-derived. Default 0.5 (≈ 50%% of baseline as "
        "buffer for variant cold starts: torch.compile AOTI compile, "
        "fresh aiter shapes, spec-decoding draft load). Bump for "
        "workloads with heavy compile cost; lower to tighten the "
        "backstop. No effect when --explore-variant-timeout-sec is "
        "set to a positive value.",
    )
    # Controls a resumed session's leftover legacy scoreboard (action_scores).
    opt.add_argument(
        "--legacy-action-scores",
        dest="legacy_action_scores",
        type=str,
        choices=("drop", "warn"),
        default="drop",
        help="Resume-mode handling of the legacy scoreboard "
        "(``action_scores`` and friends). 'drop' (default) "
        "silently discards. 'warn' logs a WARNING + adds a "
        "breakdown.warnings entry. KB_design §3.9 §7.",
    )
    # SharedState migration controls.
    opt.add_argument(
        "--migration-mode",
        dest="migration_mode",
        type=str,
        choices=("strict", "lenient"),
        default="strict",
        help="Strictness of the legacy → v0.8 state.json migration. "
        "'strict' (default) aborts on fact-layer field loss; "
        "'lenient' logs WARNING and continues. KB_design §3.10 §5.3.",
    )
    opt.add_argument(
        "--reset-state",
        dest="reset_state",
        action="store_true",
        default=False,
        help="Back up the existing ``state.json`` (if any) to "
        "``state.json.preReset.<unix_ts>`` and start the session "
        "from a blank SharedState. Cortex KB is NOT touched. "
        "KB_design §3.10 §5.3.",
    )
    # observability
    opt.add_argument(
        "--breakdown-include-transcripts",
        dest="breakdown_include_transcripts",
        type=str,
        choices=("true", "false"),
        default="false",
        help="Inline specialist transcript bodies into "
        "``specialist_runs`` (true) or reference them by path "
        "only (false, default). KB_design §3.12 §7.",
    )
    # plateau threshold tuning: override defaults; locked at session start.
    opt.add_argument(
        "--plateau-explore-keep-gain",
        dest="plateau_explore_keep_gain",
        type=float,
        default=None,
        help="EXPLORE plateau: max cumulative KEEP-gain (%%) across the "
        "lookback window below which the AND condition fires. "
        "Default 0.5.",
    )
    opt.add_argument(
        "--plateau-explore-empty-streak",
        dest="plateau_explore_empty_streak",
        type=int,
        default=None,
        help="EXPLORE plateau: required count of *consecutive* specialist "
        "rounds with empty proposal_set before the AND condition "
        "fires. Default 3.",
    )
    opt.add_argument(
        "--plateau-explore-lookback",
        dest="plateau_explore_lookback",
        type=int,
        default=None,
        help="EXPLORE plateau: number of trailing rounds the gain sum is computed over. Default 5.",
    )
    opt.add_argument(
        "--plateau-kernel-revert-streak",
        dest="plateau_kernel_revert_streak",
        type=int,
        default=None,
        help="KERNEL plateau: consecutive REVERT / NEEDS_REVIEW integrate "
        "attempts to count as plateau (one half of the OR). "
        "Default 3.",
    )
    opt.add_argument(
        "--plateau-kernel-keep-gain",
        dest="plateau_kernel_keep_gain",
        type=float,
        default=None,
        help="KERNEL plateau: max cumulative KEEP-gain (%%) across the "
        "lookback window below which the OR fires. Default 0.5.",
    )
    opt.add_argument(
        "--plateau-kernel-lookback",
        dest="plateau_kernel_lookback",
        type=int,
        default=None,
        help="KERNEL plateau: number of trailing integrate attempts the gain sum is computed over. Default 5.",
    )
    # IR-6 — EXPLORE hard force-exit thresholds (either condition fires; locked at start).
    opt.add_argument(
        "--explore-force-exit-hours-remaining",
        dest="explore_force_exit_hours_remaining",
        type=float,
        default=None,
        help="EXPLORE force-exit: total wall-clock remaining (hours) "
        "below which EXPLORE exits immediately to the next phase, "
        "regardless of plateau / steward. Default 3.0 (IR-6).",
    )
    opt.add_argument(
        "--explore-force-exit-budget-pct",
        dest="explore_force_exit_budget_pct",
        type=float,
        default=None,
        help="EXPLORE force-exit: phase-budget remaining fraction "
        "(0..1) below which EXPLORE exits immediately. Default "
        "0.20 (IR-6).",
    )
    # phase budget percentages: each phase claims a fraction of the wall-clock
    # budget (caps; may exit earlier). Both ``--max-minutes-*-pct`` and
    # ``--phase-budget-*-pct`` spellings are accepted.
    opt.add_argument(
        "--max-minutes-prelude-pct",
        "--phase-budget-prelude-pct",
        dest="phase_budget_prelude_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for PRELUDE as a fraction of --max-hours. Default: 0.03.",
    )
    opt.add_argument(
        "--max-minutes-framework-pct",
        "--phase-budget-framework-pct",
        dest="phase_budget_framework_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for FRAMEWORK_AGENT. Default: 0.15.",
    )
    opt.add_argument(
        "--max-minutes-explore-pct",
        "--phase-budget-explore-pct",
        dest="phase_budget_explore_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for EXPLORE. Default: 0.375.",
    )
    opt.add_argument(
        "--max-minutes-kernel-pct",
        "--phase-budget-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL_AGENT. Default: 0.305.",
    )
    opt.add_argument(
        "--max-minutes-sweep-pct",
        "--phase-budget-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.12.",
    )
    opt.add_argument(
        "--max-minutes-close-pct",
        "--phase-budget-close-pct",
        dest="phase_budget_close_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for CLOSE. Default: 0.02.",
    )
    opt.add_argument(
        "--strict-phase",
        dest="strict_phase",
        action="store_true",
        default=True,
        help="(v0.8 M2 default) Enforce PolicyGate R1 phase_incompatible. "
        "Action proposals outside the current phase's allowlist "
        "return policy_denied so the LLM self-corrects.",
    )
    opt.add_argument(
        "--no-strict-phase",
        dest="strict_phase",
        action="store_false",
        help="Disable R1 enforcement (warn-only). Useful for back-compat smoke tests; production should stay strict.",
    )

    rec = sub.add_parser(
        "recover-session",
        help="Rebuild + push the session_breakdown for a session that exited "
        "abnormally (crash / SIGKILL) so its breakdown lands on Langfuse.",
    )
    rec.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory of the crashed run (contains state.json / reports/trace/ and the recorder fragments).",
    )
    rec.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the session already looks complete (close_sequence_done / breakdown already recorded).",
    )
    rec.add_argument(
        "--backfill-trace",
        action="store_true",
        help="Also replay reports/trace/llm_calls.jsonl as Langfuse "
        "generations. Use ONLY when the live emitter never ran for this "
        "session (e.g. it was disabled during the run); otherwise it "
        "duplicates generations already pushed live.",
    )

    return p
