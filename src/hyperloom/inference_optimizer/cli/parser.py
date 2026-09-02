# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI argument parser — ``_build_parser`` and its purely-computational helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NoReturn

from .. import framework_registry
from .backends import CRITIC_PROTOCOL_CHOICES
from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES
from hyperloom.common.llm_config import provider_model_defaults
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


# Substrings that mark a flag or a NAME=VALUE name as carrying a credential.
# Deliberately broad: over-redacting a pod env var costs nothing, while missing
# one writes a live token into a log the platform ships elsewhere.
_SECRET_NAME_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "key",
    "auth",
)
_REDACTED = "***"


def _is_secret_name(name: str) -> bool:
    """Report whether a flag or variable name suggests a credential.

    Args:
        name: A flag (leading dashes tolerated) or a ``NAME=VALUE`` name.

    Returns:
        bool: ``True`` when the name matches any known credential hint.
    """
    lowered = name.lstrip("-").lower()
    return any(hint in lowered for hint in _SECRET_NAME_HINTS)


def _redact_unknown_args(tokens: list[str]) -> str:
    """Render unrecognised CLI tokens for logging with credential values masked.

    Names are always kept and only values are dropped, so a misspelled real flag
    stays as diagnosable as before. A value is masked when its own flag looks
    sensitive (``--api-key foo``) or when it is a ``NAME=VALUE`` pair with a
    sensitive name (``--extra-env HF_TOKEN=foo``), which is how credentials
    normally reach the pods.

    Args:
        tokens: The leftover argv entries argparse could not place.

    Returns:
        str: A space-joined, log-safe rendering of ``tokens``.
    """

    def _mask(value: str, flag_is_secret: bool) -> str:
        name, sep, _ = value.partition("=")
        if sep and _is_secret_name(name):
            return f"{name}={_REDACTED}"
        return _REDACTED if flag_is_secret else value

    rendered: list[str] = []
    # A bare value belongs to the flag before it, so its sensitivity carries over.
    flag_is_secret = False
    for token in tokens:
        if token.startswith("-"):
            flag, sep, inline = token.partition("=")
            flag_is_secret = _is_secret_name(flag)
            if sep:
                rendered.append(f"{flag}={_mask(inline, flag_is_secret)}")
                flag_is_secret = False
            else:
                rendered.append(flag)
        else:
            rendered.append(_mask(token, flag_is_secret))
            flag_is_secret = False
    return " ".join(rendered)


class RedactingArgumentParser(argparse.ArgumentParser):
    """Parser whose unrecognised-argument error cannot print a credential.

    argparse renders the offending tokens verbatim. The platform hands its whole
    prompt FLAGS block to this CLI, pod credentials included, so an argument
    this parser does not know -- a flag the platform added before Hyperloom
    declared it, or a plain typo -- would otherwise write a live token into a
    log shipped elsewhere. Only the values are masked; every name survives, so
    the message still says exactly which argument was rejected.
    """

    _UNRECOGNIZED_PREFIX = "unrecognized arguments: "

    def error(self, message: str) -> NoReturn:
        """Exit 2 like argparse, with credential values masked.

        Args:
            message: The argparse-generated error message.
        """
        if message.startswith(self._UNRECOGNIZED_PREFIX):
            tail = message[len(self._UNRECOGNIZED_PREFIX) :]
            message = self._UNRECOGNIZED_PREFIX + _redact_unknown_args(tail.split())
        super().error(message)


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
    """Resolve the default Claude model from env.

    Runs before ``_preflight`` normalizes the environment, so it consults
    :func:`provider_model_defaults` itself: a gateway that only serves its own
    models must not be handed the AMD Claude default.
    """
    explicit = (os.environ.get("CLAUDE_MODEL") or "").strip()
    if explicit:
        return explicit
    if os.environ.get("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX") == "1":
        return (os.environ.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
    gateway_model = provider_model_defaults().get("CLAUDE_MODEL", "")
    if gateway_model:
        return gateway_model
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
    if anthropic_url and not openai_url:
        return (os.environ.get("CLAUDE_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL
    explicit = (os.environ.get("CODEX_MODEL") or "").strip()
    if explicit:
        return explicit
    gateway_model = provider_model_defaults().get("CODEX_MODEL", "")
    if gateway_model:
        return gateway_model
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

    p = RedactingArgumentParser(
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
        "--resume-from is set — model is read from manifest.json/"
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
        "--resume-from.",
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
        "--quantize-provider",
        choices=("claude", "codex", "hermes"),
        default=None,
        help="Agent runtime for the original Quark skill prelude. Resolution: "
        "--quantize-provider > HYPERLOOM_QUANT_PROVIDER > claude.",
    )
    opt.add_argument(
        "--gpu-type",
        type=str.lower,
        # Sorted for a stable --help listing, and derived so a board added to
        # the identities table is accepted here without a second edit.
        choices=sorted(AMD_GPU_DISPATCH_IDENTITIES),
        default=None,
        help="Hint for the real target GPU. The rocm-smi probe always "
        "wins when both are present and disagree; a WARN is "
        "emitted to stderr so the operator sees the typo. Used "
        "verbatim only when the probe fails (CPU sandbox / no "
        "rocm-smi). Magpie runner_type is derived separately; "
        "mi308x and mi325x currently run with mi300x runner scripts because "
        "Magpie does not yet ship MI308X/MI325X-specific SGLang/vLLM scripts. "
        "Radeon 8060S uses the dedicated radeon8060s/gfx1151 runner.",
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
        # None, not 2, so a resume can tell "not passed" from "passed 2" and
        # let the persisted value stand. The 2 is applied where it is resolved.
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
    # Platform-owned, inert here. Primus-Claw parses ONE prompt FLAGS block for
    # both itself and this CLI, and these configure the cluster it provisions
    # (pod image, per-pod cpu/mem, pod env) before the optimizer ever starts.
    # Nothing reads them; they are declared so strict parsing accepts a real
    # platform prompt.
    #
    # Declared individually rather than tolerating unknown arguments wholesale,
    # so a misspelled Hyperloom flag still fails fast instead of running for
    # hours on a default. No similarity heuristic could stand in for this list:
    # ``--cpus-per-node`` and ``--gpus-per-node`` are one character apart.
    opt.add_argument("--mn-image", default=None, help=argparse.SUPPRESS)
    opt.add_argument("--cpus-per-node", type=int, default=None, help=argparse.SUPPRESS)
    opt.add_argument("--mem-per-node", type=int, default=None, help=argparse.SUPPRESS)
    opt.add_argument("--extra-env", action="append", default=[], help=argparse.SUPPRESS)
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
        'Example: --server-args "--kv-cache-dtype fp8_e4m3 '
        '--gpu-memory-utilization 0.85".',
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
        "state.json under `explore_search.last_round.skipped_dup`, and in the "
        "action's per-variant outcomes, tagged `user_skip`.",
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
            "deserve separate KB rows). When omitted, auto-detected via "
            "importing the framework's top-level package and reading "
            "``__version__`` (sglang/vllm/atom supported); auto-detect "
            "failure degrades to 'unknown_version'. Override with "
            "--framework-version=0.4.5 to pin a specific tag for the run."
        ),
    )
    # ``None`` rather than True: the deprecated alias below has to be able to
    # tell "not passed" from an explicit value, or it could never be honoured
    # without also overriding this flag. bootstrap applies the real default.
    opt.add_argument(
        "--auto-kernel-opt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "At KERNEL entry, dispatch the source-level kernel_opt batch "
            "without waiting for an orchestration request — every untried "
            "candidate above the dispatch floor goes in one batch. Runs on "
            "both entry routes (with and without FP8 GEMM tuning), since "
            "tuning GEMM shape tables and rewriting kernel source are "
            "unrelated. --no-auto-kernel-opt leaves source-level kernel_opt "
            "to explicit orchestration requests; it does not disable the "
            "forge-fusion or collective lanes, which have their own gates. "
            "Default: enabled."
        ),
    )
    # Deprecated spelling of --auto-kernel-opt, still honoured so a launcher
    # template that passes it keeps its opt-out instead of silently getting the
    # default back. ``None`` default distinguishes "not passed" from an explicit
    # value; bootstrap resolves the precedence and warns. Unlike the retired
    # no-ops below, this one still feeds the same dest.
    opt.add_argument(
        "--continue-kernel-after-gemm",
        dest="continue_kernel_after_gemm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
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
        help="Stop when current best reaches N (serving: tok/s/GPU; xDiT: img/s)",
    )
    grp.add_argument(
        "--target-baseline-dir", type=str, default=None, help="Stop when current best matches the baseline in DIR"
    )
    opt.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Per-session subdir to resume; the only way to resume a session. "
        "Skips the SharedState seed and lets the Coordinator replay the "
        "prior event log + state.json. Must be an absolute path under "
        "$USER_DATA_PATH (workspace_root), which MUST stay at workspace "
        "level (/shared/hyperloom-sessions, not the per-session subdir) so "
        "runtime/ resolution works.",
    )
    opt.add_argument(
        "--force-resume",
        action="store_true",
        default=False,
        help=(
            "Allow ``--resume-from`` to push past a terminal "
            "``stop_reason='target_reached'``. "
            "Without this flag the resume aborts (Issue-G guard, per "
            "SKILL.md 'Run-time signals': that terminal requires an "
            "operator-side workload / strategy change before resuming). "
            "No-op without ``--resume-from``."
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
            "the SKILL launcher writes <session_dir>/model_arch.json, which "
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
            "(e.g. b300 / mi355x / h200). target_analysis ALWAYS runs first, "
            "before baseline, and always writes "
            "$SESSION_DIR/target_analysis/target_baseline.json + a short "
            "MD report. When this flag is set, the JSON carries the "
            "matching InferenceX (https://inferencex.semianalysis.com) "
            "reference data point; when unset, the JSON carries a "
            "structured reason='no_target_gpu_configured' marker so the "
            "report still has a deterministic 'External baseline' "
            "section. The reference is API-measured and never influences "
            "the Objective, scoring, or any KEEP/REVERT gate; a matching "
            "row is surfaced to the gap advisory as direction only. "
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
    # Retired with the kernel LLM role; accepted as no-ops so a launcher or
    # operator template that still passes them does not exit 2. Nothing reads
    # the dests. ``--kernel-prompt`` took a path, so it has to keep consuming
    # one: as a store_true its value would land as a stray positional and
    # argparse would exit 2 anyway, which is the failure this exists to avoid.
    for _retired in ("--kernel-codex", "--kernel-claude"):
        opt.add_argument(_retired, action="store_true", default=False, help=argparse.SUPPRESS)
    opt.add_argument("--kernel-prompt", type=str, default=None, help=argparse.SUPPRESS)
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
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip the accuracy eval everywhere. The baseline anchors on "
        "throughput alone instead of halting on a missing accuracy "
        "reference, and every candidate is graded on throughput only. "
        "Useful for CI/CD tuning runs that care about performance and not "
        "accuracy; the run is not accuracy-validated. Default: eval enabled.",
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
        "'auto' (default) derives it from the configured credentials; an "
        "explicit value fails at startup when that side has no credential. "
        "Ignored (with a warning) under --critic-mock, which runs no review "
        "inference.",
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
        "--orch-prompt", type=str, default=None, help="Override Orchestration system prompt (file path or inline)"
    )
    opt.add_argument("--critic-prompt", type=str, default=None, help="Override Critic system prompt")
    opt.add_argument(
        "--local-kb-root",
        dest="local_kb_root",
        type=str,
        default=None,
        help="Filesystem root for the local recipe-snapshot KB store. "
        "All writes (put_recipe / append_attempt / delete_recipe) go "
        "here. Defaults to "
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
        help="Skip the recipe KB integration entirely (T0/T2/T3/T4 become "
        "no-ops). Also short-circuits any legacy IR-3 KB probe marker. "
        "Manifest records the reason as ``explicit_flag`` when set "
        "explicitly.",
    )
    opt.add_argument(
        "--recipe-kb-strict-fingerprint",
        dest="recipe_kb_strict_fingerprint",
        action="store_true",
        default=False,
        help="When set, T0 refuses warm_start_recipe rows whose "
        "stack_fingerprint does not match the current pod (recorded "
        "in manifest.json). Default: lenient (M1 records the flag "
        "in manifest only; consumed by M5 specialist assembly).",
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
        default=0.7,
        help="Minimum ``warm_start_recipe.confidence`` required to "
        "trigger the auto-replay. Default 0.7 means an ``exact`` "
        "seven-tuple hit (conf 1.0) and a server-returned ``relative`` "
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
        "``status=drift`` and continue with the regular optimisation "
        "flow without inheriting the warm config.",
    )
    # PR Monitor REST + MCP are co-hosted by KB Store and derived from
    # $KB_STORE_URL. There are deliberately no independent endpoint flags.
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
        "enable scoring. Default 'claude-opus-5,gpt-5.6-sol,"
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
    # Specialist model override; provider shape selects Claude or Codex.
    opt.add_argument(
        "--specialist-model",
        dest="specialist_model",
        type=str,
        default=None,
        help="Generic model override for the selected specialist backend. "
        "When omitted, Codex specialists use --codex-model and Claude "
        "specialists use --claude-model. KB_design §3.5 §6.",
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
        "a fresh selected-provider agent CLI per task. 'inprocess' uses "
        "the matching Claude or Codex Agent SDK backend in the orchestrator "
        "process.",
    )
    opt.add_argument(
        "--specialist-mcp-config",
        dest="specialist_mcp_config",
        type=str,
        default=None,
        help="Optional MCP config JSON for specialist subprocesses. Claude "
        "receives it via --mcp-config; Codex translates supported HTTP/stdio "
        "servers into private task-local config. Default: None.",
    )

    # Integration toggles. Roofline refresh is unconditional (fires at PRELUDE
    # and every 10% cumulative_gain_validated crossing).
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
        default=None,
        help="Comma-separated CONC ladder for --enable-conc-sweep. Ordered "
        "high-to-low internally for single-server arm reuse, so the order given "
        "does not matter. Defaults to the ladder for the workload: "
        "256,128,64,32,16,8,4,2 synthetic, 1,4,8,10,14,20,28 under "
        "HYPERLOOM_AGENTX (an agentic request carries orders of magnitude more "
        "prompt, so the same card saturates far lower).",
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
        "optimisation phase. ``0`` (default) auto-derives from "
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
        "top of --explore-overtime-kill-ratio when the explore hard "
        "cap is auto-derived. Default 0.5 (≈ 50%% of baseline as "
        "buffer for variant cold starts: torch.compile AOTI compile, "
        "fresh aiter shapes, spec-decoding draft load). Bump for "
        "workloads with heavy compile cost; lower to tighten the "
        "backstop. No effect when --explore-variant-timeout-sec is "
        "set to a positive value.",
    )
    opt.add_argument(
        "--reset-state",
        dest="reset_state",
        action="store_true",
        default=False,
        help="Back up the existing ``state.json`` (if any) to "
        "``state.json.preReset.<unix_ts>`` and start the session "
        "from a blank SharedState. Recipe KB is NOT touched.",
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
        # The EXPLORE spellings land on the same option: configuration search
        # and source landing are two arms of one phase with one budget, so a
        # separate share for either would be a number nothing reads.
        "--max-minutes-explore-pct",
        "--phase-budget-explore-pct",
        dest="phase_budget_framework_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for the OPTIMIZE (FRAMEWORK_AGENT) phase. Default: 0.40.",
    )
    opt.add_argument(
        "--max-minutes-kernel-pct",
        "--phase-budget-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL_AGENT. Default: 0.50.",
    )
    opt.add_argument(
        "--max-minutes-sweep-pct",
        "--phase-budget-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.05.",
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
        help="Enforce PolicyGate R1 phase_incompatible. "
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
