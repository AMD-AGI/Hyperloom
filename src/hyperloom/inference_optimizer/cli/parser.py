# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI argument parser — ``_build_parser`` and its purely-computational helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NoReturn

from .constraint_args import add_constraint_arguments
from .gpu_power_args import add_gpu_power_arguments
from .optimize_args import (
    add_cluster_arguments,
    add_conc_sweep_arguments,
    add_critic_arguments,
    add_framework_agent_arguments,
    add_framework_arguments,
    add_gpu_partition_arguments,
    add_kb_arguments,
    add_pd_role_arguments,
    add_pd_topology_arguments,
    add_phase_budget_arguments,
    add_plateau_arguments,
    add_recon_arguments,
    add_research_arguments,
    add_run_mode_arguments,
    add_service_arguments,
    add_target_arguments,
    add_workload_arguments,
)
from .recover import add_recover_subparser
from .session_target import add_session_dir_argument
from hyperloom_common.llm.llm_config import is_anthropic_only, is_openai_only, provider_model_defaults

# Workload knob fallbacks live in ``hyperloom.common`` so that the orchestrator
# can read the same numbers without importing this module, which would close a
# cycle. Only the ones this module quotes in help text are pulled in here.
from hyperloom.common.workload_defaults import (
    DEFAULT_CONC,
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_TP,
)
from hyperloom_common.llm.llm_config import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)
from hyperloom.orchestrator.scoring.proposal_scorer import DEFAULT_SCORER_MODELS

#: Fallback wall-clock budget. Named because ``--resume-from`` reads it back as
#: the "operator did not set this" signal: unlike the target flags, this one has
#: a real default, so the value alone cannot say whether it was asked for.
DEFAULT_MAX_HOURS = 2.0

# Substrings that mark a flag or a NAME=VALUE name as carrying a credential.
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
    """Report whether a flag or variable name suggests a credential."""
    lowered = name.lstrip("-").lower()
    return any(hint in lowered for hint in _SECRET_NAME_HINTS)


def _redact_unknown_args(tokens: list[str]) -> str:
    """Render unrecognised CLI tokens for logging with credential values masked."""

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
    """Parser whose unrecognised-argument error cannot print a credential."""

    _UNRECOGNIZED_PREFIX = "unrecognized arguments: "

    def error(self, message: str) -> NoReturn:
        """Exit 2 like argparse, with credential values masked."""
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
    """Resolve the default Claude model from env."""
    explicit = (os.environ.get("CLAUDE_MODEL") or "").strip()
    if explicit:
        return explicit
    gateway_model = provider_model_defaults().get("CLAUDE_MODEL", "")
    if gateway_model:
        return gateway_model
    if is_openai_only():
        return (os.environ.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
    return DEFAULT_CLAUDE_MODEL


def _default_codex_model_env() -> str:
    """Resolve the default Codex-style model from env."""
    if is_anthropic_only():
        return (os.environ.get("CLAUDE_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL
    explicit = (os.environ.get("CODEX_MODEL") or "").strip()
    if explicit:
        return explicit
    gateway_model = provider_model_defaults().get("CODEX_MODEL", "")
    if gateway_model:
        return gateway_model
    return DEFAULT_CODEX_MODEL


def _default_research_lane_capacity() -> int:
    """Default ``--research-lane-capacity`` to the GPU ceiling (2×GPU)."""
    from hyperloom.orchestrator.policy.gate import research_lane_ceiling

    return research_lane_ceiling()


def _default_gpu_specialist_capacity() -> int:
    """Default ``--gpu-specialist-capacity`` to the whole visible machine."""
    from hyperloom.common.visible_devices import detect_gpu_count

    return detect_gpu_count()


def _add_custom_specialist_args(opt: argparse.ArgumentParser) -> None:
    """The ``--custom-specialist-*`` flags of ``optimize``."""
    opt.add_argument(
        "--custom-specialist-prompt-file",
        dest="custom_specialist_prompt_file",
        type=str,
        default=None,
        help="Register an operator-defined specialist type (custom_specialist) whose domain focus is this file's "
        "text (max 16 KiB). Requires --custom-specialist-description. The Coordinator alone dispatches it, once "
        "per macro cycle in FRAMEWORK_AGENT, ahead of other queued work, and benches its proposals first; "
        "Orchestration cannot. Persisted for --resume-from; passing it on a "
        "resume replaces the stored definition. The text is written in clear to state.json and every task's "
        "prompt.md: do not put credentials in it. HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1 runs FRAMEWORK_AGENT on it alone.",
    )
    opt.add_argument(
        "--custom-specialist-description",
        dest="custom_specialist_description",
        type=str,
        default=None,
        help="One line (max 200 chars) describing custom_specialist in its own prompt. Requires "
        "--custom-specialist-prompt-file.",
    )


def _add_resume_arguments(opt: argparse.ArgumentParser) -> None:
    """The ``--resume-from`` and ``--force-resume`` flags of ``optimize``."""
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


def _add_plug_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the flags that select which implementation fills each plug socket; absent, every socket stays local."""
    opt.add_argument(
        "--plug-profile",
        dest="plug_profile",
        default=None,
        metavar="NAME",
        help="Built-in plug profile to load (local, hosted-gpu, hosted-cpu). Without it, and without the other plug "
        "flags, nothing changes from a plain local run.",
    )
    opt.add_argument(
        "--manifest",
        dest="plug_manifest",
        default=None,
        metavar="PATH",
        help="Custom plug manifest (JSON) laid over the profile. Falls back to $HYPERLOOM_MANIFEST.",
    )
    for flag, dest, metavar, text in (
        (
            "--plug",
            "plug",
            "SOCKET=IMPL",
            "Choose the implementation of one socket; repeatable. The last layer, over profile, manifest and "
            "$HYPERLOOM_MANIFEST_OVERRIDES.",
        ),
        (
            "--plug-config",
            "plug_config",
            "SOCKET.KEY=VALUE",
            "Set one configuration key of a socket's implementation; repeatable. VALUE is read as JSON when it "
            "parses as JSON, as text otherwise.",
        ),
    ):
        opt.add_argument(flag, dest=dest, action="append", default=[], metavar=metavar, help=text)


def _add_model_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the model path and quantization-prelude flags of ``optimize``."""
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
        help="Optional natural-language quantization request. When set, "
        "`hyperloom quantize` runs ONCE as a prelude before the "
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


def _add_parallelism_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the parallelism, concurrency and server-args flags of ``optimize``."""
    opt.add_argument(
        "--tp",
        type=_positive_int_arg,
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
        "--max-model-len > $MAX_MODEL_LEN > auto(ISL+OSL+headroom, "
        "clamped to native context). Explicit values must fit the known native context and are "
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
        "expert parallelism: sglang adds `--ep-size N`, vllm and atom "
        "add `--enable-expert-parallel` (EP then spans all TP ranks). "
        "Typical: EP=TP. Default: 1. On a single node an EP that does "
        "not divide TP is dropped with a warning; on multi-node EP > TP "
        "is rejected at server-restart time.",
    )


def _add_budget_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the variant-skip list, wall-clock budget and sequence-length flags of ``optimize``."""
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
    opt.add_argument(
        "--max-hours",
        type=float,
        # No argparse default: ``--resume-from`` must tell "the operator asked
        # for this many hours" from "the operator said nothing", and a default
        # would make an explicit value indistinguishable from absence.
        default=None,
        help=f"Wall-clock budget in hours (default {DEFAULT_MAX_HOURS})",
    )
    opt.add_argument(
        "--extend-hours",
        dest="extend_hours",
        type=float,
        default=0.0,
        help="On --resume-from, grant this many additional hours to the "
        "session budget. Elapsed time is summed forward across every leg and "
        "never reset, so this is the only way to lengthen a run that has "
        "already spent its budget. The grant is recorded in the session state "
        "with its reason.",
    )
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


def _add_model_context_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the model-class, reference-GPU, tick and LLM model flags of ``optimize``."""
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


def _add_capacity_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the research-lane and GPU-specialist capacity flags of ``optimize``."""
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


def _add_proposal_scorer_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the advisory proposal-scorer flags of ``optimize``."""
    # Advisory specialist-proposal scorer (ProposalScorer): scores each proposal_set with gateway models as a
    # reference for Orchestration; never gates.
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
        "enable scoring. Default '%(default)s'. "
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


def _add_specialist_arguments(opt: argparse.ArgumentParser) -> None:
    """Add the specialist model, turn-cap and MCP-config flags of ``optimize``."""
    from hyperloom.orchestrator.specialists.domains import (
        DEFAULT_SPECIALIST_MAX_TURNS as _DEFAULT_SPECIALIST_MAX_TURNS,
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
        "--specialist-mcp-config",
        dest="specialist_mcp_config",
        type=str,
        default=None,
        help="Optional MCP config JSON for specialist subprocesses. Claude "
        "receives it via --mcp-config; Codex translates supported HTTP/stdio "
        "servers into private task-local config. Default: None.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser and subcommands."""
    p = RedactingArgumentParser(
        prog="hyperloom",
        description="Inference Optimizer — multi-agent inference optimization (SGLang/vLLM/Atom/xDiT)",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", "-v", action="count", default=0, help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)
    opt = sub.add_parser("optimize", parents=[common], help="Drive a multi-agent optimization run on a model")
    _add_model_arguments(opt)
    add_gpu_partition_arguments(opt)
    add_framework_arguments(opt)
    add_cluster_arguments(opt)
    _add_parallelism_arguments(opt)
    add_pd_topology_arguments(opt)
    add_pd_role_arguments(opt)
    _add_budget_arguments(opt)
    add_workload_arguments(opt)
    add_target_arguments(opt)
    add_constraint_arguments(opt)
    add_gpu_power_arguments(opt)
    _add_resume_arguments(opt)
    _add_plug_arguments(opt)
    _add_model_context_arguments(opt)
    add_run_mode_arguments(opt)
    add_framework_agent_arguments(opt)
    add_critic_arguments(opt)
    _add_custom_specialist_args(opt)
    add_kb_arguments(opt)
    add_service_arguments(opt)
    _add_capacity_arguments(opt)
    _add_proposal_scorer_arguments(opt)
    _add_specialist_arguments(opt)
    add_research_arguments(opt)
    add_recon_arguments(opt)
    add_conc_sweep_arguments(opt)
    add_plateau_arguments(opt)
    add_phase_budget_arguments(opt)

    add_session_dir_argument(opt)
    add_recover_subparser(sub, common)

    return p
