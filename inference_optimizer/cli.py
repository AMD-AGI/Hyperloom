"""CLI entry

Usage::

    inference_optimizer optimize \\
        --model /wekafs/models/<your-model> \\
        --target-gain 10 \\
        --max-hours 2

    # or via env (matches the rest of the pipeline / Dockerfile convention):
    export MODEL_PATH=/wekafs/models/<your-model>
    inference_optimizer optimize --target-gain 10 --max-hours 2

Single subcommand for now (``optimize``). Wires Claude+Codex backends,
registers all available action_executors, builds the requested objective,
and starts ``Coordinator.run()`` until target / time / SIGTERM.

Env vars consumed (besides the standard backend creds):

  MODEL_PATH                                   — required if --model not passed;
                                                 also exported back to subprocess
                                                 env so Magpie YAMLs get the
                                                 correct model path injected
                                                 instead of the YAML's hardcoded
                                                 fallback.
  OPENAI_BASE_URL + SAFE_API_KEY — canonical LiteLLM endpoint; compatibility aliases are exported for Claude/OOB/GEAK
  ROCR_VISIBLE_DEVICES                         — pin the GPU
  CLAUDE_MODEL                                 — default claude-opus-4-7
  CODEX_MODEL                                  — default gpt-5.4
  USER_DATA_PATH                               — override session dir
                                                 (default: /workspace/hyperloom).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
)
from .orchestrator.action_executors import (
    TargetAnalysisExecutor,
    baseline_executor,
    explore_executor,
    recover_executor,
    report_executor,
    session_breakdown_executor,
    sweep_executor,
)
from .orchestrator.action_executors.integrate_patch import IntegratePatchExecutor
from .orchestrator.action_executors.framework_pr import FrameworkPrExecutor
from .orchestrator.action_executors.recover import recover_executor
from .orchestrator.action_executors.profile import profile_executor
from .orchestrator.action_executors.roofline import make_roofline_executor
from .orchestrator.backends import (
    ClaudeBackend,
    CodexBackend,
    CriticAgentBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    RobustnessAgentBackend,
)
from .manifest import load_manifest, write_manifest
from .orchestrator.action_registry import ActionRegistry
from .orchestrator.coordinator import Coordinator
from .orchestrator.cortex_t0 import run_t0_anchor
from .orchestrator.framework_paths import resolve_source_file_allowlist
from .orchestrator.objective import Objective, build_objective
from .orchestrator.shared_state import SharedState
from .orchestrator.system_prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from .paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    _SESSION_SKELETON,
    asset_system_prompts_dir,
    make_session_dir,
    mn_profile_trace_root,
    session_dir as _session_dir_resolve,
)
from .session_paths import (
    agent_prompt_snapshot,
)


log = logging.getLogger("inference_optimizer.cli")


class _RetiredFlag(argparse.Action):
    """Argparse action that hard-fails on a retired CLI flag.

    Reaching for the old flag spelling should produce an immediate,
    explicit error (``parser.error`` exits 2) with a one-line
    migration hint — silent aliases would mask the behaviour change
    when the underlying semantics differ.
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        hint: str,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs.setdefault("help", argparse.SUPPRESS)
        self._hint = hint
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        parser.error(f"{option_string} was removed. {self._hint}")


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only fragment consumed by ``prompt_builder``.

    Phase 1 collapsed ``orchestration.md`` to a small "rules + output protocol"
    fragment; the full system prompt is composed at runtime from
    :class:`ActionMetadata` and run-level parameters by
    :func:`build_orchestration_prompt`. The legacy
    ``orchestration.no_kernel.md`` was deleted — kernel-vs-no-kernel is now
    a builder parameter, not a separate file.
    """
    return asset_system_prompts_dir() / "orchestration.md"


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Return ``(kind, value)`` strings consumed by the prompt builder."""
    kind = objective.kind()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


def _build_orchestration_prompt(
    *,
    no_kernel: bool,
    framework: str,
    objective: Objective,
    max_minutes: int,
    action_registry: ActionRegistry | None = None,
) -> str:
    """Compose the Orchestration system prompt for this run.

    The legacy ``_load_orchestration_prompt(no_kernel)`` returned a
    hand-maintained markdown file; this replacement assembles the prompt
    from typed inputs so kernel-enabled / no-kernel split is just a
    parameter and every enabled action carries a 1-line description.

    Callers may still pass ``--orch-prompt`` to fully override the result.
    """
    registry = action_registry or ActionRegistry().load()
    enabled = default_enabled_actions(no_kernel=no_kernel)
    kind, value = _objective_summary_for_prompt(objective)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework=framework,
        kernel_enabled=not no_kernel,
        objective_kind=kind,
        objective_value=value,
        max_minutes=int(max_minutes),
        rules_fragment_path=_orchestration_rules_fragment_path(),
        framework_source_roots=resolve_source_file_allowlist(),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``system_prompts/critic.md``."""
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


_DEFAULT_KERNEL_PROMPT = (
    "You are the Kernel agent — responder-only. You receive `request`\n"
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request, emit ONE `response` intent in reply.\n"
    "Schema:\n"
    "  intent_type: response\n"
    "  payload: {\n"
    "    in_reply_to: <request msg_id>,\n"
    "    kind:        '<request.kind>_done',\n"
    "    status:      'ok' | 'failed' | 'needs_review',\n"
    "    result:      { /* whatever the request asked for */ }\n"
    "  }\n\n"
    "Native-only rule: run_optimization must refuse runtime-generated\n"
    "torch.compile/Inductor/Triton cache kernels. Only reusable framework\n"
    "sources under stable repos (aiter/sglang/vllm source trees) are valid\n"
    "kernel-opt targets; otherwise return status='failed' with a clear reason.\n\n"
    "SESSION_DIR contract: every path you emit in result.* must be either\n"
    "verbatim from the request payload, prefixed by SESSION_DIR (injected\n"
    "per tick), or under one of `/sgl-workspace/aiter/`, `/sgl-workspace/\n"
    "sglang/`, `/sgl-workspace/vllm/` (the framework source allowlists).\n"
    "PolicyGate rejects responses whose path fields escape this set.\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)


_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140. Listed here so
    # we can log the resolved value at session start instead of waiting for
    # Magpie subprocess output deep in the run.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    normalized = str(gpu_type or "").strip().lower()
    if normalized == "mi325x":
        return "mi300x"
    return normalized


# ---------------------------------------------------------------------------
# Hard model allowlist. Orchestration MUST resolve to one of these two ids
# before Coordinator boots; anything else is a configuration error and the
# CLI refuses to start the service. Per operator direction (2026-05-09):
# only Opus 4-7 (preferred) or Opus 4-6 (fallback when 4-7 is missing from
# the gateway catalog) are acceptable for the long-running orchestration
# loop. Other catalog entries (haiku / opus 4-5) drift the agent's behaviour
# enough that prior runs degraded measurably.
_CLAUDE_PREFERRED_MODEL = "claude-opus-4-7"
_CLAUDE_FALLBACK_MODEL  = "claude-opus-4-6"
_CLAUDE_ALLOWED_MODELS  = (_CLAUDE_PREFERRED_MODEL, _CLAUDE_FALLBACK_MODEL)

# Catalog probe retry contract (gateway is documented-flaky; the launcher
# we replaced had to manually curl this endpoint a couple of times before it
# returned 200). Sleep N seconds before attempt i+1; len(_CATALOG_RETRY_DELAYS_SEC)
# is the retry count after the initial attempt.
_CATALOG_RETRY_DELAYS_SEC = (1.0, 3.0, 5.0)
_CATALOG_REQUEST_TIMEOUT_SEC = 5.0

# /dev/shm threshold: vLLM IPC + NCCL shm segments routinely need >8GB; when
# free space drops below this the next launch tends to collide with stale
# segments and hang for 5 minutes inside zmq.
_DEV_SHM_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB


# Critic-agent skill root resolution. Env wins; otherwise we look at the
# sibling ``critic-agent/`` directory next to this package's repo root.
# The runtime is invoked with ``cwd=<root>`` (mirrors critic-agent's own
# pytest.ini ``pythonpath = .``) so ``python -m runtime.cli`` resolves
# without needing a pip-install of critic-agent.
_CRITIC_AGENT_ROOT_ENV = "CRITIC_AGENT_ROOT"


def _resolve_critic_agent_root() -> Path | None:
    """Return the critic-agent skill root, or ``None`` if not found.

    Order:
    1. ``$CRITIC_AGENT_ROOT`` env var (operator override).
    2. Sibling ``critic-agent/`` next to the inference_optimizer package
       (i.e. ``$REPO_ROOT/critic-agent``).
    """
    override = os.environ.get(_CRITIC_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    # PACKAGE_ROOT/.. == repo root (since the package lives at $REPO/inference_optimizer/).
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "critic-agent"
    return candidate if (candidate / "runtime" / "cli.py").is_file() else None


def _validate_critic_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m runtime.cli --help`` doesn't work.

    Raises :class:`SystemExit` with a clear message when the runtime cannot
    start; printed in the operator's voice so they know which env to fix.
    """
    cmd = [sys.executable, "-m", "runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: critic-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix CRITIC_AGENT_ROOT, install critic-agent at "
            f"$REPO_ROOT/critic-agent, or pass --critic-mock / "
            f"--critic-codex-bare to bypass critic-agent.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: critic-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Robustness-agent runtime location resolution. Mirrors the critic-agent
# helpers above: hosts pick a backend through --robustness-mock /
# --robustness-agent (env override INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND),
# and when "agent" is selected we shell out to
# ``python -m robustness_agent.runtime.cli`` with cwd=<root> and
# PYTHONPATH=<root>/src.
_ROBUSTNESS_AGENT_ROOT_ENV = "ROBUSTNESS_AGENT_ROOT"


def _resolve_robustness_agent_root() -> Path | None:
    """Return the robustness-agent skill root, or ``None`` if not found.

    Order:
    1. ``$ROBUSTNESS_AGENT_ROOT`` env var (operator override).
    2. Sibling ``robustness-agent/`` next to the inference_optimizer package
       (i.e. ``$REPO_ROOT/robustness-agent``).
    """
    override = os.environ.get(_ROBUSTNESS_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "src" / "robustness_agent" / "runtime" / "cli.py").is_file() else None
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "robustness-agent"
    cli_module = candidate / "src" / "robustness_agent" / "runtime" / "cli.py"
    return candidate if cli_module.is_file() else None


def _validate_robustness_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m robustness_agent.runtime.cli --help`` doesn't work."""
    src = str(root / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    cmd = [sys.executable, "-m", "robustness_agent.runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: robustness-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix ROBUSTNESS_AGENT_ROOT, install robustness-agent at "
            f"$REPO_ROOT/robustness-agent, or pass --robustness-mock to bypass.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: robustness-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """B3: tighten incompatible CLI knobs when --framework atom is selected.

    atom in Magpie v1 has neither a torch_profiler integration (so
    roofline cannot produce a trace) nor a source-patcher for the
    framework-agent's PR loop (the kernel-agent / framework-agent assume
    sglang/vllm source layouts). Auto-disabling kernel-agent +
    framework-agent + roofline phases keeps the rest of the run sensible
    without forcing the operator to remember three extra flags. Explicit
    user opt-in for any of these is preserved (we only flip a value when
    it is still at its enabled default).

    atom multi-node is also unsupported (Magpie wrapper / atom server
    have no multi-node TP wiring) — fail-fast on ``--nodes >= 2`` so
    operators don't burn a ~6-min cold start on a doomed run.

    Returns the list of flag names auto-disabled (for callers that want
    to log / assert). Calls ``sys.exit(2)`` on the multi-node guard
    failure.
    """
    auto_disabled: list[str] = []
    if not getattr(args, "no_kernel", False):
        args.no_kernel = True
        auto_disabled.append("--no-kernel")
    if not getattr(args, "no_framework", False):
        args.no_framework = True
        auto_disabled.append("--no-framework")
    if getattr(args, "enable_roofline", True):
        args.enable_roofline = False
        auto_disabled.append("--no-enable-roofline")
    if auto_disabled:
        print(
            f"  framework=atom: auto-disabling "
            f"{', '.join(auto_disabled)} (atom has no profiler / "
            "sglang/vllm-specific source patcher; see "
            "atom_boost_tutorials.md §6)"
        )
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    return auto_disabled


def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi325x|mi355x or None if undetectable.

    Tries `rocm-smi --showproductname` first (most reliable), then falls
    back to torch.cuda.get_device_properties(0).gcnArchName parsing. Both
    are best-effort — on CPU-only or non-ROCm boxes we silently return
    None so the caller can defer to Magpie's own detection layer.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None


async def _noop_prep(ctx) -> dict:
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


def _build_backends(
    *,
    claude_model: str,
    codex_model: str,
    kernel_codex: bool,
    critic_choice: str,
    session_dir: Path,
    critic_agent_root: Path | None = None,
    critic_kb_mode: str = "inmemory",
    robustness_choice: str = "mock",
    robustness_agent_root: Path | None = None,
    robustness_options: dict[str, Any] | None = None,
    no_kernel: bool = False,
) -> dict[str, Any]:
    """Construct all per-role backends.

    ``critic_choice`` ∈ {``"mock"``, ``"agent"``, ``"codex_bare"``}:

    * ``mock`` — always-approve adapter (smoke / offline tests).
    * ``agent`` — :class:`CriticAgentBackend` driving the ``critic-agent``
      skill runtime. Adds KB priors / writes, session memory, and
      ``review_constraints``-gated verdicts. Requires ``critic_agent_root``.
    * ``codex_bare`` — legacy direct :class:`CodexBackend` chat-completion
      path. Kept available for debugging the LLM layer in isolation.

    ``robustness_choice`` ∈ {``"mock"``, ``"agent"``}:

    * ``mock`` — heartbeat-only :class:`MockRobustnessBackend` (default;
      keeps optimizer self-contained).
    * ``agent`` — :class:`RobustnessAgentBackend` driving
      ``python -m robustness_agent.runtime.cli`` in a subprocess. Mirrors
      the critic-agent transport. Requires ``robustness_agent_root``.
    """
    if critic_choice not in ("mock", "agent", "codex_bare"):
        raise ValueError(
            f"_build_backends: critic_choice={critic_choice!r} not in "
            "{'mock','agent','codex_bare'}"
        )

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    elif critic_choice == "codex_bare":
        critic_backend = CodexBackend(model=codex_model)
    else:  # "agent"
        if critic_agent_root is None:
            raise ValueError(
                "_build_backends: critic_choice='agent' requires critic_agent_root"
            )
        # N38 (May 2026) — feed the registry-derived per-action
        # verdict policy so the critic-agent runtime sees
        # ``review_constraints.action_verdict_policy[<action_name>]``
        # and approves exploration / archival actions without
        # demanding the before/after evidence they themselves produce.
        # Replaces the prompt-hardcoded carve-out lists N33/N35/N37
        # had to patch one action at a time.
        try:
            from inference_optimizer.orchestrator.action_registry import (
                ActionRegistry,
            )
            _reg = ActionRegistry().load()
            _policy = {a.name: a.verdict_class for a in _reg.all()}
        except Exception:  # noqa: BLE001 — degrade to empty policy
            _policy = {}
        critic_backend = CriticAgentBackend(
            critic_agent_root=critic_agent_root,
            session_dir=session_dir,
            codex_model=codex_model,
            kb_mode=critic_kb_mode,
            action_verdict_policy=_policy,
        )

    if robustness_choice not in ("mock", "agent"):
        raise ValueError(
            f"_build_backends: robustness_choice={robustness_choice!r} not in "
            "{'mock','agent'}"
        )
    if robustness_choice == "mock":
        robustness_backend: Any = MockRobustnessBackend()
    else:  # "agent"
        if robustness_agent_root is None:
            raise ValueError(
                "_build_backends: robustness_choice='agent' requires "
                "robustness_agent_root"
            )
        robustness_backend = RobustnessAgentBackend(
            robustness_agent_root=robustness_agent_root,
            session_dir=session_dir,
            options=robustness_options,
        )

    backends: dict[str, Any] = {
        "orchestration": ClaudeBackend(model=claude_model, max_turns_default=4),
        "critic":        critic_backend,
        "robustness":    robustness_backend,
    }
    if not no_kernel:
        if kernel_codex:
            backends["kernel"] = CodexBackend(model=codex_model)
        else:
            backends["kernel"] = ClaudeBackend(model=claude_model, max_turns_default=4)
    return backends


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
) -> SharedState:
    # research_lane capacity is locked here for the lifetime
    # of the session. Clamp to [0, MAX_RESEARCH_LANE_CAPACITY] (M6
    # ceiling; values above this are silently clamped down). The cap
    # protects LLM quota and PR Monitor load (§3.14 R-07).
    from inference_optimizer.orchestrator.policy import (
        MAX_RESEARCH_LANE_CAPACITY,
    )
    research_lane_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 1
    )
    research_lane_capacity = max(
        0, min(MAX_RESEARCH_LANE_CAPACITY, research_lane_capacity),
    )
    # collect plateau threshold overrides into a single dict
    #. Only non-None CLI overrides
    # land in the dict; absent keys fall through to the
    # ``DEFAULT_PLATEAU_*`` library constants at phase-compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)
    if getattr(args, "plateau_kernel_revert_streak", None) is not None:
        plateau_overrides["kernel_revert_streak"] = int(args.plateau_kernel_revert_streak)
    if getattr(args, "plateau_kernel_keep_gain", None) is not None:
        plateau_overrides["kernel_keep_gain_pct"] = float(args.plateau_kernel_keep_gain)
    if getattr(args, "plateau_kernel_lookback", None) is not None:
        plateau_overrides["kernel_lookback"] = int(args.plateau_kernel_lookback)
    # EXPLORE HARD force-exit thresholds (IR-6). Either condition fires
    # an ``explore_force_exit_low_budget`` exit (overrides plateau /
    # steward / LLM proposals).
    if getattr(args, "explore_force_exit_hours_remaining", None) is not None:
        plateau_overrides["force_exit_hours_remaining"] = float(
            args.explore_force_exit_hours_remaining
        )
    if getattr(args, "explore_force_exit_budget_pct", None) is not None:
        plateau_overrides["force_exit_budget_pct"] = float(
            args.explore_force_exit_budget_pct
        )
    # IR-7 steward controls.
    if getattr(args, "steward_disabled", False):
        plateau_overrides["steward_disabled"] = True
    if getattr(args, "steward_continuation_cap", None) is not None:
        plateau_overrides["steward_continuation_cap"] = int(
            args.steward_continuation_cap
        )

    # Resolve workload metadata from CLI flags first, then env. The
    # same fields are mirrored into manifest.json by manifest.build_manifest
    # (single source of truth); we duplicate the parse here so SharedState
    # can be populated without re-reading the file we just wrote.
    def _int_env_or_arg(arg_name: str, env_name: str) -> int:
        val = getattr(args, arg_name, None)
        if val is None or val == 0:
            raw = (os.environ.get(env_name, "") or "").strip()
            return int(raw) if raw.isdigit() else 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    # Fix E (--explore-overtime-kill-ratio): mirror the CLI value into
    # the fresh SharedState so the ExploreExecutor can read it via the
    # Coordinator-injected task.params on the very first explore round.
    # ``0`` (and any non-positive) disables the gate.
    explore_overtime_kill_ratio_raw = getattr(
        args, "explore_overtime_kill_ratio", None,
    )
    try:
        explore_overtime_kill_ratio = (
            float(explore_overtime_kill_ratio_raw)
            if explore_overtime_kill_ratio_raw is not None else 1.10
        )
    except (TypeError, ValueError):
        explore_overtime_kill_ratio = 1.10

    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI / env at fresh-session time
        # so downstream consumers (specialist prompt builder,
        # orchestration tick prompt) see the real values. Without this
        # the specialist prompt's "## 2. HARDWARE CONTEXT" silently uses
        # the SpecialistPromptInputs dataclass defaults (e.g. TP=1) and
        # comm_specialist self-vetoes on TP=8 sessions.
        tp=_int_env_or_arg("tp", "TP"),
        precision=(
            str(getattr(args, "precision", None) or os.environ.get("PRECISION", "") or "").strip()
        ),
        conc=_int_env_or_arg("conc", "CONC"),
        isl=_int_env_or_arg("isl", "ISL"),
        osl=_int_env_or_arg("osl", "OSL"),
        max_model_len=_int_env_or_arg("max_model_len", "MAX_MODEL_LEN"),
        kernel_enabled=not getattr(args, "no_kernel", False),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        plateau_overrides=plateau_overrides,
        explore_overtime_kill_ratio=explore_overtime_kill_ratio,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_PR phase (PRELUDE → FRAMEWORK_PR →
        # EXPLORE). ``--no-framework`` skips it; default on. Mirrors
        # the ``--no-kernel`` / ``kernel_enabled`` pattern.
        framework_phase_enabled=not bool(getattr(args, "no_framework", False)),
        gain_driven_kernel_opt=bool(
            getattr(args, "gain_driven_kernel_opt", False),
        ),
    )
    state.save(session_dir)
    return state


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout."""
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print(f"  [ok] manifest.json (written first)")


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
) -> None:
    """Persist each persistent agent's effective system prompt.

    Writes to ``agents/<role>/system_prompt.snapshot.md`` so resume
    runs and post-mortem inspection can compare against the in-memory
    prompt without re-deriving it from CLI args.
    """
    for role, body in prompts.items():
        target = agent_prompt_snapshot(session_dir, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")


def _default_target_summary(args: argparse.Namespace) -> str:
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        return (
            f"Establish baseline on {Path(args.model).name} then reach "
            f"{args.target_tput} tok/s/GPU within {args.max_hours}h."
        )
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."


# --- Executor wiring tables ------------------------------------------------
# Declarative mappings of action_kind → ExecutorFn so tests can introspect
# what's actually wired without re-parsing the imperative body of
# ``_register_executors``. Adding a new action with a real executor MUST
# update these tables; the regression test in
# ``tests/test_p1_2_full_action_catalogue.py`` enforces consistency between
# these tables and ``session_paths._runs_actions()``.

# Real executors enabled in every run mode (kernel + no-kernel).
#
# v0.8 M3 + KB_gaps/Gap-10: the legacy ``backends`` / ``params`` /
# ``validate_stack`` registrations have been removed alongside
# PolicyGate's ``action_deprecated`` rule. The merged ``explore``
# action subsumes the per-variant KEEP/REVERT plus the per-KEEP stack
# rebench; validate_stack is no longer a
# standalone action.
#
# The legacy executor Python modules (``action_executors/backends.py``,
# ``params.py``, ``validate_stack.py``) have been physically deleted
# from the tree. The legacy resume audit trails (``backends_attempts``
# etc.) keep their meaning on disk so legacy session resumes still
# render correctly. New sessions never see these action names because
# PolicyGate denies them with ``rule='action_deprecated'`` before a
# task is ever queued.
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline":          baseline_executor,
    # ``profile`` is registered so the Coordinator-internal task path
    # (kind switched via ``--no-enable-roofline``) can dispatch it
    # through SubAgentRunner. PolicyGate denies LLM-proposed
    # delegate{action_name='profile'} via
    # ``analysis_action_not_llm_proposable``, so this registration is
    # effectively Coordinator-only.
    "profile":           profile_executor,
    "explore":           explore_executor,
    "sweep":             sweep_executor,
    "report":            report_executor,
    "session_breakdown": session_breakdown_executor,
    # ``recover`` re-enabled in 2026-05 alongside the robustness-agent
    # ``gpu_memory_leaked`` signal (Change A/B): a real executor now
    # cleans up leaked VRAM owners and, behind
    # ``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1``, optionally shells out to
    # ``rocm-smi --gpureset``. See
    # ``orchestrator/action_executors/recover.py``. ``validate_stack``
    # has been retired and is intentionally absent.
    "recover":           recover_executor,
}

# Kernel-only real executors. The composite ``roofline`` action is
# registered separately by ``_register_executors`` below. ``profile``
# is registered in ``_REAL_EXECUTORS_FULL`` so the Coordinator's
# auto-managed analysis path can dispatch it through SubAgentRunner
# when ``--no-enable-roofline`` is set; the same ``profile_executor``
# is also called directly from RooflineExecutor's ``_wrap_profile_ctx``
# in the default roofline mode. PolicyGate denies LLM-proposed
# delegate{action_name='profile'} regardless of mode.
_REAL_EXECUTORS_KERNEL_ONLY: dict[str, Any] = {}

# Kernel-owned action kinds dispatched via
# ``request{target_agent='kernel', kind=...}``. The executor body is a
# no-op in this process — actual work happens inside the kernel agent's
# request handlers — but the names must stay registered so SubAgentRunner
# does not raise ``no_executor`` on a stale task.
_NOOP_KINDS_KERNEL_ONLY: tuple[str, ...] = (
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "operator_tuning", "vendor_kernel_config",
)


def _build_specialist_executor(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> "Callable[[Any], Awaitable[dict]]":
    """v0.8 §3.5 / §3.13 M5 + PR-A2 (Arbor-into-Hyperloom) — build the
    specialist executor adapter.

    Returns an ``async fn(ctx: RunnerContext) -> dict`` compatible with
    :data:`SubAgentRunner.ExecutorFn`. The adapter wraps a
    :class:`SpecialistRunner` and translates its
    :class:`SpecialistRunResult` dataclass into the dict the dispatcher
    expects to publish onto the bus.

    Dispatch mode (PR-A2): production wires the
    :class:`SpecialistSubprocessDispatcher` so each specialist runs in
    a fresh ``claude`` subprocess inside a per-task git worktree
    (``runs/specialist/<task_id>/worktree/``). The
    ``--specialist-dispatch-mode`` flag can fall back to the legacy M5
    in-process :class:`ClaudeBackend` path (used by tests + when the
    ``claude`` binary is missing from $PATH).

    Backend choice: every specialist runs on Claude (matching the
    orchestration role). The CLI flag ``--specialist-model`` overrides
    the default model (``--claude-model``) so operators can dedicate a
    cheaper / faster model to specialist research without touching the
    main orchestration loop. KB_design §3.5 §6 / §3.14 R-05.

    The factory captures ``session_dir`` + ``knowledge_plane`` once at
    cli boot; the same runner instance handles every specialist task
    for the session.
    """
    import shutil

    from .orchestrator.specialist_mcp_config import write_specialist_mcp_config
    from .orchestrator.specialist_runner import (
        DEFAULT_SPECIALIST_TOOLS,
        SpecialistRunner,
    )
    from .orchestrator.specialist_subprocess import SpecialistSubprocessConfig
    from .orchestrator.sub_agent_runner import SubAgentResult

    claude_model = (
        (getattr(args, "specialist_model", None) or args.claude_model)
        .strip()
    )
    max_turns = int(getattr(args, "specialist_max_turns", 8) or 8)
    per_turn_max_seconds = float(
        getattr(args, "specialist_per_turn_max_seconds", 600.0) or 600.0
    )
    dispatch_mode = (
        str(getattr(args, "specialist_dispatch_mode", "subprocess") or "subprocess")
        .strip().lower()
    )

    # PR-A2: subprocess dispatch is the production default. We fall
    # back to in-process when (a) the operator picks ``inprocess`` or
    # (b) the ``claude`` binary is not on $PATH (e.g. dev environments
    # using only the in-process SDK). The fallback is logged so it's
    # visible in the manifest.
    # PR-A2: derive framework_source_roots from the canonical resolver so
    # the specialist worktree is rooted at the same set the orchestration
    # prompt + PolicyGate path-validator already trust.
    framework_source_roots = tuple(resolve_source_file_allowlist())
    claude_bin = shutil.which("claude") or ""
    use_subprocess = dispatch_mode != "inprocess" and bool(claude_bin)
    if dispatch_mode == "subprocess" and not claude_bin:
        log.warning(
            "specialist_dispatch_mode=subprocess requested but `claude` "
            "binary not found on PATH; falling back to in-process backend",
        )

    if use_subprocess:
        # Operator-supplied --specialist-mcp-config wins. When unset,
        # auto-generate ``<session_dir>/runtime/specialist_mcp.json``
        # from the live KnowledgePlane so the spawned claude subprocess
        # actually has the PR Monitor MCP server wired (without it the
        # ``mcp__pr_monitor__*`` tool names in the whitelist resolve to
        # nothing and the specialist falls back to WebSearch).
        mcp_config_path: str | None = str(
            getattr(args, "specialist_mcp_config", "") or ""
        ) or None
        if mcp_config_path is None and knowledge_plane is not None:
            try:
                pr_mcp_url = knowledge_plane.specialist_mcp_url()
            except AttributeError:
                pr_mcp_url = ""
            generated = write_specialist_mcp_config(
                session_dir=session_dir,
                pr_monitor_mcp_url=pr_mcp_url,
            )
            if generated is not None:
                mcp_config_path = str(generated)
        sub_config = SpecialistSubprocessConfig(
            claude_executable=claude_bin or "claude",
            model=claude_model,
            framework_source_roots=framework_source_roots,
            mcp_config_path=mcp_config_path,
            per_turn_max_seconds=per_turn_max_seconds,
        )
        runner = SpecialistRunner(
            subprocess_config=sub_config,
            session_dir=session_dir,
            default_tools=DEFAULT_SPECIALIST_TOOLS,
            default_max_turns=max_turns,
            per_turn_max_seconds=per_turn_max_seconds,
            knowledge_plane=knowledge_plane,
        )
    else:
        def _backend_factory(domain: Any) -> Any:
            # in-process Claude path (fallback).
            return ClaudeBackend(
                model=claude_model, max_turns_default=max_turns,
            )

        runner = SpecialistRunner(
            backend_factory=_backend_factory,
            session_dir=session_dir,
            default_tools=DEFAULT_SPECIALIST_TOOLS,
            default_max_turns=max_turns,
            per_turn_max_seconds=per_turn_max_seconds,
            knowledge_plane=knowledge_plane,
        )

    async def _executor(ctx: Any) -> dict:
        """Adapter: SubAgentRunner.run_task → SpecialistRunner.run.

        The SubAgentRunner wrapper has already transitioned the task to
        ``running`` and ``ctx.task`` is the live :class:`Task`. We
        always return a dict (even on failure) so the dispatcher's
        ``state.transition('succeeded', evidence={...})`` step gets a
        well-formed payload. SpecialistRunResult's distinction between
        ``succeeded`` / ``empty_synthesised`` / ``stale`` etc. is
        preserved under ``result.runner_status`` for downstream
        analytics (breakdown.specialist_runs).
        """
        run_result = await runner.run(ctx)
        # Translate dataclass → dict. The Coordinator's
        # ``_handle_intent`` Gap-03 path (when wired) will pull
        # ``specialist_done`` out of result.payload['specialist_done'].
        return {
            "runner_status": run_result.status,
            "task_id": run_result.task_id,
            "domain": run_result.domain,
            "gap_canonical_id": run_result.gap_canonical_id,
            "specialist_done": run_result.specialist_done,
            "turns_used": run_result.turns_used,
            "workspace": run_result.workspace,
            "transcript_path": run_result.transcript_path,
            "done_path": run_result.done_path,
            "error": run_result.error,
            "notes": list(run_result.notes or []),
        }

    return _executor


def _register_executors(
    coordinator: Coordinator,
    *,
    no_kernel: bool = False,
    compare_against_gpu: str | None = None,
    session_dir: Path | None = None,
    specialist_executor: "Callable[[Any], Awaitable[dict]] | None" = None,
) -> None:
    """Wire all currently-available action executors.

    Real executors are pulled from ``_REAL_EXECUTORS_FULL`` (always) and
    ``_REAL_EXECUTORS_KERNEL_ONLY`` (when kernel-mode is on). Kernel-owned
    kinds (whose work is done inside the kernel agent via emit_intent)
    get ``_noop_prep`` so SubAgentRunner doesn't fail with "no_executor".

    When ``no_kernel`` is True, the kernel-owned executor table is
    skipped and the kernel-only no-op stubs are skipped. The Coordinator-
    internal ``profile`` and ``roofline`` analysis executors are
    registered unconditionally so PRELUDE's auto-enqueued analysis task
    (kind switched by ``--enable-roofline``) can always dispatch.

    ``target_analysis`` is *always* registered with the real
    :class:`TargetAnalysisExecutor`. When ``compare_against_gpu`` is a
    non-empty string, the executor fetches the matching InferenceX
    reference; when empty / None, it writes a structured
    ``reason='no_target_gpu_configured'`` marker JSON so the report
    section is rendered uniformly in both cases.

    ``specialist_executor`` (v0.8 §3.5 / KB_gaps/Gap-01): the
    :func:`_build_specialist_executor` adapter, when the operator has
    ``--research-lane-capacity > 0``. ``None`` keeps the legacy
    behaviour where a ``delegate{action='specialist'}`` task hits
    ``no_executor`` and fails (M3 / pre-M5 path).
    """
    for kind, fn in _REAL_EXECUTORS_FULL.items():
        coordinator.sub.register_executor(kind, fn)

    coordinator.sub.register_executor(
        "target_analysis",
        TargetAnalysisExecutor(
            compare_against_gpu=(compare_against_gpu or "").strip(),
            session_dir=session_dir,
        ),
    )

    # v0.8 §3.5 + register the specialist sub-agent
    # adapter so ``delegate{action='specialist'}`` no longer hits
    # ``no_executor``. Gated by ``--research-lane-capacity`` upstream
    # (cli only passes a non-None executor when capacity > 0).
    if specialist_executor is not None:
        coordinator.sub.register_executor("specialist", specialist_executor)

    # PR-A4 (Arbor-into-Hyperloom): wire the real IntegratePatchExecutor.
    # The executor reads the specialist's worktree patches, applies them
    # to framework_source_roots via ``git apply``, runs a Magpie bench,
    # and decides KEEP / REVERT. It is the single integration point —
    # specialists never apply patches themselves (Inv-5.1 updated by
    # PR-A2; the worktree authorisation gives them write capability,
    # but the orchestrator-side ``integrate_patch`` is what makes those
    # patches visible to the serving framework).
    coordinator.sub.register_executor(
        "integrate_patch",
        IntegratePatchExecutor(session_dir=session_dir),
    )

    # FRAMEWORK_PR phase per-candidate executor — Coordinator-internal
    # only (PolicyGate denies LLM ``delegate{action='framework_pr'}``
    # via ``framework_pr_action_not_llm_proposable``).
    coordinator.sub.register_executor(
        "framework_pr",
        FrameworkPrExecutor(session_dir=session_dir),
    )

    # The composite ``roofline`` action runs profile + trace_analyze
    # atomically and surfaces analysis.md to the next orchestration
    # tick. Coordinator auto-enqueues it at PRELUDE and on every 10%
    # gain watermark crossing — independent of ``--no-kernel`` — so the
    # executor is unconditionally registered. ``profile`` is the
    # ``--no-enable-roofline`` alternative and is registered via
    # ``_REAL_EXECUTORS_FULL`` above. PolicyGate denies LLM-proposed
    # delegate{action_name='roofline'|'profile'} regardless of mode.
    coordinator.sub.register_executor(
        "roofline",
        make_roofline_executor(shared_state=coordinator.shared_state),
    )

    if log.isEnabledFor(logging.DEBUG):
        for required_kind in ("roofline", "profile"):
            if required_kind not in coordinator.sub.executor_registry:
                log.debug(
                    "register_executors: %r missing from sub-agent registry "
                    "(no_kernel=%s); PRELUDE analysis task will fail with "
                    "no_executor",
                    required_kind, no_kernel,
                )

    if no_kernel:
        return

    for kind, fn in _REAL_EXECUTORS_KERNEL_ONLY.items():
        coordinator.sub.register_executor(kind, fn)
    for kind in _NOOP_KINDS_KERNEL_ONLY:
        coordinator.sub.register_executor(kind, _noop_prep)


def _print_final_summary(state: SharedState, stop_reason: str) -> None:
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    print(f"  baseline_tput        : {state.baseline_tput:.1f} tok/s/GPU")
    print(
        f"  cumulative_gain      : {state.cumulative_gain:.2f}% "
        f"(per-round sum — informational)"
    )
    if state.cumulative_gain_validated_ts:
        stale = (
            " ⚠ stack changed since validation"
            if len(state.optimization_stack) > state.cumulative_gain_validated_stack_len
            else ""
        )
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print(
            f"  cumulative_gain_val  : 0.00% "
            f"⚠ never validated — no `explore` stack-rebench has succeeded yet"
        )
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    print("===============================================")


def _derive_anthropic_base_url(openai_base_url: str) -> str:
    """Derive ``ANTHROPIC_BASE_URL`` from ``OPENAI_BASE_URL``.

    The Anthropic SDK appends ``/v1`` itself, so we strip a trailing
    ``/v1`` from the OpenAI-style URL. Returns the input verbatim when
    there is no ``/v1`` suffix to strip.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse(parsed._replace(path=path))


def _reset_claude_config_to_upstream(
    safe_key: str, anthropic_base_url: str
) -> None:
    """Point ``~/.claude/config.json`` ``customApiUrl`` at the upstream gateway.

    Used after the auth-proxy was removed: a stale ``127.0.0.1:4002``
    value would make the Claude CLI dial a port no longer bound. We
    rewrite to the upstream Anthropic URL so the CLI talks directly to
    the gateway with its ``x-api-key`` header (the gateway accepts it).
    """
    import json as _json

    if not anthropic_base_url:
        return
    claude_config_path = Path.home() / ".claude" / "config.json"
    config_data: dict = {}
    if claude_config_path.exists():
        try:
            config_data = _json.loads(
                claude_config_path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            config_data = {}
        current_url = config_data.get("customApiUrl", "")
        if current_url == anthropic_base_url:
            print("Preflight: ~/.claude/config.json already points at upstream")
            return

    config_data.setdefault("theme", "dark")
    config_data.setdefault("hasCompletedOnboarding", True)
    if safe_key:
        config_data["primaryApiKey"] = safe_key
    elif "primaryApiKey" not in config_data:
        config_data["primaryApiKey"] = ""
    config_data["customApiUrl"] = anthropic_base_url
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    claude_config_path.write_text(
        _json.dumps(config_data, indent=2) + "\n", encoding="utf-8",
    )
    claude_config_path.chmod(0o600)
    print(
        f"Preflight: updated ~/.claude/config.json customApiUrl -> "
        f"{anthropic_base_url}"
    )


def _load_dotenv_fallback() -> None:
    """Env always wins over .env. If SAFE_API_KEY or OPENAI_BASE_URL is
    missing from os.environ, source ``$REPO_ROOT/.env`` (defaults to
    ``os.getcwd()``) but never overwrite a key that is already in env.
    """
    if os.environ.get("SAFE_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    if not env_file.exists():
        return
    loaded = 0
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")


def _load_kernel_agent_env_fallback() -> None:
    """If ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset, auto-source the
    kernel-agent env file produced by ``inference_optimizer/scripts/
    install.sh`` at ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``
    (overridable via ``$KERNEL_AGENT_ENV``).

    Background: the May 2026 R1 N14 run stalled for 1h 12min because
    the launcher only sourced the user's basic ``.env`` (3 vars) and
    missed kernel-agent.env.sh. ``RooflineExecutor``'s trace_analyze
    sub-step imports ``kernel_request_handlers.HYPERLOOM_KERNEL_AGENT_ROOT``
    at module load — that read happens before any user code can fix the
    env, so the only way to recover without a restart is to source the
    file here, before any orchestrator import. Setting the env in this
    process also propagates to all subprocesses launched by Magpie /
    TraceLens / GEAK runners.

    Hard-fail contract (revised after the May 2026 Qwen1.5-7B 10h
    silent-stall: env file at the wrong path -> silent WARN-only ->
    5 rooflines all profile-success / trace_analyze-fail / snapshot
    stays None / LLM heartbeat-only 7.5h). The function now:

    * Looks ONLY at ``$KERNEL_AGENT_ENV`` (if set) or
      ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``. USER_DATA_PATH
      MUST be the workspace root (per N17 split: ``runtime/`` is
      workspace-shared, not per-session). No parent-dir fallback.
    * If the file is missing OR parses 0 vars OR
      HYPERLOOM_KERNEL_AGENT_ROOT is still unset after sourcing,
      ``sys.exit(2)`` with a clear actionable message instead of
      silently warning and letting trace_analyze fail later. Fail
      early, fail loud: the operator sees the problem in the first
      30 seconds, not 10 hours in.
    * Skip entirely when HYPERLOOM_KERNEL_AGENT_ROOT is already set
      (operator pre-sourced manually, or running from a sandbox that
      injected it — both fine).
    """
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = os.environ.get("USER_DATA_PATH")
        if not user_data:
            print(
                "Preflight: ERROR — neither $HYPERLOOM_KERNEL_AGENT_ROOT "
                "nor $KERNEL_AGENT_ENV nor $USER_DATA_PATH is set. Cannot "
                "resolve kernel-agent.env.sh. Run "
                "inference_optimizer/scripts/install.sh and export "
                "USER_DATA_PATH=/path/to/sessions first.",
                file=sys.stderr,
            )
            sys.exit(2)
        candidate = str(Path(user_data) / "runtime" / "kernel-agent.env.sh")
    env_path = Path(candidate)
    if not env_path.is_file():
        print(
            f"Preflight: ERROR — kernel-agent env file not found at "
            f"{env_path}. USER_DATA_PATH must be the workspace root "
            f"(parent of <model>/<ts>/ per-session subdirs); runtime/ "
            f"is workspace-shared, not per-session. Either "
            f"(a) re-run inference_optimizer/scripts/install.sh under "
            f"USER_DATA_PATH={os.environ.get('USER_DATA_PATH','?')}, "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file, or "
            f"(c) set $HYPERLOOM_KERNEL_AGENT_ROOT directly to skip this "
            f"fallback entirely. Aborting now (was: silently warning and "
            f"letting trace_analyze fail 10h in).",
            file=sys.stderr,
        )
        sys.exit(2)
    loaded = 0
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(
            f"Preflight: ERROR — failed to read {env_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if "HYPERLOOM_KERNEL_AGENT_ROOT" not in os.environ:
        print(
            f"Preflight: ERROR — sourced {env_path} ({loaded} vars) but "
            f"HYPERLOOM_KERNEL_AGENT_ROOT is still unset. The env file is "
            f"malformed or stale. Re-run inference_optimizer/scripts/"
            f"install.sh to regenerate it.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"Preflight: loaded {loaded} kernel-agent var(s) from "
        f"{env_path} (env wins, HYPERLOOM_KERNEL_AGENT_ROOT="
        f"{os.environ['HYPERLOOM_KERNEL_AGENT_ROOT']})"
    )


def _ensure_python_sdks(python_exe: str, pip_extra: list[str]) -> None:
    """Install Python SDKs that ``inference_optimizer`` imports at runtime.

    Even though they're declared in ``pyproject.toml``, a sandbox that only
    pulled the source tree (or used ``pip install`` without resolving the
    dep tree) lands here without them. ``ClaudeBackend`` / ``CodexBackend``
    lazy-import them, so without this guard the first reactor tick fails
    with ``BackendError: claude-agent-sdk not installed`` after baseline
    has already burned 5+ minutes of wall time.

    We probe-then-install per package using the SAME interpreter that
    will later import them (``sys.executable``); cross-interpreter installs
    are the precise failure mode that "claude-agent-sdk was not installed
    in /opt/venv → installed 0.1.77" reports come from.
    """
    candidates = (
        ("claude_agent_sdk", "claude-agent-sdk>=0.1.65"),
        ("openai",           "openai>=1.50"),
        ("httpx",             "httpx>=0.27"),
    )
    for module_name, pip_spec in candidates:
        check = subprocess.run(
            [python_exe, "-c", f"import {module_name}"],
            capture_output=True,
        )
        if check.returncode == 0:
            print(f"Preflight: {module_name} OK")
            continue
        print(f"Preflight: {module_name} not importable, installing {pip_spec} ...")
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet",
             *pip_extra, pip_spec],
            check=True,
        )
        print(f"Preflight: installed {pip_spec}")


def _unset_hip_visible_devices() -> None:
    """Drop ``HIP_VISIBLE_DEVICES`` if ``ROCR_VISIBLE_DEVICES`` is set.

    Long-standing ROCm gotcha (mirrored in ``inference_optimizer/SKILL.md``
    §"GPU Runner Type"): when both vars are set, ``HIP_VISIBLE_DEVICES``
    can mask devices in a way that makes ``torch.cuda.is_available()``
    return False inside the Magpie subprocess, which then logs "No
    accelerator" and exits non-zero. We use ``ROCR_VISIBLE_DEVICES`` as the
    canonical pinning; ``HIP_VISIBLE_DEVICES`` is a footgun left over from
    upstream CUDA scripts that copy-pasted into ROCm sandboxes.
    """
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        return
    if "ROCR_VISIBLE_DEVICES" not in os.environ:
        return
    value = os.environ.pop("HIP_VISIBLE_DEVICES")
    print(
        f"Preflight: WARNING — unset HIP_VISIBLE_DEVICES={value!r} "
        f"(ROCR_VISIBLE_DEVICES wins on ROCm; HIP_VISIBLE_DEVICES can "
        f"make torch.cuda.is_available() false inside Magpie subprocess)"
    )


def _check_gpu_visibility() -> None:
    """Best-effort sanity check on visible GPU count vs ``$TP``.

    rocm-smi may be missing on CPU-only test boxes; we return silently in
    that case and let Magpie's own detection complain later. The check is
    purely informational — TP can also be set inside the Magpie YAML, in
    which case this WARN is a false positive but cheap.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return
    if proc.returncode != 0:
        return
    visible = sum(
        1 for line in (proc.stdout or "").splitlines()
        if line.strip().startswith("GPU[")
    )
    try:
        wanted = int(os.environ.get("TP", "1") or "1")
    except ValueError:
        wanted = 1
    if visible == 0:
        print("Preflight: WARNING — rocm-smi sees 0 GPUs; benchmark will fail")
        return
    if wanted > visible:
        print(
            f"Preflight: WARNING — TP={wanted} but rocm-smi sees {visible} "
            f"GPU(s); sglang/vllm may fail to load weights. Lower TP or "
            f"adjust ROCR_VISIBLE_DEVICES."
        )


def _check_shm_disk() -> None:
    """Warn on tight ``/dev/shm`` (vLLM/NCCL IPC needs headroom).

    ``_kill_stale_servers()`` already clears stale segments before each
    Magpie variant, but if the partition itself is small the very first
    launch can still collide. We don't fail-fast — partitions can be
    short-lived (test sandboxes use tmpfs sized to the workload).
    """
    try:
        usage = shutil.disk_usage("/dev/shm")
    except (FileNotFoundError, OSError):
        return
    if usage.free < _DEV_SHM_MIN_FREE_BYTES:
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(
            f"Preflight: WARNING — /dev/shm has {free_gb:.1f} GiB free of "
            f"{total_gb:.1f} GiB total (< 16 GiB threshold). vLLM IPC + "
            f"NCCL shm segments may collide with stale entries; if the "
            f"first server launch hangs >5min, clear /dev/shm/{{vllm,nccl,cuda}}*"
        )


_TRACELENS_REQUIRED_CLIS: tuple[str, ...] = (
    "TraceLens_generate_perf_report_pytorch_inference",
)


def _check_tracelens_cli() -> None:
    """Hard-gate TraceLens CLI presence — abort BEFORE Coordinator starts.

    Why this is hard-fail (vs ``_check_node_claude_cli`` which is WARN-only):

    The ``TraceLens_*`` console_scripts are installed by
    ``kernel-agent/scripts/install.sh`` (chained from
    ``inference_optimizer/scripts/install.sh``) into the *pod-local*
    ``/opt/venv/bin/`` — they do NOT persist across pod restarts even
    when ``$USER_DATA_PATH`` (typically ``/workspace/hyperloom``) is a
    WekaFS-backed session dir that survives pod recycling. SKILL §IR-2
    therefore requires running ``install.sh`` before every launch; the
    only carve-out is ``--resume`` in the *same shell* that earlier ran
    install.sh.

    Brain-generated launchers that source only
    ``runtime/kernel-agent.env.sh`` and skip install.sh (fresh-start
    ``--model`` path) land here with no TraceLens CLI on PATH. Until this
    gate was added, the missing-CLI failure was surfaced only by the
    robustness agent's J3 signal at tick ~6 (HIGH severity
    ``tracelens_cli_missing``) — after baseline had already completed
    (or hung) and a multi-minute setup cost was wasted.
    ``select_kernels`` / ``kernel_opt`` then fail downstream when they
    shell out to ``tracelens_analysis.py``.

    Moving discovery to launch — mirroring ``_gate_claude_model``
    (SKILL §Step 2 step 10) — turns a delayed silent strike into a
    fail-fast with an actionable error pointing at the install.sh fix.
    """
    missing = [
        name for name in _TRACELENS_REQUIRED_CLIS
        if shutil.which(name) is None
    ]
    if not missing:
        return
    session_dir = os.environ.get("USER_DATA_PATH", "/workspace/hyperloom")
    print(
        f"ERROR: TraceLens CLI(s) not on PATH: {missing}. The pod-local "
        f"/opt/venv/bin/TraceLens_* console_scripts are installed by "
        f"kernel-agent/scripts/install.sh (chained from "
        f"inference_optimizer/scripts/install.sh) and do NOT persist "
        f"across pod restarts. SKILL IR-2 requires running install.sh "
        f"before every launch (carve-out applies only to --resume in "
        f"the same shell that earlier ran install.sh). Re-run:\n"
        f"  bash $REPO_ROOT/inference_optimizer/scripts/install.sh\n"
        f"  . {session_dir}/runtime/kernel-agent.env.sh\n"
        f"then retry `inference_optimizer optimize`. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_node_claude_cli() -> None:
    """WARN-only presence check for the bundled agent CLIs and ``@cursor/sdk``.

    ``claude_agent_sdk`` typically shells out to the bundled
    ``@anthropic-ai/claude-code`` CLI; without it on PATH the SDK falls
    back to a direct HTTP path against the upstream gateway, so this
    is informational rather than fatal. Same for ``codex`` (used by
    ``CodexBackend`` whenever Codex is on the wire — i.e. ``--critic-agent``
    / ``--critic-codex-bare`` and/or ``--kernel-codex``). ``node`` is a
    transitive dep — if it's missing, npm-based recovery via
    ``kernel-agent/scripts/install.sh`` won't work either.

    The cursor backend talks to Cursor's own gateway via the ``@cursor/sdk``
    Node library (not a CLI). We probe it via ``require.resolve`` against
    ``$(npm root -g)`` since ``shutil.which('cursor')`` would always miss.
    """
    missing = [t for t in ("node", "claude", "codex") if shutil.which(t) is None]
    if missing:
        print(
            f"Preflight: WARNING — CLI(s) not on PATH: {missing}. "
            f"ClaudeBackend / CodexBackend may fall back to direct HTTP. "
            f"Run kernel-agent/scripts/install.sh to bring them in."
        )
    # @cursor/sdk presence — probe via Node since it's a library, not a CLI.
    if shutil.which("node") is not None and shutil.which("npm") is not None:
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"], capture_output=True, text=True, timeout=10,
            )
            global_modules = (npm_root.stdout or "").strip()
            probe = subprocess.run(
                ["node", "-e", "require.resolve('@cursor/sdk')"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "NODE_PATH": global_modules} if global_modules else None,
            )
            if probe.returncode != 0:
                print(
                    "Preflight: WARNING — @cursor/sdk not resolvable; cursor "
                    "backend will fail to start. Run kernel-agent/scripts/"
                    "install.sh to install it globally via npm."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            print(
                "Preflight: WARNING — could not probe @cursor/sdk presence; "
                "cursor backend may be unavailable."
            )


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    anthropic_base_url: str | None,
    args: argparse.Namespace | None = None,
) -> None:
    """One canonical diagnostics block at the end of preflight.

    Replaces the half-dozen scattered ``print(f"Preflight: ...")`` lines
    with a single grep-friendly section that operators (and the launcher
    LLM) can paste verbatim into status reports without spelunking the
    source for env var names.
    """
    from .orchestrator.action_executors.baseline import (
        BASELINE_COLD_START_TIMEOUT_SEC,
        BASELINE_DEFAULT_TIMEOUT_SEC,
        _probe_aiter_jit_cache,
    )
    from .paths import asset_root

    probe = _probe_aiter_jit_cache()
    cold_cap = os.environ.get(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
        str(BASELINE_COLD_START_TIMEOUT_SEC),
    )
    if probe["probe_status"] == "found":
        kind = "COLD" if probe["is_cold"] else "WARM"
        cache_line = (
            f"{probe['kernel_count']} .so / {probe['size_mb']} MB "
            f"({kind}) at {probe['path']}"
        )
    else:
        cache_line = f"<probe_status={probe['probe_status']}>"

    print("Preflight diagnostics:")
    print(f"  asset_root          = {asset_root()}")
    print(
        f"  session_dir         = {_session_dir_resolve()}  "
        f"({ENV_USER_DATA_PATH}="
        f"{os.environ.get(ENV_USER_DATA_PATH, '<unset>')}, "
        f"default={DEFAULT_SESSION_DIR})"
    )
    print(f"  magpie_python       = {magpie_python}")
    print(
        f"  INFERENCEX_PATH     = "
        f"{os.environ.get('INFERENCEX_PATH', '<unset>')}"
    )
    print(f"  aiter jit cache     = {cache_line}")
    print(f"  cold_start_timeout  = {cold_cap}s")
    print(f"  warm_timeout        = {BASELINE_DEFAULT_TIMEOUT_SEC}s")
    if anthropic_base_url:
        print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url} (direct to gateway)")
    else:
        print(
            "  ANTHROPIC_BASE_URL  = <unset> — OPENAI_BASE_URL missing; "
            "Claude SDK will fail"
        )
    if args is not None:
        kb_enabled = bool(getattr(args, "cortex_enabled", True))
        pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
        kb_reason = getattr(args, "kb_degraded_reason", None) or "-"
        pr_reason = getattr(args, "pr_degraded_reason", None) or "-"
        kb_status = "OK" if kb_enabled else f"DEGRADED ({kb_reason})"
        pr_status = "OK" if pr_enabled else f"DEGRADED ({pr_reason})"
        print(f"  kb_status           = {kb_status}")
        print(f"  pr_monitor_status   = {pr_status}")
        print(f"  kb_degraded_reason  = {kb_reason}")
        print(f"  pr_degraded_reason  = {pr_reason}")

    # Issue-H (Saturday May 2026): surface Cortex KB offline-queue
    # state. The flusher daemon already drains ``.kb_pending.ndjson``
    # in the background, but a dead-letter pile-up is the canonical
    # signal for "the prior session's KB writes were rejected (HTTP
    # 4xx schema), this session is starting cold." Operators have no
    # in-band way to see this today — they discover it only when
    # specialists return empty proposal_set.
    try:
        _print_cortex_kb_queue_status()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  cortex_kb_queue     = <probe_failed: {exc!r}>")


def _print_cortex_kb_queue_status() -> None:
    """Emit a one-line summary of the Cortex KB offline NDJSON queue.

    Pure visibility helper. The flusher daemon does the actual draining;
    we only count rows so the operator sees ``pending=N dead=M`` next
    to the rest of the preflight diagnostics. When ``pending > 0`` the
    operator can verify the flusher is alive via
    ``runtime/cortex/.kb_flusher.pid``; the dead-letter count is the
    422-style permanent-reject signal that telegraphs an upcoming
    cold-start session for new models.
    """
    from .session_paths import (
        cortex_dead_letter_ndjson,
        cortex_flushed_ndjson,
        cortex_pending_ndjson,
    )

    sd = _session_dir_resolve()
    pending = cortex_pending_ndjson(sd)
    dead = cortex_dead_letter_ndjson(sd)
    flushed = cortex_flushed_ndjson(sd)

    def _count(p: Path) -> int:
        if not p.exists():
            return 0
        try:
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    p_n, d_n, f_n = _count(pending), _count(dead), _count(flushed)
    print(
        f"  cortex_kb_queue     = pending={p_n} dead_letter={d_n} "
        f"flushed={f_n} (root={pending.parent})"
    )
    if d_n > 0:
        print(
            f"                        ⚠ {d_n} dead-letter row(s) — "
            f"prior KB writes permanently rejected (4xx schema). "
            f"Specialists for affected anchors will start cold "
            f"(no priors). See {dead}."
        )


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | None:
    """Probe ``<base_url>/models`` with retry; return set of model ids or None.

    Mirrors what the launcher had to do by hand (``terminals/6.txt``):

        curl -H "Authorization: Bearer $SAFE_API_KEY" \
             "https://gateway/api/v1/llm-proxy/v1/models" | jq '.data[].id'

    The gateway has a documented flake rate; we retry up to
    ``len(_CATALOG_RETRY_DELAYS_SEC)`` times with exponential backoff.

    TLS verification is ON by default. For internal gateways with self-
    signed certs, set ``INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1`` to
    skip verification (a warning is printed since the probe also sends
    ``Authorization: Bearer <api_key>``).
    """
    import time

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # _ensure_python_sdks should have installed it; if it didn't, we
        # cannot probe — return None so the caller can decide how to react.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    insecure = os.environ.get(
        "INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE", "",
    ).strip().lower() in ("1", "true", "yes")
    if insecure:
        print(
            "Preflight: WARNING — INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1 "
            "is set; catalog probe will skip TLS verification while sending "
            "an Authorization: Bearer header. Use only against trusted internal "
            "gateways with self-signed certs."
        )
        try:
            import urllib3  # type: ignore[import-not-found]
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass

    probe_url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    delays = (0.0, *_CATALOG_RETRY_DELAYS_SEC)
    last_err: str = ""
    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(
                probe_url,
                headers=headers,
                timeout=_CATALOG_REQUEST_TIMEOUT_SEC,
                verify=not insecure,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"failed: {last_err}"
            )
            continue
        if resp.status_code != 200:
            last_err = (
                f"HTTP {resp.status_code}: "
                f"{(resp.text or '')[:200]}"
            )
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"got {last_err}"
            )
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"returned non-JSON: {last_err}"
            )
            continue
        ids = {
            m["id"] for m in data.get("data") or []
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        }
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(
        f"Preflight: catalog probe exhausted {len(delays)} attempts "
        f"({last_err}); cannot validate model availability"
    )
    return None


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> set[str] | None:
    """Hard-gate Claude model selection. Mutates ``args.claude_model``.

    Per operator direction (2026-05-09):

    * ``--claude-model`` MUST be one of ``_CLAUDE_ALLOWED_MODELS`` (4-7 or
      4-6). Any other value aborts boot before Coordinator starts —
      orchestration drift on opus-4-5 / haiku silently degraded prior runs.
    * Probe the gateway catalog with retry (gateway flakes). If the chosen
      model is missing but the fallback (4-6) is in catalog, rewrite the
      arg + WARN. Otherwise sys.exit(2) — refuse to start the service.

    Returns the catalog id set on success (so the codex smoke-test can
    reuse it without re-probing); returns None if catalog probe failed but
    the chosen model was already in ``_CLAUDE_ALLOWED_MODELS`` AND we
    decide to proceed (we don't — gateway unreachable means abort).
    """
    chosen = (args.claude_model or "").strip()
    if chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of {list(_CLAUDE_ALLOWED_MODELS)} "
            f"(preferred: {_CLAUDE_PREFERRED_MODEL}, "
            f"fallback: {_CLAUDE_FALLBACK_MODEL}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Catalog probe uses GET <base>/models against the upstream gateway URL.
    # ``INFERENCE_OPTIMIZER_CATALOG_PROBE_URL`` remains an explicit override
    # path for operators who need to point the probe at a different host.
    base_url = (
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    if not base_url and resolved_urls is not None:
        base_url = resolved_urls[1]

    api_key = (
        os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )

    catalog_ids = _probe_llm_catalog(base_url=base_url, api_key=api_key)
    if catalog_ids is None:
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    if _CLAUDE_FALLBACK_MODEL in catalog_ids:
        print(
            f"Preflight: WARNING — {chosen!r} not in gateway catalog; "
            f"falling back to {_CLAUDE_FALLBACK_MODEL!r}"
        )
        args.claude_model = _CLAUDE_FALLBACK_MODEL
        return catalog_ids

    print(
        f"ERROR: neither {_CLAUDE_PREFERRED_MODEL!r} nor "
        f"{_CLAUDE_FALLBACK_MODEL!r} present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    catalog_ids: set[str] | None,
) -> None:
    """WARN-only catalog check for ``--codex-model`` (no hard gate).

    Operator pinned only Claude; Codex is allowed to use whatever is
    requested. We still want to flag obvious typos / missing models BEFORE
    the Coordinator starts ticking, since the Codex backend's
    ``__post_init__`` does not pre-validate against the catalog.
    """
    if catalog_ids is None:
        return
    # Codex is needed by the Kernel agent (when kernel-codex is on) and by
    # the Critic — both via the ``codex_bare`` direct path and the
    # ``agent`` path (which also calls Codex for review reasoning).
    critic_uses_codex = args.critic_backend in ("agent", "codex_bare")
    needs_codex = critic_uses_codex or (
        args.kernel_codex and not getattr(args, "no_kernel", False)
    )
    if not needs_codex:
        return
    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return
    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog or use --critic-mock / --kernel-claude to "
        f"avoid the Codex path entirely."
    )


def _preflight(
    args: argparse.Namespace | None = None,
) -> tuple[str, str] | None:
    """Auto-install missing runtime deps and export auth aliases.

    1. Credentials fallback: env > $REPO_ROOT/.env (env always wins).
    2. Auth aliases for Claude/Codex CLIs from SAFE_API_KEY/OPENAI_BASE_URL.
    3. Python SDK (claude-agent-sdk / openai / httpx) auto-install.
    4. Resolve ANTHROPIC_BASE_URL from OPENAI_BASE_URL (strip trailing /v1)
       and reset ``~/.claude/config.json`` ``customApiUrl`` to the upstream
       gateway. The auth-proxy on :4002 has been retired — the AMD
       primus-safe gateway accepts both ``x-api-key`` and Bearer auth so
       the proxy rewrite step is no longer needed.
    5. ROCm env hygiene (HIP_VISIBLE_DEVICES unset, GPU/shm sanity).
    6. ray + Magpie + InferenceX auto-install.
    7. node / claude / codex CLI presence check (WARN-only).
    8. Single canonical diagnostics block.

    Returns ``(anthropic_base_url, openai_base_url)`` — the resolved
    upstream URLs that ``_run_optimize`` uses for the catalog probe and
    diagnostics. ``None`` only when ``OPENAI_BASE_URL`` is missing.
    """
    _load_dotenv_fallback()
    _load_kernel_agent_env_fallback()

    # --- Auth alias export ---
    safe_key = os.environ.get("SAFE_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if safe_key:
        for alias in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                      "ANTHROPIC_API_KEY", "OOB_API_KEY", "GEAK_API_KEY",
                      "LLM_API_KEY", "AMD_LLM_API_KEY"):
            if os.environ.get(alias) != safe_key:
                os.environ[alias] = safe_key
                print(f"Preflight: refreshed {alias} from SAFE_API_KEY")
    # OOB / GEAK / LLM_API_BASE inherit the upstream URL verbatim.
    if base_url:
        for alias in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
            if os.environ.get(alias) != base_url:
                prev = os.environ.get(alias, "")
                os.environ[alias] = base_url
                print(
                    f"Preflight: {alias} {prev or '<unset>'} -> {base_url} "
                    f"(direct to gateway)"
                )

    # --- Resolve install interpreters ---
    from .orchestrator.action_executors._grid_runner import _resolve_magpie_python
    magpie_python = _resolve_magpie_python()

    # Detect whether we're in a venv; if not, add --break-system-packages
    # so pip doesn't refuse to install on bare-metal Debian/Ubuntu hosts.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) ---
    # Must happen BEFORE Coordinator import, since ClaudeBackend lazy-imports
    # claude_agent_sdk at construction time and the catalog probe later in
    # this preflight needs httpx. Use sys.executable so the package lands
    # in the same site-packages this process imports from.
    _ensure_python_sdks(sys.executable, pip_extra)

    # --- Resolve ANTHROPIC_BASE_URL + reset ~/.claude/config.json ---
    # The auth-proxy on :4002 has been removed. We force-override both URL
    # vars to keep them consistent and prevent stale shell/.env/k8s values
    # (e.g. a legacy 127.0.0.1:4002 leftover) from reaching the CLIs.
    resolved_urls: tuple[str, str] | None = None
    if base_url:
        anthropic_url = _derive_anthropic_base_url(base_url)
        orig_anthropic = os.environ.get("ANTHROPIC_BASE_URL", "")
        orig_openai = os.environ.get("OPENAI_BASE_URL", "")
        for var, want, prev in (
            ("ANTHROPIC_BASE_URL", anthropic_url, orig_anthropic),
            ("OPENAI_BASE_URL", base_url, orig_openai),
        ):
            if os.environ.get(var) != want:
                os.environ[var] = want
                print(
                    f"Preflight: {var} {prev or '<unset>'} -> {want} "
                    f"(direct to gateway)"
                )
        _reset_claude_config_to_upstream(safe_key, anthropic_url)
        resolved_urls = (anthropic_url, base_url)
    else:
        print(
            "Preflight: WARNING — OPENAI_BASE_URL unset; "
            "Claude/Codex SDKs will fail at first call"
        )

    # --- ROCm env hygiene + GPU/shm sanity (defensive WARN-only) ---
    _unset_hip_visible_devices()
    _check_gpu_visibility()
    _check_shm_disk()

    # --- Runtime dep install ---
    # 1. Ray — needed by Magpie for task scheduling even without kernel-agent.
    if shutil.which("ray") is None:
        print("Preflight: ray not found, installing ray[default]==2.44.1 + click<8.3.0 ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "ray[default]==2.44.1", "click<8.3.0"],
            check=True,
        )
        print("Preflight: ray installed OK")

    # 2. Magpie — the benchmark engine all executors shell out to.
    # ``$MAGPIE_DIR`` is the operator override; install.sh defaults it
    # to ``$HYPERLOOM_RUNTIME_DIR/Magpie`` (= ``$USER_DATA_PATH/runtime/
    # Magpie``) so a missing Magpie auto-clones into the session tree.
    # Falls back to legacy ``/workspace/Magpie`` for environments still
    # on pre-migration launchers.
    check = subprocess.run(
        [magpie_python, "-c", "import Magpie"],
        capture_output=True,
    )
    if check.returncode != 0:
        magpie_env = os.environ.get("MAGPIE_DIR")
        magpie_env_explicit = bool(magpie_env)
        if magpie_env:
            magpie_dir = Path(magpie_env)
        else:
            from .paths import magpie_dir as _magpie_default
            magpie_dir = _magpie_default(_session_dir_resolve())
        magpie_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (magpie_dir / "setup.py").exists() and not (magpie_dir / "pyproject.toml").exists():
            # Refuse-to-clobber guard: when $MAGPIE_DIR was set explicitly by
            # the operator (e.g. a wekafs fork with un-pushed local commits),
            # cloning Magpie main on top would silently destroy local work.
            # Auto-clone only happens for the default path resolved from
            # `paths.magpie_dir(session_dir)`.
            if magpie_env_explicit:
                print(
                    f"Preflight: ERROR — $MAGPIE_DIR={magpie_dir} has no "
                    f"setup.py/pyproject.toml; refusing to clone Magpie "
                    f"main on top of an operator-supplied path. Fix the "
                    f"env or unset $MAGPIE_DIR to fall back to the "
                    f"session-default location.",
                    file=sys.stderr,
                )
                raise FileNotFoundError(
                    f"$MAGPIE_DIR={magpie_dir} is not a valid Magpie checkout"
                )
            print(f"Preflight: Magpie not importable and not found at {magpie_dir}; cloning ...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/AMD-AGI/Magpie.git", str(magpie_dir)],
                check=True,
            )
        print(f"Preflight: installing Magpie from {magpie_dir} ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "-e", str(magpie_dir)],
            check=True,
        )
        print("Preflight: Magpie installed OK")

    # 3. InferenceX + lm-eval — required for accuracy evaluation (GSM8K).
    # Magpie's benchmark scripts call `run_eval` which sources
    # InferenceX/benchmarks/benchmark_lib.sh → _install_lm_eval_deps.
    # We just ensure the InferenceX checkout exists; lm-eval deps are
    # auto-installed by benchmark_lib.sh at runtime.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "")
    if not inferencex_path:
        from .paths import (
            magpie_dir as _magpie_default,
            runtime_dir as _runtime_default,
        )
        runtime_root = _runtime_default(_session_dir_resolve())
        magpie_root = (
            Path(os.environ["MAGPIE_DIR"])
            if os.environ.get("MAGPIE_DIR")
            else _magpie_default(_session_dir_resolve())
        )
        # InferenceX detection order: Magpie's own InferenceX submodule
        # first (the canonical layout after install.sh), then the
        # standalone runtime checkout, then legacy host-level mounts so
        # existing pre-migration pods keep working.
        for candidate in (
            magpie_root / "InferenceX",
            runtime_root / "InferenceX",
            Path("/wekafs/hyperloom/InferenceX"),
            Path("/opt/hyperloom/InferenceX"),
            Path("/wekafs/fully-local/inference_optimization/InferenceX"),
        ):
            if candidate.is_dir():
                inferencex_path = str(candidate)
                break
    if inferencex_path and Path(inferencex_path).is_dir():
        os.environ.setdefault("INFERENCEX_PATH", inferencex_path)
    else:
        print("Preflight: WARNING — InferenceX not found. GSM8K accuracy "
              "eval will fail. Set INFERENCEX_PATH or clone Magpie with "
              "InferenceX submodule.")

    # --- node / claude / codex CLI presence (WARN-only) ---
    _check_node_claude_cli()

    # --- TraceLens CLI presence (HARD-FAIL; SKILL Step 2 step 8.5) ---
    # Catches brain-generated launchers that source only env.sh and skip
    # install.sh — without this gate the missing-CLI symptom would not
    # surface until robustness J3 fires at tick ~6, after baseline.
    _check_tracelens_cli()

    # --- IR-3: Cortex KB + PR Monitor reachability (soft degrade) ---
    if args is not None:
        _run_ir3_preflight(args)

    # --- Single canonical diagnostics block ---
    _emit_preflight_diagnostics(
        magpie_python=magpie_python,
        anthropic_base_url=(resolved_urls[0] if resolved_urls is not None else None),
        args=args,
    )

    return resolved_urls


def _run_ir3_preflight(args: argparse.Namespace) -> None:
    """IR-3 — Cortex KB + PR Monitor reachability probe (soft degrade).

    Mutates ``args``:
    * ``cortex_enabled`` / ``pr_monitor_enabled`` (bool) — final
      enablement after applying explicit flag + IR-3 marker.
    * ``kb_degraded_reason`` / ``pr_degraded_reason`` —
      ``None | "explicit_flag" | "ir3_auto"``.

    Never raises and never ``sys.exit``s — KB/PR outages downgrade to
    NDJSON fallback / specialist no-op, not an aborted launch.
    """
    explicit_kb = bool(getattr(args, "degraded_kb", False))
    explicit_pr = bool(getattr(args, "degraded_pr", False))

    args.cortex_enabled = True
    args.pr_monitor_enabled = True
    args.kb_degraded_reason = None
    args.pr_degraded_reason = None

    if explicit_kb and explicit_pr:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
        return

    user_data = Path(os.environ.get("USER_DATA_PATH", "/workspace/hyperloom"))
    marker_path = user_data / "runtime" / "cortex" / ".kb_preflight.json"
    script = (
        Path(__file__).resolve().parent / "scripts" / "preflight_kb.sh"
    )
    env = os.environ.copy()
    if explicit_kb:
        env["SKIP_KB_PROBE"] = "1"
    if explicit_pr:
        env["SKIP_PR_PROBE"] = "1"

    try:
        subprocess.run(
            ["bash", str(script)], env=env,
            check=False, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # The script itself died — treat both branches as unreachable
        # so soft-degrade kicks in (unless explicit flag overrides).
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False, "pr_reachable": False,
            "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False, "pr_reachable": False,
                "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
            }

    if explicit_kb:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
    elif not marker.get("kb_reachable", False) and not marker.get("kb_skipped", False):
        args.cortex_enabled = False
        args.kb_degraded_reason = "ir3_auto"

    if explicit_pr:
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
    elif not marker.get("pr_reachable", False) and not marker.get("pr_skipped", False):
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "ir3_auto"


# ---------------------------------------------------------------------------
# Default critic backend. Step D of the critic-agent integration plan flipped
# this from "mock" to "agent" once CriticAgentBackend is wired and tested.
# Override via env (operator runbook) or the --critic-mock / --critic-agent
# / --critic-codex-bare flags. Step C tests pin a specific value via the
# CLI flag so they're insulated from default drift.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND", "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent", "codex_bare")


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice.

    ``args.critic_backend`` is set by the matching CLI flag (or ``None`` if
    the operator passed nothing), in which case we fall back to
    :data:`DEFAULT_CRITIC_BACKEND`. Hard-fails on an invalid value.
    """
    chosen = args.critic_backend
    if chosen is None:
        chosen = DEFAULT_CRITIC_BACKEND
    if chosen not in _VALID_CRITIC_BACKENDS:
        print(
            f"ERROR: critic backend {chosen!r} not in {_VALID_CRITIC_BACKENDS!r} "
            f"(set by --critic-mock / --critic-agent / --critic-codex-bare or "
            f"INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen


# Default robustness backend. Production runs use the real subprocess
# transport; operators can still force the heartbeat-only mock with
# --robustness-mock or the env override below.
DEFAULT_ROBUSTNESS_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND", "agent",
)
_VALID_ROBUSTNESS_BACKENDS = ("mock", "agent")


def _resolve_robustness_choice(args: argparse.Namespace) -> str:
    """Resolve the active robustness backend choice.

    Mirrors :func:`_resolve_critic_choice`. Operator picks via
    ``--robustness-mock`` / ``--robustness-agent`` (sets
    ``args.robustness_backend``); ``None`` falls back to
    :data:`DEFAULT_ROBUSTNESS_BACKEND`. Hard-fails on an invalid value.

    Multi-node auto-downgrade: when ``args.nodes >= 2`` the
    robustness-agent's ``LocalProbeSource`` family targets
    sandbox-local resources only — ``ray status``, the inference
    server health URL, GPU / FD / disk / shm metrics, etc. On
    multi-node every one of those resources lives in a separate
    Kubernetes pod (head pod / worker pod / RayJob submitter),
    unreachable from the sandbox by design. Each probe failure
    surfaces as a HIGH-severity false positive symptom
    (``ray_head_dead``, ``local_server_unreachable``,
    ``gpu_memory_leaked``, ...) that
    drowns the bus, eats ActionLadder cooldown slots, and risks
    tripping ``escalate_strategy_change`` chains that stall
    Orchestration. Until ``robustness-agent`` grows multi-node-aware
    probe targeting, the cleanest path is to disable the real
    backend on multi-node and rely on the mock heartbeat. Operators
    who explicitly pass ``--robustness-agent`` get a WARNING and
    the auto-downgrade is honoured anyway; passing
    ``--robustness-mock`` suppresses the WARNING. Single-node
    behaviour is unchanged.
    """
    chosen = getattr(args, "robustness_backend", None)
    explicit = chosen is not None
    if chosen is None:
        chosen = DEFAULT_ROBUSTNESS_BACKEND
    if chosen not in _VALID_ROBUSTNESS_BACKENDS:
        print(
            f"ERROR: robustness backend {chosen!r} not in "
            f"{_VALID_ROBUSTNESS_BACKENDS!r} (set by --robustness-mock / "
            f"--robustness-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2 and chosen == "agent":
        if explicit:
            print(
                f"WARN: --robustness-agent selected but nodes={nodes} — "
                f"robustness-agent's LocalProbe family targets "
                f"sandbox-local resources (ray, inference server, "
                f"GPU, ...) that all live in separate pods "
                f"on multi-node and surface as HIGH false positives. "
                f"Auto-downgrading to --robustness-mock; pass "
                f"--robustness-mock explicitly to suppress this "
                f"warning. See inference_optimizer/multi_node/SKILL.md "
                f"(Robustness limitation in multi-node mode).",
                file=sys.stderr,
            )
        chosen = "mock"
    return chosen


def _build_robustness_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-default ``request.options`` overrides from CLI flags.

    Only emits keys the operator actually passed so the runtime CLI
    falls back to its own defaults / env-discovery for the rest.

    Multi-node auto-disable: when ``args.nodes >= 2`` the inference
    server runs in the head pod (separate Kubernetes pod from the
    sandbox where the robustness-agent subprocess lives), so the
    hardcoded ``http://127.0.0.1:8888/health`` probe baked into
    ``robustness_agent.config.Config.auto_probe_inference_server``
    can never succeed and floods the bus with false-positive
    ``local_server_unreachable`` symptoms each tick. Disable the
    auto-probe by default in multi-node so single-node semantics stay
    intact while multi-node stops emitting the bogus alert. Operators
    who configure an explicit cluster-local probe target can re-enable
    via a future ``--robustness-auto-probe-inference-server`` flag.
    """
    options: dict[str, Any] = {}
    server_url = getattr(args, "robustness_server_url", None)
    if server_url is not None:
        options["robustness_server_url"] = server_url
    llm_rca = getattr(args, "robustness_llm_rca", None)
    if llm_rca is not None:
        options["llm_rca_enabled"] = bool(llm_rca)
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2:
        options["auto_probe_inference_server"] = False
        # B3 no_levers_found floor — multi-node large-model spends
        # 35-50 min on sglang cold start + baseline + profile +
        # turnaround alone before the first explore family runs.
        # Bumping the elapsed-time floor from 45 to 60 minutes layers
        # a wall-clock buffer on top of the explore_started gate
        # (commit 97318ee) so the symptom only fires when both
        # exploration has started AND a full hour has elapsed
        # without finding a lever. Single-node default (45.0) stays
        # untouched.
        options["progress_no_levers_min_minutes"] = 60.0
    return options


# ---------------------------------------------------------------------------
# Cortex KB T0 hook
# ---------------------------------------------------------------------------
def _bootstrap_cortex_kb(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
) -> CortexKBClient:
    """Construct the :class:`CortexKBClient` and run the T0 anchor.

    T0 (PRELUDE entry) sequence per 
    1. ``session begin`` (sync, must succeed unless ``--degraded-kb``).
    2. ``propose-point workload_node`` (canonical: ``workload.<model>.<hw>``)
       — idempotent across sessions for the same pair.
    3. ``find-recipe`` snapshot to ``.kb_warm.json`` (M5 will consume).
    4. ``traps`` snapshot to ``.kb_pitfalls.json`` (M5).
    5. Persist sid to SharedState + ``.kb_sid``.

    Resume rules (M1 §7): if ``.kb_sid`` exists, reuse the sid without
    re-begin. The other T0 steps still run so the warm-start snapshots
    stay fresh for M5 consumers.

    Failure handling:

    - ``--degraded-kb`` → returns a disabled client; never raises.
    - sync ``session begin`` failure → ``sys.exit(2)``; resume can pick
      up the partial session_dir once Cortex comes back.
    - propose_point / find_recipe / traps failures fall through to NDJSON
      / warning; PRELUDE proceeds.

    Returns the constructed (possibly disabled) :class:`CortexKBClient`
    so the caller can thread it into the Coordinator.
    """
    enabled = bool(getattr(args, "cortex_enabled", True))
    kb_url = (getattr(args, "cortex_kb_url", None) or "").strip() or None
    client = CortexKBClient(
        session_dir=session_dir,
        kb_url=kb_url,
        enabled=enabled,
    )
    if not enabled:
        print("Cortex KB        : DISABLED (--degraded-kb)")
        return client

    # the cli is the *canonical* T0 entry point
    # (fail-fast banner + sys.exit on Cortex outage), but the actual
    # T0 ritual lives in :mod:`orchestrator.cortex_t0` so an SDK /
    # integration-test caller that constructs the Coordinator
    # directly can run the same sequence as a defensive fallback
    # (see :meth:`Coordinator._ensure_cortex_t0_anchored`).
    state = SharedState.load_or_init(session_dir)
    workload = (
        state.model_name
        or manifest.get("model_name", "")
        or Path(manifest.get("model_path", "") or "").name
        or "unknown_model"
    )
    hw = state.gpu_type or manifest.get("gpu_type", "") or "unknown_gpu"
    stack_fp = manifest.get("stack_fingerprint") or {}
    image_digest = manifest.get("image") or ""
    extra_attrs = {
        "marathon_dispatch_id": manifest.get("session_id", ""),
        "framework":            state.framework or manifest.get("framework", ""),
        "model_class":          state.model_class or "",
        "claw_session_id":      manifest.get("claw_session_id") or "",
    }
    try:
        run_t0_anchor(
            client,
            state,
            workload=workload,
            hw=hw,
            image_digest=image_digest,
            stack_fingerprint=stack_fp,
            extra_attrs=extra_attrs,
            resume=resume,
            fail_fast=True,
            on_status=print,
            session_dir=session_dir,
            save_state=True,
        )
    except CortexKBError as exc:
        print(
            f"ERROR: T0 Cortex `session begin` failed: {exc}\n"
            f"This is fail-fast per KB_design §3.13 M1. Pass "
            f"--degraded-kb to skip KB integration this run.",
            file=sys.stderr,
        )
        sys.exit(2)
    return client


# ---------------------------------------------------------------------------
# v0.8 KB_gaps/Dead-E — Cortex KB flusher daemon lifecycle
# ---------------------------------------------------------------------------
def _maybe_spawn_kb_flusher(
    args: argparse.Namespace,
    *,
    session_dir: Path,
) -> tuple[subprocess.Popen | None, Path]:
    """Spawn ``scripts.cortex_kb_flusher`` for this session and return the
    handle + pid path.

    KB_design §3.6 / §3.14 R-01/R-02 require a background NDJSON drainer
    so the main loop never blocks on Cortex outages. Pre-Dead-E the
    daemon code existed but the cli never launched it; this helper is
    the missing link.

    Returns ``(None, pid_path)`` when spawn is skipped (``--degraded-kb``,
    ``--no-kb-flusher``, or a healthy prior daemon is still bound to
    ``pid_path``). A status marker is always written so the breakdown
    collector can surface the boot-time decision.
    """
    from .session_paths import (
        cortex_dir,
        cortex_flusher_pid,
        cortex_flusher_status_json,
    )

    pid_path = cortex_flusher_pid(session_dir)
    status_path = cortex_flusher_status_json(session_dir)
    cortex_root = cortex_dir(session_dir)
    cortex_root.mkdir(parents=True, exist_ok=True)

    cortex_enabled = bool(getattr(args, "cortex_enabled", True))
    flusher_enabled = bool(getattr(args, "kb_flusher_enabled", True))
    interval_sec = float(getattr(args, "kb_flusher_interval_sec", 5.0) or 5.0)
    batch_size = int(getattr(args, "kb_flusher_batch_size", 50) or 50)
    cortex_kb_url = (getattr(args, "cortex_kb_url", None) or "").strip() or None

    def _write_status(
        *,
        spawned: bool,
        pid: int | None,
        cmd: list[str],
        reason: str,
    ) -> None:
        payload = {
            "enabled":       cortex_enabled and flusher_enabled,
            "spawned":       spawned,
            "pid":           pid,
            "cmd":           cmd,
            "cortex_kb_url": cortex_kb_url,
            "interval_sec":  interval_sec,
            "batch_size":    batch_size,
            "reason":        reason,
            "ts":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pid_path":      str(pid_path),
        }
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("kb_flusher status marker write failed: %s", exc)

    if not cortex_enabled:
        _write_status(spawned=False, pid=None, cmd=[], reason="cortex_disabled")
        print("Cortex KB flusher: SKIPPED (--degraded-kb)")
        return None, pid_path
    if not flusher_enabled:
        _write_status(spawned=False, pid=None, cmd=[], reason="flag_disabled")
        print("Cortex KB flusher: SKIPPED (--no-kb-flusher)")
        return None, pid_path

    if pid_path.exists():
        try:
            prior = int(pid_path.read_text(encoding="utf-8").strip().splitlines()[0])
            os.kill(prior, 0)
            _write_status(
                spawned=False, pid=prior, cmd=[],
                reason=f"prior_alive_pid={prior}",
            )
            print(f"Cortex KB flusher: REUSED (existing daemon pid={prior})")
            return None, pid_path
        except (OSError, ValueError, IndexError):
            try:
                pid_path.unlink(missing_ok=True)
            except OSError:
                pass

    cmd: list[str] = [
        sys.executable, "-m", "inference_optimizer.scripts.cortex_kb_flusher",
        "--session-dir", str(session_dir),
        "--interval-sec", str(interval_sec),
    ]
    if cortex_kb_url:
        cmd.extend(["--cortex-kb-url", cortex_kb_url])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError as exc:
        log.warning("kb_flusher spawn failed: %s", exc)
        _write_status(
            spawned=False, pid=None, cmd=cmd,
            reason=f"spawn_failed:{exc!r}"[:240],
        )
        print(f"Cortex KB flusher: SPAWN FAILED ({exc!r})")
        return None, pid_path

    _write_status(spawned=True, pid=proc.pid, cmd=cmd, reason="spawned")
    print(
        f"Cortex KB flusher: SPAWNED pid={proc.pid} "
        f"interval={interval_sec}s batch={batch_size}"
    )
    return proc, pid_path


def _stop_kb_flusher(
    proc: subprocess.Popen | None,
    pid_path: Path,
    *,
    grace_sec: float = 10.0,
) -> None:
    """Graceful shutdown for the flusher daemon spawned by
    :func:`_maybe_spawn_kb_flusher`. Best-effort: never raises.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        log.warning(
            "kb_flusher did not exit within %.1fs of SIGTERM; killing pid=%d",
            grace_sec, proc.pid,
        )
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("kb_flusher kill failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("kb_flusher graceful shutdown failed: %s", exc)
    finally:
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# PR Monitor + KnowledgePlane wiring
# ---------------------------------------------------------------------------
def _bootstrap_knowledge_plane(
    args: argparse.Namespace,
    *,
    cortex_client: "CortexKBClient | None",
    session_dir: Path | None = None,
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade for one session.

    Wires the (optional) PR Monitor REST client + the (already
    bootstrapped) Cortex KB client into a single read/write surface.
    Both backends fail-soft: if either is unreachable the facade
    returns empty / disabled-status responses so prompt assembly +
    breakdown collectors don't have to branch on availability.

    ``--degraded-pr`` short-circuits the REST probe and yields a
    disabled :class:`PRMonitorClient` (which also strips the
    ``mcp__pr_monitor__*`` tool block on the specialist side via
    :meth:`SpecialistRunner._resolve_tools`).
    """
    from .orchestrator.knowledge_plane import (
        KnowledgePlane,
        load_domain_repos,
    )
    from .orchestrator.pr_monitor import (
        DEFAULT_PR_FEED_WINDOW_DAYS,
        DEFAULT_PR_MONITOR_MCP_URL,
        PRMonitorClient,
    )

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_url = (getattr(args, "pr_monitor_url", None) or "").strip() or None
    pr_mcp_url = (
        (getattr(args, "pr_monitor_mcp_url", None) or "").strip()
        or DEFAULT_PR_MONITOR_MCP_URL
    )
    window_days = int(
        getattr(args, "pr_feed_window_days", DEFAULT_PR_FEED_WINDOW_DAYS)
        or DEFAULT_PR_FEED_WINDOW_DAYS
    )

    pr_client = PRMonitorClient.from_args(url=pr_url, enabled=pr_enabled)
    # IR-3 (``_preflight()`` step 13) already probed ``/healthz`` and
    # set ``args.pr_monitor_enabled`` + ``args.pr_degraded_reason``.
    # We trust that result here — runtime drift falls back to the
    # specialist-side empty PR feed surface.
    if not pr_enabled:
        reason = getattr(args, "pr_degraded_reason", None) or "explicit_flag"
        status_text = f"disabled ({reason})"
        print(f"PR Monitor       : DISABLED ({reason})")
        pr_reachable = False
    else:
        status_text = f"REST {pr_client.base_url} (window={window_days}d)"
        print(
            f"PR Monitor       : REST {pr_client.base_url} (window="
            f"{window_days}d, mcp={pr_mcp_url})"
        )
        pr_reachable = True

    # v0.8 §3.6 + record a one-shot status marker so
    # ``breakdown.warnings`` can surface ``pr_monitor:disabled`` /
    # ``pr_monitor:unreachable`` without scraping logs. Best-effort: a
    # write failure here only loses the breakdown row, not the
    # KnowledgePlane bootstrap itself.
    if session_dir is not None:
        try:
            from .session_paths import pr_monitor_status_json
            from .paths import asset_actions_dir  # noqa: F401 (unused import warning suppress)
            marker = pr_monitor_status_json(session_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "enabled":      bool(pr_enabled),
                "url":          (pr_client.base_url if pr_enabled else ""),
                "reachable":    bool(pr_reachable),
                "mcp_url":      pr_mcp_url if pr_enabled else "",
                "window_days":  int(window_days),
                "status_text":  status_text,
            }, sort_keys=True, indent=2))
        except OSError as exc:  # noqa: BLE001 — defensive
            log.warning(
                "pr_monitor_status marker write failed: %r "
                "(breakdown.warnings will miss pr_monitor row)", exc,
            )

    return KnowledgePlane.from_clients(
        cortex_kb=cortex_client,
        pr_monitor=pr_client,
        domain_repos=load_domain_repos(),
        pr_feed_window_days=window_days,
        pr_monitor_mcp_url=pr_mcp_url,
    )


def _reset_state_file(session_dir: Path) -> None:
    """v0.8 §3.10 — back up the existing ``state.json`` and start fresh.

    The backup name is ``state.json.preReset.<unix_ts>`` so multiple
    resets in the same session_dir don't clobber each other. Symbolic
    of the operator's nuclear option:
    the Cortex KB cross-session knowledge is *not* touched here — only
    the per-session fact-layer is reset. ``Coordinator`` will reseed
    its dataclass defaults on the next ``load_or_init`` call.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return
    import time as _time
    ts = int(_time.time())
    backup_path = session_dir / f"state.json.preReset.{ts}"
    try:
        state_path.replace(backup_path)
    except OSError as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "v0.8 §3.10 --reset-state: could not move %s → %s: %s",
            state_path, backup_path, exc,
        )
        return
    import logging as _logging
    _logging.getLogger(__name__).info(
        "v0.8 §3.10 --reset-state: backed up state.json to %s; "
        "session starts blank.", backup_path.name,
    )


def _gc_old_profile_traces(
    root: str | None = None,
    retention_days: int = 7,
    keep: str | None = None,
) -> None:
    """Best-effort GC of stale per-RayJob profile-trace dirs on the shared FS.

    Removes only top-level subdirectories whose mtime is older than
    ``retention_days``. The active session's dir (just mkdir'd seconds ago)
    is always young enough; ``keep`` adds an explicit name-match guard so
    the current run is never collected even if a clock skew flipped mtime.

    ``root`` defaults to :func:`mn_profile_trace_root` (anchored on
    ``$USER_DATA_PATH``); callers may pass an explicit override for
    tests or migration scenarios.

    Failure is logged + swallowed: GC must never block optimizer startup.

    Override knobs (env):
      HYPERLOOM_MN_TRACE_RETENTION_DAYS -- int days, default 7
      HYPERLOOM_MN_TRACE_GC_DISABLE     -- "1" disables GC entirely
    """
    if os.environ.get("HYPERLOOM_MN_TRACE_GC_DISABLE", "").strip() in (
        "1", "true", "yes",
    ):
        return
    try:
        retention_days = int(
            os.environ.get("HYPERLOOM_MN_TRACE_RETENTION_DAYS") or retention_days
        )
    except ValueError:
        retention_days = 7
    base = Path(root) if root is not None else mn_profile_trace_root()
    if not base.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    keep_name = Path(keep).name if keep else ""
    removed = 0
    kept = 0
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if keep_name and child.name == keep_name:
                kept += 1
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                kept += 1
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                print(
                    f"WARN multi-node GC: failed to rm {child}: {exc}",
                    file=sys.stderr,
                )
    except OSError as exc:
        print(f"WARN multi-node GC: scan failed under {base}: {exc}", file=sys.stderr)
        return
    if removed or kept:
        print(
            f"multi-node: GC profile-traces removed={removed} kept={kept} "
            f"retention={retention_days}d root={base}"
        )


def _provision_multi_node_rayjob_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2``, create/reuse SaFE RayJob, bootstrap once, export RAY_ADDRESS."""
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    from .multi_node.cli import cmd_bootstrap, cmd_create_rayjob, _load_state
    from .orchestrator.action_executors._multi_node_env import export_ray_address_to_os

    state_path = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))
    image = (
        (getattr(args, "rayjob_image", None) or "").strip()
        or os.environ.get("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", "").strip()
    )
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 requires a RayJob container image. Pass "
            "--rayjob-image <harbor/...> or set INFERENCE_OPTIMIZER_RAYJOB_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "rayjob_gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # Forward agent-supplied prompt env (verbatim, no defaults). When the
    # state file already has a non-terminal rayjob_id, cmd_create_rayjob
    # reuses it and skips POST CreateWorkload -- so passing extra_env on
    # a reuse call is a no-op (the original create-time env stays in
    # effect). To inject env into an existing RayJob the caller must
    # `stop-rayjob --clear-state` first or invoke create-rayjob with
    # --recreate. See multi_node/SKILL.md.
    rayjob_extra_env = list(getattr(args, "rayjob_extra_env", None) or [])

    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        display_name=None,
        description=None,
        owner_id=None,
        extra_env=rayjob_extra_env,
        extra_label=[],
        no_wait=False,
        recreate=False,
        poll_interval=6,
        poll_timeout=int(
            os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
        ),
    )
    rc = cmd_create_rayjob(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    if not state.get("last_bootstrap_submission_id"):
        ns_boot = argparse.Namespace(
            script=None,
            force=False,
            print_logs=False,
            poll_interval=6,
            poll_timeout=int(
                os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
            ),
        )
        rc_boot = cmd_bootstrap(ns_boot)
        if rc_boot != 0:
            sys.exit(rc_boot)

    export_ray_address_to_os()
    ra = os.environ.get("RAY_ADDRESS", "")
    if ra:
        print(f"multi-node: exported RAY_ADDRESS={ra} for kernel-agent Ray tasks")

    # Multi-node only: server-side sglang/vllm pods (head + worker) must
    # write torch traces to a path that the sandbox-side profile_executor
    # can read. Per-pod /tmp is invisible to the sandbox; wekafs is the
    # only fs both sides mount. Namespace the dir by ``rayjob_id`` so a
    # restart of the same RayJob reuses the directory and a new RayJob
    # gets a fresh one. This env is consumed by:
    #   * multi_node.cli._build_multinode_launch_entrypoint -> --torch-profiler-dir
    #   * orchestrator.action_executors.profile.ProfileExecutor.__call__
    state_after = _load_state()
    rid = (state_after.get("rayjob_id") or "").strip()
    if rid:
        # Anchor torch-profile shared root on $USER_DATA_PATH so the
        # operator only has to point one knob at a cluster-shared
        # filesystem (e.g. wekafs); the sandbox and the RayJob pods then
        # both see traces under the same path. See
        # ``inference_optimizer.paths.mn_profile_trace_root`` docstring
        # for the multi-node USER_DATA_PATH caveat.
        trace_root_path = mn_profile_trace_root() / rid / "torch_trace"
        trace_root = str(trace_root_path)
        try:
            trace_root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"WARN multi-node: cannot mkdir {trace_root}: {exc}; "
                f"server traces will fall back to per-pod /tmp",
                file=sys.stderr,
            )
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = trace_root
            print(
                f"multi-node: exported HYPERLOOM_MN_PROFILE_TRACE_DIR={trace_root}"
            )
            # Best-effort GC of older sibling RayJob trace dirs. Runs only
            # AFTER the current session's dir is mkdir'd, so the active
            # rayjob_id is name-guarded and its mtime is fresh.
            _gc_old_profile_traces(keep=rid)

    # RayJob recreate path: when an existing session's RayJob was killed
    # (OOM, manual recreate, SaFE rescheduling) and we provisioned a
    # fresh one above, any kernel-agent patches previously applied to
    # the OLD pods are gone from the new pods' filesystems (the pod
    # local fs is destroyed with the old pod). The sandbox-side
    # SharedState.optimization_stack still records every promoted patch
    # though, so we replay them here before any executor's first
    # restart_server_for_round. Best-effort; failures degrade to
    # warnings — the orchestrator will notice missing speedups in the
    # next baseline and re-run kernel-agent on the affected kernels.
    _replay_kernel_patches_for_multi_node(args)


def _replay_kernel_patches_for_multi_node(args: argparse.Namespace) -> None:
    """Replay every applied kernel-agent patch onto the (possibly new)
    RayJob pods.

    Scans the session's ``kernel-agent-workspace`` tree for manifest
    files with ``status="applied"`` and a ``multinode`` block, then
    invokes ``python3 -m inference_optimizer.multi_node apply-patch``
    once per patch. The fan-out is idempotent — re-running apply on
    pods that already have the new file produces a fresh backup of
    the (now-current) source and overwrites with the same bytes.

    Run only when ``--nodes >= 2``. Best-effort: stdout/stderr from
    each replay is mirrored verbatim; per-patch failures emit a
    warning but do not raise so a single broken patch doesn't block
    other replays or the subsequent optimize loop.
    """
    nodes = max(1, int(getattr(args, "nodes", 1) or 1))
    if nodes < 2:
        return
    session_dir = _session_dir_resolve()
    workspace_root = session_dir / "kernel-agent-workspace"
    if not workspace_root.is_dir():
        return

    manifests: list[Path] = sorted(workspace_root.rglob("manifest.json"))
    if not manifests:
        return

    replayed = 0
    skipped = 0
    failed = 0
    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN multi-node patch replay: skipping unreadable "
                f"manifest {mpath}: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if str(data.get("status", "")).lower() != "applied":
            continue
        mn = data.get("multinode") or {}
        if not mn:
            continue
        target_file = data.get("target_file") or ""
        patch_path = data.get("patch_path") or ""
        kernel_id = data.get("kernel_id") or ""
        backup_dir_on_pod = mn.get("backup_dir_on_pod") or "/var/kernel_patch_backups"
        if not target_file or not patch_path:
            skipped += 1
            continue
        if not Path(patch_path).is_file():
            print(
                f"WARN multi-node patch replay: source patch missing for "
                f"{target_file} (manifest={mpath} patch_path={patch_path}); "
                f"skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cmd = [
            sys.executable, "-m", "inference_optimizer.multi_node",
            "apply-patch",
            "--patch-file", str(patch_path),
            "--target-path", str(target_file),
            "--backup-dir", str(backup_dir_on_pod),
            "--kernel-id", str(kernel_id),
        ]
        print(
            f"multi-node patch replay: target={target_file} kernel_id={kernel_id!r} "
            f"(from {mpath})",
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            failed += 1
            print(
                f"WARN multi-node patch replay failed for {target_file} "
                f"rc={proc.returncode} stderr={(proc.stderr or '')[-1000:]!r}",
                file=sys.stderr,
            )
            continue
        replayed += 1
    if replayed or failed or skipped:
        print(
            f"multi-node patch replay: applied={replayed} "
            f"skipped={skipped} failed={failed} "
            f"(scanned {len(manifests)} manifest(s) under {workspace_root})"
        )


async def _run_optimize(args: argparse.Namespace) -> int:
    # Surface --nodes to the rest of the process (preflight diagnostics
    # and any executor that wants to short-circuit when the optimizer is
    # in single-node mode) by exporting it before _preflight runs. We
    # re-export even when the env var was already set so the CLI flag
    # always wins, matching the documented resolution order.
    nodes_resolved = max(1, int(args.nodes))
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    # Resolve gpus_per_node with the same priority chain
    # `_provision_multi_node_rayjob_stack` uses (CLI > env > 8) so the
    # validation here matches what SaFE actually ends up seeing.
    gpn_attr = getattr(args, "rayjob_gpus_per_node", None)
    if gpn_attr is not None:
        gpus_per_node_resolved = int(gpn_attr)
    else:
        try:
            gpus_per_node_resolved = int(
                os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8,
            )
        except ValueError:
            gpus_per_node_resolved = 8
    total_gpus = nodes_resolved * gpus_per_node_resolved

    # Topology sanity gates — multi-node ONLY (nodes >= 2). Single-pod
    # runs keep their legacy code paths untouched (Magpie owns the
    # server lifecycle there, and the TP/EP knobs flow through env
    # rather than the multi_node CLI). Rejected at CLI parse time so
    # the agent gets an immediate, attributable error instead of a
    # cryptic sglang/vllm launcher crash 30 minutes into a cold start.
    if nodes_resolved >= 2:
        # Gate 1: total cluster GPUs must hold the model's TP shards.
        #   nodes * gpus_per_node >= tp
        # Anything less means at least one TP rank has no GPU to land on.
        if total_gpus < tp_resolved:
            print(
                f"ERROR: TP={tp_resolved} exceeds total GPU count "
                f"({nodes_resolved} nodes * {gpus_per_node_resolved} "
                f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
                "--nodes, or use a larger --rayjob-gpus-per-node pod "
                "template.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Gate 2: expert-parallel size cannot exceed TP — sglang/vllm
        # cannot place more expert shards than ranks. Helper repeats
        # this check at server-restart time, but failing here keeps
        # the agent from spending bootstrap minutes on an unrunnable
        # topology.
        if ep_resolved > tp_resolved:
            print(
                f"ERROR: EP={ep_resolved} > TP={tp_resolved}. Expert-parallel "
                "size must be <= tensor-parallel size. Either lower --ep or "
                "raise --tp.",
                file=sys.stderr,
            )
            sys.exit(2)

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Re-export $TP / $CONC / $EP from the resolved CLI args, but ONLY in
    # multi-node mode. Rationale: main's single-node design carries TP /
    # CONC via `benchmark.envs.TP` / `benchmark.envs.CONC` in the YAML
    # config (see SKILL.md §"Before a new model run"). Writing the
    # argparse defaults ("1" / "8" / "1") back into os.environ here
    # would silently override those YAML values via
    # `_workload_envs.apply_runtime_benchmark_overrides` (which prefers
    # env over envs.* for these keys). The multi-node orchestrator
    # subprocess + sweep child workers still need to see the resolved
    # CLI values, so the env export stays for nodes>=2 runs.
    if nodes_resolved >= 2:
        os.environ["TP"] = str(tp_resolved)
        os.environ["CONC"] = str(max(1, int(getattr(args, "conc", 8) or 8)))
        os.environ["EP"] = str(ep_resolved)
    # User-declared grid skip list. Resolution order is already enforced
    # by argparse default (--skip-variants > $SKIP_VARIANTS); we re-export
    # so executors started later via subprocess (multi-node orchestrator,
    # sweep child workers) inherit the same spec without re-parsing argv.
    # Empty string is intentional: it clears any stale value left by a
    # prior session in the same shell.
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs the same way for executors / helper. Empty
    # string means "let helper resolve from state.json"; pd_mode
    # always exported so colocated runs explicitly clear any stale
    # PD_MODE the operator may have left set.
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    # nodes_resolved is already computed above (TP/EP gates).
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation logically requires at least one prefill pod
        # and one decode pod; with NODES=1 that would mean co-hosting
        # both server roles on the same GPU set, which defeats the
        # latency-isolation purpose and is not supported by sglang/vllm
        # in the current architecture (also rejected later by
        # _resolve_pd_args, but we fail at CLI parse so the agent gets
        # an immediate, attributable error).
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to colocated) or "
            "raise --nodes.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ["PD_MODE"] = pd_mode
    if pd_mode == "disaggregated":
        for cli_attr, env_key in (
            ("pd_prefill_nodes", "PD_PREFILL_NODES"),
            ("pd_decode_nodes", "PD_DECODE_NODES"),
            ("pd_prefill_tp", "PD_PREFILL_TP"),
            ("pd_decode_tp", "PD_DECODE_TP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v

    await asyncio.to_thread(_provision_multi_node_rayjob_stack, args)

    resolved_urls = _preflight(args)

    # Hard-gate Claude model BEFORE any session work. Mutates args.claude_model
    # in-place when falling back to opus-4-6; aborts with sys.exit(2) if the
    # gateway catalog cannot be probed or neither allowed model is present.
    catalog_ids = _validate_and_resolve_claude_model(args, resolved_urls)
    _smoke_test_codex_model(args, catalog_ids)

    # `--resume-from <path>` implies `--resume` (operator convenience).
    if args.resume_from and not args.resume:
        args.resume = True

    if args.resume:
        # Resume mode (N17-aware): USER_DATA_PATH stays at workspace
        # level so runtime/ + logs/ resolution doesn't break (N15
        # `_load_kernel_agent_env_fallback` looks at $USER_DATA_PATH/
        # runtime/kernel-agent.env.sh, which only exists at workspace
        # level after install.sh ran). We then pick the per-session
        # subdir to resume from via either:
        #
        # * --resume-from <path>  : explicit operator choice
        # * --resume alone        : auto-pick LATEST per-session subdir
        #                            under workspace_root/<model>/<ts>/.
        #                            Falls back to workspace_root itself
        #                            (legacy flat layout) when no
        #                            per-session subdir exists.
        #
        # We pin INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to the picked
        # subdir so every paths.session_dir() call later (and every
        # subprocess that inherits env) resolves consistently.
        from .paths import (
            ENV_CURRENT_SESSION_DIR, find_latest_per_session_dir,
            workspace_root,
        )
        ws = workspace_root()
        if args.resume_from:
            session_dir = Path(args.resume_from).expanduser().resolve()
            try:
                session_dir.relative_to(ws.resolve())
            except ValueError:
                print(
                    f"ERROR: --resume-from {session_dir!r} is not under "
                    f"$USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
                    f"workspace root (the parent of the per-session subdirs) "
                    f"and pass the per-session subdir via --resume-from.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not session_dir.is_dir():
                print(
                    f"ERROR: --resume-from {session_dir!r} does not exist.",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            picked = find_latest_per_session_dir()
            if picked is not None:
                session_dir = picked
                print(f"  --resume: auto-picked latest per-session subdir")
            else:
                # Legacy flat layout — workspace_root itself is the
                # session_dir. Validate it has manifest.json + state.json
                # the same way the per-session branch does below.
                session_dir = ws
                print(
                    f"  --resume: no per-session subdir found under "
                    f"{ws}/<model>/<ts>/; falling back to flat layout "
                    f"({ws})"
                )
        # Pin so every paths.session_dir() / subprocess inherits the
        # resolved location BEFORE Coordinator/SharedState load.
        os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
        # Make sure per-session skeleton exists (mkdir -p semantics,
        # idempotent; doesn't disturb existing files).
        for sub in __import__(
            "inference_optimizer.paths", fromlist=["_SESSION_SKELETON"]
        )._SESSION_SKELETON:
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        try:
            manifest = load_manifest(session_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --resume failed: {exc}", file=sys.stderr)
            sys.exit(2)
        if not (session_dir / "state.json").exists():
            print(
                f"ERROR: --resume failed: {session_dir}/state.json missing "
                f"(manifest exists but Coordinator never wrote SharedState)",
                file=sys.stderr,
            )
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  manifest.session_id    : {manifest.get('session_id')}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(f"  prior current_best    : "
              f"{(state.current_best or {}).get('action')}/"
              f"{(state.current_best or {}).get('tput')}")
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")

        # Re-export session-level env vars from persisted state so the
        # executors (baseline / profile / sweep / backends / params) resolve
        # model / framework / gpu_type correctly. Without this, a resume
        # in a fresh shell would fall back to YAML hardcoded defaults,
        # potentially benchmarking the wrong model on the wrong framework.
        if state.model_path:
            os.environ["MODEL_PATH"] = state.model_path
            print(f"  re-exported MODEL_PATH: {state.model_path}")
        if state.framework:
            os.environ["FRAMEWORK"] = state.framework
            print(f"  re-exported FRAMEWORK : {state.framework}")
        if state.gpu_type:
            runner_gpu_type = _gpu_runner_type(state.gpu_type)
            os.environ["TARGET_GPU_TYPE"] = state.gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
            if runner_gpu_type != state.gpu_type:
                print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
        # Re-export workload metadata (TP / CONC / ISL / OSL /
        # MAX_MODEL_LEN / PRECISION) from SharedState so a resume in
        # a fresh shell sees the same workload contract baseline ran
        # under. Without this, the resumed shell's executors fall
        # back to YAML defaults (TP=1, CONC=8, ISL=256, OSL=256) and
        # the specialist prompt builder renders TP=1 (which makes
        # comm_specialist self-veto on TP=8 sessions).
        for state_attr, env_name in (
            ("tp", "TP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("max_model_len", "MAX_MODEL_LEN"),
        ):
            val = getattr(state, state_attr, 0) or 0
            if val:
                os.environ[env_name] = str(val)
                print(f"  re-exported {env_name:<14s}: {val}")
        if state.precision:
            os.environ["PRECISION"] = state.precision
            print(f"  re-exported PRECISION     : {state.precision}")
        # Honour persisted kernel_enabled flag on resume; CLI --no-kernel
        # can still override on a previously-enabled session.
        if not state.kernel_enabled:
            args.no_kernel = True
            print("  kernel agent          : DISABLED (persisted from original run)")
        # Same persistence contract for the FRAMEWORK_PR phase toggle.
        if not bool(getattr(state, "framework_phase_enabled", True)):
            args.no_framework = True
            print("  framework phase       : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_framework", False)):
            # Inverse direction (P2.d): persisted state still has the
            # phase enabled, but the operator passed ``--no-framework``
            # on resume. Only honour this when the original session has
            # not yet entered FRAMEWORK_PR (or anything past it) —
            # retroactively skipping a phase we are in/past is
            # incoherent.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE"):
                state.framework_phase_enabled = False
                print(
                    "  framework phase       : DISABLING for resume "
                    "(--no-framework + phase=PRELUDE)"
                )
            else:
                print(
                    f"  framework phase       : WARN --no-framework ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )

        # CRITICAL: a leftover stop_reason from the prior run (most often
        # "time_exhausted") fools Orchestration into thinking the work is
        # already done — it just heartbeats forever. Clear it so the new
        # run has a clean signal. The Coordinator's run() always re-sets
        # stop_reason at exit anyway.
        prior_crash = state.crash_count

        # Issue-G (Saturday May 2026): the SKILL.md "Run-time signals"
        # section says ``no_more_leverage`` and ``target_reached`` are
        # intentional terminal states — "only resume if the user
        # changes workload / search space / model / strategy." The
        # original auto-clear silently nuked both, which is how an
        # over-eager monitor or a habitual ``--resume`` would push a
        # session past a steward verdict the operator never reviewed.
        # We now require ``--force-resume`` to push past those two
        # vocab terms; everything else (time_exhausted, max_ticks,
        # crash recovery) auto-clears as before.
        force_resume = bool(getattr(args, "force_resume", False))
        gated_terminal = {"no_more_leverage", "target_reached"}
        if prior_stop in gated_terminal and not force_resume:
            print(
                f"\nERROR: --resume blocked by terminal stop_reason="
                f"{prior_stop!r}.\n"
                f"\n"
                f"  SKILL.md (Run-time signals): {prior_stop!r} is a "
                f"deliberate terminal state.\n"
                f"  The optimizer will not auto-resume past it because "
                f"the prior run\n"
                f"  declared exhaustion — picking up where it left off "
                f"only repeats\n"
                f"  the same exhaustion verdict.\n"
                f"\n"
                f"  Override paths:\n"
                f"  1. Pass ``--force-resume`` if you have changed the "
                f"workload /\n"
                f"     search space / model / strategy and want to "
                f"continue regardless.\n"
                f"  2. Start a fresh session (different "
                f"$USER_DATA_PATH) for a clean run.\n"
                f"\n"
                f"  Reports for the prior run live under "
                f"{session_dir}/reports/.\n",
                file=sys.stderr,
            )
            sys.exit(2)

        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            state.closing_phase = False
            state.closing_started_unix = 0.0
            state.closing_report_task_id = ""
            # Reset persisted crash_count so a fresh resume isn't immediately
            # tripped into "emergency" by accumulated failures from prior runs
            # (e.g. authentication errors before .env was loaded).
            state.crash_count = 0
            # Reset start_ts to "now" so the elapsed_minutes value the
            # orchestration agent sees in its prompt reflects this resume's
            # budget rather than the original session start; otherwise an
            # old session that exhausted its budget will look "already
            # over budget" on resume and the LLM will refuse to propose
            # any work.
            from datetime import datetime, timezone
            state.start_ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            state.save(session_dir)
            override_note = (
                " (--force-resume override)"
                if force_resume and prior_stop in gated_terminal
                else ""
            )
            print(
                f"  → cleared stop_reason and reset crash_count "
                f"(was {prior_crash}) for fresh resume{override_note}"
            )
            print(f"  → reset start_ts to {state.start_ts} (resume budget)")
        # re-bootstrap (or pick up existing) Cortex KB session.
        # Same call as the fresh-session branch; the resume rules inside
        # ``_bootstrap_cortex_kb`` (.kb_sid + state.cortex_session_id)
        # decide whether to begin a new session or reuse the prior one.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=True,
        )
        kb_flusher_proc, kb_flusher_pid_path = _maybe_spawn_kb_flusher(
            args, session_dir=session_dir,
        )
        # KnowledgePlane facade. Bootstrapped
        # alongside the cortex client so a resumed session also gets the
        # PR Monitor + KB readonly tools wired into specialist dispatch.
        # Returns a plane that fail-soft degrades when PR Monitor or
        # Cortex is unreachable; specialists then see empty pr_feed /
        # kb_subgraph and continue. ``None`` only when --degraded-kb.
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )
        # Note: main commit 8e69732 also adds an N31 "resume backfill"
        # block that copies last_trace_analyze into
        # last_trace_analyze_baseline. The baseline-freeze field is
        # main's N31 final-roofline machinery, which F3-3 retired on
        # this branch in favour of gain-only freshness; the backfill
        # therefore has no live consumer and is omitted.
    else:
        # Resolve model path from --model first, then $MODEL_PATH env. Without
        # either, fail fast: silently falling back to the YAML's hardcoded
        # `/wekafs/models/Qwen-Qwen3-8B` was the cause of "the optimizer ran
        # the wrong model" reports — explicit > implicit.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume to continue an existing "
                "session at the canonical session_dir).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export the resolved value so downstream subprocess executors
        # (baseline / profile / sweep / backends / params) inject it into
        # the Magpie YAML instead of trusting the YAML's hardcoded `model:`.
        os.environ["MODEL_PATH"] = str(args.model)

        # Resolve framework: --framework > $FRAMEWORK env > "sglang".
        # Session-wide; mixing frameworks in one session is not supported.
        framework = (
            (args.framework or os.environ.get("FRAMEWORK", "")).strip().lower()
            or "sglang"
        )
        if framework not in ("sglang", "vllm", "atom"):
            print(
                f"ERROR: --framework must be sglang, vllm, or atom "
                f"(got {framework!r}); set $FRAMEWORK accordingly or pass "
                "--framework",
                file=sys.stderr,
            )
            sys.exit(2)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")

        # B3: --framework atom auto-tightens incompatible phases. See
        # _apply_atom_auto_tighten for rationale.
        if framework == "atom":
            _apply_atom_auto_tighten(args)

        # Resolve real target GPU: --gpu-type > $GPU_TYPE > rocm-smi probe.
        # Keep the real type in args/SharedState so TraceLens and GEAK prompts
        # see MI325X, while mapping only Magpie's runner env to mi300x.
        gpu_type = (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        if not gpu_type:
            gpu_type = _autodetect_gpu_type() or ""
            if gpu_type:
                print(f"GPU type        : {gpu_type} (auto-detected)")
        runner_gpu_type = _gpu_runner_type(gpu_type)
        if gpu_type and runner_gpu_type != gpu_type:
            print(
                "WARN: mi325x uses mi300x as Magpie runner_type (same arch; "
                "Magpie has no sglang_mi325x.sh / vllm_mi325x.sh yet)",
                file=sys.stderr,
            )
        args.gpu_type = gpu_type or None
        if runner_gpu_type:
            os.environ["TARGET_GPU_TYPE"] = gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"GPU type        : {gpu_type}")
            print(f"Magpie runner   : {runner_gpu_type} (will inject runner_type into Magpie YAML)")
        else:
            os.environ.pop("TARGET_GPU_TYPE", None)
            os.environ.pop("GPU_TYPE", None)
            args.gpu_type = None
            print("GPU type        : <unset> (Magpie will auto-detect)")

        # Compute MAX_MODEL_LEN = ISL + OSL + 4096 headroom, export for yaml injection.
        max_model_len = args.isl + args.osl + 4096
        os.environ["MAX_MODEL_LEN"] = str(max_model_len)
        os.environ["ISL"] = str(args.isl)
        os.environ["OSL"] = str(args.osl)
        os.environ["PRECISION"] = args.precision
        print(f"Workload        : ISL={args.isl} OSL={args.osl} "
              f"MAX_MODEL_LEN={max_model_len} PRECISION={args.precision}")

        # N17: session_dir is now <workspace_root>/<model>/<UTC ts>/
        # by default (per-model + per-launch). Workspace_root is
        # $USER_DATA_PATH (fallback /workspace/hyperloom). To restore
        # the legacy flat layout, set INFERENCE_OPTIMIZER_SESSION_LAYOUT=
        # flat. `make_session_dir(model_name=...)` does the layout
        # decision + pins the result via $INFERENCE_OPTIMIZER_CURRENT_
        # SESSION_DIR for every subprocess.
        session_dir = make_session_dir(model_name=args.model)
        manifest = write_manifest(session_dir, args=args)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)
        _seed_shared_state(
            session_dir, args, session_id=manifest["session_id"],
        )
        # Cortex KB T0 anchor. Must run after the SharedState
        # seed (so model_name / gpu_type / framework are populated for
        # recipe_canonical_id derivation) but before Coordinator is
        # constructed (the Coordinator stores the client + threads it
        # into T2/T3/T4 hooks). Fails fast unless --degraded-kb.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=False,
        )
        kb_flusher_proc, kb_flusher_pid_path = _maybe_spawn_kb_flusher(
            args, session_dir=session_dir,
        )
        # KnowledgePlane facade for specialist
        # sub-agents. Wraps cortex_client (already T0'd above) + a
        # PR Monitor REST client; fail-soft on either side so specialist
        # dispatch always has a non-None plane to consult.
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )

    objective = build_objective({
        "MAX_HOURS": str(args.max_hours),
        "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
        "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
        "TARGET_DIR": args.target_baseline_dir or "",
    })
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    no_kernel = getattr(args, "no_kernel", False)

    # Resolve critic backend choice + critic-agent runtime root before
    # _build_backends (which constructs CriticAgentBackend immediately and
    # would otherwise blow up on missing runtime). Fail-fast policy: if the
    # operator selected --critic-agent (or it's the default) but the
    # critic-agent runtime is unreachable, we abort with rc=2 instead of
    # silently falling back to mock/codex_bare.
    critic_choice = _resolve_critic_choice(args)
    critic_agent_root: Path | None = None
    critic_kb_mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if critic_kb_mode not in ("inmemory", "live"):
        print(
            f"ERROR: CRITIC_KB_CLIENT_MODE={critic_kb_mode!r} not in "
            "{'inmemory','live'}",
            file=sys.stderr,
        )
        sys.exit(2)
    if critic_choice == "agent":
        critic_agent_root = _resolve_critic_agent_root()
        if critic_agent_root is None:
            print(
                f"ERROR: --critic-agent selected but critic-agent runtime not "
                f"found.\n"
                f"  Set ${_CRITIC_AGENT_ROOT_ENV} to the directory containing "
                f"runtime/cli.py, or install critic-agent at "
                f"$REPO_ROOT/critic-agent/.\n"
                f"  Bypass with --critic-mock or --critic-codex-bare.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_critic_agent_runtime(critic_agent_root)
        if critic_kb_mode == "live" and not os.environ.get("KB_BASE_URL"):
            print(
                "ERROR: CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not "
                "set. Either export KB_BASE_URL or unset "
                "CRITIC_KB_CLIENT_MODE to fall back to inmemory.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Default WORKSPACE_PATH for the critic-agent runtime if the
        # operator hasn't already pinned it. NOTE: in the critic-agent
        # runtime this env names the SKILL static-asset root (not a
        # writable artefact directory), so it points at the repo root.
        # The legacy "artefact root" meaning of WORKSPACE_PATH used by
        # kernel-agent tools was retired during the
        # all-artefacts-under-USER_DATA_PATH migration.
        os.environ.setdefault("WORKSPACE_PATH", str(Path(__file__).resolve().parents[1]))

    # Resolve robustness backend choice + runtime root, mirroring critic.
    robustness_choice = _resolve_robustness_choice(args)
    robustness_agent_root: Path | None = None
    robustness_options = _build_robustness_options(args)
    if robustness_choice == "agent":
        robustness_agent_root = _resolve_robustness_agent_root()
        if robustness_agent_root is None:
            print(
                f"ERROR: --robustness-agent selected but robustness-agent "
                f"runtime not found.\n"
                f"  Set ${_ROBUSTNESS_AGENT_ROOT_ENV} to the directory "
                f"containing src/robustness_agent/runtime/cli.py, or install "
                f"robustness-agent at $REPO_ROOT/robustness-agent/.\n"
                f"  Bypass with --robustness-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_robustness_agent_runtime(robustness_agent_root)

    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        kernel_codex=args.kernel_codex,
        critic_choice=critic_choice,
        session_dir=session_dir,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
        robustness_choice=robustness_choice,
        robustness_agent_root=robustness_agent_root,
        robustness_options=robustness_options,
        no_kernel=no_kernel,
    )
    # Bug A fix: expose the active session_dir to in-process executors
    # (e.g. ReportExecutor) that don't get session_dir threaded through
    # task.params. This is read in report.py::_resolve_session_dir.
    os.environ["USER_DATA_PATH"] = str(session_dir)
    # Production: enable strict path-containment checks in PolicyGate so
    # any LLM-emitted intent whose path field escapes session_dir lands
    # as `policy_denied` in its inbox. Tests omit this and keep the
    # legacy lenient mode for fixture paths under /tmp.
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # flip on PolicyGate R1 phase_incompatible enforcement
    # for production runs (matches the legacy→v0.8 strict_paths
    # rollout). Tests construct PolicyGate directly with strict_phase
    # left at the dataclass default (False) so legacy fixtures aren't
    # broken; this env var only affects the cli boot path.
    if getattr(args, "strict_phase", True):
        os.environ["INFERENCE_OPTIMIZER_STRICT_PHASE"] = "1"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_STRICT_PHASE", None)
    # propagate the ``--legacy-action-scores`` choice so
    # ``SharedState.from_dict`` (called from anywhere — Coordinator,
    # breakdown, resume probes) handles the drop / warn behavior
    # uniformly. Default ``drop`` matches the env-unset case.
    legacy_mode = str(
        getattr(args, "legacy_action_scores", "drop") or "drop",
    ).strip().lower()
    if legacy_mode == "warn":
        os.environ["INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES"] = "warn"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", None)
    # propagate ``--migration-mode``. SharedState.from_dict
    # consults this env var to decide whether a fact-layer
    # discrepancy is fatal (strict) or a downgraded WARNING (lenient).
    migration_mode = str(
        getattr(args, "migration_mode", "strict") or "strict",
    ).strip().lower()
    if migration_mode == "lenient":
        os.environ["INFERENCE_OPTIMIZER_MIGRATION_MODE"] = "lenient"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_MIGRATION_MODE", None)
    # ``--reset-state`` nukes the existing state.json
    # (backing it up to ``state.json.preReset.<unix_ts>``) so the
    # session starts blank. Done BEFORE Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir)
    # propagate ``--breakdown-include-transcripts`` so
    # any end-of-session breakdown emitted from this run picks up the
    # inline / path-only choice.
    transcripts_flag = str(
        getattr(args, "breakdown_include_transcripts", "false") or "false",
    ).strip().lower()
    if transcripts_flag == "true":
        os.environ["INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS"] = "1"
    else:
        os.environ.pop(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", None,
        )

    # Build the phase budget pct dict from CLI flags; ``None`` values
    # fall back to library defaults inside Coordinator. See KB_design
    # §3.13 M2 §7 + §3.8 §5.3.
    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", "PRELUDE"),
        ("phase_budget_explore_pct", "EXPLORE"),
        ("phase_budget_kernel_pct",  "KERNEL"),
        ("phase_budget_sweep_pct",   "SWEEP"),
        ("phase_budget_close_pct",   "CLOSE"),
    ):
        val = getattr(args, cli_field, None)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)

    # When kernel is disabled, strip it from the role registry so
    # Coordinator does not tick a non-existent agent and PolicyGate does
    # not expect a backend for it.
    role_registry = None
    if no_kernel:
        from .orchestrator.agent_role import default_role_registry
        role_registry = {
            k: v for k, v in default_role_registry().items() if k != "kernel"
        }

    coordinator = Coordinator(
        session_dir, backends=backends, role_registry=role_registry,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        model_class=(
            getattr(args, "model_class", None)
            or os.environ.get("MODEL_CLASS")
            or ""
        ),
        cortex_kb=cortex_client,
        phase_budget_pct=phase_budget_pct or None,
        # v0.8 §3.5 + KB_gaps/Gap-01/Gap-02 — KnowledgePlane facade.
        # ``None`` when --degraded-kb; otherwise wraps Cortex KB +
        # PR Monitor for specialist prompt assembly.
        knowledge_plane=knowledge_plane,
    )
    framework_for_prompt = (
        os.environ.get("FRAMEWORK", "").strip().lower() or "sglang"
    )
    max_minutes_for_prompt = int(round(float(args.max_hours) * 60))
    prompts: dict[str, str] = {
        "orchestration": args.orch_prompt or _build_orchestration_prompt(
            no_kernel=no_kernel,
            framework=framework_for_prompt,
            objective=objective,
            max_minutes=max_minutes_for_prompt,
        ),
        "critic":        args.critic_prompt or _load_critic_prompt(),
    }
    if not no_kernel:
        prompts["kernel"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT
    coordinator.system_prompt_overrides = prompts
    # ``fa phase-discover`` timeout override. ``framework_agent_client``
    # uses DEFAULT_FA_PHASE_TIMEOUT_SEC (180s) when this is missing /
    # falsy.
    try:
        coordinator.framework_pr_discover_timeout_sec = float(
            getattr(args, "framework_pr_discover_timeout_sec", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        coordinator.framework_pr_discover_timeout_sec = 0.0
    # v0.8 §3.5 + build specialist executor when the
    # research_lane capacity is non-zero. ``args.research_lane_capacity``
    # is already clamped to [0, 32] by ``_seed_shared_state``; a value
    # of 0 means "degrade to M3 LLM-direct grid", and we keep
    # ``specialist_executor=None`` so the dispatcher falls back to the
    # legacy ``no_executor`` rejection (which PolicyGate R2 also
    # short-circuits in practice).
    specialist_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 0
    )
    specialist_executor: "Any" = None
    if specialist_capacity > 0:
        specialist_executor = _build_specialist_executor(
            args,
            session_dir=session_dir,
            knowledge_plane=knowledge_plane,
        )
    _register_executors(
        coordinator,
        no_kernel=no_kernel,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
        specialist_executor=specialist_executor,
    )
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(session_dir, prompts=prompts)

    kernel_str = "DISABLED" if no_kernel else (
        f"{'Codex' if args.kernel_codex else 'Claude'}"
    )
    if critic_choice == "mock":
        critic_str = "mock"
    elif critic_choice == "codex_bare":
        critic_str = f"codex-bare({args.codex_model})"
    else:  # "agent"
        critic_str = (
            f"critic-agent(kb={critic_kb_mode}, codex={args.codex_model}, "
            f"root={critic_agent_root})"
        )
    if robustness_choice == "mock":
        robustness_str = "mock"
    else:
        robustness_str = f"robustness-agent(root={robustness_agent_root})"
        if robustness_options:
            kvs = ",".join(f"{k}={v!r}" for k, v in sorted(robustness_options.items()))
            robustness_str += f"[{kvs}]"
    print(f"Backends        : "
          f"orchestration=Claude({args.claude_model}), "
          f"kernel={kernel_str}, "
          f"critic={critic_str}, "
          f"robustness={robustness_str}")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} "
          f"(budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    if not (getattr(args, "compare_against_gpu", None) or "").strip():
        print(
            "[target_analysis] no --compare-against-gpu set; will write a "
            "marker JSON at $SESSION_DIR/target_analysis/target_baseline.json "
            "(reason=no_target_gpu_configured) — set --compare-against-gpu "
            "to fetch real InferenceX reference data.",
            file=sys.stderr,
        )

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
            closing_grace_sec=args.closing_grace_sec,
        )
    finally:
        await coordinator.stop()
        # v0.8 KB_gaps/Dead-E — stop the flusher daemon (best-effort;
        # graceful SIGTERM with 10s budget then SIGKILL fallback). Done
        # before the breakdown safety-net so the final NDJSON drain
        # numbers reflect a stable post-shutdown queue.
        try:
            _stop_kb_flusher(kb_flusher_proc, kb_flusher_pid_path)
        except Exception:  # noqa: BLE001
            log.exception("kb_flusher stop failed (non-fatal)")
        # End-of-session safety net: always materialize session_breakdown.json
        # for downstream consumers (claw-stats-service / hyperloom-results-
        # service / offline analysis). Best-effort — a failure here MUST NOT
        # mask the actual stop_reason, so we swallow exceptions and log.
        #
        # v0.8 §3.2 §5.5 / KB_gaps/Gap-06: when the CLOSE phase sequencer
        # ran to completion, step 2 already wrote the same artifact via
        # the standard session_breakdown executor. Skip the duplicate
        # write here so the cli.finally path doesn't clobber the
        # sequencer's output (which includes the full CLOSE-step
        # evidence the sequencer stamped on phase_history). The flag
        # is locked in CORE_STATE_FIELDS so an LLM can't trick us
        # into skipping the safety net for a non-CLOSE termination.
        sequencer_done = getattr(
            coordinator.shared_state, "close_sequence_done", False,
        )
        if sequencer_done:
            print(
                "Session breakdown : (already written by CLOSE phase "
                "sequencer; skipping cli.finally safety-net write)"
            )
        else:
            try:
                from .breakdown import write_breakdown_json
                breakdown_path = write_breakdown_json(session_dir)
                print(f"Session breakdown : {breakdown_path}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "session_breakdown finalize failed (non-fatal)"
                )
            # Issue-I (Saturday May 2026): mirror the same safety-net
            # for ``reports/final.md``. The full ReportExecutor walks
            # the message bus (rich highlights), but if the close
            # sequencer never reached step 1 the operator is left
            # with an empty ``reports/`` directory and has to manually
            # piece together stop_reason / current_best from
            # state.json. ``write_minimal_final_report`` is no-op when
            # the sequencer's final.md already exists.
            try:
                from .breakdown import write_minimal_final_report
                final_md = write_minimal_final_report(session_dir)
                print(f"Final report      : {final_md}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "emergency final report write failed (non-fatal)"
                )

    _print_final_summary(coordinator.shared_state, stop_reason)
    return 0 if stop_reason in (
        "target_reached",
        "no_more_leverage",
        "time_exhausted",
        "max_ticks",
    ) else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference_optimizer",
        description="Inference Optimizer v0.6 — multi-agent SGLang/vLLM optimization",
    )
    p.add_argument("--verbose", "-v", action="count", default=0,
                    help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize",
                          help="Drive a multi-agent optimization run on a model")
    opt.add_argument("--model", "-m", type=Path, default=None,
                      help="Model path (required for new runs; ignored when "
                           "--resume is set — model is read from manifest.json/"
                           "state.json)")
    opt.add_argument(
        "--gpu-type", choices=["mi300x", "mi325x", "mi355x"], default=None,
        help="Override the real target GPU for TraceLens/GEAK prompts. Magpie "
             "runner_type is derived separately; mi325x currently runs with "
             "mi300x runner scripts because Magpie does not yet ship "
             "sglang_mi325x.sh / vllm_mi325x.sh.",
    )
    opt.add_argument(
        "--framework", choices=["sglang", "vllm", "atom"], default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
             "--framework > $FRAMEWORK env > sglang (default). Selection is "
             "session-wide; mixing frameworks in a single session is not "
             "supported. NOTE: --framework atom is single-node-only and has "
             "no profiler / framework-source-patcher integration; B3 "
             "auto-tightens incompatible phases off when atom is selected.",
    )
    opt.add_argument(
        "--nodes", type=int,
        # Resolution: --nodes (CLI) > $INFERENCE_OPTIMIZER_NODES > $NODES > 1.
        # Brain / SaFE sandbox prompts conventionally export ``NODES=2`` in
        # ``optimizer.env`` (not the ``INFERENCE_OPTIMIZER_NODES`` form), so
        # without the ``NODES`` fallback the optimizer silently fell back to
        # single-node mode and the brain-supplied nodes value never reached
        # ``args.nodes``. The downstream ``nodes_resolved >= 2`` gate in
        # ``_run_optimize`` (and ``is_multi_node()`` in
        # ``_multi_node_env.py``) still hard-requires >= 2 before any
        # multi-node code path runs, so falling through to ``NODES=1``
        # produces the same single-pod behaviour as before this fallback
        # existed.
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_NODES")
            or os.environ.get("NODES")
            or "1"
        ),
        help="Total number of GPU nodes for the inference RayJob. "
             "1 (default) keeps the legacy single-pod path. "
             ">=2: `optimize` provisions the SaFE RayJob before preflight "
             "(unless already in /tmp/multi_node_state.json), runs bootstrap "
             "once, and exports RAY_ADDRESS for kernel-agent. Does not stop the "
             "RayJob on exit; run `python3 -m inference_optimizer.multi_node "
             "stop-rayjob` when you want to release it. Requires "
             "--rayjob-image or INFERENCE_OPTIMIZER_RAYJOB_IMAGE. "
             "Resolution: --nodes > $INFERENCE_OPTIMIZER_NODES > $NODES > 1.",
    )
    opt.add_argument(
        "--rayjob-image",
        default=None,
        help="Container image for the multi-node RayJob (head+workers). "
             "Required when --nodes>=2 unless INFERENCE_OPTIMIZER_RAYJOB_IMAGE "
             "is set or state file last_create_request.image is present.",
    )
    opt.add_argument(
        "--rayjob-gpus-per-node",
        type=int,
        default=None,
        help="GPUs per RayJob pod (default: INFERENCE_OPTIMIZER_GPUS_PER_NODE "
             "or 8). Passed to multi_node create-rayjob.",
    )
    # --rayjob-extra-env is a PROMPT-DRIVEN pass-through: this CLI never
    # invents or hardcodes env keys/values. The agent maps each line of
    # the user prompt's `env:` block into one `--rayjob-extra-env K=V`
    # (same contract as `multi_node create-rayjob --extra-env`). Forwarded
    # verbatim to `_provision_multi_node_rayjob_stack` -> cmd_create_rayjob
    # -> workload_spec.env. Reserved keys (RAY_JOB_ENTRYPOINT) are stripped
    # by the multi_node layer. Credential keys (*_API_KEY / *_BASE_URL)
    # are still auto-injected by _credential_fanout() — do NOT also pass
    # them here; the prompt block deliberately excludes them per the
    # multi_node SKILL contract.
    opt.add_argument(
        "--rayjob-extra-env",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra env entries to inject into the multi-node RayJob "
             "(repeatable). Agent maps each line of the user prompt's "
             "`env:` block into one --rayjob-extra-env K=V; the CLI "
             "does not own any default. Skip *_API_KEY / *_BASE_URL "
             "(auto-injected by _credential_fanout) and RAY_JOB_ENTRYPOINT "
             "(reserved by workload_spec). Only takes effect when "
             "--nodes>=2 and this run actually creates the RayJob; "
             "idempotent reuse of an existing rayjob_id keeps the env "
             "set at original create time.",
    )
    opt.add_argument(
        "--tp", type=int,
        default=int(os.environ.get("TP", "1") or 1),
        help="Tensor parallel size. Resolution: --tp > $TP env > 1. "
             "Symmetric with --ep — historically TP only flowed in via "
             "$TP env (read by _workload_envs); the CLI flag was added "
             "so the agent can pass `--tp N` directly from the prompt's "
             "Environment block instead of having to `export TP=N` "
             "first. Either path still works.",
    )
    opt.add_argument(
        "--conc", type=int,
        default=int(os.environ.get("CONC", "8") or 8),
        help="Magpie client concurrency cap (max in-flight requests). "
             "Resolution: --conc > $CONC env > 8. Symmetric with --tp; "
             "agent can pass `--conc N` directly from the prompt.",
    )
    opt.add_argument(
        "--ep", type=int,
        default=int(os.environ.get("EP", "1") or 1),
        help="Expert-parallel size for MoE inference. 1 (default) keeps "
             "experts sharded by TP (legacy behaviour). >=2 enables true "
             "expert parallelism: sglang adds `--expert-parallel-size N`, "
             "vllm adds `--enable-expert-parallel`. Typical: EP=TP for "
             "DSr1/DSv3 on multi-node. Resolution: --ep > $EP env > 1. "
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
        "--pd-prefill-nodes", type=int,
        default=int(os.environ.get("PD_PREFILL_NODES", "0") or 0),
        help="Number of prefill nodes (disaggregated only); pn+dn=nodes",
    )
    opt.add_argument(
        "--pd-decode-nodes", type=int,
        default=int(os.environ.get("PD_DECODE_NODES", "0") or 0),
        help="Number of decode nodes (disaggregated only)",
    )
    opt.add_argument(
        "--pd-prefill-tp", type=int,
        default=int(os.environ.get("PD_PREFILL_TP", "0") or 0),
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-decode-tp", type=int,
        default=int(os.environ.get("PD_DECODE_TP", "0") or 0),
        help="TP for decode group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-transfer-backend", type=str,
        default=os.environ.get("PD_TRANSFER_BACKEND", ""),
        help="sglang: mooncake|nixl ; vllm: NixlConnector|...; empty = default",
    )
    opt.add_argument(
        "--pd-ib-device", type=str,
        default=os.environ.get("PD_IB_DEVICE", ""),
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). "
             "Empty = use $NCCL_IB_HCA from RayJob pod env at server-launch time.",
    )
    opt.add_argument(
        "--skip-variants", type=str,
        default=os.environ.get("SKIP_VARIANTS", ""),
        help="Comma/whitespace-separated list of variant names or fnmatch "
             "globs to drop from the backends/params grids before launch. "
             "Examples: `attn_aiter` (exact), `attn_aiter,sched_dfs` (two "
             "exacts), `attn_*,vllm_aiter_*` (globs). Resolution: "
             "--skip-variants > $SKIP_VARIANTS > empty. Exported back into "
             "$SKIP_VARIANTS so all executors and the multi-node orchestrator "
             "subprocess see the same value. Dropped variants surface in "
             "state.json under each action's `dropped_variants` field tagged "
             "`source=user_skip`.",
    )
    opt.add_argument("--max-hours", type=float, default=2.0,
                      help="Wall-clock budget in hours (default 2.0)")
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
    opt.add_argument("--isl", type=int, default=int(os.environ.get("ISL", "256")),
                      help="Input sequence length (default $ISL or 256)")
    opt.add_argument("--osl", type=int, default=int(os.environ.get("OSL", "256")),
                      help="Output sequence length (default $OSL or 256)")
    opt.add_argument("--precision", type=str,
                      default=os.environ.get("PRECISION", "bf16"),
                      help="Model precision (default $PRECISION or bf16)")
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument("--target-gain", type=float, default=None,
                      help="Stop when cumulative_gain >= N%% over baseline")
    grp.add_argument("--target-tput", type=float, default=None,
                      help="Stop when current best tok/s/GPU >= N")
    grp.add_argument("--target-baseline-dir", type=str, default=None,
                      help="Stop when current best matches the baseline in DIR")
    opt.add_argument("--resume", action="store_true", default=False,
                      help="Resume an existing session. Without --resume-from, "
                           "auto-picks the latest per-session subdir under "
                           "$USER_DATA_PATH/<model>/<UTC ts>/ (N17 layout) or "
                           "falls back to $USER_DATA_PATH (legacy flat layout). "
                           "USER_DATA_PATH MUST stay at workspace level "
                           "(/wekafs/.../sessions/, not the per-session subdir) "
                           "so runtime/ resolution works. Skips the SharedState "
                           "seed and lets the Coordinator replay the prior "
                           "event log + state.json.")
    opt.add_argument("--resume-from", type=str, default=None,
                      help="Explicit per-session subdir to resume from. Use "
                           "when multiple per-launch ts dirs exist under the "
                           "same model and the latest is not what you want. "
                           "Must be an absolute path under $USER_DATA_PATH "
                           "(workspace_root). Implies --resume.")
    opt.add_argument(
        "--force-resume", action="store_true", default=False,
        help=(
            "Allow ``--resume`` to push past a terminal "
            "``stop_reason='no_more_leverage'`` or ``'target_reached'``. "
            "Without this flag the resume aborts (Issue-G guard, per "
            "SKILL.md 'Run-time signals': those terminals require an "
            "operator-side workload / strategy change before resuming). "
            "No-op outside ``--resume``."
        ),
    )
    opt.add_argument(
        "--model-class", type=str,
        default=os.environ.get("MODEL_CLASS", None),
        help=(
            "Model class hint consumed by "
            "``orchestrator/scoring.MODEL_CLASS_ACTION_PRIORS`` to seed "
            "per-action base scores. Recognised values (case-insensitive, "
            "with -/+/space tolerated): dense / moe_mla / moe_swa / "
            "moe_mla_nsa. The deleted ``classify`` action used to discover "
            "this from the model files; the external SKILL caller is now "
            "expected to supply it via this flag (or the MODEL_CLASS env "
            "var). Unset / unknown values fall back to the ``moe_mla`` "
            "curated priors so DeepSeek-shaped sessions keep working."
        ),
    )
    opt.add_argument("--target-summary", type=str, default=None,
                      help="Free-text goal summary surfaced in prompts")
    opt.add_argument(
        "--compare-against-gpu", type=str, default=None,
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
            "section. The data is REPORT-ONLY: it does not influence "
            "Objective, scoring, or any agent prompt. Other dimensions "
            "(model / framework / precision / ISL / OSL) are derived "
            "from --model and the standard FRAMEWORK / PRECISION / ISL / "
            "OSL env vars."
        ),
    )
    opt.add_argument("--max-ticks", type=int, default=None,
                      help="Hard tick cap (None = unlimited; mostly for tests)")
    opt.add_argument("--tick-interval-sec", type=float, default=0.0,
                      help="Sleep between ticks (0 = no sleep)")
    opt.add_argument("--claude-model", type=str,
                      default=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"))
    opt.add_argument("--codex-model", type=str,
                      default=os.environ.get("CODEX_MODEL", "gpt-5.4"))
    opt.add_argument("--no-kernel", action="store_true", default=False,
                      help="Disable the Kernel agent entirely. The run will "
                           "only do baseline + params + backends + sweep (pure "
                           "parameter search). Useful when GEAK/OOB/GPU "
                           "compile env is unavailable or you just want the "
                           "quick-win parameter path. Default: kernel enabled.")
    opt.add_argument("--framework-pr-discover-timeout-sec", type=float,
                      default=0.0,
                      help="Override the per-call timeout for "
                           "``fa phase-discover``. 0 (the default) uses "
                           "framework_agent_client.DEFAULT_FA_PHASE_TIMEOUT_SEC "
                           "(180s). The Coordinator retries discover up to "
                           "DISCOVER_FAILURE_RETRY_LIMIT (3) consecutive "
                           "failures before marking FRAMEWORK_PR done.")
    opt.add_argument("--no-framework", action="store_true",
                      default=os.environ.get(
                          "INFERENCE_OPTIMIZER_NO_FRAMEWORK", "0",
                      ).strip() in ("1", "true", "True", "TRUE", "yes"),
                      help="Skip the FRAMEWORK_PR phase (PRELUDE → EXPLORE "
                           "directly). The phase pre-scans upstream sglang/"
                           "vllm PRs via framework-agent and lands KEPT "
                           "patches before EXPLORE starts. Disable when "
                           "the framework-agent toolchain is unavailable "
                           "or you want a faster cold start. Also read from "
                           "$INFERENCE_OPTIMIZER_NO_FRAMEWORK=1. "
                           "Default: framework phase enabled.")
    opt.add_argument("--kernel-codex", action="store_true", default=True,
                      help="Use Codex backend for Kernel agent (default — faster). "
                           "Pass --kernel-claude to switch.")
    opt.add_argument("--kernel-claude", action="store_false", dest="kernel_codex",
                      help="Use Claude backend for Kernel agent")
    # Critic backend selection. ``critic_backend`` is the canonical
    # attribute; the four flags below are convenience aliases that all
    # set the same dest. Default is filled by the CLI default block in
    # ``_resolve_critic_choice``; a single explicit flag wins, conflicts
    # are caught at runtime (mutual exclusion checked there because
    # argparse mutually-exclusive groups don't compose with default values
    # cleanly when we want one flag to be the implicit default).
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
        "--critic-codex-bare",
        dest="critic_backend",
        action="store_const",
        const="codex_bare",
        help="Force the legacy bare-Codex Critic (single chat-completion + "
             "JSON envelope; no KB, no session memory). For debugging the "
             "LLM layer in isolation.",
    )
    # Back-compat alias for the old --critic-real flag (semantically the
    # same as --critic-codex-bare). Hidden from --help to avoid promoting
    # the old name; still accepted so existing launchers don't break.
    opt.add_argument(
        "--critic-real",
        dest="critic_backend",
        action="store_const",
        const="codex_bare",
        help=argparse.SUPPRESS,
    )
    # ----- Robustness backend selection (mirrors critic) ------------------
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
    opt.add_argument("--orch-prompt", type=str, default=None,
                      help="Override Orchestration system prompt (file path or inline)")
    opt.add_argument("--critic-prompt", type=str, default=None,
                      help="Override Critic system prompt")
    opt.add_argument("--kernel-prompt", type=str, default=None,
                      help="Override Kernel system prompt")
    # ------------------------------------------------------------------
    # Cortex KB integration flags
    # ------------------------------------------------------------------
    # The defaults wire Cortex *on* (matches the "Loop 1" expectation in
    # the cortex hand-off doc). ``--degraded-kb`` is a debug escape hatch
    # that fully bypasses T0/T2/T3/T4 so a fresh sandbox can reproduce
    # the behaviour without any KB writes. ``--cortex-kb-url``
    # overrides the env value (``CORTEX_KB_URL``) without exporting one
    # process-wide. ``--cortex-strict-fingerprint`` enforces the
    # manifest stack_fingerprint matches a recipe before warm_start is
    # consumed (M5 consumer; M1 records the flag into manifest only).
    opt.add_argument(
        "--cortex-kb-url",
        dest="cortex_kb_url",
        type=str,
        default=None,
        help="Override CORTEX_KB_URL for this run (default: env value or "
             "http://kb-service.primus-cortex.svc.cluster.local).",
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
    # v0.8 KB_gaps/Dead-E — Cortex KB flusher daemon lifecycle. The cli
    # spawns ``scripts.cortex_kb_flusher`` after the T0 anchor (so the
    # NDJSON pending queue gets drained in the background while the
    # main loop runs); ``--no-kb-flusher`` short-circuits the spawn for
    # operators debugging the NDJSON path manually. ``--kb-flusher-*``
    # overrides forward the same flags to the daemon CLI.
    opt.add_argument(
        "--no-kb-flusher",
        dest="kb_flusher_enabled",
        action="store_false",
        default=True,
        help="Skip spawning the Cortex KB flusher daemon. The NDJSON "
             "pending queue still accumulates but only drains on the "
             "next session start (or via manual ``python -m "
             "inference_optimizer.scripts.cortex_kb_flusher`` run). "
             "Implied by --degraded-kb.",
    )
    opt.add_argument(
        "--kb-flusher-interval-sec",
        dest="kb_flusher_interval_sec",
        type=float,
        default=5.0,
        help="Forwarded to the flusher daemon as --interval-sec "
             "(default 5.0).",
    )
    opt.add_argument(
        "--kb-flusher-batch-size",
        dest="kb_flusher_batch_size",
        type=int,
        default=50,
        help="Forwarded to the flusher daemon as --batch-size "
             "(default 50).",
    )
    # ------------------------------------------------------------------
    # PR Monitor REST + MCP
    # ------------------------------------------------------------------
    # ``--pr-monitor-url`` overrides the in-cluster default; the
    # marathon pod is typically in a different cluster from
    # primus-cortex, so an operator running outside the primus-cortex
    # k8s namespace must port-forward + pass a localhost URL.
    # ``--degraded-pr`` switches the KnowledgePlane.pr_feed_warm
    # path to a no-op and strips ``mcp__pr_monitor__*`` from the
    # specialist tool whitelist (Inv-6.3 degrade-to-empty).
    opt.add_argument(
        "--pr-monitor-url",
        dest="pr_monitor_url",
        type=str,
        default=None,
        help="Override PR Monitor REST URL for this run. Default: "
             "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local"
             "/v1 (env: PR_MONITOR_URL). Pair with --pr-monitor-mcp-url "
             "when port-forwarding for local debug.",
    )
    opt.add_argument(
        "--pr-monitor-mcp-url",
        dest="pr_monitor_mcp_url",
        type=str,
        default=None,
        help="Override PR Monitor MCP URL handed to specialist LLM "
             "backends. Default mirrors --pr-monitor-url with /mcp/ "
             "suffix; the trailing slash is mandatory.",
    )
    opt.add_argument(
        "--degraded-pr",
        dest="degraded_pr",
        action="store_true",
        default=False,
        help="Disable the PR Monitor integration entirely. "
             "pr_feed_warm returns empty; the specialist tool "
             "whitelist drops mcp__pr_monitor__* tools. Short-circuits "
             "the IR-3 PR Monitor probe; IR-3 sets this automatically "
             "when PR Monitor is unreachable (soft degrade).",
    )
    opt.add_argument(
        "--pr-feed-window-days",
        dest="pr_feed_window_days",
        type=int,
        default=int(
            os.environ.get("PR_FEED_WINDOW_DAYS", "30") or "30"
        ),
        help="Look-back window for the PR feed warmup (days). "
             "Default: 30.",
    )
    # ------------------------------------------------------------------
    # v0.8 M5/M6 — specialist research_lane capacity
    # ------------------------------------------------------------------
    # ``--research-lane-capacity`` locks the number of LLM specialists
    # that may run concurrently on the research_lane:
    #   * 0   → degrade to M3 (no specialist dispatch; EXPLORE uses the
    #           default_grid path).
    #   * 1   → v0.8 M5 default (single specialist at a time).
    #   * 4   → PR-A3 (Arbor-into-Hyperloom) default — enough headroom
    #           for the Orchestration LLM to fan out one specialist per
    #           top-K gap inside one tick (multi-emit shape) and have
    #           the dispatcher actually run them in parallel.
    #   * 6   → M6 ceiling that matches Arbor's "six specialists across
    #           domains" pattern; hard upper bound (values above this
    #           are silently clamped down to 6 — see
    #           ``MAX_RESEARCH_LANE_CAPACITY`` in
    #           ``orchestrator/policy.py``).
    # Locked at session start (mirrored into manifest + SharedState);
    # PolicyGate denies mid-flight mutation via CORE_STATE_FIELDS.
    opt.add_argument(
        "--research-lane-capacity",
        dest="research_lane_capacity",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY", "4")
            or "4"
        ),
        help="Max concurrent LLM specialist sub-agents on the "
             "research_lane. 0 disables specialist "
             "dispatch entirely (degrades to M3 LLM-direct grid); 4 is "
             "the PR-A3 default (Arbor-into-Hyperloom); 6 is the M6 "
             "hard cap. Range [0, 6]; values above 6 are silently "
             "clamped down. Locked at session start.",
    )
    # ------------------------------------------------------------------
    # v0.8 §3.5 / §3.13 M5 + specialist sub-agent
    # backend selection. Specialists run via Claude (default) and inherit
    # the orchestration model unless overridden. Per-task turn / time
    # caps protect against runaway LLM consumption.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--specialist-model",
        dest="specialist_model",
        type=str,
        default=os.environ.get("INFERENCE_OPTIMIZER_SPECIALIST_MODEL", "")
        or None,
        help="Claude model used for specialist sub-agents (defaults to "
             "the orchestration --claude-model). KB_design §3.5 §6.",
    )
    opt.add_argument(
        "--specialist-max-turns",
        dest="specialist_max_turns",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_SPECIALIST_MAX_TURNS", "8")
            or "8"
        ),
        help="Hard cap on LLM turns per specialist task (KB_design "
             "§3.5 §6). On exhaustion the runner synthesises an empty "
             "specialist_done (Inv-5.3).",
    )
    opt.add_argument(
        "--specialist-per-turn-max-seconds",
        dest="specialist_per_turn_max_seconds",
        type=float,
        default=float(
            os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS", "600"
            )
            or "600"
        ),
        help="Wall-clock cap per specialist turn (default 600s). Used "
             "by the robustness stale-scan to detect stuck specialists "
             ".",
    )
    # ------------------------------------------------------------------
    # PR-A2 (Arbor-into-Hyperloom): specialist dispatch shape.
    # ------------------------------------------------------------------
    opt.add_argument(
        "--specialist-dispatch-mode",
        dest="specialist_dispatch_mode",
        type=str,
        choices=("subprocess", "inprocess"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_DISPATCH_MODE", "subprocess",
        ).strip() or "subprocess",
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
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_MCP_CONFIG", "",
        ).strip() or None,
        help="Optional path to an MCP config JSON forwarded to the "
             "subprocess claude (`--mcp-config`). Used to wire kb / pr "
             "MCP servers into specialists. Default: None.",
    )
    # ------------------------------------------------------------------
    # Integration toggles. The roofline composite + watermark-driven
    # refresh path is unconditional now: roofline auto-fires at PRELUDE
    # (after baseline) and on every 10% ``cumulative_gain_validated``
    # crossing over ``last_roofline_tput``. The composite/deny toggles
    # that gated the legacy single-step ``profile`` path are gone.
    # ------------------------------------------------------------------
    def _env_default_on(env_var: str) -> bool:
        return os.environ.get(env_var, "1").strip() != "0"

    # The standalone FRAMEWORK_PR phase is on by default; use
    # ``--no-framework`` to disable it (mirrors the install-side
    # ``INFERENCE_OPTIMIZER_NO_FRAMEWORK=1`` opt-out).
    opt.add_argument(
        "--gain-driven-kernel-opt",
        dest="gain_driven_kernel_opt",
        action="store_true",
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT", "0",
        ).strip() == "1",
        help="Lock ``kernel_opt`` until the 3-round moving "
             "average of ``last_explore_delta_gain_pct`` drops below "
             "epsilon (0.5%%). Prevents premature deep work while cheap "
             "exploration is still earning. Default off. Env: "
             "INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT=1.",
    )
    opt.add_argument(
        "--enable-roofline",
        dest="enable_roofline",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_ENABLE_ROOFLINE"),
        help="Select which analysis action the Coordinator enqueues at "
             "PRELUDE bootstrap and on every +10% watermark crossing. "
             "Default on: ``roofline`` (composite profile + "
             "trace_analyze + analysis.md). Pass ``--no-enable-roofline`` "
             "to use plain ``profile`` instead (lighter — captures the "
             "trace only, skips trace_analyze). Behaviour is otherwise "
             "identical (same idempotency keys, same pending-task "
             "dispatch gate, same watermark anchor update). Env: "
             "INFERENCE_OPTIMIZER_ENABLE_ROOFLINE=0.",
    )
    # Retired flags that operator scripts may still pass. We hard-fail
    # at argparse time with a one-line migration hint instead of
    # silently aliasing — silent aliases hide the behaviour change from
    # the single ``--enable-roofline`` mode-select (the old composite /
    # direct-profile bifurcation no longer exists, and the
    # PRELUDE-initial roofline is unconditional).
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
    # ------------------------------------------------------------------
    # Fix E — per-variant explore overtime kill ratio.
    #
    # Mirrored into :attr:`SharedState.explore_overtime_kill_ratio`
    # and read by :class:`ExploreExecutor` via task.params (Coordinator
    # injects ``baseline_runtime_sec`` + ``explore_overtime_kill_ratio``
    # alongside ``base_extra_args`` / ``base_tput`` on every explore
    # task).
    #
    # Default 1.10: kill a single-variant Magpie run once its
    # wall-clock exceeds the baseline run's wall-clock by +10 %. The
    # killed variant is recorded with ``outcome='KILLED_OVERTIME'`` +
    # ``runtime_sec`` + ``wall_clock_ratio_vs_baseline`` (no tput) so
    # the orchestration LLM can tell "ran too slow → early kill" from
    # "benchmark crashed" or "hit the hard variant timeout".
    #
    # Pass ``0`` (or any non-positive value) to disable the gate; the
    # legacy ``variant_timeout_sec`` hard cap is still enforced in
    # that case. The gate ONLY applies to the per-variant single-run
    # step; the inlined stack rebench (Q4) intentionally uses the
    # legacy timeout because a deep stack legitimately runs slower
    # than the bare baseline.
    # ------------------------------------------------------------------
    def _env_float_or(default: float, env_var: str) -> float:
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)
    opt.add_argument(
        "--explore-overtime-kill-ratio",
        dest="explore_overtime_kill_ratio",
        type=float,
        default=_env_float_or(
            1.10, "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO",
        ),
        help="Per-variant explore overtime kill: each single-variant "
             "Magpie run in the explore loop is reaped once its "
             "wall-clock exceeds ``baseline_runtime_sec * RATIO``. The "
             "variant is recorded with outcome=KILLED_OVERTIME + "
             "runtime_sec + wall_clock_ratio_vs_baseline (no tput) so "
             "the LLM can distinguish it from a hard timeout / crash. "
             "Default 1.10 (kill at +10%% over baseline wall-clock). "
             "Pass 0 to disable. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO.",
    )
    # ------------------------------------------------------------------
    # drop scoreboard
    # ------------------------------------------------------------------
    # v0.8 retires the legacy ``action_scores`` decision system. The
    # flag below controls how a resumed session's leftover
    # scoreboard data is handled:
    #
    # * ``drop`` (default): silently strip ``action_scores`` /
    #   ``params_no_promote_streak`` / ``score_violation`` etc. from
    #   the loaded state.json. The Coordinator never writes them
    #   again so subsequent saves are clean.
    # * ``warn``: same drop behaviour PLUS a ``WARNING`` log line +
    #   a ``breakdown.warnings`` entry so the operator sees the
    #   migration even on a fresh dashboard.
    #
    # No third "keep" mode is offered: §3.9 §7 explicitly forbids
    # carrying the field forward (the prompt builder no longer reads
    # it; keeping the bytes only inflates state.json).
    opt.add_argument(
        "--legacy-action-scores",
        dest="legacy_action_scores",
        type=str,
        choices=("drop", "warn"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "drop",
        ).strip() or "drop",
        help="Resume-mode handling of the legacy scoreboard "
             "(``action_scores`` and friends). 'drop' (default) "
             "silently discards. 'warn' logs a WARNING + adds a "
             "breakdown.warnings entry. KB_design §3.9 §7.",
    )
    # ------------------------------------------------------------------
    # SharedState evolution
    # ------------------------------------------------------------------
    # ``--migration-mode`` controls how non-fatal schema migration
    # discrepancies are surfaced:
    #
    # * ``strict`` (default): a missing fact-layer field (baseline_tput
    #   / current_best / cumulative_gain / optimization_stack) inside
    #   a non-empty state.json is fatal — CLI exits 1 so the operator
    #   can investigate before any new write corrupts the audit trail.
    # * ``lenient``: same discrepancies are downgraded to WARNING and
    #   the run continues with default values. Useful when an operator
    #   is intentionally salvaging a partially-corrupted session.
    #
    # ``--reset-state`` is the nuclear option: when set, the existing
    # ``state.json`` is backed up to ``state.json.preReset.<ts>`` and
    # the session starts fresh. Cortex KB cross-session knowledge is
    # untouched.
    opt.add_argument(
        "--migration-mode",
        dest="migration_mode",
        type=str,
        choices=("strict", "lenient"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_MIGRATION_MODE", "strict",
        ).strip() or "strict",
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
    # ------------------------------------------------------------------
    # observability
    # ------------------------------------------------------------------
    # ``--breakdown-include-transcripts`` controls whether the per-task
    # specialist transcript bodies are inlined under
    # ``specialist_runs[i].transcripts[j].body`` (true) or only
    # referenced by path (false, default). Default false keeps the
    # ``session_breakdown.json`` payload small for the cluster
    # dashboards; operators can flip it on for offline replay.
    opt.add_argument(
        "--breakdown-include-transcripts",
        dest="breakdown_include_transcripts",
        type=str,
        choices=("true", "false"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "false",
        ).strip().lower() or "false",
        help="Inline specialist transcript bodies into "
             "``specialist_runs`` (true) or reference them by path "
             "only (false, default). KB_design §3.12 §7.",
    )
    # ------------------------------------------------------------------
    # plateau threshold tuning
    # ------------------------------------------------------------------
    # These flags swap the library default plateau thresholds for the
    # ``compute_plateau_explore`` / ``compute_plateau_kernel`` pure
    # functions. They land in :attr:`SharedState.plateau_overrides`
    # at session boot; ``phase_state.compute_next_phase`` consults
    # the overrides every tick. Locked at session start so resume
    # uses the same threshold the original run picked.
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
        help="EXPLORE plateau: number of trailing rounds the gain sum is "
             "computed over. Default 5.",
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
        help="KERNEL plateau: number of trailing integrate attempts the "
             "gain sum is computed over. Default 5.",
    )
    # ------------------------------------------------------------------
    # v0.8 IR-6 — EXPLORE HARD force-exit thresholds (Saturday May 2026)
    # ------------------------------------------------------------------
    # Either condition fires an ``explore_force_exit_low_budget`` exit
    # which routes EXPLORE → KERNEL (or SWEEP when --no-kernel). Iron
    # Rule IR-6: this gate is non-negotiable — the steward / plateau /
    # LLM cannot extend EXPLORE past either threshold. Locked at session
    # start into ``SharedState.plateau_overrides``.
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
    # ------------------------------------------------------------------
    # IR-7 — session_steward_specialist controls (Saturday May 2026)
    # ------------------------------------------------------------------
    # The steward is the soft gate on plateau (HARD IR-6 still wins on
    # low budget). Operators can disable it for smoke runs or cap its
    # continuation-granting power.
    opt.add_argument(
        "--steward-disabled",
        dest="steward_disabled",
        action="store_true",
        default=False,
        help="Disable session_steward_specialist; plateau directly "
             "exits EXPLORE without the steward gate (IR-7).",
    )
    opt.add_argument(
        "--steward-continuation-cap",
        dest="steward_continuation_cap",
        type=int,
        default=None,
        help="Max times the steward may return 'continue_explore' in "
             "this session (default 1). Beyond the cap, "
             "continue_explore is coerced to advance_to_kernel.",
    )
    # ------------------------------------------------------------------
    # phase budget percentages
    # ------------------------------------------------------------------
    # Each phase claims a fraction of the total wall-clock budget. The
    # numbers below are caps — the Coordinator may exit a phase earlier
    # if the plateau judge fires. Defaults follow §3.8 §5.3.  Sum need
    # not equal 1.0 (a deliberate padding is encouraged).
    opt.add_argument(
        "--max-minutes-prelude-pct",
        dest="phase_budget_prelude_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for PRELUDE as a fraction of "
             "--max-hours. Default: 0.05.",
    )
    opt.add_argument(
        "--max-minutes-explore-pct",
        dest="phase_budget_explore_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for EXPLORE. Default: 0.60.",
    )
    opt.add_argument(
        "--max-minutes-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL. Default: 0.25.",
    )
    opt.add_argument(
        "--max-minutes-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.08.",
    )
    opt.add_argument(
        "--max-minutes-close-pct",
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
        help="Disable R1 enforcement (warn-only). Useful for "
             "back-compat smoke tests; production should stay strict.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt", "kernel_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        return asyncio.run(_run_optimize(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
