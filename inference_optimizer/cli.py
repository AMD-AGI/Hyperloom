"""CLI entry — DESIGN v0.6 §22.

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
  INFERENCE_OPTIMIZER_SESSION_DIR              — override session dir for tests
                                                 (default: /workspace/hyperloom).
  INFERENCE_OPTIMIZER_KB_ROOT                  — marathon KB dir (kb_query.py +
                                                 entries.jsonl); default:
                                                 Hyperloom/marathon/skills/kb
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .orchestrator.action_executors import (
    TargetAnalysisExecutor,
    backends_executor,
    baseline_executor,
    params_executor,
    pmc_roofline_executor,
    profile_executor,
    report_executor,
    sweep_executor,
    validate_stack_executor,
)
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
from .orchestrator.objective import Objective, build_objective
from .orchestrator.shared_state import SharedState
from .orchestrator.system_prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from .paths import (
    DEFAULT_SESSION_DIR,
    ENV_OVERRIDE_SESSION_DIR,
    _SESSION_SKELETON,
    asset_system_prompts_dir,
    make_session_dir,
    session_dir as _session_dir_resolve,
)
from .session_paths import (
    agent_prompt_snapshot,
)


log = logging.getLogger("inference_optimizer.cli")


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

# Where ``ensure_auth_proxy.sh`` expects ``auth_proxy.py`` to live, plus the
# read-only mount points that ship the source. ``OOB_SRC`` env wins; the
# defaults below match the bundle layout used by ``kernel-agent/install.sh``.
_OOB_SRC_CANDIDATES: tuple[str, ...] = (
    "/wekafs/fully-local/OOB",
    "/wekafs/fully-local/inference_optimization/OOB",
)

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
            timeout=5,
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
            timeout=5,
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
        critic_backend = CriticAgentBackend(
            critic_agent_root=critic_agent_root,
            session_dir=session_dir,
            codex_model=codex_model,
            kb_mode=critic_kb_mode,
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
    state = SharedState(
        session_id=session_id,
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        kernel_enabled=not getattr(args, "no_kernel", False),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=int((args.max_hours or 0) * 60),
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
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline":       baseline_executor,
    "backends":       backends_executor,
    "params":         params_executor,
    "sweep":          sweep_executor,
    "report":         report_executor,
    "validate_stack": validate_stack_executor,
}

# Real executors enabled only when kernel-mode is on (profile/pmc_roofline
# only feed kernel-opt and would burn lanes for nothing in --no-kernel).
_REAL_EXECUTORS_KERNEL_ONLY: dict[str, Any] = {
    "profile":      profile_executor,
    "pmc_roofline": pmc_roofline_executor,
}

# Prep / orchestration-only / agent-owned action kinds that the
# Orchestration loop still needs to dispatch but whose bodies are no-ops
# (the orchestration agent does the actual work via emit_intent). Kept
# split so --no-kernel can exclude kernel-owned kinds.
_NOOP_KINDS_COMMON: tuple[str, ...] = (
    "setup", "classify", "target_analysis",
    "dream", "re_explore", "recover",
    "comm_optimization", "compiler_tuning",
)
_NOOP_KINDS_KERNEL_ONLY: tuple[str, ...] = (
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "operator_tuning", "vendor_kernel_config",
)


def _register_executors(
    coordinator: Coordinator,
    *,
    no_kernel: bool = False,
    compare_against_gpu: str | None = None,
    session_dir: Path | None = None,
) -> None:
    """Wire all currently-available action executors.

    Real executors are pulled from ``_REAL_EXECUTORS_FULL`` (always) and
    ``_REAL_EXECUTORS_KERNEL_ONLY`` (when kernel-mode is on). Kinds that
    the Orchestration loop dispatches but whose bodies are no-ops (prep
    stubs + kernel-owned actions whose work happens in handlers) get
    ``_noop_prep`` so SubAgentRunner doesn't fail with "no_executor".

    When ``no_kernel`` is True, kernel-owned action stubs are skipped and
    ``profile`` is also skipped (profiling only feeds kernel-opt).

    When ``compare_against_gpu`` is a non-empty string, ``target_analysis``
    is registered with the real :class:`TargetAnalysisExecutor` instead of
    the default ``_noop_prep`` stub; the runner pulls the rest of the
    reference query (model / framework / precision / ISL / OSL) from
    process env at call time. Empty / None falls back to the no-op so
    existing runs are byte-identical.
    """
    for kind, fn in _REAL_EXECUTORS_FULL.items():
        coordinator.sub.register_executor(kind, fn)

    if no_kernel:
        noop_kinds = _NOOP_KINDS_COMMON
    else:
        for kind, fn in _REAL_EXECUTORS_KERNEL_ONLY.items():
            coordinator.sub.register_executor(kind, fn)
        noop_kinds = _NOOP_KINDS_COMMON + _NOOP_KINDS_KERNEL_ONLY

    compare_gpu = (compare_against_gpu or "").strip()
    for kind in noop_kinds:
        if kind == "target_analysis" and compare_gpu:
            coordinator.sub.register_executor(
                "target_analysis",
                TargetAnalysisExecutor(
                    compare_against_gpu=compare_gpu,
                    session_dir=session_dir,
                ),
            )
            continue
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
            f"⚠ never validated — no `validate_stack` action ran"
        )
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    print("===============================================")


def _derive_proxy_urls(upstream_url: str, proxy_port: int) -> tuple[str, str]:
    """Mirror of bash ``derive_proxy_urls`` in ``ensure_auth_proxy.sh``.

    Returns ``(proxy_anthropic_url, proxy_openai_url)`` from a LiteLLM-style
    ``upstream_url`` like ``https://host/api/v1/llm-proxy/v1``:

    * The OpenAI URL keeps the path verbatim → ``http://127.0.0.1:4002/api/v1/llm-proxy/v1``.
    * The Anthropic URL strips a trailing ``/v1`` because the Anthropic SDK
      appends it itself → ``http://127.0.0.1:4002/api/v1/llm-proxy``.
    """
    from urllib.parse import urlparse

    path = urlparse(upstream_url).path.rstrip("/")
    anthropic_path = path[: -len("/v1")] if path.endswith("/v1") else path
    base = f"http://127.0.0.1:{proxy_port}"
    return f"{base}{anthropic_path}", f"{base}{path}"


def _proxy_alive(proxy_port: int, timeout: float = 2.0) -> bool:
    """TCP-probe ``127.0.0.1:proxy_port``. Returns True iff connect succeeds."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(("127.0.0.1", proxy_port))
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _ensure_auth_proxy_and_claude_config(
    safe_key: str, base_url: str
) -> tuple[str, str] | None:
    """Start auth-proxy on :4002 and ensure ~/.claude/config.json uses it.

    The AMD primus-safe gateway rejects x-api-key (returns "token not
    present"). Claude CLI only sends x-api-key. The auth_proxy bridges
    the gap by rewriting to Authorization: Bearer. We must:

    1. Start the proxy (idempotent — reuses ``ensure_auth_proxy.sh`` logic).
       If the supervisor reports success but the port is not actually open,
       retry the supervisor once before giving up (the "127 retry" leg).
    2. Point ~/.claude/config.json at the proxy, not directly at the gateway.

    Returns ``(proxy_anthropic_url, proxy_openai_url)`` when the proxy is
    confirmed alive on ``127.0.0.1:proxy_port``; ``None`` otherwise. The
    caller is responsible for force-overriding ``ANTHROPIC_BASE_URL`` /
    ``OPENAI_BASE_URL`` based on this return value.
    """
    import json as _json

    proxy_port = int(os.environ.get("AUTH_PROXY_PORT", "4002"))
    upstream_url = base_url or os.environ.get(
        "ANTHROPIC_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", ""),
    )
    if not upstream_url:
        print("Preflight: no LLM base URL set; skipping auth-proxy setup")
        return None

    proxy_anthropic_url, proxy_openai_url = _derive_proxy_urls(
        upstream_url, proxy_port
    )

    proxy_script = (
        Path(__file__).resolve().parent.parent
        / "kernel-agent"
        / "scripts"
        / "ensure_auth_proxy.sh"
    )

    def _run_supervisor() -> bool:
        """Run ensure_auth_proxy.sh once. Returns True if it exited 0."""
        if not proxy_script.exists():
            return False
        env_for_proxy = os.environ.copy()
        env_for_proxy["OOB_BASE_URL"] = upstream_url
        env_for_proxy["OOB_API_KEY"] = safe_key or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        env_for_proxy["AUTH_PROXY_PORT"] = str(proxy_port)
        result = subprocess.run(
            ["bash", str(proxy_script)],
            capture_output=True,
            text=True,
            env=env_for_proxy,
        )
        for line in (result.stdout or "").strip().splitlines():
            print(f"  {line}")
        if result.returncode != 0:
            print(
                f"Preflight: WARNING — ensure_auth_proxy.sh failed "
                f"(rc={result.returncode})"
            )
            for line in (result.stderr or "").strip().splitlines()[-5:]:
                print(f"  {line}")
            return False
        return True

    proxy_ready = False
    if proxy_script.exists():
        if _run_supervisor() and _proxy_alive(proxy_port):
            proxy_ready = True
        else:
            # 127 retry leg — supervisor may have just unblocked a stuck
            # port or swapped credentials. One re-run is cheap and recovers
            # from "port_open but probe timed out" races.
            print("Preflight: auth-proxy not alive after first attempt; retrying")
            if _run_supervisor() and _proxy_alive(proxy_port):
                proxy_ready = True
    else:
        # Supervisor missing — best-effort: trust the port if it is open.
        if _proxy_alive(proxy_port):
            print("Preflight: auth-proxy :4002 already open")
            proxy_ready = True
        else:
            print(
                "Preflight: WARNING — auth-proxy :4002 not running and "
                f"ensure_auth_proxy.sh not found at {proxy_script}"
            )

    if not proxy_ready:
        print(
            "Preflight: WARNING — auth-proxy could not be brought up; "
            "falling back to original env"
        )
        return None

    # Proxy is alive. Update ~/.claude/config.json to match.
    claude_config_path = Path.home() / ".claude" / "config.json"
    needs_update = False
    config_data: dict = {}
    if claude_config_path.exists():
        try:
            config_data = _json.loads(
                claude_config_path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            config_data = {}
        current_url = config_data.get("customApiUrl", "")
        if "127.0.0.1" not in current_url and "localhost" not in current_url:
            needs_update = True
    else:
        needs_update = True

    if needs_update:
        config_data.setdefault("theme", "dark")
        config_data.setdefault("hasCompletedOnboarding", True)
        config_data["primaryApiKey"] = safe_key or config_data.get(
            "primaryApiKey", ""
        )
        config_data["customApiUrl"] = proxy_anthropic_url
        claude_config_path.parent.mkdir(parents=True, exist_ok=True)
        claude_config_path.write_text(
            _json.dumps(config_data, indent=2) + "\n", encoding="utf-8",
        )
        claude_config_path.chmod(0o600)
        print(
            f"Preflight: updated ~/.claude/config.json customApiUrl -> "
            f"{proxy_anthropic_url}"
        )
    else:
        print("Preflight: ~/.claude/config.json already points at proxy")

    return proxy_anthropic_url, proxy_openai_url


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


def _ensure_oob_proxy_source() -> bool:
    """Make sure ``auth_proxy.py`` exists at the path supervisor expects.

    ``ensure_auth_proxy.sh`` looks for the script at
    ``${HYPERLOOM_ROOT}/OOB/oob_cli/auth_proxy.py`` (default
    ``/opt/hyperloom/OOB/oob_cli/auth_proxy.py``). On a fresh sandbox where
    ``kernel-agent/scripts/install.sh`` has NOT run yet, that file is absent
    and the supervisor silently noops + returns 1, leaving :4002 dead and
    Claude SDK requests hitting the gateway directly with ``x-api-key`` →
    HTTP 401 → "Waiting for first result" hang.

    This helper bootstraps just the ``auth_proxy.py`` source from a known
    bundle mount (``$OOB_SRC`` > ``/wekafs/fully-local/OOB`` > sibling
    ``inference_optimization/OOB``) so the supervisor can find + start it.
    Returns True if the file is present afterwards, False otherwise.
    """
    hyperloom_root = Path(os.environ.get("HYPERLOOM_ROOT", "/opt/hyperloom"))
    target_dir = hyperloom_root / "OOB" / "oob_cli"
    proxy_py = target_dir / "auth_proxy.py"
    if proxy_py.is_file():
        return True

    candidates: list[Path] = []
    env_src = os.environ.get("OOB_SRC", "").strip()
    if env_src:
        candidates.append(Path(env_src))
    candidates.extend(Path(p) for p in _OOB_SRC_CANDIDATES)

    for cand in candidates:
        try:
            if not cand or not cand.is_dir():
                continue
            if not (cand / "auth_proxy.py").is_file():
                continue
        except OSError:
            continue
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cand, target_dir, dirs_exist_ok=True)
        except (OSError, shutil.Error) as exc:
            print(
                f"Preflight: WARNING — failed to copy OOB source {cand} -> "
                f"{target_dir}: {exc}"
            )
            continue
        print(
            f"Preflight: bootstrapped auth_proxy.py from {cand} -> {target_dir}"
        )
        return True

    print(
        f"Preflight: WARNING — auth_proxy.py source not located at any of "
        f"$OOB_SRC / {_OOB_SRC_CANDIDATES}. The :4002 supervisor will "
        f"warn-and-skip; ANTHROPIC_BASE_URL stays at upstream and Claude "
        f"SDK may 401."
    )
    return False


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


def _check_node_claude_cli() -> None:
    """WARN-only presence check for the bundled ``claude`` / ``codex`` CLIs.

    ``claude_agent_sdk`` typically shells out to the bundled
    ``@anthropic-ai/claude-code`` CLI; without it on PATH the SDK falls
    back to a direct HTTP path that our auth-proxy still services, so this
    is informational rather than fatal. Same for ``codex`` (used by
    ``CodexBackend`` whenever Codex is on the wire — i.e. ``--critic-agent``
    / ``--critic-codex-bare`` and/or ``--kernel-codex``). ``node`` is a
    transitive dep — if it's missing, npm-based recovery via
    ``kernel-agent/scripts/install.sh`` won't work either.
    """
    missing = [t for t in ("node", "claude", "codex") if shutil.which(t) is None]
    if missing:
        print(
            f"Preflight: WARNING — CLI(s) not on PATH: {missing}. "
            f"ClaudeBackend / CodexBackend may fall back to direct HTTP. "
            f"Run kernel-agent/scripts/install.sh to bring them in."
        )


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    proxy_anthropic: str | None,
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
        f"({ENV_OVERRIDE_SESSION_DIR}="
        f"{os.environ.get(ENV_OVERRIDE_SESSION_DIR, '<unset>')}, "
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
    if proxy_anthropic:
        print(f"  proxy URLs          = {proxy_anthropic} (auth-proxy alive)")
    else:
        print(
            "  proxy URLs          = DIRECT — auth-proxy unavailable; "
            "Claude SDK may 401"
        )


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | None:
    """Probe ``<base_url>/models`` with retry; return set of model ids or None.

    Mirrors what the launcher had to do by hand (``terminals/6.txt``):

        curl -k -H "Authorization: Bearer $SAFE_API_KEY" \
             "https://gateway/api/v1/llm-proxy/v1/models" | jq '.data[].id'

    The gateway has a documented flake rate; we retry up to
    ``len(_CATALOG_RETRY_DELAYS_SEC)`` times with exponential backoff. SSL
    verify is OFF because the gateway cert is occasionally self-signed;
    we suppress the urllib3 InsecureRequestWarning so the diagnostics
    section stays readable.
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
                verify=False,
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
    proxy_urls: tuple[str, str] | None,
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

    base_url = ""
    if proxy_urls is not None:
        base_url = proxy_urls[1]  # OpenAI-compat URL keeps the /v1 suffix
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "")

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


def _preflight() -> tuple[str, str] | None:
    """Auto-install missing runtime deps and export auth aliases.

    1. Credentials fallback: env > $REPO_ROOT/.env (env always wins).
    2. Auth aliases for Claude/Codex CLIs from SAFE_API_KEY/OPENAI_BASE_URL.
    3. Python SDK (claude-agent-sdk / openai / httpx) auto-install.
    4. ``auth_proxy.py`` source bootstrap (so ensure_auth_proxy.sh has fuel).
    5. Auth-proxy + ~/.claude/config.json supervision.
    6. ROCm env hygiene (HIP_VISIBLE_DEVICES unset, GPU/shm sanity).
    7. ray + Magpie + InferenceX auto-install.
    8. node / claude / codex CLI presence check (WARN-only).
    9. Single canonical diagnostics block.

    Returns the ``(proxy_anthropic_url, proxy_openai_url)`` tuple from
    :func:`_ensure_auth_proxy_and_claude_config` so the caller
    (``_run_optimize``) can route the catalog probe through the proxy.
    """
    _load_dotenv_fallback()

    # --- Auth alias export ---
    safe_key = os.environ.get("SAFE_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if safe_key:
        for alias in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                      "ANTHROPIC_API_KEY", "OOB_API_KEY", "GEAK_API_KEY",
                      "LLM_API_KEY", "AMD_LLM_API_KEY"):
            os.environ.setdefault(alias, safe_key)
    # OOB / GEAK / LLM_API_BASE keep upstream URL: those clients speak Bearer
    # auth natively and do NOT need the auth-proxy. ANTHROPIC_BASE_URL and
    # OPENAI_BASE_URL are handled separately below — the auth-proxy step
    # force-overrides them so any externally-preset value (shell rc, .env,
    # k8s secret, container env) cannot bypass :4002.
    if base_url:
        for alias in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
            os.environ.setdefault(alias, base_url)

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

    # --- Bootstrap auth_proxy.py source (so the supervisor has fuel) ---
    _ensure_oob_proxy_source()

    # --- Ensure auth-proxy is running + ~/.claude/config.json points at it ---
    # The AMD primus-safe gateway only accepts Authorization: Bearer, but the
    # Claude CLI (bundled in claude_agent_sdk) sends x-api-key. The auth_proxy
    # on :4002 rewrites the header. Without this, Claude SDK hangs at
    # "Waiting for first result" / exits with code 1.
    #
    # Snapshot any externally-preset URLs BEFORE supervision so we can either
    # force-override them (proxy alive) or restore them (proxy unavailable).
    # The two env vars MUST stay consistent — either both proxy or both orig.
    orig_anthropic = os.environ.get("ANTHROPIC_BASE_URL", "")
    orig_openai = os.environ.get("OPENAI_BASE_URL", "")
    proxy_urls = _ensure_auth_proxy_and_claude_config(safe_key, base_url)
    if proxy_urls is not None:
        proxy_anthropic, proxy_openai = proxy_urls
        for var, want, prev in (
            ("ANTHROPIC_BASE_URL", proxy_anthropic, orig_anthropic),
            ("OPENAI_BASE_URL", proxy_openai, orig_openai),
        ):
            if os.environ.get(var) != want:
                os.environ[var] = want
                print(
                    f"Preflight: {var} {prev or '<unset>'} -> {want} "
                    f"(auth-proxy)"
                )
    else:
        # Proxy unavailable after retry — restore the originals so both vars
        # stay consistent with whatever the user had (no half-overridden state).
        for var, prev in (
            ("ANTHROPIC_BASE_URL", orig_anthropic),
            ("OPENAI_BASE_URL", orig_openai),
        ):
            if prev:
                os.environ[var] = prev
            else:
                os.environ.pop(var, None)
        print(
            "Preflight: WARNING — Claude/Codex SDKs may receive 401 "
            "without auth-proxy"
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
    check = subprocess.run(
        [magpie_python, "-c", "import Magpie"],
        capture_output=True,
    )
    if check.returncode != 0:
        workspace_root = os.environ.get("WORKSPACE_ROOT") or "/workspace"
        magpie_dir = Path(workspace_root) / "Magpie"
        if not (magpie_dir / "setup.py").exists() and not (magpie_dir / "pyproject.toml").exists():
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
        workspace_root = os.environ.get("WORKSPACE_ROOT") or "/workspace"
        # Check common mount points
        for candidate in (
            Path(workspace_root) / "Magpie" / "InferenceX",
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

    # --- Single canonical diagnostics block ---
    _emit_preflight_diagnostics(
        magpie_python=magpie_python,
        proxy_anthropic=(proxy_urls[0] if proxy_urls is not None else None),
    )

    return proxy_urls


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
    """
    chosen = getattr(args, "robustness_backend", None)
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
    return chosen


def _build_robustness_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-default ``request.options`` overrides from CLI flags.

    Only emits keys the operator actually passed so the runtime CLI
    falls back to its own defaults / env-discovery for the rest.
    """
    options: dict[str, Any] = {}
    server_url = getattr(args, "robustness_server_url", None)
    if server_url is not None:
        options["robustness_server_url"] = server_url
    llm_rca = getattr(args, "robustness_llm_rca", None)
    if llm_rca is not None:
        options["llm_rca_enabled"] = bool(llm_rca)
    return options


async def _run_optimize(args: argparse.Namespace) -> int:
    proxy_urls = _preflight()

    # Hard-gate Claude model BEFORE any session work. Mutates args.claude_model
    # in-place when falling back to opus-4-6; aborts with sys.exit(2) if the
    # gateway catalog cannot be probed or neither allowed model is present.
    catalog_ids = _validate_and_resolve_claude_model(args, proxy_urls)
    _smoke_test_codex_model(args, catalog_ids)

    if args.resume:
        # Resume mode: session_dir is fixed at /workspace/hyperloom (or
        # $INFERENCE_OPTIMIZER_SESSION_DIR for tests). We re-mkdir the
        # skeleton (idempotent) so a partially-initialised previous run
        # is healed before we touch state.
        session_dir = make_session_dir()
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
        # Honour persisted kernel_enabled flag on resume; CLI --no-kernel
        # can still override on a previously-enabled session.
        if not state.kernel_enabled:
            args.no_kernel = True
            print("  kernel agent          : DISABLED (persisted from original run)")

        # CRITICAL: a leftover stop_reason from the prior run (most often
        # "time_exhausted") fools Orchestration into thinking the work is
        # already done — it just heartbeats forever. Clear it so the new
        # run has a clean signal. The Coordinator's run() always re-sets
        # stop_reason at exit anyway.
        prior_crash = state.crash_count
        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            # Reset persisted crash_count so a fresh resume isn't immediately
            # tripped into "emergency" by accumulated failures from prior runs
            # (e.g. authentication errors before .env was loaded).
            state.crash_count = 0
            state.save(session_dir)
            print(
                f"  → cleared stop_reason and reset crash_count "
                f"(was {prior_crash}) for fresh resume"
            )
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
        # Session-wide; mixing sglang/vllm in one session is not supported.
        framework = (
            (args.framework or os.environ.get("FRAMEWORK", "")).strip().lower()
            or "sglang"
        )
        if framework not in ("sglang", "vllm"):
            print(
                f"ERROR: --framework must be sglang or vllm (got {framework!r}); "
                "set $FRAMEWORK accordingly or pass --framework",
                file=sys.stderr,
            )
            sys.exit(2)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")

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

        # session_dir is fixed at /workspace/hyperloom (override:
        # $INFERENCE_OPTIMIZER_SESSION_DIR). Each sandbox is single-use,
        # so collision detection is unnecessary; mkdir -p is enough.
        session_dir = make_session_dir()
        manifest = write_manifest(session_dir, args=args)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)
        _seed_shared_state(
            session_dir, args, session_id=manifest["session_id"],
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
        # operator hasn't already pinned it.
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
    os.environ["INFERENCE_OPTIMIZER_SESSION_DIR"] = str(session_dir)
    # Production: enable strict path-containment checks in PolicyGate so
    # any LLM-emitted intent whose path field escapes session_dir lands
    # as `policy_denied` in its inbox. Tests omit this and keep the
    # legacy lenient mode for fixture paths under /tmp.
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"

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
    _register_executors(
        coordinator,
        no_kernel=no_kernel,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
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

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
        )
    finally:
        await coordinator.stop()

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
        "--framework", choices=["sglang", "vllm"], default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
             "--framework > $FRAMEWORK env > sglang (default). Selection is "
             "session-wide; mixing sglang and vllm in a single session is "
             "not supported.",
    )
    opt.add_argument("--max-hours", type=float, default=2.0,
                      help="Wall-clock budget in hours (default 2.0)")
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
                      help="Resume the session at the canonical session_dir "
                           "(/workspace/hyperloom by default, or "
                           "$INFERENCE_OPTIMIZER_SESSION_DIR for tests). "
                           "Skips the SharedState seed and lets the "
                           "Coordinator replay the prior event log + "
                           "state.json. Refuses to start if manifest.json or "
                           "state.json is missing.")
    opt.add_argument("--model-class", type=str, default=None,
                      help="Optional model class hint (dense_8B / moe_mla / ...)")
    opt.add_argument("--target-summary", type=str, default=None,
                      help="Free-text goal summary surfaced in prompts")
    opt.add_argument(
        "--compare-against-gpu", type=str, default=None,
        help=(
            "Reference GPU hardware key for external baseline comparison "
            "(e.g. b300 / mi355x / h200). When set, the target_analysis "
            "action fetches the matching reference data point from "
            "InferenceX (https://inferencex.semianalysis.com) and writes "
            "$SESSION_DIR/target_analysis/target_baseline.json + a short "
            "MD report. The data is REPORT-ONLY: it does not influence "
            "Objective, scoring, or any agent prompt. Other dimensions "
            "(model / framework / precision / ISL / OSL) are derived "
            "from --model and the standard FRAMEWORK / PRECISION / ISL / "
            "OSL env vars. Unset = keep target_analysis as a no-op stub."
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
