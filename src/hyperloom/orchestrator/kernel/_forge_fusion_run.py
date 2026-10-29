# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One autonomous kernel-fusion run: the agent, the sandbox, the trace and the CLI call."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common import codex_session, llm_config

from . import forge_fusion
from ..actions.executors._kernel_agent_tool import HandlerResult
from ._forge_lane_budget import (
    _forge_fusion_timeout_sec,
    _fusion_session_serve_args,
    _lane_budget,
    _unfunded_lane_skip,
)
from .lane_budget import LANE_FUSION


# forge-fusion (autonomous kernel fusion)
def _forge_fusion_available() -> bool:
    """Check that KernelForge's fusion pipeline is importable.

    Probes the subpackage rather than ``kernelforge``: an installation
    predating the fusion absorption would satisfy the parent import and only
    fail once the subprocess rejected ``forge-fuse``. PATH is not consulted
    because the tool is invoked through ``sys.executable -m``.
    """
    try:
        return importlib.util.find_spec("kernelforge.workflows.forge_fusion") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _resolve_fusion_decode_trace(state) -> str:
    """Reuse this run's PRELUDE/roofline decode trace for fusion discovery.

    forge-fusion's discover stage needs a CUDA-graph-disabled decode kineto trace.
    PRELUDE/roofline already captured one and promoted it into
    ``state.last_profile_trace`` alongside the workload it was measured on
    (``SharedState.record_profile_workload``), so the lane reuses that rather than
    re-profiling. ``state.last_profile_trace`` is the only source: the fusion
    opportunities discovered below are attributed to the run that produced the trace,
    and a caller-supplied path would file this run's decisions under another workload.

    ``last_profile_trace`` holds a single trace file for AgentX profiles and for any
    round that merged its ranks, and the capture directory otherwise --
    ``_preferred_main_trace_path`` in ``actions/executors/profile.py`` returns
    ``trace_dir`` when no ``merged-*`` file exists, which is every non-AgentX
    sglang/vLLM profile. A directory therefore resolves to the
    newest capture in it, which is this run's too: the directory is the round's own
    ``trace_dir``.
    """
    trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    if not trace:
        return ""
    path = Path(trace)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return ""
    captures = sorted(
        list(path.glob("*.trace.json.gz")) + list(path.glob("*.trace.json")) + list(path.glob("*.json.gz")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(captures[0]) if captures else ""


def _active_forge_fusion_env_flags(state: Any) -> dict[str, str]:
    """Return active env flags only when forge-fusion itself is current_best."""
    current_best = getattr(state, "current_best", None) or {}
    if str(current_best.get("action") or "") != "fusion":
        return {}
    envs = current_best.get("extra_envs")
    if not isinstance(envs, dict):
        return {}
    active: dict[str, str] = {}
    for key, val in envs.items():
        name = str(key)
        value = str(val)
        if "_FUSED" not in name.upper():
            continue
        if value.strip().lower() in ("", "0", "false", "no", "off", "none"):
            continue
        active[name] = value
    return active


def _resolve_forge_agent(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Resolve the Forge agent backend and model as one decision.

    :func:`llm_config.resolve_agent_backend` decides the backend (an explicit
    ``INFERENCE_OPTIMIZER_AGENT_BACKEND``, else the credential-ranked choice), so
    this role cannot disagree with the orchestration role or the Forge registry
    about which backend a box is configured for. Model id precedence is owned by
    :func:`llm_config.resolve_forge_llm_model`.

    Raises:
        ValueError: If ``INFERENCE_OPTIMIZER_AGENT_BACKEND`` names no known backend.
    """
    source = env if env is not None else os.environ
    agent_backend = llm_config.resolve_agent_backend(source)
    return agent_backend, llm_config.resolve_forge_llm_model(agent_backend, env=source)


def _resolve_forge_fusion_sandbox_mode(*, agent_backend: str, env: Mapping[str, str] | None = None) -> str:
    """Resolve and validate the sandbox policy recorded for forge-fusion.

    Codex delegates the default and validation to the canonical Hyperloom
    resolver, including its operator opt-in for ``bypass``. Claude records
    ``workspace-write`` as the stable audit default.

    Raises:
        CodexSessionUnavailableError: If the configured mode is unknown or bypass
            lacks the operator mode confirmation.
    """
    if agent_backend == "claude":
        return codex_session.DEFAULT_CODEX_SANDBOX_MODE
    return codex_session.resolve_codex_sandbox_mode(env=dict(env) if env is not None else None)


def _fusion_failure(error_class: str, error: str, *, status: str = "failed") -> HandlerResult:
    """A forge-fusion run that ended before forge-fuse started."""
    return {
        "status": status,
        "backend": "forge",
        "engine": "forge_fusion",
        "error_class": error_class,
        "error": error,
        "decision": "REVERT",
        "kept": False,
    }


def _fusion_precondition_refusal(state: Any, *, model_path: str, trace_path: str) -> HandlerResult | None:
    """Why this session cannot run forge-fusion now, or ``None`` when it can."""
    active_fusion_flags = _active_forge_fusion_env_flags(state)
    if active_fusion_flags:
        return {
            "status": "complete",
            "backend": "forge",
            "engine": "forge_fusion",
            "micro_decision": "already_active",
            "decision": "REVERT",
            "kept": False,
            "requires_e2e_validation": False,
            "active_env_flags": active_fusion_flags,
            "reason": (
                "current_best is already a forge-fusion KEEP; "
                "skip forge-fusion to avoid rerunning the same adopted source patch"
            ),
            "source": "forge_fusion",
        }
    if not _forge_fusion_available():
        return _fusion_failure(
            "forge_fusion_not_found",
            "KernelForge fusion pipeline not found. Install via 'pip install <KernelForge>[claude,codex]'.",
        )
    if not model_path:
        return _fusion_failure("model_path_missing", "model_path is required")
    if not trace_path:
        recorded = str(getattr(state, "last_profile_trace", "") or "").strip()
        return _fusion_failure(
            "decode_trace_missing",
            "no decode trace available for fusion discovery "
            f"(state.last_profile_trace={recorded or '(empty)'}; run profile/roofline first)",
            status="skipped",
        )
    return None


async def run_fusion_handler(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Autonomous kernel fusion via the forge-fusion CLI, awaited by KernelPhase at KERNEL entry.

    Resolves one provider-compatible agent backend, model, and validated sandbox
    policy, then runs forge-fuse through :func:`forge_fusion.run_forge_fusion`,
    which returns the fusion contract result. A KEPT fusion carries a source patch + env flags
    and ``requires_e2e_validation`` so the integrate gate confirms the
    end-to-end gain. Reuses the PRELUDE decode trace (no re-profiling); only
    ``payload['task_id']`` is read, to name the workspace.
    """
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    model_path = str(getattr(state, "model_path", "") or os.environ.get("MODEL_PATH") or "").strip()
    trace_path = _resolve_fusion_decode_trace(state)
    refusal = _fusion_precondition_refusal(state, model_path=model_path, trace_path=trace_path)
    if refusal is not None:
        return refusal

    framework = str(getattr(state, "framework", "") or "sglang").strip().lower()
    try:
        agent_backend, llm_model = _resolve_forge_agent()
    except ValueError as exc:
        return _fusion_failure("invalid_agent_backend", str(exc))
    try:
        agent_sandbox_mode = _resolve_forge_fusion_sandbox_mode(agent_backend=agent_backend)
    except RuntimeError as exc:
        return _fusion_failure("invalid_agent_sandbox_mode", str(exc))
    fusion_lane = _lane_budget(state, LANE_FUSION)
    if fusion_lane is None:
        return {**_unfunded_lane_skip(LANE_FUSION), "engine": "forge_fusion"}

    workspace = session_dir / "runs" / "fusion" / str(payload.get("task_id") or "kernel_entry_fusion")
    workspace.mkdir(parents=True, exist_ok=True)
    result = await forge_fusion.run_forge_fusion(
        {
            "trace_path": trace_path,
            "model_path": model_path,
            "framework": framework,
            "output_dir": str(workspace),
            "discover_mode": "llm",
            "agent_backend": agent_backend,
            "llm_model": llm_model,
            "agent_sandbox_mode": agent_sandbox_mode,
            "max_turns": int(os.environ.get("FORGE_FUSION_MAX_TURNS") or 100),
            "gpu": "0",
            "timeout": _forge_fusion_timeout_sec(lane_budget_sec=fusion_lane.budget_sec),
            # How many recipes the lane's share pays for. Omitted when none could be
            # derived, which leaves forge-fuse on every discovered recipe.
            **({"max_recipes": fusion_lane.max_targets} if fusion_lane.max_targets > 0 else {}),
            **_fusion_session_serve_args(state, framework=framework, model_path=model_path),
        }
    )
    return {
        "backend": "forge",
        "engine": "forge_fusion",
        "workspace": str(workspace),
        "framework": framework,
        "model_path": model_path,
        "agent_backend": agent_backend,
        "llm_model": llm_model,
        "agent_sandbox_mode": agent_sandbox_mode,
        "source": "forge_fusion",
        **result,
    }
