# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run()."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common import llm_config
from hyperloom.common.env import EnvValueError, env_bool
from hyperloom.common.gpu_partition import PartitionModel
from hyperloom.common.llm_config import CLAUDE_OAUTH_TOKEN_ENV
from hyperloom.common.llm_headers import parse_custom_headers
from .executors import (
    _build_specialist_executor,
    _register_executors,
)
from .kb import (
    _bootstrap_recipe_kb,
    _bootstrap_knowledge_plane,
)
from .backends import (
    CriticSetup,
    ModelRouting,
    _build_backends,
    _build_proposal_scorer,
    _orchestration_banner,
    critic_review_target,
    critic_on_codex_side,
)
from .model_gate import (
    _load_model_max_position_embeddings,
    _finish_model_gate,
    _preflight_context_window,
    _preflight_model_config_compat,
    _preflight_unsupported_model_arch,
    _record_resumed_model_gate,
    _resolve_max_model_len,
    _start_model_gate,
)
from ..gpu_types import (
    _autodetect_gpu_type,
    gpu_runner_type,
    _resolve_amd_gpu_type,
    _resolve_gpu_type,
    require_runner_framework,
    warn_family_runner,
)
from ..model_config_utils import (
    summarize_model_config,
)
from .bootstrap import (
    LaunchInfo,
    _emit_launch_info,
    constraint_budget_resume_conflict,
    resolve_launch_gpu_power,
    _begin_resume_leg,
    _resume_budget_lines,
    _print_final_summary,
    _print_session_skeleton,
    _reconcile_crash_count,
    open_plug_host,
    resolve_plug_manifest,
    warn_without_target_gpu,
    running_plug_host,
    _seed_shared_state,
    _snapshot_system_prompts,
    agentx_state_is_stale,
    resume_gpu_power_settings,
    latency_budget_scope_error,
    parse_operator_extra_env,
    resolve_framework_version,
    resolve_model_display_name,
)
from hyperloom.orchestrator.actions.executors._aiter_jit import clean_stale_aiter_locks
from hyperloom.common.perf_metric import agentx_enabled as _agentx_enabled

from .credentials import (
    _CLAUDE_ALLOWED_MODELS as _CLAUDE_ALLOWED_MODELS,
    _CODEX_FALLBACK_MODELS as _CODEX_FALLBACK_MODELS,
    _CATALOG_RETRY_DELAYS_SEC as _CATALOG_RETRY_DELAYS_SEC,
    _CRITIC_AGENT_ROOT_ENV as _CRITIC_AGENT_ROOT_ENV,
    _resolve_agent_root as _resolve_agent_root,
    _validate_agent_runtime as _validate_agent_runtime,
)
from .multi_node import (
    _prepare_multi_node_state as _prepare_multi_node_state,
    _resolve_gpus_per_node,
    _resolve_mn_backend as _resolve_mn_backend,
)
from .quantization import (
    _run_quantization_prelude as _run_quantization_prelude,
)
from .recover import (
    _run_recover_session as _run_recover_session,
)


__all__ = ["main"]
from .. import framework_registry
from ..session.manifest import load_manifest, write_manifest
from ..protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator, CoordinatorSettings, RunOptions
from hyperloom.orchestrator.collaborator import OrchestrationPrompt as _OrchestrationPrompt
from hyperloom.inference_optimizer.framework_paths import resolve_framework_tree, resolve_kernel_search_roots
from hyperloom.orchestrator.state.objective import AnyObjective, Objective, build_objective
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    TRANSPORT_TOOLS,
    build_orchestration_prompt,
    default_enabled_actions,
    PromptAssets,
    PromptCycle,
    SessionContext,
)
from hyperloom.inference_optimizer.breakdown.stop_reasons import SUPERVISOR_RESTART_REASON

from ..session.lock import SessionAlreadyRunning, SessionLock, SessionLockPathError
from ..session.paths import ENV_USER_DATA_PATH, asset_system_prompts_dir, make_session_dir


log = logging.getLogger("hyperloom.inference_optimizer.cli")

from hyperloom.common.workload_defaults import (
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from .gpu_power_apply import gpu_power_restore_guard
from .custom_specialist import (
    CUSTOM_SPECIALIST_ONLY_ENV,
    _apply_custom_specialist_resume as _apply_custom_specialist_resume,
    _custom_specialist_conflict as _custom_specialist_conflict,
    _load_custom_specialist_args as _load_custom_specialist_args,
)
from .parser import (
    DEFAULT_MAX_HOURS,
    _build_parser as _build_parser,
    _positive_int_arg as _positive_int_arg,
    _redact_unknown_args as _redact_unknown_args,
)
from .preflight import (
    _check_gfx_arch_resolvable,
    _mark_pending_install_event_failed,
    _persist_install_event,
    _preflight as _preflight,
)


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only ``orchestration.md`` fragment consumed by ``prompt_builder``."""
    return asset_system_prompts_dir() / "orchestration.md"


def _normalise_framework_name(value: str | None) -> str:
    """Normalize a framework string for equality checks."""
    return str(value or "").strip().lower().replace("_", "-")


_XDIT_MODEL_NAME_ATTR = "xdit_model_name"
_XDIT_MODEL_NAME_ENV = "XDIT_MODEL_NAME"


def _apply_operator_supplied_paths(args: Any, framework: str) -> None:
    """Publish ``--framework-path`` / ``--benchmark-scripts-dir`` as env."""
    fatal: list[str] = []
    if framework == "custom":
        from hyperloom.orchestrator.actions.executors.benchmark_backend import (
            BENCHMARK_BACKEND_ENV,
            DEFAULT_BENCHMARK_BACKEND,
        )

        # Matches install.sh's normalisation so the two gates agree on a value like " Bypass "; anything else,
        # including unset, is refused.
        backend = os.environ.get(BENCHMARK_BACKEND_ENV, "").strip().lower()
        if backend != "bypass":
            shown = f"{backend!r}" if backend else f"unset (defaults to {DEFAULT_BENCHMARK_BACKEND!r})"
            fatal.append(
                f"--framework custom requires {BENCHMARK_BACKEND_ENV}=bypass; it is {shown}. "
                "The default backend cannot run an operator-supplied script."
            )
    for flag, value, env_name in (
        ("--framework-path", getattr(args, "framework_path", None), "FRAMEWORK_REPO_PATH"),
        (
            "--benchmark-scripts-dir",
            getattr(args, "benchmark_scripts_dir", None),
            "HYPERLOOM_BYPASS_SCRIPTS_DIR",
        ),
    ):
        raw = str(value or "").strip()
        # An exported-but-empty var is not a choice, it is an unset one; treating it as set would let a stray `export
        # FRAMEWORK_REPO_PATH=` silently swallow the flag.
        already = os.environ.get(env_name, "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_dir():
                fatal.append(f"{flag}={raw!r} is not a directory")
                continue
            if not already:
                os.environ[env_name] = str(path.resolve())
        elif framework == "custom" and not already:
            fatal.append(f"{flag} is required for --framework custom (or export {env_name})")

    if framework == "xdit":
        xdit_name = str(getattr(args, _XDIT_MODEL_NAME_ATTR, None) or "").strip()
        already_xdit = os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()
        if xdit_name and not already_xdit:
            os.environ[_XDIT_MODEL_NAME_ENV] = xdit_name
        elif not xdit_name and not already_xdit:
            fatal.append(
                "--xdit-model-name is required for xDiT runs. "
                "Pass the HuggingFace repo ID, e.g. 'black-forest-labs/FLUX.2-dev'."
            )

    if fatal:
        for line in fatal:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(2)


def _restore_operator_supplied_paths_from_state(args: Any, state: SharedState) -> None:
    """Re-export the session paths persisted on ``state`` when this resume omitted them."""
    from hyperloom.inference_optimizer.framework_paths import point_session_framework_root
    from hyperloom.orchestrator.actions.executors.benchmark_backend import BENCHMARK_BACKEND_ENV

    cli_repo = str(getattr(args, "framework_path", None) or "").strip()
    cli_scripts = str(getattr(args, "benchmark_scripts_dir", None) or "").strip()
    if not cli_repo and not os.environ.get("FRAMEWORK_REPO_PATH", "").strip():
        archived = str(getattr(state, "framework_repo_path", "") or "").strip()
        if archived:
            point_session_framework_root(archived, state.framework)
    if not cli_scripts and not os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip():
        archived = str(getattr(state, "bypass_scripts_dir", "") or "").strip()
        if archived:
            os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] = archived
    if not os.environ.get(BENCHMARK_BACKEND_ENV, "").strip():
        archived = str(getattr(state, "benchmark_backend", "") or "").strip()
        if archived:
            os.environ[BENCHMARK_BACKEND_ENV] = archived
    if (
        not str(getattr(args, _XDIT_MODEL_NAME_ATTR, None) or "").strip()
        and not os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()
    ):
        archived = str(getattr(state, _XDIT_MODEL_NAME_ATTR, "") or "").strip()
        if archived:
            os.environ[_XDIT_MODEL_NAME_ENV] = archived


def _require_custom_entrypoint(framework: str, gpu_type: str | None = None) -> None:
    """Fail at launch when the selected runner cannot serve the framework."""
    require_runner_framework(gpu_type or "", framework)
    if str(framework or "").strip().lower() != "custom":
        return
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_scriptable_runtime_defaults,
    )
    from hyperloom.orchestrator.actions.executors.bypass_scriptable import (
        resolve_scriptable_script,
        scriptable_script_candidates,
    )

    bench: dict[str, Any] = {"framework": "custom"}
    envs: dict[str, Any] = {}
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=gpu_type,
        explicit_benchmark_script=False,
    )
    runner_type = str(bench.get("runner_type") or "mi355x")
    inferencex_root = (
        os.environ.get("INFERENCEX_PATH", "").strip() or os.environ.get("MAGPIE_INFERENCEX_PATH", "").strip()
    )
    script = resolve_scriptable_script("custom", runner_type, inferencex_root, bench)
    if script is not None:
        return
    name = f"custom_{runner_type}.sh"
    candidates = scriptable_script_candidates("custom", runner_type, inferencex_root, bench)
    tried = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
    print(
        f"ERROR: --framework custom could not resolve a benchmark entrypoint ({name}). Tried:\n{tried}",
        file=sys.stderr,
    )
    sys.exit(2)


def _persist_operator_supplied_paths(state: SharedState) -> None:
    """Mirror the live custom-workload env back onto ``state`` for the next resume."""
    from hyperloom.orchestrator.actions.executors.benchmark_backend import (
        BENCHMARK_BACKEND_ENV,
    )

    state.framework_repo_path = os.environ.get("FRAMEWORK_REPO_PATH", "").strip()
    state.bypass_scripts_dir = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    state.benchmark_backend = os.environ.get(BENCHMARK_BACKEND_ENV, "").strip().lower()
    state.xdit_model_name = os.environ.get(_XDIT_MODEL_NAME_ENV, "").strip()


def _registered_framework_or_exit(requested: str, *, hint: str) -> str:
    """Return ``requested`` normalised, or the default when blank; exit 2 when it names no registered framework."""
    framework = requested.strip().lower() or framework_registry.DEFAULT_FRAMEWORK
    if not framework_registry.is_supported(framework):
        print(
            f"ERROR: framework must be one of {', '.join(framework_registry.names())} (got {framework!r}); {hint}",
            file=sys.stderr,
        )
        sys.exit(2)
    return framework


def _enforce_expected_framework(
    framework: str,
    *,
    expected: str | None = None,
) -> None:
    """Fail fast when a launcher-pinned expected framework is violated."""
    actual = _normalise_framework_name(framework)
    expected_raw = (
        expected
        if expected is not None
        else (os.environ.get("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", "") or os.environ.get("EXPECTED_FRAMEWORK", ""))
    )
    wanted = _normalise_framework_name(expected_raw)
    if not wanted:
        return
    if wanted != actual:
        print(
            "ERROR: framework mismatch: "
            f"EXPECTED_FRAMEWORK={wanted!r} but resolved framework={actual!r}. "
            "Refusing to launch because this would run a different backend "
            "than the operator requested.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Summarise an objective into the ``(kind, value)`` pair the prompt expects."""
    kind = objective.kind()
    if isinstance(objective, AnyObjective):
        return kind, objective.describe()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "target_within_pct"):
        value = float(getattr(objective, "target_within_pct"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


@dataclass(frozen=True)
class OrchestrationPromptContext:
    """What the Orchestration prompt says about the session itself; fixed for the run while cycles and phases move."""

    no_kernel: bool
    framework: str
    objective: Objective
    max_minutes: int
    no_framework_agent: bool = False
    transport: str = TRANSPORT_TOOLS
    benchmark_mode: str = ""
    agentx_corpus_shape: Mapping[str, Any] | None = None
    agentx_grading: Mapping[str, Any] | None = None
    agentx_backend: str = ""


def _build_orchestration_prompt(
    context: OrchestrationPromptContext,
    *,
    macro_cycle: int = 0,
    cycle_directive: str = "",
    cycle_strategy: Mapping[str, Any] | None = None,
    phase: str = "",
) -> str:
    """Compose the Orchestration system prompt from typed inputs (``--orch-prompt`` overrides)."""
    no_kernel, no_framework_agent, framework = context.no_kernel, context.no_framework_agent, context.framework
    enabled = default_enabled_actions(no_kernel=no_kernel, no_optimize=no_framework_agent)
    kind, value = _objective_summary_for_prompt(context.objective)
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=enabled,
        session=SessionContext(
            framework=framework,
            kernel_enabled=not no_kernel,
            framework_agent_phase_enabled=not no_framework_agent,
            objective_kind=kind,
            objective_value=value,
            max_minutes=int(context.max_minutes),
            framework_source_roots=resolve_kernel_search_roots(),
            session_framework_tree=resolve_framework_tree(framework),
            transport=context.transport,
            benchmark_mode=context.benchmark_mode,
            agentx_corpus_shape=context.agentx_corpus_shape,
            agentx_grading=context.agentx_grading,
            agentx_backend=context.agentx_backend,
        ),
        cycle=PromptCycle(
            macro_cycle=int(macro_cycle), cycle_directive=cycle_directive, cycle_strategy=cycle_strategy, phase=phase
        ),
        assets=PromptAssets(rules_fragment_path=_orchestration_rules_fragment_path()),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``orchestrator/prompts/critic.md``."""
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


# Per-attempt read timeout for the gateway /models catalog probe.
try:
    _CATALOG_REQUEST_TIMEOUT_SEC = float(
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_TIMEOUT_SEC", "5.0") or "5.0"
    )
except (TypeError, ValueError):
    _CATALOG_REQUEST_TIMEOUT_SEC = 5.0


def _should_remote_probe_gpu(args: argparse.Namespace) -> bool:
    """Return whether the GPU type should be probed on the remote cluster."""
    if int(getattr(args, "nodes", 1) or 1) < 2:
        return False
    backend = (getattr(args, "mn_backend", None) or "rayjob").strip().lower()
    return backend in ("rayjob", "infera")


def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """Validate atom-specific CLI knobs: the ``--nodes>=2`` fail-fast guard (IR-8)."""
    auto_disabled: list[str] = []
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        "  framework=atom: no auto-disable applied (framework-agent + "
        "profile / roofline / TraceLens all wired for atom); "
        "--nodes>=2 guard active — see SKILL.md IR-8"
    )
    return auto_disabled


# Exit code for "another optimizer already owns this session".
SESSION_BUSY_EXIT_CODE = 3


def _acquire_session_lock_or_exit(session_dir: Path) -> SessionLock:
    """Take the single-optimizer session lock or exit ``SESSION_BUSY_EXIT_CODE``."""
    lock = SessionLock(session_dir)
    try:
        lock.acquire()
    except SessionAlreadyRunning as exc:
        print(
            f"ERROR: {exc}. Refusing to start a second optimizer on the same "
            f"session (would corrupt shared leases / state.json). If the owner "
            f"is truly dead, wait for the OS to drop the lock or remove "
            f"{lock.path} before retrying.",
            file=sys.stderr,
        )
        sys.exit(SESSION_BUSY_EXIT_CODE)
    return lock


# Sentinel returned by _probe_llm_catalog when the gateway has no /models route (HTTP 404/405).
_CATALOG_NO_MODELS_ENDPOINT: frozenset[str] = frozenset()


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | frozenset[str] | None:
    """Probe ``<base_url>/models`` with retry (gateway flakes); return set of model ids or None."""

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # httpx should already be installed; return None so the caller decides.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    probe_url = base_url.rstrip("/") + "/models"
    headers = _catalog_probe_headers(base_url=base_url, api_key=api_key)

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
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} failed: {last_err}")
            continue
        if resp.status_code in (404, 405):
            # The endpoint has no /models route; not a transient/auth error, so stop retrying and signal "no catalog
            # endpoint" distinctly.
            print(
                f"Preflight: catalog probe got HTTP {resp.status_code} for "
                f"{probe_url}; endpoint exposes no /models route"
            )
            return _CATALOG_NO_MODELS_ENDPOINT
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} got {last_err}")
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} returned non-JSON: {last_err}")
            continue
        ids: set[str] = set()
        for model in data.get("data") or []:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                continue
            model_id = model["id"]
            ids.add(model_id)
            ids.add(_catalog_compare_model_id(model_id))
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(f"Preflight: catalog probe exhausted {len(delays)} attempts ({last_err}); cannot validate model availability")
    return None


def _catalog_probe_headers(*, base_url: str, api_key: str) -> dict[str, str]:
    """Build headers for a direct ``<base_url>/models`` probe."""
    openai_base = (os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    probe = (base_url or "").strip().rstrip("/")
    probing_openai = bool(openai_base) and (probe == openai_base or probe.startswith(openai_base + "/"))
    env_name = "OPENAI_CUSTOM_HEADERS" if probing_openai else "ANTHROPIC_CUSTOM_HEADERS"
    headers = parse_custom_headers(os.environ.get(env_name))
    if api_key and not any(name.lower() == "authorization" for name in headers):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _catalog_compare_model_id(model_id: str) -> str:
    """Normalize catalog IDs for preflight comparison only."""
    text = str(model_id or "").strip()
    lowered = text.lower()
    if text.lower().startswith("claude-"):
        return lowered.replace(".", "-")
    return lowered


def _same_gateway(anthropic_url: str, openai_url: str) -> bool:
    """True when both protocol sides are served by one gateway."""
    if not anthropic_url or not openai_url:
        return False
    if anthropic_url == openai_url:
        return True
    from urllib.parse import urlsplit

    return urlsplit(anthropic_url).netloc == urlsplit(openai_url).netloc


def _codex_model_should_follow_claude() -> bool:
    """True when the operator supplied only Anthropic config."""
    return llm_config.is_anthropic_only()


def _claude_model_should_follow_codex() -> bool:
    """True when the operator supplied only OpenAI-compatible config."""
    return llm_config.is_openai_only()


def _catalog_probe_has_no_credential() -> bool:
    """True when a Claude subscription token is the only credential available."""
    if not os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip():
        return False
    # Any other Anthropic-side form can authenticate the probe, so the list of what disqualifies "oauth-only" is the
    # registry minus the token itself, derived rather than restated.
    bearer_capable = (
        *(n for n in llm_config.ANTHROPIC_CREDENTIAL_ENV_ORDER if n != CLAUDE_OAUTH_TOKEN_ENV),
        "OPENAI_API_KEY",
    )
    return not any(os.environ.get(name, "").strip() for name in bearer_capable)


def _custom_orch_model_allowed() -> bool:
    """Whether orchestration may use a model outside the AMD Claude allowlist; only an explicit false-token denies."""
    return env_bool("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", True)


def _critic_agent_runtime_needed(critic_choice: str) -> bool:
    """Whether the selected critic path will actually instantiate critic-agent."""
    return critic_choice == "agent"


def _reject_unusable_claude_model(chosen: str, *, allow_custom: bool) -> None:
    """Refuse a model id the static allowlist cannot accept, before any gateway is asked."""
    if not allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of "
            f"{list(_CLAUDE_ALLOWED_MODELS)} (best first). Refusing to start. "
            f"For a non-AMD gateway, set "
            f"INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1 to use a custom "
            f"orchestration model validated against your gateway catalog.",
            file=sys.stderr,
        )
        sys.exit(2)
    if allow_custom and not chosen:
        print(
            "ERROR: --claude-model is empty but "
            "custom orchestration model support is enabled; pass an explicit "
            "model id. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)


def _catalog_probe_candidates(
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool,
) -> list[tuple[str, str]]:
    """The gateways worth asking for a model catalog, in the order they are asked.

    The Claude catalog must come from the Anthropic side unless the Claude
    model follows Codex, in which case the OpenAI side is the authority.
    """
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not anthropic_url:
        anthropic_url = resolved_urls[0]
    if not openai_url:
        openai_url = resolved_urls[1]
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    # OpenAI-side key only.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    candidates: list[tuple[str, str]] = []
    if claude_follows_codex:
        if openai_url:
            candidates.append((openai_url, openai_key))
        elif anthropic_url:
            candidates.append((anthropic_url, anthropic_key))
        return candidates
    # A keyless Anthropic candidate can only fail — a subscription token is not a catalog credential — so it
    # must not consume the one probe slot and leave a configured OpenAI gateway unverified.
    if anthropic_url and anthropic_key:
        candidates.append((anthropic_url, anthropic_key))
    if openai_url and ((anthropic_url and _same_gateway(anthropic_url, openai_url)) or not candidates):
        # One gateway serving both protocols, or the only side left once a keyless Anthropic side was skipped
        # above.
        candidates.append((openai_url, openai_key))
    return candidates


def _probe_claude_catalog(
    resolved_urls: tuple[str, str],
    *,
    chosen: str,
    claude_follows_codex: bool,
):
    """The gateway's model catalog, and whether a probe was made at all.

    ``(None, False)`` means the probe was skipped because the only credential
    cannot authenticate one -- the static allowlist gate has already run, and
    it is the only check the probe would have added.
    """
    override_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not override_url and _catalog_probe_has_no_credential():
        print(
            "Preflight: catalog probe skipped: oauth-only credential "
            f"(CLAUDE_CODE_OAUTH_TOKEN); cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None, False
    if override_url:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        return _probe_llm_catalog(base_url=override_url, api_key=api_key), True
    catalog_ids: set[str] | frozenset[str] | None = None
    seen_urls: set[str] = set()
    for cand_url, cand_key in _catalog_probe_candidates(resolved_urls, claude_follows_codex=claude_follows_codex):
        if not cand_url or cand_url in seen_urls:
            continue
        seen_urls.add(cand_url)
        catalog_ids = _probe_llm_catalog(base_url=cand_url, api_key=cand_key)
        # A missing /models route is the only answer worth re-asking on the other side, and only because a
        # dual-protocol gateway lists its models there.
        if catalog_ids is not _CATALOG_NO_MODELS_ENDPOINT:
            break
    return catalog_ids, True


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool | None = None,
) -> set[str] | None:
    """Gate Claude model selection against the gateway catalog; mutates ``args.claude_model``."""
    if claude_follows_codex is None:
        claude_follows_codex = _claude_model_should_follow_codex()
    chosen = (args.claude_model or "").strip()
    # Custom orchestration models are enabled by default; the gateway catalog probe below is the sole gate.
    allow_custom = _custom_orch_model_allowed()
    _reject_unusable_claude_model(chosen, allow_custom=allow_custom)
    catalog_ids, probed = _probe_claude_catalog(
        resolved_urls,
        chosen=chosen,
        claude_follows_codex=claude_follows_codex,
    )
    if not probed:
        return None

    if catalog_ids is _CATALOG_NO_MODELS_ENDPOINT:
        # The gateway has no /models route; the model cannot be verified here, so proceed rather than refuse.
        print(
            f"Preflight: WARNING — gateway has no /models route (HTTP 404/405); "
            f"cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None

    if catalog_ids is None:
        # Auth/network/server/non-JSON/empty-catalog failure: genuinely unverifiable.
        if allow_custom:
            print(
                f"Preflight: WARNING — gateway catalog unreachable; cannot verify "
                f"--claude-model={chosen!r}. Proceeding with custom orchestration "
                f"model support enabled (trusting the operator id)."
            )
            return None
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids or _catalog_compare_model_id(chosen) in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    # For non-allowlisted custom ids the AMD fallback is meaningless; fail clearly on a catalog miss.
    if allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} not present in gateway catalog "
            f"(custom orchestration model support enabled; catalog has "
            f"{sorted(catalog_ids)[:20]}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Walk the allowlist in order so it acts as a real preference ladder: a gateway that carries opus-4-8 but not the
    # newer default must land on 4-8, not skip two generations down to the last entry.
    for candidate in _CLAUDE_ALLOWED_MODELS:
        if candidate == chosen:
            continue
        if candidate in catalog_ids:
            print(f"Preflight: WARNING — {chosen!r} not in gateway catalog; falling back to {candidate!r}")
            args.claude_model = candidate
            return catalog_ids

    print(
        f"ERROR: none of the allowed Claude models {list(_CLAUDE_ALLOWED_MODELS)!r} "
        f"present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _resolve_models_for_run(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool | None = None,
    codex_follows_claude: bool | None = None,
) -> None:
    """Resolve both model ids against the gateway before any session work."""
    if claude_follows_codex is None:
        claude_follows_codex = _claude_model_should_follow_codex()
    if codex_follows_claude is None:
        codex_follows_claude = _codex_model_should_follow_claude()

    if claude_follows_codex:
        # codex_model is about to become the orchestration model, so it needs the ladder whatever the critic backend
        # is.
        _smoke_test_codex_model(args, resolved_urls)
        args.claude_model = args.codex_model

    # Hard-gate the Claude model (mutates args.claude_model on fallback; sys.exit(2) on failure).
    _validate_and_resolve_claude_model(args, resolved_urls, claude_follows_codex=claude_follows_codex)

    if codex_follows_claude:
        args.codex_model = args.claude_model

    _probe_critic_review_model(
        args, codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
    )


_CRITIC_PROBE_TIMEOUT_SEC = 60.0


def _probe_critic_review_model(
    args: argparse.Namespace, *, codex_follows_claude: bool, claude_follows_codex: bool = False
) -> None:
    """Send the critic's model one real request before the session starts; exit rc=2 when it cannot answer.

    A catalog listing only proves a gateway names a model, not that its upstream serves it, and the critic has no
    fallback model: a review path that fails every call must stop the launch rather than run a session without a
    critic.
    """
    if _resolve_critic_choice(args) != "agent":
        return
    try:
        protocol, model = critic_review_target(
            args.critic_protocol,
            orchestration_on_codex=critic_on_codex_side(
                codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
            ),
            claude_model=args.claude_model,
            codex_model=args.codex_model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    messages = [{"role": "user", "content": "Reply with OK."}]
    last_error: BaseException | None = None
    for delay in (0.0, *_CATALOG_RETRY_DELAYS_SEC):
        if delay:
            time.sleep(delay)
        try:
            if protocol == "anthropic":
                llm_config.anthropic_completion(
                    request=llm_config.AnthropicCompletionRequest(
                        model=model, messages=messages, max_tokens=16, timeout_s=_CRITIC_PROBE_TIMEOUT_SEC
                    ),
                    component="critic",
                    operation="preflight",
                )
            else:
                llm_config.chat_completion(
                    llm_config.get_openai_client(timeout=_CRITIC_PROBE_TIMEOUT_SEC),
                    component="critic",
                    operation="preflight",
                    model=model,
                    messages=messages,
                    max_completion_tokens=16,
                )
        except Exception as exc:  # any failure means the review path is unusable; exits 2 after the last retry
            log.warning("critic model %r probe failed over the %s protocol: %s", model, protocol, exc, exc_info=True)
            last_error = exc
            continue
        print(f"Preflight: critic model {model!r} answered over the {protocol} protocol")
        return
    print(
        f"ERROR: critic model {model!r} did not answer over the {protocol} protocol: {last_error!r}\n"
        f"  The critic reviews with the orchestration model unless --critic-protocol selects the other side "
        f"(then CLAUDE_MODEL for anthropic, CODEX_MODEL for openai); there is no fallback model.\n"
        f"  Fix that model or its credential, or pass --critic-mock to run without a critic. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
) -> None:
    """WARN-only catalog check for ``--codex-model`` on an OpenAI-only launch, where it becomes the orchestration model.

    Steps down the ladder before Coordinator starts.
    """
    openai_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not openai_url:
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not openai_url:
        openai_url = resolved_urls[1]
    # OpenAI-side key only.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    catalog_ids = _probe_llm_catalog(base_url=openai_url, api_key=openai_key)
    if catalog_ids is None:
        # WARN-only path: don't block startup just because the OpenAI catalog is unreachable (the Claude gate already
        # validated reachability).
        print(
            "Preflight: WARNING — OpenAI-side catalog unreachable; skipping "
            "--codex-model verification (CodexBackend may fail at first turn)."
        )
        return

    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return

    if chosen in _CODEX_FALLBACK_MODELS:
        for candidate in _CODEX_FALLBACK_MODELS:
            if candidate == chosen:
                continue
            if candidate in catalog_ids:
                print(
                    f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog; falling back to {candidate!r}"
                )
                args.codex_model = candidate
                return

    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog (known-good ids, newest first: "
        f"{list(_CODEX_FALLBACK_MODELS)})."
    )


# Default critic backend; override via env or --critic-mock/--critic-agent.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
    "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent")


def _resolve_choice(
    attr: str,
    default: str,
    valid: tuple[str, ...],
    flag_hint: str,
    *,
    args: argparse.Namespace,
) -> tuple[str, bool]:
    """Resolve a backend choice from CLI args with validation and fallback to default."""
    chosen = getattr(args, attr, None)
    explicit = chosen is not None
    if chosen is None:
        chosen = default
    if chosen not in valid:
        print(
            f"ERROR: {attr.replace('_', ' ')} {chosen!r} not in {valid!r} (set by {flag_hint})",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen, explicit


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice (arg → DEFAULT_CRITIC_BACKEND); hard-fails on invalid."""
    chosen, _ = _resolve_choice(
        "critic_backend",
        DEFAULT_CRITIC_BACKEND,
        _VALID_CRITIC_BACKENDS,
        "--critic-mock / --critic-agent or INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
        args=args,
    )
    return chosen


def _reset_state_file(session_dir: Path, *, framework: str) -> None:
    """Back up ``state.json`` to ``state.json.preReset.<unix_ts>`` and start fresh (Recipe KB untouched).

    The fresh state keeps ``framework``: the session is single-framework, and
    nothing after this point re-derives it.
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
            "--reset-state: could not move %s → %s: %s",
            state_path,
            backup_path,
            exc,
        )
        return
    SharedState(framework=framework).save(session_dir)
    import logging as _logging

    _logging.getLogger(__name__).info(
        "--reset-state: backed up state.json to %s; session starts blank except for framework=%s.",
        backup_path.name,
        framework,
    )


def _preflight_agentx_backend(args: argparse.Namespace) -> None:
    """Reject the combinations where AgentX labels work it did not do."""
    if not _agentx_enabled():
        return
    from hyperloom.inference_optimizer import framework_registry
    from hyperloom.orchestrator.actions.executors.benchmark_backend import (
        resolve_backend_name,
    )

    backend = resolve_backend_name()
    if backend == "bypass":
        print(
            "ERROR: HYPERLOOM_AGENTX=1 with HYPERLOOM_BENCHMARK_BACKEND=bypass. "
            "The bypass backend ignores benchmark_script for serving frameworks, "
            "so AgentX would not run and the session would report synthetic "
            "measurements as AgentX results. Pick one.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    framework = str(getattr(args, "framework", "") or "").strip().lower()
    if framework and framework_registry.is_scriptable(framework):
        print(
            f"ERROR: HYPERLOOM_AGENTX=1 with --framework {framework!r}, which is "
            "scriptable. The AgentX switch only replaces the benchmark client of a "
            "serving framework, so no trace would be replayed -- but the session "
            "would still be labelled AgentX and run on AgentX budgets. Pick one.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _apply_agentx_budget_profile(args: argparse.Namespace) -> None:
    """Warn when AgentX is enabled without an explicit session budget."""
    if not _agentx_enabled():
        return
    # ``--max-hours`` carries no argparse default, so absence reads as ``None``
    # here: this runs before either path settles the budget. An explicit value
    # is a deliberate choice even when it equals the default, and gets no note.
    if getattr(args, "max_hours", None) is None:
        print(
            f"NOTE: AgentX replay is configured and --max-hours is at its default of {DEFAULT_MAX_HOURS}. "
            "One AgentX round (corpus load + warmup + drain + measurement window) "
            "typically exceeds that on its own; pass an explicit --max-hours sized "
            "to the number of candidates you intend to measure.",
            file=sys.stderr,
        )


def _resolve_run_max_model_len(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve planning context; native AgentX serving context remains in its YAML."""
    if getattr(args, "max_model_len", None):
        return int(args.max_model_len), "--max-model-len"
    max_model_len_env = os.environ.get("MAX_MODEL_LEN", "").strip()
    if max_model_len_env:
        try:
            return _positive_int_arg(max_model_len_env), "$MAX_MODEL_LEN"
        except argparse.ArgumentTypeError as exc:
            print(
                f"ERROR: MAX_MODEL_LEN={max_model_len_env!r} is invalid: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    # AgentX replays real agentic traces whose lengths come from the corpus, so ISL/OSL are meaningless placeholders
    # here (they default to 1024/1024) and ``ISL+OSL+headroom`` would pin the server at ~6k against traces reaching
    # ~1M tokens -- the corpus would then be silently reduced to whatever fits.
    if _agentx_enabled():
        native = _load_model_max_position_embeddings(str(args.model or ""))
        if native:
            return int(native), "agentx-native-context"
        # Model not on disk yet (uncached HF id).
        print(
            "NOTE: Model context is unavailable locally; using a planning estimate. "
            "Magpie resolves the AgentX serving context from its workload configuration.",
            file=sys.stderr,
        )
    return (
        _resolve_max_model_len(
            args.isl,
            args.osl,
            str(args.model or ""),
        ),
        "auto",
    )


def _build_phase_budget_pct(args: argparse.Namespace) -> dict[str, float]:
    """Map ``--*-pct`` CLI flags to a ``phase -> pct`` override dict.

    ENABLEMENT has no flag: nothing enforces a per-phase cap for it.
    """
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_CLOSE,
        PHASE_FRAMEWORK_AGENT,
        PHASE_KERNEL_AGENT,
        PHASE_PRELUDE,
        PHASE_SWEEP,
    )

    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", PHASE_PRELUDE),
        ("phase_budget_framework_pct", PHASE_FRAMEWORK_AGENT),
        ("phase_budget_kernel_pct", PHASE_KERNEL_AGENT),
        ("phase_budget_sweep_pct", PHASE_SWEEP),
        ("phase_budget_close_pct", PHASE_CLOSE),
    ):
        val = getattr(args, cli_field)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)
    return phase_budget_pct


def _detect_checkpoint_precision(model_path: str | None) -> str:
    """Detect weight precision from model config.json; returns '' when it names none."""
    if not model_path:
        return ""
    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(str(model_path))
    quant = (summary.get("quantization") or "").strip().lower()
    if quant:
        if quant.startswith("fp8"):
            return "fp8"
        if quant in ("fp4", "mxfp4", "nvfp4"):
            return "fp4"
        if quant in ("int8", "w8a8_int8"):
            return "int8"
        if quant in ("int4", "awq", "gptq"):
            return "int4"
        return quant
    dtype = (summary.get("torch_dtype") or "").strip().lower()
    _DTYPE_MAP = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
    }
    return _DTYPE_MAP.get(dtype, dtype) if dtype else ""


def _positive_env_int(name: str) -> int:
    """``$name`` as a positive int, or 0 so the caller falls through to the next rung of its ladder."""
    try:
        value = int(os.environ.get(name.upper(), "").strip())
    except ValueError:
        return 0
    return value if value > 0 else 0


def _seed_env_from_resumed_state(state: Any, pins: dict[str, str]) -> None:
    """Put the session's recorded workload in the environment, so a resume outranks whatever the shell exported.

    Hyperloom's own repro scripts ``export TP``/``CONC``/``ISL``/``OSL``, and sourcing one before a resume must not
    silently re-measure the session at another shape. ``pins`` -- this resume's ``--extra-env``, or, when it
    passes none, the pins persisted from the launch -- is skipped, so changing the workload stays an explicit act.
    """
    for name in ("ISL", "OSL", "CONC", "TP", "EP", "MAX_MODEL_LEN"):
        recorded = getattr(state, name.lower(), 0)
        if recorded and name not in pins:
            os.environ[name] = str(recorded)


def _resolve_workload_knobs(
    args: argparse.Namespace,
    state: Any | None = None,
) -> None:
    """Fill the unset numeric workload knobs on ``args``: flag, environment, resumed state, default.

    The environment rung carries the ``--extra-env`` pins :func:`_export_operator_launch_shape` has already written,
    and on a resume also the recorded workload :func:`_seed_env_from_resumed_state` writes over any shell export -- a
    resumed value therefore arrives through that rung rather than through ``state``. Precision has its own function
    because it reads the checkpoint, so it cannot run this early.
    """
    int_knobs = (
        ("isl", DEFAULT_ISL),
        ("osl", DEFAULT_OSL),
        ("conc", DEFAULT_CONC),
        ("tp", DEFAULT_TP),
        ("ep", DEFAULT_EP),
    )
    for name, default in int_knobs:
        val = getattr(args, name, None)
        if val is None:
            from_env = _positive_env_int(name)
            persisted = int(getattr(state, name, 0) or 0) if state is not None else 0
            val = from_env or (persisted if persisted > 0 else default)
        setattr(args, name, int(val))


def _resolve_precision(args: argparse.Namespace, state: Any | None = None) -> None:
    """Fill ``args.precision``: flag, environment, resumed state, checkpoint detection, default.

    Must run after the model is resolved: the detection rung reads the checkpoint's ``config.json``, and finding
    nothing there settles ``args.precision`` on the default with no later call able to correct it. A ``PRECISION``
    reaching this function through the environment is treated as an explicit flag, mismatch warning included.
    """
    precision = getattr(args, "precision", None)
    if not precision:
        precision = os.environ.get("PRECISION", "").strip()
    if not precision:
        persisted = (getattr(state, "precision", "") or "").strip() if state is not None else ""
        if persisted:
            precision = persisted
        else:
            detected = _detect_checkpoint_precision(getattr(args, "model", None))
            precision = detected or DEFAULT_PRECISION
    else:
        detected = _detect_checkpoint_precision(getattr(args, "model", None))
        if detected and detected != precision:
            print(
                f"WARN: --precision={precision!r} but checkpoint dtype is {detected!r}; "
                "using --precision flag as specified.",
                file=sys.stderr,
            )
    args.precision = precision


def _export_workload_envs_for_optimize(
    args: argparse.Namespace,
    *,
    tp_resolved: int,
    ep_resolved: int,
) -> None:
    """Project resolved workload knobs (TP/CONC/EP) into env for downstream Magpie YAMLs."""
    os.environ["TP"] = str(max(1, int(tp_resolved or 1)))
    os.environ["CONC"] = str(max(1, int(getattr(args, "conc", DEFAULT_CONC) or DEFAULT_CONC)))
    os.environ["EP"] = str(max(1, int(ep_resolved or 1)))


def _enforce_topology_gates(*, nodes: int, gpus_per_node: int, tp: int, ep: int) -> None:
    """Fail fast on a TP/EP shape the cluster cannot place, rather than on a cryptic launcher crash mid-cold-start.

    Multi-node only: a single node places whatever its own GPU count allows and the launcher reports that itself.
    """
    if nodes < 2:
        return
    total_gpus = nodes * gpus_per_node
    # Gate 1: total cluster GPUs (nodes*gpus_per_node) must hold the model's TP shards.
    if total_gpus < tp:
        print(
            f"ERROR: TP={tp} exceeds total GPU count "
            f"({nodes} nodes * {gpus_per_node} "
            f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
            "--nodes, or use a larger --gpus-per-node pod "
            "template.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Gate 2: EP cannot exceed TP (can't place more expert shards than ranks); fail before bootstrap.
    if ep > tp:
        print(
            f"ERROR: EP={ep} > TP={tp}. Expert-parallel "
            "size must be <= tensor-parallel size. Either lower --ep or "
            "raise --tp.",
            file=sys.stderr,
        )
        sys.exit(2)


def _export_operator_launch_shape(
    *,
    server_args: str,
    extra_env: dict[str, str],
) -> None:
    """Project the operator's ``--server-args`` / ``--extra-env`` into env.

    Every pin is exported under its own name, with no exceptions: Hyperloom's control variables are read with a
    bare ``os.environ.get``, so a pin the JSON blob alone carries is invisible to all of them. The blob is kept
    because it reaches ``state.json`` and names the previous launch's pins, which a resume dropping one must unset.
    """
    from hyperloom.common.env_safety import OPERATOR_EXTRA_ENV_VAR, operator_extra_env

    if server_args:
        os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] = server_args
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_SERVER_ARGS", None)
    # Read before the blob is rewritten: it names what the previous launch exported, which a resume dropping a pin
    # -- or a second session in the same shell -- has to unset rather than leave behind.
    for name in operator_extra_env():
        if name not in extra_env:
            os.environ.pop(name, None)
    os.environ.update(extra_env)
    if extra_env:
        os.environ[OPERATOR_EXTRA_ENV_VAR] = json.dumps(extra_env)
    else:
        os.environ.pop(OPERATOR_EXTRA_ENV_VAR, None)


def _export_predictor_settings(args: argparse.Namespace) -> None:
    """Project ``--primatune-endpoint`` / ``--primatune-mode`` into the env the predictor reads.

    An absent flag leaves the variable alone, so a value exported in the shell
    still applies. Nothing is persisted: a ``--resume-from`` relaunch needs the
    flags again unless the shell carries the variables.
    """
    from hyperloom.common import predictor_config

    endpoint = str(getattr(args, "primatune_endpoint", "") or "").strip()
    if endpoint:
        os.environ[predictor_config.ENV_ENDPOINT] = endpoint
    mode = str(getattr(args, "primatune_mode", "") or "").strip()
    if mode:
        os.environ[predictor_config.ENV_MODE] = mode
    conf = predictor_config.load()
    if conf.enabled:
        print(f"  predictor          : {conf.mode} at {conf.endpoint}")


def _partition_fanout_supported(framework: str | None) -> tuple[bool, str]:
    """Whether this framework's runner can place work per partition."""
    name = str(framework or "").strip().lower()
    if not name:
        return False, (
            "the framework is not resolved yet, so whether its runner places work "
            "per partition could not be checked here"
        )
    if framework_registry.is_scriptable(name):
        return True, ""
    return False, (
        f"{name!r} runs a server, and its benchmark does not place work per partition, "
        f"so the streams-per-partition setting would be ignored"
    )


@dataclass(frozen=True)
class PartitionRequest:
    """The compute-partition flags an operator passed: an asserted mode and the streams each partition runs."""

    declared_mode: str | None = None
    streams_per_partition: int | None = None


def _export_partition_shape(
    request: PartitionRequest,
    *,
    framework: str | None = None,
    gpu_type: str | None = None,
    nodes: int = 1,
    model: PartitionModel = PartitionModel(),
) -> dict[str, Any]:
    """Validate the session's compute-partition shape and publish it."""
    declared_mode, streams_per_partition = request.declared_mode, request.streams_per_partition
    from hyperloom.common.gpu_partition import PartitionError, parse_mode
    from hyperloom.orchestrator.actions.executors._partition_shape import (
        DEFAULT_STREAMS_PER_PARTITION,
        PARTITION_COUNT_ENV,
        PARTITION_CU_ENV,
        PARTITION_MODE_ENV,
        PARTITION_STREAMS_ENV,
        PARTITION_TOTAL_STREAMS_ENV,
        runtime_env,
        session_shape_summary,
        validate_session_shape,
    )

    # Cleared first so a second session in the same shell cannot inherit a shape the operator did not ask for this
    # time.
    for key in (
        PARTITION_MODE_ENV,
        PARTITION_COUNT_ENV,
        PARTITION_CU_ENV,
        PARTITION_STREAMS_ENV,
        PARTITION_TOTAL_STREAMS_ENV,
    ):
        os.environ.pop(key, None)

    try:
        mode = parse_mode(declared_mode)
    except PartitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    streams_named = streams_per_partition is not None
    # Tested against None rather than falsiness: `0 or DEFAULT` is DEFAULT, which would quietly honour an invalid
    # request as the default instead of refusing it, and leave the guard below unreachable for the one value most
    # likely to be passed by mistake.
    streams = DEFAULT_STREAMS_PER_PARTITION if streams_per_partition is None else int(streams_per_partition)
    if streams < 1:
        print(
            f"ERROR: --streams-per-partition must be >= 1, got {streams_per_partition}",
            file=sys.stderr,
        )
        sys.exit(2)

    if nodes >= 2:
        # Unconditional: the misleading record is produced by observing at all, not by asking for a mode.
        print(
            f"WARN: this session has --nodes {nodes}, and a compute-partition shape "
            f"describes one card. The benchmark node's topology cannot be read from here, "
            f"so no shape is recorded for this session.",
            file=sys.stderr,
        )
        if mode:
            print(
                f"ERROR: --compute-partition-mode {mode} cannot be checked on a "
                f"--nodes {nodes} session: the assertion is about the benchmark node's "
                f"card, which this process cannot read. An unverifiable assertion is not "
                f"a satisfied one. Drop the flag to run multi-node.",
                file=sys.stderr,
            )
            sys.exit(2)
        return {}

    fanout, fanout_detail = _partition_fanout_supported(framework)
    if (mode or streams_named) and not fanout:
        print(f"WARN: {fanout_detail}.", file=sys.stderr)

    verdict = validate_session_shape(
        declared_mode=mode,
        streams=streams,
        gpu_type=gpu_type,
        model=model,
        # The footprint refusal is arithmetic about streams sharing a partition.
        fanout_expected=fanout or bool(mode) or streams_named,
    )
    for warning in verdict.warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    if not verdict.ok:
        print(f"ERROR: {verdict.refusal}", file=sys.stderr)
        sys.exit(2)
    for note in verdict.notes:
        print(note)

    if verdict.layout is None:
        return {}
    # The observed shape is published either way -- the platform fingerprint reads it back from here, and provenance
    # is the point.
    os.environ.update(runtime_env(verdict.layout, streams, fanout=fanout))
    return session_shape_summary(verdict.layout, streams, fanout_expected=fanout)


def _publish_gpu_power_settings(record: Mapping[str, Any]) -> None:
    """Expose the recorded power settings to the platform fingerprint, the way the partition shape is."""
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV

    if record:
        os.environ[GPU_POWER_SETTINGS_ENV] = json.dumps(record, sort_keys=True)
    else:
        os.environ.pop(GPU_POWER_SETTINGS_ENV, None)


def _restore_partition_shape_from_state(args: Any, state: SharedState) -> None:
    """Fill the partition flags from the archive when this resume omitted them."""
    archived = dict(getattr(state, "compute_partition", None) or {})
    if not str(getattr(args, "compute_partition_mode", None) or "").strip():
        args.compute_partition_mode = str(archived.get("mode") or "")
    # `is None` rather than falsiness, so an explicit `--streams-per-partition 0` still reaches the guard that refuses
    # it instead of being read as "omitted" and quietly replaced by the archived value.
    if getattr(args, "streams_per_partition", None) is None:
        streams = archived.get("streams_per_partition")
        args.streams_per_partition = int(streams) if streams else None


#: Manifest objective kinds and the flag each one was parsed from.
_OBJECTIVE_KIND_TO_FLAG: Mapping[str, str] = {
    "gain_pct": "target_gain",
    "tput": "target_tput",
    "baseline": "target_baseline_dir",
    "roofline_pct": "target_roofline",
}


def _restore_budget_and_objective(args: Any, state: SharedState, manifest: Mapping[str, Any]) -> list[str]:
    """Fill the budget and the stop target from the archive when this resume omitted them.

    A bare resume would otherwise rebuild both from the flags: the budget would
    shorten a longer session and close the leg as ``time_exhausted``, and the
    objective would be dropped. An operator may pass only ``--resume-from``.
    An omitted ``--max-hours`` arrives as ``None``, so a smaller
    explicit budget still tightens the leg, which is what ``_start_run`` reads.

    Args:
        args: Parsed arguments for this resume; an explicit flag always wins.
        state: Loaded session state, whose ``max_minutes`` already carries every
            ``--extend-hours`` grant from earlier legs.
        manifest: The session manifest, which is where the objective is recorded.

    Returns:
        Lines to print, each already prefixed with ``  → ``.
    """
    lines: list[str] = []
    persisted_minutes = float(getattr(state, "max_minutes", 0) or 0)
    if getattr(args, "max_hours", None) is None and persisted_minutes > 0:
        args.max_hours = persisted_minutes / 60.0
        lines.append(f"  → restored budget: --max-hours {args.max_hours:.2f} (persisted)")

    # Restored as a set: a target named on this resume replaces the persisted
    # objective outright rather than joining it, which build_objective refuses.
    if any(getattr(args, flag, None) is not None for flag in _OBJECTIVE_KIND_TO_FLAG.values()):
        return lines
    recorded = manifest.get("objective") or {}
    # ``_objective_summary`` writes one entry, and ``objectives`` too when a
    # roofline target joins a throughput one.
    for entry in recorded.get("objectives") or [recorded]:
        flag = _OBJECTIVE_KIND_TO_FLAG.get(str(entry.get("kind") or ""))
        value = entry.get("value")
        if flag is None or value is None:
            continue
        setattr(args, flag, str(value) if flag == "target_baseline_dir" else float(value))
        lines.append(f"  → restored objective: --{flag.replace('_', '-')} {value}")
    return lines


def _exit_code_for_stop_reason(stop_reason: str | None, baseline_tput: float) -> int:
    """Map a terminal ``stop_reason`` to a process exit code (0 success, 1 failure).

    Reads the same classifier the breakdown grades outcomes against. A second
    copy here would decide CI's verdict on a vocabulary that had drifted from
    the one the report was written from.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import outcome_status

    return 0 if outcome_status(str(stop_reason or ""), baseline_tput) == "completed" else 1


def _write_cli_terminal_artifacts(session_dir: Path, state: SharedState, stop_reason: str | None) -> None:
    """Write the session's terminal artifacts in their established order.

    A watchdog restart ends a leg, not the session, so it produces none of them
    -- including the close-out package, which speaks for a finished run.
    """
    if stop_reason == SUPERVISOR_RESTART_REASON:
        return
    if not state.close_sequence_done:
        from ..breakdown.recorder.close_out import record_close_safety_net

        record_close_safety_net(session_dir)
    try:
        from ..breakdown import write_minimal_final_json

        final_json = write_minimal_final_json(session_dir)
        print(f"Final summary     : {final_json}")
    except Exception:
        log.exception("crash-safe final.json write failed (non-fatal)")
    if state.close_sequence_done:
        print("Session breakdown : (already written by CLOSE phase sequencer; skipping cli.finally safety-net write)")
    else:
        try:
            from ..breakdown import write_breakdown_json

            breakdown_path = write_breakdown_json(session_dir)
            print(f"Session breakdown : {breakdown_path}")
        except Exception:
            log.exception("session_breakdown finalize failed (non-fatal)")
        try:
            from ..breakdown import write_minimal_final_report

            final_md = write_minimal_final_report(session_dir)
            print(f"Final report      : {final_md}")
        except Exception:
            log.exception("emergency final report write failed (non-fatal)")
    try:
        from hyperloom.inference_optimizer.trace.langfuse_emitter import flush_session, record_session_breakdown

        flush_session(session_dir)
        from ..breakdown import patch_breakdown_langfuse

        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
    except Exception:
        log.debug("langfuse flush_session failed", exc_info=True)
    # Safety net for paths that leave close_sequence_done False and never run
    # the sequencer; ordered after langfuse so the package carries its patch.
    try:
        from ..breakdown import package_session_artifacts

        pkg_path = package_session_artifacts(session_dir, session_id=str(getattr(state, "session_id", "") or ""))
        if pkg_path is not None:
            print(f"Artifact package  : {pkg_path}")
    except Exception:
        log.exception("session artifact package failed (non-fatal)")


def _new_preflight_failure_session_dir(
    args: argparse.Namespace,
    *,
    failed_attempt: bool = False,
) -> Path:
    """Create a standalone session for a failed preflight attempt."""
    model_name = resolve_model_display_name(args)
    if not model_name:
        model_name = Path(os.environ.get("MODEL_PATH", "")).name
    if not model_name:
        resume_from = str(getattr(args, "resume_from", "") or "").strip()
        if resume_from:
            model_name = Path(resume_from).expanduser().parent.name
    if failed_attempt:
        model_name = f"{model_name or 'preflight'}-failed-attempt"
    return make_session_dir(model_name=model_name or "preflight-failure")


def _persist_preflight_failure_artifacts(
    args: argparse.Namespace,
    exc: Exception,
) -> Path | None:
    """Best-effort materialize the failed install event and final SBD."""
    _mark_pending_install_event_failed(args, exc)
    session_dir = _new_preflight_failure_session_dir(
        args,
        failed_attempt=bool(str(getattr(args, "resume_from", "") or "").strip()),
    )

    session_lock = SessionLock(session_dir)
    try:
        session_lock.acquire()
    except (SessionAlreadyRunning, SessionLockPathError, OSError):
        session_lock.release()
        log.warning("failed to lock SBD V6 preflight failure session", exc_info=True)
        return None

    try:
        if not (session_dir / "manifest.json").is_file():
            try:
                manifest_args = argparse.Namespace(**vars(args))
                if not getattr(manifest_args, "model", None):
                    manifest_args.model = os.environ.get("MODEL_PATH", "")
                write_manifest(session_dir, args=manifest_args)
            except Exception as write_exc:  # isolation boundary: the preflight-failure manifest write
                log.warning("failed to write manifest for SBD V6 preflight failure", exc_info=True)
                from ..session.sbd_v6 import record_write_warning

                record_write_warning(session_dir, component="preflight_failure.manifest", exc=write_exc)

        _persist_install_event(args, session_dir)
        from ..breakdown.recorder.close_out import record_close_safety_net
        from ..breakdown.recorder import record_stage_reached

        record_close_safety_net(session_dir)
        record_stage_reached(session_dir, "install")
        from ..breakdown import write_breakdown_json

        write_breakdown_json(session_dir)
    finally:
        session_lock.release()
    print(f"Preflight failure artifacts: {session_dir}", file=sys.stderr)
    return session_dir


def _start_experience_kb() -> None:
    """Bring the workspace's Experience KB service to serving, then check its Experiences can be collected."""
    from hyperloom_kb import ConfigurationError, LocalServiceError, RemoteClientError
    from hyperloom_kb.collect import MappingError

    from ..experience_collect import validate_config as validate_experience_collection
    from ..experience_kb_service import ensure_service

    # The run must not depend on the Experience KB: reads come back empty and writes are spooled or skipped.
    try:
        ensure_service()
    except (LocalServiceError, RemoteClientError) as exc:
        log.warning("Experience KB service is not serving (%s); Experience writes are spooled until it is", exc)
    try:
        validate_experience_collection()
    except (ConfigurationError, MappingError, RemoteClientError) as exc:
        log.warning("Experience KB cannot take this run's Experiences (%s); the run continues without them", exc)


@dataclass(frozen=True)
class _SessionStart:
    """The session this run operates on, however it was opened."""

    session_dir: Path
    session_lock: SessionLock
    manifest: dict[str, Any]


def _restore_platform_state(args: argparse.Namespace, state: SharedState, session_lock: SessionLock) -> None:
    """Re-assert the partition shape, the power settings and the topology this session was measured under.

    Each resumes on the same restore / apply / persist path: an omitted flag
    re-asserts the archived value rather than silently taking a new default.
    """
    # The partition shape is part of the measurement contract, so it resumes on the same restore / apply / persist
    # path as the paths above.
    _restore_partition_shape_from_state(args, state)
    from hyperloom.orchestrator.loop.coordinator import reap_orphaned_servers_best_effort

    reap_orphaned_servers_best_effort(session_lock.session_dir, phase="resume")  # before an apply reads the cards
    # The power settings resume on the same assert / record path: an omitted flag re-asserts the archived value.
    gpu_power, _gpu_power_error = resume_gpu_power_settings(args, state)
    if _gpu_power_error:
        session_lock.release()
        print(f"ERROR: cannot resume this session -- {_gpu_power_error}.", file=sys.stderr)
        sys.exit(2)
    _publish_gpu_power_settings(gpu_power)
    state.gpu_power_settings = gpu_power
    state.compute_partition = _export_partition_shape(
        PartitionRequest(getattr(args, "compute_partition_mode", None), getattr(args, "streams_per_partition", None)),
        framework=state.framework or getattr(args, "framework", None),
        gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
        # A resume must re-pass --nodes, so the persisted count is the one that says whether this session was ever
        # multi-node.
        nodes=max(int(getattr(args, "nodes", 1) or 1), int(getattr(state, "nodes", 1) or 1)),
        # Passed for the persisted model identity.
        model=PartitionModel(
            state.model_path or str(getattr(args, "model", "") or ""),
            state.precision or getattr(args, "precision", None),
            shared_state=state,
        ),
    )
    if state.compute_partition.get("mode"):
        print(f"  re-exported partition shape: {state.compute_partition['mode']}")
    if state.framework_repo_path:
        print(f"  re-exported FRAMEWORK_REPO_PATH: {state.framework_repo_path}")
    if state.bypass_scripts_dir:
        print(f"  re-exported HYPERLOOM_BYPASS_SCRIPTS_DIR: {state.bypass_scripts_dir}")
    if state.benchmark_backend:
        print(f"  re-exported HYPERLOOM_BENCHMARK_BACKEND: {state.benchmark_backend}")
    # Feeds the IR-8 check.
    if state.nodes > 1 and int(getattr(args, "nodes", 1) or 1) <= 1:
        args.nodes = state.nodes
        os.environ["INFERENCE_OPTIMIZER_NODES"] = str(state.nodes)
        print(f"  re-exported nodes         : {state.nodes}")


def _restore_phase_toggles(args: argparse.Namespace, state, session_dir: Path) -> None:
    """Honour the phase and gate decisions the original run persisted.

    A CLI flag can still turn a phase off on this resume; it cannot turn one
    back on that the session recorded as disabled.
    """
    # Warm-replay gates: a non-default value is the only "operator set this" signal, since the parser gives these
    # flags real defaults rather than None.
    if args.no_warm_replay:
        state.warm_replay_enabled = False
    _wr_value = getattr(args, "warm_replay_min_confidence", None)
    if _wr_value is not None:
        state.warm_replay_min_confidence = _wr_value
    # Honour persisted kernel_enabled on resume; CLI --no-kernel can still override.
    if not state.kernel_enabled:
        args.no_kernel = True
        print("  Kernel-agent          : DISABLED (persisted from original run)")
    # Same persistence contract for the FRAMEWORK_AGENT phase toggle.
    if not bool(getattr(state, "framework_agent_phase_enabled", True)):
        args.no_framework_agent = True
        print("  framework phase       : DISABLED (persisted from original run)")
    elif bool(getattr(args, "no_framework_agent", False)):
        # Inverse: honour --no-framework-agent on resume only before FRAMEWORK is entered.
        cur_phase = (getattr(state, "phase", "") or "").strip().upper()
        if cur_phase in ("", "PRELUDE"):
            state.framework_agent_phase_enabled = False
            # Persist immediately; the later conditional save only runs on prior stop_reason/crash.
            state.save(session_dir)
            print("  framework phase       : DISABLING for resume (--no-framework-agent + phase=PRELUDE)")
        else:
            print(
                f"  framework phase       : WARN --no-framework-agent ignored; "
                f"session is already in phase={cur_phase!r} "
                f"(cannot retroactively skip)"
            )


@dataclass(frozen=True)
class _OperatorLaunchShape:
    """The operator's server args and extra env a resume re-applies: this resume's flags, else the session's record."""

    server_args: str
    extra_env: dict


def _reexport_session_environment(
    args: argparse.Namespace,
    state,
    *,
    session_dir: Path,
    session_id: str,
    launch_shape: _OperatorLaunchShape,
) -> None:
    """Put the session's recorded environment back, so a fresh shell resumes what was measured.

    Without this a resume falls back to the YAML defaults for anything the
    original run pinned. Every knob resumes on the same ladder: an explicit
    flag on this resume wins, otherwise the archived value is re-exported, so a
    re-passed pin can still change it.
    """
    # Re-export session-level env from persisted state so a fresh-shell resume doesn't fall back to YAML defaults.
    if state.model_path:
        resume_model_path = str(state.model_path)
        os.environ["MODEL_PATH"] = resume_model_path
        print(f"  re-exported MODEL_PATH: {resume_model_path}")
        # Backfill model_info for sessions created before the field existed (or whose config was unreadable at
        # launch); fail-soft to {}.
        if not state.model_info:
            state.model_info = summarize_model_config(resume_model_path)
            if state.model_info:
                state.save(session_dir)
                print("  backfilled model_info (from config.json)")
    state.framework = _registered_framework_or_exit(
        state.framework or os.environ.get("FRAMEWORK", ""),
        hint="it comes from the resumed session, or from $FRAMEWORK when the session recorded none",
    )
    _enforce_expected_framework(state.framework)
    os.environ["FRAMEWORK"] = state.framework
    print(f"  re-exported FRAMEWORK : {state.framework}")
    if state.framework == "atom":
        _apply_atom_auto_tighten(args)
    if state.gpu_type:
        runner_gpu_type = gpu_runner_type(state.gpu_type)
        os.environ["TARGET_GPU_TYPE"] = state.gpu_type
        os.environ["GPU_TYPE"] = runner_gpu_type
        print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
        if runner_gpu_type != state.gpu_type:
            print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
    info = LaunchInfo(
        pid=os.getpid(),
        session_dir=session_dir,
        session_id=session_id,
        run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
        gpu_type=state.gpu_type or "",
        framework=state.framework,
        model=str(state.model_path or ""),
    )
    _emit_launch_info(info, launch_info_file=getattr(args, "launch_info_file", None))
    _seed_env_from_resumed_state(state, launch_shape.extra_env)
    # The seeding above puts the recorded workload on the environment rung, so it outranks a shell export while an
    # explicit --isl/--conc/... on this resume still wins. The resolved values are then projected back into env so
    # resume sees the same workload contract (not YAML defaults).
    _resolve_workload_knobs(args, state)
    _resolve_precision(args, state)
    # Same ladder as the knobs above; the environment rung is what lets a re-passed pin change it.
    _resume_max_model_len = (
        getattr(args, "max_model_len", None) or _positive_env_int("MAX_MODEL_LEN") or getattr(state, "max_model_len", 0)
    )
    for env_name, val in (
        ("TP", args.tp),
        ("EP", args.ep),
        ("CONC", args.conc),
        ("ISL", args.isl),
        ("OSL", args.osl),
        ("MAX_MODEL_LEN", _resume_max_model_len),
    ):
        if val:
            os.environ[env_name] = str(int(val))
            print(f"  re-exported {env_name:<14s}: {int(val)}")
    # Profile-scoped OSL: an explicit --profile-osl on this resume wins; otherwise re-export the value persisted
    # from the original run.
    _resume_profile_osl = getattr(args, "profile_osl", None) or getattr(state, "profile_osl", 0)
    if _resume_profile_osl:
        os.environ["PROFILE_OSL"] = str(int(_resume_profile_osl))
        state.profile_osl = int(_resume_profile_osl)
        print(f"  re-exported PROFILE_OSL   : {int(_resume_profile_osl)}")
    os.environ["PRECISION"] = args.precision
    print(f"  re-exported PRECISION     : {args.precision}")
    if getattr(state, "framework_version", ""):
        os.environ["FRAMEWORK_VERSION"] = state.framework_version
        print(f"  re-exported FRAMEWORK_VERSION: {state.framework_version}")
    # Operator launch shape: already exported above, ahead of the staleness guard that reads the pins.
    state.operator_server_args = launch_shape.server_args
    state.operator_extra_env = launch_shape.extra_env
    if launch_shape.server_args:
        print(f"  re-exported server_args   : {launch_shape.server_args}")
    if launch_shape.extra_env:
        print(f"  re-exported extra_env     : {','.join(sorted(launch_shape.extra_env))}")
    _apply_custom_specialist_resume(args, state)
    # Custom-workload paths: an explicit --framework-path / --benchmark-scripts-dir on this resume wins, else the
    # persisted value.
    _restore_operator_supplied_paths_from_state(args, state)
    _apply_operator_supplied_paths(args, state.framework)
    _require_custom_entrypoint(
        state.framework,
        gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
    )
    _persist_operator_supplied_paths(state)


def _open_resumed_session(args: argparse.Namespace) -> _SessionStart:
    """Lock the session ``--resume-from`` names, refusing a directory that is not a resumable session.

    ``--resume-from`` names a subdirectory under the workspace root; USER_DATA_PATH itself stays where it is.
    """
    from ..session.paths import ENV_CURRENT_SESSION_DIR, _SESSION_SKELETON, is_path_within, workspace_root

    ws = workspace_root()
    session_dir = Path(args.resume_from).expanduser().resolve()
    if getattr(args, "session_dir", None):
        print("ERROR: --session-dir names a new session; it cannot be combined with --resume-from.", file=sys.stderr)
        sys.exit(2)
    if not is_path_within(session_dir, ws):
        print(
            f"ERROR: --resume-from {session_dir!r} is not under $USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
            "workspace root (the parent of the per-session subdirs) and pass the per-session subdir via --resume-from.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not session_dir.is_dir():
        print(f"ERROR: --resume-from {session_dir!r} does not exist.", file=sys.stderr)
        sys.exit(2)
    # Pin before Coordinator/SharedState load so paths/subprocesses inherit the resolved location.
    os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
    for sub in _SESSION_SKELETON:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)

    # Single-optimizer guard: take the session lock before any state.json / lease access.
    session_lock = _acquire_session_lock_or_exit(session_dir)
    from ..session.resume_guard import ResumeBlocked, ensure_resume_safe
    from hyperloom.orchestrator.bus.resource_lock import local_owner_scope

    try:
        ensure_resume_safe(session_dir, owner_scope=local_owner_scope())
    except ResumeBlocked as exc:
        session_lock.release()
        print(f"ERROR: --resume-from {exc}", file=sys.stderr)
        sys.exit(2)
    _persist_install_event(args, session_dir)

    try:
        manifest = load_manifest(session_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: --resume-from failed: {exc}", file=sys.stderr)
        sys.exit(2)
    if not (session_dir / "state.json").exists():
        print(
            f"ERROR: --resume-from failed: {session_dir}/state.json missing "
            "(manifest exists but Coordinator never wrote SharedState)",
            file=sys.stderr,
        )
        sys.exit(2)
    return _SessionStart(session_dir=session_dir, session_lock=session_lock, manifest=manifest)


def _open_next_leg(args: argparse.Namespace, state: SharedState, session: _SessionStart) -> None:
    """Begin the resumed session's next leg: clear the last leg's stop, grant ``--extend-hours``, report the budget.

    ``target_reached`` is a terminal state requiring ``--force-resume`` to push past it; other reasons auto-clear.
    """
    prior_stop = state.stop_reason
    prior_crash = state.crash_count
    if prior_stop == "target_reached" and not getattr(args, "force_resume", False):
        print(
            f"\nERROR: --resume-from blocked by terminal stop_reason={prior_stop!r}.\n\n"
            f"  SKILL.md (Run-time signals): {prior_stop!r} is a deliberate terminal state.\n"
            "  The optimizer will not auto-resume past it because the prior run\n"
            "  declared exhaustion — picking up where it left off only repeats\n"
            "  the same exhaustion verdict.\n\n"
            "  Override paths:\n"
            "  1. Pass ``--force-resume`` if you have changed the workload /\n"
            "     search space / model / strategy and want to continue regardless.\n"
            "  2. Start a fresh session (different $USER_DATA_PATH) for a clean run.\n\n"
            f"  Reports for the prior run live under {session.session_dir}/reports/.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    _begin_resume_leg(state)
    extend_hours = float(args.extend_hours)
    if extend_hours > 0.0:
        state.extend_budget_minutes(extend_hours * 60.0)
    state.save(session.session_dir)
    _record_resumed_model_gate(
        args,
        session.session_dir,
        workload_overrides={
            key: str(getattr(state, key) or session.manifest.get(key) or "")
            for key in ("model_path", "model_name", "framework", "gpu_type")
        },
    )
    override_note = " (--force-resume override)" if prior_stop == "target_reached" else ""
    print(f"  → cleared stop_reason and crash_count (was {prior_crash}) for this leg{override_note}")
    for line in _restore_budget_and_objective(args, state, session.manifest):
        print(line)
    if args.max_hours is None:
        args.max_hours = DEFAULT_MAX_HOURS
    for line in _resume_budget_lines(state, extend_hours=extend_hours):
        print(line)


async def _resume_session(args: argparse.Namespace) -> _SessionStart:
    """Reopen a recorded session, restoring the environment it was measured under."""
    session = _open_resumed_session(args)
    session_dir = session.session_dir
    state = SharedState.load_or_init(session_dir)
    # Restored before the staleness guard below, which resolves the AgentX backend from the environment: a
    # session whose backend came from a pin would otherwise be refused as a workload mismatch. An explicit flag
    # on this resume wins, else the persisted value.
    _resume_server_args = str(getattr(args, "server_args", "") or "").strip() or state.operator_server_args
    _resume_extra_env = parse_operator_extra_env(args) or dict(state.operator_extra_env)
    _export_operator_launch_shape(server_args=_resume_server_args, extra_env=_resume_extra_env)
    _stale = agentx_state_is_stale(state)
    if _stale:
        print(
            f"ERROR: cannot resume this session -- {_stale}. "
            "Start a fresh session instead of mixing the two measurement sets.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Checked before any resume step persists state, so a refused resume leaves the session as it was.
    _resume_conflict = (
        latency_budget_scope_error(state.framework, getattr(args, "max_latency_ms", None))
        or constraint_budget_resume_conflict(args, state)
        or _custom_specialist_conflict(args, state)
    )
    if _resume_conflict:
        session.session_lock.release()
        print(f"ERROR: cannot resume this session -- {_resume_conflict}.", file=sys.stderr)
        sys.exit(2)
    print(f"Resuming session: {session_dir}")
    print(f"  manifest.session_id    : {session.manifest.get('session_id')}")
    print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
    print(f"  prior cumul_gain      : {state.cumulative_gain_validated:.2f}%")
    print(f"  prior current_best    : {state.current_best.get('action')}/{state.current_best.get('tput')}")
    print(f"  prior stop_reason     : {state.stop_reason or '(none)'}")

    _reexport_session_environment(
        args,
        state,
        session_dir=session_dir,
        session_id=str(session.manifest.get("session_id") or ""),
        launch_shape=_OperatorLaunchShape(server_args=_resume_server_args, extra_env=_resume_extra_env),
    )
    # After the re-export, not before: both read the GPU_TYPE and the framework it put back.
    _restore_platform_state(args, state, session.session_lock)
    _restore_phase_toggles(args, state, session_dir)
    _open_next_leg(args, state, session)
    return session


def _resolve_target_gpu(args: argparse.Namespace, framework: str) -> None:
    """Settle ``args.gpu_type`` and the Magpie runner, then refuse a board this session cannot serve."""
    # Resolve real target GPU: probe > --gpu-type hint; probe wins to catch wrong-host typos that corrupt KB.
    user_specified = _resolve_amd_gpu_type(args.gpu_type) or ""
    if _should_remote_probe_gpu(args):
        from ..multi_node._internal.gpu_probe import remote_autodetect_gpu_type

        probed = remote_autodetect_gpu_type() or ""
        if probed:
            print(f"GPU probe       : {probed} (remote {(args.mn_backend or 'rayjob').lower()})")
    else:
        probed = _autodetect_gpu_type() or ""
    gpu_type, gpu_warnings = _resolve_gpu_type(user_specified=user_specified, probed=probed)
    for line in gpu_warnings:
        print(line, file=sys.stderr)
    if probed and not user_specified:
        print(f"GPU type        : {gpu_type} (auto-detected)")
    runner_gpu_type = gpu_runner_type(gpu_type)
    warn_family_runner(gpu_type, framework)
    args.gpu_type = gpu_type or None
    if runner_gpu_type:
        os.environ["TARGET_GPU_TYPE"] = gpu_type
        os.environ["GPU_TYPE"] = runner_gpu_type
        print(f"GPU type        : {gpu_type}")
        print(f"Magpie runner   : {runner_gpu_type} (will inject runner_type into Magpie YAML)")
    else:
        os.environ.pop("TARGET_GPU_TYPE", None)
        os.environ.pop("GPU_TYPE", None)
        print("GPU type        : <unset> (Magpie will auto-detect)")
    _require_custom_entrypoint(framework, gpu_type=runner_gpu_type)

    # Runs here, not in _preflight, because the question it asks -- will provenance be able to name the ISA? -- is
    # unanswerable until args.gpu_type is final.
    _check_gfx_arch_resolvable(args.gpu_type)


def _create_fresh_session(args: argparse.Namespace, *, framework: str, nodes_resolved: int) -> _SessionStart:
    """Create and lock the session directory, then write its manifest, launch info and seeded state."""
    session_dir = make_session_dir(
        model_name=resolve_model_display_name(args), target=getattr(args, "session_dir", None)
    )
    # Single-optimizer guard: take the lock so the contract holds uniformly and the owner pid is published for the
    # session diagnostics.
    session_lock = _acquire_session_lock_or_exit(session_dir)
    from .benchmark_config import bind_benchmark

    bind_benchmark(session_dir, args)
    manifest = write_manifest(session_dir, args=args)
    _persist_install_event(args, session_dir)
    # One-shot Langfuse startup marker so a run killed before a breakdown still leaves a correlatable trace.
    try:
        from hyperloom.inference_optimizer.trace.langfuse_emitter import record_session_start

        record_session_start(session_dir)
    except Exception:  # outermost telemetry sink: the Langfuse session_start marker
        log.debug("langfuse record_session_start failed (non-fatal)", exc_info=True)
    print(f"Session dir     : {session_dir}")
    print(f"Session id      : {manifest['session_id']}  (manifest label only)")
    _print_session_skeleton(session_dir)

    # Machine-readable launch info: stable point for launcher scripts to harvest pid/session_dir/run_log.
    info = LaunchInfo(
        pid=os.getpid(),
        session_dir=session_dir,
        session_id=str(manifest["session_id"]),
        run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
        gpu_type=args.gpu_type or "",
        framework=args.framework or "",
        model=str(args.model),
    )
    _emit_launch_info(info, launch_info_file=getattr(args, "launch_info_file", None))
    # Placed here, not at the top of _run_optimize, because everything it weighs is resolved by now and none of it
    # was then: the framework (whether anything fans out), args.gpu_type (the CU fallback), and args.model, which
    # --quantize rewrites to the exported checkpoint -- sizing partitions against the source model would weigh the
    # wrong weights.
    compute_partition = _export_partition_shape(
        PartitionRequest(getattr(args, "compute_partition_mode", None), getattr(args, "streams_per_partition", None)),
        framework=framework,
        gpu_type=args.gpu_type,
        nodes=nodes_resolved,
        model=PartitionModel(str(args.model), getattr(args, "precision", None)),
    )
    gpu_power = resolve_launch_gpu_power(args, nodes=nodes_resolved, owner=str(manifest["session_id"]))
    _publish_gpu_power_settings(gpu_power)
    _seed_shared_state(
        session_dir,
        args,
        session_id=manifest["session_id"],
        compute_partition=compute_partition,
        gpu_power_settings=gpu_power,
    )
    return _SessionStart(session_dir=session_dir, session_lock=session_lock, manifest=manifest)


async def _seed_fresh_session(args: argparse.Namespace, *, nodes_resolved: int) -> _SessionStart:
    """Open a new session: resolve the target, seed its state, and gate the model before bring-up."""
    # Resolve model path: --model > $MODEL_PATH; fail fast rather than silently use the YAML hardcoded model.
    args.model = args.model or os.environ.get("MODEL_PATH") or ""
    if not args.model:
        print(
            "ERROR: model is required. Pass --model <path> or set MODEL_PATH env (or use --resume-from <session_dir> "
            "to continue an existing session).",
            file=sys.stderr,
        )
        sys.exit(2)
    # Re-export so subprocess executors inject the resolved model into the Magpie YAML, not its hardcoded model.
    from hyperloom.common.model_paths import resolve_serving_model_path

    os.environ["MODEL_PATH"] = resolve_serving_model_path(str(args.model)) or str(args.model)

    # Quantization prelude (one-shot, before any session/baseline work): if --quantize was passed, quantize the
    # source model now and rewrite args.model to the exported quantized model.
    await _run_quantization_prelude(args)

    # Resolve framework: --framework > $FRAMEWORK > "sglang" (session-wide; no framework mixing).
    framework = _registered_framework_or_exit(
        args.framework or os.environ.get("FRAMEWORK", ""),
        hint="set $FRAMEWORK accordingly or pass --framework",
    )
    _enforce_expected_framework(framework)
    os.environ["FRAMEWORK"] = framework
    print(f"Framework       : {framework}")
    _apply_operator_supplied_paths(args, framework)
    if framework == "atom":
        _apply_atom_auto_tighten(args)
    _resolve_target_gpu(args, framework)

    # ISL/OSL/CONC/TP/EP were settled and projected before _preflight; precision could not be, because it reads
    # the checkpoint and the model is only resolved above (issue #903).
    _resolve_precision(args)
    # MAX_MODEL_LEN is operator-overridable.
    max_model_len, max_model_len_source = _resolve_run_max_model_len(args)
    args.max_model_len = max_model_len
    os.environ["MAX_MODEL_LEN"] = str(max_model_len)
    os.environ["ISL"] = str(args.isl)
    os.environ["OSL"] = str(args.osl)
    # Profile-scoped OSL (issue #571): exported only when explicitly set, so the profile/roofline materializer can
    # decouple its OSL from the served workload.
    if getattr(args, "profile_osl", None) is not None:
        os.environ["PROFILE_OSL"] = str(args.profile_osl)
    os.environ["PRECISION"] = args.precision
    _fw_version_for_env = resolve_framework_version(args)
    if _fw_version_for_env:
        os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
    if _agentx_enabled():
        print(f"Workload        : AgentX replay, CONC={args.conc}; corpus and model context resolved by Magpie")
    else:
        print(
            f"Workload        : ISL={args.isl} OSL={args.osl} MAX_MODEL_LEN={max_model_len} ({max_model_len_source}) "
            f"PRECISION={args.precision} FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}"
        )

    session = _create_fresh_session(args, framework=framework, nodes_resolved=nodes_resolved)
    _start_model_gate(args, session.session_dir)
    # After the seed, before the heavy server bring-up: each check rejects a model that bring-up would fail on.
    if (
        _preflight_unsupported_model_arch(args, session.session_dir)
        or _preflight_model_config_compat(args, session.session_dir)
        or _preflight_context_window(args, session.session_dir)
    ):
        sys.exit(2)
    _finish_model_gate(args, session.session_dir)
    return session


def _resolve_run_topology(args: argparse.Namespace) -> int:
    """Settle this run's launch shape, and return the node count the caller gates on.

    Everything else it settles -- the operator's pins, TP/EP, ISL/OSL/CONC, the
    PD split, ``SKIP_VARIANTS`` -- leaves through ``os.environ``, which is how
    the subprocess executors that inherit it read these. Only the node count has
    a caller in this process, so only it is returned. A resume runs its own
    ladder against the session's recorded values and skips the projection.
    """
    # Surface --nodes (CLI flag wins) before _preflight runs.
    nodes_resolved = max(1, int(args.nodes))
    # The operator's pins have to be in the environment before the ladder below reads them.
    _export_operator_launch_shape(
        server_args=str(getattr(args, "server_args", "") or "").strip(),
        extra_env=parse_operator_extra_env(args),
    )
    # Settle the knobs before the gates, the projection and ``_preflight`` derive from them. A resume runs its own
    # ladder once the state is loaded.
    _is_resume = bool(getattr(args, "resume_from", None))
    if not _is_resume:
        _resolve_workload_knobs(args)
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    gpus_per_node_resolved = _resolve_gpus_per_node(args)

    # Project before ``_preflight``: ``check_gpu_visibility`` reads ``$TP`` to warn when the run asks for more
    # shards than rocm-smi can see.
    if not _is_resume:
        _export_workload_envs_for_optimize(args, tp_resolved=tp_resolved, ep_resolved=ep_resolved)

    _enforce_topology_gates(
        nodes=nodes_resolved,
        gpus_per_node=gpus_per_node_resolved,
        tp=tp_resolved,
        ep=ep_resolved,
    )

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Multi-node topology handoff: export the CLI-flag-resolved backend / gpus-per-node so downstream subprocesses
    # (kernel agent, benchmark, KB topology, state synthesis) read a single stable source.
    if nodes_resolved >= 2:
        os.environ["INFERENCE_OPTIMIZER_GPUS_PER_NODE"] = str(gpus_per_node_resolved)
        os.environ["INFERENCE_OPTIMIZER_MN_BACKEND"] = _resolve_mn_backend(args)
    _export_predictor_settings(args)
    # The partition shape is deliberately NOT exported here.

    # User-declared grid skip list; re-export so subprocess executors inherit it (empty clears stale values).
    # Not pinnable: it is the policy for one run, not part of the session's measurement contract, so a resume takes
    # it from the flag it was passed rather than from what the original launch pinned.
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs for executors; empty means "resolve from state.json", pd_mode always exported.
    pd_mode = (getattr(args, "pd_mode", "") or "aggregated").lower()
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation needs >=2 nodes (separate prefill + decode pods); fail at parse time.
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to aggregated) or "
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
            ("pd_prefill_ep", "PD_PREFILL_EP"),
            ("pd_decode_ep", "PD_DECODE_EP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
            ("pd_prefill_extra_args", "PD_PREFILL_EXTRA_ARGS"),
            ("pd_decode_extra_args", "PD_DECODE_EXTRA_ARGS"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v
    return nodes_resolved


@dataclass(frozen=True)
class _AgentLineup:
    """Which phases and which critic this run brings, once the toggles are read."""

    no_kernel: bool
    no_framework_agent: bool
    critic_choice: str
    critic_agent_root: Path | None
    critic_kb_mode: str


def _resolve_agent_lineup(args: argparse.Namespace) -> _AgentLineup:
    """Read the phase toggles and settle the critic backend before any backend is built.

    An unreachable ``--critic-agent`` runtime aborts here rather than at the
    first critic call, so the run does not pay for bring-up to find out.
    """
    no_kernel = getattr(args, "no_kernel", False)
    no_framework_agent = bool(getattr(args, "no_framework_agent", False))
    # Mirror the kernel banner so both phase toggles surface their state.
    if no_framework_agent:
        print(
            "Optimize phase  : DISABLED (--no-framework-agent); "
            f"{'baseline -> SWEEP' if no_kernel else 'baseline -> KERNEL -> SWEEP'}"
        )
    else:
        print("Optimize phase  : ENABLED")
    if no_framework_agent and no_kernel:
        print(
            "WARNING: --no-framework-agent and --no-kernel are both set; the "
            "run collapses to baseline -> SWEEP over an empty "
            "optimization_stack (no config or source search, no KERNEL "
            "rewrites). SWEEP only re-validates the baseline recipe. "
            "Continuing as requested.",
            file=sys.stderr,
        )
    if bool(getattr(args, "research_scout", True)):
        print(
            "Research scout  : ENABLED at PRELUDE (re-dispatch every "
            f"{max(1, int(getattr(args, 'research_scout_interval', 3) or 3))} "
            "explore rounds)"
        )
    else:
        print("Research scout  : DISABLED (--no-research-scout)")
    if bool(getattr(args, "target_advisory", True)):
        print("Target advisory : ENABLED (External target gap injected into prompts; advisory-only)")
    else:
        print("Target advisory : DISABLED (--no-target-advisory)")
    if bool(getattr(args, "recipe_sediment", True)):
        print("Recipe sediment : ENABLED (KEEP/REVERT provenance written to persistent recipe)")
    else:
        print("Recipe sediment : DISABLED (--no-recipe-sediment)")
    # Resolve critic backend + runtime root before _build_backends; abort rc=2 if --critic-agent runtime unreachable.
    critic_choice = _resolve_critic_choice(args)
    if critic_choice == "mock" and args.critic_protocol != "auto":
        # The mock critic issues no review inference, so there is no transport for the flag to select.
        print(
            f"WARNING: --critic-protocol={args.critic_protocol} is ignored with the mock critic, "
            "which performs no review inference.",
            file=sys.stderr,
        )
    critic_agent_root: Path | None = None
    critic_kb_mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if critic_kb_mode not in ("inmemory", "live"):
        print(
            f"ERROR: CRITIC_KB_CLIENT_MODE={critic_kb_mode!r} not in {{'inmemory','live'}}",
            file=sys.stderr,
        )
        sys.exit(2)
    if _critic_agent_runtime_needed(critic_choice):
        critic_agent_root = _resolve_agent_root()
        if critic_agent_root is None:
            print(
                f"ERROR: --critic-agent selected but critic-agent runtime not "
                f"found.\n"
                f"  Set ${_CRITIC_AGENT_ROOT_ENV} to the directory containing "
                f"runtime/cli.py, or check the "
                f"src/hyperloom/agents/critic/ install.\n"
                f"  Bypass with --critic-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_agent_runtime(critic_agent_root)
        if critic_kb_mode == "live" and not os.environ.get("KB_BASE_URL"):
            print(
                "ERROR: CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not "
                "set. Either export KB_BASE_URL or unset "
                "CRITIC_KB_CLIENT_MODE to fall back to inmemory.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Default WORKSPACE_PATH for critic-agent runtime: SKILL static-asset root (repo root), not artefact dir.
        os.environ.setdefault("WORKSPACE_PATH", str(Path(__file__).resolve().parents[4]))
    return _AgentLineup(
        no_kernel=no_kernel,
        no_framework_agent=no_framework_agent,
        critic_choice=critic_choice,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
    )


def _install_system_prompts(
    coordinator: Coordinator,
    args: argparse.Namespace,
    objective: Objective,
    lineup: _AgentLineup,
    *,
    macro_cycle: int,
) -> dict[str, str]:
    """Install each role's boot system prompt on ``coordinator``, an operator override winning, and return them.

    The Coordinator has already entered the session's phase; it re-scopes the prompt at every later phase seam.
    """
    state = coordinator.shared_state
    directive = str((dict(getattr(state, "orchestration_memory", {}) or {})).get("next_cycle_directive", "") or "")
    prompt_context = OrchestrationPromptContext(
        no_kernel=lineup.no_kernel,
        no_framework_agent=lineup.no_framework_agent,
        framework=os.environ["FRAMEWORK"],
        objective=objective,
        max_minutes=int(round(float(args.max_hours) * 60)),
        # The role record always names Claude; the deployment's provider shape is what actually decides, so read the
        # transport off the built backend.
        transport=getattr(coordinator.backends.get("orchestration"), "transport", TRANSPORT_TOOLS),
        benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""),
        agentx_corpus_shape=state.agentx_corpus_shape,
        agentx_grading=state.grading,
        agentx_backend=state.agentx_backend,
    )
    prompts = {
        "orchestration": args.orch_prompt
        or _build_orchestration_prompt(
            prompt_context, macro_cycle=macro_cycle, cycle_directive=directive, phase=state.phase
        ),
        "critic": args.critic_prompt or _load_critic_prompt(),
    }
    import functools as _functools

    coordinator.orch_prompt = _OrchestrationPrompt(
        overrides=prompts,
        is_user_supplied=bool(args.orch_prompt),
        rebuild=_functools.partial(_build_orchestration_prompt, prompt_context),
    )
    return prompts


def _print_backend_banner(args: argparse.Namespace, backends: dict[str, Any], lineup: _AgentLineup) -> None:
    """Print the backend each role runs on and the run's tick limits."""
    orchestration_str = _orchestration_banner(
        backends.get("orchestration"),
        claude_model=args.claude_model,
        codex_model=args.codex_model,
    )
    kernel_str = "DISABLED" if lineup.no_kernel else "programmatic"
    if lineup.critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        # Read off the backend rather than re-derived from the environment: --critic-protocol can override that
        # derivation, and the banner would then report the model of a transport this run is not using.
        _review_protocol = backends["critic"].protocol
        _review_model = args.claude_model if _review_protocol == "anthropic" else args.codex_model
        critic_str = (
            f"critic-agent(kb={lineup.critic_kb_mode}, protocol={_review_protocol}, "
            f"model={_review_model}, root={lineup.critic_agent_root})"
        )
    print(f"Backends        : orchestration={orchestration_str}, kernel={kernel_str}, critic={critic_str}")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} (budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()


def _settle_custom_specialist_only(args: argparse.Namespace) -> None:
    """Read only mode once the operator's pins are in the environment, and refuse a fresh launch's custom conflicts.

    A resume checks its conflicts against the stored state once that is loaded.
    """
    try:
        args.custom_specialist_only = env_bool(CUSTOM_SPECIALIST_ONLY_ENV)
    except EnvValueError as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        raise SystemExit(2) from exc
    conflict = "" if args.resume_from else _custom_specialist_conflict(args, SharedState())
    if conflict:
        print(f"ERROR: {conflict}.", file=sys.stderr)
        raise SystemExit(2)


async def _run_optimize(args: argparse.Namespace) -> int:
    """Run the ``optimize`` subcommand end to end."""
    plug_manifest = resolve_plug_manifest(args)  # first: a refused flag must leave any session as it was
    nodes_resolved = _resolve_run_topology(args)
    _settle_custom_specialist_only(args)

    # Stale aiter JIT lock sweep: killed runs leave locks that block subsequent starts (locks <5min preserved).
    aiter_sweep = clean_stale_aiter_locks()
    if aiter_sweep["dir"] and aiter_sweep["deleted"]:
        print(
            f"Stale aiter locks cleared: "
            f"dir={aiter_sweep['dir']} "
            f"deleted={aiter_sweep['deleted']} "
            f"skipped_fresh={aiter_sweep['skipped_fresh']} "
            f"errors={aiter_sweep['errors']}"
        )

    claude_follows_codex = _claude_model_should_follow_codex()
    if claude_follows_codex:
        args.claude_model = args.codex_model

    # Capture provider intent before _preflight() fills missing endpoints (preflight may populate OPENAI_BASE_URL from
    # ANTHROPIC_BASE_URL).
    codex_follows_claude = _codex_model_should_follow_claude()
    try:
        resolved_urls = _preflight(args)
    except Exception as exc:
        try:
            _persist_preflight_failure_artifacts(args, exc)
        except Exception:
            log.warning("failed to preserve SBD V6 preflight failure", exc_info=True)
        raise

    # Model resolution makes blocking LLM calls; the subscription-token client refuses to run inside this loop.
    await asyncio.to_thread(
        _resolve_models_for_run,
        args,
        resolved_urls,
        claude_follows_codex=claude_follows_codex,
        codex_follows_claude=codex_follows_claude,
    )
    _start_experience_kb()
    # Before either session branch: these are read by the fresh-launch seeding AND by the resume path, so this is the
    # one place that covers both.
    _preflight_agentx_backend(args)
    if not args.resume_from:
        _scope_error = latency_budget_scope_error(
            getattr(args, "framework", None) or os.environ.get("FRAMEWORK", ""), getattr(args, "max_latency_ms", None)
        )
        if _scope_error:
            print(f"ERROR: {_scope_error}.", file=sys.stderr)
            raise SystemExit(2)
    _apply_agentx_budget_profile(args)
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    resolve_benchmark_timeouts()
    # A fresh launch has nothing to restore, so settle the budget before the
    # manifest and the workload env read it. A resume keeps ``None`` until
    # ``_restore_budget_and_objective`` has had its chance at the session's own.
    if args.resume_from:
        session = await _resume_session(args)
    else:
        if args.max_hours is None:
            args.max_hours = DEFAULT_MAX_HOURS
        session = await _seed_fresh_session(args, nodes_resolved=nodes_resolved)
    session_dir = session.session_dir
    session_lock = session.session_lock
    manifest = session.manifest
    plug_host = open_plug_host(plug_manifest, session_dir)

    # Recipe KB T0 anchor (after seed/restore for recipe_canonical_id, before Coordinator); skipped when --degraded-kb.
    recipe_kb_client = _bootstrap_recipe_kb(
        args,
        session_dir=session_dir,
        manifest=manifest,
        resume=bool(args.resume_from),
    )
    # KnowledgePlane owns Recipe KB even when PR Monitor is degraded.
    knowledge_plane = _bootstrap_knowledge_plane(args, recipe_kb_client=recipe_kb_client)

    from ..multi_node.state_paths import bind_state_file_to_session

    bind_state_file_to_session(session_dir)
    if nodes_resolved >= 2:
        await asyncio.to_thread(_prepare_multi_node_state, args)

    objective_env = {"MAX_HOURS": str(args.max_hours)}
    for env_name, target in (
        ("TARGET_GAIN_PCT", args.target_gain),
        ("TARGET_TPUT_PER_GPU", args.target_tput),
        ("TARGET_DIR", args.target_baseline_dir),
        ("TARGET_WITHIN_ROOFLINE_PCT", args.target_roofline),
    ):
        objective_env[env_name] = str(target) if target else ""
    objective = build_objective(objective_env)
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    lineup = _resolve_agent_lineup(args)

    # T0 may have persisted warm-start state; preserve it before constructing the Coordinator.
    state = SharedState.load_or_init(session_dir)

    backends = _build_backends(
        ModelRouting(
            claude_model=args.claude_model,
            codex_model=args.codex_model,
            codex_follows_claude=codex_follows_claude,
            claude_follows_codex=claude_follows_codex,
            deepseek_model=llm_config.resolve_forge_llm_model(llm_config.AGENT_BACKEND_DEEPSEEK),
        ),
        CriticSetup(
            choice=lineup.critic_choice,
            agent_root=lineup.critic_agent_root,
            kb_mode=lineup.critic_kb_mode,
            protocol=args.critic_protocol,
        ),
        session_dir=session_dir,
    )
    # Expose active session_dir to in-process executors via the canonical pin env var; reinforced here for resume
    # paths.
    os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session_dir)
    # Production: enable strict PolicyGate path-containment (escaping intents land as policy_denied).
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # --reset-state backs up state.json and starts blank, before Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir, framework=state.framework)

    coordinator = Coordinator(
        session_dir,
        backends=backends,
        settings=CoordinatorSettings(
            model_class=getattr(args, "model_class", None) or os.environ.get("MODEL_CLASS", ""),
            phase_budget_pct=_build_phase_budget_pct(args),
        ),
        advisory=AdvisoryServices(
            # KnowledgePlane facade (None when --degraded-pr).
            knowledge_plane=knowledge_plane,
            # Advisory multi-model specialist-proposal scorer, disabled by default (enable via --proposal-scoring).
            proposal_scorer=_build_proposal_scorer(args, session_dir),
        ),
    )
    _initial_macro_cycle = int(getattr(coordinator.shared_state, "macro_cycle", 0) or 0)
    prompts = _install_system_prompts(coordinator, args, objective, lineup, macro_cycle=_initial_macro_cycle)
    # Build specialist executor only when research_lane capacity > 0 (0 degrades to LLM-direct grid).
    specialist_executor: "Any" = None
    if args.research_lane_capacity > 0:
        specialist_executor = _build_specialist_executor(
            args,
            session_dir=session_dir,
            knowledge_plane=knowledge_plane,
        )
    _register_executors(
        coordinator,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
        specialist_executor=specialist_executor,
    )
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(
        session_dir,
        prompts=prompts,
        orchestration_phase=coordinator.shared_state.phase,
        macro_cycle=_initial_macro_cycle,
    )
    _print_backend_banner(args, backends, lineup)

    warn_without_target_gpu(args)

    stop_reason: str | None = None
    try:
        with running_plug_host(plug_host):
            stop_reason = await coordinator.run(
                objective=objective,
                options=RunOptions(
                    max_minutes=args.max_hours * 60.0,
                    tick_interval_sec=args.tick_interval_sec,
                    max_ticks=args.max_ticks,
                    install_signal_handlers=True,
                    closing_grace_sec=args.closing_grace_sec,
                ),
            )
    finally:
        state = coordinator.shared_state
        effective_stop_reason = stop_reason or coordinator.stop_classification
        await coordinator.stop()
        # Drop the single-optimizer session lock once the
        # coordinator has released its leases. The OS would drop it on process
        # exit anyway; this just frees it promptly for an intentional resume.
        session_lock.release()
        _write_cli_terminal_artifacts(session_dir, state, effective_stop_reason)
        try:
            state.save(session_dir)
        except Exception:
            log.exception("failed to persist post-stop state (non-fatal)")

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep is now a SWEEP-phase action auto-enqueued by the Coordinator, not a post-hook here.

    _print_final_summary(coordinator.shared_state, stop_reason, session_dir)
    return _exit_code_for_stop_reason(stop_reason, coordinator.shared_state.baseline_tput)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and dispatch the requested subcommand."""
    # Force line-buffering so output piped through a non-TTY sink flushes every line immediately instead of
    # block-buffering, which would otherwise freeze the top-level log for the duration of a blocking Magpie
    # subprocess.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # Absolutise before the parser defaults or any session path derive from it: subprocesses run with their own cwd,
    # so a relative value would diverge.
    user_data = os.environ.get(ENV_USER_DATA_PATH, "")
    if user_data and not Path(user_data).is_absolute():
        os.environ[ENV_USER_DATA_PATH] = str(Path(user_data).expanduser().absolute())

    parser = _build_parser()
    # Strict on purpose.
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        from .benchmark_config import configure_benchmark

        try:
            configure_benchmark(args)
            _load_custom_specialist_args(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        with gpu_power_restore_guard():
            return asyncio.run(_run_optimize(args))
    if args.command == "recover":
        return _run_recover_session(args)
    parser.print_help()
    return 2
