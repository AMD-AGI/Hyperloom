# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL / ANTHROPIC_BASE_URL +
OPENAI_API_KEY / ANTHROPIC_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common import llm_config
from hyperloom.common.llm_config import CLAUDE_OAUTH_TOKEN_ENV, parse_custom_headers
from .executors import (
    _build_specialist_executor,
    _register_executors,
)
from .kb import (
    _bootstrap_recipe_kb,
    _bootstrap_knowledge_plane,
)
from .backends import (
    _build_backends,
    _build_proposal_scorer,
    _official_anthropic_only,
    _official_openai_only,
    resolve_robustness_options,
)
from .model_gate import (
    _autodetect_gpu_type,
    _gpu_runner_type,
    _load_model_max_position_embeddings,
    _finish_model_gate,
    _preflight_context_window,
    _preflight_model_config_compat,
    _preflight_unsupported_model_arch,
    _record_resumed_model_gate,
    _resolve_gpu_type,
    _resolve_max_model_len,
    _start_model_gate,
)
from ..model_config_utils import (
    summarize_model_config,
)
from .bootstrap import (
    _begin_resume_leg,
    _clean_stop_resume_budget_lines,
    _print_final_summary,
    _print_session_skeleton,
    _reconcile_crash_count,
    _seed_shared_state,
    _snapshot_system_prompts,
    agentx_state_is_stale,
    parse_operator_extra_env,
    resolve_model_display_name,
)
from hyperloom.orchestrator.actions.executors._aiter_jit import clean_stale_aiter_locks
from hyperloom.orchestrator.actions.executors._workload_envs import (
    agentx_enabled as _agentx_enabled,
)

from .credentials import (
    _CLAUDE_ALLOWED_MODELS as _CLAUDE_ALLOWED_MODELS,
    _CODEX_FALLBACK_MODELS as _CODEX_FALLBACK_MODELS,
    _CATALOG_RETRY_DELAYS_SEC as _CATALOG_RETRY_DELAYS_SEC,
    _CRITIC_AGENT_ROOT_ENV as _CRITIC_AGENT_ROOT_ENV,
    _ROBUSTNESS_AGENT_ROOT_ENV as _ROBUSTNESS_AGENT_ROOT_ENV,
    _resolve_agent_root as _resolve_agent_root,
    _validate_agent_runtime as _validate_agent_runtime,
)
from .multi_node import (
    _prepare_multi_node_state as _prepare_multi_node_state,
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
from ..protocol.action_surfaces import ACTION_CATALOGUE, ActionMetadata
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist
from hyperloom.orchestrator.state.objective import Objective, build_objective
from hyperloom.orchestrator.state.shared_state import SharedState, timed_teardown_step
from hyperloom.orchestrator.prompts.prompt_builder import (
    TRANSPORT_TOOLS,
    build_orchestration_prompt,
    default_enabled_actions,
)
from ..session.lock import SessionAlreadyRunning, SessionLock
from ..session.paths import (
    ENV_USER_DATA_PATH,
    asset_system_prompts_dir,
    make_session_dir,
)


log = logging.getLogger("hyperloom.inference_optimizer.cli")

from .parser import (
    _build_parser as _build_parser,
    _positive_int_arg as _positive_int_arg,
    _redact_unknown_args as _redact_unknown_args,
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from .preflight import (
    _check_gfx_arch_resolvable,
    _mark_pending_install_event_failed,
    _persist_install_event,
    _preflight as _preflight,
)


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only ``orchestration.md`` fragment consumed by ``prompt_builder``.

    Returns:
        Path: The path to the bundled ``orchestration.md`` fragment.
    """
    return asset_system_prompts_dir() / "orchestration.md"


def _normalise_framework_name(value: str | None) -> str:
    """Normalize a framework string for equality checks."""
    return str(value or "").strip().lower().replace("_", "-")


def _apply_operator_supplied_paths(args: Any, framework: str) -> None:
    """Publish ``--framework-path`` / ``--benchmark-scripts-dir`` as env.

    Both flags are friendlier spellings of env vars the executors already read,
    so this only forwards them; an explicitly exported env still wins, matching
    how every other path override in the loop behaves.

    ``custom`` has no shipped checkout or entrypoint to fall back on, so a
    missing or non-existent directory is fatal here rather than at the first
    benchmark, half an hour in. The benchmark backend is checked for the same
    reason: it defaults to Magpie, which cannot run an operator's script at all,
    and nothing downstream rejects the combination — the run would simply take
    the wrong executor and fail somewhere less obvious. Resume re-invokes this
    after restoring archived paths so a re-passed ``--benchmark-scripts-dir``
    is not silently dropped.
    """
    fatal: list[str] = []
    if framework == "custom":
        from hyperloom.orchestrator.actions.executors.benchmark_backend import (
            BENCHMARK_BACKEND_ENV,
            DEFAULT_BENCHMARK_BACKEND,
        )

        # Matches install.sh's normalisation so the two gates agree on a value
        # like " Bypass "; anything else, including unset, is refused.
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
        # An exported-but-empty var is not a choice, it is an unset one; treating
        # it as set would let a stray `export FRAMEWORK_REPO_PATH=` silently
        # swallow the flag.
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

    if fatal:
        for line in fatal:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(2)


def _restore_operator_supplied_paths_from_state(args: Any, state: SharedState) -> None:
    """Fill custom-workload env from persisted state when this resume omitted it.

    Priority is CLI flag > already-exported env > archived SharedState. The
    CLI flag is not written here: :func:`_apply_operator_supplied_paths`
    publishes it, and must see a vacant env for the flag to win. Archive is
    therefore applied only when neither the flag nor the env is set, matching
    how ``--server-args`` / ``--extra-env`` resume. ``benchmark_backend`` has
    no CLI flag (env / install.sh only), so the env check alone is the full
    precedence chain for that field.

    Args:
        args: Parsed CLI namespace (reads ``framework_path`` /
            ``benchmark_scripts_dir``).
        state: Resumed session state.
    """
    from hyperloom.orchestrator.actions.executors.benchmark_backend import (
        BENCHMARK_BACKEND_ENV,
    )

    cli_repo = str(getattr(args, "framework_path", None) or "").strip()
    cli_scripts = str(getattr(args, "benchmark_scripts_dir", None) or "").strip()
    if not cli_repo and not os.environ.get("FRAMEWORK_REPO_PATH", "").strip():
        archived = str(getattr(state, "framework_repo_path", "") or "").strip()
        if archived:
            os.environ["FRAMEWORK_REPO_PATH"] = archived
    if not cli_scripts and not os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip():
        archived = str(getattr(state, "bypass_scripts_dir", "") or "").strip()
        if archived:
            os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] = archived
    if not os.environ.get(BENCHMARK_BACKEND_ENV, "").strip():
        archived = str(getattr(state, "benchmark_backend", "") or "").strip()
        if archived:
            os.environ[BENCHMARK_BACKEND_ENV] = archived


def _require_custom_entrypoint(framework: str, gpu_type: str | None = None) -> None:
    """Fail at launch when ``--framework custom`` cannot resolve its script.

    Walks the same chain the bypass runner uses (``apply_scriptable_runtime_defaults``
    then :func:`resolve_scriptable_script`) so a missing ``custom_{runner}.sh``
    (or lone operator script) exits in seconds with the candidate list, instead
    of looping ``magpie_nonzero_invalid_measurement`` until robustness stops
    the session.

    No-op for shipped frameworks. Must run after GPU_TYPE is resolved: the
    runner suffix is part of the filename.

    Side effect: publishes ``CUSTOM_REPO_PATH`` / ``CUSTOM_DIR`` into
    ``os.environ`` via ``apply_scriptable_runtime_defaults``, which PolicyGate
    needs anyway to allowlist the custom checkout.

    Args:
        framework: Resolved session framework.
        gpu_type: Magpie runner GPU type (e.g. ``mi355x``).
    """
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


def _enforce_expected_framework(
    framework: str,
    *,
    expected: str | None = None,
) -> None:
    """Fail fast when a launcher-pinned expected framework is violated.

    Long-running launches often pass through generated shell scripts. A stale
    script that mutates ``$FRAMEWORK`` can otherwise silently run a different
    backend than the operator requested. ``EXPECTED_FRAMEWORK`` is the compact
    launcher-facing guard; ``INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK`` is the
    namespaced equivalent for platform integrations.
    """
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
    """Summarise an objective into the ``(kind, value)`` pair the prompt expects.

    Inspects the objective for the first recognised target attribute
    (``target_gain_pct`` → float, ``target_tput_per_gpu`` → float,
    ``baseline_dir`` → str) and pairs it with the objective's ``kind()``.

    Args:
        objective (Objective): The run objective to summarise.

    Returns:
        tuple[str, float | str | None]: ``(kind, value)`` where ``value`` is the
        objective's numeric / string target, or ``None`` when none is present.
    """
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
    no_framework_agent: bool = False,
    macro_cycle: int = 0,
    cycle_directive: str = "",
    phase: str = "",
    transport: str = TRANSPORT_TOOLS,
    action_registry: Mapping[str, ActionMetadata] | None = None,
) -> str:
    """Compose the Orchestration system prompt from typed inputs (``--orch-prompt`` overrides).

    Args:
        no_kernel (bool): When ``True`` the kernel actions are disabled.
        framework (str): The serving framework name (e.g. ``sglang``).
        objective (Objective): The run objective summarised into the prompt.
        max_minutes (int): The wall-clock budget in minutes.
        no_framework_agent (bool): When ``True`` the FRAMEWORK_AGENT phase is disabled.
        macro_cycle (int): Current macro-cycle counter; shown in the CYCLE DIRECTIVE section.
        cycle_directive (str): LLM-authored focus text for this cycle; empty renders the default arc.
        phase (str): Current pipeline phase; omits the prompt modules whose
            behaviour that phase cannot reach. Empty renders every module.
        transport (str): How the orchestration backend carries an intent; omits
            the prompt modules describing a tool surface it does not mount.
        action_registry (Mapping[str, ActionMetadata] | None): The action
            catalogue to use; defaults to :data:`ACTION_CATALOGUE`.

    Returns:
        str: The composed Orchestration system prompt.
    """
    registry = action_registry or ACTION_CATALOGUE
    enabled = default_enabled_actions(no_kernel=no_kernel, no_optimize=no_framework_agent)
    kind, value = _objective_summary_for_prompt(objective)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework=framework,
        kernel_enabled=not no_kernel,
        framework_agent_phase_enabled=not no_framework_agent,
        objective_kind=kind,
        objective_value=value,
        max_minutes=int(max_minutes),
        macro_cycle=int(macro_cycle),
        cycle_directive=cycle_directive,
        phase=phase,
        transport=transport,
        rules_fragment_path=_orchestration_rules_fragment_path(),
        framework_source_roots=resolve_source_file_allowlist(),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``orchestrator/prompts/critic.md``.

    Returns:
        str: The contents of ``critic.md``.
    """
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


# Per-attempt read timeout for the gateway /models catalog probe. Operator
# override via env (default 5.0s) for slow-gateway windows.
try:
    _CATALOG_REQUEST_TIMEOUT_SEC = float(
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_TIMEOUT_SEC", "5.0") or "5.0"
    )
except (TypeError, ValueError):
    _CATALOG_REQUEST_TIMEOUT_SEC = 5.0


def _should_remote_probe_gpu(args: argparse.Namespace) -> bool:
    """Return whether the GPU type should be probed on the remote cluster.

    Multi-node runs the CLI on a GPU-less sandbox, so the real inference GPU
    lives on the handed-over cluster and must be probed via the Ray head
    (rayjob) or Infera SSH. Single-node runs keep the local probe.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``nodes`` /
            ``mn_backend``).

    Returns:
        bool: True when ``--nodes >= 2`` and the backend is rayjob/infera
        (``--mn-backend`` unset defaults to rayjob).
    """
    if int(getattr(args, "nodes", 1) or 1) < 2:
        return False
    backend = (getattr(args, "mn_backend", None) or "rayjob").strip().lower()
    return backend in ("rayjob", "infera")


def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """Validate atom-specific CLI knobs: sole job is the ``--nodes>=2`` fail-fast guard (IR-8).

    No auto-tightening is applied; kernel/framework/profile all work on atom. Multi-node TP wiring
    is unimplemented so ``--nodes>=2`` exits 2. Returns the list of auto-disabled flags (always empty).

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``nodes``).

    Returns:
        list[str]: The auto-disabled flags (always empty).

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` (unsupported on atom).
    """
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
        "  framework=atom: no auto-disable applied (kernel-agent + "
        "framework-agent + profile / roofline / TraceLens all wired "
        "for atom); --nodes>=2 guard active — see SKILL.md IR-8"
    )
    return auto_disabled


def _emit_launch_info(
    *,
    pid: int,
    session_dir: Path,
    session_id: str,
    run_log: str,
    gpu_type: str,
    framework: str,
    model: str,
    launch_info_file: str | None,
) -> dict[str, Any]:
    """Print the machine-readable HYPERLOOM_LAUNCH stdout line; optionally JSON-dump to ``launch_info_file``.

    Returns the launch_info dict for callers/tests.

    Args:
        pid (int): The launched process id.
        session_dir (Path): The session root directory.
        session_id (str): The session identifier.
        run_log (str): The run-log path string.
        gpu_type (str): The resolved GPU type.
        framework (str): The serving framework name.
        model (str): The model name / path.
        launch_info_file (str | None): Optional path to JSON-dump the launch
            info; skipped when ``None``.

    Returns:
        dict[str, Any]: The launch-info dict that was printed (and optionally
            written).
    """
    launch_info: dict[str, Any] = {
        "event": "launch",
        "pid": pid,
        "session_dir": str(session_dir),
        "session_id": session_id,
        "run_log": run_log,
        "manifest": str(session_dir / "manifest.json"),
        "gpu_type": gpu_type,
        "framework": framework,
        "model": model,
    }
    kv_body = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in launch_info.items())
    print(f"HYPERLOOM_LAUNCH {kv_body}")
    if launch_info_file:
        path = Path(launch_info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(launch_info, indent=2))
        print(f"Launch info file: {path}")
    return launch_info


# Exit code for "another optimizer already owns this session". Distinct from
# generic config/usage failures (``2``) so the robustness monitor can tell a
# refused duplicate launch from a real misconfiguration.
SESSION_BUSY_EXIT_CODE = 3


def _acquire_session_lock_or_exit(session_dir: Path) -> SessionLock:
    """Take the single-optimizer session lock or exit ``SESSION_BUSY_EXIT_CODE``.

    Guards both fresh ``optimize`` and ``--resume-from`` against a second optimizer
    attaching to the same ``session_dir``. When a live optimizer already owns
    the session this refuses to run before any ``state.json`` / lease mutation.

    Args:
        session_dir (Path): The resolved session root directory.

    Returns:
        SessionLock: The acquired lock; the caller must keep it referenced for
            the optimizer's lifetime.
    """
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


# Sentinel returned by _probe_llm_catalog when the gateway has no /models route
# (HTTP 404/405). Distinct from None (auth/network/server error / empty catalog)
# so the caller can proceed for an endpoint that exposes no catalog.
_CATALOG_NO_MODELS_ENDPOINT: frozenset[str] = frozenset()


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | frozenset[str] | None:
    """Probe ``<base_url>/models`` with retry (gateway flakes); return set of model ids or None.

    Args:
        base_url (str): The gateway base URL; ``""`` returns ``None``.
        api_key (str): Optional bearer key sent in the ``Authorization``
            header.

    Returns:
        set[str] | None: The set of model ids from ``<base_url>/models``, or
            ``None`` when the probe is skipped or exhausts its retries.
    """

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
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} failed: {last_err}")
            continue
        if resp.status_code in (404, 405):
            # The endpoint has no /models route; not a transient/auth error, so
            # stop retrying and signal "no catalog endpoint" distinctly.
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
    """Build headers for a direct ``<base_url>/models`` probe.

    Applies the operator's custom headers for the side being probed: the OpenAI
    base uses ``OPENAI_CUSTOM_HEADERS``; the Anthropic base and any manual probe
    override use ``ANTHROPIC_CUSTOM_HEADERS`` (Anthropic is the primary catalog
    target, so it is the default). Gateway-specific headers (e.g. an AMD
    ``Ocp-Apim-Subscription-Key``) must be supplied via those env vars — no
    host-specific auto-injection.
    """
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
    """True when both protocol sides are served by one gateway.

    Either the same URL, or the same host with a different path per protocol
    (``/anthropic`` vs ``/v1``) — the shape the installers call a dual-protocol
    gateway.
    """
    if not anthropic_url or not openai_url:
        return False
    if anthropic_url == openai_url:
        return True
    from urllib.parse import urlsplit

    return urlsplit(anthropic_url).netloc == urlsplit(openai_url).netloc


def _codex_model_should_follow_claude() -> bool:
    """True when the operator supplied only Anthropic config."""
    return _official_anthropic_only()


def _claude_model_should_follow_codex() -> bool:
    """True when the operator supplied only OpenAI-compatible config."""
    if os.environ.get("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX") == "1":
        return True
    return _official_openai_only()


def _catalog_probe_has_no_credential() -> bool:
    """True when a Claude subscription token is the only credential available.

    The catalog probe is a bearer-authenticated ``GET <base>/models``; a
    subscription token is not accepted there, so probing would only produce a
    misleading auth failure.
    """
    if not os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip():
        return False
    # Any other Anthropic-side form can authenticate the probe, so the list of
    # what disqualifies "oauth-only" is the registry minus the token itself,
    # derived rather than restated. OPENAI_API_KEY joins it because the probe
    # accepts either side's bearer. Read off the module so the registry stays a
    # single source at call time, not a tuple frozen at import.
    bearer_capable = (
        *(n for n in llm_config.ANTHROPIC_CREDENTIAL_ENV_ORDER if n != CLAUDE_OAUTH_TOKEN_ENV),
        "OPENAI_API_KEY",
    )
    return not any(os.environ.get(name, "").strip() for name in bearer_capable)


def _custom_orch_model_allowed() -> bool:
    """Whether orchestration may use a model outside the AMD Claude allowlist.

    Custom orchestration models are enabled by default so provider-specific
    model IDs (for example DeepSeek) can run when they are present in the
    configured gateway catalog. Operators can set
    ``INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`` (or false/no/off) to
    restore the stricter AMD Claude allowlist.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _custom_orch_model_explicitly_disabled() -> bool:
    """Whether the operator explicitly requested strict AMD model allowlisting."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL")
    return raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}


def _critic_agent_runtime_needed(critic_choice: str) -> bool:
    """Whether the selected critic path will actually instantiate critic-agent.

    Every provider shape runs the full critic-agent: its KB two-phase runtime is
    protocol-independent, and both review transports keep it. There is no
    degraded critic to fall back to — ``--critic-mock`` is the only opt-out.
    """
    return critic_choice == "agent"


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> set[str] | None:
    """Gate Claude model selection against the gateway catalog; mutates ``args.claude_model``.

    Probes the gateway catalog (retries); on a miss it walks
    ``_CLAUDE_ALLOWED_MODELS`` in order and falls back to the first id the
    gateway serves with a WARN, else sys.exit(2). Returns the
    catalog id set on success (reused by the codex smoke-test). The AMD
    ``_CLAUDE_ALLOWED_MODELS`` allowlist is enforced only when the operator sets
    ``INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL`` to 0/false/no/off (and is
    additionally waived on the codex-follow path); otherwise the catalog probe
    is the sole gate.

    Args:
        args (argparse.Namespace): The parsed CLI namespace; ``claude_model``
            may be mutated to the fallback model.
        resolved_urls (tuple[str, str] | None): Optional
            ``(anthropic_base_url, openai_base_url)`` used to resolve the probe
            base URL when env is unset.

    Returns:
        set[str] | None: The gateway catalog id set on success.

    Raises:
        SystemExit: With code 2 when the model is disallowed, the catalog is
            unreachable, or no acceptable model is present.
    """
    chosen = (args.claude_model or "").strip()
    # Custom orchestration models are enabled by default; the gateway catalog
    # probe below is the sole gate. Set INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0
    # to restore the stricter AMD Claude allowlist.
    allow_custom = _custom_orch_model_allowed()
    if not _custom_orch_model_explicitly_disabled():
        allow_custom = allow_custom or _claude_model_should_follow_codex()
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

    # Catalog probe GETs <base>/models. Probe the Anthropic side first (the
    # orchestration model is Claude); fall back to the OpenAI side when it has
    # no reachable catalog. INFERENCE_OPTIMIZER_CATALOG_PROBE_URL overrides the
    # host outright (single probe, no fallback).
    catalog_ids: set[str] | frozenset[str] | None = None
    override_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not override_url and _catalog_probe_has_no_credential():
        # The static allowlist gate above already ran; this is the only check the
        # probe would have added, so proceed with the operator's id.
        print(
            "Preflight: catalog probe skipped: oauth-only credential "
            f"(CLAUDE_CODE_OAUTH_TOKEN); cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None
    if override_url:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        catalog_ids = _probe_llm_catalog(base_url=override_url, api_key=api_key)
    else:
        anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not anthropic_url and resolved_urls is not None:
            anthropic_url = resolved_urls[0]
        if not openai_url and resolved_urls is not None:
            openai_url = resolved_urls[1]
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        # OpenAI-side key only.
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        # The Claude catalog must come from the Anthropic side. The OpenAI side
        # is offered as a second candidate only when both sides are the same
        # gateway -- identical URLs, or one host serving each protocol on its
        # own path -- and the loop below only reaches it when the first side
        # turns out to have no /models route at all.
        candidates: list[tuple[str, str]] = []
        if _claude_model_should_follow_codex():
            if openai_url:
                candidates.append((openai_url, openai_key))
            elif anthropic_url:
                candidates.append((anthropic_url, anthropic_key))
        else:
            # A keyless Anthropic candidate can only fail — a subscription token
            # is not a catalog credential — so it must not consume the one probe
            # slot and leave a configured OpenAI gateway unverified.
            if anthropic_url and anthropic_key:
                candidates.append((anthropic_url, anthropic_key))
            if openai_url and ((anthropic_url and _same_gateway(anthropic_url, openai_url)) or not candidates):
                # One gateway serving both protocols, or the only side left once
                # a keyless Anthropic side was skipped above.
                candidates.append((openai_url, openai_key))
        seen_urls: set[str] = set()
        for cand_url, cand_key in candidates:
            if not cand_url or cand_url in seen_urls:
                continue
            seen_urls.add(cand_url)
            catalog_ids = _probe_llm_catalog(base_url=cand_url, api_key=cand_key)
            # A missing /models route is the only answer worth re-asking on the
            # other side, and only because a dual-protocol gateway lists its
            # models there. An unreachable side (None) means the gateway is
            # flaky rather than route-less: probing the OpenAI side would then
            # answer a Claude question with an OpenAI catalog and fail every
            # allowlisted id, so stop and let the caller degrade to its WARN.
            if catalog_ids is not _CATALOG_NO_MODELS_ENDPOINT:
                break

    if catalog_ids is _CATALOG_NO_MODELS_ENDPOINT:
        # The gateway has no /models route; the model cannot be verified here,
        # so proceed rather than refuse.
        print(
            f"Preflight: WARNING — gateway has no /models route (HTTP 404/405); "
            f"cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None

    if catalog_ids is None:
        # Auth/network/server/non-JSON/empty-catalog failure: genuinely
        # unverifiable. Only proceed under the explicit opt-out.
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

    # For non-allowlisted custom ids the AMD fallback is meaningless; fail
    # clearly on a catalog miss. Allowlisted Claude ids keep the fallback below.
    if allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} not present in gateway catalog "
            f"(custom orchestration model support enabled; catalog has "
            f"{sorted(catalog_ids)[:20]}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Walk the allowlist in order so it acts as a real preference ladder: a
    # gateway that carries opus-4-8 but not the newer default must land on 4-8,
    # not skip two generations down to the last entry.
    # Allowlist ids are already in the probe's normalized form, so a plain
    # membership test is enough here.
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
    resolved_urls: tuple[str, str] | None,
    *,
    claude_follows_codex: bool | None = None,
    codex_follows_claude: bool | None = None,
) -> None:
    """Resolve both model ids against the gateway before any session work.

    Order matters in the single-provider deploys. When the Codex model also
    becomes the orchestration model, its ladder has to run *first*: otherwise
    the Claude gate sees an id derived from a Codex default the gateway may not
    serve, and aborts on a model the operator never chose.

    Args:
        args (argparse.Namespace): Parsed CLI namespace; both ``claude_model``
            and ``codex_model`` may be rewritten.
        resolved_urls (tuple[str, str] | None): ``(anthropic_url, openai_url)``
            from preflight.
        claude_follows_codex (bool | None): Whether the orchestration model is
            derived from ``codex_model`` (OpenAI-only deploy). Callers pass the
            value captured before ``_preflight`` filled in missing endpoints;
            ``None`` re-derives it from the environment.
        codex_follows_claude (bool | None): The Anthropic-only mirror image.

    Raises:
        SystemExit: With code 2 when the Claude gate rejects the model.
    """
    if claude_follows_codex is None:
        claude_follows_codex = _claude_model_should_follow_codex()
    if codex_follows_claude is None:
        codex_follows_claude = _codex_model_should_follow_claude()

    if claude_follows_codex:
        # codex_model is about to become the orchestration model, so it needs
        # the ladder whatever the critic backend is.
        _smoke_test_codex_model(args, resolved_urls, required=True)
        args.claude_model = args.codex_model

    # Hard-gate the Claude model (mutates args.claude_model on fallback; sys.exit(2) on failure).
    _validate_and_resolve_claude_model(args, resolved_urls)

    if codex_follows_claude:
        args.codex_model = args.claude_model
    elif not claude_follows_codex:
        # Codex smoke probes the OpenAI side independently (split entrypoints).
        _smoke_test_codex_model(args, resolved_urls)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
    *,
    required: bool = False,
) -> None:
    """WARN-only catalog check for ``--codex-model``; flags typos and steps down the ladder before Coordinator starts.

    Probes the OpenAI-side catalog independently of the Claude check: in a
    split-entrypoint deploy the Claude catalog lives on the Anthropic gateway
    and would not list ``gpt-*``, so reusing it would always false-warn.

    Unlike the Claude gate this never aborts, but it mirrors its ladder: a
    ``_CODEX_FALLBACK_MODELS`` id the gateway does not serve is rewritten to the
    newest one it does, so a gateway lagging behind the default degrades at
    preflight instead of on the first Codex turn. Ids outside that tuple are the
    operator's own choice and are only reported.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``codex_model`` / ``critic_backend``); ``codex_model`` may be
            mutated to a fallback.
        resolved_urls (tuple[str, str] | None): ``(anthropic_url, openai_url)``
            from preflight; the OpenAI side is probed for the Codex catalog.
        required (bool): Check even when no critic-agent will run. Set on the
            OpenAI-only path, where ``codex_model`` also drives orchestration
            and so matters regardless of the critic backend.
    """
    if not required:
        if _codex_model_should_follow_claude():
            return
        if args.critic_backend != "agent":
            return

    openai_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not openai_url:
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not openai_url and resolved_urls is not None:
        openai_url = resolved_urls[1]
    # OpenAI-side key only.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    catalog_ids = _probe_llm_catalog(base_url=openai_url, api_key=openai_key)
    if catalog_ids is None:
        # WARN-only path: don't block startup just because the OpenAI catalog
        # is unreachable (the Claude gate already validated reachability).
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
        f"{list(_CODEX_FALLBACK_MODELS)}) or use --critic-mock to "
        f"avoid the Codex path entirely."
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
    """Resolve a backend choice from CLI args with validation and fallback to default.

    Args:
        attr (str): The ``args`` attribute name to read (e.g. ``"critic_backend"``).
        default (str): The fallback value when the attribute is ``None``.
        valid (tuple[str, ...]): Allowable backend names; hard-fails outside this set.
        flag_hint (str): Human-readable hint for the error message describing how to
            set the value (e.g. ``"--critic-mock / --critic-agent or
            INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND"``).
        args (argparse.Namespace): The parsed CLI namespace.

    Returns:
        tuple[str, bool]: ``(chosen, explicit)`` where ``chosen`` is the resolved
        backend name and ``explicit`` is ``True`` when the arg was set by the
        caller (not defaulted).

    Raises:
        SystemExit: With code 2 when the resolved backend is not in ``valid``.
    """
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
    """Resolve the active critic backend choice (arg → DEFAULT_CRITIC_BACKEND); hard-fails on invalid.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``critic_backend``).

    Returns:
        str: The resolved critic backend (one of ``_VALID_CRITIC_BACKENDS``).

    Raises:
        SystemExit: With code 2 when the chosen backend is invalid.
    """
    chosen, _ = _resolve_choice(
        "critic_backend",
        DEFAULT_CRITIC_BACKEND,
        _VALID_CRITIC_BACKENDS,
        "--critic-mock / --critic-agent or INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
        args=args,
    )
    return chosen


# Default robustness backend ("agent"); force heartbeat-only mock via --robustness-mock or env.
DEFAULT_ROBUSTNESS_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND",
    "agent",
)
_VALID_ROBUSTNESS_BACKENDS = ("mock", "agent")


def _resolve_robustness_choice(args: argparse.Namespace) -> str:
    """Resolve the active robustness backend choice (arg → DEFAULT_ROBUSTNESS_BACKEND); hard-fails on invalid.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``robustness_backend``).

    Returns:
        str: The resolved robustness backend (one of
            ``_VALID_ROBUSTNESS_BACKENDS``).

    Raises:
        SystemExit: With code 2 when the chosen backend is invalid.
    """
    chosen, _ = _resolve_choice(
        "robustness_backend",
        DEFAULT_ROBUSTNESS_BACKEND,
        _VALID_ROBUSTNESS_BACKENDS,
        "--robustness-mock / --robustness-agent or INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND",
        args=args,
    )
    return chosen


def _reset_state_file(session_dir: Path) -> None:
    """Back up ``state.json`` to ``state.json.preReset.<unix_ts>`` and start fresh (Recipe KB untouched).

    Args:
        session_dir (Path): The session root directory holding ``state.json``.
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
    import logging as _logging

    _logging.getLogger(__name__).info(
        "--reset-state: backed up state.json to %s; session starts blank.",
        backup_path.name,
    )


# AgentX conc-sweep budgets, sized from a measured round: 35B / conc 64 /
# 3600s window = ~111 min end to end (46 min per-lane warmup + 60 min
# measurement + ~5 min boot, corpus load and mapping).
#
# The overtime kill ratio is deliberately NOT raised. It was, until the measured
# round showed the reasoning was wrong: a duration-based replay COMPRESSES
# runtime spread rather than widening it, because the measurement window is
# fixed. Only warmup scales with how slow a config is, so a variant with 3x
# slower warmup still lands at 1.8x the baseline total -- comfortably inside the
# stock 2.0x. Raising it would also have inverted the hard-cap/soft-kill
# layering (see AGENTX_EXPLORE_TIMEOUT_CEILING_SEC), which is the real fix.
#
# The total is that measured round times the ladder: seven agentic rungs on each
# of the baseline and optimized arms, so fourteen. It is a ceiling rather than a
# reservation -- the SWEEP phase clamps it to the session's own remaining time
# before the task is enqueued, and an exhausted budget skips the rungs it cannot
# pay for rather than failing the sweep. Sizing it below the ladder would instead
# truncate every run by default.
_AGENTX_CONC_SWEEP_TIMEOUT_SEC = 9000  # 2.5 h per variant
_AGENTX_CONC_SWEEP_TOTAL_BUDGET_SEC = 93600  # 26 h: ~111 min x 14 rungs


def _preflight_agentx_backend(args: argparse.Namespace) -> None:
    """Reject the combinations where AgentX labels work it did not do.

    The AgentX switch works by overwriting ``benchmark.benchmark_script``, and
    there are two ways for that overwrite not to happen while every other
    AgentX-gated behaviour still fires:

    * The bypass backend never reads that field for a serving framework, so
      ``HYPERLOOM_BENCHMARK_BACKEND=bypass`` produces a full run of ordinary
      synthetic measurements labelled as an AgentX session.
    * ``apply_agentx_switch`` returns early for a scriptable framework (xdit and
      friends have no serving endpoint for aiperf to drive). ``agentx_enabled()``
      is a bare env read, though, so the session still disables eval, turns the
      concurrency sweep off, collapses the ISL/OSL grid, widens every budget,
      and stamps ``benchmark_mode="agentx"`` on its state -- over a workload that
      never touched a trace.

    Neither is set on purpose; refuse instead of handing back numbers that mean
    something other than what they claim.

    Args:
        args: Parsed CLI namespace; ``framework`` decides the scriptable case.

    Raises:
        SystemExit: When AgentX is combined with either.
    """
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
    """Widen the per-variant time budgets for AgentX's much longer runs.

    A synthetic benchmark round is minutes; an AgentX round replays a real trace
    corpus for a fixed measurement window plus dataset load, per-lane warmup and
    drain -- measured at ~111 min for a 35B model at concurrency 64. The stock
    conc-sweep budgets are sized for the former and would record every AgentX
    variant as ``budget_exhausted`` before it finished.

    Only knobs still at their parser default are touched; a value the operator
    typed is left exactly as typed. Two things are deliberately NOT changed:

    - ``--max-hours`` is the operator's contract with the scheduler / pod lease.
      Silently extending it gets the job killed from outside the process, which
      is far harder to diagnose than running out of budget. It only warns.
    - ``--explore-overtime-kill-ratio`` stays at its default, because a
      duration-based replay compresses runtime spread instead of widening it
      (only warmup scales with a slow config; the measurement window is fixed).
      The per-variant hard cap is what needed adjusting, and that lives in
      ``explore.AGENTX_EXPLORE_TIMEOUT_CEILING_SEC``.

    No-op unless ``HYPERLOOM_AGENTX`` is on.

    Args:
        args: Parsed CLI namespace, mutated in place.
    """
    if not _agentx_enabled():
        return
    if float(getattr(args, "max_hours", 0) or 0) == 2.0:
        print(
            "NOTE: HYPERLOOM_AGENTX is on and --max-hours is at its default of 2.0. "
            "One AgentX round (corpus load + warmup + drain + measurement window) "
            "typically exceeds that on its own; pass an explicit --max-hours sized "
            "to the number of candidates you intend to measure.",
            file=sys.stderr,
        )
    if int(getattr(args, "conc_sweep_timeout_sec", 0) or 0) == 1800:
        args.conc_sweep_timeout_sec = _AGENTX_CONC_SWEEP_TIMEOUT_SEC
    if int(getattr(args, "conc_sweep_total_budget_sec", 0) or 0) == 9000:
        args.conc_sweep_total_budget_sec = _AGENTX_CONC_SWEEP_TOTAL_BUDGET_SEC


# Largest input+output a single request reaches in the 256k-capped weka corpus,
# measured over all 30,141 requests of semianalysis_cc_traces_weka_062126_256k
# (max input alone is 255,808). That variant is what every model family outside
# upstream's 1M-context whitelist replays, and it is deliberately the
# conservative of the two: the whitelist lives in aiperf_client.sh and
# duplicating it here would give two copies to keep in sync, while the families
# on it carry ~1M contexts and so never trip this bound anyway.
AGENTX_CAPPED_CORPUS_PEAK_TOKENS = 255_999


def _warn_if_context_too_small_for_corpus(resolved: int, source: str) -> None:
    """Say up front when the served window cannot hold the replay corpus.

    Not fatal: the run is still the honest one to make, and a model that cannot
    hold the corpus cannot hold a leaderboard row either. But the failure would
    otherwise surface an hour in, as requests the server refuses accumulate into
    a context-overflow rate the scenario turns into submission_valid=false (or,
    past ``--failed-request-threshold``, an abort) -- with nothing pointing at
    the cause. Deliberately checked against the RESOLVED value whatever its
    source: an explicit ``--max-model-len 8192`` or a stale exported
    ``$MAX_MODEL_LEN`` are the likeliest ways to get this wrong, and gating the
    check on the auto-resolved branch would stay silent for exactly those.
    """
    if not _agentx_enabled() or resolved >= AGENTX_CAPPED_CORPUS_PEAK_TOKENS:
        return
    print(
        f"WARNING: HYPERLOOM_AGENTX is on but the served context window "
        f"({resolved}, from {source}) is below the replay corpus peak "
        f"({AGENTX_CAPPED_CORPUS_PEAK_TOKENS} input+output tokens). Requests that "
        f"do not fit will be refused by the server; the run is expected to come "
        f"back non-submittable, and a model this size is not comparable on the "
        f"AgentX board.",
        file=sys.stderr,
    )


def _resolve_run_max_model_len(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve run-wide MAX_MODEL_LEN with explicit operator values winning."""
    resolved = _resolve_run_max_model_len_inner(args)
    _warn_if_context_too_small_for_corpus(*resolved)
    return resolved


def _resolve_run_max_model_len_inner(args: argparse.Namespace) -> tuple[int, str]:
    """Resolution proper; the corpus-fit warning is applied by the caller."""
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
    # AgentX replays real agentic traces whose lengths come from the corpus, so
    # ISL/OSL are meaningless placeholders here (they default to 1024/1024) and
    # ``ISL+OSL+headroom`` would pin the server at ~6k against traces reaching
    # ~1M tokens -- the corpus would then be silently reduced to whatever fits.
    # Upstream's agentic path requires the model's own default context; mirror
    # that. Explicit operator values above still win.
    if _agentx_enabled():
        native = _load_model_max_position_embeddings(str(args.model or ""))
        if native:
            return int(native), "agentx-native-context"
        # Model not on disk yet (uncached HF id). Nothing downstream recomputes
        # this -- the client deliberately never emits a context cap and the
        # server phase is a bare delegation -- so the fallback below really is
        # what the server gets. It is a synthetic-shape derivation that does not
        # describe this workload; the corpus-fit warning fires on it.
        print(
            "WARNING: HYPERLOOM_AGENTX is on but the model's native context could "
            "not be read (weights not on disk yet), so MAX_MODEL_LEN falls back "
            "to the synthetic ISL+OSL derivation and the server will be booted at "
            "that width. Pre-fetch the weights, or pass --max-model-len.",
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


def _resume_can_disable_eval(baseline_accuracy: float) -> bool:
    """Whether ``--no-eval`` may still disable the accuracy eval for a resumed session.

    The cutoff is the anchored accuracy rather than a phase: the baseline runs
    inside PRELUDE, so the phase alone cannot say whether a reference exists.
    Once it does, every KEEP so far was graded against it.

    Args:
        baseline_accuracy (float): The persisted ``state.baseline_accuracy``.

    Returns:
        bool: ``True`` while no accuracy is anchored yet.
    """
    return float(baseline_accuracy or 0.0) <= 0.0


def _build_phase_budget_pct(args: argparse.Namespace) -> dict[str, float]:
    """Map ``--*-pct`` CLI flags to a ``phase -> pct`` override dict.

    Keys MUST be the canonical phase names from
    :mod:`hyperloom.orchestrator.phases.machine_state`; otherwise
    :func:`normalize_budget_pct` silently drops the entry and the phase falls
    back to its library default.
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
        val = getattr(args, cli_field, None)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)
    return phase_budget_pct


def _detect_checkpoint_precision(model_path: str | None) -> str:
    """Detect weight precision from model config.json; returns '' on failure."""
    if not model_path:
        return ""
    try:
        from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

        summary = summarize_model_config(str(model_path))
    except Exception:  # noqa: BLE001
        return ""
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


def _resolve_workload_knobs(
    args: argparse.Namespace,
    state: Any | None = None,
) -> None:
    """Fill unset workload knobs on ``args`` from a fixed priority ladder.

    Priority: explicit CLI flag (non-``None``) > resumed ``SharedState`` value >
    fallback default. Writes the resolved values back onto ``args`` so every
    downstream consumer (SharedState seed, manifest, env projection) reads one
    authoritative source instead of racing argparse defaults against env
    (issue #903). Inherited process env is deliberately NOT a config source.

    Args:
        args: Parsed CLI namespace; mutated in place.
        state: Resumed ``SharedState`` whose persisted knobs win over defaults
            when the flag is unset; ``None`` on a fresh launch.
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
            persisted = int(getattr(state, name, 0) or 0) if state is not None else 0
            val = persisted if persisted > 0 else default
        setattr(args, name, int(val))
    precision = getattr(args, "precision", None)
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
    nodes_resolved: int,
    tp_resolved: int,
    ep_resolved: int,
    argv: list[str] | None = None,
) -> None:
    """Project resolved workload knobs (TP/CONC/EP) into env for downstream Magpie YAMLs.

    After ``_resolve_workload_knobs`` the values on ``args`` are already the
    authoritative resolution (flag > resume-state > default), so export them
    unconditionally. This keeps SharedState, the manifest, and the materialized
    YAML in agreement instead of the old gated export that only fired for
    explicit flags / multi-node and left SharedState and the served value split
    (issue #903).

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``conc``).
        nodes_resolved (int): The resolved node count (unused; retained for the
            call-site contract).
        tp_resolved (int): The resolved tensor-parallel size to export as ``TP``.
        ep_resolved (int): The resolved expert-parallel size to export as ``EP``.
        argv (list[str] | None): Unused; retained for the call-site contract.
    """
    os.environ["TP"] = str(max(1, int(tp_resolved or 1)))
    os.environ["CONC"] = str(max(1, int(getattr(args, "conc", DEFAULT_CONC) or DEFAULT_CONC)))
    os.environ["EP"] = str(max(1, int(ep_resolved or 1)))


def _export_operator_launch_shape(
    *,
    server_args: str,
    extra_env: dict[str, str],
) -> None:
    """Project the operator's ``--server-args`` / ``--extra-env`` into env.

    Internal handoff for the in-process executors, not a public config API. A
    fresh launch feeds it from argv, a resume from the persisted SharedState.
    Empty inputs clear the variables so a second session in the same shell
    cannot inherit a stale value.

    Args:
        server_args: The resolved ``--server-args`` string, possibly empty.
        extra_env: The resolved ``--extra-env`` pins, possibly empty.
    """
    if server_args:
        os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] = server_args
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_SERVER_ARGS", None)
    if extra_env:
        os.environ["INFERENCE_OPTIMIZER_EXTRA_ENV"] = json.dumps(extra_env)
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_EXTRA_ENV", None)


def _partition_fanout_supported(framework: str | None) -> tuple[bool, str]:
    """Whether this framework's runner can place work per partition.

    Split out and returned rather than tested inline because the interesting
    case is the third one. ``--framework`` defaults to ``None`` and is resolved
    later, so a truthiness guard here reads as "checked and fine" while actually
    meaning "not checked" -- and the operator sees nothing either way.

    Args:
        framework: The session framework, possibly unresolved.

    Returns:
        ``(supported, detail)``. ``detail`` is empty when supported, and
        otherwise says whether the answer is *no* or *not yet known*.
    """
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


def _export_partition_shape(
    *,
    declared_mode: str | None,
    streams_per_partition: int | None,
    framework: str | None = None,
    gpu_type: str | None = None,
    nodes: int = 1,
    model_path: str | None = None,
    precision: str | None = None,
    shared_state: Any = None,
) -> dict[str, Any]:
    """Validate the session's compute-partition shape and publish it.

    Read-only. The mode belongs to the card and is set outside the optimizer;
    this observes what the card is in, refuses a session that cannot work in
    that shape, and hands the shape to the benchmark entrypoint that fans work
    out across partitions.

    Validating at launch is the whole point. Every failure this catches -- a mode
    that was never applied, a partition too small for the workload -- otherwise
    surfaces hours later as an out-of-memory crash or, worse, as a perfectly
    good measurement filed under a topology the card was never in.

    Args:
        declared_mode: The raw ``--compute-partition-mode`` value, if any.
        streams_per_partition: The raw ``--streams-per-partition`` value, if any.
        framework: The resolved session framework. Decides whether anything will
            place work per partition, which gates both the footprint refusal and
            the runtime hand-off. Must be resolved before this is called: an
            unresolved framework reads as "cannot fan out".
        gpu_type: Board name, used only if the device's CU count cannot be read.
        nodes: Resolved node count. The shape describes the card this process
            can see, which on a multi-node session is not the benchmark's, so
            nothing is observed or recorded there.
        model_path: Checkpoint to size the workload from. Without it the
            feasibility check has nothing to weigh and can only warn.
        precision: Resolved precision, which sets the bytes per weight.
        shared_state: Persisted state on a resume, consulted for the model
            identity and for a ``peak_gib_per_stream`` that nothing in this
            repository writes -- so in practice the footprint is the
            weight-bytes bound either way.

    Returns:
        The published shape, or ``{}`` when this session has none.
    """
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

    # Cleared first so a second session in the same shell cannot inherit a shape
    # the operator did not ask for this time.
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
    # Tested against None rather than falsiness: `0 or DEFAULT` is DEFAULT, which
    # would quietly honour an invalid request as the default instead of refusing
    # it, and leave the guard below unreachable for the one value most likely to
    # be passed by mistake.
    streams = DEFAULT_STREAMS_PER_PARTITION if streams_per_partition is None else int(streams_per_partition)
    if streams < 1:
        print(
            f"ERROR: --streams-per-partition must be >= 1, got {streams_per_partition}",
            file=sys.stderr,
        )
        sys.exit(2)

    if nodes >= 2:
        # Unconditional: the misleading record is produced by observing at all,
        # not by asking for a mode. A controller that happens to have GPUs would
        # otherwise publish its own topology as the session's while the
        # benchmark ran somewhere else entirely.
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
        params={
            "model_path": str(model_path or ""),
            "precision": str(precision or ""),
        },
        shared_state=shared_state,
        # The footprint refusal is arithmetic about streams sharing a partition.
        # An operator who named the flags has asserted that shape and is held to
        # it; otherwise it only applies where something will actually fan out.
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
    # The observed shape is published either way -- the platform fingerprint
    # reads it back from here, and provenance is the point. Only the fan-out
    # instruction is withheld from a session whose benchmark cannot act on it.
    os.environ.update(runtime_env(verdict.layout, streams, fanout=fanout))
    return session_shape_summary(verdict.layout, streams, fanout_expected=fanout)


def _restore_partition_shape_from_state(args: Any, state: SharedState) -> None:
    """Fill the partition flags from the archive when this resume omitted them.

    Priority is CLI flag > archived ``SharedState``, the same chain
    :func:`_restore_operator_supplied_paths_from_state` applies to the
    custom-workload paths. Resolved values are written back onto ``args`` so
    :func:`_export_partition_shape` stays the only writer of the env it owns,
    and so a restored declaration is re-checked against the live card rather
    than trusted -- the card may have been repartitioned while the session was
    stopped, which is exactly the case the declaration exists to catch.

    Args:
        args: Parsed CLI namespace, updated in place.
        state: Resumed session state.
    """
    archived = dict(getattr(state, "compute_partition", None) or {})
    if not str(getattr(args, "compute_partition_mode", None) or "").strip():
        args.compute_partition_mode = str(archived.get("mode") or "")
    # `is None` rather than falsiness, so an explicit `--streams-per-partition 0`
    # still reaches the guard that refuses it instead of being read as "omitted"
    # and quietly replaced by the archived value.
    if getattr(args, "streams_per_partition", None) is None:
        streams = archived.get("streams_per_partition")
        args.streams_per_partition = int(streams) if streams else None


# Terminal stop_reasons that represent a clean, successful optimizer run (exit 0).
# Anything else (baseline / preflight failures, crashes, enablement stalls) exits
# non-zero so CI surfaces genuine problems.
_SUCCESS_STOP_REASONS: frozenset[str] = frozenset(
    {
        "target_reached",
        "global_converged",
        "time_exhausted",
        "max_ticks",
        # SWEEP finished cleanly: exit_normal_sweep returns sweep_done once the
        # concurrency ladder settles, which means the run optimized and
        # closed normally (e.g. the no-kernel path), so neither is a CI failure.
        "sweep_done",
    }
)


def _exit_code_for_stop_reason(stop_reason: str | None) -> int:
    """Map a terminal ``stop_reason`` to a process exit code (0 success, 1 failure)."""
    return 0 if (stop_reason or "") in _SUCCESS_STOP_REASONS else 1


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
    try:
        session_dir = _new_preflight_failure_session_dir(
            args,
            failed_attempt=bool(str(getattr(args, "resume_from", "") or "").strip()),
        )
    except Exception:  # noqa: BLE001 — never replace the original preflight failure
        log.warning("failed to create a session for SBD V6 preflight failure", exc_info=True)
        return None

    session_lock = SessionLock(session_dir)
    try:
        session_lock.acquire()
    except Exception:  # noqa: BLE001 — never replace the original preflight failure
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
            except Exception as write_exc:  # noqa: BLE001 — the install event can still stand alone
                log.warning("failed to write manifest for SBD V6 preflight failure", exc_info=True)
                from ..session.sbd_v6 import record_write_warning

                record_write_warning(session_dir, component="preflight_failure.manifest", exc=write_exc)

        _persist_install_event(args, session_dir)
        try:
            from ..breakdown import write_breakdown_json

            write_breakdown_json(session_dir)
        except Exception as write_exc:  # noqa: BLE001 — never replace the original preflight failure
            log.warning("failed to write SBD V6 preflight failure breakdown", exc_info=True)
            from ..session.sbd_v6 import record_write_warning

            record_write_warning(session_dir, component="preflight_failure.breakdown", exc=write_exc)
    finally:
        session_lock.release()
    print(f"Preflight failure artifacts: {session_dir}", file=sys.stderr)
    return session_dir


async def _run_optimize(args: argparse.Namespace) -> int:
    """Run the ``optimize`` subcommand end to end.

    Resolves topology arguments (nodes, TP/EP, GPUs per node), runs
    preflight, and drives the optimization session to completion.

    Args:
        args: Parsed CLI arguments for the ``optimize`` subcommand.

    Returns:
        Process exit code (``0`` on success).
    """
    # Surface --nodes (CLI flag wins) before _preflight runs.
    nodes_resolved = max(1, int(args.nodes))
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    # Resolve gpus_per_node from the explicit CLI flag or the policy default.
    gpn_attr = getattr(args, "gpus_per_node", None)
    if gpn_attr is not None:
        gpus_per_node_resolved = int(gpn_attr)
    else:
        gpus_per_node_resolved = 8
    total_gpus = nodes_resolved * gpus_per_node_resolved

    # Topology sanity gates — multi-node only (nodes>=2); fail fast vs a cryptic launcher crash mid-cold-start.
    if nodes_resolved >= 2:
        # Gate 1: total cluster GPUs (nodes*gpus_per_node) must hold the model's TP shards.
        if total_gpus < tp_resolved:
            print(
                f"ERROR: TP={tp_resolved} exceeds total GPU count "
                f"({nodes_resolved} nodes * {gpus_per_node_resolved} "
                f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
                "--nodes, or use a larger --gpus-per-node pod "
                "template.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Gate 2: EP cannot exceed TP (can't place more expert shards than ranks); fail before bootstrap.
        if ep_resolved > tp_resolved:
            print(
                f"ERROR: EP={ep_resolved} > TP={tp_resolved}. Expert-parallel "
                "size must be <= tensor-parallel size. Either lower --ep or "
                "raise --tp.",
                file=sys.stderr,
            )
            sys.exit(2)

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Multi-node topology handoff: export the CLI-flag-resolved backend /
    # gpus-per-node so downstream subprocesses (kernel agent, benchmark, KB
    # topology, state synthesis) read a single stable source. These are internal
    # handoff envs, not a public config API; users pass --mn-backend /
    # --gpus-per-node instead. The pod image is not among them: the platform
    # builds the pods before the optimizer starts.
    if nodes_resolved >= 2:
        os.environ["INFERENCE_OPTIMIZER_GPUS_PER_NODE"] = str(gpus_per_node_resolved)
        os.environ["INFERENCE_OPTIMIZER_MN_BACKEND"] = _resolve_mn_backend(args)
    _export_operator_launch_shape(
        server_args=str(getattr(args, "server_args", "") or "").strip(),
        extra_env=parse_operator_extra_env(args),
    )
    # The partition shape is deliberately NOT exported here. It needs the
    # resolved framework, GPU type and post-quantization model path, none of
    # which exist yet, and each fresh-launch and resume branch calls it once at
    # the point where they do. Exporting here as well made a resume validate
    # twice -- the first time against an unresolved model it could only warn
    # about.

    # Project resolved workload knobs into env for the fresh-launch path only.
    # A resume must NOT export here: ``args.tp``/etc. are still unresolved
    # (``None`` -> 1) because the persisted SharedState is loaded later; the
    # resume branch re-exports the real values after ``_resolve_workload_knobs``
    # so downstream (incl. preflight) never sees the placeholder default.
    if not args.resume_from:
        _export_workload_envs_for_optimize(
            args,
            nodes_resolved=nodes_resolved,
            tp_resolved=tp_resolved,
            ep_resolved=ep_resolved,
        )
    # User-declared grid skip list; re-export so subprocess executors inherit it (empty clears stale values).
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
        os.environ["INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX"] = "1"
        args.claude_model = args.codex_model
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX", None)

    # Capture provider intent before _preflight() fills missing endpoints
    # (preflight may populate OPENAI_BASE_URL from ANTHROPIC_BASE_URL).
    codex_follows_claude = _codex_model_should_follow_claude()
    try:
        resolved_urls = _preflight(args)
    except Exception as exc:  # noqa: BLE001 — only unexpected defects create diagnostic sessions
        try:
            _persist_preflight_failure_artifacts(args, exc)
        except Exception:  # noqa: BLE001 — preserve the original failure exactly
            log.warning("failed to preserve SBD V6 preflight failure", exc_info=True)
        raise

    _resolve_models_for_run(
        args,
        resolved_urls,
        claude_follows_codex=claude_follows_codex,
        codex_follows_claude=codex_follows_claude,
    )
    # Before either session branch: these are read by the fresh-launch seeding
    # AND by the resume path, so this is the one place that covers both.
    _preflight_agentx_backend(args)
    _apply_agentx_budget_profile(args)

    if args.resume_from:
        # USER_DATA_PATH stays at the workspace root; --resume-from names a subdir under it.
        from ..session.paths import (
            ENV_CURRENT_SESSION_DIR,
            workspace_root,
        )

        ws = workspace_root()
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
        # Pin before Coordinator/SharedState load so paths/subprocesses inherit the resolved location.
        os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
        # Ensure per-session skeleton exists (idempotent mkdir -p).
        for sub in __import__(
            "hyperloom.inference_optimizer.session.paths", fromlist=["_SESSION_SKELETON"]
        )._SESSION_SKELETON:
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        # Single-optimizer guard: take the session lock before any state.json /
        # lease access. Held for the whole run.
        session_lock = _acquire_session_lock_or_exit(session_dir)
        _persist_install_event(args, session_dir)

        try:
            manifest = load_manifest(session_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --resume-from failed: {exc}", file=sys.stderr)
            sys.exit(2)
        if not (session_dir / "state.json").exists():
            print(
                f"ERROR: --resume-from failed: {session_dir}/state.json missing "
                f"(manifest exists but Coordinator never wrote SharedState)",
                file=sys.stderr,
            )
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        _stale = agentx_state_is_stale(state)
        if _stale:
            print(
                f"ERROR: cannot resume this session -- {_stale}. "
                "Start a fresh session instead of mixing the two measurement sets.",
                file=sys.stderr,
            )
            sys.exit(2)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  manifest.session_id    : {manifest.get('session_id')}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain_validated:.2f}%")
        print(
            f"  prior current_best    : "
            f"{(state.current_best or {}).get('action')}/"
            f"{(state.current_best or {}).get('tput')}"
        )
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")

        # Re-export session-level env from persisted state so a fresh-shell resume doesn't fall back to YAML defaults.
        if state.model_path:
            resume_model_path = str(state.model_path)
            os.environ["MODEL_PATH"] = resume_model_path
            print(f"  re-exported MODEL_PATH: {resume_model_path}")
            # Backfill model_info for sessions created before the field existed
            # (or whose config was unreadable at launch); fail-soft to {}.
            if not state.model_info:
                state.model_info = summarize_model_config(resume_model_path)
                if state.model_info:
                    state.save(session_dir)
                    print("  backfilled model_info (from config.json)")
        if state.framework:
            _enforce_expected_framework(state.framework)
            os.environ["FRAMEWORK"] = state.framework
            print(f"  re-exported FRAMEWORK : {state.framework}")
        if state.gpu_type:
            runner_gpu_type = _gpu_runner_type(state.gpu_type)
            os.environ["TARGET_GPU_TYPE"] = state.gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
            if runner_gpu_type != state.gpu_type:
                print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
        # Resolve workload knobs with the resumed state as the fallback source
        # (explicit --isl/--conc/... on this resume still win), then project the
        # resolved values into env so resume sees the same workload contract
        # (not YAML defaults). ``ep`` mirrors EP so single-node vLLM MoE resume
        # still injects --enable-expert-parallel.
        _resolve_workload_knobs(args, state)
        _resume_max_model_len = getattr(args, "max_model_len", None) or getattr(state, "max_model_len", 0) or 0
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
        # Profile-scoped OSL: an explicit --profile-osl on this resume wins;
        # otherwise re-export the value persisted from the original run.
        _resume_profile_osl = getattr(args, "profile_osl", None) or getattr(state, "profile_osl", 0)
        if _resume_profile_osl:
            os.environ["PROFILE_OSL"] = str(int(_resume_profile_osl))
            state.profile_osl = int(_resume_profile_osl)
            print(f"  re-exported PROFILE_OSL   : {int(_resume_profile_osl)}")
        if args.precision:
            os.environ["PRECISION"] = args.precision
            print(f"  re-exported PRECISION     : {args.precision}")
        if getattr(state, "framework_version", ""):
            os.environ["FRAMEWORK_VERSION"] = state.framework_version
            print(f"  re-exported FRAMEWORK_VERSION: {state.framework_version}")
        # Operator launch shape: an explicit flag on this resume wins, else the
        # persisted value.
        _resume_server_args = str(getattr(args, "server_args", "") or "").strip() or state.operator_server_args
        _resume_extra_env = parse_operator_extra_env(args) or dict(state.operator_extra_env)
        _export_operator_launch_shape(
            server_args=_resume_server_args,
            extra_env=_resume_extra_env,
        )
        state.operator_server_args = _resume_server_args
        state.operator_extra_env = _resume_extra_env
        if _resume_server_args:
            print(f"  re-exported server_args   : {_resume_server_args}")
        if _resume_extra_env:
            print(f"  re-exported extra_env     : {','.join(sorted(_resume_extra_env))}")
        # Custom-workload paths: an explicit --framework-path /
        # --benchmark-scripts-dir on this resume wins, else the persisted
        # value. Without this, a fresh-shell resume leaves
        # HYPERLOOM_BYPASS_SCRIPTS_DIR unset and every grid variant exits
        # rc=2 before spawn (magpie_nonzero_invalid_measurement, blank error).
        _restore_operator_supplied_paths_from_state(args, state)
        _apply_operator_supplied_paths(args, state.framework or "sglang")
        _require_custom_entrypoint(
            state.framework,
            gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
        )
        _persist_operator_supplied_paths(state)
        # The partition shape is part of the measurement contract, so it resumes
        # on the same restore / apply / persist path as the paths above. The
        # re-check is not ceremony: a card can be repartitioned while a session
        # is stopped, and resuming into a different topology would compare
        # candidates measured under one shape against a baseline from another.
        _restore_partition_shape_from_state(args, state)
        state.compute_partition = _export_partition_shape(
            declared_mode=getattr(args, "compute_partition_mode", None),
            streams_per_partition=getattr(args, "streams_per_partition", None),
            framework=state.framework or getattr(args, "framework", None),
            gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
            # A resume must re-pass --nodes, so the persisted count is the one
            # that says whether this session was ever multi-node.
            nodes=max(int(getattr(args, "nodes", 1) or 1), int(getattr(state, "nodes", 1) or 1)),
            model_path=state.model_path or str(getattr(args, "model", "") or ""),
            precision=state.precision or getattr(args, "precision", None),
            # Passed for the persisted model identity. It is also where a
            # measured per-stream peak would be read from, but nothing writes
            # one, so a resume sizes against the same weight-bytes bound as a
            # fresh launch.
            shared_state=state,
        )
        if state.compute_partition.get("mode"):
            print(f"  re-exported partition shape: {state.compute_partition['mode']}")
        if state.framework_repo_path:
            print(f"  re-exported FRAMEWORK_REPO_PATH: {state.framework_repo_path}")
        if state.bypass_scripts_dir:
            print(f"  re-exported HYPERLOOM_BYPASS_SCRIPTS_DIR: {state.bypass_scripts_dir}")
        if state.benchmark_backend:
            print(f"  re-exported HYPERLOOM_BENCHMARK_BACKEND: {state.benchmark_backend}")
        # Feeds the robustness defaults and the IR-8 check only. ``nodes_resolved``
        # was fixed from argv at the top of this function, so the cluster hand-off
        # is already settled and a multi-node resume must re-pass --nodes.
        if state.nodes > 1 and int(getattr(args, "nodes", 1) or 1) <= 1:
            args.nodes = state.nodes
            os.environ["INFERENCE_OPTIMIZER_NODES"] = str(state.nodes)
            print(f"  re-exported nodes         : {state.nodes}")
        # Warm-replay gates: a non-default value is the only "operator set this"
        # signal, since the parser gives these flags real defaults rather than None.
        if args.no_warm_replay:
            state.warm_replay_enabled = False
        for _wr_attr, _wr_default in (
            ("warm_replay_min_confidence", 0.7),
            ("warm_replay_min_reproduce_pct", 0.8),
        ):
            _wr_value = getattr(args, _wr_attr)
            if _wr_value != _wr_default:
                setattr(state, _wr_attr, _wr_value)
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
        # Same persistence contract for the eval toggle.
        if state.eval_disabled:
            args.no_eval = True
            print("  accuracy eval         : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_eval", False)):
            anchored = float(state.baseline_accuracy or 0.0)
            if _resume_can_disable_eval(anchored):
                state.eval_disabled = True
                state.save(session_dir)
                print("  accuracy eval         : DISABLING for resume (--no-eval + no anchored accuracy)")
            else:
                print(
                    f"  accuracy eval         : WARN --no-eval ignored; "
                    f"session already anchored accuracy={anchored:.4f} "
                    f"(cannot retroactively ungrade prior KEEPs)"
                )

        prior_crash = state.crash_count

        # target_reached is a terminal state requiring --force-resume to push
        # past it; other reasons auto-clear.
        force_resume = bool(getattr(args, "force_resume", False))
        gated_terminal = {"target_reached"}
        if prior_stop in gated_terminal and not force_resume:
            print(
                f"\nERROR: --resume-from blocked by terminal stop_reason="
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

        reanchor_budget = bool(prior_stop or prior_crash >= 3)
        _begin_resume_leg(state, reanchor_budget=reanchor_budget)
        state.save(session_dir)
        _record_resumed_model_gate(
            args,
            session_dir,
            workload_overrides={
                "model_path": str(state.model_path or manifest.get("model_path") or ""),
                "model_name": str(state.model_name or manifest.get("model_name") or ""),
                "framework": str(state.framework or manifest.get("framework") or ""),
                "gpu_type": str(state.gpu_type or manifest.get("gpu_type") or ""),
            },
        )
        if reanchor_budget:
            override_note = " (--force-resume override)" if force_resume and prior_stop in gated_terminal else ""
            print(f"  → cleared stop_reason and reset crash_count (was {prior_crash}) for fresh resume{override_note}")
            print(f"  → reset start_ts to {state.start_ts} (resume budget)")
        else:
            for line in _clean_stop_resume_budget_lines(
                state,
                max_hours=float(getattr(args, "max_hours", 0) or 0),
            ):
                print(line)
        # Re-bootstrap the recipe KB client (recreates client + reruns T0 warm-start); skipped when --degraded-kb.
        recipe_kb_client = _bootstrap_recipe_kb(
            args,
            session_dir=session_dir,
            manifest=manifest,
            resume=True,
        )
        # KnowledgePlane owns Recipe KB even when PR Monitor is degraded.
        knowledge_plane = _bootstrap_knowledge_plane(
            args,
            recipe_kb_client=recipe_kb_client,
            session_dir=session_dir,
        )
        # No resume backfill needed for roofline (roofline_snapshots restored by SharedState.from_dict).
    else:
        # Resolve model path: --model > $MODEL_PATH; fail fast rather than silently use the YAML hardcoded model.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume-from <session_dir> to "
                "continue an existing session).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export so subprocess executors inject the resolved model into the Magpie YAML, not its hardcoded model.
        from hyperloom.common.model_paths import resolve_serving_model_path

        os.environ["MODEL_PATH"] = resolve_serving_model_path(str(args.model)) or str(args.model)

        # Quantization prelude (one-shot, before any session/baseline work):
        # if --quantize was passed, quantize the source model now and rewrite
        # args.model to the exported quantized model. No-op otherwise.
        await _run_quantization_prelude(args)

        # Resolve framework: --framework > $FRAMEWORK > "sglang" (session-wide; no framework mixing).
        framework = (
            args.framework or os.environ.get("FRAMEWORK", "")
        ).strip().lower() or framework_registry.DEFAULT_FRAMEWORK
        if not framework_registry.is_supported(framework):
            print(
                f"ERROR: --framework must be one of "
                f"{', '.join(framework_registry.names())} "
                f"(got {framework!r}); set $FRAMEWORK accordingly or pass "
                "--framework",
                file=sys.stderr,
            )
            sys.exit(2)
        _enforce_expected_framework(framework)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")
        _apply_operator_supplied_paths(args, framework)

        # B3: --framework atom auto-tightens incompatible phases (see _apply_atom_auto_tighten).
        if framework == "atom":
            _apply_atom_auto_tighten(args)

        # Resolve real target GPU: probe > --gpu-type hint; probe wins to catch wrong-host typos that corrupt KB.
        # Multi-node runs the CLI on a GPU-less sandbox, so the local probe sees
        # nothing; probe the real inference GPU on the handed-over cluster (Ray
        # head / Infera SSH) instead. Best-effort: on failure probed stays empty
        # and the --gpu-type / $GPU_TYPE hint wins.
        user_specified = (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        if _should_remote_probe_gpu(args):
            from ..multi_node._internal.gpu_probe import remote_autodetect_gpu_type

            probed = remote_autodetect_gpu_type() or ""
            if probed:
                print(f"GPU probe       : {probed} (remote {(args.mn_backend or 'rayjob').lower()})")
        else:
            probed = _autodetect_gpu_type() or ""
        gpu_type, gpu_warnings = _resolve_gpu_type(
            user_specified=user_specified,
            probed=probed,
        )
        for line in gpu_warnings:
            print(line, file=sys.stderr)
        if probed and not user_specified:
            print(f"GPU type        : {gpu_type} (auto-detected)")
        runner_gpu_type = _gpu_runner_type(gpu_type)
        if gpu_type and runner_gpu_type != gpu_type:
            print(
                f"WARN: {gpu_type} uses {runner_gpu_type} as Magpie "
                f"runner_type (same gfx942/CDNA3 arch; Magpie has no "
                f"sglang_{gpu_type}.sh / vllm_{gpu_type}.sh yet)",
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
        _require_custom_entrypoint(framework, gpu_type=runner_gpu_type or gpu_type)

        # Runs here, not in _preflight, because the question it asks -- will
        # provenance be able to name the ISA? -- is unanswerable until
        # args.gpu_type is final. Asked earlier it sees only the raw hint, so it
        # warns on the bare-metal hosts the probe resolves correctly moments
        # later, and a check that cries wolf is one people stop reading.
        _check_gfx_arch_resolvable(args.gpu_type)

        # Resolve workload knobs (flag > default; no resume state on a fresh
        # launch) so ISL/OSL/CONC/TP/EP are authoritative reals before
        # MAX_MODEL_LEN auto-derivation and env projection (issue #903).
        _resolve_workload_knobs(args)
        # MAX_MODEL_LEN is operator-overridable. Auto resolution only runs when
        # neither --max-model-len nor $MAX_MODEL_LEN was supplied.
        max_model_len, max_model_len_source = _resolve_run_max_model_len(args)
        args.max_model_len = max_model_len
        os.environ["MAX_MODEL_LEN"] = str(max_model_len)
        os.environ["ISL"] = str(args.isl)
        os.environ["OSL"] = str(args.osl)
        # Profile-scoped OSL (issue #571): exported only when explicitly set, so
        # the profile/roofline materializer can decouple its OSL from the served
        # workload. Unset leaves the profile phase on the global --osl.
        if getattr(args, "profile_osl", None) is not None:
            os.environ["PROFILE_OSL"] = str(args.profile_osl)
        os.environ["PRECISION"] = args.precision
        # Mirror resolved framework_version into env (explicit > auto-detect > unset; see _resolve_framework_version).
        _fw_version_for_env = (getattr(args, "framework_version", None) or "").strip() or (
            os.environ.get("FRAMEWORK_VERSION", "") or ""
        ).strip()
        if not _fw_version_for_env:
            from ..recipe_snapshot_constants import (
                DEFAULT_FRAMEWORK_VERSION_SLUG,
                detect_framework_version,
            )

            _detected = detect_framework_version(
                (getattr(args, "framework", None) or "").strip() or os.environ.get("FRAMEWORK", "")
            )
            if _detected and _detected != DEFAULT_FRAMEWORK_VERSION_SLUG:
                _fw_version_for_env = _detected
        if _fw_version_for_env:
            os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
        print(
            f"Workload        : ISL={args.isl} OSL={args.osl} "
            f"MAX_MODEL_LEN={max_model_len} ({max_model_len_source}) "
            f"PRECISION={args.precision} "
            f"FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}"
        )

        # session_dir defaults to <workspace_root>/<model>/<UTC ts>-<rand8>/.
        # Use the resolved identity so a quantized run is named after the source
        # model (e.g. "<model>-quantized") instead of the generic export-dir
        # basename "quantized".
        session_dir = make_session_dir(model_name=resolve_model_display_name(args))
        # Single-optimizer guard: take the lock so the contract holds uniformly
        # and the owner pid is published for the robustness monitor.
        session_lock = _acquire_session_lock_or_exit(session_dir)
        manifest = write_manifest(session_dir, args=args)
        _persist_install_event(args, session_dir)
        # One-shot Langfuse startup marker so a run killed before a breakdown
        # still leaves a correlatable trace. Best-effort, never fatal.
        try:
            from hyperloom.orchestrator.trace.langfuse_emitter import record_session_start

            record_session_start(session_dir)
        except Exception:  # noqa: BLE001 — startup marker must never break launch
            log.debug("langfuse record_session_start failed (non-fatal)", exc_info=True)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)

        # Machine-readable launch info: stable point for launcher scripts to harvest pid/session_dir/run_log.
        _emit_launch_info(
            pid=os.getpid(),
            session_dir=session_dir,
            session_id=str(manifest["session_id"]),
            run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
            gpu_type=gpu_type or "",
            framework=args.framework or "",
            model=str(args.model) if args.model else "",
            launch_info_file=getattr(args, "launch_info_file", None),
        )
        # Placed here, not at the top of _run_optimize, because everything it
        # weighs is resolved by now and none of it was then: the framework
        # (whether anything fans out), args.gpu_type (the CU fallback), and
        # args.model, which --quantize rewrites to the exported checkpoint --
        # sizing partitions against the source model would weigh the wrong
        # weights. Still before the seed, so the shape it returns is the one
        # persisted rather than a lossy re-read from the environment.
        compute_partition = _export_partition_shape(
            declared_mode=getattr(args, "compute_partition_mode", None),
            streams_per_partition=getattr(args, "streams_per_partition", None),
            framework=framework,
            gpu_type=args.gpu_type,
            nodes=nodes_resolved,
            model_path=str(args.model or os.environ.get("MODEL_PATH") or ""),
            precision=getattr(args, "precision", None),
        )
        state = _seed_shared_state(
            session_dir,
            args,
            session_id=manifest["session_id"],
            compute_partition=compute_partition,
        )
        _start_model_gate(args, session_dir)
        # Unsupported-model preflight: reject multimodal/vision configs (runs after seed, before heavy bring-up).
        if _preflight_unsupported_model_arch(args, session_dir):
            sys.exit(2)
        # Model-config compatibility preflight: reject statically-broken
        # configs before the heavy server bring-up.
        if _preflight_model_config_compat(args, session_dir):
            sys.exit(2)
        # Context-window preflight: reject when ISL+OSL+headroom exceeds max_position_embeddings (no stretch by policy).
        if _preflight_context_window(args, session_dir):
            sys.exit(2)
        _finish_model_gate(args, session_dir)
        # Recipe KB T0 anchor (after seed for recipe_canonical_id, before Coordinator); skipped when --degraded-kb.
        recipe_kb_client = _bootstrap_recipe_kb(
            args,
            session_dir=session_dir,
            manifest=manifest,
            resume=False,
        )
        # KnowledgePlane owns Recipe KB even when PR Monitor is degraded.
        knowledge_plane = _bootstrap_knowledge_plane(
            args,
            recipe_kb_client=recipe_kb_client,
            session_dir=session_dir,
        )

    from ..multi_node.state_paths import bind_state_file_to_session

    bind_state_file_to_session(session_dir)
    if nodes_resolved >= 2:
        await asyncio.to_thread(_prepare_multi_node_state, args)

    objective = build_objective(
        {
            "MAX_HOURS": str(args.max_hours),
            "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
            "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
            "TARGET_DIR": args.target_baseline_dir or "",
        }
    )
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
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
        # The mock critic issues no review inference, so there is no transport
        # for the flag to select. Warned rather than rejected: the pairing is
        # harmless, and failing here would break scripts that pass a protocol
        # unconditionally and toggle the mock per run.
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
        critic_agent_root = _resolve_agent_root("critic")
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
        _validate_agent_runtime(critic_agent_root, agent="critic")
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

    # Resolve robustness backend choice + runtime root, mirroring critic.
    robustness_choice = _resolve_robustness_choice(args)
    robustness_agent_root: Path | None = None
    # T0 anchor writes warm_start_* via its own SharedState load; reload before
    # persisting launch-shape fields so seed/resume memory cannot clobber them.
    state = SharedState.load_or_init(session_dir)
    robustness_options = resolve_robustness_options(args, state)
    state.robustness_options = robustness_options
    # Persist before the Coordinator reads SharedState off disk, or the launch
    # shape stays in-memory only and the next resume falls back to defaults.
    state.save(session_dir)
    if robustness_choice == "agent":
        robustness_agent_root = _resolve_agent_root("robustness")
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
        _validate_agent_runtime(robustness_agent_root, agent="robustness")

    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        critic_choice=critic_choice,
        session_dir=session_dir,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
        robustness_choice=robustness_choice,
        robustness_agent_root=robustness_agent_root,
        robustness_options=robustness_options,
        codex_follows_claude=codex_follows_claude,
        critic_protocol=args.critic_protocol,
    )
    # Expose active session_dir to in-process executors via the canonical pin
    # env var; reinforced here for resume paths. Do NOT overwrite
    # USER_DATA_PATH — it must remain the workspace root for concurrent sessions
    # and install.sh on shared filesystems (WekaFS).
    os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session_dir)
    # Production: enable strict PolicyGate path-containment (escaping intents land as policy_denied).
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # PolicyGate R1 phase_incompatible enforcement for production runs (env affects cli boot path only).
    if getattr(args, "strict_phase", True):
        os.environ["INFERENCE_OPTIMIZER_STRICT_PHASE"] = "1"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_STRICT_PHASE", None)
    # --reset-state backs up state.json and starts blank, before Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir)
    from hyperloom.inference_optimizer.breakdown.exporter import set_default_include_transcripts

    transcripts_flag = str(getattr(args, "breakdown_include_transcripts", "false") or "false").strip().lower()
    set_default_include_transcripts(transcripts_flag == "true")
    # Build phase budget pct dict from CLI flags; absent values fall back to Coordinator library defaults.
    phase_budget_pct = _build_phase_budget_pct(args)

    coordinator = Coordinator(
        session_dir,
        backends=backends,
        model_class=(getattr(args, "model_class", None) or os.environ.get("MODEL_CLASS") or ""),
        recipe_kb=recipe_kb_client,
        phase_budget_pct=phase_budget_pct or None,
        # KnowledgePlane facade (None when --degraded-pr).
        knowledge_plane=knowledge_plane,
        # Advisory multi-model specialist-proposal scorer, disabled by default
        # (enable via --proposal-scoring). When active it scores each
        # proposal_set and surfaces results to Orchestration without gating.
        # Not persisted across a resume. ``session_dir`` lets it append per-model
        # token usage to the full-trace ledger (component=proposal_scorer).
        proposal_scorer=_build_proposal_scorer(args, session_dir),
        # Warm-recipe replay controls. Default ON; fires when
        # warm_start_recipe.confidence >= min_confidence and the measured gain
        # reproduces at least min_reproduce_pct of the recipe's claim.
        # SharedState is the persistent authority across restarts.
        warm_replay_enabled=state.warm_replay_enabled,
        warm_replay_min_confidence=state.warm_replay_min_confidence,
        warm_replay_min_reproduce_pct=state.warm_replay_min_reproduce_pct,
    )
    framework_for_prompt = os.environ.get("FRAMEWORK", "").strip().lower() or "sglang"
    max_minutes_for_prompt = int(round(float(args.max_hours) * 60))
    _initial_macro_cycle = int(getattr(coordinator.shared_state, "macro_cycle", 0) or 0)
    _initial_directive = str(
        (dict(getattr(coordinator.shared_state, "orchestration_memory", {}) or {})).get("next_cycle_directive", "")
        or ""
    )
    # A fresh session has no phase recorded yet and always begins at PRELUDE;
    # the Coordinator re-scopes the prompt at every later phase seam.
    from hyperloom.orchestrator.phases.machine_state import PHASE_PRELUDE as _PHASE_PRELUDE

    _initial_phase = coordinator.shared_state.phase or _PHASE_PRELUDE
    # The role record always names Claude; the deployment's provider shape is
    # what actually decides, so read the transport off the built backend.
    _orch_transport = getattr(coordinator.backends.get("orchestration"), "transport", TRANSPORT_TOOLS)
    prompts: dict[str, str] = {
        "orchestration": args.orch_prompt
        or _build_orchestration_prompt(
            no_kernel=no_kernel,
            no_framework_agent=no_framework_agent,
            framework=framework_for_prompt,
            objective=objective,
            max_minutes=max_minutes_for_prompt,
            macro_cycle=_initial_macro_cycle,
            cycle_directive=_initial_directive,
            phase=_initial_phase,
            transport=_orch_transport,
        ),
        "critic": args.critic_prompt or _load_critic_prompt(),
    }
    coordinator.system_prompt_overrides = prompts
    # Cache a pure rebuild closure so the macro-cycle boundary can re-focus the
    # orchestration prompt without reaching back into argparse. A user-supplied
    # --orch-prompt is never clobbered.
    import functools as _functools

    coordinator._orch_prompt_is_user_supplied = bool(args.orch_prompt)
    coordinator._rebuild_orch_prompt = _functools.partial(
        _build_orchestration_prompt,
        no_kernel=no_kernel,
        no_framework_agent=no_framework_agent,
        framework=framework_for_prompt,
        objective=objective,
        max_minutes=max_minutes_for_prompt,
        transport=_orch_transport,
    )
    # Build specialist executor only when research_lane capacity > 0 (0 degrades to LLM-direct grid).
    specialist_capacity = int(getattr(args, "research_lane_capacity", 1) or 0)
    specialist_executor: "Any" = None
    if specialist_capacity > 0:
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
    _snapshot_system_prompts(session_dir, prompts=prompts, orchestration_phase=_initial_phase)

    def _backend_kind(role: str) -> str:
        backend = backends.get(role)
        name = str(getattr(backend, "name", "") or "").strip().lower()
        if name == "claude":
            return "Claude"
        if name == "codex":
            return "Codex"
        if backend is None:
            return "DISABLED"
        return backend.__class__.__name__

    orchestration_str = (
        f"Claude({args.claude_model})"
        if _backend_kind("orchestration") == "Claude"
        else f"{_backend_kind('orchestration')}({args.codex_model})"
    )
    kernel_str = "DISABLED" if no_kernel else "programmatic"
    if critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        # Read off the backend rather than re-derived from the environment:
        # --critic-protocol can override that derivation, and the banner would
        # then report the model of a transport this run is not using.
        _review_protocol = str(getattr(backends.get("critic"), "protocol", "") or "openai")
        _review_model = args.claude_model if _review_protocol == "anthropic" else args.codex_model
        critic_str = (
            f"critic-agent(kb={critic_kb_mode}, protocol={_review_protocol}, "
            f"model={_review_model}, root={critic_agent_root})"
        )
    if robustness_choice == "mock":
        robustness_str = "mock"
    else:
        robustness_str = f"robustness-agent(root={robustness_agent_root})"
        if robustness_options:
            kvs = ",".join(f"{k}={v!r}" for k, v in sorted(robustness_options.items()))
            robustness_str += f"[{kvs}]"
    print(
        f"Backends        : "
        f"orchestration={orchestration_str}, "
        f"kernel={kernel_str}, "
        f"critic={critic_str}, "
        f"robustness={robustness_str}"
    )
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} (budget = {args.max_hours}h)")
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
        state = coordinator.shared_state
        with timed_teardown_step(state, "coordinator_stop"):
            await coordinator.stop()
        # Drop the single-optimizer session lock once the
        # coordinator has released its leases. The OS would drop it on process
        # exit anyway; this just frees it promptly for an intentional resume.
        with timed_teardown_step(state, "session_lock"):
            session_lock.release()
        # Crash-safe reports/final.json. Runs unconditionally and first so a
        # machine-readable summary always exists even when the CLOSE sequencer
        # never ran. Idempotent: a no-op when ReportExecutor already wrote it.
        try:
            from ..breakdown import write_minimal_final_json

            with timed_teardown_step(state, "final_json"):
                final_json = write_minimal_final_json(session_dir)
            print(f"Final summary     : {final_json}")
        except Exception:  # noqa: BLE001 — safety net must never mask stop_reason
            log.exception("crash-safe final.json write failed (non-fatal)")
        # End-of-session safety net: always materialize session_breakdown.json (best-effort; never mask stop_reason).
        # Skip when the CLOSE sequencer already wrote it (close_sequence_done is locked in CORE_STATE_FIELDS).
        sequencer_done = getattr(state, "close_sequence_done", False)
        if sequencer_done:
            print(
                "Session breakdown : (already written by CLOSE phase sequencer; skipping cli.finally safety-net write)"
            )
            # Re-run the Langfuse flush idempotently as a safety net.
            try:
                from hyperloom.orchestrator.trace.langfuse_emitter import (
                    flush_session,
                    record_session_breakdown,
                )

                with timed_teardown_step(state, "langfuse"):
                    flush_session(session_dir)
                    from ..breakdown import patch_breakdown_langfuse

                    patch_breakdown_langfuse(session_dir)
                    record_session_breakdown(session_dir)
            except Exception:  # noqa: BLE001
                log.debug("langfuse flush_session (post-sequencer) failed", exc_info=True)
        else:
            try:
                from ..breakdown import write_breakdown_json

                with timed_teardown_step(state, "session_breakdown"):
                    breakdown_path = write_breakdown_json(session_dir)
                print(f"Session breakdown : {breakdown_path}")
            except Exception:  # noqa: BLE001
                log.exception("session_breakdown finalize failed (non-fatal)")
            # Safety-net reports/final.md write (no-op when the sequencer's final.md already exists).
            try:
                from ..breakdown import write_minimal_final_report

                with timed_teardown_step(state, "final_md"):
                    final_md = write_minimal_final_report(session_dir)
                print(f"Final report      : {final_md}")
            except Exception:  # noqa: BLE001
                log.exception("emergency final report write failed (non-fatal)")
            # Live Langfuse push (opt-in, default off): reconcile + flush, then
            # splice the post-flush receipt into the session_breakdown.json
            # langfuse section. Runs before the artifact package so the bundled
            # SBD carries counts_final=true. No-op unless HYPERLOOM_LANGFUSE_ENABLE
            # + LANGFUSE_* are set; idempotent.
            try:
                from hyperloom.orchestrator.trace.langfuse_emitter import (
                    flush_session,
                    record_session_breakdown,
                )

                with timed_teardown_step(state, "langfuse"):
                    flush_session(session_dir)
                    from ..breakdown import patch_breakdown_langfuse

                    patch_breakdown_langfuse(session_dir)
                    record_session_breakdown(session_dir)
            except Exception:  # noqa: BLE001
                log.debug("langfuse flush_session failed (non-fatal)", exc_info=True)

        # Safety-net artifact package -> /workspace, for paths that leave
        # close_sequence_done False and never run the sequencer. Best-effort;
        # runs after the SBD/final.md + Langfuse flush so the freshest products
        # are bundled.
        try:
            from ..breakdown import package_session_artifacts

            with timed_teardown_step(state, "artifact_package"):
                pkg_path = package_session_artifacts(
                    session_dir,
                    session_id=str(getattr(state, "session_id", "") or ""),
                )
            if pkg_path is not None:
                print(f"Artifact package  : {pkg_path}")
        except Exception:  # noqa: BLE001
            log.exception("session artifact package failed (non-fatal)")
        try:
            state.save(session_dir)
        except Exception:  # noqa: BLE001
            log.exception("failed to persist teardown timings (non-fatal)")

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep is now a SWEEP-phase action auto-enqueued by the Coordinator, not a post-hook here.

    _print_final_summary(coordinator.shared_state, stop_reason, session_dir)
    return _exit_code_for_stop_reason(stop_reason)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and dispatch the requested subcommand.

    Configures logging from the ``-v`` count, resolves any ``--*-prompt`` flag
    that points at a file (reading its contents in place), and runs the
    ``optimize`` subcommand via :func:`asyncio.run`. Prints help and returns a
    non-zero code for unknown commands.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        int: The process exit code (``optimize`` result, or ``2`` for no/unknown
        command).
    """
    # Force line-buffering so output piped through a non-TTY sink flushes every
    # line immediately instead of block-buffering, which would otherwise freeze
    # the top-level log for the duration of a blocking Magpie subprocess.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # Absolutise before the parser defaults or any session path derive from it:
    # subprocesses run with their own cwd, so a relative value would diverge.
    user_data = os.environ.get(ENV_USER_DATA_PATH, "")
    if user_data and not Path(user_data).is_absolute():
        os.environ[ENV_USER_DATA_PATH] = str(Path(user_data).expanduser().absolute())

    parser = _build_parser()
    # Strict on purpose. The platform's prompt FLAGS block is authored by hand,
    # so a typo is likelier here than in generated argv, and the flags the
    # platform consumes on its own side are declared as inert no-ops rather than
    # waved through — tolerating unknown arguments wholesale would let
    # `--target-gai 5` run for hours on the default instead of failing now.
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
        return asyncio.run(_run_optimize(args))
    if args.command == "recover-session":
        return _run_recover_session(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
