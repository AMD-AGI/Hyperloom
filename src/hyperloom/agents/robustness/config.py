# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration for the Robustness Agent.

Env-first with auto-detection fallbacks: session_dir (``SESSION_DIR``) and the
LLM endpoint / credentials (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``, plus
Anthropic and DeepSeek variants) fall back to probing well-known paths and
endpoints when unset. ``ROBUSTNESS_DISABLE_LOCAL_PROBE`` and
``ROBUSTNESS_NODES`` are env-only with fixed defaults.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hyperloom.common.env import env_bool, env_int
from hyperloom.common.llm_config import (
    CLAUDE_OAUTH_TOKEN_ENV,
    anthropic_synthesizable_key,
    deepseek_compat_env,
)

log = logging.getLogger(__name__)

SESSION_DIR_CANDIDATES: list[Path] = [
    Path("/workspace/session"),
    Path(tempfile.gettempdir()) / "robustness-session",
]


@dataclass
class Config:
    """Runtime configuration for the Robustness Agent.

    Holds every tunable knob the agent uses: auto-detected service
    endpoints, monitoring intervals, alert thresholds, LLM RCA settings,
    and the reactor signal parameters. Most fields have sensible defaults;
    the discovered service URLs and LLM credentials are populated by
    :meth:`discover`.

    Attributes:
        session_dir (Path): Directory containing the session's storage
            (including ``coordinator.db``).
        llm_model (str): Model name used for LLM-driven root-cause
            analysis.
        llm_base_url (str): LLM API base URL discovered from the sandbox. Empty
            for a Claude subscription host, which has no endpoint to resolve.
        llm_api_key (str): LLM API key discovered from the sandbox. Empty when
            the credential is a subscription token, which the Claude CLI spends
            without ever handing it to this process.
        llm_rca_enabled (Optional[bool]): Tri-state RCA activation flag;
            ``None`` auto-enables when credentials are present.

    Note:
        Many additional threshold, interval, and per-signal fields exist
        on this dataclass; see the inline comments grouped by signal
        family for their meaning.
    """

    session_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "robustness-session")

    @property
    def coordinator_db_path(self) -> Path:
        """Filesystem path to the session's Coordinator SQLite database.

        Returns:
            Path: ``session_dir/storage/coordinator.db``.
        """
        return self.session_dir / "storage" / "coordinator.db"

    # -- thresholds --
    gpu_temp_warn_c: float = 85.0
    agent_stall_timeout_s: float = 300.0

    # -- LLM for RCA (auto-detected from Claw sandbox env) --
    llm_model: str = "claude-opus-5"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_provider: str = "openai"

    # -- LLM RCA throttle / activation --
    # ``None`` = auto-enable when the discovered provider can authenticate a
    # call, which for the Anthropic side is a usable transport rather than a
    # base_url + api_key pair; ``False`` = force-disable.
    llm_rca_enabled: Optional[bool] = None
    llm_rca_severity_min: str = "high"  # one of low/medium/high
    llm_rca_cooldown_s: float = 60.0
    llm_rca_max_calls_per_tick: int = 1
    llm_rca_timeout_s: float = 8.0
    llm_rca_max_chars: int = 1500

    # -- reactor knobs --
    cooldown_ticks: int = 5
    source_fail_threshold: int = 3
    source_recheck_interval_s: float = 30.0
    standalone_tick_interval_s: float = 10.0

    # -- LocalProbe extras --
    health_probe_targets: list[str] = field(default_factory=list)
    health_probe_timeout_s: float = 1.5

    # -- multi-node knobs --
    # Required in multi-node runs: per-pod ps/HTTP/rocm-smi probes false-fire
    # local_server_unreachable / ray_head_dead on Ray workers.
    disable_local_probe: bool = False
    # Informational only (mirrors --nodes); policy driven by the flags above.
    nodes: int = 1

    # -- gpu_memory_leaked signal --
    # GPU "full" when util_mem_pct > threshold OR free MiB < threshold; fires
    # only after holding for min_consecutive_ticks (anti-flap vs cold-start).
    gpu_leak_util_mem_pct_threshold: float = 99.0
    gpu_leak_free_mb_threshold: float = 500.0
    gpu_leak_min_consecutive_ticks: int = 2

    # -- deadline_imminent / budget_burn_no_gain signals --
    # warn/imminent_pct = budget consumed before medium/high rungs fire;
    # min_minutes avoids short smoke tests; productive_gain_pct is the
    # validated-gain cliff above which the run finishes naturally (below it,
    # wind down via delegate(report)).
    budget_warn_pct: float = 0.70
    budget_imminent_pct: float = 0.85
    budget_min_minutes: float = 30.0
    budget_productive_gain_pct: float = 0.5
    # -- budget signal extensions --
    # strategy_drift_pct = earliest gate: MEDIUM when half-burnt with no gain.
    # Absolute-time thresholds back-stop very long budgets.
    budget_strategy_drift_pct: float = 0.5
    budget_deadline_warning_minutes: float = 30.0
    budget_deadline_hard_cutoff_minutes: float = 5.0

    # -- same_payload_loop signal --
    # Consecutive identical-payload failures before firing.
    repeated_payload_streak_threshold: int = 3
    repeated_payload_lookback_events: int = 80

    # -- aiter_jit_regressed signal --
    aiter_jit_cold_so_count: int = 20
    aiter_jit_regression_ratio: float = 0.8
    aiter_jit_stale_build_threshold: int = 1
    aiter_jit_stale_build_persist_ticks: int = 5

    # -- gain_plateau / no_levers_found signals --
    progress_gain_window_ticks: int = 6
    progress_gain_epsilon_pct: float = 0.5
    progress_no_levers_min_minutes: float = 45.0
    progress_no_levers_min_ticks: int = 8

    # -- disk / shm signals --
    disk_used_warn_pct: float = 85.0
    disk_used_crit_pct: float = 95.0
    shm_used_warn_pct: float = 75.0
    shm_used_crit_pct: float = 90.0

    # -- fd_pressure signal --
    fd_warn_used_pct: float = 80.0
    fd_crit_used_pct: float = 95.0
    fd_probe_enabled: bool = True
    fd_probe_pid: int | None = None

    # -- ray_head_dead signal --
    ray_probe_enabled: bool = True
    ray_probe_timeout_s: float = 5.0

    # -- idempotency_replay signal --
    idempotency_replay_threshold: int = 2

    # -- decision-audit signals --
    # Off skips the LocalProbe sub-probe (e.g. ``runs/`` on read-only storage).
    decision_audit_enabled: bool = True
    decision_audit_max_integrate: int = 20
    decision_audit_max_oob_attempts: int = 50
    # Noise-floor cliff; mirrors executors' 1.0% keep threshold.
    decision_audit_min_keep_gain_pct: float = 1.0
    decision_audit_dispatch_bypass_epsilon_pct: float = 0.5

    # -- preflight signals --
    # Off disables manifest + kernel_breakdown probes (C1/C2); C3 gated by JIT.
    preflight_enabled: bool = True
    # Fire when projected HBM headroom < min_headroom_pct. 5% floor: the
    # projection over-estimates HBM by ~5%, so below it headroom is ~0.
    preflight_min_headroom_pct: float = 5.0
    preflight_activation_buf_gib: float = 8.0
    # Amdahl kernel ceiling; 1.5x single-kernel speedup, 5% noise-floor.
    preflight_amdahl_single_kernel_speedup: float = 1.5
    preflight_amdahl_min_e2e_ceiling_pct: float = 5.0
    # Cold-start vs budget. ``cold_start_minutes=None`` reads
    # ``$INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC`` (default 3600s).
    preflight_cold_start_so_count: int = 20
    preflight_cold_start_minutes: float | None = None

    # -- multi-source server logs --
    # Colon-separated extra log globs (env-supplied) added to the
    # ``runs/*/*/server.log`` defaults; "" disables extras.
    server_log_extra_globs: str = ""
    # Max number of extra log files scanned per tick.
    server_log_max_extra: int = 5

    # -- critic-health signals --
    critic_health_enabled: bool = True
    critic_health_max_judge_bundles: int = 20
    critic_health_min_outage_judges: int = 3
    critic_health_min_unavailable_verdicts: int = 3
    critic_health_max_workdir_count: int = 100

    # -- kernel-pipeline signals --
    kernel_pipeline_pending_count_threshold: int = 1
    kernel_pipeline_min_pending_ticks: int = 3
    kernel_pipeline_min_geak_sigterm_attempts: int = 2
    kernel_pipeline_min_kernels_with_no_progress: int = 3
    # Auto-append inference server health URL to ``health_probe_targets`` so
    # sglang SIGSTOP fires a symptom. 127.0.0.1:8888 = Magpie wrapper default.
    auto_probe_inference_server: bool = True
    inference_server_health_url: str = "http://127.0.0.1:8888/health"

    # -- state-integrity signals --
    state_integrity_enabled: bool = True
    state_wal_bytes_warn_threshold: int = 1 * 1024 * 1024 * 1024  # 1 GiB
    state_wal_bytes_critical_threshold: int = 4 * 1024 * 1024 * 1024  # 4 GiB
    state_stale_lease_min_age_s: float = 60.0
    state_inbox_bloat_warn_bytes: int = 100 * 1024 * 1024  # 100 MiB
    state_inbox_bloat_critical_bytes: int = 500 * 1024 * 1024  # 500 MiB

    # -- external-deps signals --
    # Disable for hosts that audit gateway / mounts externally.
    external_deps_enabled: bool = True
    external_mount_stat_timeout_s: float = 5.0
    external_gateway_probe_url: str = ""  # empty → derive from OPENAI_BASE_URL
    external_mount_latency_warn_ms: float = 5000.0
    external_mount_latency_critical_ms: float = 15000.0

    # -- postmortem finalizer --
    # False disables the flashpoint + decision_trace writers.
    finalize_enabled: bool = True
    finalize_reports_subdir: str = "reports"
    finalize_max_findings_in_report: int = 20
    finalize_max_tasks_per_action: int = 50

    # -- phase budget / conversation progress signals --
    phase_budget_warn_used_pct: float = 90.0
    conversation_progress_enabled: bool = True

    # -- cross-tick state persistence --
    # Subprocess-per-tick transport needs disk-backed state for any
    # consecutive-tick rule (gpu leak, gain_plateau, cooldowns). ``False`` =
    # in-memory only (unit tests / single-process drivers).
    state_store_enabled: bool = True

    # -- server process patterns --
    # Mirrors ``local_probe._DEFAULT_PROCESS_PATTERNS`` so the
    # gpu_memory_leaked "no live owner" check matches every legitimate VRAM
    # holder.
    server_process_patterns: list[str] = field(
        default_factory=lambda: [
            # SGLang
            "sglang.srt",
            "sglang.launch_server",
            # vLLM
            "vllm.entrypoints",
            "vllm serve",
            "vllm.v1.engine.core",
            "vllm.engine.async_llm_engine",
            "EngineCore",
            # Magpie / InferenceX
            "Magpie",
            "inferencex",
            # Ray + JIT compilation
            "ray::IDLE",
            "raylet",
            "hipcc",
        ]
    )
    benchmark_process_patterns: list[str] = field(
        default_factory=lambda: [
            "benchmark_serving",
        ]
    )

    @classmethod
    def discover(cls) -> "Config":
        """Auto-detect all configuration from the runtime environment.

        Scans for the session directory and reads the LLM credentials and
        the env-only knobs from the sandbox environment.

        Returns:
            Config: A new instance populated with the discovered values.
        """
        session_dir = _discover_session_dir()
        llm_base_url, llm_api_key, llm_provider = _discover_llm_credentials()
        disable_local_probe = env_bool("ROBUSTNESS_DISABLE_LOCAL_PROBE", False)
        nodes = env_int("ROBUSTNESS_NODES", 1)

        config = cls(
            session_dir=session_dir,
            llm_model=_discover_llm_model(llm_provider),
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            disable_local_probe=disable_local_probe,
            nodes=nodes,
        )

        log.info(
            "Config discovered: session_dir=%s llm=%s nodes=%d disable_local_probe=%s",
            config.session_dir,
            # A subscription-token host resolves no base_url at all, so the URL
            # alone would report "(not available)" for an RCA engine that is
            # about to start issuing calls.
            "(configured)" if (config.llm_base_url or config.llm_provider == "anthropic") else "(not available)",
            config.nodes,
            config.disable_local_probe,
        )
        return config


def _discover_session_dir() -> Path:
    """Find session directory by scanning well-known paths.

    Checks the ``SESSION_DIR`` environment variable, then the known
    candidate paths and the current working directory for a
    ``storage/coordinator.db`` marker.

    Returns:
        Path: The discovered session directory, or the last candidate
        as a fallback when none is found.
    """
    if "SESSION_DIR" in os.environ:
        p = Path(os.environ["SESSION_DIR"])
        if p.exists():
            log.info("Session dir from SESSION_DIR env: %s", p)
            return p

    for candidate in SESSION_DIR_CANDIDATES:
        db = candidate / "storage" / "coordinator.db"
        if db.exists():
            log.info("Session dir discovered at: %s", candidate)
            return candidate

    cwd = Path.cwd()
    db = cwd / "storage" / "coordinator.db"
    if db.exists():
        log.info("Session dir is cwd: %s", cwd)
        return cwd

    fallback = SESSION_DIR_CANDIDATES[-1]
    log.warning("No session dir found, using fallback: %s", fallback)
    return fallback


def _provider_env() -> dict[str, str]:
    """Return the process environment with retired provider variables normalized.

    The robustness agent also runs standalone, outside the CLI preflight that
    normally performs this rewrite, so a sandbox still carrying ``DEEPSEEK_*``
    would otherwise resolve no credentials and silently degrade RCA to a no-op.
    """
    env = dict(os.environ)
    env.update(deepseek_compat_env(env))
    return env


def _discover_llm_credentials() -> tuple[str, str, str]:
    """Pick up LLM credentials already in the Claw sandbox environment.

    Returns:
        tuple[str, str, str]: A ``(base_url, api_key, provider)`` tuple read
        from sandbox environment variables; URL/key may be empty if unset.
    """
    env = _provider_env()
    openai_base = env.get("OPENAI_BASE_URL", "").strip()
    openai_key = env.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return openai_base or "https://api.openai.com/v1", openai_key, "openai"
    gateway_key = env.get("LLM_API_KEY", "").strip() or env.get("LLM_GATEWAY_KEY", "").strip()
    if gateway_key and openai_base:
        return openai_base, gateway_key, "openai"

    # The synthesizable subset, which is exactly the set that may be handed on
    # as an api_key: a subscription token is spent by the CLI and never travels
    # as a key, so it is excluded here by construction rather than by omission.
    anthropic_key = anthropic_synthesizable_key(env)
    if anthropic_key:
        return (
            env.get("ANTHROPIC_BASE_URL", "").strip() or "https://api.anthropic.com",
            anthropic_key,
            "anthropic",
        )
    # A Claude Max/Pro subscription token is resolved by the CLI itself, so it is
    # deliberately not returned as an api_key; the provider alone selects it.
    if env.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip():
        return env.get("ANTHROPIC_BASE_URL", "").strip(), "", "anthropic"

    return "", "", "openai"


def _discover_llm_model(provider: str) -> str:
    """Resolve the RCA model for the discovered provider."""
    env = _provider_env()
    explicit = env.get("ROBUSTNESS_LLM_MODEL", "").strip() or env.get("LLM_MODEL", "").strip()
    if explicit:
        return explicit
    if provider == "openai":
        return env.get("OPENAI_MODEL", "").strip() or env.get("CODEX_MODEL", "").strip() or "gpt-5.6-sol"
    return env.get("ANTHROPIC_MODEL", "").strip() or env.get("CLAUDE_MODEL", "").strip() or "claude-opus-5"


