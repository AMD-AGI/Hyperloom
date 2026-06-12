# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Configuration for the Robustness Agent.

Claw v2 cannot inject env vars, so config is auto-detected: session_dir,
robust-analyzer endpoint, and LLM endpoint (OPENAI_BASE_URL / SAFE_API_KEY).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

ROBUST_ANALYZER_CANDIDATES: list[str] = [
    "http://robust-analyzer.primus-robust.svc.cluster.local:8085",
    "http://robust-analyzer:8085",
]

# Primary data source: cluster DNS first, then a local dev port-forward.
ROBUSTNESS_SERVER_CANDIDATES: list[str] = [
    "http://robustness-server.robustness.svc.cluster.local:8000",
    "http://robustness-server.primus-safe.svc.cluster.local:8000",
    "http://robustness-server:8000",
    "http://localhost:8000",
]

SESSION_DIR_CANDIDATES: list[Path] = [
    Path("/workspace/session"),
    Path("/tmp/robustness-session"),
]


@dataclass
class Config:
    """Runtime configuration for the Robustness Agent.

    Holds every tunable knob the agent uses: auto-detected service
    endpoints, monitoring intervals, alert thresholds, LLM RCA settings,
    and the many M1/M1.5 reactor signal parameters. Most fields have
    sensible defaults; the discovered service URLs and LLM credentials
    are populated by :meth:`discover`.

    Attributes:
        session_dir (Path): Directory containing the session's storage
            (including ``conductor.db``).
        robust_analyzer_url (str): Auto-detected robust-analyzer
            endpoint; empty means local-only mode.
        robustness_server_url (str): Primary M1 data source endpoint;
            empty means skip the server and use only the local probe.
        llm_model (str): Model name used for LLM-driven root-cause
            analysis.
        llm_base_url (str): LLM API base URL discovered from the sandbox.
        llm_api_key (str): LLM API key discovered from the sandbox.
        llm_rca_enabled (Optional[bool]): Tri-state RCA activation flag;
            ``None`` auto-enables when credentials are present.
        metrics_window_s (int): Rolling window, in seconds, over which
            metrics-based signals are evaluated.

    Note:
        Many additional threshold, interval, and per-signal fields exist
        on this dataclass; see the inline comments grouped by signal
        family (A–L) for their meaning.
    """

    session_dir: Path = field(default_factory=lambda: Path("/tmp/robustness-session"))

    # Filled by auto-detection; empty means local-only mode.
    robust_analyzer_url: str = ""

    # Primary data source; empty means "skip server, only use local probe".
    robustness_server_url: str = ""

    @property
    def conductor_db_path(self) -> Path:
        """Filesystem path to the session's conductor SQLite database.

        Returns:
            Path: ``session_dir/storage/conductor.db``.
        """
        return self.session_dir / "storage" / "conductor.db"

    # -- monitoring intervals (seconds) --
    process_check_interval: float = 10.0
    gpu_check_interval: float = 15.0
    disk_check_interval: float = 60.0
    event_poll_interval: float = 5.0
    health_check_interval: float = 30.0

    # -- thresholds --
    gpu_vram_warn_pct: float = 90.0
    gpu_vram_crit_pct: float = 95.0
    gpu_temp_warn_c: float = 85.0
    gpu_util_drop_pct: float = 50.0
    disk_usage_warn_pct: float = 85.0
    disk_usage_crit_pct: float = 95.0
    agent_stall_timeout_s: float = 300.0
    benchmark_timeout_s: float = 600.0
    server_start_timeout_s: float = 480.0

    # -- LLM for RCA (auto-detected from Claw sandbox env) --
    llm_model: str = "claude-opus-4-7"
    llm_base_url: str = ""
    llm_api_key: str = ""
    rca_max_turns: int = 10

    # -- LLM RCA throttle / activation --
    # ``None`` = auto-enable when llm_base_url + llm_api_key are both set;
    # ``False`` = force-disable (ROBUSTNESS_LLM_RCA_DISABLED=1 also flips off).
    llm_rca_enabled: Optional[bool] = None
    llm_rca_severity_min: str = "high"  # one of low/medium/high
    llm_rca_cooldown_s: float = 60.0
    llm_rca_max_calls_per_tick: int = 1
    llm_rca_timeout_s: float = 8.0
    llm_rca_max_chars: int = 1500

    # -- reactor knobs --
    cooldown_ticks: int = 5
    metrics_window_s: int = 300
    server_request_timeout_s: float = 5.0
    source_fail_threshold: int = 3
    source_recheck_interval_s: float = 30.0
    standalone_tick_interval_s: float = 10.0

    # -- LocalProbe extras --
    health_probe_targets: list[str] = field(default_factory=list)
    health_probe_timeout_s: float = 1.5

    # -- multi-node knobs --
    # Required in multi-node runs: per-pod ps/HTTP/rocm-smi probes false-fire
    # local_server_unreachable / ray_head_dead on Ray workers without a server.
    disable_local_probe: bool = False
    # RobustnessServerSource fan-out so local_health sees per-pod GPU snapshots.
    enable_cluster_pod_metrics: bool = False
    pod_metrics_categories: tuple[str, ...] = ("gpu",)
    # Resolves the full multi-node RayJob pod set via the hierarchy endpoint.
    workload_uid: str = ""
    # Informational only (mirrors --nodes); policy driven by the flags above.
    nodes: int = 1

    # -- gpu_memory_leaked signal (2026-05) --
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
    # -- H1 / H2 budget signal extensions (2026-05-18) --
    # strategy_drift_pct (50%) = earliest gate: MEDIUM when half-burnt with no
    # gain. Absolute-time thresholds back-stop very long budgets.
    budget_strategy_drift_pct: float = 0.5
    budget_deadline_warning_minutes: float = 30.0
    budget_deadline_hard_cutoff_minutes: float = 5.0

    # -- same_payload_loop signal (B1, 2026-05-18) --
    # Consecutive identical-payload failures before firing.
    repeated_payload_streak_threshold: int = 3
    repeated_payload_lookback_events: int = 80

    # -- aiter_jit_regressed signal (A7, 2026-05-18) --
    aiter_jit_cold_so_count: int = 20
    aiter_jit_regression_ratio: float = 0.8
    aiter_jit_stale_build_threshold: int = 1
    aiter_jit_stale_build_persist_ticks: int = 5

    # -- gain_plateau / no_levers_found signals (B2 / B3, 2026-05-18) --
    progress_gain_window_ticks: int = 6
    progress_gain_epsilon_pct: float = 0.5
    progress_no_levers_min_minutes: float = 45.0
    progress_no_levers_min_ticks: int = 8

    # -- A3 / A4 disk / shm signals (2026-05-18) --
    disk_used_warn_pct: float = 85.0
    disk_used_crit_pct: float = 95.0
    shm_used_warn_pct: float = 75.0
    shm_used_crit_pct: float = 90.0

    # -- A5 fd_pressure signal (2026-05-18) --
    fd_warn_used_pct: float = 80.0
    fd_crit_used_pct: float = 95.0
    fd_probe_enabled: bool = True
    fd_probe_pid: int | None = None

    # -- A6 ray_head_dead signal (2026-05-18) --
    ray_probe_enabled: bool = True
    ray_probe_timeout_s: float = 5.0

    # -- B4 idempotency_replay signal (2026-05-18) --
    idempotency_replay_threshold: int = 2

    # -- G decision-audit signals (2026-05-18) --
    # Off skips the LocalProbe sub-probe (e.g. ``runs/`` on read-only storage).
    decision_audit_enabled: bool = True
    decision_audit_max_integrate: int = 20
    decision_audit_max_oob_attempts: int = 50
    # G2 noise-floor cliff; mirrors upstream executors' 1.0% keep threshold.
    decision_audit_min_keep_gain_pct: float = 1.0
    decision_audit_dispatch_bypass_epsilon_pct: float = 0.5

    # -- C preflight signals (2026-05-19) --
    # Off disables manifest + kernel_breakdown probes (C1/C2); C3 gated by JIT.
    preflight_enabled: bool = True
    # C1 — fire when projected HBM headroom < min_headroom_pct. 5% floor: the
    # projection over-estimates HBM by ~5%, so below it headroom is ~0.
    preflight_min_headroom_pct: float = 5.0
    preflight_activation_buf_gib: float = 8.0
    # C2 — Amdahl kernel ceiling; 1.5x single-kernel speedup = 2026-05 GEAK
    # average, 5% min_e2e_ceiling_pct = SKILL noise-floor convention.
    preflight_amdahl_single_kernel_speedup: float = 1.5
    preflight_amdahl_min_e2e_ceiling_pct: float = 5.0
    # C3 — cold-start vs budget. ``cold_start_minutes=None`` reads
    # ``$INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC`` (default 3600s).
    preflight_cold_start_so_count: int = 20
    preflight_cold_start_minutes: float | None = None

    # -- D2 multi-source server logs (2026-05-19) --
    # Colon-separated extra log globs (env-supplied) added to the
    # ``runs/*/*/server.log`` defaults; "" disables extras.
    server_log_extra_globs: str = ""
    # Max number of extra log files scanned per tick.
    server_log_max_extra: int = 5

    # -- E critic-health signals (2026-05-19) --
    critic_health_enabled: bool = True
    critic_health_max_judge_bundles: int = 20
    critic_health_min_outage_judges: int = 3
    critic_health_min_unavailable_verdicts: int = 3
    critic_health_max_workdir_count: int = 100

    # -- F kernel-pipeline signals (2026-05-19) --
    kernel_pipeline_pending_count_threshold: int = 1
    kernel_pipeline_min_pending_ticks: int = 3
    kernel_pipeline_min_geak_sigterm_attempts: int = 2
    kernel_pipeline_min_cursor_401_hits: int = 3
    kernel_pipeline_min_kernels_with_no_progress: int = 3
    # Auto-append inference server health URL to ``health_probe_targets`` so
    # sglang SIGSTOP fires a symptom. 127.0.0.1:8888 = Magpie wrapper default.
    auto_probe_inference_server: bool = True
    inference_server_health_url: str = "http://127.0.0.1:8888/health"

    # -- I state-integrity signals (2026-05-19) --
    state_integrity_enabled: bool = True
    state_wal_bytes_warn_threshold: int = 1 * 1024 * 1024 * 1024     # 1 GiB
    state_wal_bytes_critical_threshold: int = 4 * 1024 * 1024 * 1024  # 4 GiB
    state_stale_lease_min_age_s: float = 60.0
    state_inbox_bloat_warn_bytes: int = 100 * 1024 * 1024     # 100 MiB
    state_inbox_bloat_critical_bytes: int = 500 * 1024 * 1024  # 500 MiB

    # -- J external-deps signals (2026-05-19) --
    # Disable for hosts that audit gateway / mounts externally (e.g. Primus-SaFE).
    external_deps_enabled: bool = True
    external_mount_stat_timeout_s: float = 5.0
    external_gateway_probe_url: str = ""  # empty → derive from OPENAI_BASE_URL
    external_mount_latency_warn_ms: float = 5000.0
    external_mount_latency_critical_ms: float = 15000.0

    # -- L1 + L2 postmortem finalizer (2026-05-19) --
    # False disables the L1 flashpoint + L2 decision_trace writers.
    finalize_enabled: bool = True
    finalize_reports_subdir: str = "reports"
    finalize_max_findings_in_report: int = 20
    finalize_max_tasks_per_action: int = 50

    # -- cross-tick state persistence --
    # Subprocess-per-tick transport needs disk-backed state for any
    # consecutive-tick rule (gpu leak, gain_plateau, cooldowns). ``False`` =
    # in-memory only (unit tests / single-process drivers).
    state_store_enabled: bool = True

    # -- ring buffer (local mode only) --
    local_metrics_history_s: int = 3600
    local_metrics_sample_interval: int = 5

    # -- server process patterns --
    # Mirrors ``local_probe._DEFAULT_PROCESS_PATTERNS`` so the
    # gpu_memory_leaked "no live owner" check matches every legitimate VRAM
    # holder; vLLM v1 / Ray / aiter JIT entries are critical or EngineCore-
    # children get mis-classified as "not a server".
    server_process_patterns: list[str] = field(default_factory=lambda: [
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
    ])
    benchmark_process_patterns: list[str] = field(default_factory=lambda: [
        "benchmark_serving",
    ])

    @classmethod
    async def discover(cls) -> "Config":
        """Auto-detect all configuration from the runtime environment.

        Discovers the session directory, probes the robust-analyzer and
        robustness-server endpoints, and reads LLM credentials from the
        sandbox environment.

        Returns:
            Config: A new instance populated with the discovered values.
        """
        session_dir = _discover_session_dir()
        analyzer_url = await _probe_robust_analyzer()
        server_url = await _probe_robustness_server()
        llm_base_url, llm_api_key = _discover_llm_credentials()
        workload_uid = _discover_workload_uid()
        disable_local_probe = _env_bool("ROBUSTNESS_DISABLE_LOCAL_PROBE", False)
        enable_cluster_pod_metrics = _env_bool(
            "ROBUSTNESS_ENABLE_CLUSTER_POD_METRICS", False,
        )
        nodes = _env_int("ROBUSTNESS_NODES", 1)

        config = cls(
            session_dir=session_dir,
            robust_analyzer_url=analyzer_url,
            robustness_server_url=server_url,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            workload_uid=workload_uid,
            disable_local_probe=disable_local_probe,
            enable_cluster_pod_metrics=enable_cluster_pod_metrics,
            nodes=nodes,
        )

        log.info(
            "Config discovered: session_dir=%s server=%s analyzer=%s llm=%s "
            "nodes=%d workload_uid=%s disable_local_probe=%s "
            "enable_cluster_pod_metrics=%s",
            config.session_dir,
            config.robustness_server_url or "(local-only)",
            config.robust_analyzer_url or "(local mode)",
            "(configured)" if config.llm_base_url else "(not available)",
            config.nodes,
            config.workload_uid or "(unset)",
            config.disable_local_probe,
            config.enable_cluster_pod_metrics,
        )
        return config


def _discover_session_dir() -> Path:
    """Find session directory by scanning well-known paths.

    Checks the ``SESSION_DIR`` environment variable, then the known
    candidate paths and the current working directory for a
    ``storage/conductor.db`` marker.

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
        db = candidate / "storage" / "conductor.db"
        if db.exists():
            log.info("Session dir discovered at: %s", candidate)
            return candidate

    cwd = Path.cwd()
    db = cwd / "storage" / "conductor.db"
    if db.exists():
        log.info("Session dir is cwd: %s", cwd)
        return cwd

    fallback = SESSION_DIR_CANDIDATES[-1]
    log.warning("No session dir found, using fallback: %s", fallback)
    return fallback


async def _probe_robust_analyzer() -> str:
    """Try known robust-analyzer endpoints, return first reachable one.

    Returns:
        str: The first candidate URL whose ``/health`` endpoint returns
        200, or an empty string if none are reachable.
    """
    for url in ROBUST_ANALYZER_CANDIDATES:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    log.info("Robust-analyzer reachable at %s", url)
                    return url
        except Exception:
            continue

    log.info("Robust-analyzer not reachable, will use local provider")
    return ""


async def _probe_robustness_server() -> str:
    """Probe known robustness-server endpoints + ROBUSTNESS_SERVER_URL env.

    Returns:
        str: The first candidate URL whose ``/healthz`` endpoint returns
        200, or an empty string if none are reachable.
    """
    candidates: list[str] = []
    env_url = os.environ.get("ROBUSTNESS_SERVER_URL", "").strip()
    if env_url:
        candidates.append(env_url.rstrip("/"))
    candidates.extend(ROBUSTNESS_SERVER_CANDIDATES)

    for url in candidates:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                resp = await client.get(f"{url}/healthz")
                if resp.status_code == 200:
                    log.info("Robustness-server reachable at %s", url)
                    return url
        except Exception:
            continue

    log.info("Robustness-server not reachable, will use local-only fallback")
    return ""


def _discover_llm_credentials() -> tuple[str, str]:
    """Pick up LLM credentials already in the Claw sandbox environment.

    Returns:
        tuple[str, str]: A ``(base_url, api_key)`` pair read from the
        sandbox environment variables; either element may be empty if
        unset.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = (
        os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("LLM_API_KEY", "")
    )
    return base_url, api_key


_WORKLOAD_UID_ENV_KEYS: tuple[str, ...] = (
    "ROBUSTNESS_WORKLOAD_UID",
    "CLAW_WORKLOAD_UID",
    "WORKLOAD_UID",
    "KUBE_WORKLOAD_UID",
    "RAY_JOB_ID",
)


def _discover_workload_uid() -> str:
    """Resolve the multi-node workload uid (first non-empty env key above).

    Lets a RayJob sandbox opt into hierarchy-based pod discovery; single-node
    runs leave every key unset and fall back to ``list_session_pods``.
    """
    for key in _WORKLOAD_UID_ENV_KEYS:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean configuration value from the environment.

    Args:
        name: Name of the environment variable to read.
        default: Value to return when the variable is unset.

    Returns:
        ``True`` when the variable is set to one of ``1``, ``true``, ``yes``,
        or ``on`` (case-insensitive); ``False`` for any other set value; and
        ``default`` when the variable is unset.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer configuration value from the environment.

    Args:
        name: Name of the environment variable to read.
        default: Value to return when the variable is unset, empty, or not a
            valid integer.

    Returns:
        The parsed integer, or ``default`` when the variable is missing or
        cannot be parsed.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
