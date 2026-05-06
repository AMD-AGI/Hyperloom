"""Configuration for the Robustness Agent.

All tunables are loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # -- deployment mode --
    # If set, use Primus-Robust-Internal analyzer API for GPU/RDMA/fault metrics.
    # If empty, fall back to local shell commands.
    robust_analyzer_url: str = ""

    # -- conductor integration --
    session_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "SESSION_DIR", "/tmp/robustness-session",
    )))

    @property
    def conductor_db_path(self) -> Path:
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

    # -- LLM for RCA --
    llm_model: str = "claude-opus-4-7"
    llm_base_url: str = ""
    llm_api_key: str = ""
    rca_max_turns: int = 10

    # -- ring buffer (local mode only) --
    local_metrics_history_s: int = 3600
    local_metrics_sample_interval: int = 5

    # -- server process patterns --
    server_process_patterns: list[str] = field(default_factory=lambda: [
        "sglang.srt", "vllm.entrypoints", "vllm serve",
    ])
    benchmark_process_patterns: list[str] = field(default_factory=lambda: [
        "benchmark_serving",
    ])

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            robust_analyzer_url=os.environ.get("ROBUST_ANALYZER_URL", ""),
            session_dir=Path(os.environ.get("SESSION_DIR", "/tmp/robustness-session")),
            process_check_interval=float(os.environ.get("PROCESS_CHECK_INTERVAL", "10")),
            gpu_check_interval=float(os.environ.get("GPU_CHECK_INTERVAL", "15")),
            disk_check_interval=float(os.environ.get("DISK_CHECK_INTERVAL", "60")),
            event_poll_interval=float(os.environ.get("EVENT_POLL_INTERVAL", "5")),
            health_check_interval=float(os.environ.get("HEALTH_CHECK_INTERVAL", "30")),
            gpu_vram_warn_pct=float(os.environ.get("GPU_VRAM_WARN_PCT", "90")),
            gpu_vram_crit_pct=float(os.environ.get("GPU_VRAM_CRIT_PCT", "95")),
            gpu_temp_warn_c=float(os.environ.get("GPU_TEMP_WARN_C", "85")),
            disk_usage_warn_pct=float(os.environ.get("DISK_USAGE_WARN_PCT", "85")),
            disk_usage_crit_pct=float(os.environ.get("DISK_USAGE_CRIT_PCT", "95")),
            agent_stall_timeout_s=float(os.environ.get("AGENT_STALL_TIMEOUT_S", "300")),
            benchmark_timeout_s=float(os.environ.get("BENCHMARK_TIMEOUT_S", "600")),
            server_start_timeout_s=float(os.environ.get("SERVER_START_TIMEOUT_S", "480")),
            llm_model=os.environ.get("LLM_MODEL", "claude-opus-4-7"),
            llm_base_url=os.environ.get("OPENAI_BASE_URL", ""),
            llm_api_key=os.environ.get("SAFE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            local_metrics_history_s=int(os.environ.get("LOCAL_METRICS_HISTORY_S", "3600")),
            local_metrics_sample_interval=int(os.environ.get("LOCAL_METRICS_SAMPLE_INTERVAL", "5")),
        )
