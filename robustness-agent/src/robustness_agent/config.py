"""Configuration for the Robustness Agent.

No custom environment variables — everything is auto-detected or uses
hardcoded defaults.  Claw v2 brain-hands mode does not allow injecting
extra env vars into the sandbox, so the agent must discover its runtime
context on its own.

Discovery strategy:
  1. session_dir: scan well-known paths for conductor.db
  2. robust-analyzer: probe known service endpoints, fallback to local
  3. LLM endpoint: reuse OPENAI_BASE_URL / SAFE_API_KEY already present
     in the Claw sandbox (set by Claw brain, not by us)
"""

from __future__ import annotations

import asyncio
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

SESSION_DIR_CANDIDATES: list[Path] = [
    Path("/workspace/session"),
    Path("/tmp/robustness-session"),
]


@dataclass
class Config:
    session_dir: Path = field(default_factory=lambda: Path("/tmp/robustness-session"))

    # Filled by auto-detection; empty means local-only mode.
    robust_analyzer_url: str = ""

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

    # -- LLM for RCA (auto-detected from Claw sandbox env) --
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
    async def discover(cls) -> "Config":
        """Auto-detect all configuration from the runtime environment."""
        session_dir = _discover_session_dir()
        analyzer_url = await _probe_robust_analyzer()
        llm_base_url, llm_api_key = _discover_llm_credentials()

        config = cls(
            session_dir=session_dir,
            robust_analyzer_url=analyzer_url,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )

        log.info(
            "Config discovered: session_dir=%s analyzer=%s llm=%s",
            config.session_dir,
            config.robust_analyzer_url or "(local mode)",
            "(configured)" if config.llm_base_url else "(not available)",
        )
        return config


def _discover_session_dir() -> Path:
    """Find session directory by scanning well-known paths."""
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
    """Try known robust-analyzer endpoints, return first reachable one."""
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


def _discover_llm_credentials() -> tuple[str, str]:
    """Pick up LLM credentials already in the Claw sandbox environment."""
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = (
        os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("LLM_API_KEY", "")
    )
    return base_url, api_key
