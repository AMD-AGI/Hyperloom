# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Central configuration for kernelforge."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from kernelforge.knowledge.experience_store import KnowledgeConfig
from kernelforge.resources import default_project_root, resource_path

log = logging.getLogger(__name__)


@cache
def _warn_removed_max_turns_env() -> None:
    """Warn once when the removed max-turns environment variable is present."""
    log.warning(
        "KERNEL_AGENTS_MAX_TURNS is no longer supported and will be "
        "ignored; forge-loop derives its turn cap from --max-hours"
    )


def _env_bool(name: str, default: bool) -> bool:
    """Parse one conventional boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_json_object(name: str) -> dict:
    """Parse one optional JSON object environment variable."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


@dataclass
class Config:
    """Runtime configuration loaded from environment + optional overrides."""

    # GPU environment
    # ROCm compilation target.
    gpu_target: str = "gfx942"
    # Hardware model used in KB identities.
    gpu_type: str = "mi355x"
    # System owning the candidate stream this run files under; a producer has its
    # own index in the KB identity scheme. Empty means the forge-loop's own.
    producer: str = ""

    # Workspace where kernel source trees live
    workspace: str = ""

    # Generic local Agent provider settings.
    agent_backend: str = "auto"
    agent_model: str = ""
    agent_cli: str = ""
    agent_timeout_sec: int = 1800
    agent_reasoning_effort: str = "high"
    agent_sandbox_mode: str = "bypass"
    agent_precheck: bool = True
    agent_fallback_provider: str = "claude"
    agent_options: dict = field(default_factory=dict)
    # Provider conversation-turn ceiling. Kept HIGH and used only as a runaway
    # backstop: the intended per-session stop is the in-session gate's block
    # budget (max_blocks), which ends the session on a clean, resumable path.
    # Claude enforces this in the SDK and preserves the resume handle when the
    # cap raises; providers without a native turn cap rely on their timeout and
    # the same block budget.
    max_turns: int = 500

    # Paths (derived)
    project_root: Path = field(default_factory=default_project_root)
    experiments_dir: Path = field(default=None)
    # There is no `knowledge_dir` here any more. It used to resolve the packaged
    # `data/knowledge_base` tree, which no caller ever read; the tree is gone and
    # the field went with it. Knowledge the loop *produces* goes to
    # `resources.writable_knowledge_root()`, which is a different directory.
    # Curated per-backend knowledge tree injected into the forge-loop system
    # prompt as an on-demand index (hardware / common_methodology / flydsl).
    local_knowledge_dir: Path = field(default=None)

    # Kernel-specific benchmark harness injected into the kernel backend prompt.
    bench_setup: str = ""

    # Bounded scratch measurement for the read-only planning specialists (see
    # orchestrator.specialists.SpecialistProbeConfig). On by default: a
    # specialist that can only argue about a dispatch constant is the failure
    # this answers, and the probe's own budgets are what make it safe. The two
    # budgets are the analysis PHASE's, shared by every specialist of the round.
    specialist_probe: bool = True
    specialist_probe_max: int = 6
    specialist_probe_budget_sec: float = 600.0
    # Where the round scratch trees are created. Empty derives it from
    # experiments_dir; it must be absolute -- a relative value would resolve
    # against whatever the process CWD happens to be -- and it must lie outside
    # the canonical workspace, which is the one place the probe refuses to run.
    specialist_probe_scratch_root: str = ""

    # Experience storage. gbrain_url/gbrain_token remain compatibility fields for
    # the broader remote knowledge index and are populated only in remote mode.
    gbrain_url: str = field(default="")
    gbrain_token: str = field(default="")
    knowledge_config: KnowledgeConfig | None = field(default=None)

    # Experimental / off by default: inject framework/mori/ into the forge-loop
    # knowledge block alongside framework/aiter/. Not wired to any CLI flag yet
    # (ablation-only knob) — set via KERNELFORGE_INCLUDE_MORI_KB=1.
    # None means "unset, defer to the env var" -- using a plain bool here
    # (default False) made an explicit `Config(include_mori_kb=False)` and
    # "not specified" indistinguishable, so __post_init__ would silently
    # overwrite an explicit False with whatever the env var said.
    include_mori_kb: bool | None = field(default=None)
    # Appended to preserve the positional order of the pre-existing Config
    # contract. None keeps the provider default; empty/none/off disables it.
    agent_fallback_model: str | None = None

    def __post_init__(self):
        """Derive paths and validate provider-specific runtime settings."""
        from kernelforge.agent_backends.registry import get_agent_provider

        self.project_root = Path(self.project_root)
        self.agent_backend = (self.agent_backend or "auto").strip().lower()
        if self.agent_backend != "auto":
            get_agent_provider(self.agent_backend)
        self.agent_reasoning_effort = (self.agent_reasoning_effort or "high").strip()
        self.agent_sandbox_mode = (self.agent_sandbox_mode or "bypass").strip().lower()
        self.agent_fallback_provider = (self.agent_fallback_provider or "").strip().lower()
        if self.agent_fallback_provider == "none":
            self.agent_fallback_provider = ""
        if self.agent_fallback_provider:
            get_agent_provider(self.agent_fallback_provider)
        if self.agent_fallback_model is not None:
            fallback_model = str(self.agent_fallback_model).strip()
            self.agent_fallback_model = "" if fallback_model.lower() in {"", "none", "off"} else fallback_model
        if self.agent_timeout_sec <= 0:
            raise ValueError("agent_timeout_sec must be greater than zero")
        if self.specialist_probe_max <= 0:
            raise ValueError("specialist_probe_max must be greater than zero")
        if self.specialist_probe_budget_sec <= 0:
            raise ValueError("specialist_probe_budget_sec must be greater than zero")
        if (
            self.specialist_probe_scratch_root
            and not Path(self.specialist_probe_scratch_root).expanduser().is_absolute()
        ):
            raise ValueError(
                "specialist_probe_scratch_root must be an absolute path: "
                f"{self.specialist_probe_scratch_root!r} would resolve against "
                "whatever the process working directory happens to be"
            )
        if not isinstance(self.agent_options, dict):
            raise ValueError("agent_options must be a dict")
        if self.experiments_dir is None:
            self.experiments_dir = self.project_root / "experiments"
        if self.local_knowledge_dir is None:
            self.local_knowledge_dir = resource_path("local_knowledge", self.project_root)
        if self.knowledge_config is None:
            self.knowledge_config = KnowledgeConfig.from_env(
                gbrain_base_url=self.gbrain_url or None,
                gbrain_token=self.gbrain_token or None,
            )
        self.gbrain_url = self.knowledge_config.gbrain_base_url
        self.gbrain_token = self.knowledge_config.gbrain_token
        # Only fall back to the env var when the caller didn't pass an
        # explicit value at all -- an explicit True/False (from either
        # direct construction or `from_env(include_mori_kb=...)`) always
        # wins over the environment.
        if self.include_mori_kb is None:
            self.include_mori_kb = os.getenv("KERNELFORGE_INCLUDE_MORI_KB", "").strip().lower() in ("1", "true", "yes")

    def agent_runtime(self):
        """Resolve the selected provider into one complete runtime config."""
        from kernelforge.agent_backends.registry import (
            resolve_agent_runtime,
            select_default_agent_provider,
        )

        provider = self.agent_backend
        if provider == "auto":
            provider = select_default_agent_provider(self.agent_model).name
        return resolve_agent_runtime(
            provider,
            model=self.agent_model,
            fallback_model=self.agent_fallback_model,
            executable=self.agent_cli,
            timeout_sec=self.agent_timeout_sec,
            reasoning_effort=self.agent_reasoning_effort,
            sandbox_mode=self.agent_sandbox_mode,
            precheck=self.agent_precheck,
            fallback_provider=self.agent_fallback_provider,
            options=self.agent_options,
        )

    @classmethod
    def from_env(cls, **overrides) -> Config:
        """Load config from environment variables with optional overrides."""
        if os.getenv("KERNEL_AGENTS_MAX_TURNS") is not None:
            _warn_removed_max_turns_env()
        knowledge_config = overrides.get("knowledge_config")
        if knowledge_config is None:
            knowledge_config = KnowledgeConfig.from_env(
                mode=overrides.get("knowledge_store_mode"),
                local_root=overrides.get("knowledge_local_root"),
                gbrain_base_url=overrides.get("gbrain_url"),
                gbrain_token=overrides.get("gbrain_token"),
            )
        return cls(
            gpu_target=overrides.get("gpu_target", os.getenv("GPU_TARGET", "gfx942")),
            gpu_type=str(overrides["gpu_type"] if "gpu_type" in overrides else "mi355x").strip().lower(),
            producer=str(overrides.get("producer", "")).strip().lower(),
            workspace=overrides.get("workspace", os.getenv("KERNEL_WORKSPACE", "")),
            agent_backend=overrides.get(
                "agent_backend",
                os.getenv("FORGE_AGENT_BACKEND", "auto"),
            ),
            agent_model=overrides.get(
                "agent_model",
                os.getenv("FORGE_AGENT_MODEL", "").strip() or os.getenv("KERNEL_AGENTS_MODEL", "").strip(),
            ),
            agent_cli=overrides.get("agent_cli", os.getenv("FORGE_AGENT_CLI", "")),
            agent_timeout_sec=int(
                overrides.get(
                    "agent_timeout_sec",
                    os.getenv("FORGE_AGENT_TIMEOUT_SEC", "1800"),
                )
            ),
            agent_reasoning_effort=overrides.get(
                "agent_reasoning_effort",
                os.getenv("FORGE_AGENT_REASONING_EFFORT", "high"),
            ),
            agent_sandbox_mode=overrides.get(
                "agent_sandbox_mode",
                os.getenv("FORGE_AGENT_SANDBOX_MODE", "bypass"),
            ),
            agent_precheck=overrides.get("agent_precheck", _env_bool("FORGE_AGENT_PRECHECK", True)),
            agent_fallback_provider=overrides.get(
                "agent_fallback_provider",
                os.getenv("FORGE_AGENT_FALLBACK_PROVIDER", "claude"),
            ),
            agent_fallback_model=overrides.get(
                "agent_fallback_model",
                os.getenv("FORGE_AGENT_FALLBACK_MODEL") if "FORGE_AGENT_FALLBACK_MODEL" in os.environ else None,
            ),
            agent_options=overrides.get("agent_options")
            if "agent_options" in overrides
            else _env_json_object("FORGE_AGENT_OPTIONS_JSON"),
            max_turns=int(overrides.get("max_turns", 500)),
            specialist_probe=overrides.get("specialist_probe", _env_bool("FORGE_SPECIALIST_PROBE", True)),
            specialist_probe_max=int(
                overrides.get(
                    "specialist_probe_max",
                    os.getenv("FORGE_SPECIALIST_PROBE_MAX", "6"),
                )
            ),
            specialist_probe_budget_sec=float(
                overrides.get(
                    "specialist_probe_budget_sec",
                    os.getenv("FORGE_SPECIALIST_PROBE_BUDGET_SEC", "600"),
                )
            ),
            specialist_probe_scratch_root=str(
                overrides.get(
                    "specialist_probe_scratch_root",
                    os.getenv("FORGE_SPECIALIST_PROBE_SCRATCH_ROOT", ""),
                )
            ),
            gbrain_url=knowledge_config.gbrain_base_url,
            gbrain_token=knowledge_config.gbrain_token,
            knowledge_config=knowledge_config,
            include_mori_kb=overrides.get("include_mori_kb"),
        )
