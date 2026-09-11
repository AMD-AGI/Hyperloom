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

from hyperloom.common.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
    normalize_reasoning_effort,
)
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


def resolve_agent_model(agent_backend: str) -> str:
    """Resolve the model id from the environment ladder Hyperloom publishes.

    Forge no longer ships alongside Hyperloom, it ships *inside* it, and an
    operator configuring a box should not have to learn a second vocabulary for
    the same decision. There is exactly one rung, and it is the platform's:
    ``CLAUDE_MODEL`` / ``CODEX_MODEL``, the same pair
    :func:`hyperloom.common.llm_config.resolve_forge_llm_model` reads. The two
    are written out separately rather than sharing one helper because they
    answer different questions -- that one picks the model for Hyperloom's own
    calls into a Forge campaign, this one picks the model an agent session
    runs -- and the shared piece worth deduplicating is the variable names,
    which is exactly what this change makes identical.

    Forge used to consult private variables above that pair -- first
    ``FORGE_CLAUDE_MODEL`` / ``FORGE_CODEX_MODEL``, then a provider-neutral
    ``FORGE_AGENT_MODEL`` -- from when it was a separate project that had to
    name its own settings. Inside Hyperloom every one of those is a second
    spelling of a setting the platform already names, and a second spelling is
    only ever a second place for a box to be misconfigured. The Hyperloom-side
    resolver never had them, so deleting them is what makes the two ladders the
    same ladder rather than two that agree by coincidence.

    Only a settled backend has an answer here. ``auto`` gets ``""``: which
    provider runs is not known until :meth:`Config.agent_runtime` has checked
    which CLI is actually installed, and answering early with ``CLAUDE_MODEL``
    would hand a Claude model id to Codex on a box where the Claude CLI is
    missing -- a 400 from the gateway, not a fallback.
    """
    backend = (agent_backend or "").strip().lower()
    if backend == "codex":
        return os.getenv("CODEX_MODEL", "").strip()
    if backend == "claude":
        return os.getenv("CLAUDE_MODEL", "").strip()
    return ""


def resolve_agent_reasoning_effort() -> str:
    """Resolve the reasoning effort, honouring Hyperloom's project-wide value.

    ``HYPERLOOM_REASONING_EFFORT`` already sets the effort for Hyperloom's own
    LLM calls; a box that sets it means it for the whole run, and a Forge
    campaign that ignored it would be the one component quietly running at a
    different depth than the operator asked for.
    ``FORGE_AGENT_REASONING_EFFORT`` stays above it for the run that wants Forge
    specifically turned up or down. Both name a level in
    :data:`REASONING_EFFORT_LEVELS`; a value outside it is refused here, by
    name, rather than carried into the campaign to fail at the provider once
    the run is already hours deep.
    """
    for name in ("FORGE_AGENT_REASONING_EFFORT", "HYPERLOOM_REASONING_EFFORT"):
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        effort = normalize_reasoning_effort(raw)
        if not effort:
            raise ValueError(
                f"{name}={raw!r} is not a reasoning effort; expected one of {', '.join(REASONING_EFFORT_LEVELS)}"
            )
        return effort
    return DEFAULT_REASONING_EFFORT


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
    agent_reasoning_effort: str = DEFAULT_REASONING_EFFORT
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
    # the field went with it.
    # Curated per-backend knowledge tree injected into the forge-loop system
    # prompt as an on-demand index (hardware / common_methodology / flydsl).
    local_knowledge_dir: Path = field(default=None)

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

    # On by default: render every knowledge pillar as a one-line pointer instead
    # of inlining its whole INDEX.md map. The maps are re-read on every turn of
    # every session, and the index is carried into each specialist and synthesis
    # payload as well, so the cost is far larger than one copy: measured on the
    # analysis role, the first-turn prefix falls from 51,275 tokens to 8,415
    # (-83.6%), and on the implementer lanes from a mean 62,022 (n=18) to 17,704
    # (n=4, -71.5%, ranges disjoint).
    #
    # The behaviour question -- does an agent still go looking once the map is a
    # pointer -- was the reason this stayed opt-in, and a four-a-side A/B on
    # forge-loop softmax answered it. Deferred: speedup 1.2036 / 1.1562 / 1.1068
    # / 1.1193, improved 4 of 4. Inlined: 1.0800 / 1.0481 / 1.1447 / 1.0000,
    # improved 3 of 4. The deferred arm's worst run beats the inlined arm's mean
    # (1.1068 vs 1.0682). Agents do follow the pointer: two INDEX.md reads in the
    # deferred arm were each followed by a card read, where the inlined arm read
    # INDEX.md zero times in 98 sessions.
    #
    # Set KERNELFORGE_DEFER_KNOWLEDGE_MAPS=0 to inline the maps again. Note the
    # A/B covers one kernel at n=4 a side, so that escape hatch is deliberate.
    # None means "unset, defer to the env var" -- see include_mori_kb above for
    # why a plain bool would make an explicit False indistinguishable.
    defer_knowledge_maps: bool | None = field(default=None)

    def __post_init__(self):
        """Derive paths and validate provider-specific runtime settings."""
        from kernelforge.agent_backends.registry import get_agent_provider

        self.project_root = Path(self.project_root)
        self.agent_backend = (self.agent_backend or "auto").strip().lower()
        if self.agent_backend != "auto":
            get_agent_provider(self.agent_backend)
        self.agent_reasoning_effort = normalize_reasoning_effort(
            self.agent_reasoning_effort or DEFAULT_REASONING_EFFORT
        )
        if not self.agent_reasoning_effort:
            raise ValueError(f"agent_reasoning_effort must be one of {', '.join(REASONING_EFFORT_LEVELS)}")
        self.agent_sandbox_mode = (self.agent_sandbox_mode or "bypass").strip().lower()
        self.agent_fallback_provider = (self.agent_fallback_provider or "").strip().lower()
        if self.agent_fallback_provider:
            get_agent_provider(self.agent_fallback_provider)
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
        if self.defer_knowledge_maps is None:
            # Defaults on, so the env var reads as an opt-*out*: anything that
            # is not an explicit "off" leaves the pointers in place.
            self.defer_knowledge_maps = os.getenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", "").strip().lower() not in (
                "0",
                "false",
                "no",
            )

    def agent_runtime(self):
        """Resolve the selected provider into one complete runtime config."""
        from kernelforge.agent_backends.registry import (
            resolve_agent_runtime,
            select_default_agent_provider,
        )

        provider = self.agent_backend
        if provider == "auto":
            provider = select_default_agent_provider(self.agent_model).name
        # The model variable is per-provider, so it can only be read once the
        # provider is settled -- reading it before ``auto`` resolves is how a
        # Claude model id reaches Codex.
        model = self.agent_model or resolve_agent_model(provider)
        return resolve_agent_runtime(
            provider,
            model=model,
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
        agent_backend = overrides.get("agent_backend", os.getenv("FORGE_AGENT_BACKEND", "auto"))
        return cls(
            gpu_target=overrides.get("gpu_target", os.getenv("GPU_TARGET", "gfx942")),
            gpu_type=str(overrides["gpu_type"] if "gpu_type" in overrides else "mi355x").strip().lower(),
            producer=str(overrides.get("producer", "")).strip().lower(),
            workspace=overrides.get("workspace", os.getenv("KERNEL_WORKSPACE", "")),
            agent_backend=agent_backend,
            agent_model=overrides.get("agent_model", resolve_agent_model(agent_backend)),
            agent_cli=overrides.get("agent_cli", os.getenv("FORGE_AGENT_CLI", "")),
            agent_timeout_sec=int(
                overrides.get(
                    "agent_timeout_sec",
                    os.getenv("FORGE_AGENT_TIMEOUT_SEC", "1800"),
                )
            ),
            # Resolved lazily: the env ladder refuses an off-ladder value by
            # raising, and a caller who named an effort explicitly must not be
            # made to answer for a variable their value was going to override.
            agent_reasoning_effort=(
                overrides["agent_reasoning_effort"]
                if "agent_reasoning_effort" in overrides
                else resolve_agent_reasoning_effort()
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
            defer_knowledge_maps=overrides.get("defer_knowledge_maps"),
        )
