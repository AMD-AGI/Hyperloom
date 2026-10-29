# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind concurrent Implementer lanes to their own workspace copies, and refuse providers that cannot isolate them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from kernelforge.config import Config
from kernelforge.loop.bootstrap.errors import ForgeLoopError

if TYPE_CHECKING:
    from kernelforge.orchestrator.agent import ImplementerSetup


# Suffix naming one lane's private AITER build cache, placed beside the lane copy rather than inside it.
_LANE_AITER_CACHE_SUFFIX = ".aiter-cache"


def _lane_workspace_path(value: str, *, lane_dir: str, workspace_dir: str, label: str) -> str:
    """Rebind one canonical workspace path onto a lane's own copy of it."""
    workspace_root = Path(workspace_dir).resolve()
    value_path = Path(value)
    resolved = value_path.resolve() if value_path.is_absolute() else (workspace_root / value_path).resolve()
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(
            f"lane isolation cannot rebind the {label} {resolved} onto "
            f"{lane_dir}: it is outside the campaign workspace {workspace_root}"
        ) from error
    return str(Path(lane_dir).resolve() / relative)


def _assert_lane_session_cwd(*, kernel_path: str, workspace: str, lane_dir: str) -> None:
    """Fail unless every directory the session could start in is inside the lane."""
    lane_root = Path(lane_dir).resolve()
    candidates = {"kernel directory": Path(kernel_path).resolve().parent}
    if workspace:
        candidates["configured workspace"] = Path(workspace).resolve()
    outside = sorted(
        f"{label} {candidate}"
        for label, candidate in candidates.items()
        if candidate != lane_root and lane_root not in candidate.parents
    )
    if outside:
        raise ValueError(f"lane session would run outside its lane {lane_root}: " + "; ".join(outside))


# What a lane needs its provider to do, and what a provider that does not do it costs.
#
# Refusal is reserved for a guarantee whose absence makes a lane's own
# measurement meaningless. Every lane compiles a different edit of one kernel,
# and aiter loads a JIT module by name without checking the binary against the
# source it was built from, so lanes sharing one build cache measure each
# other's binaries. Nothing downstream can tell that apart from a real result,
# which is what makes it a refusal rather than a warning.
_LANE_PROVIDER_REQUIREMENTS = (
    (
        "session_env",
        "apply AgentRunSpec.env to the session it spawns, which is what gives "
        "each lane its own AITER build cache instead of one they share, where "
        "a lane can measure a module a sibling compiled",
    ),
)


# Guarantees a lane is better off with and can still run without.
#
# What ``stop_hooks`` buys a lane is in-session denial: the PreToolUse
# callbacks refuse an edit to the driver, harness or oracle, and refuse a driver
# run that skips the shared device lock. A provider that ignores
# ``AgentRunSpec.hooks`` gets neither.
#
# It is an advisory rather than a refusal because a campaign's published
# numbers do not rest on it. A lane candidate that touches the measurement
# surface is refused by ``IterationRunner._lane_rejection`` on its patch paths,
# the canonical driver is re-checked byte-for-byte by
# ``_validate_driver_integrity`` after the patch applies, and every candidate is
# re-measured one at a time in the canonical tree -- a lane's own timings never
# decide a KEEP. Unlocked concurrent benchmarking is caught after the fact too,
# by the lane teardown's contention report, which voids the round out loud.
#
# So the cost of running without it is wasted budget, reported when it happens,
# rather than a result nobody can distinguish from a real one.
_LANE_PROVIDER_ADVISORIES = (
    (
        "stop_hooks",
        "run the callbacks in AgentRunSpec.hooks, which is what denies a lane "
        "an edit to the driver, harness or oracle while its session can still "
        "be saved, and denies a driver run that skips the device lock its "
        "siblings are queueing on. Without them a lane can lose its whole "
        "session at the boundary check, and concurrent unlocked benchmarks can "
        "corrupt their own and their siblings' timings -- the round is voided "
        "rather than trusted when that happens",
    ),
)


def _require_lane_provider_capabilities(provider: str, lanes: int) -> None:
    """Refuse concurrent lanes a provider cannot measure, warn about the rest."""
    if lanes < 2:
        return
    from kernelforge.agent_backends.registry import get_agent_provider

    capabilities = get_agent_provider(provider).capabilities
    # Warned before any refusal below, so one re-run is still enough: an
    # operator who switches to a provider that clears the refusal has already
    # been told what that provider will and will not do for a lane. Each is
    # named so a voided round is a known cost of this provider rather than a
    # mystery found in the logs months later.
    for name, detail in _LANE_PROVIDER_ADVISORIES:
        if getattr(capabilities, name, False):
            continue
        print(f"  [lanes] WARNING: agent provider {provider!r} does not declare {name}; it cannot {detail}")
    missing = [
        f"{name} (it must {detail})"
        for name, detail in _LANE_PROVIDER_REQUIREMENTS
        if not getattr(capabilities, name, False)
    ]
    if not missing:
        return
    raise ForgeLoopError(
        f"--lanes {lanes} needs guarantees that agent provider {provider!r} "
        "does not declare: " + "; ".join(missing) + ". Re-run with --lanes 1, "
        "or select a provider that declares them."
    )


@dataclass(frozen=True)
class LaneBinding:
    """The canonical workspace paths a lane rebinds onto its own copy."""

    workspace_dir: str
    driver: str
    source_files: Iterable[str]


def _make_lane_agent_factory(
    *,
    make_agent,
    config: Config,
    binding: LaneBinding,
    setup: ImplementerSetup,
    usage=None,
):
    """Build the factory that binds one Implementer session to one lane."""
    from kernelforge.agent_backends.base import session_environment
    from kernelforge.tools.aiter_cache import child_cache_environment

    def factory(lane_dir: str, serialized_driver: str | None):
        # Each lane gets its own Config: a provider that requires the workspace as its cwd would otherwise start every
        # lane's session in the shared canonical workspace.
        lane_root = Path(lane_dir).resolve()
        lane_config = replace(config, workspace=str(lane_root))
        lane_driver = _lane_workspace_path(
            binding.driver,
            lane_dir=lane_dir,
            workspace_dir=binding.workspace_dir,
            label="driver",
        )
        lane_sources = [
            _lane_workspace_path(
                path,
                lane_dir=lane_dir,
                workspace_dir=binding.workspace_dir,
                label="source file",
            )
            for path in binding.source_files
        ]
        lane_setup = replace(
            setup,
            task=replace(setup.task, source_files=lane_sources, profiling_enabled=False),
            # The lock-taking wrapper the round installed for this lane.
            measurement=replace(setup.measurement, driver_script=lane_driver, interposed_driver_path=serialized_driver),
            session=replace(setup.session, insession_gate=True, insession_gate_stop_check=False),
        )
        lane_agent = make_agent(lane_config, lane_setup, usage=usage)
        # Each lane compiles a different edit of the same kernel. aiter imports a JIT module by name and never checks
        # the .so against the source it was built from, so lanes sharing one build cache load each other's binaries
        # and each one measures a kernel it did not write.
        lane_env = child_cache_environment(lane_root.with_name(lane_root.name + _LANE_AITER_CACHE_SUFFIX))

        async def session(kernel_path: str, plan: str, *, session_sink: dict | None = None) -> str:
            _assert_lane_session_cwd(
                kernel_path=kernel_path,
                workspace=lane_config.workspace,
                lane_dir=lane_dir,
            )
            try:
                with session_environment(lane_env):
                    return await lane_agent(kernel_path, plan, session_sink=session_sink)
            finally:
                # The backend was built for this one session and nothing resumes it once the lane returns.
                lane_agent.close()

        return session

    return factory
