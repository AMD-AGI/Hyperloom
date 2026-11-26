# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind each lane that runs beside others to its own workspace copy, and refuse what cannot run lanes side by side."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from kernelforge.config import Config
from kernelforge.evaluation.scoring import GateScoring
from kernelforge.utils.device_queue import RUNNER_CONFIG_NAME, hidden_device_environment
from kernelforge.utils.git import repository_selected_by_environment
from kernelforge.utils.import_root import import_name_of, import_path_first
from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopError, ForgeLoopUsageError
from kernelforge.workflows.forge_loop.gate.gate_settings import DeviceAccess
from kernelforge.workflows.forge_loop.roles.agent_extras import declared_extras

if TYPE_CHECKING:
    from kernelforge.workflows.forge_loop.iteration.fanout import LaneSeat
    from kernelforge.workflows.forge_loop.roles.agent import ImplementerSetup


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
# What ``stop_hooks`` buys a lane is in-session denial: the PreToolUse callbacks refuse an edit to the driver, harness
# or oracle, and refuse a command that selects a device around the queue the lanes share. A provider that ignores
# ``AgentRunSpec.hooks`` gets neither.
#
# It is an advisory rather than a refusal because a campaign's published numbers do not rest on it: a lane's patch
# that touches the measurement surface is refused before it reaches the canonical tree, the canonical driver is
# re-checked byte-for-byte after the patch applies, and a KEEP is decided by the canonical measurement alone.
_LANE_PROVIDER_ADVISORIES = (
    (
        "stop_hooks",
        "run the callbacks in AgentRunSpec.hooks, which is what denies a lane "
        "an edit to the driver, harness or oracle while its session can still "
        "be saved, and denies a command that selects a device around the queue "
        "its siblings wait in. Without them a lane can lose its whole session "
        "at the boundary check",
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


def _require_copyable_workspace(lanes: int) -> None:
    """Refuse side-by-side lanes on a workspace that cannot be copied.

    Lanes that run side by side each work in a copy of the workspace. A workspace whose repository is selected
    through the environment -- a framework tree edited where it is installed, its history kept elsewhere -- has no
    repository a copy could carry. One lane works in the workspace itself and needs none. It is the lane count the
    caller passed that cannot be used here, whichever Agent profile would run, so the run is refused as a usage
    error when it is asked for, not while a profile is being started.
    """
    selected = repository_selected_by_environment()
    if lanes < 2 or not selected:
        return
    raise ForgeLoopUsageError(
        f"--lanes {lanes} runs lanes in copies of the workspace, and this workspace's repository is selected "
        f"through {' and '.join(selected)}, which a copy cannot carry. Re-run with --lanes 1."
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
    from kernelforge.utils.aiter_cache import child_cache_environment

    def factory(seat: LaneSeat):
        # Each lane gets its own Config: a provider that requires the workspace as its cwd would otherwise start every
        # lane's session in the shared canonical workspace.
        lane_dir = str(seat.lane_dir)
        lane_root = seat.lane_dir.resolve()
        lane_config = replace(config, workspace=str(lane_root))

        def rebound(value: str, label: str) -> str:
            return _lane_workspace_path(value, lane_dir=lane_dir, workspace_dir=binding.workspace_dir, label=label)

        # Each lane compiles a different edit of the same kernel. aiter imports a JIT module by name and never checks
        # the .so against the source it was built from, so lanes sharing one build cache load each other's binaries
        # and each one measures a kernel it did not write.
        measurement_env = child_cache_environment(lane_root.with_name(lane_root.name + _LANE_AITER_CACHE_SUFFIX))
        if import_name_of(Path(binding.workspace_dir)):
            # The lane copy stands under the workspace's name in a directory of its own, which has to come before the
            # one holding the canonical workspace: by that name the lane's session would otherwise import the
            # canonical workspace, and time a kernel the lane did not write.
            measurement_env["PYTHONPATH"] = import_path_first(lane_root.parent, os.environ.get("PYTHONPATH", ""))
        device = DeviceAccess(
            runner=str(seat.device_runner),
            queue_root=str(seat.queue_root),
            hold_limit_sec=seat.hold_limit_sec,
            usage_log=str(seat.usage_log),
            lane=seat.assignment.lane_id,
            environment=measurement_env,
        )
        runner_files = [str(seat.device_runner), str(seat.device_runner.with_name(RUNNER_CONFIG_NAME))]
        lane_setup = replace(
            setup,
            task=replace(
                setup.task,
                source_files=[rebound(path, "source file") for path in binding.source_files],
                extra_protected_paths=[*(setup.task.extra_protected_paths or []), *runner_files],
            ),
            measurement=replace(
                setup.measurement,
                driver_script=rebound(binding.driver, "driver"),
                device=device,
                # A lane runs beside others: a correct candidate may stop, and what it gained is the merge's to use.
                gain_required=False,
            ),
            session=replace(setup.session, insession_gate=True),
        )
        lane_agent = make_agent(lane_config, lane_setup, usage=usage)
        # The session itself sees no device: every command that needs one goes through the lane's runner.
        session_env = {**measurement_env, **hidden_device_environment()}

        async def session(
            kernel_path: str,
            history: str,
            session_sink: dict | None = None,
            gate_scoring: GateScoring = GateScoring(),
        ) -> str:
            """Run the lane's Implementer session under the Implementer callback's own contract.

            The loop hands a session only the optional arguments it names, so both are named here: the scoring state
            the lane's stop check measures against, and the sink the round reads the lane's result from.
            """
            _assert_lane_session_cwd(
                kernel_path=kernel_path,
                workspace=lane_config.workspace,
                lane_dir=lane_dir,
            )
            extras = declared_extras(lane_agent, {"session_sink": session_sink, "gate_scoring": gate_scoring})
            try:
                with session_environment(session_env):
                    return await lane_agent(kernel_path, history, **extras)
            finally:
                # The backend was built for this one session; its record turn resumes the session by id.
                lane_agent.close()

        return session

    return factory
