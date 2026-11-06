# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Freeze a minimal base launch into a test session, the way a fresh fixed session does."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom.inference_optimizer.session.base_launch import write_base_launch
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts import (
    BaseLaunch,
    ClientLaunch,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    Redacted,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)


def session_base(args: Sequence[str] = (), env: Mapping[str, str | Redacted] | None = None) -> BaseLaunch:
    return BaseLaunch(
        query=LaunchQuery(
            model_id="org/model",
            framework="sglang",
            board="mi355x",
            workload=Workload.FIXED,
            concurrencies=(4,),
            isl=1024,
            osl=1024,
        ),
        source=LaunchSource(kind=SourceKind.USER_FILE, path="launch.sh", files={"launch.sh": "c" * 64}),
        point=LaunchPoint(conc=4, exact=True),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        framework="sglang",
        entrypoint=("sglang", "serve"),
        model=LaunchModel("org/model", "/models/model"),
        topology=Topology(tp=1),
        args=tuple(args),
        env=dict(env or {}),
        client=ClientLaunch(tool="aiperf", commit="d" * 40),
        resolved_unix=1_700_000_000,
    )


def freeze_session_base(
    session_dir: Path, args: Sequence[str] = (), env: Mapping[str, str | Redacted] | None = None
) -> BaseLaunch:
    """Write ``base_launch.json`` into ``session_dir`` and record its digest in the session's state."""
    base = session_base(args, env)
    write_base_launch(session_dir, base)
    state = SharedState.load_or_init(session_dir)
    state.base_launch_digest = base.digest()
    state.save(session_dir)
    return base
