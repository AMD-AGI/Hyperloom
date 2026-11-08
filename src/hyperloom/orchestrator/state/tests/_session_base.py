# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Freeze a minimal base launch into a test session, the way a fresh fixed session does."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom.orchestrator.state import base_launch as state_base_launch
from inference_suite.contracts import (
    BaseLaunch,
    ClientLaunch,
    GoldenAcceptance,
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


def session_base(
    args: Sequence[str] = (),
    env: Mapping[str, str | Redacted] | None = None,
    golden: GoldenAcceptance | None = None,
) -> BaseLaunch:
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
        golden_al=golden,
    )


def freeze_session_base(
    session_dir: Path,
    args: Sequence[str] = (),
    env: Mapping[str, str | Redacted] | None = None,
    golden: GoldenAcceptance | None = None,
) -> BaseLaunch:
    """Write ``base_launch.json`` into ``session_dir`` and record its digest in the session's state."""
    return freeze_base(session_dir, session_base(args, env, golden))


def freeze_base(session_dir: Path, base: BaseLaunch) -> BaseLaunch:
    """Write ``base`` as ``session_dir``'s ``base_launch.json`` and record its digest in the session's state."""
    return state_base_launch.freeze_base(session_dir, base).launch
