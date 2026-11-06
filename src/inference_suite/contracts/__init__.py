# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The documents ``inference_suite`` reads and writes; each validates itself and refuses unknown keys."""

from __future__ import annotations

from inference_suite.contracts.base_launch import (
    BASE_LAUNCH_SCHEMA,
    BaseLaunch,
    BaseLaunchError,
    ClientLaunch,
    GoldenAcceptance,
    GoldenCurve,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    PointAlternative,
    Redacted,
    RunnerOwned,
    SourceApi,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)

__all__ = [
    "BASE_LAUNCH_SCHEMA",
    "BaseLaunch",
    "BaseLaunchError",
    "ClientLaunch",
    "GoldenAcceptance",
    "GoldenCurve",
    "ImageFacts",
    "LaunchModel",
    "LaunchPoint",
    "LaunchQuery",
    "LaunchSource",
    "PointAlternative",
    "Redacted",
    "RunnerOwned",
    "SourceApi",
    "SourceKind",
    "Topology",
    "VersionBasis",
    "Workload",
]
