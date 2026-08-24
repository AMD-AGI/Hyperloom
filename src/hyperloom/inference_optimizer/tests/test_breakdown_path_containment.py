# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path-boundary tests for the breakdown collectors.

Artifact locations recorded in ``state.json`` are data, not trusted input:
a session directory lives on a shared filesystem and the exporter reads the
files those fields point at. These tests pin the boundary so a recorded path
can only ever resolve inside its own session.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors._common import _resolve_under_session


def _session(tmp_path: Path) -> Path:
    sd = tmp_path / "sessions" / "sid-1"
    (sd / "runs").mkdir(parents=True)
    return sd


def test_existing_absolute_path_inside_the_session_is_used(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    workspace = sd / "runs" / "benchmark_1"
    workspace.mkdir()
    assert _resolve_under_session(sd, str(workspace)) == workspace


def test_existing_absolute_path_outside_the_session_is_refused(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    assert _resolve_under_session(sd, str(outside)) is None


def test_a_recorded_path_cannot_reach_an_arbitrary_host_directory(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    assert _resolve_under_session(sd, "/etc") is None


def test_container_path_is_rerooted_at_the_session(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    workspace = sd / "runs" / "benchmark_1"
    workspace.mkdir()

    assert _resolve_under_session(sd, "/workspace/runs/benchmark_1") == workspace


def test_rerooting_wins_over_a_colliding_path_on_this_host(tmp_path: Path) -> None:
    """A container path that also exists here must not resolve to another session."""
    sd = _session(tmp_path)
    mine = sd / "runs" / "benchmark_1"
    mine.mkdir()
    # Stand in for the container-side root existing on this host and holding a
    # different session's data.
    foreign_root = tmp_path / "container"
    (foreign_root / "runs" / "benchmark_1").mkdir(parents=True)

    resolved = _resolve_under_session(sd, str(foreign_root / "runs" / "benchmark_1"))

    assert resolved == mine


def test_traversal_in_the_rerooted_suffix_is_refused(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    secret = tmp_path / "secret"
    secret.mkdir()

    assert _resolve_under_session(sd, "/workspace/runs/../../secret") is None


def test_symlink_out_of_the_session_is_refused(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = sd / "runs" / "escape"
    link.symlink_to(outside)

    assert _resolve_under_session(sd, str(link)) is None


def test_empty_and_unusable_values_resolve_to_nothing(tmp_path: Path) -> None:
    sd = _session(tmp_path)
    assert _resolve_under_session(sd, None) is None
    assert _resolve_under_session(sd, "") is None
    assert _resolve_under_session(sd, "relative/never/existed") is None


def test_another_sessions_path_never_resolves_to_that_session(tmp_path: Path) -> None:
    """``sid-1`` must not read ``sid-10``, whose path is also a string prefix match."""
    sd = _session(tmp_path)
    sibling = tmp_path / "sessions" / "sid-10" / "runs"
    sibling.mkdir(parents=True)

    resolved = _resolve_under_session(sd, str(sibling))

    assert resolved != sibling
    assert resolved is None or resolved.is_relative_to(sd)
