# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checkpoint fault that belongs to the host."""

from __future__ import annotations

from hyperloom.orchestrator.bringup import env_preflight as ep


def test_a_checkpoint_path_that_does_not_resolve_is_a_fault(tmp_path):
    """An absolute weights path with nothing behind it stops the round."""
    verdict = ep.check_environment(model=str(tmp_path / "absent-checkpoint"))
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.CHECKPOINT_UNRESOLVED
    assert verdict.terminal


def test_a_repository_id_is_not_judged_as_a_path():
    """A hub id names no local path, so nothing about this host refutes it."""
    assert ep.check_environment(model="org/Model-8B").status == ep.OK
