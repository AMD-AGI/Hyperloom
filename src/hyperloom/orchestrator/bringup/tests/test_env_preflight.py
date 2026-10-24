# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checkpoint fault that belongs to the host."""

from __future__ import annotations

from hyperloom.common.bringup import LadderStage
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


def test_the_observation_carries_the_fault_and_the_stage_the_launch_would_die_at():
    """A fault records like any other bring-up observation, and says it is one."""
    verdict = ep.EnvVerdict(status=ep.FAULT, fault=ep.CHECKPOINT_UNRESOLVED, detail="no such path")
    observation = ep.env_fault_observation(verdict)
    assert observation.stage_failed == LadderStage.CONFIG_VALIDATE
    assert observation.env_fault == ep.CHECKPOINT_UNRESOLVED
    assert ep.is_env_fault(observation)
    assert not observation.booted


def test_a_boot_failure_the_classifier_placed_is_not_an_environment_fault():
    """Only an observation carrying ``env_fault`` ends the run as infrastructure."""
    from hyperloom.orchestrator.bringup.ladder import BootStreams, classify

    observation = classify(
        BootStreams("ValueError: Architectures ['XForCausalLM'] are not supported for now."), server_elapsed_sec=1.0
    )
    assert not ep.is_env_fault(observation)


def test_a_host_resource_shortfall_is_not_one_of_the_terminal_faults():
    """A configuration that asks for more than the host has is still the loop's to fix."""
    from hyperloom.common.failure_signature import RESOURCE_CONSTRAINT
    from hyperloom.orchestrator.bringup.ladder import BootStreams, classify

    observation = classify(
        BootStreams("RuntimeError: HIP out of memory. Tried to allocate 4.00 GiB"), server_elapsed_sec=1.0
    )
    assert observation.env_fault == RESOURCE_CONSTRAINT
    assert not ep.is_env_fault(observation)
