# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node grid restarts must carry the same reference server envs as baseline.

Single-node reads the reference envs out of the materialized YAML, because Magpie
launches the server from that YAML. Multi-node launches through
``restart_server_for_round``, which never reads it, and the baseline forwards them
explicitly -- so a grid that forwarded only ``variant.extra_envs`` measured every
candidate on a server missing the envs the baseline ran with, and attributed the
difference to the candidate's flags.
"""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _mn_restart_env


def test_reference_envs_reach_an_arg_only_variant():
    """An arg-only variant still needs the reference envs the baseline had."""
    merged = _mn_restart_env({"SGLANG_REF_KNOB": "1"}, GridVariant(name="args-only"), [])

    assert merged == {"SGLANG_REF_KNOB": "1"}


def test_variant_env_wins_over_reference():
    """Priority is reference < variant, mirroring the args layering."""
    variant = GridVariant(name="v", extra_envs={"SGLANG_REF_KNOB": "candidate"})

    merged = _mn_restart_env({"SGLANG_REF_KNOB": "reference", "OTHER": "keep"}, variant, [])

    assert merged == {"SGLANG_REF_KNOB": "candidate", "OTHER": "keep"}


def test_unset_beats_the_reference():
    """Unset must still mean unset once the reference supplies the key."""
    variant = GridVariant(name="v", unset_envs=["SGLANG_REF_KNOB"])

    merged = _mn_restart_env({"SGLANG_REF_KNOB": "reference"}, variant, ["SGLANG_REF_KNOB"])

    assert merged == {}


def test_no_reference_is_byte_for_byte_the_old_behaviour():
    """No reference recipe configured => exactly the variant's own envs."""
    variant = GridVariant(name="v", extra_envs={"MORI_X": "1"})

    assert _mn_restart_env({}, variant, []) == {"MORI_X": "1"}
