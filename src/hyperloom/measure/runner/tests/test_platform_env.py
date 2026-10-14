# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.measure.runner.platform_env import align_platform_env, take_topology


def test_topology_variables_are_removed_and_returned():
    env = {"MASTER_ADDR": "a", "RANK": "0", "PET_NNODES": "1", "PATH": "/bin"}

    taken = take_topology(env)

    assert taken == {"MASTER_ADDR": "a", "RANK": "0", "PET_NNODES": "1"}
    assert env == {"PATH": "/bin"}


def test_platform_variables_are_set_deleted_and_the_changed_names_reported():
    env = {"NCCL_IB_SL": "3", "HSA_ENABLE_SDMA": "0", "NCCL_CROSS_NIC": "1", "OTHER": "x"}

    changed = align_platform_env(env, {"HSA_ENABLE_SDMA": "0", "NCCL_CROSS_NIC": "2", "NCCL_IB_TIMEOUT": "20"})

    assert env == {"HSA_ENABLE_SDMA": "0", "NCCL_CROSS_NIC": "2", "NCCL_IB_TIMEOUT": "20", "OTHER": "x"}
    assert changed == ("NCCL_CROSS_NIC", "NCCL_IB_SL", "NCCL_IB_TIMEOUT")


def test_a_variable_that_is_not_a_platform_variable_is_refused():
    with pytest.raises(ValueError, match="not platform variables: \\['PATH'\\]"):
        align_platform_env({}, {"PATH": "/x"})
