###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The named selector dimensions a grouped-kernel backend reads off one TraceLens shape row."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.trace_analysis._task_group_contract import named_dimensions


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([[256, 4096, 7168], [256, 7168, 2048]], id="down-projection-weight"),
        pytest.param([[256, 4096, 7168]], id="fused-gate-up-weight-only"),
    ],
)
def test_a_fused_moe_row_names_tokens_hidden_size_topk_experts_and_intermediate_size(weights):
    """N is read off the down-projection weight when the row has one, else off half the fused gate/up weight."""
    row = {"operation": "fused_moe", "input_shapes": [[64, 7168], [64, 8], *weights]}

    assert named_dimensions(row) == {"M": 64, "K": 7168, "TOPK": 8, "E": 256, "N": 2048}
