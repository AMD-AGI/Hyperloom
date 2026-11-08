# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The old grid_server_args and framework_registry names multibranch-dev imports stay importable from their old paths."""

from __future__ import annotations

import pytest

from inference_suite import argv
from inference_suite.frameworks import Framework, framework_facts


@pytest.mark.parametrize("name", ["remove_server_args", "tokenize_server_args_preserving_json"])
def test_each_argv_name_multibranch_dev_imports_is_the_inference_suite_one(name: str) -> None:
    from hyperloom.inference_optimizer import grid_server_args

    assert getattr(grid_server_args, name) is getattr(argv, name)


def test_the_expert_parallel_table_multibranch_dev_imports_holds_each_frameworks_flag() -> None:
    from hyperloom.inference_optimizer.framework_registry import EXPERT_PARALLEL_FLAGS

    expected = {}
    for framework in Framework:
        ep = framework_facts(framework).expert_parallel
        if ep is not None:
            expected[framework.value] = (ep.flag, ep.aliases, ep.takes_size)
    assert EXPERT_PARALLEL_FLAGS == expected
    assert EXPERT_PARALLEL_FLAGS["sglang"] == ("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True)
