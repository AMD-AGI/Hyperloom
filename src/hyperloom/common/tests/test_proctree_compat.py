# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The old ``hyperloom.common.proctree`` path re-exports the names multibranch-dev imports from it."""

from __future__ import annotations

import pytest

from hyperloom_common import proctree

DEV_NAMES = ("TERM_GRACE_SEC", "collect_tree", "descendants", "group_members", "kill_tree", "proc_identity", "running")


@pytest.mark.parametrize("name", DEV_NAMES)
def test_each_name_multibranch_dev_imports_is_the_hyperloom_common_one(name: str) -> None:
    from hyperloom.common import proctree as old

    assert getattr(old, name) is getattr(proctree, name)
