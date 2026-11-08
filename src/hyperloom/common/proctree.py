# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The old path of :mod:`hyperloom_common.proctree`, kept for callers this branch does not migrate.

multibranch-dev's ``scheduling.processes``, ``specialist.agent_sandbox``, ``specialist.agent_watch`` and their tests
import these names from here; removed with 5.12 once they are gone.
"""

from hyperloom_common.proctree import (
    TERM_GRACE_SEC,
    collect_tree,
    descendants,
    group_members,
    kill_tree,
    proc_identity,
    running,
)

__all__ = ["TERM_GRACE_SEC", "collect_tree", "descendants", "group_members", "kill_tree", "proc_identity", "running"]
