# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Transitional re-export of :mod:`hyperloom_common.proctree` for importers not yet migrated.

Consumers: ``_server_lifecycle``, ``_subprocess_kill``, ``roofline``, ``geak_runner``,
``loop.dispatcher``, ``kernel_geak_dispatch``, ``specialists.launch_env``,
``specialists.reap_watch``, ``specialists.subprocess_``. Removed in unit 2.01c.
"""

from hyperloom_common.proctree import (
    REAP_MARK_ENV,
    TERM_GRACE_SEC,
    boot_unix,
    collect_tree,
    descendants,
    group_members,
    kill_tree,
    marked_pids,
    process_uids,
    running,
    signal_group,
)

__all__ = [
    "REAP_MARK_ENV",
    "TERM_GRACE_SEC",
    "boot_unix",
    "collect_tree",
    "descendants",
    "group_members",
    "kill_tree",
    "marked_pids",
    "process_uids",
    "running",
    "signal_group",
]
