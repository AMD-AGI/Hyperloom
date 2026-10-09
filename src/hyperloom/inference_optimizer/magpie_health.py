# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Magpie package health checks shared by the installer and CLI preflight."""

from __future__ import annotations


def magpie_health_code(*, native_agentx: bool) -> str:
    """Keep generic imports compatible; audit native AgentX's execution contract."""
    if not native_agentx:
        return "import Magpie\n"
    from .agentx.native import _MAGPIE_SOURCE_IDENTITY_CODE

    return (
        _MAGPIE_SOURCE_IDENTITY_CODE
        + """
import inspect
import re
import sys
from pathlib import Path

import Magpie
from Magpie.modes.benchmark import AgentXConfig
from Magpie.modes.benchmark.agentx import _expand_single_node_agentx_entries

assert AgentXConfig.from_value("enable")
assert "run-eval" in inspect.getsource(_expand_single_node_agentx_entries)
expected = sys.argv[1].strip().lower()
package_root = Path(Magpie.__file__).resolve().parent
commit, _source_url = _resolve_magpie_source_identity(package_root)
_validate_magpie_execution_tree(package_root, commit)
if re.fullmatch(r"[0-9a-f]{7,40}", expected):
    assert commit and (commit.startswith(expected) or expected.startswith(commit))
"""
    )
