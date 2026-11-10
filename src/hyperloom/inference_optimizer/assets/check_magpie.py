# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check the revision installed by pip without importing the GPU runtime."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
import json
import sys


def installed_revision_matches(expected: str) -> bool:
    """PEP 610 metadata binds the installed package to its immutable commit."""
    if importlib.util.find_spec("Magpie") is None:
        return False
    try:
        direct = distribution("magpie-eval").read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    vcs = json.loads(direct or "{}").get("vcs_info", {})
    return expected in {vcs.get("commit_id"), vcs.get("requested_revision")}


if __name__ == "__main__":
    sys.exit(0 if installed_revision_matches(sys.argv[1]) else 1)
