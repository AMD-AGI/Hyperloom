# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load ``install_kernel_tools.sh`` the way ``install.sh`` does, for tests that drive its steps."""

from __future__ import annotations

import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
INSTALL_SH = ASSETS / "install.sh"
KERNEL_TOOLS_SH = ASSETS / "install_kernel_tools.sh"

# Defined by install.sh and called by the kernel-agent steps.
_INSTALL_SH_HELPERS = ("log", "warn", "die", "run", "acquire_install_lock", "_resolve_ref_sha", "_prune_dep_cache")


def install_sh_function(name: str) -> str:
    """Return the source of the top-level function ``name`` in install.sh."""
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(name)}\(\)\s*\{{", line):
            if line.rstrip().endswith("}"):
                return line
            end = next(j for j in range(i + 1, len(lines)) if lines[j] == "}")
            return "\n".join(lines[i : end + 1])
    raise AssertionError(f"install.sh defines no function {name!r}")


def kernel_tools_preamble() -> str:
    """Shell text giving the kernel-agent steps what install.sh has set up before it sources them."""
    helpers = "\n".join(install_sh_function(name) for name in _INSTALL_SH_HELPERS)
    return (
        f"{helpers}\n"
        ': "${CHECK_ONLY:=0}"\n'
        ': "${DRY_RUN:=0}"\n'
        ': "${PYTHON:=$(command -v python3)}"\n'
        ': "${_open_source_root:=${HYPERLOOM_CACHE_DIR:-${REPO_ROOT}/.cache}}"\n'
    )


def sourceable_kernel_tools(dest_dir: Path) -> Path:
    """Write the preamble plus the kernel-agent steps to one file a test can ``source``."""
    copy = dest_dir / "kernel_tools_sourceable.sh"
    copy.write_text(kernel_tools_preamble() + KERNEL_TOOLS_SH.read_text(encoding="utf-8"), encoding="utf-8")
    return copy
