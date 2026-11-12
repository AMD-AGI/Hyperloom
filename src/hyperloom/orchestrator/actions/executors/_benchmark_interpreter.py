# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The serving framework's interpreter, which a round's server and the framework probes run under."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path


#: The interpreter of the framework image's own environment.
_IMAGE_PYTHON = "/opt/venv/bin/python"


def _resolve_probe_python(framework: str = "vllm", *, env: Mapping[str, str] | None = None) -> str:
    """Resolve the interpreter a build-accuracy probe must use.

    Args:
        framework: The serving framework whose interpreter is wanted.
        env: The environment to resolve against, defaulting to the ambient one.
            A KEEP probe passes the graded launch's environment so that a
            ``PATH`` the override rewrote selects the same executable the launch
            selected, rather than the ambient one.
    """
    resolved_env = os.environ if env is None else env
    if (framework or "").strip().lower() == "vllm":
        venv_root = resolved_env.get("VLLM_VENV_ROOT", "").strip()
        if venv_root:
            venv_python = str(Path(venv_root) / "bin" / "python")
            if os.access(venv_python, os.X_OK):
                return venv_python
    vllm_exe = shutil.which("vllm", path=resolved_env.get("PATH"))
    if vllm_exe:
        vllm_python = os.path.join(os.path.dirname(vllm_exe), "python")
        if os.path.exists(vllm_python):
            return vllm_python
    if Path(_IMAGE_PYTHON).is_file():
        return _IMAGE_PYTHON
    return shutil.which("python3", path=resolved_env.get("PATH")) or "python3"
