# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Interpreter selection shared by benchmark backends and framework probes."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._subprocess_kill import run_with_session_kill

log = logging.getLogger(__name__)


def _resolve_magpie_python(env: Mapping[str, str] | None = None) -> str:
    """Resolve the Python interpreter for Magpie subprocesses.

    Args:
        env: The environment to resolve against, defaulting to the ambient one.
            A caller resolving the interpreter a *graded* launch used passes the
            environment that launch ran under, because ``PATH`` decides which
            interpreter a bare name resolves to.
    """
    # An ambient resolution keeps the exact call shape it always had -- no `path=`
    # on `shutil.which`, no `env=` on the probe -- so that every existing caller
    # and its test doubles are unaffected. Only an explicit environment routes
    # through the overridden lookups.
    resolved_env = os.environ if env is None else env
    which_kwargs: dict[str, Any] = {} if env is None else {"path": resolved_env.get("PATH")}
    probe_env = None if env is None else dict(resolved_env)

    def _can_import_magpie(py: str) -> bool:
        """Whether an interpreter can import Magpie and its ``yaml`` dep."""
        try:
            # find_spec keeps expected missing-module probes out of the run log.
            proc = run_with_session_kill(
                [
                    py,
                    "-c",
                    "import importlib.util as u, sys; sys.exit(0 if u.find_spec('Magpie') and u.find_spec('yaml') else 1)",
                ],
                timeout=10,
                env=probe_env,
            )
            return getattr(proc, "returncode", 1) == 0
        except Exception:  # noqa: BLE001 - probe subprocess; absence answers False
            return False

    env_val = resolved_env.get("MAGPIE_PYTHON", "").strip()
    if env_val:
        if _can_import_magpie(env_val):
            return env_val
        log.warning(
            "MAGPIE_PYTHON=%s cannot import Magpie; ignoring it and "
            "auto-detecting an interpreter that can. (A stale value is often "
            "baked into kernel-agent.env.sh when install.sh resolved it "
            "before Magpie was pip-installed.)",
            env_val,
        )

    candidate = shutil.which("python3", **which_kwargs)
    if candidate and _can_import_magpie(candidate):
        return candidate

    opt_venv = "/opt/venv/bin/python"
    if Path(opt_venv).is_file():
        return opt_venv
    if candidate:
        return candidate
    return "python3"


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
    which_kwargs: dict[str, Any] = {} if env is None else {"path": resolved_env.get("PATH")}
    if (framework or "").strip().lower() == "vllm":
        venv_root = resolved_env.get("VLLM_VENV_ROOT", "").strip()
        if venv_root:
            venv_python = str(Path(venv_root) / "bin" / "python")
            if os.access(venv_python, os.X_OK):
                return venv_python
    magpie_python = _resolve_magpie_python() if env is None else _resolve_magpie_python(resolved_env)
    # On a single-venv host the harness already uses the serving interpreter.
    if magpie_python and magpie_python != "/opt/venv/bin/python":
        return magpie_python
    vllm_exe = shutil.which("vllm", **which_kwargs)
    if vllm_exe:
        vllm_python = os.path.join(os.path.dirname(vllm_exe), "python")
        if os.path.exists(vllm_python):
            return vllm_python
    return magpie_python
