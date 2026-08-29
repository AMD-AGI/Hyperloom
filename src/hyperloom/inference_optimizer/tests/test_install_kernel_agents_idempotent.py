# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_kernel_agents()``.

kernel_agents (the KernelForge forge-loop CLI, invoked as ``python -m
kernel_agents.cli``) is installed from the KernelForge repo *root* resolved via
``$FORGE_PATH``. The installer should:

  * pip-install the root when FORGE_PATH points at a checkout that contains
    kernel_agents and it is not yet importable,
  * skip pip when ``import kernel_agents.cli`` already works (idempotent),
  * fail-soft (log + return 0, no pip, no error) when FORGE_PATH is unset or its
    checkout does not contain kernel_agents.

Regression cover for the 2026-07-28 bug where kernel_agents was never installed
in the container and every forge-loop attempt died with
``ModuleNotFoundError: No module named 'kernel_agents'``.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"

PIP_MARKER = "pip-install-called"


def _extract_fn(name: str) -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, import_ok: bool) -> Path:
    """Stub ``$PYTHON``.

    * ``-m pip install ...``: touches PIP_MARKER, exits 0.
    * ``-c 'import ...'``: exits per ``import_ok``; once pip has run (marker
      present) it always succeeds, so the post-install verify passes.
    """
    marker = tmp_path / PIP_MARKER
    import_check = "exit 0" if import_ok else f'[ -f "{marker}" ] && exit 0 || exit 1'
    body = f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  touch "{marker}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  {import_check}
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _make_checkout(tmp_path: Path, *, with_kernel_agents: bool) -> Path:
    """A fake KernelForge checkout root that _kernel_forge_root() probes for."""
    root = tmp_path / "KernelForge"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'kernel-agents'\n", encoding="utf-8")
    if with_kernel_agents:
        ka = root / "src" / "kernel_agents"
        ka.mkdir(parents=True)
        (ka / "__init__.py").write_text("", encoding="utf-8")
    return root


def _run(tmp_path: Path, *, forge_path: str | None, import_ok: bool) -> tuple[str, int, bool]:
    """Run the extracted _kernel_forge_root + ensure_kernel_agents body.

    Returns (stdout, returncode, pip_called).
    """
    fake_py = _fake_python(tmp_path, import_ok=import_ok)
    forge_line = f'export FORGE_PATH="{forge_path}"' if forge_path is not None else "unset FORGE_PATH || true"
    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
die() {{ echo "[die] $*"; exit 1; }}
CHECK_ONLY=0
DRY_RUN=0
PYTHON="{fake_py}"
PIP_EXTRA=()
{forge_line}

{_extract_fn("_kernel_forge_root")}

{_extract_fn("_kernel_agents_ready")}

{_extract_fn("ensure_kernel_agents")}

ensure_kernel_agents
"""
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    pip_called = (tmp_path / PIP_MARKER).exists()
    return proc.stdout, proc.returncode, pip_called


def test_installs_when_forge_path_valid_and_not_importable(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_kernel_agents=True)
    out, rc, pip_called = _run(tmp_path, forge_path=str(root), import_ok=False)
    assert rc == 0, out
    assert pip_called, f"pip install should have run:\n{out}"
    assert "ensuring kernel_agents from" in out
    assert "kernel_agents installed OK from" in out


def test_skip_reinstall_when_already_importable(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_kernel_agents=True)
    out, rc, pip_called = _run(tmp_path, forge_path=str(root), import_ok=True)
    assert rc == 0, out
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "kernel_agents already importable with forge-gemm-tune; skipping install" in out


def test_fail_soft_when_forge_path_unset(tmp_path: Path) -> None:
    out, rc, pip_called = _run(tmp_path, forge_path=None, import_ok=False)
    assert rc == 0, f"unset FORGE_PATH must be fail-soft (rc 0):\n{out}"
    assert not pip_called, f"no pip when FORGE_PATH unset:\n{out}"
    assert "FORGE_PATH not set" in out


def test_fail_soft_when_checkout_lacks_kernel_agents(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_kernel_agents=False)
    out, rc, pip_called = _run(tmp_path, forge_path=str(root), import_ok=False)
    assert rc == 0, out
    assert not pip_called, f"no pip when checkout has no kernel_agents:\n{out}"
    assert "skipping optional forge-loop install" in out


# --- Static guards: keep the fix wired in ---------------------------------


def test_static_kernel_forge_root_keys_on_forge_path_only() -> None:
    body = _extract_fn("_kernel_forge_root")
    assert "FORGE_PATH" in body
    # The consolidated resolution must not reintroduce the dropped aliases.
    assert "KERNEL_FORGE_ROOT" not in body
    assert "KERNEL_FORGE_PATH" not in body


def test_static_ensure_kernel_agents_install_and_verify_wired() -> None:
    body = _extract_fn("ensure_kernel_agents")
    assert "pip install" in body, "install path must exist for the miss case"
    assert "kernel_agents installed OK" in body
    assert "already importable" in body, "idempotent skip must stay wired"
    assert "die " in body, "post-install import must be verified (die on failure)"


def test_static_readiness_probe_covers_the_fusion_package() -> None:
    """A checkout from before fusion was absorbed imports the CLI fine.

    Probing only the CLI lets such a pod skip the install and pass the check,
    and the run then dies at forge-fuse with fusion missing.
    """
    probe = _extract_fn("_kernel_agents_ready")
    assert "kernel_agents.fusion" in probe, "the readiness probe must require fusion"
    assert '"forge-gemm-tune" in getattr(main, "commands", {})' in probe, (
        "the readiness probe must require the GEMM command without assuming main is a click Group"
    )

    body = _extract_fn("ensure_kernel_agents")
    skip_probe, _, verify = body.partition("pip install")
    assert "_kernel_agents_ready" in skip_probe, "the skip gate must use the shared readiness probe"
    assert "_kernel_agents_ready" in verify, "the post-install check must use the shared readiness probe"


def test_static_standalone_forge_gemm_tune_install_is_removed() -> None:
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert "ensure_forge_gemm_tune" not in text
    assert "FORGE_GEMM_TUNE_ROOT" not in text
