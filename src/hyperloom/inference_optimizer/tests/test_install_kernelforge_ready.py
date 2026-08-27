# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``_check_kernelforge_ready()``.

The built-in kernel-opt agent (``kernelforge``, invoked as ``python -m
kernelforge.cli``) ships inside this distribution, so there is nothing left to
install for it — only something to verify. This replaces the old installer step
that pip-installed forge as a separate distribution from a KernelForge checkout
resolved via ``$FORGE_PATH``.

The probe must:

  * pass silently when the packages import,
  * abort the install when they do not (a partial install is not something a
    later forge run can recover from),
  * downgrade that abort to a warning under ``--check-only``,
  * treat a missing ``openai_codex`` as a warning only (it matters solely to an
    OpenAI-only deployment).

Regression cover for the 2026-07-28 bug where the forge CLI was never installed
in the container and every forge-loop attempt died with ``ModuleNotFoundError``.
The failure mode the vendoring cannot reintroduce is the install itself; the
failure mode it CAN reintroduce is an incomplete package tree, which is what the
two-module probe is for.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"


def _extract_fn(name: str) -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"could not locate {name}() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, kernelforge_rc: int, codex_rc: int) -> Path:
    """Stub ``$PYTHON``: decide each ``-c "import ..."`` probe by module name."""
    body = f"""#!/usr/bin/env bash
if [ "${{1:-}}" = "-c" ]; then
  case "${{2:-}}" in
    *kernelforge*) exit {kernelforge_rc} ;;
    *openai_codex*) exit {codex_rc} ;;
  esac
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _run(
    tmp_path: Path,
    *,
    kernelforge_rc: int = 0,
    codex_rc: int = 0,
    check_only: int = 0,
) -> tuple[str, int]:
    fake_py = _fake_python(tmp_path, kernelforge_rc=kernelforge_rc, codex_rc=codex_rc)
    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
die() {{ echo "[die] $*"; exit 1; }}
CHECK_ONLY={check_only}
DRY_RUN=0
PYTHON="{fake_py}"

{_extract_fn("_check_kernelforge_ready")}

_check_kernelforge_ready
echo "[harness] reached-end"
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
    return proc.stdout, proc.returncode


def test_passes_when_the_packaged_forge_imports(tmp_path: Path) -> None:
    out, rc = _run(tmp_path)
    assert rc == 0, out
    assert "kernelforge (built-in kernel-opt agent) OK" in out
    assert "openai_codex OK" in out


def test_aborts_when_forge_is_not_importable(tmp_path: Path) -> None:
    # A partial install: the wheel is there but its forge tree is not. Failing
    # here is the whole point — the alternative is a ModuleNotFoundError in the
    # middle of a kernel attempt, hours later.
    out, rc = _run(tmp_path, kernelforge_rc=1)
    assert rc != 0, f"a missing kernelforge must abort the install:\n{out}"
    assert "kernelforge missing" in out


def test_check_only_downgrades_the_abort_to_a_warning(tmp_path: Path) -> None:
    out, rc = _run(tmp_path, kernelforge_rc=1, check_only=1)
    assert rc == 0, f"--check-only must report, not abort:\n{out}"
    assert "kernelforge not importable" in out
    assert "reached-end" in out


def test_missing_codex_runtime_is_a_warning_only(tmp_path: Path) -> None:
    # It matters only to an OpenAI-only deployment; a claude deployment is fine
    # without it, so this must never abort a working install.
    out, rc = _run(tmp_path, codex_rc=1)
    assert rc == 0, out
    assert "openai_codex not importable" in out
    assert "reached-end" in out


# --- Static guards: keep the fix wired in ---------------------------------


def test_static_probe_covers_the_fusion_package() -> None:
    """A tree missing the fusion subpackage imports the CLI fine.

    Probing only the CLI lets such a pod pass the check, and the run then dies
    at forge-fuse with fusion missing.
    """
    body = _extract_fn("_check_kernelforge_ready")
    assert "kernelforge.cli" in body, "the CLI entry point must be probed"
    assert "kernelforge.fusion" in body, "the probe must require fusion too"


def test_static_probe_installs_nothing() -> None:
    # forge ships in this distribution; a pip install here would mean something
    # upstream failed to install it, and papering over that is how the old
    # $FORGE_PATH-resolved side install went stale.
    body = _extract_fn("_check_kernelforge_ready")
    assert "pip install" not in body
    assert "FORGE_PATH" not in body


def test_static_probe_is_called_from_both_install_paths() -> None:
    # Packaged-wheel and editable installs are separate branches of
    # ensure_inference_optimizer(); the probe must guard both.
    body = _extract_fn("ensure_inference_optimizer")
    assert body.count("_check_kernelforge_ready") == 2, (
        "both the packaged and the editable branch must run the readiness probe:\n" + body
    )
