# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Behavioural + static guards for ensure_geak()'s pip-install path.

GEAK dropped its setup.sh and now ships as a pip package. ensure_geak() must:

* install the GEAK package from the local ${GEAK_ROOT} checkout (NOT a
  git+<remote>@<ref> URL), so the installed package matches the
  interface/run_e2e.py we run and honours GEAK_REPO/GEAK_REF overrides
  (local mirror / fork / SSH URL) that are not valid pip URLs;
* pass GEAK_HOME=${GEAK_ROOT} so GEAK's bootstrap reuses our checkout;
* skip the package install with a clear warning when the checkout carries no
  pyproject.toml/setup.py, instead of failing obscurely;
* never mention setup.sh again.

The behavioural test extracts the real ensure_geak body from install.sh and
runs it with stubbed log/warn/run (run only echoes, so nothing hits the
network or pip).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
INSTALL_SH = REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"


def _extract_ensure_geak() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(r"^ensure_geak\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_geak() in install.sh"
    return m.group(0)


def _run_ensure_geak(tmp_path: Path, *, package_metadata: bool) -> str:
    """Run the extracted ensure_geak body with stubs; return combined output."""
    geak_root = tmp_path / "os" / "GEAK"
    (geak_root / ".git").mkdir(parents=True)          # take the "already present" path
    (geak_root / "interface").mkdir(parents=True)
    (geak_root / "interface" / "run_e2e.py").write_text("# runner\n", encoding="utf-8")
    if package_metadata:
        (geak_root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log()  {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
run()  {{ echo "RUN: $*"; }}
CHECK_ONLY=0
DRY_RUN=0
GEAK_ROOT="{geak_root}"
# A non-HTTPS override that is a valid `git clone` target but NOT a valid
# `git+...` pip URL — proves we never build such a URL.
GEAK_REPO="git@github.com:acme/GEAK.git"
GEAK_REF="main"
GEAK_E2E_RUNNER="${{GEAK_ROOT}}/interface/run_e2e.py"

{_extract_ensure_geak()}

ensure_geak
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
    assert proc.returncode == 0, f"ensure_geak harness failed:\n{proc.stdout}"
    return proc.stdout


def test_installs_local_checkout_not_git_url(tmp_path: Path) -> None:
    out = _run_ensure_geak(tmp_path, package_metadata=True)
    geak_root = tmp_path / "os" / "GEAK"
    # Installs the local checkout, with GEAK_HOME pointing at it.
    assert f"GEAK_HOME={geak_root} python3 -m pip install" in out, out
    assert f"pip install -q --no-cache-dir --break-system-packages {geak_root}" in out, out
    # Never refetches from the remote via a git+ pip URL.
    assert "git+" not in out, out
    # setup.sh is gone.
    assert "setup.sh" not in out, out
    # SDK still installed; runner present so no "missing" warnings.
    assert "pip install -q --no-cache-dir --break-system-packages claude-agent-sdk anyio" in out, out
    assert "package metadata missing" not in out, out
    assert "e2e runner not found" not in out, out


def test_skips_pip_with_warning_when_no_package_metadata(tmp_path: Path) -> None:
    out = _run_ensure_geak(tmp_path, package_metadata=False)
    geak_root = tmp_path / "os" / "GEAK"
    assert "package metadata missing" in out, out
    # The package install must be skipped (no pip install of the checkout dir)...
    assert f"pip install -q --no-cache-dir --break-system-packages {geak_root}" not in out, out
    # ...but the SDK install still runs.
    assert "claude-agent-sdk anyio" in out, out


def test_static_guards_pip_from_checkout() -> None:
    body = _extract_ensure_geak()
    assert 'python3 -m pip install ${_PIP_FLAGS} "${GEAK_ROOT}"' in body, \
        "ensure_geak must pip-install the local ${GEAK_ROOT} checkout"
    assert 'GEAK_HOME="${GEAK_ROOT}"' in body, "must pass GEAK_HOME to reuse the checkout"
    assert "git+" not in body, "must not build a git+<remote> pip URL"
    assert "setup.sh" not in body, "setup.sh path must be fully removed"
