# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests: a ``site-packages`` MAGPIE_PATH must not reach PYTHONPATH.

Otherwise the main venv's torch shadows an isolated vLLM venv's torch and
crashes vLLM's C extension; a source-checkout MAGPIE_PATH must still be kept.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from hyperloom.common.env_safety import is_python_package_root
from hyperloom.orchestrator.actions.executors._grid_runner import (
    _prepend_magpie_pythonpath,
)


class TestIsPythonPackageRoot:
    def test_site_packages_is_package_root(self) -> None:
        assert is_python_package_root("/opt/venv/lib/python3.12/site-packages")

    def test_dist_packages_is_package_root(self) -> None:
        assert is_python_package_root("/usr/lib/python3/dist-packages")

    def test_trailing_slash_still_detected(self) -> None:
        assert is_python_package_root("/opt/venv/lib/python3.12/site-packages/")

    def test_source_checkout_is_not_package_root(self) -> None:
        assert not is_python_package_root("/data/.cache/Magpie@abc1234")

    def test_empty_is_not_package_root(self) -> None:
        assert not is_python_package_root("")
        assert not is_python_package_root(None)


class TestGridRunnerPrepend:
    def test_site_packages_not_prepended(self) -> None:
        result = _prepend_magpie_pythonpath(
            "/opt/venv/lib/python3.12/site-packages", "/primus/shuoshuo"
        )
        assert result == "/primus/shuoshuo"
        assert "site-packages" not in result

    def test_source_checkout_prepended(self) -> None:
        result = _prepend_magpie_pythonpath("/data/.cache/Magpie@abc1234", "/existing")
        assert result == "/data/.cache/Magpie@abc1234:/existing"

    def test_source_checkout_prepended_empty_current(self) -> None:
        result = _prepend_magpie_pythonpath("/data/.cache/Magpie@abc1234", "")
        assert result == "/data/.cache/Magpie@abc1234"

    def test_empty_magpie_dir_no_change(self) -> None:
        assert _prepend_magpie_pythonpath("", "/existing") == "/existing"


ROOT = Path(__file__).resolve().parents[3] / "hyperloom" / "agents" / "kernel"
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"


def _sourceable_installer(dest_dir: Path) -> Path:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    patched = re.sub(r"(?m)^main \"\$@\"\s*$", "", text)
    copy = dest_dir / "install_sourceable.sh"
    copy.write_text(patched, encoding="utf-8")
    return copy


def _compose_pythonpath(magpie_path: str, repo_root: str = "/target-install") -> str:
    """Source install.sh and evaluate the top-level PYTHONPATH composition."""
    with tempfile.TemporaryDirectory() as td:
        sourceable = _sourceable_installer(Path(td))
        script = f"""
set -euo pipefail
export ANTHROPIC_API_KEY=ak-test-key
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export REPO_ROOT={repo_root}
export MAGPIE_PATH={magpie_path}
CHECK_ONLY=1
DRY_RUN=1
source {sourceable!s}
printf '%s' "$PYTHONPATH"
"""
        proc = subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True, timeout=60
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        return proc.stdout.strip()


class TestInstallComposePythonPath:
    def test_site_packages_magpie_not_composed(self) -> None:
        pp = _compose_pythonpath("/opt/venv/lib/python3.12/site-packages")
        assert "site-packages" not in pp
        assert "/target-install" in pp

    def test_source_checkout_magpie_composed(self) -> None:
        pp = _compose_pythonpath("/data/.cache/Magpie@abc1234")
        assert "/data/.cache/Magpie@abc1234" in pp
        assert "/target-install" in pp
