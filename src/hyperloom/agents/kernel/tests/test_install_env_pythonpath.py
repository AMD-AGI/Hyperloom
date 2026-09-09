#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the PYTHONPATH written into kernel-agent runtime env."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"


def _sourceable_installer(dest_dir: Path) -> Path:
    """Write a copy of install.sh with the top-level ``main \"$@\"`` call removed."""
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    # Drop only the final top-level ``main "$@"`` dispatch line.
    patched = re.sub(r"(?m)^main \"\$@\"\s*$", "", text)
    copy = dest_dir / "install_sourceable.sh"
    copy.write_text(patched, encoding="utf-8")
    return copy


class WriteEnvFilePythonPathTest(unittest.TestCase):
    """``write_env_file`` must keep the target-install root on PYTHONPATH."""

    def _run_write_env_file(
        self,
        workdir: Path,
        *,
        repo_root: Path,
        magpie_path: str,
        preexisting_pythonpath: str | None = None,
        preexisting_dotenv_pythonpath: str | None = None,
    ) -> tuple[str, str]:
        """Source the installer and invoke ``write_env_file`` in isolation."""
        sourceable = _sourceable_installer(workdir)
        runtime_dir = workdir / "runtime"
        kernel_agent_env = runtime_dir / "kernel-agent.env.sh"
        # The installer persists .env at ``$REPO_ROOT/.env`` (single .env source), so read it back from there.
        dotenv = repo_root / ".env"

        # Seed a stale .env so the installer's credentials-fallback sourcing re-imports a PYTHONPATH that lacks
        # REPO_ROOT (the re-install bug).
        if preexisting_dotenv_pythonpath is not None:
            dotenv.write_text(
                "ANTHROPIC_API_KEY=ak-from-dotenv\n"
                "ANTHROPIC_BASE_URL=https://api.anthropic.com\n"
                f"PYTHONPATH={preexisting_dotenv_pythonpath}\n",
                encoding="utf-8",
            )

        # Set only the globals write_env_file reads; keep everything else empty so the assertion isolates the
        # PYTHONPATH composition behavior.
        script = f"""
set -euo pipefail
export ANTHROPIC_API_KEY=ak-test-key
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export REPO_ROOT={repo_root!s}
export USER_DATA_PATH={workdir!s}
export HYPERLOOM_RUNTIME_DIR={runtime_dir!s}
export KERNEL_AGENT_ENV={kernel_agent_env!s}
export MAGPIE_PATH={magpie_path}
"""
        if preexisting_pythonpath is not None:
            script += f"export PYTHONPATH={preexisting_pythonpath}\n"
        script += f"""
CHECK_ONLY=0
DRY_RUN=0
source {sourceable!s}
write_env_file
"""
        proc = subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"write_env_file failed\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        return (
            kernel_agent_env.read_text(encoding="utf-8"),
            dotenv.read_text(encoding="utf-8") if dotenv.exists() else "",
        )

    def test_kernel_agent_env_pythonpath_includes_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            env_text, dotenv_text = self._run_write_env_file(
                work,
                repo_root=repo_root,
                magpie_path="/data/.cache/Magpie@abc1234",
            )

        pythonpath_line = next(
            (ln for ln in env_text.splitlines() if "export PYTHONPATH=" in ln),
            "",
        )
        self.assertTrue(pythonpath_line, f"no PYTHONPATH export in env file:\n{env_text}")
        # The --target root (REPO_ROOT) must be reachable so subprocesses can ``import hyperloom``.
        self.assertIn(
            str(repo_root),
            pythonpath_line,
            f"REPO_ROOT missing from kernel-agent env PYTHONPATH: {pythonpath_line}",
        )
        # MAGPIE_PATH must still be present (append, not replace).
        self.assertIn("/data/.cache/Magpie@abc1234", pythonpath_line)
        # And .env must carry the same reachable PYTHONPATH.
        self.assertIn(str(repo_root), dotenv_text)

    def test_pythonpath_append_preserves_existing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            env_text, _ = self._run_write_env_file(
                work,
                repo_root=repo_root,
                magpie_path="/data/.cache/Magpie@abc1234",
                preexisting_pythonpath="/pre/existing/entry",
            )

        pythonpath_line = next(
            (ln for ln in env_text.splitlines() if "export PYTHONPATH=" in ln),
            "",
        )
        self.assertIn(str(repo_root), pythonpath_line)
        self.assertIn("/data/.cache/Magpie@abc1234", pythonpath_line)
        # A previously-set PYTHONPATH entry must not be clobbered.
        self.assertIn("/pre/existing/entry", pythonpath_line)

    def test_reinstall_stale_dotenv_pythonpath_does_not_drop_repo_root(self) -> None:
        """A re-install must not lose REPO_ROOT to a stale .env PYTHONPATH."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            env_text, dotenv_text = self._run_write_env_file(
                work,
                repo_root=repo_root,
                magpie_path="/data/.cache/Magpie@abc1234",
                preexisting_dotenv_pythonpath="/data/.cache/Magpie@abc1234:",
            )

        pythonpath_line = next(
            (ln for ln in env_text.splitlines() if "export PYTHONPATH=" in ln),
            "",
        )
        self.assertIn(
            str(repo_root),
            pythonpath_line,
            f"re-install dropped REPO_ROOT to stale .env PYTHONPATH: {pythonpath_line}",
        )
        self.assertIn("/data/.cache/Magpie@abc1234", pythonpath_line)
        self.assertIn(str(repo_root), dotenv_text)
        # No duplicate Magpie entry after recomposition.
        self.assertEqual(
            pythonpath_line.count("/data/.cache/Magpie@abc1234"),
            1,
            f"duplicate MAGPIE_PATH entry in PYTHONPATH: {pythonpath_line}",
        )


if __name__ == "__main__":
    unittest.main()
