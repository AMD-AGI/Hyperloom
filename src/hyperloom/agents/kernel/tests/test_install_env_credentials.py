#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for credential precedence in kernel-agent runtime env.

Context: every documented launch path sources ``.env`` first and the generated
``kernel-agent.env.sh`` second, and only the first install runs ``install.sh``.
A credential the file re-exports unconditionally therefore outranks the one the
operator just rotated in ``.env``, on every launch, until the installer is run
again -- so rotating a key does not actually rotate it (#1169).

Credentials are runtime input owned by ``.env``/the caller; the paths this file
also carries are install-time results owned by the installer. These tests pin
both halves: credentials fall back, paths still win.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"

_INSTALL_KEY = "ak-install-time-key"
_INSTALL_URL = "https://gateway.install-time.example/v1"
_INSTALL_TOKEN = "sk-ant-oat01-install-time-token"
_ROTATED_KEY = "ak-rotated-key"
_ROTATED_URL = "https://gateway.rotated.example/v1"
_ROTATED_TOKEN = "sk-ant-oat01-rotated-token"


def _sourceable_installer(dest_dir: Path) -> Path:
    """Write a copy of install.sh with the top-level ``main "$@"`` call removed.

    The installer ends with an unguarded ``main "$@"``; sourcing it directly
    would run the whole install. Strip that single trailing invocation so the
    copy only defines functions and top-level variables, letting a test source
    it and call an individual function (``write_env_file``) in isolation.
    """
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    patched = re.sub(r"(?m)^main \"\$@\"\s*$", "", text)
    copy = dest_dir / "install_sourceable.sh"
    copy.write_text(patched, encoding="utf-8")
    return copy


class KernelAgentEnvCredentialPrecedenceTest(unittest.TestCase):
    """``kernel-agent.env.sh`` must not outrank a rotated ``.env`` credential."""

    def _write_env_file(self, workdir: Path, repo_root: Path) -> Path:
        """Generate ``kernel-agent.env.sh`` with the install-time credentials.

        Returns:
            Path: The generated env file.
        """
        sourceable = _sourceable_installer(workdir)
        kernel_agent_env = workdir / "runtime" / "kernel-agent.env.sh"
        # GEAK_ROOT preset: an unset one makes the installer resolve GEAK_REF
        # through ``git ls-remote``, which would put this test on the network.
        script = f"""
set -euo pipefail
export ANTHROPIC_API_KEY={_INSTALL_KEY}
export ANTHROPIC_BASE_URL={_INSTALL_URL}
export CLAUDE_CODE_OAUTH_TOKEN={_INSTALL_TOKEN}
export REPO_ROOT={repo_root!s}
export USER_DATA_PATH={workdir!s}
export HYPERLOOM_RUNTIME_DIR={workdir / "runtime"!s}
export KERNEL_AGENT_ENV={kernel_agent_env!s}
export MAGPIE_PATH=/data/.cache/Magpie@abc1234
export GEAK_ROOT={workdir / "geak"!s}
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
        return kernel_agent_env

    def _source_and_report(self, env_file: Path, exported: dict[str, str]) -> tuple[dict[str, str], str]:
        """Source ``env_file`` after exporting ``exported``, like a real launch.

        Returns:
            tuple[dict[str, str], str]: The resulting values of the reported
            variables, and the file's stderr output.
        """
        reported = (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "MAGPIE_PATH",
        )
        exports = "".join(f"export {k}={v}\n" for k, v in exported.items())
        script = exports + f". {env_file!s}\n" + "".join(f'echo "{name}=${{{name}:-}}"\n' for name in reported)
        proc = subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"sourcing the env file failed\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        values = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        return values, proc.stderr

    def test_rotated_dotenv_credentials_survive_the_env_file(self) -> None:
        """The launch order (.env then this file) must keep the rotated values."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "repo"
            repo_root.mkdir()
            env_file = self._write_env_file(work, repo_root)
            values, stderr = self._source_and_report(
                env_file,
                {
                    "ANTHROPIC_API_KEY": _ROTATED_KEY,
                    "ANTHROPIC_BASE_URL": _ROTATED_URL,
                    "CLAUDE_CODE_OAUTH_TOKEN": _ROTATED_TOKEN,
                },
            )

        self.assertEqual(
            values["ANTHROPIC_API_KEY"],
            _ROTATED_KEY,
            "the install-time key snapshot overrode the rotated .env key",
        )
        self.assertEqual(values["ANTHROPIC_BASE_URL"], _ROTATED_URL)
        self.assertEqual(values["CLAUDE_CODE_OAUTH_TOKEN"], _ROTATED_TOKEN)
        # A silent override is what made #1169 expensive to diagnose, so the
        # mismatch must be announced -- without disclosing either value.
        self.assertIn("ANTHROPIC_API_KEY", stderr)
        self.assertNotIn(_INSTALL_KEY, stderr)
        self.assertNotIn(_ROTATED_KEY, stderr)

    def test_env_file_still_supplies_credentials_when_unset(self) -> None:
        """Slurm/Ray paths source only this file, so the snapshot must remain."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "repo"
            repo_root.mkdir()
            env_file = self._write_env_file(work, repo_root)
            values, _ = self._source_and_report(env_file, {})

        self.assertEqual(values["ANTHROPIC_API_KEY"], _INSTALL_KEY)
        self.assertEqual(values["ANTHROPIC_BASE_URL"], _INSTALL_URL)
        self.assertEqual(values["CLAUDE_CODE_OAUTH_TOKEN"], _INSTALL_TOKEN)

    def test_matching_credentials_are_not_announced(self) -> None:
        """No rotation happened, so there is nothing to warn about."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "repo"
            repo_root.mkdir()
            env_file = self._write_env_file(work, repo_root)
            values, stderr = self._source_and_report(env_file, {"ANTHROPIC_API_KEY": _INSTALL_KEY})

        self.assertEqual(values["ANTHROPIC_API_KEY"], _INSTALL_KEY)
        self.assertEqual(stderr, "")

    def test_install_resolved_paths_still_win(self) -> None:
        """Paths are install-time results, not runtime input: the file owns them.

        Losing this would reintroduce the workspace mixup that made the
        installer-written path vars authoritative in the first place.
        """
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "repo"
            repo_root.mkdir()
            env_file = self._write_env_file(work, repo_root)
            values, _ = self._source_and_report(env_file, {"MAGPIE_PATH": "/stale/other-workspace/Magpie"})

        self.assertEqual(values["MAGPIE_PATH"], "/data/.cache/Magpie@abc1234")


if __name__ == "__main__":
    unittest.main()
