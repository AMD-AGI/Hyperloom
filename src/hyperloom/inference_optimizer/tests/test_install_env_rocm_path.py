#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the ROCM_PATH written into kernel-agent runtime env.

Ray's native AMD-GPU detector (ray._private.thirdparty.pyamdsmi) looks for
$ROCM_PATH/lib/librocm_smi64.so, defaulting ROCM_PATH=/opt/rocm when unset. A
rocm-sdk wheel venv never populates /opt/rocm, so without this the cluster
silently reports 0 GPUs. These tests cover the resolution being wired into
write_env_file's persistence, not the GPU-detection consequence itself.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ._kernel_tools_harness import sourceable_kernel_tools


class WriteEnvFileRocmPathTest(unittest.TestCase):
    """``write_env_file`` must persist ROCM_PATH when it resolves, and never clobber an operator's own."""

    def _run_write_env_file(
        self,
        workdir: Path,
        *,
        repo_root: Path,
        python_stub: Path,
        preexisting_rocm_path: str | None = None,
    ) -> tuple[str, str]:
        sourceable = sourceable_kernel_tools(workdir)
        runtime_dir = workdir / "runtime"
        kernel_agent_env = runtime_dir / "kernel-agent.env.sh"
        dotenv = repo_root / ".env"

        script = f"""
set -euo pipefail
export ANTHROPIC_API_KEY=ak-test-key
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export REPO_ROOT={repo_root!s}
export USER_DATA_PATH={workdir!s}
export HYPERLOOM_RUNTIME_DIR={runtime_dir!s}
export KERNEL_AGENT_ENV={kernel_agent_env!s}
export PYTHON={python_stub!s}
"""
        if preexisting_rocm_path is not None:
            script += f"export ROCM_PATH={preexisting_rocm_path}\n"
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

    @staticmethod
    def _rocm_sdk_wheel_python_stub(tmp: Path, root: Path) -> Path:
        """A fake interpreter whose ``-m rocm_sdk path --root`` prints ``root``."""
        stub = tmp / "fake-python-wheel"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "-m" ] && [ "$2" = "rocm_sdk" ]; then echo ' + str(root) + "; exit 0; fi\n"
            'exec python3 "$@"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        return stub

    @staticmethod
    def _system_rocm_python_stub(tmp: Path) -> Path:
        """A fake interpreter with no ``rocm_sdk`` module -- the system-ROCm-install case."""
        stub = tmp / "fake-python-system"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "-m" ] && [ "$2" = "rocm_sdk" ]; then echo "No module named rocm_sdk" >&2; exit 1; fi\n'
            'exec python3 "$@"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        return stub

    def test_resolved_rocm_sdk_wheel_root_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            wheel_root = work / "venv" / "lib" / "python3.12" / "site-packages" / "_rocm_sdk_devel"
            wheel_root.mkdir(parents=True)
            python_stub = self._rocm_sdk_wheel_python_stub(work, wheel_root)

            env_text, dotenv_text = self._run_write_env_file(work, repo_root=repo_root, python_stub=python_stub)

        rocm_path_line = next(
            (ln for ln in env_text.splitlines() if "export ROCM_PATH=" in ln),
            "",
        )
        self.assertTrue(rocm_path_line, f"no ROCM_PATH export in env file:\n{env_text}")
        self.assertIn(str(wheel_root), rocm_path_line)
        self.assertIn(str(wheel_root), dotenv_text)

    def test_operator_set_rocm_path_is_not_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            wheel_root = work / "venv" / "_rocm_sdk_devel"
            wheel_root.mkdir(parents=True)
            python_stub = self._rocm_sdk_wheel_python_stub(work, wheel_root)

            env_text, _ = self._run_write_env_file(
                work,
                repo_root=repo_root,
                python_stub=python_stub,
                preexisting_rocm_path="/opt/rocm",
            )

        rocm_path_line = next(
            (ln for ln in env_text.splitlines() if "export ROCM_PATH=" in ln),
            "",
        )
        self.assertIn("/opt/rocm", rocm_path_line)
        self.assertNotIn(str(wheel_root), rocm_path_line)

    def test_system_rocm_install_leaves_rocm_path_unset(self) -> None:
        """No ``rocm_sdk`` package (a traditional /opt/rocm install) must no-op, not error."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            repo_root = work / "target-install"
            repo_root.mkdir()
            python_stub = self._system_rocm_python_stub(work)

            env_text, dotenv_text = self._run_write_env_file(work, repo_root=repo_root, python_stub=python_stub)

        self.assertNotIn("ROCM_PATH", env_text)
        self.assertNotIn("ROCM_PATH", dotenv_text)


if __name__ == "__main__":
    unittest.main()
