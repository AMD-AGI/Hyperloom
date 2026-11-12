"""Regression tests for Windows-only process and socket behavior."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import unittest

import psutil


@unittest.skipUnless(sys.platform == "win32", "Windows-only regression")
class WindowsRuntimeGuardTests(unittest.TestCase):
    def test_kill_process_tree_uses_windows_supported_termination(self) -> None:
        child_code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True); time.sleep(60)"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline().strip())

        try:
            from vllm.utils.system_utils import kill_process_tree

            kill_process_tree(parent.pid)
            parent.wait(timeout=10)
            for _ in range(50):
                if not psutil.pid_exists(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(psutil.pid_exists(child_pid))
        finally:
            parent.stdout.close()
            if parent.poll() is None:
                parent.kill()
            if psutil.pid_exists(child_pid):
                try:
                    psutil.Process(child_pid).kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    @unittest.skipIf(hasattr(socket, "AF_UNIX"), "Python supports Unix sockets")
    def test_uds_reports_clear_windows_error(self) -> None:
        from vllm.entrypoints.openai.api_server import create_server_unix_socket

        with self.assertRaisesRegex(RuntimeError, "--uds is not supported"):
            create_server_unix_socket("unused.sock")


if __name__ == "__main__":
    unittest.main()
