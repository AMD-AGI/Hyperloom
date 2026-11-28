# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-key scanning and host-key error detection of the multi-node SSH layer."""

from __future__ import annotations

import pytest

from inference_suite.multinode import SshError, is_host_key_error, keyscan
from inference_suite.multinode.ssh import SshResult


def _runner(returncode: int, stdout: str, stderr: str = ""):
    """A runner that returns ``stdout`` and records the argv it is given."""
    calls: list[list[str]] = []

    def run(argv, stdin, timeout):
        calls.append(list(argv))
        return SshResult(returncode, stdout, stderr)

    return run, calls


def test_keyscan_returns_the_hashed_keys_with_a_trailing_newline():
    run, calls = _runner(0, "|1|abc| ssh-ed25519 AAAATEST")
    assert keyscan("10.0.0.1", 2222, run) == "|1|abc| ssh-ed25519 AAAATEST\n"
    assert calls == [["ssh-keyscan", "-p", "2222", "-H", "10.0.0.1"]]


def test_keyscan_refuses_a_host_without_a_key():
    run, _ = _runner(1, "")
    with pytest.raises(SshError, match="no host key for 10.0.0.1:22"):
        keyscan("10.0.0.1", 22, run)


def test_is_host_key_error_detects_a_changed_key():
    assert is_host_key_error("Host key verification failed.")
    assert not is_host_key_error("connection refused")
