# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for Dynamo SSH known_hosts management."""

from __future__ import annotations

from hyperloom.inference_optimizer.multi_node._internal import ssh_known_hosts


def test_is_host_key_error_detects_verification_failure():
    assert ssh_known_hosts.is_host_key_error("Host key verification failed.")
    assert not ssh_known_hosts.is_host_key_error("connection refused")


def test_refresh_known_hosts_appends_scanned_keys(tmp_path, monkeypatch):
    dest = tmp_path / "mn_ssh" / "known_hosts"

    def _fake_scan(argv, **kwargs):
        class _Proc:
            returncode = 0
            stdout = "|1|abc| ssh-ed25519 AAAATEST\n"
            stderr = ""

        return _Proc()

    monkeypatch.setattr(ssh_known_hosts.subprocess, "run", _fake_scan)
    out = ssh_known_hosts.refresh_known_hosts([("10.0.0.1", 2222)], dest)
    assert out == dest
    assert "ssh-ed25519" in dest.read_text(encoding="utf-8")
