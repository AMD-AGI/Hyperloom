# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for Infera SSH known_hosts management."""

from __future__ import annotations

from hyperloom.inference_optimizer.multi_node._internal import ssh_known_hosts


def test_refresh_known_hosts_appends_scanned_keys(tmp_path, monkeypatch):
    dest = tmp_path / "mn_ssh" / "known_hosts"
    monkeypatch.setattr(ssh_known_hosts, "keyscan", lambda host, port: "|1|abc| ssh-ed25519 AAAATEST\n")
    out = ssh_known_hosts.refresh_known_hosts([("10.0.0.1", 2222)], dest)
    assert out == dest
    assert "ssh-ed25519" in dest.read_text(encoding="utf-8")
