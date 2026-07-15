# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SSH host-key retry for the Infera multi-node control plane."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node import cli as mn_cli


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.parametrize(
    "helper_name",
    ["_infera_ssh_run_script", "_infera_ssh_bash_with_env"],
)
def test_host_key_mismatch_refreshes_known_hosts_and_retries(
    tmp_path, monkeypatch, helper_name
):
    state = {"ssh_key_path": str(tmp_path / "id_ed25519"), "ssh_port": 2222}
    (tmp_path / "id_ed25519").write_text("fake-key\n", encoding="utf-8")
    calls = {"n": 0}
    refreshed = tmp_path / "known_hosts_refreshed"

    def _fake_run(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeCompleted(255, stderr="WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!")
        return _FakeCompleted(0, stdout='{"status":"ok"}')

    monkeypatch.setattr(mn_cli.ssh_client.subprocess, "run", _fake_run)
    monkeypatch.setattr(mn_cli, "_infera_known_hosts_path", lambda _state: tmp_path / "known_hosts")
    monkeypatch.setattr(
        mn_cli,
        "_refresh_infera_known_hosts",
        lambda *_a, **_k: refreshed,
    )

    if helper_name == "_infera_ssh_run_script":
        cp = mn_cli._infera_ssh_run_script(
            state,
            "10.0.0.1",
            "echo ok",
            "python3",
            "--help",
            timeout=30,
        )
    else:
        cp = mn_cli._infera_ssh_bash_with_env(
            state,
            "10.0.0.1",
            "echo ok",
            {},
            timeout=30,
        )
    assert cp.returncode == 0
    assert calls["n"] == 2
