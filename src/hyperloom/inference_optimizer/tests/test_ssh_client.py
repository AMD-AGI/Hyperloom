# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the Infera multi-node SSH control plane (``ssh_client``).

These guard the command-construction / credential-forwarding logic that is the
sole channel for reaching the Infera idle pods: argv shape, base64 script
shipping, per-variant env injection (e.g. MORI_* MoE-dispatch tuning), and the
stdin-only secret path. All tests stub ``subprocess.run`` so nothing is spawned.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess

import pytest

from hyperloom.inference_optimizer.multi_node._internal import ssh_client


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def known_hosts(tmp_path):
    kh = tmp_path / "known_hosts"
    kh.write_text("", encoding="utf-8")
    return kh


@pytest.fixture
def key_file(tmp_path):
    k = tmp_path / "mn_id_ed25519"
    k.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\nDUMMY\n")
    return k


def test_ssh_identity_is_the_on_disk_path(key_file):
    """The identity handed to ``ssh -i`` must be the key's own path.

    Regression guard. This used to be a ``/dev/fd/N`` anonymous pipe so the key
    was never a discoverable path on argv, but ssh closes every inherited fd
    above stderr before opening the identity file, so it always reported

        Warning: Identity file /dev/fd/N not accessible: No such file or directory

    and fell through to ``Permission denied (publickey)`` -- which broke every
    Infera restart-server SSH fan-out. Reproduced against OpenSSH_8.9p1 with a
    regular-file fd, a pipe fd, and both the /dev/fd and /proc/self/fd
    spellings: all fail identically, only the plain path authenticates.
    """
    assert ssh_client._ssh_identity(key_file) == str(key_file)
    assert not ssh_client._ssh_identity(key_file).startswith("/dev/fd/")


def test_ssh_identity_does_not_pipe_fds_to_subprocess(monkeypatch, known_hosts, key_file):
    """No ``pass_fds`` may be handed to ssh: it cannot use an inherited fd."""
    captured = {}

    def _run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeCompleted(0, "out", "")

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    ssh_client.ssh_run("10.0.0.1", "echo hi", key_path=key_file, known_hosts=known_hosts)
    assert "pass_fds" not in captured["kwargs"]


def test_ssh_run_builds_expected_argv(monkeypatch, known_hosts, key_file):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeCompleted(0, "out", "")

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    cp = ssh_client.ssh_run(
        "10.0.0.1",
        "echo hi",
        key_path=key_file,
        known_hosts=known_hosts,
        port=2345,
        timeout=42,
    )
    assert cp.returncode == 0
    argv = captured["argv"]
    assert argv[0] == "ssh"
    # Identity must be the key path itself; ssh cannot read an inherited fd.
    assert argv[argv.index("-i") + 1] == str(key_file)
    assert "IdentitiesOnly=yes" in argv
    assert "-p" in argv and argv[argv.index("-p") + 1] == "2345"
    assert "root@10.0.0.1" in argv
    assert argv[-3:] == ["bash", "-lc", shlex.quote("echo hi")]
    assert "StrictHostKeyChecking=yes" in argv
    assert str(known_hosts) in " ".join(argv)
    assert captured["kwargs"]["timeout"] == 42
    assert captured["kwargs"]["capture_output"] is True


def test_ssh_run_script_base64_wraps_and_prepends_env(monkeypatch, known_hosts, key_file):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeCompleted(0)

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    script = "print('multi\nnode')"
    ssh_client.ssh_run_script(
        "h",
        script,
        "python3",
        "--flag v",
        key_path=key_file,
        known_hosts=known_hosts,
        env={"MORI_DISPATCH": "1", "SGLANG_FOO": "a b"},
        remote_path="/tmp/run.py",
    )
    remote_cmd = captured["argv"][-1]
    enc = base64.b64encode(script.encode()).decode()
    assert enc in remote_cmd
    assert "base64 -d" in remote_cmd
    assert "/tmp/run.py" in remote_cmd
    assert "MORI_DISPATCH=1" in remote_cmd
    assert "'a b'" in remote_cmd
    assert "python3" in remote_cmd and "--flag v" in remote_cmd


def test_ssh_run_script_rejects_invalid_env_key(monkeypatch, known_hosts):
    monkeypatch.setattr(ssh_client.subprocess, "run", lambda *a, **kw: _FakeCompleted(0))
    with pytest.raises(ValueError, match="disallowed SSH forward env keys"):
        ssh_client.ssh_run_script(
            "h",
            "x=1",
            "python3",
            "",
            key_path="/k",
            known_hosts=known_hosts,
            env={"BAD KEY": "1"},
        )


def test_ssh_run_script_no_env_has_no_prefix(monkeypatch, known_hosts, key_file):
    captured = {}
    monkeypatch.setattr(
        ssh_client.subprocess,
        "run",
        lambda argv, **kw: captured.setdefault("argv", argv) or _FakeCompleted(0),
    )
    ssh_client.ssh_run_script("h", "x=1", "python3", "", key_path=key_file, known_hosts=known_hosts)
    remote_cmd = captured["argv"][-1]
    assert "&& python3 /tmp/mn_infera_launch" in remote_cmd


def test_ssh_run_bash_with_env_pipes_secrets_via_stdin(monkeypatch, known_hosts, key_file):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["input"] = kwargs.get("input")
        captured["kwargs"] = kwargs
        return _FakeCompleted(0)

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    ssh_client.ssh_run_bash_with_env(
        "h",
        "echo body",
        {"OOB_API_KEY": "secret-123"},
        key_path=key_file,
        known_hosts=known_hosts,
    )
    assert captured["argv"][-2:] == ["bash", "-s"]
    assert "export OOB_API_KEY=secret-123" in captured["input"]
    assert "echo body" in captured["input"]
    assert "set -uo pipefail" in captured["input"]
    assert not any("secret-123" in a for a in captured["argv"])
    # Secrets still stay off argv (they go over stdin); the identity is the key path.
    assert captured["argv"][captured["argv"].index("-i") + 1] == str(key_file)
    assert "pass_fds" not in captured["kwargs"]


def test_probe_ssh_true_on_marker(monkeypatch, known_hosts):
    monkeypatch.setattr(
        ssh_client,
        "ssh_run",
        lambda *a, **kw: _FakeCompleted(0, "mn_ssh_ok\n", ""),
    )
    assert ssh_client.probe_ssh("h", key_path="/k", known_hosts=known_hosts) is True


def test_probe_ssh_false_on_bad_rc(monkeypatch, known_hosts):
    monkeypatch.setattr(
        ssh_client,
        "ssh_run",
        lambda *a, **kw: _FakeCompleted(255, "", "conn refused"),
    )
    assert ssh_client.probe_ssh("h", key_path="/k", known_hosts=known_hosts) is False


def test_probe_ssh_false_on_timeout(monkeypatch, known_hosts):
    def _boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

    monkeypatch.setattr(ssh_client, "ssh_run", _boom)
    assert ssh_client.probe_ssh("h", key_path="/k", known_hosts=known_hosts) is False


def test_generate_session_keypair_idempotent_reuse(tmp_path, monkeypatch):
    priv = tmp_path / "mn_id_ed25519"
    pub = tmp_path / "mn_id_ed25519.pub"
    priv.write_text("PRIV", encoding="utf-8")
    pub.write_text("ssh-ed25519 AAAA reuse\n", encoding="utf-8")

    def _no_keygen(*a, **kw):
        raise AssertionError("ssh-keygen must not run when keys already exist")

    monkeypatch.setattr(ssh_client.subprocess, "run", _no_keygen)
    out_priv, pub_str = ssh_client.generate_session_keypair(tmp_path)
    assert out_priv == priv
    assert pub_str == "ssh-ed25519 AAAA reuse"


def test_generate_session_keypair_uses_unencrypted_key(tmp_path, monkeypatch):
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ssh-keygen":
            assert argv[argv.index("-N") + 1] == ""
            (tmp_path / "mn_id_ed25519").write_text("PRIVATE", encoding="utf-8")
            (tmp_path / "mn_id_ed25519.pub").write_text(
                "ssh-ed25519 AAAA generated\n",
                encoding="utf-8",
            )
            return _FakeCompleted(0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    out_priv, pub_str = ssh_client.generate_session_keypair(tmp_path)
    assert out_priv == tmp_path / "mn_id_ed25519"
    assert pub_str == "ssh-ed25519 AAAA generated"
    assert [c[0][0] for c in calls] == ["ssh-keygen"]
    assert not (tmp_path / "mn_id_ed25519.pass").exists()
    assert not (tmp_path / "mn_ssh_askpass.sh").exists()


def test_generate_session_keypair_raises_on_keygen_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ssh_client.subprocess,
        "run",
        lambda *a, **kw: _FakeCompleted(1, "", "boom"),
    )
    with pytest.raises(RuntimeError, match="ssh-keygen failed"):
        ssh_client.generate_session_keypair(tmp_path)
