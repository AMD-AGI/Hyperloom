# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Infera multi-node SSH control plane (``ssh_client``).

These guard the command-construction / credential-forwarding logic that is the
sole channel for reaching the Infera idle pods: argv shape, base64 script
shipping, per-variant env injection (e.g. MORI_* MoE-dispatch tuning), and the
stdin-only secret path. All tests stub ``subprocess.run`` so nothing is spawned.
"""

from __future__ import annotations

import base64
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


def test_ssh_run_builds_expected_argv(monkeypatch, known_hosts):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeCompleted(0, "out", "")

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    cp = ssh_client.ssh_run(
        "10.0.0.1",
        "echo hi",
        key_path="/k/id",
        known_hosts=known_hosts,
        port=2345,
        timeout=42,
    )
    assert cp.returncode == 0
    argv = captured["argv"]
    assert argv[0] == "ssh"
    assert "-i" in argv and argv[argv.index("-i") + 1] == "/k/id"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "2345"
    assert "root@10.0.0.1" in argv
    assert argv[-3:] == ["bash", "-lc", shlex.quote("echo hi")]
    assert "StrictHostKeyChecking=yes" in argv
    assert str(known_hosts) in " ".join(argv)
    assert captured["kwargs"]["timeout"] == 42
    assert captured["kwargs"]["capture_output"] is True


def test_ssh_run_script_base64_wraps_and_prepends_env(monkeypatch, known_hosts):
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
        key_path="/k",
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


def test_ssh_run_script_no_env_has_no_prefix(monkeypatch, known_hosts):
    captured = {}
    monkeypatch.setattr(
        ssh_client.subprocess,
        "run",
        lambda argv, **kw: captured.setdefault("argv", argv) or _FakeCompleted(0),
    )
    ssh_client.ssh_run_script("h", "x=1", "python3", "", key_path="/k", known_hosts=known_hosts)
    remote_cmd = captured["argv"][-1]
    assert "&& python3 /tmp/mn_infera_launch" in remote_cmd


def test_ssh_run_bash_with_env_pipes_secrets_via_stdin(monkeypatch, known_hosts):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["input"] = kwargs.get("input")
        return _FakeCompleted(0)

    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    ssh_client.ssh_run_bash_with_env(
        "h",
        "echo body",
        {"OOB_API_KEY": "secret-123"},
        key_path="/k",
        known_hosts=known_hosts,
    )
    assert captured["argv"][-2:] == ["bash", "-s"]
    assert "export OOB_API_KEY=secret-123" in captured["input"]
    assert "echo body" in captured["input"]
    assert "set -uo pipefail" in captured["input"]
    assert not any("secret-123" in a for a in captured["argv"])


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


def test_generate_session_keypair_reloads_reused_encrypted_key(tmp_path, monkeypatch):
    priv = tmp_path / "mn_id_ed25519"
    pub = tmp_path / "mn_id_ed25519.pub"
    passfile = tmp_path / "mn_id_ed25519.pass"
    priv.write_text("PRIV", encoding="utf-8")
    pub.write_text("ssh-ed25519 AAAA reuse\n", encoding="utf-8")
    passfile.write_text("cached-passphrase", encoding="utf-8")
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ssh-agent":
            return _FakeCompleted(
                0,
                "SSH_AUTH_SOCK=/tmp/hyperloom-agent.sock; export SSH_AUTH_SOCK;\n"
                "SSH_AGENT_PID=12345; export SSH_AGENT_PID;\n",
                "",
            )
        if argv[0] == "ssh-add":
            assert kwargs["env"]["SSH_ASKPASS_REQUIRE"] == "force"
            assert kwargs["env"]["SSH_AUTH_SOCK"] == "/tmp/hyperloom-agent.sock"
            return _FakeCompleted(0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    out_priv, pub_str = ssh_client.generate_session_keypair(tmp_path)
    assert out_priv == priv
    assert pub_str == "ssh-ed25519 AAAA reuse"
    assert [c[0][0] for c in calls] == ["ssh-agent", "ssh-add"]
    assert passfile.read_text(encoding="utf-8") == "cached-passphrase"
    assert not (tmp_path / "mn_ssh_askpass.sh").exists()


def test_generate_session_keypair_uses_passphrase_and_agent(tmp_path, monkeypatch):
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ssh-keygen":
            passphrase = argv[argv.index("-N") + 1]
            assert passphrase
            assert passphrase != ""
            (tmp_path / "mn_id_ed25519").write_text("PRIVATE", encoding="utf-8")
            (tmp_path / "mn_id_ed25519.pub").write_text(
                "ssh-ed25519 AAAA generated\n",
                encoding="utf-8",
            )
            return _FakeCompleted(0, "", "")
        if argv[0] == "ssh-agent":
            return _FakeCompleted(
                0,
                "SSH_AUTH_SOCK=/tmp/hyperloom-agent.sock; export SSH_AUTH_SOCK;\n"
                "SSH_AGENT_PID=12345; export SSH_AGENT_PID;\n",
                "",
            )
        if argv[0] == "ssh-add":
            assert kwargs["env"]["SSH_ASKPASS_REQUIRE"] == "force"
            assert kwargs["env"]["SSH_AUTH_SOCK"] == "/tmp/hyperloom-agent.sock"
            return _FakeCompleted(0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(ssh_client.subprocess, "run", _run)
    out_priv, pub_str = ssh_client.generate_session_keypair(tmp_path)
    assert out_priv == tmp_path / "mn_id_ed25519"
    assert pub_str == "ssh-ed25519 AAAA generated"
    assert [c[0][0] for c in calls] == ["ssh-keygen", "ssh-agent", "ssh-add"]
    assert (tmp_path / "mn_id_ed25519.pass").exists()
    assert not (tmp_path / "mn_ssh_askpass.sh").exists()


def test_generate_session_keypair_raises_on_keygen_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ssh_client.subprocess,
        "run",
        lambda *a, **kw: _FakeCompleted(1, "", "boom"),
    )
    with pytest.raises(RuntimeError, match="ssh-keygen failed"):
        ssh_client.generate_session_keypair(tmp_path)
