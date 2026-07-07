# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Dynamo SSH host-key retry error handling."""

from __future__ import annotations

import subprocess

import pytest


def _host_key_failure() -> subprocess.CompletedProcess:
    """Build a CompletedProcess that looks like an SSH host-key mismatch."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=255,
        stdout="",
        stderr="Host key verification failed.",
    )


@pytest.mark.parametrize(
    "helper_name",
    ["_dynamo_ssh_run_script", "_dynamo_ssh_bash_with_env"],
)
def test_host_key_refresh_runtime_error_returns_original_cp(
    monkeypatch,
    tmp_path,
    helper_name: str,
):
    """A failed known_hosts refresh must not abort the per-pod SSH attempt."""
    from inference_optimizer.multi_node import cli as mn_cli

    state = {"ssh_key_path": str(tmp_path / "key"), "ssh_port": 22}
    (tmp_path / "key").write_text("fake-key")
    first_cp = _host_key_failure()
    run_calls = {"count": 0}

    if helper_name == "_dynamo_ssh_run_script":

        def _fake_run(*_args, **_kwargs):
            run_calls["count"] += 1
            return first_cp

        monkeypatch.setattr(mn_cli.ssh_client, "ssh_run_script", _fake_run)

        def helper():
            return mn_cli._dynamo_ssh_run_script(
                state,
                "10.0.0.1",
                "print('hi')",
                "python3",
                "",
                timeout=30,
            )
    else:

        def _fake_run(*_args, **_kwargs):
            run_calls["count"] += 1
            return first_cp

        monkeypatch.setattr(mn_cli.ssh_client, "ssh_run_bash_with_env", _fake_run)

        def helper():
            return mn_cli._dynamo_ssh_bash_with_env(
                state,
                "10.0.0.1",
                "echo hi",
                None,
                timeout=30,
            )

    monkeypatch.setattr(mn_cli, "_dynamo_known_hosts_path", lambda _state: tmp_path / "known_hosts")
    monkeypatch.setattr(
        mn_cli,
        "_refresh_dynamo_known_hosts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ssh-keyscan produced no keys")),
    )
    monkeypatch.setattr(mn_cli, "_save_state", lambda _state: None)

    cp = helper()
    assert cp is first_cp
    assert run_calls["count"] == 1


def test_host_key_refresh_success_retries_once(monkeypatch, tmp_path):
    """A successful refresh should still perform the single SSH retry."""
    from inference_optimizer.multi_node import cli as mn_cli

    state = {"ssh_key_path": str(tmp_path / "key"), "ssh_port": 22}
    (tmp_path / "key").write_text("fake-key")
    first_cp = _host_key_failure()
    second_cp = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
    outcomes = [first_cp, second_cp]

    def _fake_run(*_args, **_kwargs):
        return outcomes.pop(0)

    refreshed = tmp_path / "known_hosts_refreshed"
    refreshed.write_text("refreshed\n")

    monkeypatch.setattr(mn_cli.ssh_client, "ssh_run_script", _fake_run)
    monkeypatch.setattr(mn_cli, "_dynamo_known_hosts_path", lambda _state: tmp_path / "known_hosts")
    monkeypatch.setattr(mn_cli, "_refresh_dynamo_known_hosts", lambda *_a, **_k: refreshed)
    saved = {"called": False}
    monkeypatch.setattr(mn_cli, "_save_state", lambda _state: saved.update(called=True))

    cp = mn_cli._dynamo_ssh_run_script(
        state,
        "10.0.0.1",
        "print('hi')",
        "python3",
        "",
        timeout=30,
    )
    assert cp.returncode == 0
    assert saved["called"] is True
