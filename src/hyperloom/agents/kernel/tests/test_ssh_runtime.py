# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``backends/ssh_runtime.py`` (Dynamo multi-node GEAK SSH).

Guards the isolation switch (``ssh_placement_active``), the MN_SSH_* target
resolution, the unconfigured-guard error contract, and the pod-stdout JSON
extraction that tolerates leading log noise + nested objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR / "backends"))
sys.path.insert(0, str(_TOOLS_DIR))

import ssh_runtime  # noqa: E402


def test_ssh_placement_active_only_when_env_is_ssh(monkeypatch):
    monkeypatch.setenv("KERNEL_AGENT_GPU_PLACEMENT", "ssh")
    assert ssh_runtime.ssh_placement_active() is True
    monkeypatch.setenv("KERNEL_AGENT_GPU_PLACEMENT", "SSH")  # case-insensitive
    assert ssh_runtime.ssh_placement_active() is True
    monkeypatch.setenv("KERNEL_AGENT_GPU_PLACEMENT", "ray")
    assert ssh_runtime.ssh_placement_active() is False
    monkeypatch.delenv("KERNEL_AGENT_GPU_PLACEMENT", raising=False)
    assert ssh_runtime.ssh_placement_active() is False


def test_ssh_target_parses_env_with_default_port(monkeypatch):
    monkeypatch.setenv("MN_SSH_HOST", "10.0.0.5")
    monkeypatch.setenv("MN_SSH_KEY", "/tmp/k")
    monkeypatch.delenv("MN_SSH_PORT", raising=False)
    host, port, key = ssh_runtime.ssh_target()
    assert host == "10.0.0.5"
    assert port == ssh_runtime.DEFAULT_SSH_PORT == 2222
    assert key == "/tmp/k"
    monkeypatch.setenv("MN_SSH_PORT", "2345")
    assert ssh_runtime.ssh_target()[1] == 2345


def test_ssh_unconfigured_returns_error_when_host_or_key_missing(monkeypatch):
    monkeypatch.delenv("MN_SSH_HOST", raising=False)
    monkeypatch.delenv("MN_SSH_KEY", raising=False)
    err = ssh_runtime._ssh_unconfigured()
    assert err is not None and err["returncode"] == 1
    assert "MN_SSH_HOST" in err["stderr_tail"]
    # Both present -> None (configured).
    monkeypatch.setenv("MN_SSH_HOST", "h")
    monkeypatch.setenv("MN_SSH_KEY", "/k")
    assert ssh_runtime._ssh_unconfigured() is None
    # Host without key still unconfigured.
    monkeypatch.delenv("MN_SSH_KEY", raising=False)
    assert ssh_runtime._ssh_unconfigured() is not None


def test_extract_last_json_simple_object():
    assert ssh_runtime._extract_last_json('{"returncode": 0}') == {"returncode": 0}


def test_extract_last_json_skips_leading_log_noise_and_keeps_nested():
    text = (
        "2026-06-12 some pod log line\n"
        "another line\n"
        '{"returncode": 0, "gpu_ids": "0,1", "nested": {"a": 1, "b": [2, 3]}}\n'
    )
    out = ssh_runtime._extract_last_json(text)
    assert out["returncode"] == 0
    assert out["nested"] == {"a": 1, "b": [2, 3]}


def test_extract_last_json_picks_last_object_when_multiple():
    text = '{"first": 1}\nmid\n{"second": 2}'
    assert ssh_runtime._extract_last_json(text) == {"second": 2}


def test_extract_last_json_none_on_empty_or_no_brace_or_malformed():
    assert ssh_runtime._extract_last_json("") is None
    assert ssh_runtime._extract_last_json("no json here") is None
    assert ssh_runtime._extract_last_json('{"a": }') is None  # invalid JSON


# -- GEAK pod-runner defaults ----
import os as _os  # noqa: E402
import stat as _stat  # noqa: E402
import subprocess as _sp  # noqa: E402


def _run_geak_pod_runner(tmp_path, extra_env):
    """Execute the real GEAK _POD_RUNNER with a fake `geak` on PATH; return its JSON dict.

    The fake geak echoes its argv + GEAK_SKIP_PROFILE so the test can assert both
    the forwarded ``--target`` and the profiler default the pod runner sets.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "geak"
    fake.write_text('#!/usr/bin/env bash\necho "ARGV:$@"\necho "SKIP=$GEAK_SKIP_PROFILE"\nexit 0\n')
    fake.chmod(fake.stat().st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)
    cfg = tmp_path / "local.yaml"
    cfg.write_text("model_class: litellm\n")
    prompt = tmp_path / "p.md"
    prompt.write_text("opt this kernel")
    outdir = tmp_path / "out"
    outdir.mkdir()
    runner = tmp_path / "pod_runner.py"
    runner.write_text(ssh_runtime._POD_RUNNER)
    env = dict(_os.environ)
    env["PATH"] = f"{bindir}:{env.get('PATH','')}"
    env["GEAK_CONFIG"] = str(cfg)
    env.pop("GEAK_SCORE_TARGET", None)
    env.pop("GEAK_SKIP_PROFILE", None)
    env.update(extra_env)
    proc = _sp.run(
        [sys.executable, str(runner), "--prompt", str(prompt), "--output", str(outdir),
         "--num-gpus", "1", "--timeout-s", "30"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    return ssh_runtime._extract_last_json(proc.stdout) or {}


def test_geak_pod_runner_defaults_to_kernel_target_and_skip_profile(tmp_path):
    out = _run_geak_pod_runner(tmp_path, extra_env={})
    cmd = out.get("cmd") or []
    assert "--target" in cmd and cmd[cmd.index("--target") + 1] == "kernel", cmd
    # pod runner set GEAK_SKIP_PROFILE=1 by default -> visible to the geak child
    assert "SKIP=1" in (out.get("stdout_tail") or ""), out.get("stdout_tail")


def test_geak_pod_runner_respects_explicit_overrides(tmp_path):
    out = _run_geak_pod_runner(tmp_path, extra_env={"GEAK_SCORE_TARGET": "wall", "GEAK_SKIP_PROFILE": "0"})
    cmd = out.get("cmd") or []
    assert "--target" in cmd and cmd[cmd.index("--target") + 1] == "wall", cmd
    assert "SKIP=0" in (out.get("stdout_tail") or ""), out.get("stdout_tail")
