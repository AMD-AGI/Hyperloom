# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK-dispatch correctness regressions.

* A ``backends`` payload supplied as a JSON list (``["forge"]``) must be
  serialized into a bare ``--backends forge`` token, never ``str(["forge"])`` →
  ``"['forge']"`` (which the kernel-agent validator rightly rejects).
* A non-GEAK attempt (e.g. a Claude subprocess that times out) must not
  be silently bucketed under the GEAK lane the collectors project; the backend
  that ran is stamped on the row and an unattributable failure never defaults
  to GEAK.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh


# --------------------------------------------------------------------------- #
# --backends serialization
# --------------------------------------------------------------------------- #
def test_backends_cli_arg_list_is_comma_joined_bare_names():
    assert krh._backends_cli_arg(["forge"]) == "forge"
    assert krh._backends_cli_arg(["forge", "claude"]) == "forge,claude"
    assert krh._backends_cli_arg("forge") == "forge"
    assert krh._backends_cli_arg("forge,claude") == "forge,claude"
    assert krh._backends_cli_arg(None) == ""
    assert krh._backends_cli_arg(["forge"]) != "['forge']"  # never the list repr


def _bypass_single_kernel_guards(monkeypatch):
    """Disable the pre-dispatch validators so a unit test reaches cmd-building."""
    monkeypatch.setattr(krh, "_validate_reusable_native_kernel", lambda payload: None)
    monkeypatch.setattr(
        krh,
        "_validate_kernel_shape_and_paths",
        lambda payload, *, session_dir: None,
    )
    monkeypatch.setattr(krh, "_kernel_agent_root_error", lambda: "")


@pytest.mark.asyncio
async def test_run_optimization_single_serializes_list_backends(tmp_path: Path, monkeypatch):
    # A list backends payload reaches the subprocess as bare "forge", not "['forge']".
    _bypass_single_kernel_guards(monkeypatch)
    captured: dict[str, list[str]] = {}

    async def _fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, '{"status": "ok", "kernel_id": "k001"}', ""

    monkeypatch.setattr(krh, "_run_subprocess", _fake_run_subprocess)

    payload = {"kernel_id": "k001", "backends": ["forge"], "_single_kernel": True}
    await krh._run_optimization_single(payload, session_dir=tmp_path)

    cmd = captured["cmd"]
    assert "--backends" in cmd
    val = cmd[cmd.index("--backends") + 1]
    assert val == "forge"
    assert val != "['forge']"


# --------------------------------------------------------------------------- #
# timeout backend attribution
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_timeout_result_is_attributed_to_dispatched_backend(tmp_path: Path, monkeypatch):
    # A claude subprocess that overruns the timeout is shaped into a failed
    # result carrying backend="claude", not left for the GEAK fallback to claim.
    _bypass_single_kernel_guards(monkeypatch)

    async def _fake_timeout(cmd, *, timeout_sec):
        raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=timeout_sec)

    monkeypatch.setattr(krh, "_run_subprocess", _fake_timeout)

    payload = {"kernel_id": "k008", "backends": "claude", "_single_kernel": True}
    result = await krh._run_optimization_single(payload, session_dir=tmp_path)

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert result["error_class"] == "subprocess_timeout"
    assert result["backend"] == "claude"










# --------------------------------------------------------------------------- #
# single per-kernel backend architecture
# --------------------------------------------------------------------------- #
def test_removed_backend_ladder_helpers_stay_removed():
    assert not hasattr(krh, "_run_backend_ladder")
    assert not hasattr(krh, "_kernel_ladder_budget_sec")


@pytest.mark.asyncio
async def test_removed_ladder_budget_is_rejected_instead_of_silently_ignored(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_KERNEL_BUDGET_MIN", raising=False)
    result = await krh.run_optimization_handler({"kernel_budget_min": 1}, session_dir=tmp_path)
    assert result["status"] == "failed"
    assert result["error_class"] == "unsupported_option"


@pytest.mark.asyncio
async def test_removed_ladder_budget_env_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_KERNEL_BUDGET_MIN", "1")
    result = await krh.run_optimization_handler({}, session_dir=tmp_path)
    assert result["status"] == "failed"
    assert result["error_class"] == "unsupported_option"


@pytest.mark.asyncio
async def test_backend_sequence_dispatches_forge_once(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    calls: list[dict] = []

    async def _fake_single(payload, *, session_dir):
        calls.append(payload)
        return {"status": "failed", "backend": payload["backends"]}

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    result = await krh._run_kernel_backend_sequence(
        {},
        {"kernel_id": "k1", "source_file": "x"},
        session_dir=tmp_path,
    )

    assert [call["backends"] for call in calls] == ["forge"]
    assert result["batch_kernel_id"] == "k1"
    assert "backend_fallback_attempts" not in result
