# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK-dispatch correctness regressions.

* A ``backends`` payload supplied as a JSON list (``["forge"]``) must be
  serialized into a bare ``--backends forge`` token, never ``str(["forge"])`` →
  ``"['forge']"`` (which the kernel-agent validator rightly rejects).
* A non-GEAK attempt (e.g. a Claude subprocess that times out) must not
  be silently bucketed under the GEAK invocation lane; the backend that ran is
  stamped on the result and an unattributable failure never defaults to GEAK.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder import instrument
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


def test_record_kernel_invocations_claude_timeout_is_not_recorded_as_kernel_lane(tmp_path: Path):
    # A backend-stamped (claude) no-attempts failure must not contaminate
    # kernel backend lanes.
    result = {
        "kernel_id": "k008",
        "status": "failed",
        "error_class": "subprocess_timeout",
        "error": "TimeoutExpired ... --backends claude",
        "backend": "claude",
        "attempts": [],
    }
    instrument.record_kernel_invocations(tmp_path, result)
    sections = assemble_parts(tmp_path)
    assert not sections.get("geak_invocations"), "must NOT contaminate the geak lane"
    assert not sections.get("forge_invocations"), "must NOT contaminate the forge lane"


def test_record_kernel_invocations_unknown_backend_is_not_geak(tmp_path: Path):
    # A pre-dispatch failure with no resolvable backend must NOT be fabricated as GEAK.
    result = {
        "kernel_id": "k001",
        "status": "failed",
        "error_class": "kernel_agent_root_missing",
        "error": "kernel-agent root missing",
        "attempts": [],
    }
    instrument.record_kernel_invocations(tmp_path, result)
    sections = assemble_parts(tmp_path)
    assert not sections.get("geak_invocations"), "unknown backend must not default to geak"
    assert not sections.get("forge_invocations")


def test_record_kernel_invocations_geak_still_records(tmp_path: Path):
    # A real GEAK pre-dispatch failure still records on the geak lane.
    result = {
        "kernel_id": "k001",
        "status": "failed",
        "error_class": "non_reusable_kernel",
        "error": "non reusable",
        "backend": "geak",
        "attempts": [],
    }
    instrument.record_kernel_invocations(tmp_path, result)
    sections = assemble_parts(tmp_path)
    assert sections.get("geak_invocations"), "geak failure must record on the geak lane"


# --------------------------------------------------------------------------- #
# per-kernel ladder budget (option 1)
# --------------------------------------------------------------------------- #
def test_kernel_ladder_budget_sec_priority(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_KERNEL_BUDGET_MIN", raising=False)
    # payload override wins.
    assert krh._kernel_ladder_budget_sec({"kernel_budget_min": 10}) == 10 * 60 + 180
    # env is next in priority.
    monkeypatch.setenv("KERNEL_OPT_KERNEL_BUDGET_MIN", "5")
    assert krh._kernel_ladder_budget_sec({}) == 5 * 60 + 180


@pytest.mark.asyncio
async def test_ladder_continues_to_fallback_after_timeout(tmp_path: Path, monkeypatch):
    # A timed-out backend does not abort the ladder; the next backend still runs.
    calls: list[str] = []

    async def _fake_single(payload, *, session_dir, timeout_override_sec=None):
        calls.append(payload["backends"])
        return {"status": "failed", "backend": payload["backends"], "error_class": "subprocess_timeout"}

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    deadline = time.monotonic() + 10_000  # plenty of budget left
    best, attempts = await krh._run_backend_ladder(
        {},
        {"kernel_id": "k1", "source_file": "x"},
        "k1",
        ["forge", "claude"],
        session_dir=tmp_path,
        deadline=deadline,
    )
    assert calls == ["forge", "claude"], "fallback must run after a forge timeout"
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_ladder_caps_backend_timeout_to_remaining_budget(tmp_path: Path, monkeypatch):
    # Each backend's subprocess timeout is capped to the remaining per-kernel budget.
    seen: list[int | None] = []

    async def _fake_single(payload, *, session_dir, timeout_override_sec=None):
        seen.append(timeout_override_sec)
        return {"status": "failed", "backend": payload["backends"]}

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    deadline = time.monotonic() + 300  # ~5 min left
    await krh._run_backend_ladder(
        {},
        {"kernel_id": "k1"},
        "k1",
        ["forge"],
        session_dir=tmp_path,
        deadline=deadline,
    )
    assert seen[0] is not None
    assert 250 <= seen[0] <= 300


@pytest.mark.asyncio
async def test_ladder_skips_backends_when_budget_exhausted(tmp_path: Path, monkeypatch):
    # When the per-kernel budget is already spent, remaining backends are skipped.
    calls: list[str] = []

    async def _fake_single(payload, *, session_dir, timeout_override_sec=None):
        calls.append(payload["backends"])
        return {"status": "failed", "backend": payload["backends"]}

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    deadline = time.monotonic() - 1  # budget already exhausted
    best, attempts = await krh._run_backend_ladder(
        {},
        {"kernel_id": "k1"},
        "k1",
        ["forge"],
        session_dir=tmp_path,
        deadline=deadline,
    )
    assert calls == [], "no backend should run once the budget is exhausted"
    assert best is None
    assert attempts == []
