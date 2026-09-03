# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One owner writes the state a nomination round produces.

The phase entry runs the handler and then saves its own SharedState in full.
``SharedState.save`` serialises the whole object, so a record the handler wrote
to a second instance loaded from the same directory does not survive that save:
the siblings vanish, SWEEP finds nothing to validate, and the cycle still latches
as a completed pass.

These drive the real chain -- ``_run_kernel_opt_nomination`` -> handler ->
landing -> final save -- with only the forge subprocess faked, because that is
the seam the loss happens across.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState

_AUTO_ENV = "HYPERLOOM_FORGE_NOMINATION_AUTO"


class _Bus:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def append_and_seq(self, message: Any) -> None:
        self.sent.append(message)


def _candidates(session_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = session_dir / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": rows}), encoding="utf-8")
    return path


def _row(kernel_id: str, root: Path) -> dict[str, Any]:
    return {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "gpu_pct": 30.0,
        "source_file": str(root / f"{kernel_id}.py"),
        "reusable_native_kernel": True,
        "skip_reason": "",
    }


def _envelope(patches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patches": patches,
        "nomination": {"candidates_seen": 1, "resolved": 1, "selected": len(patches)},
    }


def _sibling(kernel: str) -> dict[str, Any]:
    return {
        "kernel_name": kernel,
        "patch_path": f"/repo/{kernel}.patch",
        "target_file": f"/repo/{kernel}.py",
        "micro_speedup": 1.4,
    }


@pytest.fixture
def phase(tmp_path, monkeypatch):
    """A KERNEL phase wired to a real SharedState, with forge faked."""
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    monkeypatch.setenv(_AUTO_ENV, "1")
    trace = tmp_path / "decode.trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidates = _candidates(tmp_path, [_row("k001", tmp_path)])

    state = SharedState.load_or_init(tmp_path)
    state.max_minutes = 600.0
    state.last_profile_trace = str(trace)
    state.last_trace_analyze = {"candidates_path": str(candidates)}
    state.save(tmp_path)

    obj = KernelPhase.__new__(KernelPhase)
    object.__setattr__(obj, "shared_state", state)
    object.__setattr__(obj, "session_dir", tmp_path)
    object.__setattr__(obj, "bus", _Bus())
    obj._record_kernel_opt_partial = lambda _result: None
    obj._record_kernel_opt_dispatch_skip = lambda _reason: None
    return obj


def _fake_forge(monkeypatch, envelope: dict[str, Any]) -> None:
    from hyperloom.agents.kernel.tools.backends import forge_submit

    monkeypatch.setattr(forge_submit, "submit_auto", lambda **_: envelope)


def _queued_on_disk(session_dir: Path) -> list[str]:
    payload = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    return sorted(
        str(record.get("kernel_id") or "") for record in (payload.get("pending_kernel_integrations") or {}).values()
    )


@pytest.mark.asyncio
async def test_a_nominated_sibling_survives_the_phase_save(phase, monkeypatch):
    """The whole point of the round: SWEEP must find the patch on disk."""
    _fake_forge(monkeypatch, _envelope([_sibling("paged_attention_v1")]))

    await phase._run_kernel_opt_nomination()

    assert _queued_on_disk(phase.session_dir) == ["paged_attention_v1"]


@pytest.mark.asyncio
async def test_every_sibling_of_one_round_survives(phase, monkeypatch):
    """A batch lands as a set, so a partial survival is still a loss."""
    _fake_forge(monkeypatch, _envelope([_sibling("attn"), _sibling("rmsnorm")]))

    await phase._run_kernel_opt_nomination()

    assert _queued_on_disk(phase.session_dir) == ["attn", "rmsnorm"]


@pytest.mark.asyncio
async def test_the_queued_count_is_reported_from_what_was_queued(phase, monkeypatch):
    """The reported figure has to come from the state that was persisted."""
    _fake_forge(monkeypatch, _envelope([_sibling("attn")]))

    await phase._run_kernel_opt_nomination()

    (message,) = phase.bus.sent
    assert message.payload["result"]["queued"] == 1


@pytest.mark.asyncio
async def test_the_live_instance_and_the_disk_agree(phase, monkeypatch):
    """A record on disk but not on the live object is the same defect mirrored."""
    _fake_forge(monkeypatch, _envelope([_sibling("attn")]))

    await phase._run_kernel_opt_nomination()

    live = sorted(
        str(record.get("kernel_id") or "") for record in (phase.shared_state.pending_kernel_integrations or {}).values()
    )
    assert live == _queued_on_disk(phase.session_dir) == ["attn"]


@pytest.mark.asyncio
async def test_an_empty_nomination_queues_nothing_and_still_reports(phase, monkeypatch):
    """A clean empty pass is valid; it must not invent a record."""
    _fake_forge(monkeypatch, _envelope([]))

    await phase._run_kernel_opt_nomination()

    assert _queued_on_disk(phase.session_dir) == []
    (message,) = phase.bus.sent
    assert message.payload["result"]["status"] == "complete"


def _latched_cycle(state: Any) -> Any:
    """The latched cycle as stored, so an unset latch is not read as cycle 0."""
    return getattr(state, "kernel_auto_pass_cycle", "absent")


@pytest.mark.asyncio
async def test_a_clean_empty_nomination_latches_the_pass(phase, monkeypatch):
    """The latch exists for this case: nothing selected, and forge said so."""
    phase.shared_state.macro_cycle = 3
    _fake_forge(monkeypatch, _envelope([]))

    await phase._run_kernel_opt_nomination()

    assert _latched_cycle(phase.shared_state) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "envelope",
    [
        {"nomination": {}},
        {"patches": {}, "nomination": {}},
        {"patches": "none"},
        _envelope([{"micro_speedup": 1.4}]),
        _envelope(["not-an-object"]),
    ],
    ids=["no-key", "object", "string", "no-identity", "not-an-object"],
)
async def test_an_unreadable_envelope_fails_and_leaves_the_latch_dormant(phase, monkeypatch, envelope):
    """A schema skew must not read as a pass that decided against every kernel."""
    phase.shared_state.macro_cycle = 3
    _fake_forge(monkeypatch, envelope)

    await phase._run_kernel_opt_nomination()

    (message,) = phase.bus.sent
    assert message.payload["result"]["status"] == "failed"
    assert _queued_on_disk(phase.session_dir) == []
    assert _latched_cycle(phase.shared_state) is None


@pytest.mark.asyncio
async def test_an_unreadable_envelope_still_names_what_it_refused(phase, monkeypatch):
    """The operator needs the reason, or a schema skew looks like a dead lane."""
    _fake_forge(monkeypatch, _envelope([{"micro_speedup": 1.4}]))

    await phase._run_kernel_opt_nomination()

    (message,) = phase.bus.sent
    result = message.payload["result"]
    assert "unreadable" in result["error"]
    assert sum(result["dropped"].values()) == 1


def _staged_workspaces(monkeypatch) -> list[str]:
    """Record the workspace each submit_auto call would stage into."""
    from hyperloom.agents.kernel.tools.backends import forge_submit

    seen: list[str] = []

    def _fake(**kwargs):
        seen.append(str(kwargs["output_dir"]))
        return _envelope([])

    monkeypatch.setattr(forge_submit, "submit_auto", _fake)
    return seen


@pytest.mark.asyncio
async def test_each_attempt_stages_into_its_own_directory(phase, monkeypatch):
    """Staging retains its worktree and refuses to reuse a path.

    A directory fixed per session therefore fails every cycle after the first,
    before the subprocess even starts, and a retry after a mid-run failure fails
    the same way.
    """
    seen = _staged_workspaces(monkeypatch)

    await phase._run_kernel_opt_nomination()
    phase.shared_state.macro_cycle = 1
    phase.shared_state.save(phase.session_dir)
    await phase._run_kernel_opt_nomination()

    assert len(seen) == 2
    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_two_attempts_in_one_cycle_do_not_collide(phase, monkeypatch):
    """A retry within a cycle needs its own directory too."""
    seen = _staged_workspaces(monkeypatch)

    await phase._run_kernel_opt_nomination()
    await phase._run_kernel_opt_nomination()

    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_the_attempt_directory_names_its_cycle(phase, monkeypatch):
    """Grouped by cycle so a session's attempts stay readable on disk."""
    seen = _staged_workspaces(monkeypatch)
    # The handler reads the persisted state, which is where the loop stamps the
    # cycle; uniqueness itself comes from the attempt id, not this number.
    phase.shared_state.macro_cycle = 4
    phase.shared_state.save(phase.session_dir)

    await phase._run_kernel_opt_nomination()

    assert "cycle-4" in seen[0]
