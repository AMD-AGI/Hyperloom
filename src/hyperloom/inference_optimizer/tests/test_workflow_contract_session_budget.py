# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A phase exit's frozen budget follows the contract the session was stamped with."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest
from jsonschema import validate

from hyperloom.inference_optimizer.breakdown.exporter import build
from hyperloom.inference_optimizer.breakdown.recorder.session_metadata import record_metadata_identity
from hyperloom.inference_optimizer.breakdown.workflow_contract import WORKFLOW_CONTRACT_V1, workflow_schema
from hyperloom.inference_optimizer.session.manifest import write_manifest
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.phases.machine_state import compute_next_phase, record_phase_transition
from hyperloom.orchestrator.phases.session_contract import bound_session_declares
from hyperloom.orchestrator.state.shared_state import SharedState


def _stamp_session(tmp_path, *, legacy: bool) -> str:
    manifest = write_manifest(tmp_path, session_id="session-budget")
    if not legacy:
        return manifest["workflow_contract_version"]
    unstamped = {
        key: value
        for key, value in manifest.items()
        if key not in {"workflow_contract_version", "workflow_contract_digest"}
    }
    (tmp_path / "manifest.json").write_text(json.dumps(unstamped), encoding="utf-8")
    record_metadata_identity(tmp_path, unstamped)
    return WORKFLOW_CONTRACT_V1


@pytest.mark.parametrize("phase", ["FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP"])
@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "v2"])
def test_phase_exit_budget_exports_under_session_contract(tmp_path, phase: str, legacy: bool):
    """A v1 session resumed on v2 code exports a phase exit its own schema accepts."""
    version = _stamp_session(tmp_path, legacy=legacy)
    state = SharedState(session_id="session-budget", phase=phase, baseline_tput=100.0)
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase=phase, reason="phase_entered")
        state.set_stop_reason("target_reached")
        target, reason, evidence = compute_next_phase(state)
        assert (target, reason) == ("CLOSE", "target_reached")
        budget = evidence["predicate_inputs"]["budget"]
        assert ("current_balance" in budget) is not legacy
        if not legacy:
            assert budget["current_balance"] == budget["remaining_sec"]
        record_phase_transition(state, to_phase=target, reason=reason, evidence=evidence)
    state.save(tmp_path)

    exported = build(tmp_path)
    assert exported["metadata"]["workflow"]["workflow_contract_version"] == version
    exits = [
        segment["exit_evidence"]
        for event in exported["timeline"]
        if event.get("type") == "phase"
        for segment in event["ext"]["segments"]
        if segment.get("exit_reason") == "target_reached"
    ]
    assert [row["predicate_inputs"]["budget"] for row in exits] == [budget]
    validate(instance=exported, schema=workflow_schema(version))


@pytest.mark.parametrize(
    ("manifest_text", "declared"),
    [
        (None, True),
        (json.dumps({"workflow_contract_version": "hyperloom.workflow_evaluation.v2"}), True),
        (json.dumps({}), False),
        (json.dumps({"workflow_contract_version": "hyperloom.workflow_evaluation.v9"}), False),
        ("{not json", True),
        (json.dumps(["not", "a", "manifest"]), True),
    ],
    ids=["no-manifest", "v2", "unstamped", "unknown-version", "corrupt", "not-a-mapping"],
)
def test_bound_session_declares_follows_manifest_identity(tmp_path, manifest_text, declared: bool):
    """A readable stamp decides; an unreadable manifest answers with the current contract."""
    if manifest_text is not None:
        (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    with session_scope(tmp_path):
        assert bound_session_declares("phase_budget", "current_balance") is declared
    assert bound_session_declares("phase_budget", "current_balance") is True


def _fail_manifest_io(monkeypatch, session_dir: Path, err: int, *, stat: bool = False) -> list[Path]:
    """Make reads (and optionally stats) of the session manifest raise ``err``; return the read log."""
    target = session_dir / "manifest.json"
    reads: list[Path] = []
    real_read_text = Path.read_text
    real_stat = Path.stat

    def read_text(self, *args, **kwargs):
        if self == target:
            reads.append(self)
            raise OSError(err, "injected", str(self))
        return real_read_text(self, *args, **kwargs)

    def fake_stat(self, *args, **kwargs):
        if self == target:
            raise PermissionError(err, "injected", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    if stat:
        monkeypatch.setattr(Path, "stat", fake_stat)
    return reads


def _count_manifest_reads(monkeypatch, session_dir: Path) -> list[Path]:
    target = session_dir / "manifest.json"
    reads: list[Path] = []
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == target:
            reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    return reads


@pytest.mark.parametrize("err", [errno.EMFILE, errno.EIO, errno.ESTALE])
def test_transient_manifest_read_failure_keeps_v2_current_balance(tmp_path, monkeypatch, err: int):
    """A v2 exit decided on a tick whose manifest read fails still exports a v2-valid budget."""
    version = _stamp_session(tmp_path, legacy=False)
    state = SharedState(session_id="session-budget", phase="SWEEP", baseline_tput=100.0)
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase="SWEEP", reason="phase_entered")
        state.set_stop_reason("target_reached")
        with monkeypatch.context() as patched:
            reads = _fail_manifest_io(patched, tmp_path, err)
            target, reason, evidence = compute_next_phase(state)
        assert reads, "the injected failure never reached the manifest read"
        assert (target, reason) == ("CLOSE", "target_reached")
        budget = evidence["predicate_inputs"]["budget"]
        assert budget["current_balance"] == budget["remaining_sec"]
        record_phase_transition(state, to_phase=target, reason=reason, evidence=evidence)
    state.save(tmp_path)

    exported = build(tmp_path)
    assert exported["metadata"]["workflow"]["workflow_contract_version"] == version
    validate(instance=exported, schema=workflow_schema(version))


def test_manifest_stat_error_does_not_abort_the_tick(tmp_path, monkeypatch):
    """EACCES on the manifest is a failed read, not an exception out of the phase decision."""
    _stamp_session(tmp_path, legacy=False)
    state = SharedState(session_id="session-budget", phase="SWEEP", baseline_tput=100.0)
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase="SWEEP", reason="phase_entered")
        state.set_stop_reason("target_reached")
        _fail_manifest_io(monkeypatch, tmp_path, errno.EACCES, stat=True)
        assert bound_session_declares("phase_budget", "current_balance") is True
        target, reason, evidence = compute_next_phase(state)
    assert (target, reason) == ("CLOSE", "target_reached")
    assert "current_balance" in evidence["predicate_inputs"]["budget"]
    assert "baseline_tput" in evidence["predicate_inputs"]["global"]


def test_successful_manifest_read_is_reused(tmp_path, monkeypatch):
    """Once read, the stamp answers without touching the manifest, even after reads start failing."""
    _stamp_session(tmp_path, legacy=True)
    with session_scope(tmp_path):
        reads = _count_manifest_reads(monkeypatch, tmp_path)
        assert bound_session_declares("phase_budget", "current_balance") is False
        assert bound_session_declares("phase_budget", "current_balance") is False
        assert len(reads) == 1
        _fail_manifest_io(monkeypatch, tmp_path, errno.EIO)
        assert bound_session_declares("phase_budget", "current_balance") is False


def test_failed_manifest_read_is_retried(tmp_path, monkeypatch):
    """A failed read is not remembered: the next call reads again and then keeps the stamp."""
    _stamp_session(tmp_path, legacy=True)
    with session_scope(tmp_path):
        with monkeypatch.context() as patched:
            failed = _fail_manifest_io(patched, tmp_path, errno.EIO)
            assert bound_session_declares("phase_budget", "current_balance") is True
            assert bound_session_declares("phase_budget", "current_balance") is True
        assert len(failed) == 2
        reads = _count_manifest_reads(monkeypatch, tmp_path)
        assert bound_session_declares("phase_budget", "current_balance") is False
        assert bound_session_declares("phase_budget", "current_balance") is False
        assert len(reads) == 1
