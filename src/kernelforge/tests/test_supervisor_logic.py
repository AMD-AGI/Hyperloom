# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for orchestrator/supervisor.py pure logic (no LLM/subprocess).

Covers the prompt builder's three file-access modes and interaction persistence.
Provider dispatch is covered through the shared registry in
``test_supervisor_backend.py``."""

from __future__ import annotations

from kernelforge.orchestrator.supervisor import (
    _build_task_prompt,
    _persist_interaction,
    load_latest_supervisor_ruling,
    persist_supervisor_ruling,
)


# ─── _build_task_prompt ───


def test_build_task_prompt_uses_bounded_artifact_access():
    p = _build_task_prompt("PROG", "TRAJ", "budget exhausted", "gfx942", 3)
    assert "gfx942" in p
    assert "budget exhausted" in p
    assert "PROG" in p
    assert "TRAJ" in p
    assert "AT MOST ~8 files" in p
    assert "Recommend at most 3 concrete directions" in p
    assert "remaining headroom for EVERY scored case" in p
    assert "no required output schema" in p


def test_build_task_prompt_empty_digest_placeholder():
    p = _build_task_prompt("PROG", "", "reason", "gfx942", 3)
    assert "(no archived trajectory yet)" in p


def test_build_task_prompt_includes_structured_profile_evidence():
    prompt = _build_task_prompt(
        "PROG",
        "TRAJ",
        "stall",
        "gfx942",
        3,
        evidence_context='{"case_id": "case-a", "bottleneck": "memory"}',
    )

    assert '"case_id": "case-a"' in prompt
    assert '"bottleneck": "memory"' in prompt
    assert "historical lesson documents" in prompt
    assert "hard floor" in prompt


# ─── _persist_interaction ───


def test_persist_interaction_writes_file(tmp_path):
    reply = "\n  REPLY with intentional surrounding whitespace  \n"
    _persist_interaction(
        str(tmp_path),
        7,
        "stall",
        "SYSTEM",
        "USER",
        reply,
        backend="codex",
        model="gpt-5.3-codex",
    )
    f = tmp_path / "forge_experiments" / "supervisor" / "intervention_iter_007.md"
    assert f.exists()
    body = f.read_text()
    assert "iteration 7" in body
    assert "backend: codex" in body
    assert "model: gpt-5.3-codex" in body
    assert reply in body
    assert load_latest_supervisor_ruling(str(tmp_path)) == reply


def test_persist_interaction_empty_reply_placeholder(tmp_path):
    _persist_interaction(str(tmp_path), 1, "stall", "SYS", "USR", "", backend="claude", model="gpt-5.5")
    f = tmp_path / "forge_experiments" / "supervisor" / "intervention_iter_001.md"
    assert "(empty" in f.read_text()
    assert load_latest_supervisor_ruling(str(tmp_path)) == ""


def test_empty_attempt_expires_latest_ruling(tmp_path):
    _persist_interaction(
        str(tmp_path),
        1,
        "stall",
        "SYS",
        "USR",
        "FIRST",
        backend="claude",
        model="gpt-5.5",
    )
    _persist_interaction(
        str(tmp_path),
        2,
        "stall",
        "SYS",
        "USR",
        "",
        backend="claude",
        model="gpt-5.5",
    )

    assert load_latest_supervisor_ruling(str(tmp_path)) == ""


def test_injected_ruling_gets_fallback_audit_artifact(tmp_path):
    reply = "\nFree-form injected ruling.\n\n"

    interaction, latest = persist_supervisor_ruling(
        str(tmp_path),
        3,
        "stall",
        reply,
    )

    assert interaction is not None
    assert latest is not None
    assert "source: injected callback" in interaction.read_text()
    assert reply in interaction.read_text()
    assert latest.read_text() == reply


def test_persist_interaction_swallows_errors():
    # A bogus workspace path must not raise (best-effort persistence).
    _persist_interaction("\x00bad", 1, "r", "s", "u", "reply", backend="codex", model="m")
