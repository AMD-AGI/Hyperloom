# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the on-demand reactor reference index and read_reference tool."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.phases import machine_state as _ps
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.prompts.prompt_builder import (
    _section_reference_index,
    build_orchestration_prompt,
    default_enabled_actions,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mcp_context_tools import (
    CONTEXT_TOOL_NAMES,
    ContextProvider,
)


@pytest.fixture(scope="module")
def refs_dir():
    return asset_prompt_references_dir()


@pytest.fixture(scope="module")
def registry() -> dict:
    return ACTION_CATALOGUE


@pytest.fixture
def file_reader(refs_dir):
    """ContextProvider wired to the real references directory."""

    def _read(name: str) -> str:
        candidate = (refs_dir / name).with_suffix(".md").resolve()
        if candidate.parent != refs_dir.resolve():
            return f"(rejected: {name!r})"
        if not candidate.exists():
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(not found: {name!r}; available: {available})"
        return candidate.read_text(encoding="utf-8")

    return ContextProvider(shared_state=None, reference_reader=_read)


# ---------------------------------------------------------------------------
# Reference directory contract
# ---------------------------------------------------------------------------


def test_all_reference_files_have_when_tag(refs_dir):
    """Every shipped reference doc must carry a <!-- when: ... --> tag."""
    files = sorted(refs_dir.glob("*.md"))
    assert files, "no reference docs found"
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        has_when = any(line.strip().startswith("<!-- when:") for line in lines[:5])
        assert has_when, f"{path.name} is missing a <!-- when: ... --> header tag"


def test_all_reference_files_appear_in_unscoped_index(refs_dir):
    """Every reference doc must be reachable from the unscoped index."""
    index_lines = _section_reference_index(references_dir=refs_dir, phase="")
    for path in refs_dir.glob("*.md"):
        assert path.stem in "\n".join(index_lines), f"{path.name} not found in the unscoped reference index"


def test_index_entries_resolve_to_real_files(refs_dir):
    """Each index entry must correspond to an existing document."""
    index_lines = _section_reference_index(references_dir=refs_dir, phase="")
    for line in index_lines:
        if not line.startswith("- **"):
            continue
        stem = line.split("**")[1]
        assert (refs_dir / f"{stem}.md").exists(), f"index entry {stem!r} points at a missing file"


def test_empty_refs_dir_produces_no_section(tmp_path):
    """An empty directory silently produces no section."""
    assert _section_reference_index(references_dir=tmp_path, phase="") == []


# ---------------------------------------------------------------------------
# Phase filtering
# ---------------------------------------------------------------------------


def test_specialist_rescue_only_in_explore_and_framework(refs_dir):
    """specialist_rescue is phase-tagged EXPLORE,FRAMEWORK_AGENT; other phases hide it."""
    rescue_path = refs_dir / "specialist_rescue.md"
    if not rescue_path.exists():
        pytest.skip("specialist_rescue.md not present")
    for phase in _ps.PHASE_NAMES:
        index = "\n".join(_section_reference_index(references_dir=refs_dir, phase=phase))
        if phase in ("EXPLORE", "FRAMEWORK_AGENT"):
            assert "specialist_rescue" in index, f"specialist_rescue missing from index in {phase}"
        else:
            assert "specialist_rescue" not in index, f"specialist_rescue leaked into index in {phase}"


def test_failure_recovery_present_in_every_phase(refs_dir):
    """failure_recovery has no phase restriction and must appear everywhere."""
    if not (refs_dir / "failure_recovery.md").exists():
        pytest.skip("failure_recovery.md not present")
    for phase in _ps.PHASE_NAMES:
        index = "\n".join(_section_reference_index(references_dir=refs_dir, phase=phase))
        assert "failure_recovery" in index, f"failure_recovery missing from index in {phase}"


# ---------------------------------------------------------------------------
# Orchestration prompt integration
# ---------------------------------------------------------------------------


def test_reference_index_present_in_prompt(registry, refs_dir):
    """The ON-DEMAND REFERENCE INDEX section appears in every phase build."""
    for phase in _ps.PHASE_NAMES:
        text = build_orchestration_prompt(
            action_registry=registry,
            enabled_actions=default_enabled_actions(no_kernel=False),
            framework="sglang",
            kernel_enabled=True,
            framework_agent_phase_enabled=True,
            objective_kind="gain_pct",
            objective_value=15.0,
            max_minutes=480,
            phase=phase,
            references_dir=refs_dir,
        )
        assert "## 8. ON-DEMAND REFERENCE INDEX" in text, f"reference index missing from {phase} prompt"


# ---------------------------------------------------------------------------
# read_reference tool
# ---------------------------------------------------------------------------


def test_read_reference_in_context_tool_names():
    """read_reference must appear in CONTEXT_TOOL_NAMES so it is granted to orchestration."""
    assert "read_reference" in CONTEXT_TOOL_NAMES


def test_read_reference_granted_to_orchestration():
    """PolicyGate must list read_reference in orchestration's allowed tools."""
    gate = PolicyGate(role_registry=default_role_registry())
    assert "read_reference" in gate.allowed_tools_for_agent("orchestration")


def test_read_reference_returns_doc(refs_dir, file_reader):
    """A valid stem returns the document text with its when: header tag."""
    docs = sorted(refs_dir.glob("*.md"))
    if not docs:
        pytest.skip("no reference docs")
    result = file_reader.read_reference(docs[0].stem)
    assert result.strip()
    assert "when:" in result.splitlines()[0]


def test_read_reference_rejects_path_traversal(file_reader):
    """Path traversal names must return an error string, not crash."""
    for bad in ("../orchestration", "/etc/passwd", ".hidden"):
        result = file_reader.read_reference(bad)
        assert "rejected" in result or "not found" in result or "invalid" in result, (
            f"traversal not caught for {bad!r}: got {result!r}"
        )


def test_read_reference_unknown_name_returns_available_list(file_reader):
    """Unknown names return the list of valid names."""
    result = file_reader.read_reference("nonexistent_reference_xyz")
    assert "not found" in result
    assert "available" in result


def test_read_reference_not_wired_returns_marker():
    """A ContextProvider with no reference_reader returns a clear marker string."""
    provider = ContextProvider(shared_state=None)
    assert "not wired" in provider.read_reference("anything")
