"""Tests for framework_agent.kb.

Hermetic - all tests redirect KB_ROOT via FRAMEWORK_AGENT_KB_DIR env so
no real workspace KB is touched.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import framework_agent.kb as kb
from framework_agent.models import Finding


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KB resolution at a clean tmp_path for every test."""
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    return tmp_path


# _resolve_kb_root -------------------------------------------------------


def test_resolve_kb_root_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FRAMEWORK_AGENT_KB_DIR beats FRAMEWORK_AGENT_ROOT/kb fallback."""
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "root"))
    assert kb._resolve_kb_root() == tmp_path / "explicit"


def test_resolve_kb_root_uses_framework_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback to FRAMEWORK_AGENT_ROOT/kb when explicit env is unset."""
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "root"))
    assert kb._resolve_kb_root() == tmp_path / "root" / "kb"


# list_domains / get_domain_files ---------------------------------------


def test_list_domains_empty_when_root_missing(kb_root: Path) -> None:
    """list_domains returns [] when no domain directory has been created."""
    assert kb.list_domains() == []


def test_list_domains_and_files(kb_root: Path) -> None:
    """list_domains + get_domain_files reflect on-disk layout."""
    (kb_root / "framework").mkdir()
    (kb_root / "framework" / "README.md").write_text("# fw")
    (kb_root / "framework" / "empirical_kb.md").write_text("# empirical")
    (kb_root / "kernel").mkdir()
    assert kb.list_domains() == ["framework", "kernel"]
    names = sorted(p.name for p in kb.get_domain_files("framework"))
    assert names == ["README.md", "empirical_kb.md"]


# _match_domains --------------------------------------------------------


def test_match_domains_keyword_hit() -> None:
    """A task description containing whitelisted keywords picks the right domain."""
    domains = kb._match_domains("improve sglang vllm cudagraph attention")
    assert "framework" in domains
    assert "kernel" in domains


def test_match_domains_no_hit() -> None:
    """When no keyword matches, the result is an empty list."""
    assert kb._match_domains("totally unrelated free-form text") == []


# select_kb -------------------------------------------------------------


def test_select_kb_prioritises_empirical_then_pitfalls(kb_root: Path) -> None:
    """select_kb returns empirical_kb.md before shared_pitfalls.md before rest."""
    d = kb_root / "framework"
    d.mkdir()
    (d / "README.md").write_text("# r")
    (d / "shared_pitfalls.md").write_text("# pitfalls")
    (d / "empirical_kb.md").write_text("# emp")
    (d / "model_taxonomy.md").write_text("# tax")
    out = kb.select_kb("sglang scheduler tuning", domains=["framework"])
    names = [p.path.name for p in out]
    assert names[0] == "empirical_kb.md"
    assert names[1] == "shared_pitfalls.md"
    assert set(names[2:]) == {"README.md", "model_taxonomy.md"}


def test_select_kb_auto_matches_when_domains_none(kb_root: Path) -> None:
    """select_kb derives domains from task_description when omitted."""
    d = kb_root / "framework"
    d.mkdir()
    (d / "empirical_kb.md").write_text("# emp")
    out = kb.select_kb("improve sglang throughput")
    assert any(p.domain == "framework" for p in out)


def test_select_kb_full_text_fallback(kb_root: Path) -> None:
    """When no keyword matches, fallback scans file contents for the lower-cased query."""
    d = kb_root / "kernel"
    d.mkdir()
    (d / "empirical_kb.md").write_text("Quirk-Wibble is the new HotAcronym for stuff.")
    out = kb.select_kb("quirk-wibble")
    assert out and out[0].domain == "kernel"


# contribute_to_kb -------------------------------------------------------


def test_contribute_creates_domain_and_appends(kb_root: Path) -> None:
    """contribute_to_kb auto-creates the domain dir + empirical_kb.md."""
    path = kb.contribute_to_kb(
        domain="framework",
        finding="hello world",
        source="unit-test",
        session_id="s1",
    )
    assert path.exists()
    text = path.read_text()
    assert "hello world" in text
    assert "source=`unit-test`" in text
    assert "session=`s1`" in text


# synthesize_findings (pure-Python) -------------------------------------


def test_synthesize_pure_python_renders_findings() -> None:
    """The pure-Python path produces a deterministic markdown digest."""
    findings = [
        Finding(
            title="winner PR:1",
            body="explanation",
            source="fa explore --execute",
            session_id="s1",
            candidate_ref="PR:1",
            metrics={"throughput": 1234.5, "throughput_ratio": 1.10},
        ),
        Finding(
            title="winner PR:2",
            body="explanation 2",
            source="fa explore --execute",
            session_id="s1",
            candidate_ref="PR:2",
            metrics={"throughput": 1500.0, "throughput_ratio": 1.30},
        ),
    ]
    out = kb.synthesize_findings("framework", findings)
    assert "## Synthesised findings - framework" in out
    assert "### winner PR:1" in out
    assert "### winner PR:2" in out
    assert "1234.5" in out
    # The throughput / throughput_ratio metric keys repeat, so they
    # should show up in the aggregate-metrics tail.
    assert "## Aggregate metrics" in out
    assert "throughput" in out
    assert "throughput_ratio" in out


def test_synthesize_empty_findings() -> None:
    """An empty list still produces a valid header + placeholder."""
    out = kb.synthesize_findings("framework", [])
    assert "## Synthesised findings - framework" in out
    assert "_no findings_" in out


# synthesize_findings (with_llm=True, lazy-import failure) --------------


def test_synthesize_with_llm_missing_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """with_llm=True must raise a clear RuntimeError when claude_agent_sdk is absent."""
    # Block claude_agent_sdk import even if it happens to be installed.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(RuntimeError, match="claude_agent_sdk not installed"):
        kb.synthesize_findings(
            "framework",
            [Finding(title="x", body="y")],
            with_llm=True,
        )


# search_kb -------------------------------------------------------------


def test_search_kb_finds_matching_content(kb_root: Path) -> None:
    """search_kb returns only the files whose content contains the needle."""
    a = kb_root / "framework"
    a.mkdir()
    (a / "empirical_kb.md").write_text("FlashInfer NVFP4 winners")
    (a / "shared_pitfalls.md").write_text("nothing special here")
    b = kb_root / "kernel"
    b.mkdir()
    (b / "empirical_kb.md").write_text("torch.nn.functional.gelu shenanigans")
    hits = kb.search_kb("flashinfer")
    assert len(hits) == 1
    assert hits[0].domain == "framework"
