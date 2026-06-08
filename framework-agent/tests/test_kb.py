# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for framework_agent.kb and the `fa kb <op>` CLI surface.

Hermetic - all tests redirect KB_ROOT via FRAMEWORK_AGENT_KB_DIR env so
no real workspace KB is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import framework_agent.kb as kb
import framework_agent.runtime.cli as cli
from framework_agent.models import Finding


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KB resolution at a clean tmp_path for every test."""
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# framework_agent.kb module
# ---------------------------------------------------------------------------


class TestResolveKbRoot:
    """_resolve_kb_root precedence rules."""

    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FRAMEWORK_AGENT_KB_DIR beats FRAMEWORK_AGENT_ROOT/kb fallback."""
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "explicit"))
        monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "root"))
        assert kb._resolve_kb_root() == tmp_path / "explicit"

    def test_uses_framework_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fallback to FRAMEWORK_AGENT_ROOT/kb when explicit env is unset."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "root"))
        assert kb._resolve_kb_root() == tmp_path / "root" / "kb"


class TestListAndMatch:
    """list_domains / get_domain_files / _match_domains."""

    def test_list_domains_empty_when_root_missing(self, kb_root: Path) -> None:
        """list_domains returns [] when no domain directory has been created."""
        assert kb.list_domains() == []

    def test_list_domains_and_files(self, kb_root: Path) -> None:
        """list_domains + get_domain_files reflect on-disk layout."""
        (kb_root / "framework").mkdir()
        (kb_root / "framework" / "README.md").write_text("# fw")
        (kb_root / "framework" / "empirical_kb.md").write_text("# empirical")
        (kb_root / "kernel").mkdir()
        assert kb.list_domains() == ["framework", "kernel"]
        names = sorted(p.name for p in kb.get_domain_files("framework"))
        assert names == ["README.md", "empirical_kb.md"]

    def test_match_domains_keyword_hit(self) -> None:
        """A task description containing whitelisted keywords picks the right domain."""
        domains = kb._match_domains("improve sglang vllm cudagraph attention")
        assert "framework" in domains
        assert "kernel" in domains

    def test_match_domains_no_hit(self) -> None:
        """When no keyword matches, the result is an empty list."""
        assert kb._match_domains("totally unrelated free-form text") == []

    def test_match_domains_atom_keyword_hit(self) -> None:
        """``atom`` lives under the framework domain alongside sglang /
        vllm so a gap description mentioning atom can pull
        framework-domain KB priors. Pinned here so a future trim of
        DOMAIN_KEYWORDS doesn't silently drop it."""
        domains = kb._match_domains("improve atom moe throughput on mi300x")
        assert "framework" in domains, (
            f"atom must hit the framework domain; got {domains!r}"
        )

    def test_atom_in_framework_domain_keywords_constant(self) -> None:
        """Constant-level guard: ``atom`` must appear in the
        ``framework`` domain's keyword list."""
        assert "atom" in kb.DOMAIN_KEYWORDS["framework"]


class TestSelectKb:
    """select_kb priority and fallback behaviour."""

    def test_prioritises_empirical_then_pitfalls(self, kb_root: Path) -> None:
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

    def test_auto_matches_when_domains_none(self, kb_root: Path) -> None:
        """select_kb derives domains from task_description when omitted."""
        d = kb_root / "framework"
        d.mkdir()
        (d / "empirical_kb.md").write_text("# emp")
        out = kb.select_kb("improve sglang throughput")
        assert any(p.domain == "framework" for p in out)

    def test_full_text_fallback(self, kb_root: Path) -> None:
        """When no keyword matches, fallback scans file contents for the lower-cased query."""
        d = kb_root / "kernel"
        d.mkdir()
        (d / "empirical_kb.md").write_text("Quirk-Wibble is the new HotAcronym for stuff.")
        out = kb.select_kb("quirk-wibble")
        assert out and out[0].domain == "kernel"


class TestContributeAndSynthesize:
    """contribute_to_kb + synthesize_findings."""

    def test_contribute_creates_domain_and_appends(self, kb_root: Path) -> None:
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

    def test_synthesize_pure_python_renders_findings(self) -> None:
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
        assert "## Aggregate metrics" in out
        assert "throughput" in out
        assert "throughput_ratio" in out

    def test_synthesize_empty_findings(self) -> None:
        """An empty list still produces a valid header + placeholder."""
        out = kb.synthesize_findings("framework", [])
        assert "## Synthesised findings - framework" in out
        assert "_no findings_" in out

    def test_synthesize_with_llm_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """with_llm=True must raise a clear RuntimeError when claude_agent_sdk is absent."""
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        with pytest.raises(RuntimeError, match="claude_agent_sdk not installed"):
            kb.synthesize_findings(
                "framework",
                [Finding(title="x", body="y")],
                with_llm=True,
            )


class TestSearchKb:
    def test_search_kb_finds_matching_content(self, kb_root: Path) -> None:
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


# ---------------------------------------------------------------------------
# `fa kb <op>` CLI surface
# ---------------------------------------------------------------------------


class TestKbCli:
    """End-to-end exercises of the `fa kb` argparse subcommand."""

    def test_list_empty(self, kb_root: Path, capsys: pytest.CaptureFixture) -> None:
        """fa kb list on a clean KB returns an empty domains array."""
        rc = cli.main(["kb", "list"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["domains"] == []
        assert payload["kb_root"] == str(kb_root)

    def test_list_after_contribute(self, kb_root: Path, capsys: pytest.CaptureFixture) -> None:
        """A successful contribute makes the new domain show up in list."""
        rc = cli.main([
            "kb", "contribute",
            "--domain", "framework",
            "--body", "hello",
            "--source", "test", "--session-id", "s1",
        ])
        assert rc == 0
        capsys.readouterr()  # discard contribute output
        rc = cli.main(["kb", "list"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["domains"] == ["framework"]

    def test_show_unknown_domain_exit_two(self, kb_root: Path, capsys: pytest.CaptureFixture) -> None:
        """fa kb show on a non-existent domain exits 2 with a clear error."""
        rc = cli.main(["kb", "show", "--domain", "nope"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_search_returns_hits(self, kb_root: Path, capsys: pytest.CaptureFixture) -> None:
        """fa kb search returns the file hits whose content matches the query."""
        d = kb_root / "framework"
        d.mkdir()
        (d / "empirical_kb.md").write_text("FlashInfer NVFP4 winner")
        rc = cli.main(["kb", "search", "--query", "flashinfer"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 1
        assert payload["hits"][0]["domain"] == "framework"

    def test_contribute_requires_body(self, kb_root: Path, capsys: pytest.CaptureFixture) -> None:
        """contribute with neither --body nor --body-file should rc=2."""
        rc = cli.main(["kb", "contribute", "--domain", "framework"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "--body" in err

    def test_synthesize_pure_python_smoke(
        self, kb_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """fa kb synthesize without --with-llm emits a deterministic digest."""
        findings = [
            {
                "title": "winner PR:1",
                "body": "explanation",
                "source": "fa explore --execute",
                "session_id": "s1",
                "candidate_ref": "PR:1",
                "metrics": {"throughput": 1234.5, "throughput_ratio": 1.10},
            }
        ]
        findings_path = tmp_path / "findings.json"
        findings_path.write_text(json.dumps(findings), encoding="utf-8")
        rc = cli.main([
            "kb", "synthesize",
            "--domain", "framework",
            "--findings", str(findings_path),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Synthesised findings - framework" in out
        assert "### winner PR:1" in out
        assert "1234.5" in out

    def test_synthesize_with_llm_missing_sdk_exit_two(
        self, kb_root: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--with-llm but claude_agent_sdk absent must surface as rc=2 with hint."""
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        rc = cli.main(["kb", "synthesize", "--domain", "framework", "--with-llm"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "claude_agent_sdk not installed" in err


# ---------------------------------------------------------------------------
# Per-framework KB partition (`framework_optimization/<fw>/`)
# ---------------------------------------------------------------------------


class TestPathForFramework:
    """``path_for_framework`` resolves the per-framework KB partition."""

    def test_atom_partition_path_resolves(self, kb_root: Path) -> None:
        path = kb.path_for_framework("atom")
        assert path == kb_root / "framework_optimization" / "atom"

    @pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
    def test_path_resolves_for_all_known_frameworks(
        self, kb_root: Path, framework: str,
    ) -> None:
        path = kb.path_for_framework(framework)
        assert path == kb_root / "framework_optimization" / framework

    def test_path_lowercases_and_strips(self, kb_root: Path) -> None:
        # ``"  Atom  "`` and ``"ATOM"`` must resolve to the same
        # partition as ``"atom"`` — keeps the partition stable across
        # casing variation in CLI inputs / config files.
        path_a = kb.path_for_framework("  Atom  ")
        path_b = kb.path_for_framework("ATOM")
        path_c = kb.path_for_framework("atom")
        assert path_a == path_b == path_c

    def test_empty_framework_returns_partition_root(self, kb_root: Path) -> None:
        # Per the helper's docstring, an empty / whitespace framework
        # resolves to the partition root so callers can detect
        # "framework not selected".
        assert kb.path_for_framework("") == kb_root / "framework_optimization"
        assert kb.path_for_framework("   ") == kb_root / "framework_optimization"

    def test_partition_dir_not_created_eagerly(self, kb_root: Path) -> None:
        # ``path_for_framework`` is read-only; it must NOT create the
        # partition directory on disk.
        _ = kb.path_for_framework("atom")
        assert not (kb_root / "framework_optimization" / "atom").exists()


class TestContributeToKbForFramework:
    """``contribute_to_kb_for_framework`` writes to the framework
    sub-partition lazily."""

    def test_creates_partition_on_first_write(self, kb_root: Path) -> None:
        path = kb.contribute_to_kb_for_framework(
            "atom",
            finding="MTP with `--num-speculative-tokens 3` regressed on FP8 Qwen3-32B",
            source="explore",
            session_id="sess-atom-001",
        )
        expected = kb_root / "framework_optimization" / "atom" / "empirical_kb.md"
        assert path == expected
        assert expected.is_file()
        body = expected.read_text()
        assert "source=`explore`" in body
        assert "sess-atom-001" in body
        assert "MTP" in body

    def test_appends_subsequent_findings(self, kb_root: Path) -> None:
        kb.contribute_to_kb_for_framework(
            "atom", finding="first", source="explore", session_id="s1",
        )
        kb.contribute_to_kb_for_framework(
            "atom", finding="second", source="explore", session_id="s1",
        )
        body = (
            kb_root / "framework_optimization" / "atom" / "empirical_kb.md"
        ).read_text()
        # Both findings present, separated by the `---` divider the
        # helper emits.
        assert body.count("---") == 2
        assert "first" in body
        assert "second" in body
