# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.kb and the `fa kb <op>` CLI surface. Hermetic - redirects the KB root via INFERENCE_OPTIMIZER_FA_KB_PATH."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import hyperloom.agents.framework.kb as kb
import hyperloom.agents.framework.runtime.cli as cli
from hyperloom.agents.framework.models import Finding


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KB resolution at a clean tmp_path for every test."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# hyperloom.agents.framework.kb module
# ---------------------------------------------------------------------------


class TestResolveKbRoot:
    """_resolve_kb_root, which has exactly one override and no fallback chain."""

    def test_io_override_matches_writeback_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fa reader honours the KB override so write/read paths match."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "io-kb"))
        assert kb._resolve_kb_root() == tmp_path / "io-kb"

    def test_defaults_to_its_own_workspace_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no override the KB lives in its own directory under the workspace.

        Not ``<workspace>/kb``: the recipe KB owns that, and ``list_domains``
        treats every directory under this root as a framework domain.
        """
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb._resolve_kb_root() == tmp_path / "workspace" / "framework-kb"

    def test_withdrawn_override_is_ignored_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolution stays total; rejecting the withdrawn override is start-up's job.

        Read paths treat KB lookups as advisory and swallow their own failures,
        so raising here would be absorbed by the caller rather than surfaced —
        turning a misconfiguration into a silently disabled gate.
        """
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "legacy"))
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "io-kb"))
        assert kb._resolve_kb_root() == tmp_path / "io-kb"

    def test_framework_agent_root_is_not_a_kb_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """That variable means "where the skill is installed" and must not raise."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "skill"))
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb._resolve_kb_root() == tmp_path / "workspace" / "framework-kb"


class TestCheckKbConfiguration:
    """The start-up gate that rejects a KB layout this build cannot honour."""

    def test_withdrawn_override_fails_loudly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run still setting the reader-only override is stopped at start-up.

        Honouring it moved reads without moving writes, which is how the ledger
        came to be written in one place and read from another. A start-up
        failure naming the replacement is recoverable; a silent split is not.
        """
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "legacy"))
        with pytest.raises(kb.KBConfigurationError, match="INFERENCE_OPTIMIZER_FA_KB_PATH"):
            kb.check_kb_configuration()

    def test_blank_value_is_not_a_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exported-but-empty variable expresses no intent, so it cannot fail the run."""
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", "   ")
        kb.check_kb_configuration()

    def test_passes_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        kb.check_kb_configuration()

    def test_fa_cli_refuses_to_start(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`fa` runs standalone, so it carries its own copy of the gate.

        Guards the reason the check moved out of the resolver: the failure has
        to reach the operator, and every read-path caller swallows exceptions.
        """
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "legacy"))
        assert cli.main(["kb", "list"]) == 2


class TestMigrateLegacyPartition:
    """The one-time carry-over from ``<workspace>/kb`` to ``<workspace>/framework-kb``."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        return tmp_path

    @staticmethod
    def _seed_legacy(tmp_path: Path, body: str = '{"pr_url": "PR-1"}') -> Path:
        legacy = tmp_path / "workspace" / "kb" / "framework_optimization"
        legacy.mkdir(parents=True)
        (legacy / "lessons.jsonl").write_text(body, encoding="utf-8")
        return legacy

    def test_carries_the_ledger_to_the_new_root(self, tmp_path: Path) -> None:
        """The writer was already working, so a real deployment has data to move."""
        self._seed_legacy(tmp_path)

        destination = kb.migrate_legacy_partition_once()

        assert destination == tmp_path / "workspace" / "framework-kb" / "framework_optimization"
        assert (destination / "lessons.jsonl").read_text(encoding="utf-8") == '{"pr_url": "PR-1"}'
        assert kb.read_pr_ledger() == [{"pr_url": "PR-1"}]

    def test_leaves_the_source_in_place(self, tmp_path: Path) -> None:
        """A copy, not a move: the operator decides when the old root goes."""
        legacy = self._seed_legacy(tmp_path)

        kb.migrate_legacy_partition_once()

        assert (legacy / "lessons.jsonl").exists()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        self._seed_legacy(tmp_path)

        assert kb.migrate_legacy_partition_once() is not None
        assert kb.migrate_legacy_partition_once() is None

    def test_never_overwrites_a_live_partition(self, tmp_path: Path) -> None:
        """A destination already in use wins; the migration is not a repair tool."""
        self._seed_legacy(tmp_path, '{"pr_url": "OLD"}')
        live = tmp_path / "workspace" / "framework-kb" / "framework_optimization"
        live.mkdir(parents=True)
        (live / "lessons.jsonl").write_text('{"pr_url": "LIVE"}', encoding="utf-8")

        assert kb.migrate_legacy_partition_once() is None
        assert kb.read_pr_ledger() == [{"pr_url": "LIVE"}]

    def test_skipped_when_the_operator_named_a_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The legacy default was never the operator's location to inherit."""
        self._seed_legacy(tmp_path)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "chosen"))

        assert kb.migrate_legacy_partition_once() is None

    def test_no_legacy_data_is_not_an_error(self, tmp_path: Path) -> None:
        assert kb.migrate_legacy_partition_once() is None


class TestMigrationCannotStopTheRun:
    """A convenience copy must never be able to take a session down with it.

    A missing ledger is a cold start, which the FRAMEWORK phase handles, so the
    worst outcome of a failed migration is re-proposing a PR. A session that
    disabled the phase entirely never reads this KB at all. Neither justifies
    refusing to start, which is what a full disk or one unreadable file would
    otherwise have caused.
    """

    @pytest.fixture(autouse=True)
    def _legacy_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        legacy = tmp_path / "workspace" / "kb" / "framework_optimization"
        legacy.mkdir(parents=True)
        (legacy / "lessons.jsonl").write_text('{"pr_url": "PR-1"}', encoding="utf-8")

    @staticmethod
    def _break_copy(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
        def _raise(*_a: object, **_k: object) -> None:
            raise exc

        monkeypatch.setattr(kb.shutil, "copytree", _raise)

    def test_io_failure_warns_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        with caplog.at_level("WARNING"):
            assert kb.migrate_legacy_partition_once() is None

        assert "could not carry the legacy partition over" in caplog.text

    def test_permission_failure_warns_and_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._break_copy(monkeypatch, PermissionError(13, "Permission denied"))

        assert kb.migrate_legacy_partition_once() is None

    def test_unexpected_failure_still_does_not_escape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not just OSError: nothing this helper can hit is worth the session."""
        self._break_copy(monkeypatch, ValueError("something nobody predicted"))

        assert kb.migrate_legacy_partition_once() is None

    def test_start_up_still_completes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole start-up sequence, not just the helper, survives it."""
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        kb.prepare_kb_environment()

    def test_a_failed_copy_leaves_nothing_behind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No staging dir survives; ``list_domains`` would report it as a domain."""
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        kb.migrate_legacy_partition_once()

        workspace = tmp_path / "workspace"
        assert not [p for p in workspace.glob("*.migrating*")]
        assert kb.list_domains() == []

    def test_staging_is_unique_and_outside_the_kb_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Concurrent start-ups must not stage onto one another's directory.

        A fixed staging name let one process's cleanup delete the copy another
        was still writing, turning two healthy start-ups into a failed one.
        """
        seen: list[Path] = []
        real_copytree = kb.shutil.copytree

        def _record(src: object, dst: object, *a: object, **k: object) -> object:
            seen.append(Path(str(dst)))
            return real_copytree(src, dst, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(kb.shutil, "copytree", _record)

        kb.migrate_legacy_partition_once()

        assert len(seen) == 1
        staging = seen[0]
        assert str(os.getpid()) in staging.name
        assert kb.mutable_kb_root() not in staging.parents
        assert staging.parent == tmp_path / "workspace"

    def test_the_loser_of_a_race_gives_up_quietly(self, tmp_path: Path) -> None:
        """Second writer finds the destination populated and stands down."""
        assert kb.migrate_legacy_partition_once() is not None
        assert kb.migrate_legacy_partition_once() is None
        assert kb.read_pr_ledger() == [{"pr_url": "PR-1"}]


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
        (kb_root / "kernel_agent").mkdir()
        assert kb.list_domains() == ["framework", "kernel_agent"]
        names = sorted(p.name for p in kb.get_domain_files("framework"))
        assert names == ["README.md", "empirical_kb.md"]

    def test_match_domains_keyword_hit(self) -> None:
        """A task description containing whitelisted keywords picks the right domain."""
        domains = kb._match_domains("improve sglang vllm cudagraph attention")
        assert "framework" in domains
        assert "kernel_agent" in domains

    def test_match_domains_no_hit(self) -> None:
        """When no keyword matches, the result is an empty list."""
        assert kb._match_domains("totally unrelated free-form text") == []

    def test_match_domains_atom_keyword_hit(self) -> None:
        """``atom`` must hit the framework domain."""
        domains = kb._match_domains("improve atom moe throughput on mi300x")
        assert "framework" in domains, f"atom must hit the framework domain; got {domains!r}"

    def test_atom_in_framework_domain_keywords_constant(self) -> None:
        """Constant-level guard: ``atom`` must appear in the framework domain's keyword list."""
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
        d = kb_root / "kernel_agent"
        d.mkdir()
        (d / "empirical_kb.md").write_text("Quirk-Wibble is the new HotAcronym for stuff.")
        out = kb.select_kb("quirk-wibble")
        assert out and out[0].domain == "kernel_agent"


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
        b = kb_root / "kernel_agent"
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
        rc = cli.main(
            [
                "kb",
                "contribute",
                "--domain",
                "framework",
                "--body",
                "hello",
                "--source",
                "test",
                "--session-id",
                "s1",
            ]
        )
        assert rc == 0
        capsys.readouterr()
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
        self,
        kb_root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
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
        rc = cli.main(
            [
                "kb",
                "synthesize",
                "--domain",
                "framework",
                "--findings",
                str(findings_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Synthesised findings - framework" in out
        assert "### winner PR:1" in out
        assert "1234.5" in out

    def test_synthesize_with_llm_missing_sdk_exit_two(
        self,
        kb_root: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
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
        self,
        kb_root: Path,
        framework: str,
    ) -> None:
        path = kb.path_for_framework(framework)
        assert path == kb_root / "framework_optimization" / framework

    def test_path_lowercases_and_strips(self, kb_root: Path) -> None:
        path_a = kb.path_for_framework("  Atom  ")
        path_b = kb.path_for_framework("ATOM")
        path_c = kb.path_for_framework("atom")
        assert path_a == path_b == path_c

    def test_empty_framework_returns_partition_root(self, kb_root: Path) -> None:
        assert kb.path_for_framework("") == kb_root / "framework_optimization"
        assert kb.path_for_framework("   ") == kb_root / "framework_optimization"

    def test_partition_dir_not_created_eagerly(self, kb_root: Path) -> None:
        # path_for_framework is read-only; it must NOT create the dir.
        _ = kb.path_for_framework("atom")
        assert not (kb_root / "framework_optimization" / "atom").exists()
