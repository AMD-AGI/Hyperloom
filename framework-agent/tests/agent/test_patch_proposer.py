"""patch_proposer LLM-loop tests (P3 PR-G, driver-injected)."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.agent.ast_scanner import AstScanResult
from framework_agent.agent.flag_discovery import DiscoveredFlag
from framework_agent.agent.kb_priors import KbEntry
from framework_agent.agent.patch_proposer import (
    ProposedPatch,
    ProposerInput,
    propose_patch,
)


def _flag(name: str = "--max-num-seqs") -> DiscoveredFlag:
    return DiscoveredFlag(
        flag_name=name,
        module="vllm.engine.arg_utils",
        source_path="/sgl/vllm/vllm/engine/arg_utils.py",
        line=10,
        via="argparse",
        type_hint="int",
        default_repr="256",
        help_text="",
        surface="cli",
        framework="vllm",
    )


def _ast_result(mode: str = "libcst") -> AstScanResult:
    return AstScanResult(
        flags=[_flag(), _flag("max_num_seqs")],
        mode=mode,  # type: ignore[arg-type]
        files_scanned=2,
        parse_failures=0,
    )


_FAKE_DIFF = """\
--- a/vllm/engine/arg_utils.py
+++ b/vllm/engine/arg_utils.py
@@ -10 +10 @@
-    max_num_seqs: int = 256
+    max_num_seqs: int = 512
"""


def _driver_keep(prompt: str, *, max_turns: int) -> tuple[str, str, float]:
    """Fixture LLM driver: returns a real diff with predicted +5% gain."""
    return _FAKE_DIFF, "Bump max_num_seqs to 512", 5.0


def _driver_no_patch(prompt: str, *, max_turns: int) -> tuple[str, str, float]:
    """Fixture LLM driver: returns empty diff (flag-discovery-only)."""
    return "", "No patch warranted; flags already in grid", 0.0


def _driver_high_gain(prompt: str, *, max_turns: int) -> tuple[str, str, float]:
    return _FAKE_DIFF, "Major refactor", 12.0


# ---------------------------------------------------------------------------
def test_propose_patch_writes_diff_to_runs_framework(tmp_path: Path):
    inp = ProposerInput(
        ast_findings=_ast_result(),
        kb_priors=[],
        target_framework="vllm",
        task_id="fw-test-001",
        session_dir=tmp_path,
    )
    result = propose_patch(inp, driver=_driver_keep)
    assert isinstance(result, ProposedPatch)
    expected = tmp_path / "runs" / "framework" / "fw-test-001" / "proposal.diff"
    assert Path(result.path) == expected.resolve()
    assert expected.read_text(encoding="utf-8") == _FAKE_DIFF


def test_propose_patch_records_predicted_gain_and_rationale(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_keep,
    )
    assert result.predicted_gain_pct == 5.0
    assert result.rationale == "Bump max_num_seqs to 512"


def test_propose_patch_extracts_files_touched_from_unified_diff(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_keep,
    )
    assert result.files_touched == ("vllm/engine/arg_utils.py",)


def test_propose_patch_empty_diff_yields_empty_path(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_no_patch,
    )
    assert result.path == ""
    assert result.diff_text == ""
    assert result.predicted_gain_pct == 0.0


def test_propose_patch_confidence_downgrades_on_grep_fallback(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(mode="grep_fallback"),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_high_gain,
    )
    # Even with 12% gain, fallback mode pins confidence to "low".
    assert result.confidence == "low"


def test_propose_patch_confidence_high_at_8pct_gain(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(mode="libcst"),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_high_gain,
    )
    assert result.confidence == "high"


def test_propose_patch_confidence_medium_at_5pct(tmp_path: Path):
    result = propose_patch(
        ProposerInput(
            ast_findings=_ast_result(mode="libcst"),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=_driver_keep,
    )
    assert result.confidence == "medium"


def test_propose_patch_prompt_includes_kb_priors(tmp_path: Path):
    """The driver receives a prompt that mentions KB priors."""
    captured: dict[str, str] = {}

    def driver(prompt: str, *, max_turns: int) -> tuple[str, str, float]:
        captured["prompt"] = prompt
        return _FAKE_DIFF, "ok", 5.0

    priors = [
        KbEntry(
            entry_id="fw-pitfall-001",
            title="block_manager refactor OOM",
            body="...",
            category="pitfall",
            target_framework="vllm",
        ),
    ]
    propose_patch(
        ProposerInput(
            ast_findings=_ast_result(),
            kb_priors=priors,
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=driver,
    )
    assert "fw-pitfall-001" in captured["prompt"]
    assert "pitfall" in captured["prompt"]


def test_propose_patch_prompt_lists_ast_flags(tmp_path: Path):
    captured: dict[str, str] = {}

    def driver(prompt: str, *, max_turns: int) -> tuple[str, str, float]:
        captured["prompt"] = prompt
        return _FAKE_DIFF, "ok", 5.0

    propose_patch(
        ProposerInput(
            ast_findings=_ast_result(),
            target_framework="vllm",
            task_id="fw-t",
            session_dir=tmp_path,
        ),
        driver=driver,
    )
    assert "--max-num-seqs" in captured["prompt"]
