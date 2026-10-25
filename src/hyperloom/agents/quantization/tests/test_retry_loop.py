"""Tests for `quantize_via_prompt` — retry orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from hyperloom.agents.quantization.driver.outcomes import OutcomeId
from hyperloom.agents.quantization.driver.retry import (
    QuantSkillRunResult,
    _decide_next_step,
    _read_counter,
    _resolve_interactive,
    quantize_via_prompt,
)
from hyperloom.agents.quantization.driver.runner import AttemptContext, AttemptResult, _attempt_finished


# ───────────────────────────────────────────────────────────────────────────── helpers
# ─────────────────────────────────────────────────────────────────────────────


class _StubRunner:
    """Records calls + lets each attempt mutate workspace before returning."""

    def __init__(self, side_effects: list[Callable[[Path, int], str | None]]):
        self.side_effects = side_effects
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, attempt_context: AttemptContext, **kwargs: Any) -> AttemptResult:
        self.calls.append(attempt_context)
        workspace = Path(attempt_context.workspace)
        attempt = attempt_context.attempt_number
        idx = attempt - 1
        sdk_error = ""
        if idx < len(self.side_effects):
            err = self.side_effects[idx](workspace, attempt)
            if err:
                sdk_error = err
        return AttemptResult(workspace=workspace, sdk_error=sdk_error)


def _materialize_clean_workspace(build_workspace, **overrides):
    """Build a complete-success workspace at a known path."""

    return build_workspace(**overrides)


@pytest.fixture
def quark_root(tmp_path: Path) -> Path:
    """A dummy quark_root dir so the bootstrap check passes."""

    qr = tmp_path / "quark_root"
    qr.mkdir()
    return qr


# ───────────────────────────────────────────────────────────────────────────── bootstrap — quark_root resolution
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quark_root_missing_fast_path(tmp_path, monkeypatch):
    # QUARK_ROOT unset -> bootstrap fails without using an internal default path.
    monkeypatch.delenv("QUARK_ROOT", raising=False)

    async def _never_called(**kwargs: Any) -> AttemptResult:  # pragma: no cover
        raise AssertionError("runner should not be invoked when bootstrap fails")

    result = await quantize_via_prompt(
        "do it",
        workspace=tmp_path / "ws",
        runner_fn=_never_called,
    )
    assert result.status == "failed"
    assert result.assessment.final == OutcomeId.quark_root_missing
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
async def test_quark_root_nonexistent_dir_fast_path(tmp_path):
    async def _never_called(**kwargs: Any) -> AttemptResult:  # pragma: no cover
        raise AssertionError("runner should not be invoked")

    result = await quantize_via_prompt(
        "do it",
        workspace=tmp_path / "ws",
        quark_root=tmp_path / "does_not_exist",
        runner_fn=_never_called,
    )
    assert result.status == "failed"
    assert result.assessment.final == OutcomeId.quark_root_missing


# ───────────────────────────────────────────────────────────────────────────── happy path — single clean attempt
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_clean_attempt_returns_success(tmp_path, quark_root, build_workspace):
    ws = tmp_path / "ws"

    def populate(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace)
        return None

    runner = _StubRunner([populate])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
    )
    assert result.status == "success"
    assert result.assessment.final is None
    assert result.assessment.attempts == (None,)
    assert result.quantized_model_dir is not None
    assert result.quantized_model_dir.exists()
    assert len(runner.calls) == 1


# ───────────────────────────────────────────────────────────────────────────── retry hypothesis gate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_retry_without_fix_hypothesis(tmp_path, quark_root, build_workspace):
    """ASK_RETRYABLE outcome but no fix_hypothesis_attempt_2.md → no retry."""

    ws = tmp_path / "ws"

    def populate(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_weights=False,
            include_validation_report=False,
        )
        return None

    runner = _StubRunner([populate])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=3,
    )
    assert len(runner.calls) == 1  # gated; no retry
    assert result.assessment.final == OutcomeId.must_have_weights_missing
    assert any("no_fix_hypothesis" in n for n in result.assessment.notes)


@pytest.mark.asyncio
async def test_retry_with_hypothesis_then_recover(tmp_path, quark_root, build_workspace):
    """Attempt 1 fails with weights missing, hypothesis written, attempt 2 clean."""

    ws = tmp_path / "ws"

    def fail_then_hypothesize(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_weights=False,
            include_validation_report=False,
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text(
            "## Fix\nRerun export with disk space cleared.\n", encoding="utf-8"
        )
        return None

    def succeed(workspace: Path, attempt: int) -> str | None:
        # Clear any prior broken state, then build a clean workspace.
        for f in workspace.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                import shutil

                shutil.rmtree(f)
        build_workspace(workspace=workspace)
        return None

    runner = _StubRunner([fail_then_hypothesize, succeed])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=2,
    )
    assert len(runner.calls) == 2
    assert result.assessment.final is None
    assert result.assessment.recovered is True
    assert result.assessment.attempts == (OutcomeId.must_have_weights_missing, None)
    assert result.status == "success"


@pytest.mark.asyncio
async def test_a_retry_does_not_start_with_the_previous_attempts_end(tmp_path, quark_root, build_workspace):
    """Every attempt shares the workspace, and an attempt's end is judged from the files in it.

    Attempt 1 stops the way SKILL.md section 8 says: ``blocked.md`` plus a fix
    hypothesis. If those end markers are still there, attempt 2 counts as
    finished before its session does anything, so a session that stops after
    one reply is never continued and its outcome is attempt 1's.
    """

    ws = tmp_path / "ws"
    finished_at_start: list[bool] = []

    def block_with_hypothesis(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_weights=False,
            last_phase="export",
            blocked_md="outcome_id: export_crashed\n",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("## Fix\nExport shard by shard.\n", encoding="utf-8")
        return None

    def stop_at_once(workspace: Path, attempt: int) -> str | None:
        finished_at_start.append(_attempt_finished(workspace))
        return None

    runner = _StubRunner([block_with_hypothesis, stop_at_once])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=1,
    )

    assert len(runner.calls) == 2
    assert finished_at_start == [False]
    assert result.assessment.attempts[0] == OutcomeId.export_crashed
    assert result.assessment.attempts[1] != OutcomeId.export_crashed
    assert "export_crashed" in (ws / "blocked.md.attempt_1").read_text(encoding="utf-8")
    assert (ws / "last_phase.txt.attempt_1").read_text(encoding="utf-8").strip() == "export"


@pytest.mark.asyncio
async def test_a_retry_that_writes_nothing_does_not_pass_on_the_previous_attempts_model(
    tmp_path, quark_root, build_workspace
):
    """The previous attempt's model and validation report stay for the retry to build on.

    With that attempt's end archived, they are also everything a retry that
    stops at once leaves behind, and the classifier would grade them as a
    clean success the retry never produced.
    """

    ws = tmp_path / "ws"

    def block_after_validation(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_eval_report=False,
            last_phase="eval",
            blocked_md="outcome_id: unclassified_failure\n",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("## Fix\nPin the eval seed.\n", encoding="utf-8")
        return None

    runner = _StubRunner([block_after_validation, lambda workspace, attempt: None])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=1,
    )

    assert len(runner.calls) == 2
    assert result.assessment.attempts == (OutcomeId.unclassified_failure, OutcomeId.unclassified_failure)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_attempt,expected",
    [
        (lambda workspace, attempt: None, OutcomeId.unclassified_failure),
        (
            lambda workspace, attempt: (
                (workspace / "last_phase.txt").write_text("exec\n", encoding="utf-8") and "CUDA out of memory"
            ),
            OutcomeId.exec_oom,
        ),
    ],
)
async def test_an_unfinished_retry_is_judged_on_what_it_raised_not_on_inherited_reports(
    tmp_path, quark_root, build_workspace, second_attempt, expected
):
    """An inherited validation report with an auto-recover failure would grade the retry ``partial``."""

    ws = tmp_path / "ws"

    def block_after_validation(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            validation_tag="aux_fail",
            include_eval_report=False,
            last_phase="validate",
            blocked_md="outcome_id: unclassified_failure\n",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("## Fix\nCopy the aux files.\n", encoding="utf-8")
        return None

    runner = _StubRunner([block_after_validation, second_attempt])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=1,
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure, expected)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
async def test_a_retry_that_only_marks_the_eval_phase_does_not_pass_on_the_previous_model(
    tmp_path, quark_root, build_workspace
):
    """``last_phase.txt`` is written at the start of a phase, so it says nothing about what the attempt produced."""

    ws = tmp_path / "ws"

    def block_after_validation(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_eval_report=False,
            last_phase="eval",
            blocked_md="outcome_id: unclassified_failure\n",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("## Fix\nPin the eval seed.\n", encoding="utf-8")
        return None

    def mark_eval(workspace: Path, attempt: int) -> str | None:
        (workspace / "last_phase.txt").write_text("eval\n", encoding="utf-8")
        return None

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=_StubRunner([block_after_validation, mark_eval]),
        interactive=False,
        max_requantize_attempts=1,
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure, OutcomeId.unclassified_failure)
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_a_rerun_after_a_run_that_wrote_nothing_does_not_pass_on_an_older_model(
    tmp_path, quark_root, build_workspace
):
    """The run in between leaves no end to archive, yet the workspace still holds the older run's outputs."""

    ws = tmp_path / "ws"

    async def run(side_effect):
        return await quantize_via_prompt(
            "fp8 it", workspace=ws, quark_root=quark_root, runner_fn=_StubRunner([side_effect]), interactive=False
        )

    first = await run(lambda workspace, attempt: build_workspace(workspace=workspace, last_phase="eval") and None)
    await run(lambda workspace, attempt: None)
    third = await run(lambda workspace, attempt: None)

    assert first.status == "success"
    assert third.assessment.attempts == (OutcomeId.unclassified_failure,)
    assert third.status == "failed"


@pytest.mark.asyncio
async def test_a_rerun_on_the_same_workspace_is_judged_on_its_own_end(tmp_path, quark_root, build_workspace):
    """The --quantize prelude gives every run of one model the same workspace."""

    ws = tmp_path / "ws"
    build_workspace(workspace=ws, include_weights=False, last_phase="export", blocked_md="outcome_id: export_crashed\n")
    (ws / "blocked.md.attempt_1").write_text("outcome_id: exec_oom\n", encoding="utf-8")
    finished_at_start: list[bool] = []

    def succeed(workspace: Path, attempt: int) -> str | None:
        finished_at_start.append(_attempt_finished(workspace))
        build_workspace(workspace=workspace, last_phase="eval")
        return None

    result = await quantize_via_prompt(
        "fp8 it", workspace=ws, quark_root=quark_root, runner_fn=_StubRunner([succeed]), interactive=False
    )

    assert finished_at_start == [False]
    assert result.status == "success"
    assert result.assessment.attempts == (None,)
    assert "exec_oom" in (ws / "blocked.md.attempt_1").read_text(encoding="utf-8")
    assert "export_crashed" in (ws / "blocked.md.attempt_2").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_counter_persists_and_caps_retries(tmp_path, quark_root, build_workspace):
    """Even with hypothesis present, counter ≥ max → no retry."""

    ws = tmp_path / "ws"
    ws.mkdir()
    # Pre-seed the counter at the max.
    (ws / "requantize_attempts.txt").write_text("1", encoding="utf-8")

    def fail(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_weights=False,
            include_validation_report=False,
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
        return None

    runner = _StubRunner([fail])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=1,
    )
    assert len(runner.calls) == 1
    assert any("max_attempts_exhausted" in n for n in result.assessment.notes)
    # Counter should still be 1 (no bump beyond the cap).
    assert _read_counter(ws) == 1


@pytest.mark.asyncio
async def test_max_requantize_attempts_zero_no_retry(tmp_path, quark_root, build_workspace):
    ws = tmp_path / "ws"

    def fail(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_weights=False,
            include_validation_report=False,
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
        return None

    runner = _StubRunner([fail])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=0,
    )
    assert len(runner.calls) == 1
    assert any("max_attempts_exhausted" in n for n in result.assessment.notes)


# ───────────────────────────────────────────────────────────────────────────── auto_fail stops immediately
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_fail_stops_immediately(tmp_path, quark_root, build_workspace):
    ws = tmp_path / "ws"

    def md5_fail(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, validation_tag="md5_fail")
        # AUTO_FAIL stops the loop even with a hypothesis present.
        (workspace / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
        return None

    runner = _StubRunner([md5_fail])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=3,
    )
    assert len(runner.calls) == 1
    assert result.assessment.final == OutcomeId.must_validate_md5_mismatch
    assert result.status == "failed"
    assert any("auto_fail" in n for n in result.assessment.notes)


# ───────────────────────────────────────────────────────────────────────────── eval_gap_exceeded with operator
# acceptance ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eval_gap_exceeded_accepted_by_operator_promotes_to_accepted(
    tmp_path, quark_root, build_workspace, monkeypatch
):
    ws = tmp_path / "ws"

    def gap_exceeded(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            eval_report={
                "metric_name": "gsm8k",
                "dataset": "gsm8k",
                "backend": "vllm",
                "source_score": 0.5,
                "quantized_score": 0.4,
                "relative_gap": 0.20,
            },
        )
        return None

    # Stub the operator prompt to answer "y".
    from hyperloom.agents.quantization.driver import retry as retry_mod

    monkeypatch.setattr(retry_mod, "_ask_operator", lambda msg: True)

    runner = _StubRunner([gap_exceeded])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=True,
        acceptable_eval_gap=0.03,
    )
    assert result.assessment.final == OutcomeId.eval_gap_accepted
    assert result.assessment.attempts[-1] == OutcomeId.eval_gap_accepted
    assert result.status == "success"
    assert result.quantized_model_dir is not None


@pytest.mark.asyncio
async def test_eval_gap_exceeded_rejected_stays_partial(tmp_path, quark_root, build_workspace, monkeypatch):
    ws = tmp_path / "ws"

    def gap_exceeded(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            eval_report={
                "metric_name": "gsm8k",
                "dataset": "gsm8k",
                "backend": "vllm",
                "source_score": 0.5,
                "quantized_score": 0.4,
                "relative_gap": 0.20,
            },
        )
        return None

    from hyperloom.agents.quantization.driver import retry as retry_mod

    monkeypatch.setattr(retry_mod, "_ask_operator", lambda msg: False)

    runner = _StubRunner([gap_exceeded])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=True,
        acceptable_eval_gap=0.03,
    )
    assert result.assessment.final == OutcomeId.eval_gap_exceeded
    assert result.status == "partial"


# ───────────────────────────────────────────────────────────────────────────── auto_recover surfaces as partial
# (Python doesn't loop on it) ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_recover_surfaced_does_not_retry(tmp_path, quark_root, build_workspace):
    ws = tmp_path / "ws"

    def env_unavailable(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_eval_report=False,
            eval_skipped_reason="docker missing",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
        return None

    runner = _StubRunner([env_unavailable])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        quark_root=quark_root,
        runner_fn=runner,
        interactive=False,
        max_requantize_attempts=3,
    )
    assert len(runner.calls) == 1
    assert result.assessment.final == OutcomeId.eval_env_unavailable
    assert result.status == "partial"
    assert any("auto_recover_unresolved" in n for n in result.assessment.notes)


# ───────────────────────────────────────────────────────────────────────────── _resolve_interactive
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_interactive_explicit_true():
    assert _resolve_interactive(True) is True


def test_resolve_interactive_explicit_false():
    assert _resolve_interactive(False) is False


def test_resolve_interactive_auto_no_tty(monkeypatch):
    import sys

    class _NotTTY:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _NotTTY())
    monkeypatch.setattr(sys, "stderr", _NotTTY())
    assert _resolve_interactive(None) is False


# ───────────────────────────────────────────────────────────────────────────── _decide_next_step unit table
# ─────────────────────────────────────────────────────────────────────────────


def test_decide_next_step_none_outcome(tmp_path):
    d = _decide_next_step(
        None,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=3,
        counter=0,
    )
    assert d.retry is False
    assert d.promote_to is None


def test_decide_next_step_auto_fail(tmp_path):
    d = _decide_next_step(
        OutcomeId.must_validate_md5_mismatch,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=3,
        counter=0,
    )
    assert d.retry is False
    assert "auto_fail" in d.note


def test_decide_next_step_auto_recover(tmp_path):
    d = _decide_next_step(
        OutcomeId.eval_env_unavailable,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=3,
        counter=0,
    )
    assert d.retry is False
    assert "auto_recover_unresolved" in d.note


def test_decide_next_step_ask_retryable_with_hypothesis(tmp_path):
    (tmp_path / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
    d = _decide_next_step(
        OutcomeId.exec_oom,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=2,
        counter=0,
    )
    assert d.retry is True


def test_decide_next_step_ask_retryable_without_hypothesis(tmp_path):
    d = _decide_next_step(
        OutcomeId.exec_oom,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=2,
        counter=0,
    )
    assert d.retry is False
    assert d.note == "no_fix_hypothesis"


def test_decide_next_step_unclassified_failure(tmp_path):
    (tmp_path / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
    d = _decide_next_step(
        OutcomeId.unclassified_failure,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=False,
        max_requantize_attempts=2,
        counter=0,
    )
    assert d.retry is True


def test_decide_next_step_checkpoint_aborted_never_retries(tmp_path):
    (tmp_path / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
    d = _decide_next_step(
        OutcomeId.checkpoint_aborted,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=True,
        max_requantize_attempts=2,
        counter=0,
    )
    assert d.retry is False
    assert "checkpoint_aborted" in d.note


def test_decide_next_step_eval_gap_accepted_promotes(tmp_path, monkeypatch):
    from hyperloom.agents.quantization.driver import retry as rmod

    monkeypatch.setattr(rmod, "_ask_operator", lambda msg: True)
    d = _decide_next_step(
        OutcomeId.eval_gap_exceeded,
        fix_hypothesis=tmp_path / "fix_hypothesis_attempt_2.md",
        interactive=True,
        max_requantize_attempts=0,
        counter=0,
    )
    assert d.retry is False
    assert d.promote_to == OutcomeId.eval_gap_accepted


# ───────────────────────────────────────────────────────────────────────────── QuantSkillRunResult shape
# ─────────────────────────────────────────────────────────────────────────────


def test_quant_skill_run_result_is_frozen():
    from hyperloom.agents.quantization.driver.assessment import Assessment

    r = QuantSkillRunResult(
        status="success",
        quantized_model_dir=None,
        assessment=Assessment(final=None, attempts=(None,), recovered=False, eval_gap=None),
    )
    with pytest.raises(Exception):
        r.status = "failed"  # type: ignore[misc]
