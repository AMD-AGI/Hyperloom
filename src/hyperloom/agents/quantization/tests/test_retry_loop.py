"""Tests for `quantize_via_prompt` — retry orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from hyperloom.agents.quantization.driver.outcomes import OutcomeId
from hyperloom.agents.quantization.driver.retry import (
    DEFAULT_QUARK_GIT_URL,
    QuantSkillRunOptions,
    QuantSkillRunResult,
    _decide_next_step,
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
        return AttemptResult(sdk_error=sdk_error)


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
    assert result.assessment.notes == (
        "quark_root is not configured (set $QUARK_ROOT or pass options=QuantSkillRunOptions(quark_root=...); "
        f"clone from {DEFAULT_QUARK_GIT_URL})",
    )
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
async def test_quark_root_nonexistent_dir_fast_path(tmp_path):
    async def _never_called(**kwargs: Any) -> AttemptResult:  # pragma: no cover
        raise AssertionError("runner should not be invoked")

    result = await quantize_via_prompt(
        "do it",
        workspace=tmp_path / "ws",
        options=QuantSkillRunOptions(quark_root=tmp_path / "does_not_exist"),
        runner_fn=_never_called,
    )
    assert result.status == "failed"
    assert result.assessment.final == OutcomeId.quark_root_missing
    assert result.assessment.notes == (
        f"quark_root path does not exist or is not a directory: {tmp_path / 'does_not_exist'} "
        "(set $QUARK_ROOT or pass options=QuantSkillRunOptions(quark_root=...); "
        f"clone from {DEFAULT_QUARK_GIT_URL})",
    )


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
        "fp8 it", workspace=ws, options=QuantSkillRunOptions(quark_root=quark_root, interactive=False), runner_fn=runner
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=3),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=2),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=runner,
    )

    assert len(runner.calls) == 2
    assert finished_at_start == [False]
    assert result.assessment.attempts[0] == OutcomeId.export_crashed
    assert result.assessment.attempts[1] != OutcomeId.export_crashed
    assert "export_crashed" in (result.run_dir / "blocked.md.attempt_1").read_text(encoding="utf-8")
    assert (result.run_dir / "last_phase.txt.attempt_1").read_text(encoding="utf-8").strip() == "export"


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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=runner,
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure, expected)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised",
    [
        None,
        # What a DeepSeek session that keeps replying without tool calls ends on (common/deepseek_task.py).
        "the owed artifact was still not written after 3 continuations",
        "CUDA out of memory",
    ],
)
async def test_a_retry_that_only_marks_the_eval_phase_does_not_pass_on_the_previous_model(
    tmp_path, quark_root, build_workspace, raised
):
    """``last_phase.txt`` is written at the start of a phase, so it says nothing about what the attempt produced.

    An error at eval would otherwise map to an auto-recover outcome, which ends
    the run ``partial`` on the earlier attempt's model.
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

    def mark_eval(workspace: Path, attempt: int) -> str | None:
        (workspace / "last_phase.txt").write_text("eval\n", encoding="utf-8")
        return raised

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=_StubRunner([block_after_validation, mark_eval]),
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure, OutcomeId.unclassified_failure)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
async def test_a_retry_that_stops_on_opening_eval_is_sent_back_to_finish_it(tmp_path, quark_root, build_workspace):
    """The owed-end check and the classifier must agree on what ends an attempt.

    A retry that stops right after writing ``last_phase.txt=eval`` has produced
    no result yet. Counting that as finished stops the session from being
    continued, and then the classifier fails it for having no end of its own.
    """

    ws = tmp_path / "ws"

    def block_after_validation(workspace: Path, attempt: int) -> str | None:
        build_workspace(
            workspace=workspace,
            include_eval_report=False,
            last_phase="validate",
            blocked_md="outcome_id: unclassified_failure\n",
        )
        (workspace / "fix_hypothesis_attempt_2.md").write_text("## Fix\nPin the eval seed.\n", encoding="utf-8")
        return None

    def stop_on_opening_eval(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, include_eval_report=False, last_phase="eval")
        if not _attempt_finished(workspace):
            # The owed-end check sends the session back, and it finishes eval.
            build_workspace(workspace=workspace, last_phase="eval")
        return None

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=_StubRunner([block_after_validation, stop_on_opening_eval]),
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure, None)
    assert result.status == "success"


@pytest.mark.asyncio
async def test_a_rerun_after_a_run_that_wrote_nothing_does_not_pass_on_an_older_model(
    tmp_path, quark_root, build_workspace
):
    """The first run's model stays in its own run directory; a later run that writes nothing does not pass on it."""

    ws = tmp_path / "ws"

    async def run(side_effect):
        return await quantize_via_prompt(
            "fp8 it",
            workspace=ws,
            options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
            runner_fn=_StubRunner([side_effect]),
        )

    first = await run(lambda workspace, attempt: build_workspace(workspace=workspace, last_phase="eval") and None)
    await run(lambda workspace, attempt: None)
    third = await run(lambda workspace, attempt: None)

    assert first.status == "success"
    assert third.assessment.attempts == (OutcomeId.unclassified_failure,)
    assert third.status == "failed"


@pytest.mark.asyncio
async def test_a_rerun_does_not_read_files_left_directly_in_the_model_workspace(tmp_path, quark_root, build_workspace):
    """Runs before per-run directories wrote straight into the model workspace; a new run neither reads nor moves them."""

    ws = tmp_path / "ws"
    build_workspace(workspace=ws, include_weights=False, last_phase="export", blocked_md="outcome_id: export_crashed\n")
    (ws / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
    finished_at_start: list[bool] = []

    def succeed(workspace: Path, attempt: int) -> str | None:
        finished_at_start.append(_attempt_finished(workspace))
        build_workspace(workspace=workspace, last_phase="eval")
        return None

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([succeed]),
    )

    assert finished_at_start == [False]
    assert result.status == "success"
    assert result.assessment.attempts == (None,)
    assert "export_crashed" in (ws / "blocked.md").read_text(encoding="utf-8")
    assert (ws / "fix_hypothesis_attempt_2.md").is_file()


@pytest.mark.asyncio
async def test_an_attempt_that_writes_no_end_does_not_succeed(tmp_path, quark_root):
    """A session that stops before writing anything leaves an empty run directory, which is no quantized model."""

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=tmp_path / "ws",
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([lambda workspace, attempt: None]),
    )

    assert result.assessment.attempts == (OutcomeId.unclassified_failure,)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


@pytest.mark.asyncio
async def test_an_exported_model_without_an_eval_end_does_not_succeed(tmp_path, quark_root, build_workspace):
    """SKILL.md owes an eval report or the reason eval was skipped; a model nobody evaluated is not a success."""

    def export_without_eval(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, include_eval_report=False, last_phase="eval")
        return None

    result = await quantize_via_prompt(
        "fp8 it",
        workspace=tmp_path / "ws",
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([export_without_eval]),
    )

    assert (result.run_dir / "run_manifest.yaml").is_file()
    assert result.assessment.attempts == (OutcomeId.unclassified_failure,)
    assert result.status == "failed"
    assert result.quantized_model_dir is None


def _fail_with_hypothesis(build_workspace):
    def fail(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, include_weights=False, include_validation_report=False)
        (workspace / f"fix_hypothesis_attempt_{attempt + 1}.md").write_text("x", encoding="utf-8")
        return None

    return fail


@pytest.mark.asyncio
async def test_retries_are_capped_per_run_and_a_rerun_gets_its_own(tmp_path, quark_root, build_workspace):
    """The --quantize prelude reruns a model on the same workspace; the earlier run's retries are not this run's."""

    ws = tmp_path / "ws"
    fail = _fail_with_hypothesis(build_workspace)
    for _ in range(2):
        runner = _StubRunner([fail, fail, fail])
        logged: list[str] = []
        result = await quantize_via_prompt(
            "fp8 it",
            workspace=ws,
            options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
            runner_fn=runner,
            log=logged.append,
        )
        assert len(runner.calls) == 2
        assert any("max_attempts_exhausted:counter=1/1" in n for n in result.assessment.notes)
        assert [m for m in logged if "retrying after" in m] == [
            "quantization-agent: retrying after outcome=must_have_weights_missing (retry 1/1)"
        ]


@pytest.mark.asyncio
async def test_a_rerun_does_not_retry_on_a_hypothesis_an_earlier_run_wrote(tmp_path, quark_root, build_workspace):
    ws = tmp_path / "ws"
    fail = _fail_with_hypothesis(build_workspace)

    def fail_without_hypothesis(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, include_weights=False, include_validation_report=False)
        return None

    first = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([fail]),
    )
    runner = _StubRunner([fail_without_hypothesis, fail])
    result = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=1),
        runner_fn=runner,
    )

    assert len(runner.calls) == 1
    assert any("no_fix_hypothesis" in n for n in result.assessment.notes)
    assert (first.run_dir / "fix_hypothesis_attempt_2.md").is_file()


@pytest.mark.asyncio
async def test_each_run_quantizes_in_its_own_run_directory(tmp_path, quark_root, build_workspace):
    """A rerun of the same model must not overwrite the model an earlier session may still be serving."""

    ws = tmp_path / "ws"
    attempt_dirs: list[Path] = []

    def succeed(workspace: Path, attempt: int) -> str | None:
        attempt_dirs.append(workspace)
        build_workspace(workspace=workspace, last_phase="eval")
        return None

    first = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([succeed]),
    )
    second = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([succeed]),
    )

    assert attempt_dirs == [first.run_dir, second.run_dir]
    assert first.run_dir != second.run_dir
    assert {run_dir.parent for run_dir in attempt_dirs} == {ws.resolve() / "runs"}
    assert first.quantized_model_dir.is_relative_to(first.run_dir)
    assert second.quantized_model_dir.is_relative_to(second.run_dir)


@pytest.mark.asyncio
async def test_a_rerun_does_not_take_the_eval_threshold_an_earlier_prompt_set(tmp_path, quark_root, build_workspace):
    """SKILL.md writes ``eval_gap_threshold.txt`` from the prompt it was given, so it belongs to that run only."""

    ws = tmp_path / "ws"
    gap = {"metric_name": "gsm8k", "source_score": 0.5, "quantized_score": 0.4, "relative_gap": 0.2}

    def accept_a_wide_gap(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, last_phase="eval", eval_report=gap, eval_gap_threshold=0.5)
        return None

    def measure_the_same_gap(workspace: Path, attempt: int) -> str | None:
        build_workspace(workspace=workspace, last_phase="eval", eval_report=gap)
        return None

    first = await quantize_via_prompt(
        "fp8 it, a 50% gap is fine",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([accept_a_wide_gap]),
    )
    second = await quantize_via_prompt(
        "fp8 it",
        workspace=ws,
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False),
        runner_fn=_StubRunner([measure_the_same_gap]),
    )

    assert first.status == "success"
    assert second.assessment.final == OutcomeId.eval_gap_exceeded


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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=0),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=3),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=True, acceptable_eval_gap=0.03),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=True, acceptable_eval_gap=0.03),
        runner_fn=runner,
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
        options=QuantSkillRunOptions(quark_root=quark_root, interactive=False, max_requantize_attempts=3),
        runner_fn=runner,
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
