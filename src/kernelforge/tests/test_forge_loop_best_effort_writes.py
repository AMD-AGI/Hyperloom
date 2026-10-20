"""Best-effort forge-loop writes report I/O and data failures and let any other failure surface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernelforge.cli.forge_loop_command as forge_loop_command

_REPORTER = forge_loop_command._RunReporter


class _FailingTracker:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    def set_llm_usage(self, experiment_id, usage):
        raise self.failure

    def set_kb_experience(self, experiment_id, kb_experience):
        raise self.failure

    def set_checkpoint(self, experiment_id, checkpoint):
        raise self.failure


class _LlmUsageOnlyTracker(_FailingTracker):
    def set_kb_experience(self, experiment_id, kb_experience):
        return None


def _llm_usage(failure: Exception) -> None:
    forge_loop_command._record_experiment_accounting(
        _LlmUsageOnlyTracker(failure), experiment_id="exp1", llm_usage={"calls": 1}, kb_experience={"read": True}
    )


class _KbOnlyTracker(_FailingTracker):
    def set_llm_usage(self, experiment_id, usage):
        return None


def _kb_experience(failure: Exception) -> None:
    forge_loop_command._record_experiment_accounting(
        _KbOnlyTracker(failure), experiment_id="exp1", llm_usage={"calls": 1}, kb_experience={"read": True}
    )


def _result_json(failure: Exception, monkeypatch, tmp_path) -> None:
    def fail(path, payload):
        raise failure

    monkeypatch.setattr(forge_loop_command, "atomic_write_json", fail)
    reporter = SimpleNamespace(
        run=SimpleNamespace(request=SimpleNamespace(artifacts=SimpleNamespace(result_json=str(tmp_path / "r.json"))))
    )
    _REPORTER.write_result_json(reporter, {})


def _checkpoint(failure: Exception, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        forge_loop_command,
        "_search_anchors",
        lambda loop_runner: SimpleNamespace(search_start_ms=1.0, search_start_mean_case_speedup=1.0, pristine_ms=1.0),
    )
    reporter = SimpleNamespace(
        loop_runner=SimpleNamespace(experiment=None),
        run=SimpleNamespace(
            caller_experiment_id="caller1",
            campaign=SimpleNamespace(base_commit="base"),
            iter_config=SimpleNamespace(workspace_dir=str(tmp_path)),
            tracker=_FailingTracker(failure),
        ),
        build_result=lambda kb_experience: {},
        write_result_json=lambda result: None,
    )
    _REPORTER.checkpoint_on_best_committed(reporter, SimpleNamespace(kept=True, commit_hash="abc"))


_SITES = {
    "llm_usage": lambda failure, monkeypatch, tmp_path: _llm_usage(failure),
    "kb_experience": lambda failure, monkeypatch, tmp_path: _kb_experience(failure),
    "result_json": _result_json,
    "checkpoint": _checkpoint,
}

_REPORTED = {
    "llm_usage": ("err", "Warning: failed to record LLM usage for experiment exp1: {failure}\n"),
    "kb_experience": ("err", "Warning: failed to record KB experience for experiment exp1: {failure}\n"),
    "result_json": ("out", "  [forge-loop] result-json snapshot skipped ({failure})\n"),
    "checkpoint": ("out", "  [checkpoint] skipped ({failure})\n"),
}


@pytest.mark.parametrize("site", list(_SITES))
def test_unexpected_failure_surfaces(site, monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="^programming error$"):
        _SITES[site](RuntimeError("programming error"), monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "failure",
    [OSError("disk full"), ValueError("corrupt record"), TypeError("not serializable")],
    ids=lambda failure: type(failure).__name__,
)
@pytest.mark.parametrize("site", list(_SITES))
def test_expected_failure_is_reported(site, failure, monkeypatch, tmp_path, capsys):
    _SITES[site](failure, monkeypatch, tmp_path)

    captured = capsys.readouterr()
    stream, template = _REPORTED[site]
    assert getattr(captured, stream) == template.format(failure=failure)
