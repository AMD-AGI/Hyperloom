"""Tests for CLI wiring of upstream PR references."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hyperloom_common.env import EnvValueError
from kernelforge.cli import forge_loop
from kernelforge.config import Config
from kernelforge.workflows.forge_loop.bootstrap import run as forge_loop_run
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationConfig
from kernelforge.knowledge.pr_knowledge.context import PrReferenceQuery
from kernelforge.knowledge.pr_knowledge.references import PRRefsResult, refs_dir
from kernelforge.knowledge.pr_knowledge.search import PRReference
from kernelforge.workflows.forge_loop.bootstrap.pr_references import (
    _collect_pr_references,
    _git_remote_url,
    _pr_kb_enabled,
    _pr_refs_event_fields,
)


def test_switch_defaults_to_off(monkeypatch):
    """Default the feature off when flag and environment are unset."""
    monkeypatch.delenv("PR_KB_ENABLE", raising=False)

    assert _pr_kb_enabled(None) is False


def test_cli_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    assert _pr_kb_enabled(False) is False

    monkeypatch.setenv("PR_KB_ENABLE", "0")
    assert _pr_kb_enabled(True) is True


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_environment_fallback_accepts_common_truthy_spellings(monkeypatch, raw):
    monkeypatch.setenv("PR_KB_ENABLE", raw)

    assert _pr_kb_enabled(None) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
def test_environment_fallback_rejects_everything_else(monkeypatch, raw):
    monkeypatch.setenv("PR_KB_ENABLE", raw)

    assert _pr_kb_enabled(None) is False


def test_an_unreadable_environment_fallback_is_not_silently_off(monkeypatch):
    """Whether the loop writes to the KB is not a question to answer by guessing."""
    monkeypatch.setenv("PR_KB_ENABLE", "maybe")

    with pytest.raises(EnvValueError, match="PR_KB_ENABLE"):
        _pr_kb_enabled(None)


def test_both_switch_forms_are_exposed():
    names = {param.name: param for param in forge_loop.params}

    assert "pr_kb" in names
    assert names["pr_kb"].default is None, "unset must fall through to the env"
    assert set(names["pr_kb"].opts) >= {"--pr-kb"}
    assert set(names["pr_kb"].secondary_opts) >= {"--no-pr-kb"}


def test_switch_is_independent_of_the_experience_kb_flag():
    """--no-experience-kb must not disable an unrelated feature."""
    names = {param.name: param for param in forge_loop.params}

    assert names["experience_kb"].default is True
    assert names["pr_kb"].default is None


def test_the_collected_pr_context_reaches_the_implementer_task(monkeypatch, tmp_path):
    from kernelforge.knowledge.pr_knowledge import references as pr_monitor_refs

    found = PRRefsResult(
        prompt_context="PR CONTEXT",
        references=(PRReference(repo="ROCm/aiter", number=1),),
        repo="ROCm/aiter",
    )
    monkeypatch.setattr(pr_monitor_refs, "collect_references", lambda **_kwargs: found)
    run = SimpleNamespace(
        iter_config=IterationConfig(kernel_file="kernel.py", driver_script="driver.py", workspace_dir=str(tmp_path)),
        config=Config(workspace=str(tmp_path), agent_backend="codex"),
        campaign=SimpleNamespace(kernel_backend="aiter", source_files=(), operator_name=""),
        deadline=SimpleNamespace(remaining=lambda reserve: 3600.0, finalize_reserve_sec=0.0),
        request=SimpleNamespace(
            budget=SimpleNamespace(max_hours=1.0),
            roles=SimpleNamespace(permission_mode=None),
            knowledge=SimpleNamespace(pr_kb=True),
        ),
    )

    task = forge_loop_run._implementer_setup(run, ["kernel.py"], 600, False).task

    assert (task.pre_task_context, task.pr_kb_repo) == ("PR CONTEXT", "ROCm/aiter")


def test_remote_url_is_read_from_the_workspace(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "remote", "add", "origin", "git@github.com:ROCm/aiter.git"],
        check=True,
    )

    assert _git_remote_url(tmp_path) == "git@github.com:ROCm/aiter.git"


def test_missing_remote_yields_an_empty_string(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    assert _git_remote_url(tmp_path) == ""


def test_non_repository_yields_an_empty_string(tmp_path):
    assert _git_remote_url(tmp_path) == ""


def test_remote_lookup_failure_is_contained(monkeypatch, tmp_path):
    def explode(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(subprocess, "run", explode)

    assert _git_remote_url(tmp_path) == ""


def test_refresh_event_carries_its_counters():
    fields = _pr_refs_event_fields("", {"candidates": 5, "surfaced": 3, "injected_entries": 3, "http_calls": 8})

    assert fields["position"] == "A"
    assert fields["reason"] == "ok"
    assert fields["candidates"] == 5
    assert fields["http_calls"] == 8


def test_reason_is_recorded_when_nothing_was_injected():
    assert _pr_refs_event_fields("repo_unresolved", {})["reason"] == "repo_unresolved"


def test_degraded_reason_is_surfaced_alongside_an_ok_reason():
    fields = _pr_refs_event_fields("", {"degraded_reason": "service_unreachable"})

    assert fields["reason"] == "ok"
    assert fields["degraded_reason"] == "service_unreachable"


def test_a_failed_lookup_degrades_instead_of_raising(monkeypatch, tmp_path):
    """A broken PR KB must never decide the outcome of the campaign."""
    from kernelforge.knowledge.pr_knowledge import references as pr_monitor_refs

    def full_disk(**_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(pr_monitor_refs, "collect_references", full_disk)

    assert (
        _collect_pr_references(
            str(tmp_path),
            PrReferenceQuery(
                kernel_backend="aiter",
                git_remote="",
                source_files=(),
                operator_name="moe",
                target_functions=(),
            ),
            1.0,
        )
        is None
    )


def test_a_service_failure_is_absorbed_like_a_local_one(monkeypatch, tmp_path):
    from kernelforge.knowledge.pr_knowledge import references as pr_monitor_refs
    from kernelforge.knowledge.pr_knowledge.client import PRTransportError

    def unreachable(**_kwargs):
        raise PRTransportError("timeout on /healthz")

    monkeypatch.setattr(pr_monitor_refs, "collect_references", unreachable)

    assert (
        _collect_pr_references(
            str(tmp_path),
            PrReferenceQuery(
                kernel_backend="aiter",
                git_remote="",
                source_files=(),
                operator_name="moe",
                target_functions=(),
            ),
            1.0,
        )
        is None
    )


def test_an_unexpected_failure_is_not_swallowed(monkeypatch, tmp_path):
    """Only service, parsing, and filesystem failures are absorbed."""
    from kernelforge.knowledge.pr_knowledge import references as pr_monitor_refs

    def bug(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pr_monitor_refs, "collect_references", bug)

    with pytest.raises(KeyboardInterrupt):
        _collect_pr_references(
            str(tmp_path),
            PrReferenceQuery(
                kernel_backend="aiter",
                git_remote="",
                source_files=(),
                operator_name="moe",
                target_functions=(),
            ),
            1.0,
        )


def test_the_caller_has_the_result_before_provenance_is_written(tmp_path):
    provenance = refs_dir(str(tmp_path)) / "provenance.json"
    provenance_at_publication = []
    run = SimpleNamespace(
        deadline=SimpleNamespace(remaining=lambda reserve: 0.0),
        usage=SimpleNamespace(totals=dict),
        tracker=None,
        request=SimpleNamespace(
            artifacts=SimpleNamespace(result_json=""),
            campaign=SimpleNamespace(workspace_dir=str(tmp_path)),
        ),
        iter_config=SimpleNamespace(pr_reference_labels=("ROCm/aiter#1",)),
        publish_result=lambda _result: provenance_at_publication.append(provenance.exists()),
    )
    loop_runner = SimpleNamespace(_checkpoint_llm_usage=lambda: None, experiment=None)
    reporter = SimpleNamespace(
        publish_final_best=lambda llm_summary: {},
        build_result=lambda kb_experience: {"best_iteration": 1, "experiment_id": "exp-1"},
    )

    forge_loop_run._publish(run, loop_runner, reporter)

    assert provenance_at_publication == [False]
    assert provenance.exists()
