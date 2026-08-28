"""Tests for CLI wiring of upstream PR references."""

from __future__ import annotations

import inspect
import subprocess

import pytest
from click.testing import CliRunner

from kernelforge.cli import (
    _collect_pr_references,
    _git_remote_url,
    _pr_kb_enabled,
    _pr_refs_event_fields,
    forge_loop,
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


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "maybe"])
def test_environment_fallback_rejects_everything_else(monkeypatch, raw):
    monkeypatch.setenv("PR_KB_ENABLE", raw)

    assert _pr_kb_enabled(None) is False


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


def test_pr_context_is_wired_only_to_the_implementer():
    """Keep external PR text out of measured Analysis and planning evidence."""
    source = inspect.getsource(forge_loop.callback)
    implementer = source[source.index("agent_fn = make_agent_fn(") : source.index("effective_implementer =")]
    analysis = source[source.index("analysis_service = make_analysis_agent_service(") : source.index("analysis_mode =")]
    orchestration = source[
        source.index("orchestration_service = make_orchestration_service(") : source.index(
            "supervisor_fn = make_supervisor_fn("
        )
    ]
    supervisor = source[
        source.index("supervisor_fn = make_supervisor_fn(") : source.index("loop_runner = IterationLoop(")
    ]

    assert "pre_task_context=pr_task_context" in implementer
    assert "pr_kb_repo=pr_kb_repo" in implementer
    assert "pr_task_context" not in analysis
    assert "pr_task_context" not in orchestration
    assert "pr_task_context" not in supervisor


def test_help_text_mentions_the_default_being_off():
    result = CliRunner().invoke(forge_loop, ["--help"])

    assert result.exit_code == 0
    assert "--pr-kb" in result.output
    assert "--no-pr-kb" in result.output


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


def test_position_a_never_writes_events_itself(tmp_path):
    """Do not create the campaign sentinel before loop initialization."""
    _pr_refs_event_fields("", {"candidates": 1})

    assert not (tmp_path / "forge_experiments" / "events.jsonl").exists()


def test_loop_appends_the_event_only_after_the_campaign_guard(tmp_path):
    """Append the event only after campaign initialization."""
    from kernelforge.loop.runner import IterationConfig, IterationLoop

    assert "pr_kb_event" in {f for f in IterationConfig.__dataclass_fields__}
    source = inspect.getsource(IterationLoop._run_locked)
    guard = source.index("already contains a campaign")
    append = source.index("pr_refs_refreshed")
    assert append > guard, "the event append must follow the fresh-campaign guard"


def test_a_failed_lookup_degrades_instead_of_raising(monkeypatch, tmp_path):
    """A broken PR KB must never decide the outcome of the campaign."""
    from kernelforge.knowledge import pr_monitor_refs

    def full_disk(**_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(pr_monitor_refs, "collect_references", full_disk)

    assert (
        _collect_pr_references(
            workspace_dir=str(tmp_path),
            kernel_backend="aiter",
            git_remote="",
            source_files=(),
            operator_name="moe",
            target_functions=(),
            budget_sec=1.0,
        )
        is None
    )


def test_a_service_failure_is_absorbed_like_a_local_one(monkeypatch, tmp_path):
    from kernelforge.knowledge import pr_monitor_refs
    from kernelforge.knowledge.pr_monitor_client import PRTransportError

    def unreachable(**_kwargs):
        raise PRTransportError("timeout on /healthz")

    monkeypatch.setattr(pr_monitor_refs, "collect_references", unreachable)

    assert (
        _collect_pr_references(
            workspace_dir=str(tmp_path),
            kernel_backend="aiter",
            git_remote="",
            source_files=(),
            operator_name="moe",
            target_functions=(),
            budget_sec=1.0,
        )
        is None
    )


def test_an_unexpected_failure_is_not_swallowed(monkeypatch, tmp_path):
    """Only service, parsing, and filesystem failures are absorbed."""
    from kernelforge.knowledge import pr_monitor_refs

    def bug(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pr_monitor_refs, "collect_references", bug)

    with pytest.raises(KeyboardInterrupt):
        _collect_pr_references(
            workspace_dir=str(tmp_path),
            kernel_backend="aiter",
            git_remote="",
            source_files=(),
            operator_name="moe",
            target_functions=(),
            budget_sec=1.0,
        )


def test_the_event_append_cannot_abort_the_campaign():
    """A failed observability write may only print, never propagate."""
    from kernelforge.loop.runner import IterationLoop

    source = inspect.getsource(IterationLoop._run_locked)
    append = source.index("pr_refs_refreshed")
    handler = source.index("except (OSError, ValueError)", append)
    following = source.index("self.ic.pr_kb_event = {}", append)

    assert handler < following, "the append must be guarded before it is cleared"


def test_result_is_emitted_before_provenance_is_written():
    source = inspect.getsource(forge_loop.callback)
    result = source.rindex("result = _build_result")
    emitted = source.index('click.echo(f"__FORGE_RESULT__', result)
    provenance = source.index("_write_pr_provenance(", emitted)

    assert emitted < provenance
