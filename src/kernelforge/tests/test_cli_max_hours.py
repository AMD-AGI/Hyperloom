# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``--max-hours`` guard on the forge-loop command.

A run shorter than MIN_MAX_HOURS can't complete a productive campaign (the time
reserve would block iterations, or the budget exhausts immediately), so the CLI
rejects it up front. These tests require no LLM / GPU / gateway."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

import kernelforge.config as config_module
from kernelforge.cli import (
    LONG_HORIZON_THRESHOLD_HOURS,
    MIN_MAX_HOURS,
    _is_long_horizon,
    _initial_remote_publication_state,
    _record_remote_publication_result,
    _remote_publication_view,
    _warm_start_publication_covers,
    _validate_max_hours,
    main,
)


def test_validate_max_hours_rejects_below_minimum():
    with pytest.raises(click.BadParameter):
        _validate_max_hours(None, None, MIN_MAX_HOURS - 0.1)


def test_validate_max_hours_accepts_minimum_and_above():
    assert _validate_max_hours(None, None, MIN_MAX_HOURS) == MIN_MAX_HOURS
    assert _validate_max_hours(None, None, 8.0) == 8.0
    # None (option unset) passes through untouched.
    assert _validate_max_hours(None, None, None) is None


def test_validate_max_hours_floor_is_not_env_overridable(monkeypatch):
    # The floor exists because the loop won't start an iteration once less than
    # budget_reserve_sec (900s) of the budget remains: below the floor a campaign
    # finalizes having done little or nothing and still exits 0. No env escape
    # hatch may weaken it, otherwise CI can go green on an empty campaign.
    monkeypatch.setenv("KF_CI_SMOKE", "1")
    with pytest.raises(click.BadParameter):
        _validate_max_hours(None, None, 0.1)


@pytest.mark.parametrize(
    ("max_hours", "expected"),
    [
        (1.0, False),
        (LONG_HORIZON_THRESHOLD_HOURS, False),
        (LONG_HORIZON_THRESHOLD_HOURS + 0.0001, True),
        (8.0, True),
    ],
)
def test_long_horizon_requires_more_than_two_hours(
    max_hours,
    expected,
):
    assert _is_long_horizon(max_hours) is expected


def test_removed_max_turns_environment_variable_warns_and_is_ignored(
    monkeypatch,
    caplog,
    request,
):
    monkeypatch.setenv("KERNEL_AGENTS_MAX_TURNS", "17")
    config_module._warn_removed_max_turns_env.cache_clear()
    request.addfinalizer(config_module._warn_removed_max_turns_env.cache_clear)
    with caplog.at_level("WARNING", logger=config_module.log.name):
        assert config_module.Config.from_env().max_turns == 500
    assert config_module.Config.from_env(max_turns=321).max_turns == 321
    assert caplog.text.count("KERNEL_AGENTS_MAX_TURNS is no longer supported") == 1


def test_forge_loop_rejects_short_max_hours():
    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--kernel",
            "k.py",
            "--driver",
            "d.py",
            "--workspace",
            ".",
            "--max-hours",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "must be >=" in result.output


def test_max_hours_help_describes_long_horizon_agents():
    result = CliRunner().invoke(main, ["forge-loop", "--help"])

    assert result.exit_code == 0
    assert "Analysis profiling" in result.output
    assert "Implementer" in result.output
    assert "Plan Critic" in result.output


def test_forge_loop_rejects_an_unregistered_producer():
    # A producer names an index in the KB identity scheme. Accepting a free
    # string here would publish under an address nothing ever reads back,
    # and the failure would only surface as a permanently cold warm start.
    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--kernel",
            "k.py",
            "--driver",
            "d.py",
            "--workspace",
            ".",
            "--producer",
            "not-a-producer",
        ],
    )
    assert result.exit_code != 0
    assert "--producer must be one of" in result.output
    assert "fusion" in result.output


def test_forge_loop_refuses_to_return_after_a_read_it_was_told_not_to_do():
    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--kernel",
            "k.py",
            "--driver",
            "d.py",
            "--workspace",
            ".",
            "--no-kb-warmstart",
            "--return-after-read-kb",
        ],
    )
    assert result.exit_code != 0
    assert "--no-kb-warmstart" in result.output


def test_legacy_loop_command_is_removed():
    result = CliRunner().invoke(main, ["loop", "--help"])

    assert result.exit_code != 0
    assert "No such command 'loop'" in result.output


def test_remote_publication_view_marks_only_latest_best_authoritative():
    published = _remote_publication_view(
        {
            "status": "published",
            "pending_commit": "",
            "published_commit": "best-2",
            "last_attempted_commit": "best-2",
            "last_result": {"written": True},
        },
        "best-2",
    )
    assert published["best_commit"] == "best-2"
    assert published["local_best_commit"] == "best-2"
    assert published["published_commit"] == "best-2"
    assert published["pending_commit"] == ""
    assert published["latest_best_published"] is True
    assert "last_result" not in published

    pending = _remote_publication_view(
        {
            "status": "pending_retry",
            "pending_commit": "best-3",
            "published_commit": "best-2",
        },
        "best-3",
    )
    assert pending["best_commit"] == "best-3"
    assert pending["pending_commit"] == "best-3"
    assert pending["published_commit"] == "best-2"
    assert pending["latest_best_published"] is False

    refined = _remote_publication_view(
        {
            "status": "not_better_than_kb",
            "pending_commit": "",
            "published_commit": "best-2",
        },
        "best-2",
    )
    assert refined["latest_best_published"] is True


def test_zero_keep_warmstart_is_authoritatively_already_published():
    state = _initial_remote_publication_state(
        {
            "applied": True,
            "applied_commit": "warm-local-commit",
            "solution_slug": "kernelforge-exp/op/existing-solution",
        }
    )

    assert _warm_start_publication_covers(state, "warm-local-commit")
    publication = _remote_publication_view(state, "warm-local-commit")
    assert publication["latest_best_published"] is True
    assert publication["published_commit"] == "warm-local-commit"
    assert publication["best_commit"] == "warm-local-commit"
    assert publication["pending_commit"] == ""
    assert publication["source"] == "existing_warm_start_solution"
    assert publication["state"] == "materialized_from_remote"
    assert publication["solution_slug"] == ("kernelforge-exp/op/existing-solution")
    assert state["last_result"] == {
        "written": False,
        "reason": "existing_warm_start_solution",
        "solution": "kernelforge-exp/op/existing-solution",
    }


def test_later_keep_publication_supersedes_warmstart_authority():
    state = _initial_remote_publication_state(
        {
            "applied": True,
            "applied_commit": "warm-local-commit",
            "solution_slug": "kernelforge-exp/op/existing-solution",
        }
    )
    state["pending_commit"] = "keep-commit"
    state["last_attempted_commit"] = "keep-commit"

    _record_remote_publication_result(
        state,
        commit="keep-commit",
        result={
            "written": True,
            "solution": "kernelforge-exp/op/campaign-solution",
        },
    )

    publication = _remote_publication_view(state, "keep-commit")
    assert publication["latest_best_published"] is True
    assert publication["published_commit"] == "keep-commit"
    assert publication["best_commit"] == "keep-commit"
    assert publication["pending_commit"] == ""
    assert publication["source"] == "campaign_publication"
    assert publication["state"] == "published"
    assert publication["solution_slug"] == ("kernelforge-exp/op/campaign-solution")


def test_failed_later_keep_preserves_warm_publication_and_marks_pending():
    state = _initial_remote_publication_state(
        {
            "applied": True,
            "applied_commit": "warm-local-commit",
            "solution_slug": "kernelforge-exp/op/existing-solution",
        }
    )
    state["pending_commit"] = "keep-commit"

    _record_remote_publication_result(
        state,
        commit="keep-commit",
        result={"written": False, "reason": "error:timeout"},
    )

    publication = _remote_publication_view(state, "keep-commit")
    assert publication["latest_best_published"] is False
    assert publication["published_commit"] == "warm-local-commit"
    assert publication["pending_commit"] == "keep-commit"
    assert publication["status"] == "pending_retry"
