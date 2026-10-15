# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A config lever that only drops a flag is still a config lever."""

from __future__ import annotations

from hyperloom.orchestrator.phases.framework import _framework_config_levers_from_done

_RADIX = "--disable-radix-cache"


def _done(*proposals, patches=None):
    return {"patches_written": list(patches or []), "proposal_set": list(proposals)}


def test_a_proposal_that_only_removes_a_flag_is_the_lever():
    """``--disable-radix-cache`` is baked into the launcher; removing it is the whole change."""
    lever = _framework_config_levers_from_done(
        _done(
            {"name": "radix-cache-on", "extra_args": "", "extra_envs": {}, "remove_args": [_RADIX]},
            {"name": "radix-plus-mtp-eagle", "extra_args": "--speculative-algorithm EAGLE"},
        )
    )

    assert lever["name"] == "radix-cache-on"
    assert lever["remove_args"] == [_RADIX]
    assert lever["extra_server_args"] == ""


def test_the_lever_names_the_proposal_its_arguments_came_from():
    """Reading the name off proposal[0] launched one proposal's name against another's arguments."""
    lever = _framework_config_levers_from_done(
        _done(
            {"name": "nothing-to-apply", "extra_args": "", "extra_envs": {}},
            {"name": "the-real-one", "extra_args": "--speculative-algorithm NEXTN"},
        )
    )

    assert lever["name"] == "the-real-one"
    assert lever["extra_server_args"] == "--speculative-algorithm NEXTN"


def test_an_unsetting_proposal_counts_too():
    lever = _framework_config_levers_from_done(
        _done({"name": "drop-env", "extra_args": "", "extra_envs": {}, "unset_envs": ["SGLANG_X"]})
    )

    assert lever["name"] == "drop-env"
    assert lever["unset_envs"] == ["SGLANG_X"]


def test_a_replace_mode_survives_to_the_launcher():
    lever = _framework_config_levers_from_done(
        _done({"name": "replace-all", "extra_args": "--tp 8", "args_mode": "replace"})
    )

    assert lever["args_mode"] == "replace"


def test_an_unknown_args_mode_falls_back_to_append():
    lever = _framework_config_levers_from_done(_done({"name": "x", "extra_args": "--tp 8", "args_mode": "clobber"}))

    assert lever["args_mode"] == "append"


def test_a_proposal_with_nothing_in_it_is_still_not_a_lever():
    assert _framework_config_levers_from_done(_done({"name": "empty", "extra_args": "", "extra_envs": {}})) == {}
    assert _framework_config_levers_from_done(_done()) == {}
    assert _framework_config_levers_from_done(None) == {}
