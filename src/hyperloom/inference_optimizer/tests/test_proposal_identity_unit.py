# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared proposal/variant identity.

The fingerprints below were captured from the explore executor's own identity
block before it was refactored onto this helper. They are pinned rather than
recomputed: a change here re-keys ``explore_search["tested"]``, so every
resumed session would re-bench its whole history.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._proposal_identity import (
    controls_of,
    effective_fingerprint,
    is_executable,
    normalize_proposal,
)


# (label, extra_args, extra_envs, variant controls, base controls, expected fingerprint)
_GOLDEN = [
    ("plain", "--max-num-seqs 128", {"A": "1"}, ([], [], "append"), ([], [], ""), "caacd6bf1da76201"),
    ("variant-only", "--max-num-seqs 128", {"A": "1"}, (["--x"], ["E"], "append"), ([], [], ""), "39cfaf303d04885d"),
    ("base-only", "--max-num-seqs 128", {"A": "1"}, ([], [], "append"), (["--b"], ["BE"], ""), "e85521a43890c743"),
    ("both", "--max-num-seqs 128", {"A": "1"}, (["--x"], ["E"], "append"), (["--b"], ["BE"], ""), "1e630973fad523ab"),
    ("overlap", "--max-num-seqs 128", {}, (["--b"], [], "append"), (["--b"], [], ""), "9cb8463804e7a11c"),
    ("variant-replace", "", {"A": "1"}, ([], [], "replace"), ([], [], ""), "7f4d4d32a84df525"),
    ("base-replace", "", {"A": "1"}, ([], [], "append"), ([], [], "replace"), "7f4d4d32a84df525"),
    ("removal-only", "", {}, (["--enable-prefix-caching"], [], "append"), ([], [], ""), "ca4d2e9e9760543a"),
    ("empty", "", {}, ([], [], "append"), ([], [], ""), "164374825086dc65"),
]


def _controls(remove_args, unset_envs, args_mode) -> dict:
    return controls_of(
        normalize_proposal({"remove_args": remove_args, "unset_envs": unset_envs, "args_mode": args_mode})
    )


@pytest.mark.parametrize(
    "args,envs,variant,base,expected",
    [row[1:] for row in _GOLDEN],
    ids=[row[0] for row in _GOLDEN],
)
def test_effective_fingerprint_matches_the_pinned_executor_values(args, envs, variant, base, expected):
    b_remove, b_unset, b_mode = base
    assert (
        effective_fingerprint(
            args,
            envs,
            controls=_controls(*variant),
            base_remove_args=b_remove,
            base_unset_envs=b_unset,
            base_args_mode=b_mode,
        )
        == expected
    )


def test_base_controls_change_the_fingerprint():
    controls = _controls(["--x"], [], "append")
    assert effective_fingerprint("--a 1", {}, controls=controls) != effective_fingerprint(
        "--a 1", {}, controls=controls, base_remove_args=["--b"]
    )


def test_removal_union_is_base_first_and_deduped():
    both = effective_fingerprint("", {}, controls=_controls(["--b", "--v"], [], "append"), base_remove_args=["--b"])
    assert both == effective_fingerprint("", {}, controls=_controls(["--b", "--v"], [], "append"))


@pytest.mark.parametrize(
    "proposal,expected",
    [
        ({"extra_args": "--a 1"}, True),
        ({"extra_server_args": "--a 1"}, True),
        ({"extra_envs": {"A": "1"}}, True),
        ({"remove_args": ["--a"]}, True),
        ({"unset_envs": ["A"]}, True),
        ({"args_mode": "replace"}, True),
        ({"name": "research-only", "reason": "read the scheduler"}, False),
        ({"extra_args": "  ", "extra_envs": {}}, False),
    ],
)
def test_is_executable(proposal, expected):
    assert is_executable(normalize_proposal(proposal)) is expected


def test_normalize_resolves_the_args_alias_and_keeps_atomic():
    fields = normalize_proposal(
        {
            "name": " coupled ",
            "extra_server_args": " --a 1 ",
            "extra_envs": {"A": 1},
            "remove_args": "--drop",
            "args_mode": "REPLACE",
            "atomic": True,
            "reason": "needs the paired headroom",
        }
    )
    assert fields == {
        "name": "coupled",
        "extra_args": "--a 1",
        "extra_envs": {"A": "1"},
        "remove_args": ["--drop"],
        "unset_envs": [],
        "args_mode": "replace",
        "atomic": True,
        "reason": "needs the paired headroom",
    }


def test_controls_of_drops_defaults():
    assert controls_of(normalize_proposal({"extra_args": "--a 1"})) == {}
    assert controls_of(normalize_proposal({"remove_args": ["--x"], "args_mode": "replace"})) == {
        "remove_args": ["--x"],
        "args_mode": "replace",
    }
