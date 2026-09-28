# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The invariants ``EnablementRound`` keeps over its own stack records."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import EnablementRound


def test_the_first_recorded_base_sha_is_never_replaced():
    rnd = EnablementRound()
    assert rnd.record_base_sha("/repo", "aaa") is True
    assert rnd.record_base_sha("/repo", "bbb") is False
    assert rnd.recorded_base_shas() == {"/repo": "aaa"}


def test_inherited_base_shas_prefer_the_durable_map_over_root_records():
    rnd = EnablementRound(
        base_sha_by_root={"/repo": "first"},
        roots=[{"path": "/repo", "base_sha": "later"}, {"path": "/aiter", "base_sha": "a1"}, "junk"],
    )
    assert rnd.inherited_base_shas() == {"/repo": "first", "/aiter": "a1"}


def test_the_setup_ledger_only_grows():
    rnd = EnablementRound(setup_executions=[{"seq": 3}])
    assert rnd.append_setup_executions([{"seq": 4}, "not-a-row"]) is True
    assert rnd.append_setup_executions([]) is False
    assert rnd.setup_executions == [{"seq": 3}, {"seq": 4}]
    assert rnd.last_execution_seq() == 4


def test_accepted_patch_roots_keep_an_earlier_binding_and_default_to_this_root():
    rnd = EnablementRound(kept_patches=["1.patch"], patch_roots={"1.patch": "/aiter"})
    roots = rnd.accepted_patch_roots(done_payload=None, applied=["2.patch"], framework_root="/repo")
    assert roots == {"1.patch": "/aiter", "2.patch": "/repo"}
