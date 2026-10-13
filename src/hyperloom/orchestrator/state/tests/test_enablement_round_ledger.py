# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The invariants ``EnablementRound`` keeps over its own stack records."""

from __future__ import annotations

from hyperloom.orchestrator.state.shared_state import EnablementRound


def _row(seq, task, *, outcome="applied", digest="d"):
    return {
        "seq": seq,
        "round_task_id": task,
        "cmd_digest": digest,
        "outcome": outcome,
        "round_disposition": "unreported",
        "at_accepted_round": False,
        "present_at_final_launch": False,
        "replayed_at_final_launch": False,
    }


def test_the_first_recorded_base_sha_is_never_replaced():
    rnd = EnablementRound()
    assert rnd.record_base_sha("/repo", "aaa") is True
    assert rnd.record_base_sha("/repo", "bbb") is False
    assert rnd.base_sha_by_root == {"/repo": "aaa"}


def test_the_first_recorded_patch_root_is_never_replaced():
    rnd = EnablementRound()
    rnd.record_patch_roots({"1.patch": "/aiter"})
    rnd.record_patch_roots({"1.patch": "/repo", "2.patch": "/repo"})
    assert rnd.patch_roots == {"1.patch": "/aiter", "2.patch": "/repo"}


def test_inherited_base_shas_prefer_the_durable_map_over_root_records():
    rnd = EnablementRound(
        base_sha_by_root={"/repo": "first"},
        roots=[
            {"path": "/repo", "base_sha": "later"},
            {"path": "/aiter", "base_sha": "a1"},
            {"path": "/wheel", "base_sha": ""},
        ],
    )
    assert rnd.inherited_base_shas() == {"/repo": "first", "/aiter": "a1"}


def test_accepted_patch_roots_keep_an_earlier_binding_and_default_to_this_root():
    rnd = EnablementRound(kept_patches=["1.patch"], patch_roots={"1.patch": "/aiter"})
    roots = rnd.accepted_patch_roots(
        done_payload={"patch_roots": {"1.patch": "/repo", "stray.patch": "/elsewhere"}},
        applied=["2.patch"],
        framework_root="/repo",
    )
    assert roots == {"1.patch": "/aiter", "2.patch": "/repo"}


def test_the_accepted_stack_is_kept_patches_then_this_round():
    rnd = EnablementRound(kept_patches=["1.patch", "2.patch"])
    assert rnd.accepted_stack(["3.patch"]) == ["1.patch", "2.patch", "3.patch"]


def test_kept_rounds_project_deduped_patches_and_last_wins_artifacts():
    rnd = EnablementRound()
    rnd.push_kept_round(
        task_id="r1",
        patches=["1.patch"],
        artifacts=[{"target": "/t/a.py", "source": "r1-a"}, {"target": "/t/b.py", "source": "r1-b"}],
    )
    rnd.push_kept_round(
        task_id="r2", patches=["1.patch", "2.patch"], artifacts=[{"target": "/t/a.py", "source": "r2-a"}]
    )

    assert [r["task_id"] for r in rnd.kept_rounds] == ["r1", "r2"]
    assert rnd.kept_patches == ["1.patch", "2.patch"]
    assert {a["target"]: a["source"] for a in rnd.kept_artifacts} == {"/t/a.py": "r2-a", "/t/b.py": "r1-b"}


def test_setup_commands_stack_once_each_in_the_order_they_ran():
    rnd = EnablementRound(setup_commands=["pip install a"])
    rnd.stack_setup_commands(["pip install b", "pip install a", "pip install b"])
    assert rnd.setup_commands == ["pip install a", "pip install b"]


def test_a_keep_replaces_observations_and_keeps_an_unobservable_tristate():
    rnd = EnablementRound(patch_targets={"old": {}}, base_sha="kept", build_extensions_not_carried=[])
    rnd.record_keep({"enablement_build_extensions_not_carried": None, "enablement_launch_argv_refused": True})
    assert rnd.patch_targets == {}
    assert rnd.base_sha == "kept"
    assert rnd.build_extensions_not_carried is None
    assert rnd.levers_without_readers == []
    assert rnd.launch_argv_refused is True


def test_the_setup_ledger_only_grows():
    rnd = EnablementRound(setup_executions=[_row(3, "r1")])
    rnd.append_setup_execution(_row(4, "r1"))
    assert [r["seq"] for r in rnd.setup_executions] == [3, 4]
    assert rnd.last_execution_seq() == 4
    assert EnablementRound().last_execution_seq() == 0


def test_the_accepted_round_takes_presence_from_an_earlier_one():
    rnd = EnablementRound(setup_executions=[_row(1, "r1", digest="foo"), _row(2, "r2", digest="foo")])
    rnd.mark_setup_round("r1", "kept", accepted=True)
    rnd.mark_setup_round("r2", "kept", accepted=True)
    first, second = rnd.setup_executions
    assert (first["at_accepted_round"], first["present_at_final_launch"]) == (False, False)
    assert (second["at_accepted_round"], second["present_at_final_launch"]) == (True, True)
    assert first["replayed_at_final_launch"] is True


def test_a_discarded_round_keeps_its_rows_off_the_final_launch():
    rnd = EnablementRound(setup_executions=[_row(1, "r1")])
    rnd.mark_setup_round("r1", "reverted", accepted=False)
    (row,) = rnd.setup_executions
    assert row["round_disposition"] == "reverted"
    assert row["present_at_final_launch"] is False


def test_a_round_with_no_task_id_claims_no_ledger_rows():
    rnd = EnablementRound(setup_executions=[_row(1, "r1")])
    rnd.mark_setup_round("r1", "kept", accepted=True)
    rnd.mark_setup_round("", "kept", accepted=True)
    (row,) = rnd.setup_executions
    assert row["round_disposition"] == "kept"
    assert row["present_at_final_launch"] is True
