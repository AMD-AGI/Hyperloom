# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.fake_gain import FAKE_GAINS
from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord, InboxItem
from hyperloom_multibranch.contracts.notebook.vocab import InboxKind
from hyperloom_multibranch.specialist import prompts
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.prompts import (
    ENTRY_CHARS,
    RECENT_ENTRIES,
    VERSION_HEX,
    launch_prompt,
    prompt_versions,
    role_prompt,
)
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.notebook.tools import UNFINISHED_TITLE
from hyperloom_multibranch.toolbox.mcp_tools import REPLY_INBOX_HEADER

WORKLOAD = "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64, InferenceX"
TOOLS = (
    "notebook_branch_log",
    "notebook_claim",
    "start_analysis",
    "worktree_sync",
    "worktree_check",
    "worktree_patch",
    "submit_test",
    "wait_test",
    "poll_inbox",
    "test_result",
    "notebook_append_branch_log",
    "notebook_learn",
    "request_repair",
    "decline_repair",
    "fix_case",
)


def _record(n: int, content: str = "", request_ids: tuple[str, ...] = ()) -> BranchLogRecord:
    entry = BranchLogEntry("s1", BranchLogType.RESULT, content or f"entry {n}", request_ids)
    return BranchLogRecord(f"e{n}", float(n), entry)


def _launch(*, brief: str = "", direction: str = "", **overrides) -> str:
    fields = {"head": "c7", "worktree": "/nfs/s1/worktree", "recent": ()}
    return launch_prompt(SessionStart("s1", "c0", 3, BranchTask(brief, direction=direction)), **{**fields, **overrides})


def test_the_role_names_the_workload_every_step_s_tool_and_the_rules():
    prompt = role_prompt(WORKLOAD.replace(", ", ",\n  "))

    assert f"The session optimizes one workload: {WORKLOAD}." in prompt
    assert all(tool in prompt for tool in TOOLS)
    assert 'Never submit purpose "confirm"' in prompt
    assert "target_share_pct" in prompt and "w/(1-w)" in prompt
    assert BASH_KILL_SAFETY_PREAMBLE in prompt
    assert "A fixable defect" in prompt and "An inherent trade-off" in prompt
    assert prompt.index("notebook_learn, citing the candidate") < prompt.index("then call decline_repair")
    with pytest.raises(ValueError, match="names the workload"):
        role_prompt(" \n")


def test_the_role_profiles_through_a_profile_test_and_tracelens_never_roofline():
    prompt = role_prompt(WORKLOAD)

    assert 'submit_test with kind "profile", purpose "profile"' in prompt
    assert 'start_analysis with tool "tracelens" and that test\'s id as profile_request_id' in prompt
    assert "roofline" not in prompt


def test_the_role_profiles_the_root_only_while_no_brief_lists_its_ceilings():
    prompt = role_prompt(WORKLOAD)

    assert "never profile the root" not in prompt
    assert "Profile the session's root only while no brief lists ceilings for it" in prompt
    assert "Profile a later head only once it carries gains you committed" in prompt


def test_the_role_says_an_explore_delta_carries_the_claimed_change():
    assert "an empty delta tests nothing" in role_prompt(WORKLOAD)


def test_the_role_names_the_config_delta_keys_that_recipe_delta_reads():
    prompt = role_prompt(WORKLOAD)
    example = prompt.split("with exactly these keys: ", 1)[1].split(". extra_server_args", 1)[0]

    config = RecipeDelta.from_document(json.loads(example)).config
    assert (config.server_args, dict(config.envs)) == ("--flag value --other-flag", {"NAME": "value"})
    assert "a key the delta does not define is refused, and the error lists the keys it takes" in prompt


def test_the_role_says_a_direction_another_branch_tests_comes_back_as_a_peer_outcome():
    prompt = role_prompt(WORKLOAD)

    assert (
        "another branch is testing it (testing_elsewhere), whose outcome then reaches you as a peer_outcome" in prompt
    )


def test_the_role_goes_back_to_choosing_after_a_result_and_never_stops_its_branch():
    prompt = role_prompt(WORKLOAD)

    assert "Then go back to step 2: the session decides when your branch stops, not you." in prompt


def test_the_role_works_its_direction_first_and_then_moves_to_one_no_branch_covers():
    prompt = role_prompt(WORKLOAD)

    assert "Your launch prompt names your branch's direction: work on it first." in prompt
    assert "dig deeper in it rather than switching to settings" in prompt
    assert "take another of the session's directions, preferring one no running branch covers" in prompt
    assert "server.log fixes; GEMM tuning; Fusion; Server arguments and environment variables; ROCm code gaps" in prompt


def test_the_role_carries_what_found_gains_on_rocm_with_examples():
    prompt = role_prompt(WORKLOAD)

    assert "What has found gains on ROCm before" in prompt
    assert "into the kernel that consumes its output" in prompt
    assert "it may already be optimal for them" in prompt
    assert prompt.index("What has found gains") < prompt.index("Examples:") < prompt.index("How a candidate is judged")


def test_the_role_points_at_a_bf16_gemm_left_among_fp8_ones_and_the_table_a_shape_really_reads():
    prompt = role_prompt(WORKLOAD)

    assert "a large GEMM still in bf16, such as lm_head (the vocabulary projection), is a candidate" in prompt
    assert "a source change on top of an FP8 GEMM kernel" in prompt
    assert "the server.log's tuned-config lookups name the table each shape reads" in prompt
    assert "a table for a path the model no longer reaches changes nothing" in prompt


def test_the_prs_direction_scans_aiter_and_the_framework_and_checks_the_installed_version():
    (direction,) = [text for text in DIRECTIONS if text.startswith("Existing PRs and patches:")]

    assert "aiter" in direction
    assert "this session's framework (sglang, vllm or atom)" in direction
    assert "merged or open" in direction
    assert "check each against the version installed in the image" in direction


def test_the_role_defines_the_layers_an_intent_names():
    prompt = role_prompt(WORKLOAD)

    assert "config for server arguments, environment variables and other settings" in prompt
    assert "kernel when the change adds or replaces a GPU kernel (a Triton, HIP or CK kernel, a fused kernel)" in prompt
    assert "or a kernel's tuned table, and source for any other code change" in prompt


def test_the_role_sends_a_tuned_table_through_put_kernel_artifact_whole_and_one_per_variable():
    prompt = role_prompt(WORKLOAD)

    assert "A tuned table goes into the recipe through put_kernel_artifact" in prompt
    assert "put the whole table for its variable, one table per variable" in prompt
    assert "add the artifact it returns to the delta's add_kernels" in prompt
    assert "never point a variable at a file of yours or set it from code" in prompt
    assert "replaces the image's default table for that variable" in prompt
    assert "carry the default's rows for the other shapes as well as the tuned ones" in prompt
    assert any("put_kernel_artifact" in direction for direction in DIRECTIONS)


def test_the_role_says_a_patch_delta_never_relabels_a_component_the_head_has():
    prompt = role_prompt(WORKLOAD)

    assert "Submit the delta worktree_patch returns as it is" in prompt
    assert "never set upstream_ref or upstream_repo on a component your head already has" in prompt


def test_the_role_says_a_change_never_selects_its_file_or_setting_with_setdefault():
    prompt = role_prompt(WORKLOAD)

    assert "must not use os.environ.setdefault" in prompt
    assert "a variable the recipe already sets silently wins" in prompt


def test_the_role_lets_small_changes_too_small_alone_go_together_as_one_intent():
    prompt = role_prompt(WORKLOAD)

    assert "Keep a candidate to one intent" not in prompt
    assert "may go together as one intent" in prompt
    assert "the summed share of the operators they speed up" in prompt


def test_the_role_says_the_conversation_lives_on_and_what_arrives_wakes_it():
    prompt = role_prompt(WORKLOAD)

    assert "This conversation lasts as long as your branch runs" in prompt
    assert "what arrives for your branch (results, judgements, briefs, the runtime's notices) wakes you" in prompt
    assert "when this conversation ends a new one starts" not in prompt


def test_the_role_says_k_is_a_ceiling_and_a_test_needs_analysis_behind_it():
    prompt = role_prompt(WORKLOAD)

    assert "K is a ceiling, not a target" in prompt
    assert "never a test just to fill a slot" in prompt


def test_the_role_says_a_profile_refused_as_existing_is_waited_on():
    assert "refused as profile_exists names the one already queued, running or measured" in role_prompt(WORKLOAD)
    assert "wait_test on it rather than profiling again" in role_prompt(WORKLOAD)


def test_the_role_keeps_working_while_tests_run_and_ends_the_turn_to_wait():
    prompt = role_prompt(WORKLOAD)

    assert "While your tests run, keep working" in prompt
    assert "when nothing is left to do until one lands, end your turn and it wakes you" in prompt
    assert "within a minute with the test's progress" in prompt
    assert "Never wait with sleep, a background shell or by reading the test's files" in prompt


def test_the_role_says_the_inbox_block_a_tool_reply_ends_with_is_not_the_tool_s_result():
    prompt = role_prompt(WORKLOAD)

    assert f'A tool reply can end with a block headed "{REPLY_INBOX_HEADER}"' in prompt
    assert "it is not part of that tool's result" in prompt


def test_the_wake_and_nudge_messages_say_what_arrived_or_that_nothing_did():
    item = InboxItem(InboxKind.JUDGEMENT, "s1", "r7", "r7 measured +2.1%: committed", 0.0)

    woken = prompts.wake_message([item], [])
    assert "this arrived for your branch" in woken and "- judgement r7: r7 measured +2.1%: committed" in woken
    nudge = prompts.nudge_message()
    assert "nothing new for your branch" in nudge
    assert "Nothing arrives on its own" in nudge and "notebook_brief" in nudge
    assert "decides when your branch stops" not in nudge


def test_the_role_says_a_test_is_how_a_specialist_without_a_gpu_measures_a_candidate():
    prompt = role_prompt(WORKLOAD)

    assert "You have no GPU of your own: submit_test is how you measure" in prompt
    assert "a flat or failed result tells the next candidate something" in prompt


def test_the_role_says_a_brief_item_asks_for_the_brief_again_before_the_next_claim():
    prompt = role_prompt(WORKLOAD)

    assert 'A "brief" item' in prompt
    assert prompt.index("call notebook_brief with the head it names") < prompt.index("before your next claim")


def test_the_role_tests_a_replayable_champion_on_the_head_before_other_directions():
    prompt = role_prompt(WORKLOAD)

    assert "get its delta with notebook_replay and test it on your head before other directions" in prompt


def test_the_role_says_settings_running_out_never_ends_a_direction_and_no_brief_comes_by_waiting():
    prompt = role_prompt(WORKLOAD)

    assert "That every setting near your direction was tried never means your direction is done" in prompt
    assert "a patch, a fused or tuned kernel, or a port of an aiter or framework PR" in prompt
    assert "Never end a turn to wait for a brief" in prompt
    assert "a branch that keeps finding nothing to test is stopped" in prompt


def test_the_nudge_sends_a_model_that_ran_out_of_settings_to_source_work():
    assert "Settings run out; source work does not" in prompts.nudge_message()


def test_the_role_says_a_bound_table_variable_takes_any_branch_s_table_under_its_target():
    prompt = role_prompt(WORKLOAD)

    assert "binds it to a target name for the session, not to a branch" in prompt
    assert "put your own table under that target" in prompt


def test_the_role_gives_the_offline_route_for_a_tunableop_table():
    prompt = role_prompt(WORKLOAD)

    assert 'env "PYTORCH_TUNABLEOP_FILENAME"' in prompt
    assert "PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=0" in prompt
    assert "online_tuning_unsupported" in prompt


def test_the_role_says_a_learning_names_its_direction_and_how_a_past_lesson_is_revised_or_retired():
    prompt = role_prompt(WORKLOAD)

    assert "A learning names its direction and cites its tests in request_ids" in prompt
    assert "revise it with supersedes" in prompt and "retire it with retires" in prompt


def test_the_role_says_how_to_read_a_hip_runtime_error_and_when_a_setting_counts_as_unusable():
    prompt = role_prompt(WORKLOAD)

    assert "hip_runtime_error" in prompt
    assert "an illegal memory access means a kernel call in your change goes out of bounds" in prompt
    assert "operation not permitted when stream is capturing" in prompt
    assert "only after a second, independent failure" in prompt


def test_the_role_points_at_the_brief_s_unfinished_leads_by_their_section_title():
    prompt = role_prompt(WORKLOAD)

    title = UNFINISHED_TITLE.split(" (")[0]
    assert f'A lead under "{title}" in the brief' in prompt


def test_the_role_says_how_to_declare_a_correctness_fix_and_what_becomes_of_it():
    prompt = role_prompt(WORKLOAD)

    assert "reachable under the session's" in prompt
    assert "$HL_BASE_URL serving $HL_MODEL" in prompt
    assert "exit 1 when the defect occurs and 0 when it does not" in prompt and "never a fix" in prompt
    assert prompt.index("Call fix_case") < prompt.index("passing the attachment fix_case returned in attachments")
    for outcome in ("committed with its case", "forked to a new branch cut at its parent", "recorded only"):
        assert outcome in prompt


def test_the_role_lists_every_known_fake_gain_with_its_refusal():
    prompt = role_prompt(WORKLOAD)

    assert all(f"{row.shows_as} Refused as {row.refusal_code}." in prompt for row in FAKE_GAINS)


def test_a_launch_opens_with_the_brief_then_says_where_the_branch_stands():
    prompt = _launch(brief="leads: fp8 KV cache measured +4% on this model")

    assert prompt.startswith("Startup brief:\nleads: fp8 KV cache")
    facts = "Your branch: s1, started at c0. Its head: c7. Tests you may have running at once: 3."
    assert prompt.index(facts) > prompt.index("leads")
    assert "first session on your branch" in prompt


def test_every_launch_of_a_repair_branch_opens_with_the_candidate_s_change_and_its_diagnosis():
    blocked = Judgement(Verdict.BLOCKED, 3.25, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
    kv = Intent("s0", "c3", "kv cache", "fp8 kv cache", IntentLayer.CONFIG)
    change = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))
    repair = RepairRequest(kv, change, blocked, "per-tensor KV scales; scale per head\n")
    start = SessionStart("s1", "c3", 3, BranchTask("leads: fp8 KV cache measured +4% on this model", repair))

    prompt = launch_prompt(start, head="c3", worktree="/nfs/s1/worktree", recent=(_record(1),))

    assert prompt.startswith(
        "Your branch repairs s0-4, a blocked candidate of s0: on c3 it measured +3.25% and was held back by the "
        "accuracy gate. Your branch starts at that commit. Its intent: kv cache / fp8 kv cache (config)."
    )
    assert json.dumps(change.to_document(), sort_keys=True) in prompt
    assert "the root cause and the repair:\nper-tensor KV scales; scale per head\nStart from that change" in prompt
    assert prompt.index("fp8 kv cache (config)") < prompt.index("Startup brief:") < prompt.index("Your branch: s1,")
    assert "Your branch repairs" not in _launch(brief="leads: none")


def test_every_launch_shows_the_branch_s_direction_before_the_brief():
    prompt = _launch(brief="leads: fp8 KV cache", direction="  try a larger max_num_seqs first\n")

    assert prompt.startswith(
        "Your branch's direction, to work on first:\n"
        "try a larger max_num_seqs first\n\nStartup brief:\nleads: fp8 KV cache"
    )
    assert "Your branch's direction" not in _launch(brief="leads: none", direction=" \n")


def test_the_session_s_directions_are_distinct_and_lead_with_the_four_a_small_session_takes():
    assert len(set(DIRECTIONS)) == len(DIRECTIONS) == 7
    assert [text.split(":")[0] for text in DIRECTIONS[:4]] == [
        "server.log fixes",
        "GEMM tuning",
        "Fusion",
        "Server arguments and environment variables",
    ]


def test_each_prompt_is_versioned_by_its_own_text(monkeypatch):
    versions = prompt_versions()

    assert set(versions) == {"specialist_role", "specialist_launch"}
    assert all(len(version) == VERSION_HEX and int(version, 16) >= 0 for version in versions.values())
    monkeypatch.setattr(prompts, "_ROLE", prompts._ROLE + "- One more rule.\n")
    edited = prompt_versions()
    assert edited["specialist_role"] != versions["specialist_role"]
    assert edited["specialist_launch"] == versions["specialist_launch"]


def test_a_launch_shows_the_latest_entries_cut_to_size_and_how_the_last_session_ended():
    entries = [_record(n) for n in range(RECENT_ENTRIES + 4)] + [_record(99, "x" * 1000, ("s1-17",))]

    prompt = _launch(recent=entries, last_end="the agent CLI exited 1")

    assert prompt.startswith("Your branch: s1,")
    assert "[e4] result: entry 4" not in prompt and "[e5] result: entry 5" in prompt
    assert f"[e99] result (s1-17): {'x' * (ENTRY_CHARS - 3)}...\n" in prompt
    assert "Your previous session ended: the agent CLI exited 1." in prompt
