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
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord, InboxItem
from hyperloom_multibranch.contracts.blackboard.vocab import InboxKind
from hyperloom_multibranch.specialist import prompts
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.exhausted import EXHAUSTED_PREFIX, ExhaustedWhy
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.specialist.prompts import (
    ENTRY_CHARS,
    RECENT_ENTRIES,
    SUMMARY_ENTRIES,
    SUMMARY_ENTRY_CHARS,
    VERSION_HEX,
    launch_prompt,
    prompt_versions,
    role_prompt,
)
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.contracts.blackboard.records import WarmReplay
from hyperloom_multibranch.blackboard.knowledge.warm_start import PATCH_REFERENCE, ReplayReference
from hyperloom_multibranch.blackboard.rsi.briefing import ON_HEAD
from hyperloom_multibranch.blackboard.tools import (
    CODE_TITLE,
    FOCUS_LINE,
    SETTINGS_TITLE,
    UNFINISHED_TITLE,
    render_replay,
)
from hyperloom_multibranch.toolbox.mcp_tools import REPLY_INBOX_HEADER

WORKLOAD = "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64, InferenceX"
TOOLS = (
    "blackboard_branch_log",
    "blackboard_claim",
    "start_analysis",
    "worktree_sync",
    "worktree_check",
    "worktree_patch",
    "submit_test",
    "wait_test",
    "poll_inbox",
    "test_result",
    "blackboard_append_branch_log",
    "blackboard_learn",
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
    assert prompt.index("blackboard_learn, citing the candidate") < prompt.index("then call decline_repair")
    with pytest.raises(ValueError, match="names the workload"):
        role_prompt(" \n")


def test_the_role_reads_the_briefs_two_direction_lists_before_claiming_and_chooses_as_before_without_them():
    prompt = " ".join(role_prompt(WORKLOAD).split())
    settings, code = (title.split(" (")[0] for title in (SETTINGS_TITLE, CODE_TITLE))

    assert f'When the brief lists "{settings}" or "{code}", read them before you claim' in prompt
    assert "A brief without those lists: choose as you would without them." in prompt


def test_the_role_profiles_through_a_profile_test_and_tracelens():
    prompt = role_prompt(WORKLOAD)

    assert 'submit_test with kind "profile", purpose "profile"' in prompt
    assert 'start_analysis with tool "tracelens" and that test\'s id as profile_request_id' in prompt


def test_the_role_uses_roofline_and_upstream_pr_search_only_when_start_analysis_lists_them():
    prompt = role_prompt(WORKLOAD)

    assert 'When start_analysis lists roofline, the same profile test with tool "roofline"' in prompt
    assert 'When start_analysis lists upstream_pr_search, search with tool "upstream_pr_search"' in prompt


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
    assert "Server arguments and environment variables; MoE chain; Small-M GEMM kernels; ROCm code gaps" in prompt
    assert "Existing PRs and patches; Communication." in prompt


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


def test_the_role_sends_tuned_rows_through_put_kernel_artifact_one_table_per_variable():
    prompt = role_prompt(WORKLOAD)

    assert "A tuned table goes into the recipe through put_kernel_artifact, one table per variable" in prompt
    assert "the artifact it returns goes into the delta's add_kernels" in prompt
    assert "never point a variable at a file of yours or set it from code" in prompt
    assert "put only the rows you tuned with rows_only=true" in prompt
    assert "a row overrides the image's default row for the same shape" in prompt
    assert "If your head's recipe already has a table for that variable, yours replaces it" in prompt
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


def test_the_role_keeps_working_after_submit_test_and_ends_the_turn_only_with_k_tests_running():
    prompt = role_prompt(WORKLOAD)

    assert "After submit_test, keep working; do not end your turn" in prompt
    assert "While fewer than K of your tests run, prepare and submit the next candidate" in prompt
    assert "End your turn only when nothing is left to read or prepare and every candidate the evidence" in prompt
    assert "within a minute with the test's progress" in prompt
    assert "Never wait with sleep, a background shell or by reading the test's files" in prompt


def test_the_role_says_what_a_pending_build_its_build_item_and_its_refusals_mean():
    prompt = role_prompt(WORKLOAD)

    assert "A test in state pending_build waits for a native build its change needs" in prompt
    assert "it is not stuck and holds one of your K" in prompt
    assert 'A "build" item says such a build is ready and the test queues for cards, or that it failed on ' in prompt
    assert "A test whose build your change broke ends candidate_failed with a compile_error" in prompt
    assert "build_after_deadline would end past your branch's deadline" in prompt


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
    assert "Nothing arrives on its own" in nudge and "blackboard_brief" in nudge
    assert "decides when your branch stops" not in nudge


def test_a_wake_after_the_integration_head_moved_carries_what_it_holds():
    item = InboxItem(InboxKind.RESULT, "s1", "r7", "r7 has its result", 0.0)

    moved = prompts.wake_message([], [], integration="The integration head is i2: server arguments --x 1.")
    with_item = prompts.wake_message([item], [], integration="The integration head is i2.")

    assert moved.startswith("Since your last turn ended, the integration head moved.")
    assert "Where the session's gains stack:\nThe integration head is i2: server arguments --x 1." in moved
    assert with_item.index("- result r7") < with_item.index("The integration head is i2.")
    assert "gains stack" not in prompts.wake_message([item], [])


def test_the_role_follows_the_integration_head_with_a_reroot_entry_and_never_replays_what_it_holds():
    prompt = role_prompt(WORKLOAD)

    assert "A replayable champion in the brief that the integration head does not hold" in prompt
    assert "Never test or replay a change it holds" in prompt
    assert 'write a branch-log entry that starts with "REROOT"' in prompt
    assert "submit nothing more on your old head" in prompt
    assert "with no candidate test of yours in flight" in prompt
    assert "one whose merge conflicted is yours to port" in prompt
    assert "Keep your head instead while your next candidate builds on a commit of yours not merged yet" in prompt


def test_the_role_says_a_cross_image_lead_is_measured_here_and_its_left_out_files_are_ported_not_replayed():
    patch = ReplayReference(PATCH_REFERENCE, "fused.patch", "a" * 64, "sha256:old", "sglang", "b" * 40)
    replay = WarmReplay("kb:c1", "c7", RecipeDelta(), IntentLayer.CONFIG, "", "sha256:old", (patch.describe(),))
    shown = render_replay(replay)
    prompt = role_prompt(WORKLOAD)

    assert "its gain was measured on image sha256:old" in shown and "left out: patch fused.patch" in shown
    assert "whose gain the brief or blackboard_replay says was measured on another image" in prompt
    assert 'Each "left out: patch" entry' in prompt
    assert "read its text with blackboard_replay and reference set to that path" in prompt
    assert "then check your synced worktree" in prompt and "port its intent to your head" in prompt
    assert "never replay it as is" in prompt


def test_a_resumed_cli_hears_why_it_stopped_its_worktree_and_what_arrived_meanwhile():
    item = InboxItem(InboxKind.RESULT, "s1", "r7", "r7 has its result", 0.0)
    notice = _record(9, "the specialist crashed")

    kept = prompts.resume_message(prompts.Resume("the agent CLI exited 1", "c3", "/wt/s1", kept=True), [item], [notice])
    reset = prompts.resume_message(prompts.Resume("the session was lost", "c4", "/wt/s1", kept=False), [], [])

    assert "the agent CLI exited 1" in kept and "same conversation" in kept
    assert "/wt/s1 is as you left it" in kept and "- result r7: r7 has its result" in kept and "[e9]" in kept
    assert "reset to your branch head c4" in reset and "uncommitted edits were dropped" in reset
    assert "Nothing arrived for your branch" in reset


def test_the_fast_path_direction_proves_a_path_ran_and_leaves_untuned_gemms_to_gemm_tuning():
    (direction,) = [text for text in DIRECTIONS if text.startswith("Fast path checks:")]

    assert "a one-time log line" in direction and "eager or plain torch path" in direction
    assert "untuned GEMM shapes belong to the GEMM tuning branch" in direction


def test_the_role_says_a_flat_patch_is_checked_to_have_run_before_its_idea_is_dropped():
    prompt = role_prompt(WORKLOAD)

    assert "put a one-time log line on the path it adds" in prompt
    assert "check the test's server.log shows that line before you call the idea dead" in prompt


def test_the_role_says_worktree_patch_names_a_new_component_s_upstream_and_when_to_set_it_by_hand():
    prompt = role_prompt(WORKLOAD)

    assert "for a component new on your head it names the upstream commit already" in prompt
    assert "set them only when its reply carries a note saying so" in prompt


def test_the_gemm_direction_and_the_role_put_a_path_switch_before_hand_picked_rows():
    (direction,) = [text for text in DIRECTIONS if text.startswith("GEMM tuning:")]
    prompt = role_prompt(WORKLOAD)

    assert "first switch it to a kernel family with tuned rows or a stronger default" in direction
    assert "picking a tile or a split-K for it without timing it is a guess" in prompt
    assert "a row at a padded M covers every batch size that pads to it" in prompt


def test_the_fast_path_direction_reads_the_branch_conditions_the_log_does_not_report():
    (direction,) = [text for text in DIRECTIONS if text.startswith("Fast path checks:")]
    prompt = role_prompt(WORKLOAD)

    assert "read the branch conditions on its call path" in direction
    assert "Work sized to a buffer rather than the live batch" in prompt
    assert "a shape someone validated, not a hardware bound" in prompt


def test_the_existing_prs_direction_says_how_to_search_check_and_record_a_pr():
    (direction,) = [text for text in DIRECTIONS if text.startswith("Existing PRs and patches:")]

    assert "WebSearch" in direction and "WebFetch" in direction
    assert "record each PR's verdict with blackboard_learn" in direction


def test_a_branch_with_no_candidate_buys_the_missing_number_before_it_stops():
    prompt = role_prompt(WORKLOAD)

    assert "no candidate clears the bar is a missing measurement, not a stop" in prompt
    assert "the cheapest test that would prove it wrong" in prompt
    assert "buy the number first" in prompts.nudge_message()


def test_the_role_reads_a_candidate_s_gain_from_its_judgement_not_from_quoted_figures():
    prompt = role_prompt(WORKLOAD)

    assert "Read a candidate's gain from its judgement" in prompt
    assert "never from figures other branches quote" in prompt


def test_the_role_says_a_branch_opened_on_a_focus_starts_from_that_operator():
    prompt = role_prompt(WORKLOAD)

    label = FOCUS_LINE.split(",")[0]
    assert f'When the brief names a "{label}", start from that operator within your direction' in prompt


def test_the_role_says_how_to_record_a_direction_done_on_a_head_with_every_reason():
    prompt = role_prompt(WORKLOAD)

    assert f'starts with "{EXHAUSTED_PREFIX}"' in prompt
    assert all(why.value in prompt for why in ExhaustedWhy)
    assert "does not stop your branch" in prompt
    assert "at least one: a direction you have not tested is not done" in prompt
    assert "direction (the name of the session's direction exactly as listed below" in prompt


def test_the_role_says_a_test_is_how_a_specialist_without_a_gpu_measures_a_candidate():
    prompt = role_prompt(WORKLOAD)

    assert "You have no GPU of your own: submit_test is how you measure" in prompt
    assert "a flat or failed result tells the next candidate something" in prompt


def test_the_role_says_a_brief_item_asks_for_the_brief_again_before_the_next_claim():
    prompt = role_prompt(WORKLOAD)

    assert 'A "brief" item' in prompt
    assert prompt.index("call blackboard_brief with the head it names") < prompt.index("before your next claim")


def test_the_role_tests_a_replayable_champion_on_the_head_before_other_directions():
    prompt = role_prompt(WORKLOAD)

    assert "get its delta with blackboard_replay and test it on your head before other directions" in prompt
    assert f'A champion the brief lists as "{ON_HEAD}" is in your head already: build on it' in prompt


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
    assert "put your own table under that target, or leave target out to use it" in prompt


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
    blocked = Judgement(
        Verdict.BLOCKED, 3.25, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, AccuracyVerdict.FAILED
    )
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
    assert len(set(DIRECTIONS)) == len(DIRECTIONS) == 11
    assert [text.split(":")[0] for text in DIRECTIONS[:4]] == [
        "Fast path checks",
        "GEMM tuning",
        "Fusion",
        "Server arguments and environment variables",
    ]


def test_the_kernel_chain_directions_go_past_tuned_tables_and_settings():
    names = [text.split(":")[0] for text in DIRECTIONS]
    moe, small_m, recurrent, communication = (
        DIRECTIONS[names.index(name)]
        for name in ("MoE chain", "Small-M GEMM kernels", "Recurrent decode chain", "Communication")
    )

    assert "router GEMV, the softmax and top-k, the sort and the expert GEMMs as one chain" in moe
    assert "fold the sort into the router" in moe
    assert "Swap the kernel family" in small_m
    assert "merge sibling projections" in small_m
    assert "fold a GEMV into the kernel that consumes its output" in small_m
    assert "conv1d, the recurrent state update and the gated norm" in recurrent
    assert "Routes already tried" in communication
    assert "--enable-aiter-allreduce-fusion" in communication
    assert "SGLANG_USE_1STAGE_ALLREDUCE" in communication


def test_the_existing_prs_direction_vendors_a_pr_that_needs_a_newer_aiter_or_triton():
    (direction,) = [text for text in DIRECTIONS if text.startswith("Existing PRs and patches:")]

    assert "needs a newer aiter or Triton than the image is vendored, not ruled out" in direction
    assert "copy its kernel into the patch" in direction


def test_the_role_takes_a_bound_from_a_tool_s_number_not_a_hand_estimate():
    prompt = role_prompt(WORKLOAD)

    assert "cites the bandwidth or roofline figure a tool reported for that operator" in prompt
    assert "a hand estimate of its bytes is a hypothesis to measure, not a bound" in prompt


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

    assert prompt.split("\n\n")[1].startswith("Your branch: s1,")
    assert "[e4] result: entry 4" not in prompt and "[e5] result: entry 5" in prompt
    assert f"[e99] result (s1-17): {'x' * (ENTRY_CHARS - 3)}...\n" in prompt
    assert "Your previous session ended: the agent CLI exited 1." in prompt


def test_the_role_checks_bandwidth_copies_and_a_flat_fusion_before_spending_tests():
    prompt = role_prompt(WORKLOAD)

    assert "each launch costs about 1.5 us on MI355X" in prompt
    assert "2 to 4 us" not in prompt
    assert "divide the bytes it must move per call" in prompt
    assert "near the device's HBM bandwidth no kernel change pays" in prompt
    assert "Count the copy, cast and elementwise kernels in a decode step" in prompt
    assert "compare each kernel's count and time with the parent's profile" in prompt
    assert "A speculative or MTP draft gains its accept length" in prompt


def test_the_directions_name_shape_whitelists_unpassed_switches_and_settings_nothing_reads():
    fast_path, settings = DIRECTIONS[0], DIRECTIONS[3]

    assert "a whitelist or table keyed by the model's shape" in fast_path
    assert "which model files pass its switch" in fast_path
    assert "Before testing a setting, grep where it is read" in settings
    assert "Bound a scheduler setting" in settings


def test_the_gemm_direction_asks_for_the_table_read_and_marks_borrowed_rows_as_untimed():
    gemm = DIRECTIONS[1]

    assert gemm.startswith("GEMM tuning:")
    assert "tune the aiter and hipBLASLt GEMM configurations" not in gemm
    assert "check which table each shape actually reads" in gemm
    assert "switch it to a kernel family with tuned rows or a stronger default" in gemm
    assert "rows borrowed from tuned tables of the same kernel family at nearby shapes are an untimed guess" in gemm
    assert "rows_only=true" in gemm
    assert "several shapes in one test" in gemm


def test_the_role_reads_draft_acceptance_by_context_host_work_host_syncs_and_capture_sizes():
    prompt = role_prompt(WORKLOAD)

    assert "Read the accept length per context-length bucket, not only its mean" in prompt
    assert "Compare TTFT with the prefill's GPU time in the same test" in prompt
    assert "A GPU profile does not show host stalls: grep the per-layer hot path for .item()" in prompt
    assert "Set graph capture sizes from the token counts the workload's steps carry" in prompt


def test_the_directions_name_prefix_recompute_config_overrides_and_fusion_through_the_quantize():
    fast_path, fusion, settings = DIRECTIONS[0], DIRECTIONS[2], DIRECTIONS[3]

    assert "compare each follow-up turn's computed prefill tokens with its new tokens" in settings
    assert "is set with --hf-overrides, nested under text_config for a multimodal wrapper" in fast_path
    assert "Carry the fusion to the quantize" in fusion


def test_the_role_names_the_session_s_reference_material_only_when_it_has_some(tmp_path):
    geak = tmp_path / "MB-reference" / "geak"

    prompt = role_prompt(WORKLOAD, references={"geak": geak})

    assert f"geak at {geak}" in prompt and "read-only" in prompt
    assert "name the file you used in your branch log" in prompt
    assert "Reference material for this session" not in role_prompt(WORKLOAD)


def test_the_role_says_what_geak_s_reference_directories_hold(tmp_path):
    prompt = role_prompt(WORKLOAD, references={"geak": tmp_path / "geak"})

    assert "perf_knowledge/expert_skills/tuning" in prompt
    assert "kernel_workflow/knowledge/learned" in prompt
    assert "perf_knowledge" not in role_prompt(WORKLOAD, references={"notes": tmp_path / "notes"})


def test_the_role_says_what_an_inferencex_copy_holds(tmp_path):
    prompt = role_prompt(WORKLOAD, references={"inferencex": tmp_path / "inferencex"})

    assert "configs/*-master.yaml" in prompt
    assert "benchmarks/single_node" in prompt


def test_the_role_reads_a_table_s_hits_before_calling_it_flat():
    prompt = role_prompt(WORKLOAD)

    assert "table_hits says, per kernel table the test loaded, how many of its rows aiter read" in prompt
    assert "a table marked not_exercised measured nothing of its rows" in prompt


def test_the_attention_and_fusion_directions_name_the_cache_layout_and_the_epilogue():
    (attention,) = [text for text in DIRECTIONS if text.startswith("Attention and decode kernels:")]
    (fusion,) = [text for text in DIRECTIONS if text.startswith("Fusion:")]

    assert "every kernel that writes the KV cache must write the layout the new one reads" in attention
    assert "revise the patch your head already has for that writer instead of adding a second one" in attention
    assert "into the GEMM epilogue that consumes it" in fusion
    assert "profile again once a large kernel on the path got faster" in fusion


def test_the_role_and_gemm_direction_tune_with_kernel_tune_when_it_is_served():
    prompt = role_prompt(WORKLOAD)
    gemm = DIRECTIONS[1]

    assert 'start_analysis with tool "kernel_tune"' in prompt
    assert "rows marked verified=false are never put" in prompt
    assert "its microseconds are advice, the test is the measurement" in prompt
    assert "test_request_id (required; optionally with shapes of your own)" in prompt
    assert "When start_analysis lists kernel_tune, tune the untuned shapes with it" in gemm


def test_the_role_profiles_a_head_right_after_its_branch_was_split_off():
    assert "or when your branch was just split off another branch's head" in role_prompt(WORKLOAD)


def test_the_role_says_an_aiter_or_framework_kernel_is_source_a_branch_can_fork_and_call_differently():
    prompt = role_prompt(WORKLOAD)

    assert "copy it into the framework tree and edit it there" in prompt
    assert "calling one differently" in prompt and "accumulating a split-K GEMM into the residual" in prompt
    assert '"Needs a C++ change" is never a reason to give up' in prompt
    assert "A kernel inside aiter or the framework is not blocked_by_tool" in prompt
    assert "a streaming read or a device-to-device copy reaches about 6.2 TB/s on MI355X" in prompt


def test_the_role_says_sglang_jit_kernels_are_framework_source_the_test_rebuilds():
    prompt = role_prompt(WORKLOAD)

    assert "python/sglang/kernels/jit/csrc" in prompt
    assert "the server's first load recompiles a kernel whose own file you edited" in prompt
    assert "a change to a shared header alone is not seen" in prompt


def test_the_dev_sandbox_times_a_kernel_with_what_its_launcher_adds():
    prompt = role_prompt(WORKLOAD, dev_leases=1)

    assert "with what its launcher adds around it (a memset, a quantize) and cache-cold weights" in prompt


def test_the_gemm_attention_and_fusion_directions_carry_the_strongest_kernel_examples():
    gemm = DIRECTIONS[1]
    (fusion,) = [text for text in DIRECTIONS if text.startswith("Fusion:")]
    (attention,) = [text for text in DIRECTIONS if text.startswith("Attention and decode kernels:")]

    assert "its split-K picks never pay the hipMemsetAsync" in gemm
    assert "launched the same asm .co directly through ctypes" in gemm
    assert "fixed by splitting the call along M, not by dropping the table" in gemm
    assert "fused_qk_norm_rope_cache_pts_quant_shuffle" in fusion and "fused_rms_fp8_group_quant" in fusion
    assert "copied into the framework tree and edited" in fusion
    assert "page size included" in attention and "pa_decode_gluon" in attention
    assert "scales from its epilogue" in attention


def test_the_role_retests_a_lesson_or_a_peer_s_flat_result_on_the_current_head_before_closing_on_it():
    prompt = role_prompt(WORKLOAD)

    assert "A lesson, a learning or another branch's flat result closes nothing on a different head" in prompt
    assert "retest it on your current head before you call it closed" in prompt


def test_the_role_says_a_blocked_or_missing_capability_exhausted_names_the_path_tried():
    prompt = role_prompt(WORKLOAD)

    assert "for blocked_by_tool or needs_capability, the path you tried: the file, the call and the error" in prompt
    assert "the error it gave, or the record is refused" in prompt


def test_the_role_checks_the_head_for_a_lead_s_change_before_replaying_it():
    prompt = role_prompt(WORKLOAD)

    assert "Before you replay a lead, check whether your head already carries its change" in prompt


def test_the_gemm_direction_checks_a_peer_s_report_that_a_kernel_family_is_missing():
    assert "A peer's report that a kernel family is missing from the image closes nothing" in DIRECTIONS[1]


def _logged(entry_id: str, kind: BranchLogType, content: str, supersedes: str | None = None) -> BranchLogRecord:
    return BranchLogRecord(entry_id, 0.0, BranchLogEntry("s1", kind, content, (f"r-{entry_id}",), supersedes))


def test_a_launch_on_a_log_longer_than_it_shows_opens_with_a_summary_of_the_whole_log():
    tried = [_logged(f"a{n}", BranchLogType.ATTEMPT, f"tried change {n}") for n in range(SUMMARY_ENTRIES + 5)]
    replaced = _logged("a99", BranchLogType.ATTEMPT, "tried change 14 again, measured right", supersedes="a14")
    results = [_logged(f"o{n}", BranchLogType.RESULT, f"change {n} flat") for n in range(SUMMARY_ENTRIES)]
    gate = _logged("g1", BranchLogType.RESULT, f"{NOTICE} r-a3 measured +2.1%: kept")
    restart = _logged(
        "n1", BranchLogType.RESULT, f"{NOTICE} the specialist crashed (RuntimeError('x')) and is restarted"
    )
    long = _logged("o99", BranchLogType.RESULT, "y" * 1000)
    plans = [_logged("p1", BranchLogType.PLAN, "profile the head"), _logged("p2", BranchLogType.PLAN, "fuse rmsnorm")]
    log = [*tried, *plans, *results, gate, restart, replaced, long]

    prompt = _launch(direction="fusions", recent=log)

    summary, direction = prompt.split("\n\n")[:2]
    assert summary.startswith(f"Your branch so far, from its whole log of {len(log)} entries:")
    assert direction.startswith("Your branch's direction")
    assert "Next, by your latest plan:\n- [p2] plan (r-p2): fuse rmsnorm" in summary and "[p1]" not in summary
    assert f"Tried, the latest {SUMMARY_ENTRIES} of {SUMMARY_ENTRIES + 5}:" in summary
    assert "[a99] attempt" in summary and "[a14]" not in summary and "[a4]" not in summary and "[a5]" in summary
    assert f"Results, the latest {SUMMARY_ENTRIES} of {SUMMARY_ENTRIES + 2}:" in summary
    assert "[g1] result" in summary and "[n1]" not in summary and "[o1]" not in summary
    assert f"[o99] result (r-o99): {'y' * (SUMMARY_ENTRY_CHARS - 3)}...\n" in summary


def test_the_summary_of_a_log_stays_bounded_however_long_its_entries_and_a_short_log_has_none():
    kinds = (BranchLogType.ATTEMPT, BranchLogType.RESULT, BranchLogType.PLAN, BranchLogType.HYPOTHESIS)
    log = [_logged(f"e{n}", kinds[n % 4], "z" * 5000) for n in range(400)]

    summary = _launch(recent=log).split("\n\n")[0]
    short = _launch(recent=[_record(n) for n in range(RECENT_ENTRIES)])

    assert summary.startswith("Your branch so far")
    bound = 2 * SUMMARY_ENTRIES * (SUMMARY_ENTRY_CHARS + 40) + ENTRY_CHARS + 200
    assert len(summary) <= bound and "[e398] hypothesis" not in summary
    assert "Your branch so far" not in short


def test_the_nudge_and_role_send_later_heads_on_a_path_whose_profile_analysis_failed_to_other_evidence():
    # Each head is profiled once already (toolbox profile_exists); the waste was profiling the next head on a path
    # whose analysis fails, so the rule has to cover later heads, not the failed one.
    failed = "Once a profile analysis failed, profile no later head on the same execution path"
    nudge = prompts.nudge_message()
    role = role_prompt(WORKLOAD)

    assert failed in nudge and failed in role
    assert "log its dispatch path or break down a flat result instead" in nudge
    assert "profile your head unless a profile analysis already failed on its execution path" in role


def test_the_role_retries_a_lead_the_integrator_refused_only_with_a_reason_it_would_now_stack():
    prompt = role_prompt(WORKLOAD)

    assert "A lead the integrator measured not stackable on the integration head" in prompt
    assert "only with a stated reason it would now stack" in prompt


def test_the_role_names_native_sources_only_when_the_session_lays_them_out(tmp_path):
    tree = NativeTree(
        SourceRoot("python/sglang/kernels/aot", tmp_path), "https://github.com/sgl-project/sglang.git", "2" * 40
    )

    native = role_prompt(WORKLOAD, native={"sgl_kernel": tree})

    assert "sgl_kernel" not in role_prompt(WORKLOAD).split("sglang/kernels/jit/csrc")[1]
    assert "python/sglang/kernels/aot" in native and "pending_build" in native and "CMakeLists.txt" in native
    assert native.startswith(role_prompt(WORKLOAD))


def test_forge_fuse_is_leased_the_session_s_tp_since_it_serves_the_model_to_check_its_fusion():
    prompt = " ".join(role_prompt(WORKLOAD, dev_leases=1).split())

    assert "forge-fuse checks its fusion by serving the model at the session's TP: lease it that many cards" in prompt
