# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS, NATIVE_SERVING, kernel_knobs
from hyperloom.common.tests.handed_names import (
    knob_words_named,
    pull_requests_named,
    server_options_named,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.runner.native.server_options import OptionInventory, ServerOption, write_inventory
from hyperloom.orchestrator.state.base_launch import ServerBase
from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord, InboxItem
from hyperloom_multibranch.contracts.blackboard.vocab import InboxKind
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.specialist import prompts
from hyperloom_multibranch.specialist.directions import CHECKOUT_DIRECTIONS, DIRECTIONS
from hyperloom_multibranch.contracts.blackboard.exhausted import EXHAUSTED_PREFIX, ExhaustedWhy
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.specialist.prompts import (
    ENTRY_CHARS,
    RECENT_ENTRIES,
    SUMMARY_ENTRIES,
    SUMMARY_ENTRY_CHARS,
    VERSION_HEX,
    RoleWorkload,
    launch_prompt,
    prompt_versions,
    RoleServer,
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

NAMED = "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64, InferenceX"
WORKLOAD = RoleWorkload(NAMED)
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
    prompt = role_prompt(RoleWorkload(NAMED.replace(", ", ",\n  ")))

    assert f"The session optimizes one workload: {NAMED}." in prompt
    assert all(tool in prompt for tool in TOOLS)
    assert 'Never submit purpose "confirm"' in prompt
    assert "target_share_pct" in prompt and "w/(1-w)" in prompt
    assert BASH_KILL_SAFETY_PREAMBLE in prompt
    assert "A fixable defect" in prompt and "An inherent trade-off" in prompt
    assert prompt.index("blackboard_learn, citing the candidate") < prompt.index("then call decline_repair")
    with pytest.raises(ValueError, match="names the workload"):
        role_prompt(RoleWorkload(" \n"))


def test_the_role_reads_the_briefs_two_direction_lists_before_claiming_and_chooses_as_before_without_them():
    prompt = " ".join(role_prompt(WORKLOAD).split())
    settings, code = (title.split(" (")[0] for title in (SETTINGS_TITLE, CODE_TITLE))

    assert f'When the brief lists "{settings}" or "{code}", read them before you claim' in prompt
    assert "A brief without those lists: choose as you would without them." in prompt


def test_the_role_names_the_config_delta_keys_that_recipe_delta_reads():
    prompt = role_prompt(WORKLOAD)
    example = prompt.split("with these keys, each optional: ", 1)[1].split(". extra_server_args", 1)[0]

    config = RecipeDelta.from_document(json.loads(example)).config
    assert (config.server_args, dict(config.envs)) == ("--flag value --other-flag", {"NAME": "value"})
    assert (config.remove_args, config.unset_envs) == (("--old-flag",), ("OLD_NAME",))
    assert "a key the delta does not define is refused, and the error lists the keys it takes" in prompt


def test_the_role_shows_the_session_s_frozen_base_launch_read_only():
    base = ServerBase(args="--tp-size 4 --enable-hierarchical-cache --hicache-size 180", env={"SGLANG_X": "true"})
    prompt = " ".join(role_prompt(WORKLOAD, server=RoleServer(base)).split())

    assert "The session's base launch, frozen when it started and read-only" in prompt
    assert "Server arguments: --tp-size 4 --enable-hierarchical-cache --hicache-size 180." in prompt
    assert "Environment: SGLANG_X=true." in prompt
    assert "an empty extra_server_args means this launch as it is" in prompt
    assert "frozen when it started" not in " ".join(role_prompt(WORKLOAD).split())
    empty = " ".join(role_prompt(WORKLOAD, server=RoleServer(ServerBase())).split())
    assert "Server arguments: none. Environment: none." in empty


def test_the_role_says_the_inbox_block_a_tool_reply_ends_with_is_not_the_tool_s_result():
    prompt = role_prompt(WORKLOAD)

    assert f'A tool reply can end with a block headed "{REPLY_INBOX_HEADER}"' in prompt
    assert "it is not part of that tool's result" in prompt


def test_the_wake_and_nudge_messages_say_what_arrived_or_that_nothing_did():
    item = InboxItem(InboxKind.JUDGEMENT, "s1", "r7", "r7 measured +2.1%: committed", 0.0)

    woken = prompts.wake_message([item], [])
    assert "this arrived for your branch" in woken and "- judgement r7: r7 measured +2.1%: committed" in woken
    nudge = prompts.nudge_message(profiled=True)
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


def test_an_inferencex_role_names_its_bar_and_defects_and_no_agentx_objective():
    role = " ".join(role_prompt(WORKLOAD).split())

    assert "clear the session's bar, set from the spread of the baseline measurements" in role
    assert "the accuracy gate, requests it did not complete (dropped_requests)" in role
    assert "(long_context_dropped). It waits for your diagnosis" in role
    assert "accuracy whatever the scales. Write the lesson" in role
    assert not any(word in role for word in ("P50", "P90", "interactivity", "output throughput"))


@pytest.mark.parametrize("objective", list(AgentXObjective))
def test_an_agentx_role_tells_its_objective_bar_and_guards(monkeypatch, objective):
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "4")
    role = " ".join(role_prompt(RoleWorkload(NAMED, objective)).split())
    guards = objective.guards

    assert f"graded on {objective.title} ({objective.metric}): {objective.description}" in role
    assert f"bar of +{objective.bar_pct:g}% on it" in role and "+2%" in role
    assert " and ".join(f"{guard.name} ({guard.metric})" for guard in guards) + " each hold within 4%" in role
    assert all(f"{guard.name} as {guard.refusal}" in role for guard in guards)
    assert f"a {objective.title} gain bought with {guards[0].name} or {guards[1].name}" in role
    assert "output throughput" not in role and "dropped_requests" not in role
    assert "spread of the baseline measurements" not in role


def test_a_resumed_cli_hears_why_it_stopped_its_worktree_and_what_arrived_meanwhile():
    item = InboxItem(InboxKind.RESULT, "s1", "r7", "r7 has its result", 0.0)
    notice = _record(9, "the specialist crashed")

    kept = prompts.resume_message(prompts.Resume("the agent CLI exited 1", "c3", "/wt/s1", kept=True), [item], [notice])
    reset = prompts.resume_message(prompts.Resume("the session was lost", "c4", "/wt/s1", kept=False), [], [])

    assert "the agent CLI exited 1" in kept and "same conversation" in kept
    assert "/wt/s1 is as you left it" in kept and "- result r7: r7 has its result" in kept and "[e9]" in kept
    assert "reset to your branch head c4" in reset and "uncommitted edits were dropped" in reset
    # The worktree is compared with its last sync: a head that moved while the CLI ran is reset all the same.
    assert "your last worktree_sync" in reset and "while you were down" not in reset.split("\n")[1]
    assert "Nothing arrived for your branch" in reset


def test_the_role_says_a_branch_opened_on_a_focus_starts_from_that_operator():
    prompt = role_prompt(WORKLOAD)

    label = FOCUS_LINE.split(",")[0]
    assert (
        f'When the brief names a "{label}", start from that operator, within your direction if you have one' in prompt
    )


def test_the_role_says_how_to_record_a_direction_done_on_a_head_with_every_reason():
    prompt = role_prompt(WORKLOAD)

    assert f'starts with "{EXHAUSTED_PREFIX}"' in prompt
    assert all(why.value in prompt for why in ExhaustedWhy)
    assert "does not stop your branch" in prompt
    assert "at least one: a direction you have not tested is not done" in prompt
    assert "direction (the name of the session's direction exactly as listed below" in prompt


def test_the_role_tests_a_replayable_champion_on_the_head_before_other_directions():
    prompt = role_prompt(WORKLOAD)

    assert "get its delta with blackboard_replay and test it on your head before other directions" in prompt
    assert f'A champion the brief lists as "{ON_HEAD}" is in your head already: build on it' in prompt


def test_the_role_points_at_the_brief_s_unfinished_leads_by_their_section_title():
    prompt = role_prompt(WORKLOAD)

    title = UNFINISHED_TITLE.split(" (")[0]
    assert f'A lead under "{title}" in the brief' in prompt


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


#: The options of the server the rule pack states, which a prompt repeats.
_STATED_OPTIONS = server_options_named(
    " ".join((*NATIVE_SERVING["llamacpp"].values(), *NATIVE_AUX_MODEL_FLAGS["llamacpp"]))
)


def _handed_over(text: str) -> set[str]:
    """What ``text`` names that a specialist is to find: a prompt teaches the method and asks for evidence, and an
    option of the server, a word of a kernel knob or a pull request it named would be the answer handed over."""
    return (server_options_named(text) - _STATED_OPTIONS) | knob_words_named(text) | pull_requests_named(text)


def test_the_directions_of_a_checkout_built_server_name_no_answer():
    assert [_handed_over(text) for text in CHECKOUT_DIRECTIONS] == [set()] * len(CHECKOUT_DIRECTIONS)


def test_each_prompt_is_versioned_by_its_own_text(monkeypatch):
    versions = prompt_versions()

    assert set(versions) == {"specialist_role", "specialist_launch"}
    assert all(len(version) == VERSION_HEX and int(version, 16) >= 0 for version in versions.values())
    monkeypatch.setattr(prompts, "_ROLE", prompts._ROLE + "- One more rule.\n")
    edited = prompt_versions()
    assert edited["specialist_role"] != versions["specialist_role"]
    assert edited["specialist_launch"] == versions["specialist_launch"]


@pytest.mark.parametrize(
    ("name", "edit", "version"),
    [
        ("_DEV_SANDBOX", lambda text: text + " One more rule.", "specialist_role"),
        ("_NATIVE", lambda text: text + " One more rule.", "specialist_role"),
        ("_CHECKOUT_CARRIED", lambda text: text + " One more rule.", "specialist_role"),
        ("_CHECKOUT_LOGS", lambda text: text + " One more rule.", "specialist_role"),
        ("DISCOVERY_TEXTS", lambda texts: (*texts, "One more rule."), "specialist_role"),
        ("_REFERENCES", lambda text: text + " One more rule.", "specialist_role"),
        ("_BASE", lambda text: text + " One more rule.", "specialist_role"),
        ("_LAUNCH_CONFIG_ONLY", lambda text: text + " One more rule.", "specialist_role"),
        ("_HOLDS", lambda holds: {**holds, "geak": holds["geak"] + " more"}, "specialist_role"),
        ("_INFERENCEX_GRADING", lambda grading: replace(grading, trade_off=", more"), "specialist_role"),
        ("_LEASE_WAKE", lambda text: text + " One more rule.", "specialist_launch"),
        ("_UNPROFILED", lambda text: text + " One more rule.", "specialist_role"),
        ("_CHECKOUT_REVIEW", lambda text: text + " One more rule.", "specialist_role"),
        ("_FIRST_IDEA_CHECKOUT", lambda text: text + " One more rule.", "specialist_role"),
        ("_WAIT_WAKE", lambda text: text + " One more rule.", "specialist_launch"),
    ],
)
def test_a_text_sent_only_under_some_settings_still_changes_its_prompt_s_version(monkeypatch, name, edit, version):
    # A session with dev leases, native trees or references is told these texts, so they belong to its fingerprint.
    before = prompt_versions()
    monkeypatch.setattr(prompts, name, edit(getattr(prompts, name)))

    assert prompt_versions()[version] != before[version]


def test_a_session_held_to_launch_configuration_ends_its_role_by_saying_what_it_keeps_and_refuses():
    kept = role_prompt(WORKLOAD)
    held = role_prompt(replace(WORKLOAD, source_changes=False))

    assert "--no-source-changes" not in kept, "a session that keeps source changes is told nothing of the option"
    assert held.startswith(kept), "the role before it is the one every session is given"
    told = " ".join(held[len(kept) :].split())
    for said in (
        "started with --no-source-changes",
        "nothing but server arguments and environment variables",
        "This overrides everything above",
        "refuses, as launch_config_only, every test whose recipe holds a patch or a kernel artifact",
        "Claim config intents only",
        "start no kernel_tune and no knob_tune",
        'A "left out:" patch or kernel of a replayed champion stays out',
        "A setting that names a file you wrote",
        "here, test the next setting",
        "EXHAUSTED with why all_tested_or_claimed",
        "a branch whose turns keep ending with nothing to test is stopped",
    ):
        assert said in told, said


def test_the_role_of_a_session_held_to_launch_configuration_still_ends_with_what_only_it_is_told(tmp_path):
    """The base launch and the references stay in it: the block overrides the source work, not the session's facts."""
    base = ServerBase(args="--tp 2", env={"SGLANG_USE_AITER": "1"})
    held = role_prompt(replace(WORKLOAD, source_changes=False), references={"notes": tmp_path}, server=RoleServer(base))

    assert held.index("Reference material for this session") < held.index("The session's base launch")
    assert held.index("The session's base launch") < held.index("Launch configuration only.")


def test_a_launch_shows_the_latest_entries_cut_to_size_and_how_the_last_session_ended():
    entries = [_record(n) for n in range(RECENT_ENTRIES + 4)] + [_record(99, "x" * 1000, ("s1-17",))]

    prompt = _launch(recent=entries, last_end="the agent CLI exited 1")

    assert prompt.split("\n\n")[1].startswith("Your branch: s1,")
    assert "[e4] result: entry 4" not in prompt and "[e5] result: entry 5" in prompt
    assert f"[e99] result (s1-17): {'x' * (ENTRY_CHARS - 3)}...\n" in prompt
    assert "Your previous session ended: the agent CLI exited 1." in prompt


def test_the_role_names_the_session_s_reference_material_only_when_it_has_some(tmp_path):
    geak = tmp_path / "MB-reference" / "geak"

    prompt = role_prompt(WORKLOAD, references={"geak": geak})

    assert f"geak at {geak}" in prompt and "read-only" in prompt
    assert "name the file you used in your branch log" in prompt
    assert "Reference material for this session" not in role_prompt(WORKLOAD)


def test_a_reference_the_role_knows_is_described_and_another_is_not(tmp_path):
    described = role_prompt(WORKLOAD, references={"geak": tmp_path / "geak"})
    plain = role_prompt(WORKLOAD, references={"notes": tmp_path / "notes"})

    assert "perf_knowledge" in described and "perf_knowledge" not in plain


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


def test_the_role_names_native_sources_only_when_the_session_lays_them_out(tmp_path):
    tree = NativeTree(
        SourceRoot("python/sglang/kernels/aot", tmp_path), "https://github.com/sgl-project/sglang.git", "2" * 40
    )

    native = role_prompt(WORKLOAD, native={"sgl_kernel": tree})

    assert "sgl_kernel" not in role_prompt(WORKLOAD).split("sglang/kernels/jit/csrc")[1]
    assert "python/sglang/kernels/aot" in native and "pending_build" in native and "CMakeLists.txt" in native
    assert native.startswith(role_prompt(WORKLOAD))


def test_the_role_tells_a_specialist_of_a_checkout_built_framework_which_trees_it_may_change():
    server = RoleServer(checkouts=("llamacpp",), test_outputs=_TEST_OUTPUTS)
    prompt = " ".join(role_prompt(WORKLOAD, server=server).split())

    assert "llamacpp's source: your worktree holds the llamacpp repository" in prompt
    assert "ggml/, src/, include/, common/, tools/server/. Only these trees are yours to change" in prompt
    assert "CMakeLists.txt" in prompt and "pending_build" in prompt and "worktree_check does not apply" in prompt
    assert "llamacpp's source" not in role_prompt(WORKLOAD)


def test_what_a_specialist_of_a_checkout_built_server_is_told_names_no_answer(tmp_path):
    server = RoleServer(checkouts=("llamacpp",), test_outputs=_inventory(tmp_path).parent, profiled=False)
    role = role_prompt(WORKLOAD, server=server)

    assert _handed_over(role) == set()
    assert not any(knob in role for knob in kernel_knobs())
    assert "the settings branch sweeps" not in role
    assert _handed_over(prompts.nudge_message(profiled=False)) == set()


#: The directory a test result's output_dir is relative to.
_TEST_OUTPUTS = Path("/mb/MB-toolbox")


def _inventory(directory: Path) -> Path:
    """The option file of a server that takes three options, one of them set by its launch, in ``directory``."""
    options = (
        ServerOption(("-a", "--alpha"), "the first", set_by_base=True),
        ServerOption(("--beta",), "the second", set_by_base=False),
        ServerOption(("--gamma", "--no-gamma"), "the third", set_by_base=False),
    )
    return write_inventory(OptionInventory("server", "0" * 64, "--alpha 1", options), directory)
