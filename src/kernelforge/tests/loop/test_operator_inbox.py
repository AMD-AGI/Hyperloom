# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator messages: how they are posted, consumed, withdrawn, and where they surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.workspace.edit_policy import protected_path_inventory
from kernelforge.loop import operator_inbox
from kernelforge.loop.operator_inbox import CLEAR, OperatorInbox, post_message
from kernelforge.loop.runner import IterationLoop, _lane_prompt, _operator_block
from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext


def _consumed(workspace: Path, iteration: int, *texts: str) -> None:
    for text in texts:
        post_message(workspace, text)
    OperatorInbox(workspace).consume(iteration)


def test_a_message_waits_until_consumed_then_stands(tmp_path: Path) -> None:
    post_message(tmp_path, "first")
    post_message(tmp_path, "second")
    inbox = OperatorInbox(tmp_path)
    assert len(inbox.pending()) == 2
    assert inbox.active_guidance() == ()

    consumed = inbox.consume(5)

    assert [message.text for message in consumed] == ["first", "second"]
    assert inbox.pending() == []
    assert all(path.name.startswith("iter_005-") for path in inbox.archive.iterdir())
    assert inbox.active_guidance() == ("second", "first")


def test_clear_withdraws_everything_before_it(tmp_path: Path) -> None:
    _consumed(tmp_path, 1, "old advice")
    _consumed(tmp_path, 2, CLEAR)
    assert OperatorInbox(tmp_path).active_guidance() == ()
    _consumed(tmp_path, 3, "new advice")
    assert OperatorInbox(tmp_path).active_guidance() == ("new advice",)


def test_a_message_still_being_written_is_not_read(tmp_path: Path) -> None:
    folder = tmp_path / "forge_experiments" / "inbox"
    folder.mkdir(parents=True)
    (folder / ".00000000000000000001-1-abcdef.md.tmp").write_text("half")
    assert OperatorInbox(tmp_path).pending() == []


def test_guidance_keeps_the_newest_within_its_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operator_inbox, "GUIDANCE_CHAR_LIMIT", 25)
    _consumed(tmp_path, 1, "a" * 10, "b" * 10, "c" * 10)
    assert OperatorInbox(tmp_path).active_guidance() == ("c" * 10, "b" * 10)


def test_forge_say_posts_and_clears(tmp_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["forge-say", "--workspace", str(tmp_path), "focus", "on", "M<=64"]).exit_code == 0
    assert runner.invoke(main, ["forge-say", "--workspace", str(tmp_path), "--clear"]).exit_code == 0
    texts = [path.read_text().strip() for path in OperatorInbox(tmp_path).pending()]
    assert sorted(texts) == sorted(["focus on M<=64", CLEAR])
    assert runner.invoke(main, ["forge-say", "--workspace", str(tmp_path)]).exit_code == 2


def _context(**overrides) -> OrchestrationContext:
    values = dict(
        analysis_commit="abc",
        workspace="/ws",
        gpu_target="gfx950",
        objective="speed",
        program_context="Optimize the kernel.",
        source_map_path="/ws/kernel.py",
        cases=(CaseEvidence(case_id="m=8", latency_ms=1.0),),
    )
    return OrchestrationContext(**(values | overrides))


def test_the_planning_payload_carries_guidance_only_when_there_is_some() -> None:
    assert "operator_guidance" not in _context().to_prompt_dict()
    payload = _context(operator_guidance=("newest", "older")).to_prompt_dict()["operator_guidance"]
    assert payload["messages_newest_first"] == ["newest", "older"]
    assert payload["priority"].startswith("highest")


def test_a_lane_reads_the_operator_first() -> None:
    plain = _lane_prompt("PLAN", serialized_driver=Path("/d.py"))
    guided = _lane_prompt("PLAN", serialized_driver=Path("/d.py"), guidance=("line one\nline two",))
    assert plain.startswith("`python3 /d.py`")
    assert guided.startswith("## Operator instructions (highest priority)")
    assert "- line one\n  line two" in guided
    assert guided.endswith(plain)
    assert _operator_block(()) == ""


def test_the_loop_archives_and_records_what_it_consumes(tmp_path: Path) -> None:
    post_message(tmp_path, "try the 16-row tile")
    events = []
    loop = SimpleNamespace(
        ic=SimpleNamespace(workspace_dir=str(tmp_path)),
        state_store=SimpleNamespace(append_event=events.append),
        persistence_errors=[],
    )
    IterationLoop._consume_operator_messages(loop, 7)
    assert [event["type"] for event in events] == ["operator_message_consumed"]
    assert events[0]["text"] == "try the 16-row tile"
    assert IterationLoop._operator_guidance(loop) == ("try the 16-row tile",)


def test_a_message_is_not_a_protected_change(tmp_path: Path) -> None:
    post_message(tmp_path, "hello")
    inventory = protected_path_inventory(tmp_path, extra_globs=("*",))
    assert not [path for path in inventory if "inbox" in path.parts]
