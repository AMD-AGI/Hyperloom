# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resuming the consistency check across machine losses (``--resume``), on fake backends."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure import consistency, consistency_resume as resume
from hyperloom.measure.consistency import ChecklistItem
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome

NODE = "node-a"


class FakeBackend:
    """One side: writes a valid benchmark report and says which node it ran on; ``script`` steers single calls."""

    def __init__(self, node: str = NODE, throughput: float = 5000.0) -> None:
        self.node, self.throughput = node, throughput
        self.calls: list[str] = []
        self.script: dict[str, object] = {}
        self.journal = SimpleNamespace(latest=lambda _id: SimpleNamespace(state="finished", extra={"verified": True}))

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call.task_id)
        action = next((v for k, v in self.script.items() if call.task_id.startswith(k)), None)
        if isinstance(action, BaseException):
            raise action
        sandbox = SandboxFacts(sandbox_id=f"sb-{call.task_id}", node_name=self.node)
        if action == "fail":
            return TestOutcome(1, "", "boom", (), sandbox, call.task_id, "candidate", "import_error")
        workspace = Path(call.output_dir) / "benchmark_local_20261014_000000"
        workspace.mkdir(parents=True)
        report = {
            "success": True,
            "throughput": {"output_throughput": self.throughput, "completed_requests": 8, "num_prompts": 8},
        }
        (workspace / "benchmark_report.json").write_text(json.dumps(report))
        return TestOutcome(0, "", "", (RoundOutcome(0, "measure", workspace, 0),), sandbox, call.task_id)


def _items(*ids: str, repeats: int = 2) -> list[ChecklistItem]:
    def call(output_dir: Path, task_id: str) -> TestCall:
        return TestCall(
            points=(),
            output_dir=output_dir,
            timeout_sec=60,
            cwd=str(output_dir),
            magpie_python=sys.executable,
            task_id=task_id,
        )

    return [ChecklistItem(i, call, repeats=repeats) for i in ids]


class Rig:
    def __init__(self, tmp: Path) -> None:
        self.local, self.hosted = FakeBackend(), FakeBackend()
        self.jsonl, self.report_dir = tmp / "shared" / "c.jsonl", tmp / "reports"
        self.started = 0

    def backends(self):
        self.started += 1
        return self.local, self.hosted

    def run(self, items, **kwargs):
        files = resume.ResumeFiles(self.jsonl, self.report_dir)
        return resume.run_resumable(items, self.backends, files, resume.ResumeSettings(**kwargs))

    def lines(self, kind: str | None = None) -> list[dict]:
        return [x for x in resume.read_lines(self.jsonl) if kind in (None, x["type"])]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    monkeypatch.setenv("NODE_NAME", NODE)
    return Rig(tmp_path)


def test_units_are_repetition_major_so_each_pass_gives_every_test_one_more_reading() -> None:
    units = resume.plan_units(_items("base", "patched", repeats=2))
    assert [u.key for u in units] == [("base", 1), ("patched", 1), ("base", 2), ("patched", 2)]


def test_a_fresh_run_writes_one_line_per_pair_with_node_and_run_and_a_summary_over_them(rig: Rig) -> None:
    summary = rig.run(_items("base", "patched"), run_id="run-1")
    units = rig.lines("unit")
    assert sorted((u["id"], u["rep"]) for u in units) == [("base", 1), ("base", 2), ("patched", 1), ("patched", 2)]
    assert {(u["node_name"], u["run_id"]) for u in units} == {(NODE, "run-1")}
    assert all(u["compared"] and u["duration_sec"] >= 0 and u["attempts"] == 1 for u in units)
    assert (summary["verdict"], summary["units_complete"], summary["units_expected"]) == ("PASS", 4, 4)
    assert (summary["nodes"], summary["node_names"]) == (1, [NODE])


def test_rerunning_a_finished_file_measures_nothing_and_does_not_even_start_the_backends(rig: Rig) -> None:
    items = _items("base", "patched")
    rig.run(items, run_id="run-1")
    before = rig.jsonl.read_text()
    summary = rig.run(items, run_id="run-2")
    assert rig.started == 1 and rig.jsonl.read_text() == before
    assert summary["verdict"] == "PASS" and summary["units_complete"] == 4


def test_a_unit_interrupted_midway_leaves_no_line_and_the_next_run_does_only_the_rest(rig: Rig) -> None:
    items = _items("base", "patched")
    rig.hosted.script = {"patched.run-1": KeyboardInterrupt()}  # the machine is lost while patched rep 1 is hosted
    with pytest.raises(KeyboardInterrupt):
        rig.run(items, run_id="run-1")
    assert [(u["id"], u["rep"]) for u in rig.lines()] == [("base", 1)], "only the pair that finished has a line"
    rig.hosted.script, rig.local.calls, rig.hosted.calls = {}, [], []
    summary = rig.run(items, run_id="run-2")
    assert sorted((u["id"], u["rep"], u["run_id"]) for u in rig.lines("unit")) == [
        ("base", 1, "run-1"), ("base", 2, "run-2"), ("patched", 1, "run-2"), ("patched", 2, "run-2"),
    ]  # fmt: skip
    assert len(rig.hosted.calls) == 3, "the finished pair was not measured again"
    assert summary["runs"] == 2 and summary["verdict"] == "PASS"


def test_a_pair_whose_sides_ran_on_different_nodes_is_never_recorded_as_a_unit(rig: Rig) -> None:
    rig.hosted.node = "node-b"
    summary = rig.run(_items("base", repeats=1), run_id="run-1")
    assert rig.lines("unit") == []
    (attempt,) = rig.lines("attempt_failed")
    assert attempt["reason"].startswith("cross_node") and "node-b" in attempt["reason"]
    assert summary["verdict"] == "INCOMPLETE" and summary["units_complete"] == 0
    rig.run(_items("base", repeats=1), run_id="run-2")
    assert rig.lines("unit") == [], "the cross-node pair is not accepted on a retry either, it is only ever retried"


def test_a_pair_that_ran_on_another_node_than_the_driver_is_not_a_unit_either(rig: Rig) -> None:
    rig.local.node = rig.hosted.node = "node-b"  # both sides agree, but not with the process that drives them
    rig.run(_items("base", repeats=1), run_id="run-1")
    assert rig.lines("unit") == [] and len(rig.lines("attempt_failed")) == 1


def _unit_line(
    item: str,
    rep: int,
    node: str,
    *,
    local: float = 5000.0,
    hosted: float = 5000.0,
    hosted_node: str | None = None,
    completed_equal: bool = True,
) -> dict:
    side = {"returncode": 0, "problem": None}
    return {
        "type": "unit", "id": item, "rep": rep, "node_name": node, "run_id": f"r-{node}", "compared": True,
        "local": {**side, "output_throughput": local, "node_name": node},
        "hosted": {**side, "output_throughput": hosted, "node_name": hosted_node or node},
        "throughput_diff_pct": abs(hosted - local) / local * 100, "completed_equal": completed_equal,
        "error_rate_equal": True, "rebuilt_files_equal": None, "signature_verified": True, "problems": [],
    }  # fmt: skip


def test_the_summary_compares_each_tests_means_over_units_from_several_nodes(tmp_path: Path) -> None:
    units = resume.plan_units(_items("t", repeats=3))
    lines = [
        _unit_line("t", 1, "n1", hosted=5010.0),
        _unit_line("t", 2, "n2", hosted=4990.0),
        _unit_line("t", 3, "n1", hosted=5005.0),
    ]
    summary = resume.resume_summary(units, lines)
    assert (summary["verdict"], summary["items"], summary["units_complete"]) == ("PASS", 1, 3)
    assert (summary["nodes"], summary["node_names"]) == (2, ["n1", "n2"])
    assert summary["median_diff_pct"] == pytest.approx(5 / 3 / 5000 * 100), "the mean of 5010, 4990, 5005 vs 5000"


def test_units_that_fail_one_by_one_pass_on_the_test_means_as_consistency_repeats_judges_them(tmp_path: Path) -> None:
    units = resume.plan_units(_items("a", "b", repeats=2))
    lines = [
        _unit_line("a", 1, "n1", hosted=5050.0),
        _unit_line("a", 2, "n2", hosted=4950.0),
        _unit_line("b", 1, "n1", hosted=5040.0),
        _unit_line("b", 2, "n2", hosted=4970.0),
    ]
    assert statistics.median(line["throughput_diff_pct"] for line in lines) == pytest.approx(0.9), "unit by unit: FAIL"
    summary = resume.resume_summary(units, lines)
    assert summary["verdict"] == "PASS", summary["failures"]
    assert (summary["median_diff_pct"], summary["p90_diff_pct"]) == (pytest.approx(0.05), pytest.approx(0.1))
    missing = resume.resume_summary(units, lines[:3])
    assert missing["verdict"] == "INCOMPLETE" and missing["failures"][0] == "missing units: 1 of 4"


def test_a_unit_whose_completed_requests_differ_fails_the_test_even_when_the_means_agree(tmp_path: Path) -> None:
    units = resume.plan_units(_items("t", repeats=2))
    lines = [_unit_line("t", 1, "n1"), _unit_line("t", 2, "n1", completed_equal=False)]
    summary = resume.resume_summary(units, lines)
    assert summary["verdict"] == "FAIL" and summary["completed_all_equal"] is False


def test_a_threshold_failure_over_the_lines_is_a_fail_not_a_pass(tmp_path: Path) -> None:
    units = resume.plan_units(_items("t", repeats=2))
    lines = [_unit_line("t", 1, "n1", hosted=5045.0), _unit_line("t", 2, "n2", hosted=5040.0)]
    summary = resume.resume_summary(units, lines)
    assert summary["verdict"] == "FAIL" and any("median" in f for f in summary["failures"])


def test_a_line_that_pairs_two_nodes_is_not_counted_in_the_summary(tmp_path: Path) -> None:
    units = resume.plan_units(_items("t", repeats=1))
    summary = resume.resume_summary(units, [_unit_line("t", 1, "n1", hosted_node="n2")])
    assert summary["units_complete"] == 0 and summary["verdict"] == "INCOMPLETE"


def test_the_first_line_of_a_unit_wins_when_two_runs_both_wrote_it(tmp_path: Path) -> None:
    units = resume.plan_units(_items("t", repeats=1))
    lines = [_unit_line("t", 1, "n1", hosted=5005.0), _unit_line("t", 1, "n2", hosted=5250.0)]
    summary = resume.resume_summary(units, lines)
    assert summary["units_complete"] == 1 and summary["nodes"] == 1 and summary["verdict"] == "PASS"


def test_a_failing_side_is_retried_and_after_max_attempts_counts_against_the_verdict(rig: Rig) -> None:
    items = _items("base", repeats=1)
    rig.hosted.script = {"base": "fail"}
    for run in ("run-1", "run-2"):
        assert rig.run(items, run_id=run, max_attempts=3)["verdict"] == "INCOMPLETE"
    assert rig.lines("unit") == [] and len(rig.lines("attempt_failed")) == 2
    summary = rig.run(items, run_id="run-3", max_attempts=3)
    (unit,) = rig.lines("unit")
    assert unit["attempts"] == 3 and not unit["compared"]
    assert summary["verdict"] == "FAIL" and summary["units_complete"] == 1


def test_a_torn_last_line_is_ignored_and_the_next_line_starts_on_a_fresh_line(rig: Rig) -> None:
    rig.jsonl.parent.mkdir(parents=True)
    rig.jsonl.write_text(json.dumps(_unit_line("base", 1, NODE)) + '\n{"type": "unit", "id": "base", "rep"')
    assert [(x["id"], x["rep"]) for x in resume.read_lines(rig.jsonl)] == [("base", 1)]
    rig.run(_items("base", repeats=2), run_id="run-1")
    assert sorted((u["id"], u["rep"]) for u in rig.lines("unit")) == [("base", 1), ("base", 2)]


def test_a_new_unit_starts_only_when_the_time_left_covers_one_but_the_first_always_starts(rig: Rig) -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 100.0
        return now[0]

    items = _items("base", "patched", repeats=2)
    summary = rig.run(items, run_id="run-1", lifetime_sec=300.0, unit_budget_sec=400.0, clock=clock)
    assert len(rig.lines("unit")) == 1, "one unit ran, the estimate of the next does not fit the time left"
    assert summary["verdict"] == "INCOMPLETE"
    rig2 = Rig(rig.jsonl.parent.parent / "other")
    rig2.run(items, run_id="run-2", lifetime_sec=1e-9, unit_budget_sec=900.0)
    assert len(rig2.lines("unit")) == 1, "a lifetime shorter than one unit still does one"


def test_the_estimate_of_a_unit_is_taken_from_the_longest_one_recorded(rig: Rig) -> None:
    done = {("a", 1): {"duration_sec": 100.0}, ("a", 2): {"duration_sec": 300.0}}
    assert resume._estimate(done, 900.0) == pytest.approx(375.0)
    assert resume._estimate({}, 900.0) == 900.0


def test_the_entry_point_resumes_from_the_file_and_exits_by_the_verdict(
    tmp_path: Path, monkeypatch, capsys, rig: Rig
) -> None:
    checklist = tmp_path / "checklist.json"
    point = tmp_path / "p.yaml"
    point.write_text("benchmark: {}\n")
    checklist.write_text(json.dumps({"items": [{"id": "base", "points": [str(point)], "repeats": 2}]}))
    config = tmp_path / "config.json"
    config.write_text("{}")
    built = []

    def build(_config: Path) -> consistency.Backends:
        built.append(1)
        return consistency.Backends(rig.local, rig.hosted, lambda: None)

    monkeypatch.setattr(resume, "build_backends", build)
    argv = [
        "--checklist",
        str(checklist),
        "--config",
        str(config),
        "--report-dir",
        str(tmp_path / "r"),
        "--resume",
        str(rig.jsonl),
    ]
    assert resume.main([*argv, "--lifetime-sec", "1e-9"]) == 10, "only the first unit starts"
    assert resume.main(argv) == 0, "the second submission measures the rest"
    assert resume.main(argv) == 0 and len(built) == 2, "the third one starts nothing"
    assert json.loads(Path(f"{rig.jsonl}.summary.json").read_text())["units_complete"] == 2
    assert capsys.readouterr().out.count('"verdict": "PASS"') == 2


# -- one run at a time: a duplicate submission, and a lost run that left a lock behind ---------------------------------


def _lock(tmp: Path, node: str = NODE, pid: int = 100, **probe) -> resume.RunLock:
    return resume.RunLock(tmp / "c.jsonl.lock", node=node, pid=pid, probe=resume.LockProbe(**probe))


def test_a_second_run_cannot_take_a_lock_that_is_held_and_fresh(tmp_path: Path) -> None:
    first, second = _lock(tmp_path, pid=1), _lock(tmp_path, node="node-b", pid=2)
    took = first.acquire()
    refused = not second.acquire()
    assert took and refused
    first.release()
    taken = second.acquire()
    assert taken, "released: the next run may go"


def test_a_lock_of_a_dead_process_on_this_node_is_taken_over_at_once(tmp_path: Path) -> None:
    taken = _lock(tmp_path, pid=1).acquire()
    assert taken
    taken = _lock(tmp_path, pid=2, pid_alive=lambda _pid: False).acquire()
    assert taken
    taken = _lock(tmp_path, pid=3, pid_alive=lambda _pid: True).acquire()
    assert not taken, "a live holder keeps it"


def test_a_lock_from_another_node_is_taken_over_only_when_its_heartbeat_stopped(tmp_path: Path) -> None:
    import os
    import time

    taken = _lock(tmp_path, node="node-a", pid=1).acquire()
    assert taken
    lost = lambda _pid: pytest.fail("the pid of another node means nothing here")  # noqa: E731
    now = [time.time()]
    other = _lock(tmp_path, node="node-b", pid=2, stale_after_sec=120.0, now=lambda: now[0], pid_alive=lost)
    taken = other.acquire()
    assert not taken, "held a moment ago"
    now[0] += 119.0
    taken = other.acquire()
    assert not taken
    now[0] += 2.0
    taken = other.acquire()
    assert taken, "silent for over 120 s: the machine that held it is gone"
    os.utime(other.path)
    assert any(p.name.startswith("c.jsonl.lock.stale.") for p in tmp_path.iterdir()), "the old lock is kept aside"


def test_the_heartbeat_keeps_a_lock_from_going_stale(tmp_path: Path) -> None:
    import os
    import time

    lock = _lock(tmp_path)
    taken = lock.acquire()
    assert taken
    os.utime(lock.path, (1.0, 1.0))
    lock.start_heartbeat(0.01)
    deadline = time.time() + 5
    while time.time() < deadline and lock.path.stat().st_mtime < 100:
        time.sleep(0.01)
    lock.release()
    assert not lock.path.exists() and time.time() < deadline, "the heartbeat touched it before the release"


def test_wait_polls_until_the_holder_lets_go_or_the_time_is_up(tmp_path: Path) -> None:
    holder = _lock(tmp_path, pid=1)
    waiter = _lock(tmp_path, node="node-b", pid=2)
    taken = holder.acquire()
    assert taken
    naps: list[float] = []
    gave_up = waiter.wait(0.0, sleep=naps.append)
    assert not gave_up and naps == []
    got_it = waiter.wait(60.0, poll_sec=5.0, sleep=lambda s: (naps.append(s), holder.release()))
    assert got_it and naps == [5.0]


def test_the_entry_point_stops_with_12_and_measures_nothing_while_another_run_holds_the_file(
    tmp_path: Path, rig: Rig, monkeypatch, capsys
) -> None:
    point = tmp_path / "p.yaml"
    point.write_text("benchmark: {}\n")
    checklist, config = tmp_path / "checklist.json", tmp_path / "config.json"
    checklist.write_text(json.dumps({"items": [{"id": "base", "points": [str(point)]}]}))
    config.write_text("{}")
    monkeypatch.setattr(resume, "build_backends", lambda _c: pytest.fail("a locked file must not start a backend"))
    held = resume.RunLock(Path(f"{rig.jsonl}.lock"), node="node-b", pid=7)
    taken = held.acquire()
    assert taken
    argv = ["--checklist", str(checklist), "--config", str(config), "--resume", str(rig.jsonl), "--lock-wait-sec", "0"]
    assert resume.main(argv) == resume.LOCKED_RC
    assert "another run" in capsys.readouterr().err and rig.lines() == []


def test_a_repetition_is_counted_from_one() -> None:
    with pytest.raises(ValueError, match="counted from 1"):
        consistency.check_repetition(_items("t")[0], Path("."), FakeBackend(), FakeBackend(), 0)
