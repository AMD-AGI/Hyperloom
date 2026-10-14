# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The consistency check on fake measurements (design 12.1 test_consistency, 12.4 stage 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import consistency
from hyperloom.measure.consistency import (
    FARM_INVENTORY_FILE,
    ChecklistItem,
    Criteria,
    ItemReport,
    Side,
    load_checklist,
    percentile,
    run_checklist,
    summarize,
)
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.tests.hosted_fixtures import FAKE_RUNNER, Rig, make_rig, point_yaml
from hyperloom.contracts.measurement import SandboxFacts

REPORT = "benchmark_local_20261014_000000"


class FakeLocal:
    """The local backend of the check: writes one report from the point's own ``local_*`` keys."""

    name = "fake-local"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.raises: Exception | None = None

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call.task_id)
        config = yaml.safe_load(Path(call.points[0]).read_text())
        sandbox = SandboxFacts(sandbox_id=f"local-{call.task_id}", backend="local")
        if config.get("local_fail"):
            return TestOutcome(1, "", "boom", (), sandbox, call.task_id, "candidate", "import_error")
        workspace = Path(call.output_dir) / REPORT
        workspace.mkdir(parents=True)
        report = {
            "success": True,
            "throughput": {
                "output_throughput": float(config.get("local_throughput", config.get("fake_throughput", 5000.0))),
                "completed_requests": int(config.get("local_completed", config.get("fake_completed", 64))),
            },
        }
        (workspace / "benchmark_report.json").write_text(json.dumps(report))
        return TestOutcome(0, "", "", (RoundOutcome(0, "measure", workspace, 0),), sandbox, call.task_id)


def _item(tmp: Path, name: str, patched: dict[str, str] | None = None, **fake) -> ChecklistItem:
    point = point_yaml(tmp, f"{name}.yaml", **fake)

    def call(output_dir: Path, task_id: str) -> TestCall:
        return TestCall(
            points=(point,),
            output_dir=output_dir,
            timeout_sec=60,
            cwd=str(tmp),
            magpie_python=sys.executable,
            task_id=task_id,
        )

    return ChecklistItem(name, call, patched or {})


@pytest.fixture
def rig(tmp_path: Path):
    built: list[Rig] = []

    def make(**kwargs) -> Rig:
        r = make_rig(tmp_path, **kwargs)
        built.append(r)
        return r

    yield make
    for r in built:
        r.backend.close()
        r.provider.close()


def _check(rig_, tmp: Path, items, local=None):
    summary = run_checklist(items, local=local or FakeLocal(), hosted=rig_.backend, report_dir=tmp / "reports")
    lines = [json.loads(x) for x in (tmp / "reports" / "consistency.jsonl").read_text().splitlines()]
    return summary, lines


def test_identical_measurements_pass_and_every_item_leaves_a_complete_record(rig, tmp_path) -> None:
    r = rig()
    items = [_item(tmp_path, n, fake_throughput=5000 + i) for i, n in enumerate(["sglang-dense", "vllm-dense"])]
    summary, lines = _check(r, tmp_path, items)
    assert summary.verdict == "PASS" and (summary.items, summary.compared) == (2, 2)
    assert summary.median_diff_pct == 0 and summary.p90_diff_pct == 0
    assert [x["type"] for x in lines] == ["item", "item", "summary"]
    item = lines[0]
    assert set(item) == {
        "type", "id", "compared", "local", "hosted", "throughput_diff_pct", "completed_equal",
        "error_rate_equal", "rebuilt_files_equal", "signature_verified", "problems",
    }  # fmt: skip
    assert item["local"]["output_throughput"] == item["hosted"]["output_throughput"] == 5000.0
    assert item["local"]["completed_requests"] == item["hosted"]["completed_requests"] == [64]
    assert item["signature_verified"] is True and item["rebuilt_files_equal"] is None and item["problems"] == []
    assert lines[-1]["verdict"] == "PASS"


def test_the_two_sides_run_one_after_the_other_never_together(rig, tmp_path) -> None:
    r = rig()
    order: list[str] = []

    class Spy(FakeLocal):
        def run_test(self, call):
            order.append(f"local-start:{call.task_id}")
            outcome = super().run_test(call)
            order.append(f"local-end:{call.task_id}")
            return outcome

    real = r.backend.run_test
    r.backend.run_test = lambda call: (order.append(f"hosted-start:{call.task_id}"), real(call))[1]
    _check(r, tmp_path, [_item(tmp_path, "a"), _item(tmp_path, "b")], local=Spy())
    assert order == [
        "local-start:a-local", "local-end:a-local", "hosted-start:a-hosted",
        "local-start:b-local", "local-end:b-local", "hosted-start:b-hosted",
    ]  # fmt: skip


def test_a_throughput_difference_beyond_the_median_fails_the_check(rig, tmp_path) -> None:
    r = rig()
    items = [_item(tmp_path, f"t{i}", fake_throughput=5000, local_throughput=5000 * 0.99) for i in range(3)]
    summary, lines = _check(r, tmp_path, items)
    assert summary.verdict == "FAIL"
    assert any("median throughput difference" in f for f in summary.failures)
    assert lines[0]["throughput_diff_pct"] == pytest.approx(1.0101, abs=1e-3)


def test_one_outlier_fails_the_p90_even_when_the_median_is_fine(rig, tmp_path) -> None:
    r = rig()
    items = [_item(tmp_path, f"t{i}", fake_throughput=5000) for i in range(8)]
    items += [_item(tmp_path, f"outlier{i}", fake_throughput=5000, local_throughput=5000 / 1.02) for i in range(2)]
    summary, _ = _check(r, tmp_path, items)
    assert summary.verdict == "FAIL" and summary.median_diff_pct == 0
    assert any("p90 throughput difference" in f for f in summary.failures)
    assert summary.p90_diff_pct == pytest.approx(2.0, abs=1e-6)


def test_a_different_completed_request_count_fails_the_check(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [_item(tmp_path, "t", fake_completed=64, local_completed=63)])
    assert summary.verdict == "FAIL" and summary.completed_all_equal is False
    assert "completed_requests_differ" in lines[0]["problems"]


def test_a_hosted_failure_is_one_failed_check_and_keeps_both_logs(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [_item(tmp_path, "t", fake_mode="node_fault")])
    assert summary.verdict == "FAIL" and summary.compared == 0
    item = lines[0]
    assert item["compared"] is False and "hosted: test_failed" in item["problems"][0]
    local_dir, hosted_dir = Path(item["local"]["log_dir"]), Path(item["hosted"]["log_dir"])
    assert (local_dir / "outcome.json").is_file() and (local_dir / REPORT).is_dir(), "the local side is kept"
    assert (hosted_dir / "outcome.json").is_file() and (hosted_dir / "hosted-server.log").is_file(), (
        "so is the hosted side"
    )


def test_a_local_failure_cannot_be_compared_and_never_passes(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [_item(tmp_path, "t", local_fail="true")])
    assert summary.verdict == "FAIL" and lines[0]["compared"] is False
    assert any(p.startswith("local: test_failed") for p in lines[0]["problems"])


def test_a_side_that_raises_is_recorded_not_fatal(rig, tmp_path) -> None:
    r = rig()

    class Broken(FakeLocal):
        def run_test(self, call):
            raise RuntimeError("no GPU")

    summary, lines = _check(r, tmp_path, [_item(tmp_path, "a"), _item(tmp_path, "b")], local=Broken())
    assert summary.verdict == "FAIL" and summary.items == 2
    assert lines[0]["problems"][0] == "local: raised: RuntimeError: no GPU"


def test_a_result_that_fails_verification_is_a_failed_check_with_no_signature_credit(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [_item(tmp_path, "t", fake_mode="wrong_key")])
    assert summary.verdict == "FAIL" and summary.signatures_all_verified is False
    assert lines[0]["signature_verified"] is False and "signature_not_verified" in lines[0]["problems"]


def test_a_hosted_pass_without_a_verified_journal_record_gets_no_signature_credit(tmp_path) -> None:
    """A hosted side that reports success but cannot show it verified the signature is not credited with it."""

    class NoRecord:
        class journal:
            @staticmethod
            def latest(request_id):
                return None

        def run_test(self, call):
            return FakeLocal().run_test(call)

    summary = run_checklist(
        [_item(tmp_path, "t")], local=FakeLocal(), hosted=NoRecord(), report_dir=tmp_path / "reports"
    )
    assert summary.verdict == "FAIL" and summary.signatures_all_verified is False


def test_an_empty_checklist_is_no_data_not_a_pass(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [])
    assert summary.verdict == "NO_DATA" and summary.items == 0
    assert lines == [summary.to_document()]


# -- rebuilt files --------------------------------------------------------------------------------------------------


def test_rebuilt_file_hashes_that_match_the_local_compile_pass(rig, tmp_path) -> None:
    r = rig()
    inventory = {"srt/a.py": "aa" * 32, "srt/b.py": "bb" * 32}
    item = _item(tmp_path, "patched", patched=dict(inventory), fake_inventory=json.dumps(inventory))
    summary, lines = _check(r, tmp_path, [item])
    assert summary.verdict == "PASS" and lines[0]["rebuilt_files_equal"] is True


def test_a_rebuilt_file_that_differs_fails_and_names_the_file(rig, tmp_path) -> None:
    r = rig()
    item = _item(
        tmp_path,
        "patched",
        patched={"srt/a.py": "aa" * 32},
        fake_inventory=json.dumps({"srt/a.py": "cc" * 32}),
    )
    summary, lines = _check(r, tmp_path, [item])
    assert summary.verdict == "FAIL" and summary.rebuilt_files_all_equal is False
    assert "rebuilt_file_differs: srt/a.py" in lines[0]["problems"]


def test_a_patched_test_without_an_inventory_cannot_be_compared_and_fails(rig, tmp_path) -> None:
    r = rig()
    summary, lines = _check(r, tmp_path, [_item(tmp_path, "patched", patched={"srt/a.py": "aa" * 32})])
    assert summary.verdict == "FAIL" and lines[0]["compared"] is False
    assert f"not_comparable: the runner listed no {FARM_INVENTORY_FILE}" in lines[0]["problems"]


def test_a_patched_file_missing_from_the_inventory_cannot_be_compared(rig, tmp_path) -> None:
    r = rig()
    item = _item(tmp_path, "p", patched={"srt/a.py": "aa" * 32}, fake_inventory=json.dumps({"srt/other.py": "aa" * 32}))
    summary, lines = _check(r, tmp_path, [item])
    assert summary.verdict == "FAIL" and lines[0]["compared"] is False


# -- the arithmetic -------------------------------------------------------------------------------------------------


def _report(item_id: str, diff: float | None, *, compared: bool = True, **overrides) -> ItemReport:
    side = Side(0, 5000.0, (64,), (None,))
    values = {
        "id": item_id, "local": side, "hosted": side, "throughput_diff_pct": diff, "completed_equal": True,
        "error_rate_equal": True, "rebuilt_files_equal": None, "signature_verified": True, "problems": (),
    }  # fmt: skip
    values.update(overrides)
    if not compared:
        values["hosted"] = Side(1, problem="x")
    return ItemReport(**values)


def test_the_limits_are_inclusive() -> None:
    reports = [_report(f"i{i}", 0.5 if i < 5 else 1.5) for i in range(10)]
    assert summarize(reports).verdict == "FAIL", "the median of five 0.5s and five 1.5s is 1.0"
    ok = [_report(f"i{i}", 0.5) for i in range(8)] + [_report(f"last{i}", 1.5) for i in range(2)]
    summary = summarize(ok)
    assert (summary.median_diff_pct, summary.p90_diff_pct, summary.verdict) == (0.5, 1.5, "PASS")


def test_a_hair_over_either_limit_fails() -> None:
    assert summarize([_report("a", 0.5001)]).verdict == "FAIL"
    assert (
        summarize([_report(f"i{i}", 0.1) for i in range(8)] + [_report(f"x{i}", 1.5001) for i in range(2)]).verdict
        == "FAIL"
    )


def test_percentile_is_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9
    assert percentile([5.0], 0.9) == 5.0


def test_one_uncompared_item_blocks_a_pass_even_when_all_others_agree() -> None:
    reports = [_report("a", 0.0), _report("b", None, compared=False)]
    summary = summarize(reports)
    assert summary.verdict == "FAIL" and summary.compared == 1
    assert any("not compared" in f for f in summary.failures)


def test_an_unverified_signature_blocks_a_pass() -> None:
    assert summarize([_report("a", 0.0, signature_verified=False)]).verdict == "FAIL"


def test_stage_two_criteria_are_configurable() -> None:
    reports = [_report(f"i{i}", 0.1) for i in range(8)] + [_report(f"x{i}", 3.0) for i in range(2)]
    assert summarize(reports).verdict == "FAIL"
    assert summarize(reports, Criteria(0.5, 3.5)).verdict == "PASS"


# -- the command ----------------------------------------------------------------------------------------------------


def test_the_checklist_file_is_checked_field_by_field(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"items": [{"id": "a", "points": ["p.yaml"], "bogus": 1}]}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_checklist(path)
    path.write_text(json.dumps({"items": [{"id": "a"}]}))
    with pytest.raises(ValueError, match="non-empty 'points'"):
        load_checklist(path)
    path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="'items' list"):
        load_checklist(path)


def _write_config(tmp: Path, **extra) -> Path:
    template = {"role": "measurement", "command": list(FAKE_RUNNER), "gpus": 1, "sealed": True, "signing": True}
    document = {
        "session_dir": str(tmp / "session"),
        "sandbox_dir": str(tmp / "sb"),
        "archive_dir": str(tmp / "archive"),
        "gpus": [0, 1],
        "templates": {"runner-1": template},
        "runner_templates": {"1": "runner-1"},
        "image_ref": "registry.invalid/fw@sha256:" + "22" * 32,
        "image_digest": "sha256:" + "22" * 32,
        "gpu_type": "FakeGPU",
        "magpie_commit": "a" * 40,
        "inferencex_commit": "a" * 40,
        "poll_interval_sec": 0.05,
        "local_backend": "hyperloom.measure.tests.test_consistency:FakeLocal",
        **extra,
    }
    path = tmp / "config.json"
    path.write_text(json.dumps(document))
    return path


def _write_checklist(tmp: Path, point: Path) -> Path:
    path = tmp / "checklist.json"
    path.write_text(json.dumps({"items": [{"id": "dense", "points": [str(point)], "timeout_sec": 60}]}))
    return path


def _main(tmp: Path, checklist: Path, config: Path) -> int:
    return consistency.main(
        ["--checklist", str(checklist), "--config", str(config), "--report-dir", str(tmp / "reports")]
    )


def test_the_entry_point_runs_a_checklist_against_the_local_sandbox_and_exits_by_verdict(tmp_path, capsys) -> None:
    point = point_yaml(tmp_path, "p.yaml", fake_throughput=4000)
    assert _main(tmp_path, _write_checklist(tmp_path, point), _write_config(tmp_path)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verdict"] == "PASS" and summary["items"] == 1
    last = json.loads((tmp_path / "reports" / "consistency.jsonl").read_text().splitlines()[-1])
    assert last["type"] == "summary"


def test_the_entry_point_exits_non_zero_when_it_could_not_compare(tmp_path, capsys) -> None:
    point = point_yaml(tmp_path, "p.yaml", fake_mode="node_fault")
    config = _write_config(tmp_path)
    assert _main(tmp_path, _write_checklist(tmp_path, point), config) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"
    assert consistency.main(["--checklist", str(tmp_path / "missing.json"), "--config", str(config)]) == 2
