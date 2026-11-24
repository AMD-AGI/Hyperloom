# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sweep of a kernel knob over fake builds and a fake microbenchmark: what it recommends, and what it refuses."""

from __future__ import annotations

import contextlib
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from hyperloom.common.outside_facts.kernel_knob import KnobSpecError
from hyperloom.measure.runner.native.backend_ops import BackendOpsError
from hyperloom.measure.runner.native.knob_sweep import SweepError, SweepHooks, SweepTiming, plan_sweep, run_sweep
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild, TreeBuildStatus

_ANCHOR = "    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {"
_MMVQ = "int calc_nwarps(int type, int ncols_dst, int table_id) {\n" + _ANCHOR + "\n        return 1;\n    }\n}\n"
_HEADER = """
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q8_0 = 8,
};
enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_DUP,
    GGML_OP_MUL_MAT, // matrix product
};
"""
_FORCED = re.compile(r"ncols_dst == (\d+)\) \{ return (\d+);")


def _source(path: str) -> str:
    return _HEADER if path.endswith("ggml.h") else _MMVQ


class _Fakes:
    """Builds that lay a directory per patch, and a benchmark whose time per key depends on the value the patch forces
    there, with a seeded jitter of one percent."""

    def __init__(
        self, tmp: Path, speed: Mapping[tuple[str, str], float], *, fail: Mapping[str, TreeBuild] = {}
    ) -> None:
        self.tmp, self.speed, self.fail = tmp, speed, fail
        self.forced: dict[Path, dict[str, str]] = {}
        self.holds = 0
        self.jitter = random.Random(7)
        self.checked: list[list[str]] = []
        self.verdict = "OK"

    def build(self, patch: bytes) -> TreeBuild:
        text = patch.decode()
        forced = dict(_FORCED.findall(text))
        value = next(iter(set(forced.values())), "")
        if value in self.fail:
            return self.fail[value]
        bin_dir = self.tmp / f"bin-{value or 'base'}"
        bin_dir.mkdir(exist_ok=True)
        self.forced[bin_dir] = forced
        return TreeBuild(TreeBuildStatus.BUILT, key=f"key-{value or 'base'}", bin=bin_dir)

    @contextlib.contextmanager
    def hold(self) -> Iterator[Mapping[str, str]]:
        self.holds += 1
        yield {"ROCR_VISIBLE_DEVICES": "3"}

    def time(self, executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str]) -> dict[str, float]:
        assert env["ROCR_VISIBLE_DEVICES"] == "3"
        forced = self.forced[executable.parent]
        times = {}
        for name in names:
            ncols = name.split("_")[2]
            value = forced.get(ncols, "")
            times[name] = 100.0 * self.speed.get((ncols, value), 1.0) * self.jitter.gauss(1.0, 0.01)
        return times

    def check(self, executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str]) -> dict[str, str]:
        self.checked.append(list(names))
        return dict.fromkeys(names, self.verdict)


def _plan(rounds: int = 5):
    return plan_sweep(
        "mmvq_nwarps_rdna3",
        {"type": ["Q8_0"], "ncols": ["1", "2", "3"]},
        ["2", "4"],
        SweepTiming([(64, 128)], rounds, 1),
    )


def _sweep(tmp_path: Path, fakes: _Fakes, rounds: int = 5):
    return run_sweep(
        _plan(rounds),
        _source,
        tmp_path / "work",
        SweepHooks(build=fakes.build, hold=fakes.hold, time=fakes.time, check=fakes.check),
    )


def test_a_value_faster_beyond_the_noise_wins_its_key_and_only_its_key(tmp_path: Path) -> None:
    fakes = _Fakes(tmp_path, {("2", "4"): 0.8, ("3", "4"): 0.998, ("1", "2"): 1.3})

    result = _sweep(tmp_path, fakes)

    assert result.winners == {("Q8_0", "2"): "4"}
    patch = result.patch.decode()
    assert patch.startswith("diff --git a/ggml/src/ggml-cuda/mmvq.cu b/ggml/src/ggml-cuda/mmvq.cu\n")
    assert "+        if (type == GGML_TYPE_Q8_0 && ncols_dst == 2) { return 4; }\n" in patch
    assert patch.count("\n+    ") == 1
    rows = {(row["key"], row["value"]): row for row in result.document["table"]}
    assert rows[("type=Q8_0,ncols=2", "4")]["verdict"] == "faster"
    assert rows[("type=Q8_0,ncols=3", "4")]["verdict"] == "within_noise"
    assert rows[("type=Q8_0,ncols=1", "2")]["verdict"] == "slower"
    assert result.document["no_winner"] == ["type=Q8_0,ncols=1", "type=Q8_0,ncols=3"]
    assert fakes.checked == [["Q8_0_2_64x128"]]
    assert fakes.holds == 1 + 5 + 1
    case_line = (tmp_path / "work" / "cases.txt").read_text().splitlines()[0]
    assert case_line == "2 0 64 1 1 1 0 2 8 128 64 1 1 0 0 0 0 0 128 1 1 1 0 0 0 0 Q8_0_1_64x128"


def test_a_winner_whose_check_against_the_cpu_fails_is_refused(tmp_path: Path) -> None:
    fakes = _Fakes(tmp_path, {("2", "4"): 0.8})
    fakes.verdict = "FAIL"

    result = _sweep(tmp_path, fakes)

    assert result.winners == {}
    assert result.patch == b""
    assert "FAIL" in result.document["refused"]["type=Q8_0,ncols=2"]
    assert any(row["verdict"] == "check_failed" for row in result.document["table"])


def test_a_value_that_does_not_compile_is_reported_and_the_others_are_compared(tmp_path: Path) -> None:
    broken = TreeBuild(TreeBuildStatus.COMPILE_ERROR, key="key-2", detail="error: too many warps")
    fakes = _Fakes(tmp_path, {("2", "4"): 0.8}, fail={"2": broken})

    result = _sweep(tmp_path, fakes)

    assert result.winners == {("Q8_0", "2"): "4"}
    arms = {arm["arm"]: arm for arm in result.document["arms"]}
    assert arms["value=2"]["status"] == "compile_error"
    assert arms["value=2"]["detail"] == "error: too many warps"
    assert arms["value=4"]["status"] == "timed"
    assert arms["value=4"]["rounds"] == 5


def test_a_value_whose_timing_fails_is_dropped_and_the_sweep_goes_on(tmp_path: Path) -> None:
    fakes = _Fakes(tmp_path, {("2", "4"): 0.8})
    timed = fakes.time

    def time(executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str]) -> dict[str, float]:
        if executable.parent.name == "bin-2":
            raise BackendOpsError("test-backend-ops exited 134: HIP error: invalid configuration")
        return timed(executable, cases, names, env)

    hooks = SweepHooks(build=fakes.build, hold=fakes.hold, time=time, check=fakes.check)
    result = run_sweep(_plan(), _source, tmp_path / "work", hooks)

    arms = {arm["arm"]: arm for arm in result.document["arms"]}
    assert arms["value=2"]["status"] == "bench_failed"
    assert "invalid configuration" in arms["value=2"]["detail"]
    assert result.winners == {("Q8_0", "2"): "4"}


def test_a_sweep_none_of_whose_values_is_timed_fails_with_each_value_s_error(tmp_path: Path) -> None:
    unbuilt = TreeBuild(TreeBuildStatus.COMPILE_ERROR, detail="blob 0a is not in the blob store")
    fakes = _Fakes(tmp_path, {}, fail={"2": unbuilt})
    timed = fakes.time

    def time(executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str]) -> dict[str, float]:
        if executable.parent.name == "bin-4":
            raise BackendOpsError("test-backend-ops exited 134")
        return timed(executable, cases, names, env)

    each = "value=2 compile_error: blob 0a is not in the blob store; value=4 bench_failed: test-backend-ops exited 134"
    with pytest.raises(SweepError, match=f"no value was timed: {each}"):
        hooks = SweepHooks(build=fakes.build, hold=fakes.hold, time=time, check=fakes.check)
        run_sweep(_plan(), _source, tmp_path / "work", hooks)


def test_the_sweep_fails_when_the_pinned_commit_does_not_build(tmp_path: Path) -> None:
    fakes = _Fakes(tmp_path, {}, fail={"": TreeBuild(TreeBuildStatus.INFRA, detail="no isolation")})

    with pytest.raises(SweepError, match="no isolation"):
        _sweep(tmp_path, fakes)
    assert fakes.holds == 0


@pytest.mark.parametrize(
    ("where", "values", "timing", "message"),
    [
        ({"warps": ["1"]}, [], SweepTiming([(1, 1)], 5, 4), "has the axes"),
        ({"ncols": ["9"]}, [], SweepTiming([(1, 1)], 5, 4), "holds"),
        ({}, ["3"], SweepTiming([(1, 1)], 5, 4), "takes the values"),
        ({}, [], SweepTiming([], 5, 4), "weight shapes"),
        ({}, [], SweepTiming([(1, 1)], 2, 4), "rounds"),
        ({}, [], SweepTiming([(1, 1)], 5, 0), "copies"),
    ],
)
def test_a_plan_the_knob_cannot_take_is_refused(where, values, timing: SweepTiming, message: str) -> None:
    with pytest.raises(KnobSpecError, match=message):
        plan_sweep("mmvq_nwarps_rdna3", where, values, timing)


def test_an_axis_the_plan_does_not_name_takes_all_of_its_values() -> None:
    plan = plan_sweep("mmvq_nwarps_rdna3", {"type": ["Q8_0"]}, [], SweepTiming([(5120, 5120)], 5, 4))

    assert plan.keys == tuple(("Q8_0", str(ncols)) for ncols in range(1, 9))
    assert plan.values == ("1", "2", "4", "8")
