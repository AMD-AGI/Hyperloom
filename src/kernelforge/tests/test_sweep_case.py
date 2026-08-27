"""Exploratory single-case sweeps: what they measure and what they refuse to."""

from __future__ import annotations

import asyncio
import pathlib
import sys
import types

import pytest

from kernelforge.loop import task_preparer
from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    EXPLORATORY_KIND,
    SELECTION_NARROWED,
    SELECTION_REJECTED,
    SELECTION_WHOLE_SUITE,
    SWEEP_CASE_FLAG,
    SWEEP_ENV_PREFIX,
    _CASE_FLAG_REJECTED,
    aggregate_benchmark_measurements,
    calculate_measurement_case_speedups,
    sweep_case,
)


@pytest.fixture(autouse=True)
def _forget_which_drivers_reject_the_flag():
    """The flag memo is process-wide; one test's driver must not answer another's."""
    _CASE_FLAG_REJECTED.clear()
    yield
    _CASE_FLAG_REJECTED.clear()


def _driver(tmp_path: pathlib.Path, body: str) -> str:
    path = tmp_path / "drv.py"
    path.write_text(body)
    return str(path)


def _sweep(driver: str, **kwargs) -> dict:
    kwargs.setdefault("case_id", "sq64")
    return asyncio.run(sweep_case(driver_script=driver, **kwargs))


# A driver that honours the flag: one case in, that case's lines out. It reads
# its one swept constant and echoes it, the way the prompt asks a source to.
_NARROWING_DRIVER = """
import argparse, os, sys
p = argparse.ArgumentParser()
p.add_argument("--bench-case", default="")
args, _ = p.parse_known_args()
cases = {"sq64": 0.5, "sq7211": 4.0}
if args.bench_case:
    cases = {args.bench_case: cases[args.bench_case]}
raw = os.environ.get("FORGE_SWEEP_BLOCK_H")
if raw is not None:
    print("sweep_const: BLOCK_H %s" % raw)
scale = float(raw or "16") / 16.0
for cid, ms in cases.items():
    print("wall_ms: %.6f" % (ms * scale))
    print("wall_ms: %.6f" % (ms * scale * 1.1))
    print("case_ms: %s %.6f" % (cid, ms * scale))
"""

# A driver written before the flag existed: parse_known_args swallows it.
_WHOLE_SUITE_DRIVER = """
print("case_ms: sq64 0.500000")
print("case_ms: sq7211 4.000000")
"""


def _counting_driver(tmp_path: pathlib.Path, body: str) -> tuple[str, pathlib.Path]:
    """A driver that tallies its own invocations, to price the flag retry."""
    tally = tmp_path / "runs.txt"
    preamble = f"""
import pathlib
_tally = pathlib.Path({str(tally)!r})
_tally.write_text(str(int(_tally.read_text()) + 1 if _tally.exists() else 1))
"""
    return _driver(tmp_path, preamble + body), tally


def _runs(tally: pathlib.Path) -> int:
    return int(tally.read_text())


# argparse with plain parse_args: an unknown flag is exit 2 before anything runs.
# This is the shape that produced 221 failures and zero measurements -- the
# driver measures perfectly well, it just will not be handed --bench-case.
_FLAG_REJECTING_DRIVER = """
import argparse
p = argparse.ArgumentParser()
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--bench-mode", action="store_true")
p.parse_args()
print("case_ms: sq64 0.500000")
print("case_ms: sq7211 4.000000")
"""


# A driver that honours the flag AND validates its argument: an undeclared case
# id is exit 2, which is byte-for-byte what argparse says about an unknown flag.
_CASE_CHECKING_DRIVER = """
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--bench-mode", action="store_true")
p.add_argument("--bench-case", default="")
args = p.parse_args()
cases = {"sq64": 0.5, "sq7211": 4.0}
if args.bench_case:
    if args.bench_case not in cases:
        print("unknown case %s" % args.bench_case, file=sys.stderr)
        sys.exit(2)
    cases = {args.bench_case: cases[args.bench_case]}
for cid, ms in cases.items():
    print("case_ms: %s %.6f" % (cid, ms))
"""


def test_narrowed_sweep_reports_one_case_as_exploratory(tmp_path):
    result = _sweep(_driver(tmp_path, _NARROWING_DRIVER))
    assert result["success"], result
    assert result["kind"] == EXPLORATORY_KIND
    assert result["case_id"] == "sq64"
    assert result["case_ms"] == pytest.approx(0.5)
    assert result["narrowed"] is True
    assert result["n_samples"] == 2
    assert result["wall_max_ms"] > result["wall_min_ms"]
    assert "EXPLORATORY, NOT AN ACCEPTANCE RESULT" in result["message"]


def test_constants_reach_the_driver_under_the_sweep_prefix(tmp_path):
    result = _sweep(_driver(tmp_path, _NARROWING_DRIVER), constants={"BLOCK_H": 32})
    assert result["success"], result
    assert result["case_ms"] == pytest.approx(1.0)
    assert result["constants"] == {"BLOCK_H": "32"}
    assert "BLOCK_H=32" in result["message"]


def test_sweep_result_carries_no_case_times(tmp_path):
    """The field every scoring path reads is the one a sweep never produces."""
    result = _sweep(_driver(tmp_path, _NARROWING_DRIVER))
    assert "case_times" not in result


def test_driver_that_ignores_the_flag_is_reported_not_hidden(tmp_path):
    result = _sweep(_driver(tmp_path, _WHOLE_SUITE_DRIVER))
    assert result["success"], result
    assert result["case_ms"] == pytest.approx(0.5)
    assert result["narrowed"] is False
    assert SWEEP_CASE_FLAG in result["message"]
    assert "whole suite" in result["message"]
    # Nothing to compare against a spread that covers other cases too.
    assert "wall_min_ms" not in result


# ---------- --bench-case is optional, in the contract and in practice ------


def test_a_driver_that_rejects_the_flag_is_retried_without_it(tmp_path):
    """The measurement exists; only the flag was refused, so ask again without it."""
    driver, tally = _counting_driver(tmp_path, _FLAG_REJECTING_DRIVER)
    result = _sweep(driver)
    assert result["success"], result
    assert result["case_ms"] == pytest.approx(0.5)
    assert result["case_selection"] == SELECTION_WHOLE_SUITE
    assert result["narrowed"] is False
    assert f"rejected {SWEEP_CASE_FLAG} (exit 2)" in result["message"]
    assert _runs(tally) == 2


def test_the_rejected_invocation_is_paid_once_per_driver(tmp_path):
    """221 probes over three campaigns paid it 221 times; a campaign pays it once."""
    driver, tally = _counting_driver(tmp_path, _FLAG_REJECTING_DRIVER)
    assert _sweep(driver)["success"]
    assert _runs(tally) == 2

    again = _sweep(driver, case_id="sq7211")
    assert again["success"], again
    assert again["case_selection"] == SELECTION_WHOLE_SUITE
    assert f"known to reject {SWEEP_CASE_FLAG}" in again["message"]
    assert _runs(tally) == 3


def test_a_whole_suite_spread_is_not_offered_as_this_case_s(tmp_path):
    """Its wall_ms lines timed every case the driver ran, not the one asked for."""
    driver = _driver(
        tmp_path,
        """
print("wall_ms: 0.400000")
print("wall_ms: 4.100000")
print("case_ms: sq64 0.500000")
print("case_ms: sq7211 4.000000")
""",
    )
    result = _sweep(driver)
    assert result["success"], result
    assert result["case_selection"] == SELECTION_WHOLE_SUITE
    assert "wall_min_ms" not in result
    assert "whole suite rather than this case" in result["message"]


def test_a_driver_that_honours_the_flag_is_untouched_by_the_retry(tmp_path):
    driver, tally = _counting_driver(tmp_path, _NARROWING_DRIVER)
    result = _sweep(driver)
    assert result["success"], result
    assert result["case_selection"] == SELECTION_NARROWED
    assert result["narrowed"] is True
    assert _runs(tally) == 1
    assert not _CASE_FLAG_REJECTED


def test_a_driver_that_fails_either_way_reports_no_time(tmp_path):
    """Failing without the flag too means the configuration broke, not the flag."""
    driver, tally = _counting_driver(
        tmp_path,
        """
import sys
print("triton compile error: out of LDS", file=sys.stderr)
sys.exit(1)
""",
    )
    result = _sweep(driver)
    assert result["success"] is False
    assert result["case_selection"] == SELECTION_REJECTED
    assert "case_ms" not in result
    assert "wall_min_ms" not in result
    assert "median_ms" not in result
    assert f"also exit 1 with {SWEEP_CASE_FLAG}" in result["message"]
    assert _runs(tally) == 2
    # Nothing was learned about the argument parser, so the next probe of a
    # working configuration still gets its case narrowed.
    assert not _CASE_FLAG_REJECTED


def test_a_bad_case_id_is_not_recorded_as_a_rejection_of_the_flag(tmp_path):
    """This driver KNOWS --bench-case; it refused the case id, and it says so
    the same way an unknown flag does -- non-zero, then fine without it. Reading
    that as a broken parser would cost every later probe of a valid case a whole
    suite and its spread, permanently, for a driver that was working."""
    driver, tally = _counting_driver(tmp_path, _CASE_CHECKING_DRIVER)
    missing = _sweep(driver, case_id="sq99")
    assert missing["success"] is False
    assert "NO TIMING FOR CASE 'sq99'" in missing["message"]
    # The whole suite the retry ran is the list of cases this driver declares,
    # and sq99 is not on it, so the flag is not what was refused.
    assert "the case id and not the flag" in missing["message"]
    assert _runs(tally) == 2
    assert not _CASE_FLAG_REJECTED

    good = _sweep(driver, case_id="sq7211")
    assert good["success"], good
    assert good["case_selection"] == SELECTION_NARROWED
    assert good["narrowed"] is True
    assert good["case_ms"] == pytest.approx(4.0)
    assert _runs(tally) == 3


def test_a_declared_case_the_flag_still_refused_does_memoise(tmp_path):
    """The other side of the same evidence: the case came back in the suite, so
    the argument was satisfiable and the parser is what would not take it."""
    driver, tally = _counting_driver(tmp_path, _FLAG_REJECTING_DRIVER)
    assert _sweep(driver)["success"]
    assert list(_CASE_FLAG_REJECTED.values()) == [True]
    assert _runs(tally) == 2


def test_a_timeout_is_not_a_flag_rejection_and_is_not_retried(tmp_path):
    """A retry would spend the probe's whole budget a second time over."""
    driver, tally = _counting_driver(
        tmp_path,
        """
import time
time.sleep(30)
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, timeout_sec=1)
    assert result["success"] is False
    assert "TIMEOUT after 1s" in result["message"]
    assert "case_selection" not in result
    assert _runs(tally) == 1


def test_sweep_runs_outside_the_tree_it_measures(tmp_path):
    driver = _driver(
        tmp_path,
        """
import pathlib
pathlib.Path("side_effect.txt").write_text("x")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver)
    assert result["success"], result
    assert not (tmp_path / "side_effect.txt").exists()
    assert not (pathlib.Path.cwd() / "side_effect.txt").exists()


# ---------- failure paths: a broken point never looks like a slow one -------


def test_unrunnable_configuration_reports_no_time(tmp_path):
    driver = _driver(
        tmp_path,
        """
import sys
print("triton compile error: out of LDS", file=sys.stderr)
sys.exit(1)
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": 512})
    assert result["success"] is False
    assert result["kind"] == EXPLORATORY_KIND
    assert "CONFIGURATION DID NOT RUN (exit 1)" in result["message"]
    assert "BLOCK_H=512" in result["message"]
    assert "case_ms" not in result
    assert "out of LDS" in result["output"]


def test_missing_case_is_a_failure_and_names_what_came_back(tmp_path):
    result = _sweep(_driver(tmp_path, "print('case_ms: sq7211 4.000000')"))
    assert result["success"] is False
    assert "NO TIMING FOR CASE 'sq64'" in result["message"]
    assert "sq7211" in result["message"]
    assert "case_ms" not in result


def test_duplicate_case_timing_is_a_failure(tmp_path):
    driver = _driver(
        tmp_path,
        """
print("case_ms: sq64 0.500000")
print("case_ms: sq64 9.000000")
""",
    )
    result = _sweep(driver)
    assert result["success"] is False
    assert "MORE THAN ONCE" in result["message"]
    assert "case_ms" not in result


def test_nonpositive_case_timing_is_a_failure(tmp_path):
    result = _sweep(_driver(tmp_path, "print('case_ms: sq64 0.000000')"))
    assert result["success"] is False
    assert "UNUSABLE TIME" in result["message"]
    assert "case_ms" not in result


def test_timeout_reports_no_time(tmp_path):
    driver = _driver(
        tmp_path,
        """
import time
time.sleep(30)
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, timeout_sec=1)
    assert result["success"] is False
    assert "TIMEOUT after 1s" in result["message"]
    assert "case_ms" not in result


@pytest.mark.parametrize(
    "constants",
    [
        {"path": 1},  # not an upper-case identifier
        {"BLOCK_H": "16; rm -rf /"},
        {"BLOCK_H": "16 32"},
    ],
)
def test_unusable_constants_are_rejected_before_the_driver_runs(tmp_path, constants):
    marker = tmp_path / "ran.txt"
    driver = _driver(
        tmp_path,
        f"""
import pathlib
pathlib.Path({str(marker)!r}).write_text("x")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants=constants)
    assert result["success"] is False
    assert "case_ms" not in result
    assert not marker.exists()


def test_a_constant_cannot_reach_the_variable_it_is_named_after(tmp_path):
    seen = tmp_path / "seen.txt"
    driver = _driver(
        tmp_path,
        f"""
import os, pathlib
pathlib.Path({str(seen)!r}).write_text(repr((
    os.environ.get("LD_PRELOAD", ""),
    os.environ.get("{SWEEP_ENV_PREFIX}LD_PRELOAD", ""),
)))
print("sweep_const: LD_PRELOAD evil.so")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"LD_PRELOAD": "evil.so"})
    assert result["success"], result
    assert seen.read_text() == repr(("", "evil.so"))


def test_invalid_case_id_is_rejected(tmp_path):
    result = _sweep(_driver(tmp_path, "print('case_ms: sq64 0.5')"), case_id="  ")
    assert result["success"] is False
    assert "INVALID CASE ID" in result["message"]


def test_a_constant_nothing_read_is_a_failure_not_a_null_result(tmp_path):
    """The default timing of a knob nobody consumed is not a measurement of it."""
    driver = _driver(
        tmp_path,
        """
print("wall_ms: 0.500000")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": 32})
    assert result["success"] is False
    assert result["kind"] == EXPLORATORY_KIND
    assert "NOTHING READ BLOCK_H" in result["message"]
    assert "default configuration" in result["message"]
    assert "case_ms" not in result


def test_only_the_unread_constant_is_named(tmp_path):
    driver = _driver(
        tmp_path,
        """
print("sweep_const: BLOCK_H 32")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": 32, "NUM_WARPS": 4})
    assert result["success"] is False
    assert "NOTHING READ NUM_WARPS" in result["message"]
    assert "BLOCK_H," not in result["message"].split("NOTHING READ")[1]


def test_a_constant_read_at_a_different_value_is_a_failure(tmp_path):
    """A source that clamps the value swept a configuration nobody asked for."""
    driver = _driver(
        tmp_path,
        """
print("sweep_const: BLOCK_H 64")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": 512})
    assert result["success"] is False
    assert "READ A DIFFERENT CONFIGURATION" in result["message"]
    assert "asked 512, read 64" in result["message"]
    assert "case_ms" not in result


# ---------- knobs the source named first ------------------------------------


# The knob one competing agent flipped to win a benchmark: read by the source
# under its own name, and no more likely to print forge's echo than any other
# third-party constant.
_THIRD_PARTY_KNOB = "GPTOSS_SWIGLU_MXFP4_BF16_BOUND"


def _knob_driver(tmp_path: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Read the knob under both names, report which one arrived, echo forge's."""
    seen = tmp_path / "seen.txt"
    driver = _driver(
        tmp_path,
        f"""
import os, pathlib
verbatim = os.environ.get({_THIRD_PARTY_KNOB!r}, "")
prefixed = os.environ.get({SWEEP_ENV_PREFIX + _THIRD_PARTY_KNOB!r}, "")
pathlib.Path({str(seen)!r}).write_text(repr((verbatim, prefixed)))
if prefixed:
    print("sweep_const: {_THIRD_PARTY_KNOB} %s" % prefixed)
print("case_ms: sq64 0.500000")
""",
    )
    return driver, seen


def test_verbatim_names_reach_a_knob_the_source_already_reads(tmp_path):
    driver, seen = _knob_driver(tmp_path)
    result = _sweep(driver, constants={_THIRD_PARTY_KNOB: 512}, prefix_constants=False)
    assert result["success"], result
    assert seen.read_text() == repr(("512", ""))
    assert result["case_ms"] == pytest.approx(0.5)


def test_by_default_a_constant_still_reaches_only_the_sweep_namespace(tmp_path):
    driver, seen = _knob_driver(tmp_path)
    result = _sweep(driver, constants={_THIRD_PARTY_KNOB: 512})
    assert result["success"], result
    assert seen.read_text() == repr(("", "512"))
    assert result["override_consumption"] == {_THIRD_PARTY_KNOB: "consumed"}


def test_a_verbatim_knob_that_echoes_nothing_is_reported_not_refused(tmp_path):
    """A third-party knob prints no sweep_const line; refusing measures nothing."""
    driver = _driver(
        tmp_path,
        """
print("wall_ms: 0.500000")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={_THIRD_PARTY_KNOB: 512}, prefix_constants=False)
    assert result["success"], result
    assert result["case_ms"] == pytest.approx(0.5)
    assert result["override_consumption"] == {_THIRD_PARTY_KNOB: "unread"}
    assert "UNCONFIRMED" in result["message"]
    assert "no-override reference measured in the same round" in result["message"]


def test_only_the_unechoed_verbatim_knob_is_marked_unread(tmp_path):
    driver = _driver(
        tmp_path,
        """
print("sweep_const: BLOCK_H 32")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(
        driver,
        constants={"BLOCK_H": 32, _THIRD_PARTY_KNOB: 512},
        prefix_constants=False,
    )
    assert result["success"], result
    assert result["override_consumption"] == {
        "BLOCK_H": "consumed",
        _THIRD_PARTY_KNOB: "unread",
    }
    assert _THIRD_PARTY_KNOB in result["message"].split("UNCONFIRMED")[1]
    assert "BLOCK_H," not in result["message"].split("UNCONFIRMED")[1]


def test_a_verbatim_knob_read_at_a_different_value_is_still_a_failure(tmp_path):
    """Silence is unconfirmed; a wrong echo is a configuration nobody asked for."""
    driver = _driver(
        tmp_path,
        """
print("sweep_const: BLOCK_H 64")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": 512}, prefix_constants=False)
    assert result["success"] is False
    assert "READ A DIFFERENT CONFIGURATION" in result["message"]
    assert "case_ms" not in result


def test_a_verbatim_name_is_still_checked_before_the_driver_runs(tmp_path):
    """No prefix to hide behind: the name goes straight into the child's env."""
    marker = tmp_path / "ran.txt"
    driver = _driver(
        tmp_path,
        f"""
import pathlib
pathlib.Path({str(marker)!r}).write_text("x")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"BLOCK_H": "16 32"}, prefix_constants=False)
    assert result["success"] is False
    assert "case_ms" not in result
    assert not marker.exists()


@pytest.mark.parametrize(
    "name",
    [
        "PATH",  # the interpreter that starts the driver
        "HIP_VISIBLE_DEVICES",  # spellable, and it would time another lane's GPU
        "LD_PRELOAD",
        "PYTHONPATH",
        "AITER_JIT_DIR",  # the cache isolation the number is attributed by
        "FORGE_NPROC_PER_NODE",
    ],
)
def test_a_verbatim_sweep_cannot_reach_what_the_measurement_runs_on(tmp_path, name):
    """The prefix guaranteed this by construction; verbatim mode by name."""
    marker = tmp_path / "ran.txt"
    driver = _driver(
        tmp_path,
        f"""
import pathlib
pathlib.Path({str(marker)!r}).write_text("x")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={name: "1"}, prefix_constants=False)
    assert result["success"] is False
    assert name in result["message"]
    assert "case_ms" not in result
    assert not marker.exists()


@pytest.mark.parametrize(
    "name",
    [
        "HOME",  # moves ~/.triton/cache and every dotfile cache
        "XDG_CACHE_HOME",  # the same, for anything honouring the spec
        "TRITON_HOME",  # ~/.triton relocated by name instead of by HOME
        "TORCH_EXTENSIONS_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
        "CC",  # a different compiler is a different binary
        "CXX",
        "CXXFLAGS",  # and so is the same compiler at -O0
        "HIPCC_COMPILE_FLAGS_APPEND",
    ],
)
def test_a_verbatim_sweep_cannot_move_the_cache_or_change_the_compiler(tmp_path, name):
    """Same class as the device and toolchain names already refused: each of
    these makes the probe compile, or compile against, something other than
    what the gate will read, so the number would not describe this source."""
    marker = tmp_path / "ran.txt"
    driver = _driver(
        tmp_path,
        f"""
import pathlib
pathlib.Path({str(marker)!r}).write_text("x")
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={name: "1"}, prefix_constants=False)
    assert result["success"] is False
    assert name in result["message"]
    assert "case_ms" not in result
    assert not marker.exists()


@pytest.mark.parametrize("name", ["HSA_XNACK", "AMD_SERIALIZE_KERNEL", "TRITON_DEBUG"])
def test_the_open_tuning_families_stay_sweepable_verbatim(tmp_path, name):
    """The reserved list must not swallow the knobs a sweep exists to vary: a
    runtime tuning variable changes how the source runs, which is the question,
    not what the source is."""
    driver = _driver(
        tmp_path,
        f"""
import os
print("sweep_const: {name} %s" % os.environ[{name!r}])
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={name: "1"}, prefix_constants=False)
    assert result["success"], result
    assert result["override_consumption"] == {name: "consumed"}


def test_the_reserved_names_are_reserved_only_verbatim(tmp_path):
    """Under the prefix they collide with nothing, so nothing needs refusing."""
    driver = _driver(
        tmp_path,
        """
import os
print("sweep_const: PATH %s" % os.environ["FORGE_SWEEP_PATH"])
print("case_ms: sq64 0.500000")
""",
    )
    result = _sweep(driver, constants={"PATH": "1"})
    assert result["success"], result


def test_an_inherited_sweep_variable_is_not_reported_as_this_sweep_s(tmp_path, monkeypatch):
    """A FORGE_SWEEP_* left in this process is not a constant this point set."""
    monkeypatch.setenv(SWEEP_ENV_PREFIX + "STALE", "99")
    result = _sweep(_driver(tmp_path, _NARROWING_DRIVER), constants={"BLOCK_H": 32})
    assert result["success"], result
    assert result["constants"] == {"BLOCK_H": "32"}
    assert result["override_consumption"] == {"BLOCK_H": "consumed"}


def test_a_one_case_suite_reached_without_the_flag_is_not_called_narrowed(
    tmp_path,
):
    """The flag was never accepted, so nothing here says the driver honoured it."""
    driver = _driver(
        tmp_path, _FLAG_REJECTING_DRIVER.replace('print("case_ms: sq7211 4.000000")', 'print("wall_ms: 0.500000")')
    )
    result = _sweep(driver)
    assert result["success"], result
    assert result["case_selection"] == SELECTION_WHOLE_SUITE
    # Only this case was timed, so its wall_ms lines really are its own spread.
    assert result["narrowed"] is True
    assert result["wall_min_ms"] == pytest.approx(0.5)
    assert f"rejected {SWEEP_CASE_FLAG}" in result["message"]


def test_a_point_with_no_measurable_spread_says_so(tmp_path):
    result = _sweep(_driver(tmp_path, "print('case_ms: sq64 0.500000')"))
    assert result["success"], result
    assert result["narrowed"] is True
    assert "wall_min_ms" not in result
    assert "no measured spread" in result["message"]


# ---------- the acceptance path refuses exploratory measurements -----------


def _exploratory_measurement() -> dict:
    return {
        "success": True,
        "kind": EXPLORATORY_KIND,
        "case_ms": 0.5,
        "case_times": {"sq64": 0.5},
        "median_ms": 0.5,
    }


def test_aggregation_refuses_an_exploratory_measurement():
    """Even one carrying case_times -- the marker decides, not the shape."""
    aggregate = aggregate_benchmark_measurements([_exploratory_measurement()])
    assert aggregate["success"] is False
    assert "EXPLORATORY SWEEP" in aggregate["message"]


def test_scoring_refuses_an_exploratory_measurement():
    benchmark = {"success": True, "measurements": [_exploratory_measurement()]}
    with pytest.raises(CaseCoverageError, match="exploratory sweep"):
        calculate_measurement_case_speedups(benchmark, {"sq64": 1.0}, expected_measurements=1)


# ---------- the contract the primitive asks drivers to satisfy -------------


def _reference_template_main(monkeypatch, argv: list[str]):
    """Execute the reference driver template's ``main`` off the device."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
    torch.manual_seed = lambda *_args: None
    harness = types.ModuleType("graph_harness")
    harness.cuda_graph_bench = lambda *_a, **_k: {"times_ms": [0.5]}
    kernel = types.ModuleType("your_kernel_module")
    kernel.your_entry_point = lambda *_a, **_k: None
    for name, module in (("torch", torch), ("graph_harness", harness), ("your_kernel_module", kernel)):
        monkeypatch.setitem(sys.modules, name, module)

    namespace: dict = {"__file__": "driver_template.py"}
    exec(compile(task_preparer.REFERENCE_DRIVER_TEMPLATE, "driver_template.py", "exec"), namespace)
    namespace["CASES"].update({"sq64": {"M": 8, "N": 8}, "sq7211": {"M": 8, "N": 9}})
    benched: list[str] = []
    namespace["_run_bench"] = lambda _d, case_id, *_a: benched.append(case_id)
    monkeypatch.setattr(sys, "argv", ["driver_template.py", *argv])
    return namespace["main"](), benched


def test_reference_template_rejects_an_undeclared_sweep_case(monkeypatch, capsys):
    status, benched = _reference_template_main(monkeypatch, ["--bench-mode", SWEEP_CASE_FLAG, "sq999"])
    assert status == 1
    assert benched == []
    assert "unknown case sq999" in capsys.readouterr().out


def test_reference_template_benchmarks_only_the_requested_case(monkeypatch):
    status, benched = _reference_template_main(monkeypatch, ["--bench-mode", SWEEP_CASE_FLAG, "sq7211"])
    assert status == 0
    assert benched == ["sq7211"]
