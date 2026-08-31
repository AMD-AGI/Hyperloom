"""Hermetic tests for the dual-path measurement driver contract preflight.

Every case here runs a real driver subprocess written by the test, so what is
verified is what a task author's driver would actually be judged on — no GPU,
no LLM, no mocking of the contract's own parsing.
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

from kernelforge.rewrite_by_flydsl import driver_contract
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

# A driver that satisfies the whole contract: it distinguishes both bench modes
# and only reaches the candidate when the ported kernel is importable.
_CONFORMING_DRIVER = """\
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--bench-mode", action="store_true")
parser.add_argument("--ref-bench-mode", action="store_true")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=30)
args, _unknown = parser.parse_known_args()

CASES = ["M4096_N1024_f32", "M8192_N1024_f32"]

if args.ref_bench_mode:
    for index, case in enumerate(CASES):
        print(f"case_ms: {case} {2.0 + index:.6f}")
    print("median_ms: 2.000000")
elif args.bench_mode:
    import kernel
    kernel.build_softmax_module(1, 1, "f32")
    for index, case in enumerate(CASES):
        print(f"case_ms: {case} {1.0 + index:.6f}")
    print("median_ms: 1.000000")
else:
    print("SNR: 61.50 dB")
    print("allclose: True")
"""


def _spec(tmp_path, *, driver_body=_CONFORMING_DRIVER, candidate="") -> tuple:
    source = tmp_path / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        candidate or "def build_softmax_module(*args):\n    raise NotImplementedError('not ported yet')\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(driver_body)
    spec = RewriteSpec(
        op_name="softmax",
        source_kernel=str(source),
        target_functions=["softmax"],
        flydsl_kernel=str(kernel),
        workspace=str(tmp_path),
    )
    return spec, str(driver)


_WORKING_CANDIDATE = "def build_softmax_module(*args):\n    return lambda *a, **k: None\n"


# ── output parsing ───────────────────────────────────────────────────────────


def test_the_canonical_timing_key_wins_over_the_deprecated_one():
    reading = driver_contract.read_driver_output("mean_ms: 9.0\ncase_ms: c0 1.0\nmedian_ms: 4.0\n")

    assert reading.timing_ms == 4.0
    assert reading.timing_metric == "median_ms"
    assert reading.case_ids == ("c0",)


def test_the_deprecated_timing_key_is_still_read():
    reading = driver_contract.read_driver_output("mean_ms: 7.5\n")

    assert reading.timing_ms == 7.5
    assert reading.timing_metric == "mean_ms"


def test_case_ids_come_from_both_reporting_conventions():
    reading = driver_contract.read_driver_output(
        "# case shape_a: relerr=0.01 ok=True\ncase_ms: shape_b 1.0\ncase_ms: shape_b 1.0\n"
    )

    assert reading.case_ids == ("shape_b", "shape_a")


def test_correctness_verdicts_are_read_from_either_metric():
    assert driver_contract.read_driver_output("SNR: 45.2 dB").snr_db == 45.2
    assert driver_contract.read_driver_output("allclose: True").allclose is True
    assert driver_contract.read_driver_output("allclose: False").allclose is False
    assert driver_contract.read_driver_output("nothing").has_correctness_verdict is False


# ── driver independence ──────────────────────────────────────────────────────


def test_a_missing_driver_is_named_as_such(tmp_path):
    spec, _driver = _spec(tmp_path)
    report = driver_contract.check_driver_independence(spec, str(tmp_path / "gone.py"))

    assert report.failure_class == driver_contract.DRIVER_MISSING


def test_a_driver_that_is_the_kernel_it_measures_is_rejected(tmp_path):
    spec, _driver = _spec(tmp_path)
    report = driver_contract.check_driver_independence(spec, spec.flydsl_kernel)

    assert report.failure_class == driver_contract.DRIVER_NOT_INDEPENDENT


def test_a_generated_forge_artifact_cannot_own_the_gate(tmp_path):
    spec, _driver = _spec(tmp_path)
    artifact = tmp_path / "forge_experiments" / "driver.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("print('drive')\n")
    report = driver_contract.check_driver_independence(spec, str(artifact))

    assert report.failure_class == driver_contract.DRIVER_NOT_INDEPENDENT


def test_a_candidate_that_would_overwrite_the_source_is_rejected(tmp_path):
    spec, driver = _spec(tmp_path)
    spec.flydsl_kernel = spec.source_kernel
    report = driver_contract.check_driver_independence(spec, driver)

    assert report.failure_class == driver_contract.SOURCE_CANDIDATE_COLLISION


def test_an_independent_driver_passes(tmp_path):
    spec, driver = _spec(tmp_path)

    assert driver_contract.check_driver_independence(spec, driver).ok is True


def test_a_module_shadowing_the_candidate_is_rejected(tmp_path):
    # The candidate moved into its attempt directory, but a kernel left at the
    # workspace root by an earlier run would still win the import.
    spec, driver = _spec(tmp_path)
    attempt = tmp_path / ".forge_rewrite" / "20260101-000000-abcdef12"
    attempt.mkdir(parents=True)
    (attempt / "kernel.py").write_text("def build_softmax_module(*a):\n    pass\n")
    spec.flydsl_kernel = str(attempt / "kernel.py")

    report = driver_contract.check_driver_independence(spec, driver)

    assert report.ok is False
    assert report.failure_class == driver_contract.CANDIDATE_SHADOWED
    assert "kernel.py" in report.detail


def test_a_candidate_alone_in_its_attempt_directory_passes(tmp_path):
    spec, driver = _spec(tmp_path)
    (tmp_path / "kernel.py").unlink()
    attempt = tmp_path / ".forge_rewrite" / "20260101-000000-abcdef12"
    attempt.mkdir(parents=True)
    (attempt / "kernel.py").write_text("def build_softmax_module(*a):\n    pass\n")
    spec.flydsl_kernel = str(attempt / "kernel.py")

    assert driver_contract.check_driver_independence(spec, driver).ok is True


# ── reference preflight ──────────────────────────────────────────────────────


def test_the_reference_preflight_returns_the_baseline_and_its_cases(tmp_path):
    spec, driver = _spec(tmp_path)
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.ok is True
    assert report.timing_ms == 2.0
    assert report.timing_metric == "median_ms"
    assert report.case_ids == ("M4096_N1024_f32", "M8192_N1024_f32")
    assert report.warnings == []


def test_a_driver_without_ref_bench_mode_is_rejected_before_porting(tmp_path):
    # A plain forge-loop driver: it ignores the flag and runs correctness.
    spec, driver = _spec(tmp_path, driver_body="print('SNR: 61.5 dB')\n")
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.ok is False
    assert report.failure_class == driver_contract.REF_MODE_UNSUPPORTED
    assert "correctness path" in report.detail


def test_a_driver_that_refuses_the_ref_flag_is_rejected(tmp_path):
    spec, driver = _spec(
        tmp_path,
        driver_body=("import argparse\nargparse.ArgumentParser().parse_args()\n"),
    )
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.failure_class == driver_contract.REF_MODE_UNSUPPORTED


def test_an_unparseable_reference_timing_is_named(tmp_path):
    spec, driver = _spec(tmp_path, driver_body="print('elapsed: quite fast')\n")
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.failure_class == driver_contract.REF_TIMING_UNPARSEABLE


def test_a_reference_mode_crash_is_named_with_its_exit_code(tmp_path):
    spec, driver = _spec(
        tmp_path,
        driver_body="import sys\nprint('boom')\nsys.exit(3)\n",
    )
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.failure_class == driver_contract.REF_MODE_FAILED
    assert "exit 3" in report.detail
    assert "boom" in report.detail


def test_a_hanging_driver_is_stopped_and_named(tmp_path):
    spec, driver = _spec(tmp_path, driver_body="import time\ntime.sleep(30)\n")
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=1)

    assert report.failure_class == driver_contract.REF_MODE_TIMEOUT


def test_the_deprecated_timing_key_is_accepted_with_a_warning(tmp_path):
    spec, driver = _spec(tmp_path, driver_body="print('mean_ms: 3.0')\n")
    report = driver_contract.preflight_reference(spec, driver, timeout_sec=60)

    assert report.ok is True
    assert report.timing_ms == 3.0
    assert "median_ms" in report.warnings[0]


# ── candidate probe before porting ───────────────────────────────────────────


def test_the_candidate_probe_accepts_a_driver_that_cannot_run_the_stub(tmp_path):
    spec, driver = _spec(tmp_path)
    report = driver_contract.probe_candidate_arguments(spec, driver, timeout_sec=60)

    assert report.ok is True


def test_a_driver_that_never_reaches_the_candidate_is_caught(tmp_path):
    # Times the source in both directions: bench mode never imports the kernel,
    # so it reports a timing even though nothing has been ported.
    spec, driver = _spec(
        tmp_path,
        driver_body=(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--bench-mode', action='store_true')\n"
            "parser.add_argument('--ref-bench-mode', action='store_true')\n"
            "parser.add_argument('--warmup', type=int, default=10)\n"
            "parser.add_argument('--iters', type=int, default=30)\n"
            "parser.parse_known_args()\n"
            "print('median_ms: 2.0')\n"
        ),
    )
    report = driver_contract.probe_candidate_arguments(spec, driver, timeout_sec=60)

    assert report.ok is False
    assert report.failure_class == driver_contract.CANDIDATE_NOT_ISOLATED
    assert "skeleton" in report.detail


def test_a_driver_without_bench_mode_is_caught_by_the_probe(tmp_path):
    spec, driver = _spec(
        tmp_path,
        driver_body=(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--ref-bench-mode', action='store_true')\n"
            "parser.add_argument('--warmup', type=int, default=10)\n"
            "parser.add_argument('--iters', type=int, default=30)\n"
            "parser.parse_args()\n"
        ),
    )
    report = driver_contract.probe_candidate_arguments(spec, driver, timeout_sec=60)

    assert report.failure_class == driver_contract.CANDIDATE_MODE_UNSUPPORTED


# ── candidate preflight after porting ────────────────────────────────────────


def test_the_candidate_preflight_accepts_matching_case_coverage(tmp_path):
    spec, driver = _spec(tmp_path, candidate=_WORKING_CANDIDATE)
    report = driver_contract.preflight_candidate(
        spec,
        driver,
        reference_case_ids=("M4096_N1024_f32", "M8192_N1024_f32"),
        timeout_sec=60,
    )

    assert report.ok is True
    assert report.timing_ms == 1.0
    assert report.case_ids == ("M4096_N1024_f32", "M8192_N1024_f32")


def test_a_candidate_that_skips_a_reference_case_is_rejected(tmp_path):
    spec, driver = _spec(tmp_path, candidate=_WORKING_CANDIDATE)
    report = driver_contract.preflight_candidate(
        spec,
        driver,
        reference_case_ids=("M4096_N1024_f32", "M8192_N1024_f32", "M16384_N1024_f32"),
        timeout_sec=60,
    )

    assert report.ok is False
    assert report.failure_class == driver_contract.CASE_COVERAGE_MISMATCH
    assert "M16384_N1024_f32" in report.detail


def test_a_candidate_that_benchmarks_an_unexpected_case_is_rejected(tmp_path):
    spec, driver = _spec(tmp_path, candidate=_WORKING_CANDIDATE)
    report = driver_contract.preflight_candidate(
        spec,
        driver,
        reference_case_ids=("M4096_N1024_f32",),
        timeout_sec=60,
    )

    assert report.failure_class == driver_contract.CASE_COVERAGE_MISMATCH
    assert "M8192_N1024_f32" in report.detail


def test_a_candidate_bench_crash_is_named(tmp_path):
    spec, driver = _spec(tmp_path)
    report = driver_contract.preflight_candidate(spec, driver, timeout_sec=60)

    assert report.failure_class == driver_contract.CANDIDATE_MODE_FAILED
    assert "NotImplementedError" in report.detail


def test_coverage_is_not_enforced_when_the_reference_reports_no_cases():
    assert driver_contract.check_case_coverage((), ("a",)).ok is True
    assert driver_contract.check_case_coverage(("a",), ("a",)).ok is True


def test_a_candidate_that_reports_no_cases_fails_coverage():
    # The reference named a case the candidate never accounted for. Passing this
    # would let the aggregate timing of a smaller workload be published as a
    # speedup, and a driver that simply never prints the per-case metric on its
    # candidate path is the likeliest way to get here.
    report = driver_contract.check_case_coverage(("a",), ())

    assert report.ok is False
    assert report.failure_class == driver_contract.CASE_COVERAGE_MISMATCH
    assert "'a'" in report.detail


# ── producer-owned environment ───────────────────────────────────────────────


def test_the_driver_receives_the_producer_owned_environment(tmp_path):
    spec, driver = _spec(
        tmp_path,
        driver_body=textwrap.dedent(
            """\
            import os
            print("median_ms: 1.0")
            print("logical:", os.environ["KERNELFORGE_REWRITE_LOGICAL_OP"])
            print("symbol:", os.environ["KERNELFORGE_REWRITE_BUILDER_SYMBOL"])
            print("candidate:", os.environ["KERNELFORGE_REWRITE_CANDIDATE_KERNEL"])
            """
        ),
    )
    run = driver_contract.run_driver(spec, driver, [], timeout_sec=60)

    assert run.ok is True
    assert "logical: softmax" in run.output
    assert "symbol: build_softmax_module" in run.output
    assert f"candidate: {spec.flydsl_kernel}" in run.output


def test_the_producer_environment_reaches_drivers_forge_does_not_launch(
    tmp_path,
    monkeypatch,
):
    # The correctness suite and the nested loop spawn the driver with the
    # ambient environment, so the contract has to be exported to it.
    monkeypatch.delenv("KERNELFORGE_REWRITE_LOGICAL_OP", raising=False)
    monkeypatch.delenv("KERNELFORGE_REWRITE_BUILDER_SYMBOL", raising=False)
    monkeypatch.delenv("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", raising=False)
    monkeypatch.delenv("KERNELFORGE_REWRITE_SOURCE_KERNEL", raising=False)
    spec, _driver = _spec(tmp_path)
    spec.op_name = "vllm::softmax"

    driver_contract.export_driver_environment(spec)

    assert os.environ["KERNELFORGE_REWRITE_LOGICAL_OP"] == "vllm::softmax"
    assert os.environ["KERNELFORGE_REWRITE_BUILDER_SYMBOL"] == spec.builder_symbol
    assert os.environ["KERNELFORGE_REWRITE_CANDIDATE_KERNEL"] == spec.flydsl_kernel
    assert os.environ["KERNELFORGE_REWRITE_SOURCE_KERNEL"] == spec.source_kernel


def test_the_driver_owns_case_selection(tmp_path):
    # Forge passes a mode and a sample count, never which shapes to run.
    spec, driver = _spec(
        tmp_path,
        driver_body="import sys\nprint('argv:', ' '.join(sys.argv[1:]))\n",
    )
    run = driver_contract.run_driver(spec, driver, ["--bench-mode"], warmup=3, iters=7, timeout_sec=60)

    assert "argv: --bench-mode --warmup 3 --iters 7" in run.output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_timed_out_driver_leaves_no_survivor(tmp_path):
    spec, driver = _spec(
        tmp_path,
        driver_body="import time\nprint('start', flush=True)\ntime.sleep(60)\n",
    )
    run = driver_contract.run_driver(spec, driver, [], timeout_sec=1)

    assert run.timed_out is True
    assert run.ok is False
