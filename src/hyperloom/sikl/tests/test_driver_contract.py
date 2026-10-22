# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The driver's output is read by KernelForge's own parsers, so assert with those: a line they skip
loses its case silently."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.sikl import workspace
from hyperloom.sikl.schema import find, load
from hyperloom.sikl.tests.conftest import CASES, SCHEMA_NAME
from kernelforge.mcp_server.tools.bench import parse_case_timings

STARTING_KERNEL = '"""Start."""\n\nimport torch\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    torch.matmul(a, b.t(), out=out)\n'
CANDIDATE_KERNEL = STARTING_KERNEL + "\n# edited: torch.matmul still passes\n"
FAILING_KERNEL = '"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n'


@pytest.fixture
def run(sikl_root: Path, tmp_path: Path):
    schema = load(find(sikl_root, SCHEMA_NAME))
    return workspace.materialize(
        schema, tmp_path / "run", sikl_root=sikl_root, language="flydsl", starting_kernel=STARTING_KERNEL
    )


def _run_driver(run, *args: str, expect_rc: int = 0) -> str:
    completed = subprocess.run(
        [sys.executable, str(run.driver), *args], cwd=run.workspace, capture_output=True, text=True
    )
    assert completed.returncode == expect_rc, completed.stdout + completed.stderr
    return completed.stdout


def _import_driver(run):
    spec = importlib.util.spec_from_file_location("sikl_forge_driver", run.driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_calls(sikl_root: Path) -> int:
    calls = sikl_root / "validate_calls"
    return int(calls.read_text()) if calls.exists() else 0


def test_bench_output_parses_into_every_case(run):
    """`parse_case_timings` must find every scored workload, none unscored."""
    stdout = _run_driver(run, "--bench-mode", "--warmup", "1", "--iters", "2")
    case_times, unscored, duplicates = parse_case_timings(stdout)
    assert set(case_times) == set(CASES)
    assert not unscored and not duplicates
    assert re.search(r"^median_ms: 0\.030000$", stdout, re.MULTILINE)


def test_correctness_output_carries_the_allclose_fallback(run):
    """SIKL grades a violation fraction, not an SNR; the loop reads allclose instead."""
    stdout = _run_driver(run)
    assert "allclose: True" in stdout
    assert re.search(r"max_diff:\s*([\d.eE+-]+)", stdout)
    assert "SNR:" not in stdout


def test_a_workload_sikl_finds_wrong_fails_its_case(run):
    run.kernel.write_text(FAILING_KERNEL, encoding="utf-8")
    stdout = _run_driver(run)
    for case_id in CASES:
        assert f"case_allclose: {case_id} False" in stdout
        assert f"sikl_findings: {case_id} numeric_mismatch" in stdout
    assert "allclose: False" in stdout


def test_one_validate_run_serves_every_mode_of_a_kernel(run, sikl_root: Path):
    """Validation times as well as checks, past forge-loop's limit on one benchmark; a kernel is validated once."""
    _run_driver(run)
    _run_driver(run, "--bench-mode")
    assert _validate_calls(sikl_root) == 1
    run.kernel.write_text(FAILING_KERNEL, encoding="utf-8")
    _run_driver(run)
    assert _validate_calls(sikl_root) == 2


def test_the_published_candidate_leaves_the_checkout(run, sikl_root: Path):
    """A candidate file left in the SIKL checkout would be served beside the next one."""
    _run_driver(run)
    published = [
        path
        for path in sikl_root.glob("sikl/ops/*/impls/*/*/*.json")
        if json.loads(path.read_text())["kernel"]["sources"]
    ]
    assert published == []


def test_finding_kinds_split_into_wrong_and_slow(run):
    """A speed finding grades a workload SIKL still serves the baseline for; only a correctness one fails it."""
    driver = _import_driver(run)
    (fast, slow) = CASES
    report = {
        "stages": [
            {"stage_name": "audit", "findings": []},
            {
                "stage_name": "bench",
                "findings": [
                    {"code": "not_faster", "axes": {"m": 8}},
                    {"code": "nondeterministic", "axes": {"m": 16}},
                ],
                "workload_records": [],
            },
        ]
    }
    outcomes = driver._outcomes(report)
    assert outcomes[fast]["incorrect"] == []
    assert outcomes[slow]["incorrect"] == ["nondeterministic"]


def test_an_audit_finding_rejects_every_workload(run):
    driver = _import_driver(run)
    report = {"stages": [{"stage_name": "audit", "findings": [{"code": "source_rejected", "axes": None}]}]}
    assert all(outcome["incorrect"] == ["source_rejected"] for outcome in driver._outcomes(report).values())


def test_driver_is_importable_without_a_gpu(run):
    """forge-loop hashes the driver before it ever runs; importing must not need ROCm or SIKL."""
    assert _import_driver(run).SCHEMA_JSON == run.schema_json


TRITON_KERNELS = """import triton
from triton import jit as compile_kernel


@triton.autotune(configs=[], key=[])
@triton.jit
def _scale(x):
    pass


@compile_kernel(do_not_specialize=["x"])
def _shift(x):
    pass
"""

FLYDSL_KERNELS = """import flydsl.compiler as flyc


@flyc.kernel
def _scale(x):
    pass


@flyc.kernel(name="gemm_bf16")
def _gemm(x):
    pass
"""


@pytest.mark.parametrize(
    ("language", "source", "launched", "own"),
    [
        ("triton", TRITON_KERNELS, "_scale", True),
        ("triton", TRITON_KERNELS, "_shift_0d1d2d", True),
        ("triton", TRITON_KERNELS, "run", False),
        ("triton", FLYDSL_KERNELS, "_scale", False),
        ("flydsl", FLYDSL_KERNELS, "_scale_0", True),
        ("flydsl", FLYDSL_KERNELS, "gemm_bf16", True),
        ("flydsl", FLYDSL_KERNELS, "_gemm_1", False),
        ("flydsl", FLYDSL_KERNELS, "Cijk_Alik_Bljk_BBS_BH_MT128x128x64", False),
        ("flydsl", FLYDSL_KERNELS, "void at::native::elementwise_kernel<128, 4>", False),
    ],
)
def test_a_launched_kernel_is_the_candidates_own_only_if_it_defines_it_in_its_language(
    sikl_root: Path, tmp_path: Path, language, source, launched, own
):
    """The profiler's kernel name is matched to the functions `kernel.py` decorates with the language's own
    kernel decorator, however it was imported."""
    schema = load(find(sikl_root, SCHEMA_NAME))
    run = workspace.materialize(
        schema, tmp_path / "run", sikl_root=sikl_root, language=language, starting_kernel=STARTING_KERNEL
    )
    driver = _import_driver(run)
    assert driver._is_defined_here(launched, driver._kernels_defined_here(source)) is own


def test_a_case_that_launches_a_foreign_kernel_fails_correctness(run, monkeypatch, capsys):
    """A workload SIKL passes still fails when its call launches work `kernel.py` does not define."""
    run.kernel.write_text(CANDIDATE_KERNEL, encoding="utf-8")
    driver = _import_driver(run)
    first, second = CASES
    monkeypatch.setattr(driver, "_foreign_kernels", lambda: {first: [], second: ["Cijk_Alik_Bljk_BBS_BH"]})
    driver._run_correctness()
    stdout = capsys.readouterr().out
    assert f"foreign_kernels: {second} Cijk_Alik_Bljk_BBS_BH" in stdout
    assert f"case_allclose: {first} True" in stdout
    assert f"case_allclose: {second} False" in stdout
    assert "allclose: False" in stdout


def test_the_starting_kernel_is_exempt_until_edited(run, monkeypatch, capsys):
    """The starting kernel anchors the first measurement; any edit makes it a candidate."""
    driver = _import_driver(run)
    monkeypatch.setattr(driver, "_foreign_kernels", lambda: {case_id: ["Cijk_Alik_Bljk_BBS_BH"] for case_id in CASES})
    assert hashlib.sha256(run.kernel.read_bytes()).hexdigest() in driver.STARTING_DIGESTS
    driver._run_correctness()
    assert "allclose: True" in capsys.readouterr().out
    run.kernel.write_text(CANDIDATE_KERNEL, encoding="utf-8")
    driver._run_correctness()
    assert "allclose: False" in capsys.readouterr().out
