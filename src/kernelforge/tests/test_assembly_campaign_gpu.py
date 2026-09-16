# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-output campaign, wrong-result rejection and clean export replay."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm PyTorch", allow_module_level=True)
if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx950"):
    pytest.skip("vector-add campaign test targets gfx950", allow_module_level=True)
pytest.importorskip("flydsl.compiler")

from kernelforge.assembly.prepare import prepare_assembly
from kernelforge.config import Config
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.mcp_server.tools.bench import bench_wallclock


def test_flydsl_capture_wrong_result_and_clean_export(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parents[3] / "examples/flydsl2asm-vector-add"
    workspace = tmp_path / "campaign"
    shutil.copytree(example, workspace)

    def git(*args, cwd=workspace):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "user.email", "forge@example.com")
    git("config", "user.name", "Forge")
    git("add", ".")
    git("commit", "-m", "source")
    base = git("rev-parse", "HEAD")

    driver = str(workspace / "driver.py")
    record = asyncio.run(
        prepare_assembly(
            config=Config(workspace=str(workspace), gpu_target="gfx950"),
            kernel=str(workspace / "kernel.py"),
            driver=driver,
            sources=[],
            base_commit=base,
            threshold=50.0,
            deadline=time.time() + 1800,
        )
    )
    assert record["build_failure_probe_passed"]
    assert record["origin"] == "flydsl_compiler"

    source = workspace / "kernel.s"
    original = source.read_text()
    edited, count = re.subn(r"\bv_add_f32(?P<encoding>_e32|_e64)?\b", r"v_sub_f32\g<encoding>", original)
    assert count == 1
    source.write_text(edited)
    try:
        report = asyncio.run(run_validation_pipeline(driver))
        assert not report.all_passed, "independent oracle accepted subtraction"
        assert report.failed_outcome == "correctness_failure", report.failed_output
    finally:
        source.write_text(original)

    replay = tmp_path / "replay"
    git("clone", str(workspace), str(replay))
    git("checkout", base, cwd=replay)
    patch = tmp_path / "roundtrip.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", base, record["preparation_commit"]],
            check=True,
            capture_output=True,
        ).stdout
    )
    git("apply", str(patch), cwd=replay)
    assert (replay / "kernel.s").read_text() == original
    assert not (replay / "forge_experiments").exists()
    assert asyncio.run(run_validation_pipeline(str(replay / "driver.py"))).all_passed
    bench = asyncio.run(
        bench_wallclock(
            str(replay / "driver.py"),
            driver_args=["--bench-case", "random"],
            warmup_iters=2,
            bench_iters=10,
            repeat=2,
        )
    )
    assert bench["success"], bench
    assert set(bench["case_times"]) == {"random"}
