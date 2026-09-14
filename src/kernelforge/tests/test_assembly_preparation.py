# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-output preparation, original-baseline selection and resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from kernelforge.assembly import prepare
from kernelforge.loop.validation import ValidationReport, ValidationResult

ASM = """.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
add:
    s_endpgm
.amdhsa_kernel add
.end_amdhsa_kernel
.amdgpu_metadata
.end_amdgpu_metadata
"""
SOURCE = "import flydsl.compiler as flyc\ndef build(*args):\n    return flyc.compile(*args)\n"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def report(passed=True, output="OK"):
    return ValidationReport([ValidationResult(1, "full", passed, output)])


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Forge Test")
    git(tmp_path, "config", "user.email", "forge@example.com")
    (tmp_path / ".gitignore").write_text("forge_experiments/\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text(SOURCE)
    driver = tmp_path / "driver.py"
    driver.write_text("# fixed independent oracle\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "original")
    options = dict(
        config=SimpleNamespace(workspace=str(tmp_path), gpu_target="gfx950"),
        kernel=str(kernel),
        driver=str(driver),
        sources=[],
        base_commit=git(tmp_path, "rev-parse", "HEAD"),
        threshold=50.0,
        deadline=time.time() + 600,
    )

    async def validate(driver, threshold, deadline):
        prepare._timeout(deadline)
        source = kernel.with_suffix(".s")
        if "export=True" in kernel.read_text():
            source.write_text(ASM)
            source.with_suffix(".s.json").write_text(
                json.dumps({"compiler_assembly_sha256": hashlib.sha256(ASM.encode()).hexdigest()})
            )
        broken = source.exists() and "FORGE_ASSEMBLY_BUILD_PROBE" in source.read_text()
        if source.exists() and "FORGE_ASSEMBLY_EXECUTION_PROBE" in source.read_text():
            return report(False, "no-op assembly failed oracle")
        return report(not broken, "FORGE_ASSEMBLY_BUILD_PROBE" if broken else "OK")

    async def bench(*args, **kwargs):
        latency = 2.0 if "_forge_assembly" in kernel.read_text() else 1.0
        return {"success": True, "median_ms": latency, "case_times": {"one": latency}}

    from kernelforge.orchestrator import agent

    def no_agent(**kwargs):
        raise AssertionError("assembly preparation must never create an LLM agent")

    monkeypatch.setattr(agent, "make_agent_fn", no_agent)
    monkeypatch.setattr(prepare, "_validate", validate)
    monkeypatch.setattr(prepare, "bench_wallclock", bench)
    return tmp_path, options


def test_capture_keeps_frontend_and_original_baseline(campaign):
    root, options = campaign
    result = asyncio.run(prepare.prepare_assembly(**options))
    assert result["origin"] == "flydsl_compiler"
    assert result["build_failure_probe_passed"]
    assert result["execution_probe_passed"]
    assert result["roundtrip_mean_case_speedup"] == 0.5
    assert "import flydsl.compiler as flyc" in (root / "kernel.py").read_text()
    assert "return _forge_assembly(*args)" in (root / "kernel.py").read_text()
    assert git(root, "diff", "--name-only", options["base_commit"], "HEAD").splitlines() == [
        "kernel.py",
        "kernel.s",
        "kernel.s.json",
    ]
    assert not git(root, "status", "--porcelain")
    config = SimpleNamespace()
    prepare.seed_source_baseline(config, result)
    assert config.baseline_case_times == {"one": 1.0}
    assert not hasattr(config, "warm_start_commit")


@pytest.mark.parametrize("changed", ["kernel.py", "kernel.s.json"])
def test_resume_accepts_instruction_edits_but_freezes_binding(campaign, changed):
    root, options = campaign
    record = asyncio.run(prepare.prepare_assembly(**options))
    (root / "kernel.s").write_text(ASM + "// later instruction edit\n")
    git(root, "add", "kernel.s")
    git(root, "commit", "-m", "candidate")
    assert asyncio.run(prepare.prepare_assembly(**options, resume=True)) == record
    (root / changed).write_text("# unauthorized\n")
    with pytest.raises(prepare.AssemblyPreparationError, match="changed after preparation"):
        asyncio.run(prepare.prepare_assembly(**options, resume=True))


def test_source_fallback_cannot_pass_build_probe(campaign, monkeypatch):
    root, options = campaign
    validate = prepare._validate

    async def always_pass(*args):
        await validate(*args)
        return report()

    monkeypatch.setattr(prepare, "_validate", always_pass)
    with pytest.raises(prepare.AssemblyPreparationError, match="deliberate assembly build failure"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE
    assert not (root / "kernel.s").exists()
    assert not (root / "kernel.s.json").exists()
    assert git(root, "rev-parse", "HEAD") == options["base_commit"]


def test_driver_must_execute_candidate_beyond_compiler_warmup(campaign, monkeypatch):
    root, options = campaign
    validate = prepare._validate

    async def accepts_noop(*args):
        result = await validate(*args)
        source = root / "kernel.s"
        if source.exists() and "FORGE_ASSEMBLY_EXECUTION_PROBE" in source.read_text():
            return report()
        return result

    monkeypatch.setattr(prepare, "_validate", accepts_noop)
    with pytest.raises(prepare.AssemblyPreparationError, match="driver accepted no-op"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE


@pytest.mark.parametrize("timeout_call", [2, 3])
def test_probe_restores_source_on_timeout(campaign, monkeypatch, timeout_call):
    root, options = campaign
    source = root / "kernel.s"
    source.write_text(ASM)
    calls = 0

    async def timeout_on_probe(*args):
        nonlocal calls
        calls += 1
        if calls == timeout_call:
            raise asyncio.TimeoutError
        if calls == 2:
            return report(False, "FORGE_ASSEMBLY_BUILD_PROBE")
        return report()

    monkeypatch.setattr(prepare, "_validate", timeout_on_probe)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            prepare.verify_assembly(root / "kernel.py", source, options["driver"], "gfx950", 50, options["deadline"])
        )
    assert source.read_text() == ASM


@pytest.mark.parametrize("failure", ["publication", "benchmark", "correctness"])
def test_failed_preparation_restores_files_and_git(campaign, monkeypatch, failure):
    root, options = campaign
    if failure == "publication":

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(prepare, "atomic_write_text", fail)
        error, message = OSError, "disk full"
    elif failure == "benchmark":

        async def invalid(*args, **kwargs):
            return {"success": True, "median_ms": 1.0, "case_times": {"one": float("nan")}}

        monkeypatch.setattr(prepare, "bench_wallclock", invalid)
        error, message = ValueError, "positive finite"
    else:

        async def invalid(*args):
            return report(False, "oracle failed")

        monkeypatch.setattr(prepare, "_validate", invalid)
        error, message = ValueError, "oracle failed"
    with pytest.raises(error, match=message):
        asyncio.run(prepare.prepare_assembly(**options))
    assert git(root, "rev-parse", "HEAD") == options["base_commit"]
    assert not git(root, "status", "--porcelain")
    assert not (root / "forge_experiments/assembly_preparation/result.json").exists()


def test_canonical_rejection_prevents_publication(campaign):
    root, options = campaign
    (root / "config.yaml").write_text(
        "compile_command: ['true']\ncorrectness_command: [\"! grep -q '_forge_assembly' kernel.py\"]\n"
    )
    git(root, "add", "config.yaml")
    git(root, "commit", "-m", "canonical acceptance")
    with pytest.raises(prepare.AssemblyPreparationError, match="correctness"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE


def test_existing_assembly_and_scope(campaign):
    root, options = campaign
    with pytest.raises(prepare.AssemblyPreparationError, match="verified preparation record"):
        asyncio.run(prepare.prepare_assembly(**options, resume=True))
    with pytest.raises(prepare.AssemblyPreparationError, match="inside the workspace"):
        prepare.assembly_edit_paths(root, "../escape.s", [])
    with pytest.raises(prepare.AssemblyPreparationError, match="exactly one"):
        prepare.assembly_edit_paths(root, "a.s", ["b.s"])
    (root / "kernel.s").write_text(ASM)
    (root / "unrelated.s").write_text(ASM)
    git(root, "add", ".")
    git(root, "commit", "-m", "existing bound assembly")
    record = asyncio.run(prepare.prepare_assembly(**options))
    assert record["origin"] == "existing_assembly"
    protected = prepare.frozen_paths(root, [root / "kernel.s"])
    assert str(root / "kernel.py") in protected
    assert str(root / "unrelated.s") in protected
    assert str(root / "kernel.s") not in protected


def test_no_keep_selects_original_and_can_resume(campaign, monkeypatch):
    from kernelforge.loop.runner import IterationConfig, IterationLoop
    from kernelforge.tracker import ExperimentTracker

    root, options = campaign
    record = asyncio.run(prepare.prepare_assembly(**options))
    config = IterationConfig(
        kernel_file=options["kernel"],
        driver_script=options["driver"],
        workspace_dir=str(root),
        campaign_base_commit=options["base_commit"],
        kernel_backend="assembly",
        source_files=[str(root / "kernel.s")],
        git_branch="assembly-test",
        max_time_hours=1.0,
    )
    prepare.seed_source_baseline(config, record)
    tracker = ExperimentTracker(root / "forge_experiments/tracker")
    loop = IterationLoop(config, tracker, config=object())
    monkeypatch.setattr(loop, "_time_remaining", lambda: 0.0)
    asyncio.run(loop.run())
    assert loop.best_mean_case_speedup == 1.0
    assert not (root / "forge_experiments/best/manifest.json").exists()
    result = {"improved": False, "best_ms": 2, "best_commit": record["preparation_commit"], "mean_case_speedup": 0.5}
    prepare.select_result(result, record)
    assert result["best_commit"] == options["base_commit"]
    assert result["best_ms"] == 1.0
    assert result["mean_case_speedup"] == 1.0
    assert result["best_manifest"] == ""
    assert result["selected_implementation"] == "original"
    resumed = IterationLoop(config, tracker, config=object(), resume=True)
    monkeypatch.setattr(resumed, "_time_remaining", lambda: 0.0)
    asyncio.run(resumed.run())
    assert resumed.best_mean_case_speedup == 1.0
    assert "export=False" in (root / "kernel.py").read_text()
    assert (root / "kernel.s").read_text() == ASM
