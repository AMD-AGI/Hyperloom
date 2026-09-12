# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly preparation, execution-path checks and source-relative scoring."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from kernelforge.assembly import port
from kernelforge.loop.validation import ValidationReport, ValidationResult

ASM = """.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.amdhsa_kernel add
.end_amdhsa_kernel
.amdgpu_metadata
.end_amdgpu_metadata
"""
LAUNCHER = "from kernelforge.assembly.compiler import assemble\nfrom kernelforge.assembly.hip import HipKernel\n"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Forge Test")
    git(tmp_path, "config", "user.email", "forge@example.com")
    (tmp_path / ".gitignore").write_text("forge_experiments/\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n")
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
        source = kernel.with_suffix(".s")
        broken = source.exists() and "FORGE_ASSEMBLY_BUILD_PROBE" in source.read_text()
        return ValidationReport(
            [ValidationResult(1, "full", not broken, "FORGE_ASSEMBLY_BUILD_PROBE" if broken else "OK")]
        )

    async def bench(*args, **kwargs):
        latency = 2.0 if kernel.read_text() == LAUNCHER else 1.0
        return {"success": True, "median_ms": latency, "case_times": {"one": latency}}

    from kernelforge.orchestrator import agent

    def make_agent(**kwargs):
        assert kwargs["correctness_only"]
        assert str(driver) in kwargs["extra_protected_paths"]

        async def run(*args, **kw):
            kernel.write_text(LAUNCHER)
            kernel.with_suffix(".s").write_text(ASM)

        return run

    monkeypatch.setattr(agent, "make_agent_fn", make_agent)
    monkeypatch.setattr(port, "_validate", validate)
    monkeypatch.setattr(port, "bench_wallclock", bench)
    return tmp_path, options


def test_slower_correct_port_is_committed_and_export_contains_launcher_and_assembly(campaign):
    root, options = campaign
    result = asyncio.run(port.prepare_assembly(**options))
    assert result["port_attempts"] == 1
    assert result["build_failure_probe_passed"]
    assert result["source_benchmark"]["median_ms"] == 1.0
    assert result["initial_assembly_benchmark"]["median_ms"] == 2.0
    assert git(root, "diff", "--name-only", options["base_commit"], "HEAD").splitlines() == ["kernel.py", "kernel.s"]
    assert not git(root, "status", "--porcelain")
    config = SimpleNamespace()
    port.seed_port_baseline(config, result)
    assert config.baseline_case_times == {"one": 1.0}
    assert config.warm_start_mean_case_speedup == 0.5
    assert config.warm_start_commit == result["port_commit"]


def test_resume_keeps_edited_assembly_but_rejects_changed_launcher(campaign):
    root, options = campaign
    record = asyncio.run(port.prepare_assembly(**options))
    (root / "kernel.s").write_text(ASM + "// valid later instruction edit\n")
    git(root, "add", "kernel.s")
    git(root, "commit", "-m", "later candidate")
    assert asyncio.run(port.prepare_assembly(**options, resume=True)) == record
    (root / "kernel.py").write_text(LAUNCHER + "# unauthorized dispatch edit\n")
    with pytest.raises(port.AssemblyPreparationError, match="launcher changed"):
        asyncio.run(port.prepare_assembly(**options, resume=True))


def test_fallback_cannot_pass_build_failure_probe(campaign, monkeypatch):
    root, options = campaign

    async def always_pass(*args):
        return ValidationReport([ValidationResult(1, "full", True, "OK")])

    monkeypatch.setattr(port, "_validate", always_pass)
    with pytest.raises(port.AssemblyPreparationError, match="deliberate assembly build failure"):
        asyncio.run(port.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == "import triton\n"
    assert not (root / "kernel.s").exists()
    assert git(root, "rev-parse", "HEAD") == options["base_commit"]


def test_probe_restores_source_on_timeout(campaign, monkeypatch):
    root, options = campaign
    kernel = root / "kernel.py"
    assembly = root / "kernel.s"
    kernel.write_text(LAUNCHER)
    assembly.write_text(ASM)
    calls = 0

    async def timeout_on_probe(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.TimeoutError
        return ValidationReport([ValidationResult(1, "full", True, "OK")])

    monkeypatch.setattr(port, "_validate", timeout_on_probe)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(port.verify_assembly(kernel, assembly, options["driver"], "gfx950", 50, options["deadline"]))
    assert assembly.read_text() == ASM


def test_invalid_source_never_calls_port_agent(campaign, monkeypatch):
    root, options = campaign

    async def fail(*args):
        return ValidationReport([ValidationResult(1, "full", False, "original oracle failed")])

    monkeypatch.setattr(port, "_validate", fail)
    with pytest.raises(port.AssemblyPreparationError, match="original oracle failed"):
        asyncio.run(port.prepare_assembly(**options))
    assert not (root / "kernel.s").exists()


def test_ready_assembly_skips_llm_and_freezes_unrelated_assembly(campaign, monkeypatch):
    root, options = campaign
    (root / "kernel.py").write_text(LAUNCHER)
    (root / "kernel.s").write_text(ASM)
    (root / "unrelated.s").write_text(ASM)
    git(root, "add", ".")
    git(root, "commit", "-m", "existing assembly implementation")
    options["base_commit"] = git(root, "rev-parse", "HEAD")
    record = asyncio.run(port.prepare_assembly(**options))
    assert record["port_attempts"] == 0
    protected = port.frozen_paths(root, [root / "kernel.s"])
    assert str(root / "kernel.py") in protected
    assert str(root / "unrelated.s") in protected
    assert str(root / "kernel.s") not in protected


def test_scope_and_resume_requirements(campaign):
    root, options = campaign
    with pytest.raises(port.AssemblyPreparationError, match="verified PORT record"):
        asyncio.run(port.prepare_assembly(**options, resume=True))
    with pytest.raises(port.AssemblyPreparationError, match="inside the workspace"):
        port.assembly_edit_paths(root, "../escape.s", [])
    with pytest.raises(port.AssemblyPreparationError, match="exactly one"):
        port.assembly_edit_paths(root, "a.s", ["b.s"])
    with pytest.raises(port.AssemblyPreparationError, match="deadline"):
        asyncio.run(port.prepare_assembly(**{**options, "deadline": time.time() - 1}))


def test_preserves_existing_assembly_on_failed_port(campaign, monkeypatch):
    root, options = campaign
    assembly = root / "kernel.s"
    assembly.write_text("// existing seed\n")
    git(root, "add", "kernel.s")
    git(root, "commit", "-m", "seed")
    options["base_commit"] = git(root, "rev-parse", "HEAD")
    from kernelforge.orchestrator import agent

    def make_agent(**kwargs):
        async def run(*args, **kwargs):
            (root / "kernel.py").write_text(LAUNCHER)
            assembly.write_text("invalid assembly\n")

        return run

    monkeypatch.setattr(agent, "make_agent_fn", make_agent)
    with pytest.raises(port.AssemblyPreparationError, match="Missing"):
        asyncio.run(port.prepare_assembly(**options))
    assert assembly.read_text() == "// existing seed\n"


def test_canonical_rejection_prevents_port_publication(campaign):
    root, options = campaign
    (root / "config.yaml").write_text(
        "compile_command: ['true']\ncorrectness_command: [\"grep -q 'import triton' kernel.py\"]\n"
    )
    git(root, "add", "config.yaml")
    git(root, "commit", "-m", "canonical source-only acceptance")
    with pytest.raises(port.AssemblyPreparationError, match="correctness"):
        asyncio.run(port.prepare_assembly(**options))
    assert not (root / "forge_experiments/assembly_port/result.json").exists()
    assert (root / "kernel.py").read_text() == "import triton\n"


@pytest.mark.parametrize("failure", ["publication", "benchmark"])
def test_failed_publication_or_measurement_restores_git_state(campaign, monkeypatch, failure):
    root, options = campaign
    if failure == "publication":

        def fail_write(*args):
            raise OSError("disk full")

        monkeypatch.setattr(port, "atomic_write_text", fail_write)
        error, message = OSError, "disk full"
    else:

        async def invalid_bench(*args, **kwargs):
            return {"success": True, "median_ms": 1.0, "case_times": {"one": float("nan")}}

        monkeypatch.setattr(port, "bench_wallclock", invalid_bench)
        error, message = port.AssemblyPreparationError, "positive finite"
    with pytest.raises(error, match=message):
        asyncio.run(port.prepare_assembly(**options))
    assert git(root, "rev-parse", "HEAD") == options["base_commit"]
    assert not git(root, "status", "--porcelain")
    assert not (root / "forge_experiments/assembly_port/result.json").exists()


def test_port_without_later_keep_publishes_complete_patch_and_resumes(campaign, monkeypatch):
    from kernelforge.loop.runner import IterationConfig, IterationLoop
    from kernelforge.tracker import ExperimentTracker

    root, options = campaign
    record = asyncio.run(port.prepare_assembly(**options))
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
    port.seed_port_baseline(config, record)
    tracker = ExperimentTracker(root / "forge_experiments/tracker")
    loop = IterationLoop(config, tracker, config=object())
    monkeypatch.setattr(loop, "_time_remaining", lambda: 0.0)
    asyncio.run(loop.run())
    best = root / "forge_experiments/best"
    manifest = json.loads((best / "manifest.json").read_text())
    assert manifest["commit_hash"] == record["port_commit"]
    assert manifest["iteration"] == 0
    assert manifest["pristine_baseline_ms"] == 1.0
    assert manifest["search_start_ms"] == 2.0
    assert not manifest["total_improved"]
    assert not manifest["incremental_improved"]
    assert set(manifest["changed_files"]) == {"kernel.py", "kernel.s"}
    resumed = IterationLoop(config, tracker, config=object(), resume=True)
    monkeypatch.setattr(resumed, "_time_remaining", lambda: 0.0)
    asyncio.run(resumed.run())
    assert resumed.best_mean_case_speedup == 0.5
    assert (root / "kernel.py").read_text() == LAUNCHER
    assert (root / "kernel.s").read_text() == ASM
