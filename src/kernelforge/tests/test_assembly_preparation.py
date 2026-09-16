# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-output preparation, original-baseline selection and resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

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
    return ValidationReport(
        [ValidationResult(1, "full", passed, output, outcome="pass" if passed else "correctness_failure")]
    )


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
    numerical = {"finite": True, "oracle_errors": [0] * 3, "repeat_errors": [0] * 2}
    (tmp_path / "numerical_reference.py").write_text(
        "import json,os\n"
        + "record="
        + repr(
            {
                "schema_version": 1,
                "cases": [{"id": "one", "source_before": numerical, "candidate": numerical, "source_after": numerical}],
            }
        )
        + "\nfrom pathlib import Path\nassembly=Path('kernel.s')\n"
        + "if assembly.exists() and 'FORGE_ASSEMBLY_EXECUTION_PROBE' in assembly.read_text():\n"
        + "    record['cases'][0]['candidate']['oracle_errors']=[1]*3\n"
        + "record['request_id']=os.environ['FORGE_NUMERICAL_REQUEST']\nprint('__FORGE_NUMERICAL__'+json.dumps(record))\n"
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "compile_command": [f'"{sys.executable}" driver.py'],
                "correctness_command": [f'"{sys.executable}" numerical_reference.py'],
                "numerical_validation": {
                    "schema_version": 1,
                    "repetitions": 3,
                    "cases": {"one": {"max_oracle_error": 0, "max_error_ratio": 1, "error_floor": 0}},
                },
            }
        )
    )
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
                json.dumps({"frontend": "flydsl", "compiler_assembly_sha256": hashlib.sha256(ASM.encode()).hexdigest()})
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
    probe = result["numerical_execution_probe_evidence"]["cases"][0]
    assert probe["source_before"]["oracle_errors"] == [0] * 3
    assert probe["source_after"]["oracle_errors"] == [0] * 3
    assert probe["candidate"]["oracle_errors"] == [1] * 3
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


@pytest.mark.parametrize(
    "wiring", ["source_before", "source_after", "masked_candidate", "unstable_source", "missing_evidence", "crash"]
)
def test_numerical_probe_rejects_miswired_measurements(campaign, wiring):
    root, options = campaign
    path = root / "numerical_reference.py"
    change = {
        "source_before": "record['cases'][0]['source_before'] = record['cases'][0]['candidate']",
        "source_after": "record['cases'][0]['source_after'] = record['cases'][0]['candidate']",
        "masked_candidate": "record['cases'][0]['candidate']['oracle_errors'] = [0]*3",
        "unstable_source": "record['cases'][0]['source_before']['repeat_errors'] = [1]*2",
        "missing_evidence": "raise SystemExit(0)",
        "crash": "raise RuntimeError('numerical driver crashed')",
    }[wiring]
    text = path.read_text().replace(
        "    record['cases'][0]['candidate']['oracle_errors']=[1]*3",
        "    record['cases'][0]['candidate']['oracle_errors']=[1]*3\n    " + change,
    )
    path.write_text(text)
    git(root, "add", path.name)
    git(root, "commit", "-m", "miswire numerical driver")
    with pytest.raises(prepare.AssemblyPreparationError, match="numerical execution probe"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE
    assert not (root / "kernel.s").exists()
    assert not (root / "forge_experiments/assembly_preparation/result.json").exists()


@pytest.mark.parametrize("comment", ["; @add", "// compiler label"])
def test_execution_probe_accepts_compiler_label_comments(campaign, monkeypatch, comment):
    monkeypatch.setattr(sys.modules[__name__], "ASM", ASM.replace("add:", "add: " + comment))
    _, options = campaign
    record = asyncio.run(prepare.prepare_assembly(**options))
    assert record["execution_probe_passed"]


def test_execution_probe_stops_preloaded_basic_block_entry(campaign, monkeypatch):
    assembly = ASM.replace("    s_endpgm", "    s_branch .LBB0_0\n.p2align 8\n.LBB0_0:\n    s_endpgm")
    monkeypatch.setattr(sys.modules[__name__], "ASM", assembly)
    root, options = campaign
    validate = prepare._validate

    async def check_preloaded_entry(*args):
        path = root / "kernel.s"
        if path.exists() and "FORGE_ASSEMBLY_EXECUTION_PROBE" in path.read_text():
            assert ".LBB0_0:\n    s_endpgm // FORGE_ASSEMBLY_EXECUTION_PROBE" in path.read_text()
        return await validate(*args)

    monkeypatch.setattr(prepare, "_validate", check_preloaded_entry)
    assert asyncio.run(prepare.prepare_assembly(**options))["execution_probe_passed"]


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


@pytest.mark.parametrize(
    "driver_source, accepted",
    [
        ("import time; time.sleep(60)", False),
        ("raise RuntimeError('hipModuleLoadData failed')", False),
        ("raise AssertionError('unclassified driver assertion')", False),
        ("print('no output comparison was performed')", False),
        ("print('max_diff: 0.5')", False),
        ("print('SNR: 0.0 dB')", True),
        ("print('allclose: False')", True),
    ],
    ids=["timeout", "load-error", "assertion", "missing-metrics", "max-diff-without-tolerance", "snr", "allclose"],
)
def test_execution_probe_requires_a_measured_correctness_failure(campaign, monkeypatch, driver_source, accepted):
    root, options = campaign
    scratch = root / "forge_experiments"
    scratch.mkdir()
    probe_driver = scratch / "probe_driver.py"
    probe_driver.write_text(driver_source)
    validate = prepare._validate

    async def invalid_probe(*args):
        result = await validate(*args)
        source = root / "kernel.s"
        if source.exists() and "FORGE_ASSEMBLY_EXECUTION_PROBE" in source.read_text():
            return await prepare.run_validation_pipeline(str(probe_driver), timeout_per_stage=1)
        return result

    monkeypatch.setattr(prepare, "_validate", invalid_probe)
    if accepted:
        record = asyncio.run(prepare.prepare_assembly(**options))
        assert record["execution_probe_passed"]
        assert (root / "kernel.s").read_text() == ASM
        return
    with pytest.raises(prepare.AssemblyPreparationError, match="measured correctness failure"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE
    assert not (root / "kernel.s").exists()
    assert not (root / "kernel.s.json").exists()
    assert git(root, "rev-parse", "HEAD") == options["base_commit"]
    assert not (scratch / "assembly_preparation/result.json").exists()


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
    settings = yaml.safe_load((root / "config.yaml").read_text())
    settings["correctness_command"].insert(0, "! grep -q '_forge_assembly' kernel.py")
    (root / "config.yaml").write_text(yaml.safe_dump(settings))
    git(root, "add", "config.yaml")
    git(root, "commit", "-m", "canonical acceptance")
    with pytest.raises(prepare.AssemblyPreparationError, match="correctness"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert (root / "kernel.py").read_text() == SOURCE


def test_missing_numerical_contract_prevents_assembly_preparation(campaign):
    root, options = campaign
    settings = yaml.safe_load((root / "config.yaml").read_text())
    del settings["numerical_validation"]
    (root / "config.yaml").write_text(yaml.safe_dump(settings))
    git(root, "add", "config.yaml")
    git(root, "commit", "-m", "legacy driver")
    with pytest.raises(prepare.AssemblyPreparationError, match="numerical_validation"):
        asyncio.run(prepare.prepare_assembly(**options))
    assert not (root / "kernel.s").exists()


def test_resume_rejects_weakened_numerical_contract(campaign):
    root, options = campaign
    record = asyncio.run(prepare.prepare_assembly(**options))
    assert record["roundtrip_numerical_evidence"]["contract_sha256"]
    settings = yaml.safe_load((root / "config.yaml").read_text())
    settings["numerical_validation"]["cases"]["one"]["max_oracle_error"] = 1
    (root / "config.yaml").write_text(yaml.safe_dump(settings))
    with pytest.raises(prepare.AssemblyPreparationError, match="unchanged numerical acceptance"):
        asyncio.run(prepare.prepare_assembly(**options, resume=True))


@pytest.mark.parametrize("committed", [False, True], ids=["working-tree", "committed"])
def test_resume_freezes_reference_helpers_from_preparation(campaign, committed):
    root, options = campaign
    asyncio.run(prepare.prepare_assembly(**options))
    reference = root / "numerical_reference.py"
    reference.write_text(reference.read_text() + "\n# changed reference implementation\n")
    if committed:
        git(root, "add", "numerical_reference.py")
        git(root, "commit", "-m", "change reference")
    with pytest.raises(prepare.AssemblyPreparationError, match="frozen assembly inputs changed"):
        asyncio.run(prepare.prepare_assembly(**options, resume=True))


@pytest.mark.parametrize("old_schema", [2, 3])
def test_resume_rejects_preparation_without_measured_execution_proof(campaign, old_schema):
    root, options = campaign
    record = asyncio.run(prepare.prepare_assembly(**options))
    record["schema_version"] = old_schema
    (root / "forge_experiments/assembly_preparation/result.json").write_text(json.dumps(record))
    with pytest.raises(prepare.AssemblyPreparationError, match="prepare a fresh campaign"):
        asyncio.run(prepare.prepare_assembly(**options, resume=True))


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
