# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real Triton-to-ASM preparation with the attributed AttnRes seed, without an LLM."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm PyTorch", allow_module_level=True)
if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx950"):
    pytest.skip("AttnRes assembly example specializes gfx950", allow_module_level=True)
pytest.importorskip("triton")

from kernelforge.assembly.port import prepare_assembly
from kernelforge.config import Config
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.orchestrator import agent


def test_attnres_port_graph_rebinding_wrong_result_and_clean_export(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parents[3] / "examples/triton2asm-attnres"
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

    def seed_port(**kwargs):
        assert kwargs["correctness_only"]

        async def run(*args, **kwargs):
            shutil.copyfile(workspace / "seed/launcher.py", workspace / "kernel.py")
            shutil.copyfile(workspace / "seed/score.s", workspace / "kernel.s")

        return run

    monkeypatch.setattr(agent, "make_agent_fn", seed_port)
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
    assert record["port_attempts"] == 1

    source = workspace / "kernel.s"
    original = source.read_text()
    entry = "kimik3_attnres_score:\n"
    assert original.count(entry) == 1
    source.write_text(original.replace(entry, entry + "s_endpgm\n"))
    try:
        report = asyncio.run(run_validation_pipeline(driver))
        assert not report.all_passed, "independent oracle accepted a no-op kernel"
    finally:
        source.write_text(original)

    replay = tmp_path / "replay"
    git("clone", str(workspace), str(replay))
    git("checkout", base, cwd=replay)
    patch = tmp_path / "port.patch"
    patch.write_text(git("diff", "--binary", base, record["port_commit"]) + "\n")
    git("apply", str(patch), cwd=replay)
    assert (replay / "kernel.s").read_text() == original
    assert not (replay / "forge_experiments").exists()
    assert asyncio.run(run_validation_pipeline(str(replay / "driver.py"))).all_passed
