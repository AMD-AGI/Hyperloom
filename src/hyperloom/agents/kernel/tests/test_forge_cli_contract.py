# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declared forge-loop option set, pinned against the argv really built.

`FORGE_LOOP_OPTIONS` is what the launch-time gate compares against the installed
KernelForge. Nothing in a running image can capture an argv, so the declaration
is the gate's only account of what the launcher sends -- and a stale declaration
makes the gate worse than useless: it would pass while the run still lost every
attempt to an option the declaration forgot.

The gate itself, and the KernelForge it reads, are covered by
``inference_optimizer/tests/test_preflight_forge_contract.py``. Nothing here
touches KernelForge, so this holds wherever the suite runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _maximal_launcher_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Build the argv with every conditional option present.

    Options the launcher only adds for some candidates are exactly the ones a
    narrower probe would miss, so each condition is satisfied here.
    """
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    program = tmp_path / "program.md"
    invocation_spec = tmp_path / "invocation_spec.json"
    for path, body in (
        (kernel, "pass\n"),
        (driver, "pass\n"),
        (program, "# Task\n"),
        (invocation_spec, "{}\n"),
    ):
        path.write_text(body)
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        pid = 43214
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "/forge/src")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit, "_openai_only_provider", lambda: True)
    monkeypatch.setenv("CODEX_MODEL", "gpt-5-codex")
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        snr_threshold=30.0,
        max_iters=8,
        max_hours=1.0,
        branch="forge/session/kernel",
        gpu_target="gfx950",
        gpu_type="mi355x",
        fellow="triton-fellow",
        program_md_file=str(program),
        invocation_spec_file=str(invocation_spec),
        experiments_dir=experiments,
        forge_log=tmp_path / "forge.log",
        timeout_s=120,
        deadline_unix=time.time() + 120.0,
        experience_id="attempt-1",
        operator_name="vllm::logical_op",
        framework="vllm",
        target_functions=["kernel_impl"],
        source_files=[str(kernel)],
    )
    return captured["command"]


def test_the_declared_option_set_matches_the_argv_the_launcher_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _maximal_launcher_argv(tmp_path, monkeypatch)
    passed = {token for token in argv if token.startswith("--")}

    assert passed, "the launcher argv carries no options; the probe is broken"
    assert passed == set(forge_submit.FORGE_LOOP_OPTIONS)
