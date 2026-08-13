# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forge-loop argv contract, checked against the installed KernelForge.

Hyperloom and KernelForge are separate repositories, wired together at runtime
through ``$FORGE_PATH``, so nothing else makes them agree on this argv. click
rejects an unknown option while parsing, before running anything, which is why a
single retired or renamed option costs an entire attempt instead of degrading
it: 22 attempts were lost to ``--shapes-json`` alone.

Reading ``--help`` is not enough. Hidden options are absent from it, so a check
built on help text cannot see the ones most likely to be retired -- ``--help``
listed neither ``--shapes-json`` nor its removal.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _installed_forge_loop_options() -> set[str]:
    """Every option the installed forge-loop accepts, hidden ones included."""

    cli = pytest.importorskip(
        "kernel_agents.cli",
        reason="KernelForge is not installed, so its argv cannot be checked",
    )
    command = cli.main.commands.get("forge-loop")
    if command is None:
        pytest.fail("the installed kernel_agents CLI has no forge-loop command")
    supported: set[str] = set()
    for param in command.params:
        supported.update(param.opts)
        supported.update(param.secondary_opts)
    return supported


def _maximal_launcher_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Build the argv with every conditional option present.

    Options this launcher only adds for some candidates are exactly the ones a
    narrower test would miss, so each condition is satisfied here.
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


def test_every_option_the_launcher_passes_exists_in_the_installed_forge_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported = _installed_forge_loop_options()
    argv = _maximal_launcher_argv(tmp_path, monkeypatch)
    passed = {token for token in argv if token.startswith("--")}

    assert passed, "the launcher argv carries no options; the probe is broken"
    missing = sorted(passed - supported)
    assert not missing, (
        "the installed forge-loop does not accept "
        + ", ".join(missing)
        + ". click exits 2 on the first unknown option, so every forge attempt "
        "would be lost. Either stop passing it or install a KernelForge that "
        "has it."
    )


def test_the_probe_would_notice_an_option_that_disappeared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A green contract test must be able to fail; pin that it can."""
    argv = _maximal_launcher_argv(tmp_path, monkeypatch)
    passed = {token for token in argv if token.startswith("--")}
    supported = _installed_forge_loop_options()
    sentinel = "--kernel"

    assert sentinel in passed and sentinel in supported
    assert sentinel in passed - (supported - {sentinel})
