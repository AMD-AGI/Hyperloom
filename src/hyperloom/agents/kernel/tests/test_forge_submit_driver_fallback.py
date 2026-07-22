"""Regression tests for Forge driver fallback delegation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _submit_with_stubbed_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    test_command: str = "",
    autogen_driver: str | None = None,
) -> tuple[dict, dict]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize kernel\n")
    output_dir = tmp_path / "forge" / "session" / "attempt"
    captured: dict = {}

    monkeypatch.setattr(forge_submit, "_needs_inplace", lambda _repo: False)
    monkeypatch.setattr(
        forge_submit,
        "_prepare_worktree",
        lambda *_args, **_kwargs: (str(workspace), str(kernel), "base-commit"),
    )
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_resolve_gpu_target", lambda _candidate: "gfx942")
    monkeypatch.setattr(
        forge_submit,
        "_autogen_forge_driver",
        lambda *_args, **_kwargs: autogen_driver,
    )
    monkeypatch.setattr(
        forge_submit,
        "_export_best_artifacts",
        lambda *_args, **_kwargs: ("", []),
    )
    monkeypatch.setattr(
        forge_submit,
        "_write_report",
        lambda out, *_args, **_kwargs: out / "optimization_report.md",
    )
    monkeypatch.setattr(forge_submit, "_remove_worktree", lambda *_args, **_kwargs: None)

    def fake_run_loop(**kwargs):
        captured.update(kwargs)
        return 1.0, 0.9, True, "loop completed", None

    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_run_loop)

    result = forge_submit.submit(
        source_file=str(kernel),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command=test_command,
        source_type="triton",
        candidate={"operation": "unsupported_op"},
        timeout_s=60,
        kernel_repo=str(workspace),
    )
    return result, captured


def test_missing_autogen_driver_reaches_forge_loop_task_preparer(monkeypatch, tmp_path):
    result, captured = _submit_with_stubbed_loop(monkeypatch, tmp_path)

    assert result["returncode"] == 0
    assert result["skipped"] is False
    assert captured["driver"].endswith("forge_task_driver.py")
    assert not Path(captured["driver"]).exists()


def test_adapter_and_autogen_failure_reaches_task_preparer(monkeypatch, tmp_path):
    result, captured = _submit_with_stubbed_loop(
        monkeypatch,
        tmp_path,
        test_command="python bench.py && echo unsafe",
    )

    assert result["returncode"] == 0
    assert result["skipped"] is False
    assert captured["driver"].endswith("forge_task_driver.py")


def test_compile_only_driver_reaches_forge_loop_task_preparer(monkeypatch, tmp_path):
    driver = tmp_path / "compile_only_driver.py"
    driver.write_text('print("compile_only: True")\n')

    result, captured = _submit_with_stubbed_loop(
        monkeypatch,
        tmp_path,
        autogen_driver=str(driver),
    )

    assert result["returncode"] == 0
    assert result["skipped"] is False
    assert captured["driver"] == str(driver)
