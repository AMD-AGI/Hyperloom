"""Regression tests for Forge driver fallback delegation."""

from __future__ import annotations

import json
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
    candidate: dict | None = None,
    invocation_spec_file: str = "",
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
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=1.0,
            best_ms=0.9,
            improved=True,
            output="loop completed",
            error=None,
            timed_out=False,
            checkpoint=None,
        )

    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_run_loop)

    result = forge_submit.submit(
        source_file=str(kernel),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command=test_command,
        source_type="triton",
        candidate=candidate or {"operation": "unsupported_op"},
        timeout_s=60,
        kernel_repo=str(workspace),
        invocation_spec_file=invocation_spec_file,
    )
    return result, captured


def _assert_staged_placeholder(driver: str, workspace: Path) -> None:
    """The delegated driver is a staged placeholder the preparer repairs.

    forge-loop resolves ``--driver`` against ``--workspace`` and requires the
    file to exist before ``preflight_task`` runs, so delegation stages a hidden
    placeholder in the workspace instead of naming a path outside it.
    """
    path = Path(driver)
    assert path.parent == workspace
    assert path.name.startswith(".forge_driver_")
    assert path.is_file()
    assert "task-preparer placeholder" in path.read_text()


def test_submit_names_the_card_alongside_the_target(monkeypatch, tmp_path):
    """The card reaches the loop, resolved from the candidate.

    KernelForge files a kernel's experience under the card and skips its KB
    without one, carrying on as though nothing were wrong, so a submit that
    resolved the target but forgot the card would optimize and remember nothing.
    """
    _result, captured = _submit_with_stubbed_loop(
        monkeypatch,
        tmp_path,
        candidate={"operation": "op", "platform": "MI300X"},
    )

    assert captured["gpu_target"] == "gfx942"
    assert captured["gpu_type"] == "mi300x"


def test_submit_reports_the_card_it_could_not_name(monkeypatch, tmp_path, caplog):
    """A candidate that names no card leaves the run without a KB address.

    Nothing downstream fails on this: the loop optimizes and the result looks
    ordinary, so the only evidence is what is said here.
    """
    monkeypatch.delenv("GPU_TYPE", raising=False)
    with caplog.at_level("WARNING"):
        _result, captured = _submit_with_stubbed_loop(
            monkeypatch,
            tmp_path,
            candidate={"operation": "op", "platform": "some-unreleased-card"},
        )

    assert captured["gpu_type"] == ""
    assert any("no known hardware model" in record.message for record in caplog.records)


def test_missing_autogen_driver_reaches_forge_loop_task_preparer(monkeypatch, tmp_path):
    result, captured = _submit_with_stubbed_loop(monkeypatch, tmp_path)

    assert result["returncode"] == 0
    assert result["skipped"] is False
    _assert_staged_placeholder(captured["driver"], tmp_path / "repo")


def test_adapter_and_autogen_failure_reaches_task_preparer(monkeypatch, tmp_path):
    result, captured = _submit_with_stubbed_loop(
        monkeypatch,
        tmp_path,
        test_command="python bench.py && echo unsafe",
    )

    assert result["returncode"] == 0
    assert result["skipped"] is False
    _assert_staged_placeholder(captured["driver"], tmp_path / "repo")


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


def test_grouped_multi_shape_task_requires_one_prepared_driver(monkeypatch, tmp_path):
    invocation_spec = tmp_path / "invocation_spec.json"
    candidate = {
        "kernel_id": "k002",
        "name": "scaled_gemm",
        "operation": "scaled_gemm",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k002",
            "kernel_ids": ["k001", "k002"],
            "rows": [
                {
                    "kernel_id": "k001",
                    "name": "scaled_gemm",
                    "input_shapes": [
                        {"shape": "(64,5120) fp8"},
                        {"shape": "(5120,5120) fp8"},
                    ],
                },
                {
                    "kernel_id": "k002",
                    "name": "scaled_gemm",
                    "input_shapes": [
                        {"shape": "(64,17408) fp8"},
                        {"shape": "(5120,17408) fp8"},
                    ],
                },
            ],
        },
    }
    selectors = [
        {"CASE_ID": "case_001", "M": 64, "N": 5120, "K": 17408},
        {"CASE_ID": "case_002", "M": 64, "N": 5120, "K": 5120},
    ]
    invocation_spec.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workload": {
                    "task_group": {
                        "cases": [
                            {"case_id": f"case_{index:03d}", "selector": selector}
                            for index, selector in enumerate(selectors, start=1)
                        ]
                    }
                },
                "tests": {
                    "driver_contract": {
                        "requires_all_cases": True,
                        "case_selectors": selectors,
                    }
                },
            }
        )
    )

    result, captured = _submit_with_stubbed_loop(
        monkeypatch,
        tmp_path,
        test_command="python existing_harness.py --correctness",
        candidate=candidate,
        invocation_spec_file=str(invocation_spec),
    )

    assert result["returncode"] == 0
    _assert_staged_placeholder(captured["driver"], tmp_path / "repo")
    # The grouped selectors no longer reach forge-loop as an argument; they stay
    # on this side as the case coverage a recovered checkpoint is validated
    # against, so the resolution itself is what has to hold.
    assert forge_submit._shapes_from_candidate(candidate)["validation"] == selectors


def test_grouped_multi_shape_task_rejects_incomplete_invocation_spec(tmp_path):
    invocation_spec = tmp_path / "invocation_spec.json"
    invocation_spec.write_text('{"schema_version": 2}\n')
    cases = [
        {"selector": {"CASE_ID": "case_001"}},
        {"selector": {"CASE_ID": "case_002"}},
    ]

    assert not forge_submit._invocation_spec_covers_cases(
        str(invocation_spec),
        cases,
    )
