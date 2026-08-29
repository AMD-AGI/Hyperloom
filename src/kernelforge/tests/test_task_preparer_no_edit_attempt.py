"""An attempt that leaves the driver untouched must be reported as such.

Observed in a real forge run: both prep attempts hit the agent timeout with the
driver byte-identical (same sha256 in ``driver_before.py`` /
``driver_at_timeout.py`` across both attempts). The retry prompt still said
"your previous attempt still did NOT pass the deterministic check", and the
operator-facing failure quoted preflight reasons — making a driver nobody had
touched look like a botched repair. These tests pin the distinction.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from kernelforge.loop import task_preparer


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    driver = workspace / "driver.py"
    driver.write_text("ORIGINAL\n", encoding="utf-8")
    return workspace, driver


def _failing_preflight(reason="driver missing"):
    return task_preparer.PreflightResult(ok=False, correctness_ok=False, bench_ok=False, reasons=[reason])


def _patch_git(monkeypatch):
    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _w: None)
    monkeypatch.setattr(task_preparer, "_git_head", lambda _w: "base-head")
    monkeypatch.setattr(task_preparer, "_git_untracked", lambda _w: set())
    monkeypatch.setattr(task_preparer, "_git_diff_patch", lambda *_a: "")
    monkeypatch.setattr(task_preparer, "_git_changed_since", lambda *_a: [])
    monkeypatch.setattr(task_preparer, "_git", lambda _w, *_a: (0, ""))


def _run(workspace, driver, audit_dir):
    return asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=str(audit_dir)),
            workspace_dir=str(workspace),
            kernel=str(workspace / "kernel.py"),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(workspace / "kernel.py")],
            preflight=_failing_preflight(),
        )
    )


def test_timeout_without_an_edit_says_so_in_prompt_and_result(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    prompts: list[str] = []

    async def agent_that_writes_nothing(**kwargs):
        prompts.append(kwargs["prompt"])
        raise asyncio.TimeoutError

    async def failing_preflight(*_a, **_k):
        return _failing_preflight("correctness mode produced no SNR/allclose metric")

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_writes_nothing)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)

    result = _run(workspace, driver, tmp_path / "experiments")

    assert result.ok is False
    assert result.attempts >= 2
    # The retry prompt must name the real problem, not imply a bad edit.
    retry = prompts[1]
    assert "made NO edit at all" in retry
    assert "still did NOT pass" not in retry
    # And so must the operator-facing message.
    assert "never edited the driver" in result.message
    # The driver really is untouched.
    assert driver.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_edited_attempt_keeps_the_original_wording(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    prompts: list[str] = []

    async def agent_that_edits_then_times_out(**kwargs):
        prompts.append(kwargs["prompt"])
        driver.write_text(f"EDIT {len(prompts)}\n", encoding="utf-8")
        raise asyncio.TimeoutError

    async def failing_preflight(*_a, **_k):
        return _failing_preflight("bench mode produced no timing")

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_then_times_out)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)

    result = _run(workspace, driver, tmp_path / "experiments")

    assert result.ok is False
    retry = prompts[1]
    assert "made NO edit at all" not in retry
    assert "Agent timed out, then deterministic preflight failed" in retry
    assert "never edited the driver" not in result.message


def test_audit_records_whether_the_driver_was_edited(tmp_path, monkeypatch):
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    experiments = tmp_path / "experiments"

    async def agent_that_writes_nothing(**_kwargs):
        raise asyncio.TimeoutError

    async def failing_preflight(*_a, **_k):
        return _failing_preflight()

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_writes_nothing)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)

    _run(workspace, driver, experiments)

    event = json.loads(
        (experiments / "task_preparation" / "attempt_01" / "agent_event.json").read_text(encoding="utf-8")
    )
    assert event["status"] == "timeout"
    assert event["driver_edited"] is False
    assert event["budget_s"] > 0


def test_system_prompt_orders_writing_before_further_reading():
    """48% of observed attempts burned their whole budget without writing.

    Every attempt that completed had edited the driver, and one that timed out
    *after* editing was still salvaged into a success by the post-timeout
    preflight — so "get a draft on disk" is the difference between a salvageable
    attempt and a total loss. The ordering has to be in the system prompt, which
    applies to every attempt, not just the retries.
    """
    prompt = task_preparer._SYSTEM_PROMPT

    assert "Working order" in prompt
    assert "WRITE a complete first draft" in prompt
    assert "ONE reference driver" in prompt
    assert "unchanged is a total loss" in prompt
    # The ordering must come after the contract it is ordering work against.
    assert prompt.index("profiling contract") < prompt.index("Working order")


def test_reference_template_covers_the_full_contract():
    """The template the agent reads must demonstrate the COMPLETE contract.

    The template must expose per-case benchmark data while leaving profile-case
    selection inside the driver.
    """
    tmpl = task_preparer.REFERENCE_DRIVER_TEMPLATE

    assert "case_ms:" in tmpl
    assert "--profile-run" in tmpl
    assert "--profile-case" not in tmpl
    assert "--shape" not in tmpl
    assert "CASES" in tmpl


def test_template_verify_uses_snr_not_allclose():
    """The verify callback must use SNR, not allclose.

    Observed: FP8 bpreshuffle GEMM at M=12288 produces SNR=44.9dB (correct)
    but allclose=False.  A verify callback using allclose caused graph capture
    to "fail" and fall back to eager timing, even though the kernel captured
    and replayed correctly.
    """
    tmpl = task_preparer.REFERENCE_DRIVER_TEMPLATE
    assert "_snr_db" in tmpl
    assert "allclose" not in tmpl.split("_run_bench")[1].split("def ")[0]

    prompt_text = task_preparer._build_prompt(
        evidence="## Task metadata",
        driver_rel=".forge_driver_x.py",
        reference_note="",
    )
    assert "SNR-based" in prompt_text or "_snr_db" in prompt_text
    assert "NOT `torch.allclose`" in prompt_text


def test_user_prompt_does_not_tell_the_agent_to_read_everything_first():
    prompt = task_preparer._build_prompt(
        evidence="## Task metadata",
        driver_rel=".forge_driver_x.py",
        reference_note="",
    )

    assert "Study the reference files first" not in prompt
    assert "get a complete draft on disk early" in prompt


def test_compile_only_driver_is_detected_and_flagged_in_evidence(tmp_path):
    """A compile-only autogen driver needs a REWRITE, not a repair.

    Observed: the agent saw a 4KB compile-only driver and spent 900s reading
    without writing, because the prompt said "Current (non-conforming) driver"
    — implying it just needs a fix. When the driver prints `compile_only: True`
    the evidence must say "rewrite it completely", not "repair".
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.cu"
    kernel.write_text("__global__ void k() {}\n", encoding="utf-8")
    driver = workspace / "driver.py"
    driver.write_text(
        "#!/usr/bin/env python3\n"
        '"""Auto-generated Forge compile-only driver."""\n'
        "import subprocess, sys\n"
        "def main():\n"
        '    print("correctness: UNVERIFIED (compile-only)")\n'
        '    print("compile_only: True")\n'
        '    print("wall_ms: 0.001")\n'
        "main()\n",
        encoding="utf-8",
    )

    evidence = task_preparer._build_evidence(
        workspace=workspace,
        kernel="kernel.cu",
        driver="driver.py",
        program_md="",
        target_functions=[],
        source_files=[],
        preflight=task_preparer.PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            reasons=["no SNR"],
        ),
    )

    assert "COMPILE-ONLY STUB" in evidence
    assert "rewrite it completely" in evidence
    assert "Current (non-conforming) driver" not in evidence


def test_compile_only_detection_ignores_comments_and_docstrings():
    """A driver that mentions compile_only in a comment or docstring must NOT be flagged."""
    comment_driver = (
        "#!/usr/bin/env python3\n"
        "# This driver replaces the old compile_only: True stub.\n"
        "import torch\n"
        "def main():\n"
        '    print("SNR: 80.0 dB")\n'
        "main()\n"
    )
    assert task_preparer._is_compile_only_driver(comment_driver) is False

    docstring_driver = (
        "#!/usr/bin/env python3\n"
        '"""\n'
        "compile_only: True\n"
        '"""\n'
        "import torch\n"
        "def main():\n"
        '    print("SNR: 80.0 dB")\n'
        "main()\n"
    )
    assert task_preparer._is_compile_only_driver(docstring_driver) is False

    real_stub = '#!/usr/bin/env python3\ndef main():\n    print("compile_only: True")\nmain()\n'
    assert task_preparer._is_compile_only_driver(real_stub) is True


def test_regular_driver_keeps_non_conforming_heading(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel(x): return x\n", encoding="utf-8")
    driver = workspace / "driver.py"
    driver.write_text("# broken measurement driver\nimport torch\n", encoding="utf-8")

    evidence = task_preparer._build_evidence(
        workspace=workspace,
        kernel="kernel.py",
        driver="driver.py",
        program_md="",
        target_functions=[],
        source_files=[],
        preflight=None,
    )

    assert "Current (non-conforming) driver" in evidence
    assert "COMPILE-ONLY STUB" not in evidence


def test_all_failures_are_timeouts_property():
    """Timeout-only preflight failures get a JIT hint, crashes don't."""
    timeout_only = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=False,
        bench_ok=False,
        reasons=[
            "correctness mode produced no SNR/allclose metric (TIMEOUT after 120s)",
            "bench mode produced no timing (TIMEOUT after 300s)",
            "cannot verify graph timing because bench produced no timing",
        ],
    )
    assert timeout_only.all_failures_are_timeouts is True

    crash_only = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=False,
        bench_ok=False,
        reasons=[
            "correctness mode produced no SNR/allclose metric (DRIVER CRASHED (exit 1))",
            "bench mode produced no timing (BENCH CRASHED (exit 1))",
        ],
    )
    assert crash_only.all_failures_are_timeouts is False

    mixed = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=False,
        bench_ok=False,
        reasons=[
            "correctness mode produced no SNR/allclose metric (TIMEOUT after 120s)",
            "bench mode produced no timing (BENCH CRASHED (exit 1))",
        ],
    )
    assert mixed.all_failures_are_timeouts is False

    passing = task_preparer.PreflightResult(
        ok=True,
        correctness_ok=True,
        bench_ok=True,
    )
    assert passing.all_failures_are_timeouts is False

    graph_probe_timeout = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=True,
        bench_ok=True,
        reasons=[
            "could not verify graph timing (probe failed): benchmark timed out",
        ],
    )
    assert graph_probe_timeout.all_failures_are_timeouts is True

    graph_probe_failed_not_timeout = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=True,
        bench_ok=True,
        reasons=[
            "could not verify graph timing (probe failed): exit code 1",
        ],
    )
    assert graph_probe_failed_not_timeout.all_failures_are_timeouts is False


def test_exception_path_tracks_driver_edits(tmp_path, monkeypatch):
    """An agent error after a partial driver edit must still track the edit.

    If the agent writes a partial driver and then the API call fails, the
    exception handler must record driver_edited=True so the final failure
    message says "could not produce a conforming driver" rather than the
    misleading "prep agent never edited the driver".
    """
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)

    async def agent_that_edits_then_crashes(**kwargs):
        driver.write_text("PARTIAL EDIT\n", encoding="utf-8")
        raise RuntimeError("API connection lost")

    async def failing_preflight(*_a, **_k):
        return _failing_preflight()

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_then_crashes)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)

    result = _run(workspace, driver, tmp_path / "experiments")

    assert result.ok is False
    assert "never edited the driver" not in result.message

    event = json.loads(
        (tmp_path / "experiments" / "task_preparation" / "attempt_01" / "agent_event.json").read_text(encoding="utf-8")
    )
    assert event["status"] == "error"
    assert event["driver_edited"] is True
    assert "budget_s" in event


def test_jit_timeout_retry_tells_agent_not_to_rewrite(tmp_path, monkeypatch):
    """When all preflight failures are timeouts, the retry must discourage rewriting.

    Observed: agent writes a correct 10KB driver, but preflight times out due to
    JIT compilation. On retry, the agent rewrites the driver differently — wasting
    the attempt. The hint should say "do NOT rewrite from scratch".
    """
    workspace, driver = _workspace(tmp_path)
    _patch_git(monkeypatch)
    prompts: list[str] = []

    async def agent_that_edits(**kwargs):
        prompts.append(kwargs["prompt"])
        driver.write_text(f"EDITED {len(prompts)}\n", encoding="utf-8")
        raise asyncio.TimeoutError

    async def timeout_preflight(*_a, **_k):
        return task_preparer.PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            reasons=[
                "correctness mode produced no SNR/allclose metric (TIMEOUT after 120s)",
                "bench mode produced no timing (TIMEOUT after 300s)",
            ],
        )

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits)
    monkeypatch.setattr(task_preparer, "_preflight_async", timeout_preflight)

    _run(workspace, driver, tmp_path / "experiments")

    assert len(prompts) >= 2
    retry = prompts[1]
    assert "TIMEOUT" in retry
    assert "Do NOT rewrite the driver from scratch" in retry
    assert "JIT compilation" in retry


def test_external_driver_prepare_publishes_on_success(tmp_path, monkeypatch):
    """When the driver lives OUTSIDE the workspace, prepare_task must stage it
    via ExternalArtifactTransaction, let the agent edit the staged copy, and
    publish the result back on success.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x\n", encoding="utf-8")

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    driver = external_dir / "driver.py"
    driver.write_text("ORIGINAL_STUB\n", encoding="utf-8")

    _patch_git(monkeypatch)

    async def agent_that_writes_a_passing_driver(**kwargs):
        workspace_dir = kwargs.get("workspace", "")
        staged_driver = Path(str(workspace_dir)) / "driver.py"
        staged_driver.write_text("PREPARED_DRIVER\n", encoding="utf-8")
        return "done"

    async def passing_preflight(*_a, **_k):
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_writes_a_passing_driver)
    monkeypatch.setattr(task_preparer, "_preflight_async", passing_preflight)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=str(tmp_path / "experiments")),
            workspace_dir=str(workspace),
            kernel=str(workspace / "kernel.py"),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(workspace / "kernel.py")],
            preflight=_failing_preflight(),
        )
    )

    assert result.ok is True
    assert driver.read_text(encoding="utf-8") == "PREPARED_DRIVER\n"


def test_external_driver_prepare_rolls_back_on_failure(tmp_path, monkeypatch):
    """When the agent fails on an external driver, the original must be restored."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x):\n    return x\n", encoding="utf-8")

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    driver = external_dir / "driver.py"
    driver.write_text("ORIGINAL_STUB\n", encoding="utf-8")

    _patch_git(monkeypatch)

    async def agent_that_edits_then_times_out(**kwargs):
        workspace_dir = kwargs.get("workspace", "")
        staged_driver = Path(str(workspace_dir)) / "driver.py"
        staged_driver.write_text("BROKEN_DRIVER\n", encoding="utf-8")
        raise asyncio.TimeoutError

    async def failing_preflight(*_a, **_k):
        return _failing_preflight("bench mode produced no timing")

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_then_times_out)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=str(tmp_path / "experiments")),
            workspace_dir=str(workspace),
            kernel=str(workspace / "kernel.py"),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(workspace / "kernel.py")],
            preflight=_failing_preflight(),
        )
    )

    assert result.ok is False
    assert result.rolled_back is True
    assert driver.read_text(encoding="utf-8") == "ORIGINAL_STUB\n"
