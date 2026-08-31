"""Hermetic tests for the forge-rewrite pipeline stages (no GPU / LLM / FlyDSL).

Complements test_rewrite_by_flydsl.py (spec / ingest / seed / report / gate).
Here we cover the orchestration stages by mocking their external processes:
  * prompts.build_port_program_md — pure string assembly.
  * optimize — forge-loop subprocess (subprocess.Popen) launch + result trust.
  * runner — setup-failure paths, the git commit helper, and the happy/fail
    end-to-end wiring with every GPU/LLM stage stubbed.
  * port_loop.run_port_loop — accept / gate-reject / validation-fail / crash,
    with make_agent_fn + the validation pipeline stubbed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.rewrite_by_flydsl import (
    driver_contract,
    flydsl_rewrite_driver_preparation,
    optimize,
    prompts,
    report,
    runner,
)
from kernelforge.rewrite_by_flydsl import port_loop
from kernelforge.rewrite_by_flydsl.attempt import create_attempt_workspace
from kernelforge.rewrite_by_flydsl.applyback import ApplybackResult
from kernelforge.rewrite_by_flydsl.kb import RewriteKbReadResult
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


@pytest.fixture(autouse=True)
def _isolate_import_path(monkeypatch):
    """Keep attempt directories exported by one test out of the next one."""
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))


def _spec(tmp_path, **kw) -> RewriteSpec:
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    return RewriteSpec(
        op_name=kw.get("op_name", "softmax"),
        source_kernel=str(src),
        target_functions=["softmax"],
        source_entry=kw.get("source_entry", "softmax"),
        flydsl_kernel=str(tmp_path / "kernel.py"),
        shapes=kw.get("shapes", [{"M": 256, "N": 1024, "dtype": "f32"}]),
        snr_threshold=30.0,
        workspace=str(tmp_path),
    )


# ── prompts.build_port_program_md ────────────────────────────────────────────


def test_port_program_md_embeds_driver_source_and_contract(tmp_path):
    s = _spec(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text("# DRIVER_MARKER\nprint('drive')\n")
    md = prompts.build_port_program_md(s, str(driver))
    assert "build_softmax_module" in md  # interface contract
    assert "DRIVER_MARKER" in md  # driver embedded read-only
    assert "def softmax(x)" in md  # source embedded read-only
    assert "source host entry `softmax`" in md  # entry hint present


@pytest.mark.parametrize(
    ("language", "fence", "banned"),
    [
        ("triton", "```python", "Triton, torch"),
        ("hip", "```cpp", "HIP, torch"),
        ("cuda", "```cpp", "CUDA, torch"),
    ],
)
def test_port_program_md_describes_the_source_in_its_own_language(
    tmp_path,
    language,
    fence,
    banned,
):
    """A HIP kernel fenced as ``python``, and a rule naming only Triton, both
    misled the agent in the block it reads most closely."""
    s = _spec(tmp_path)
    s.source_language = language
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    md = prompts.build_port_program_md(s, str(driver))

    assert fence in md
    assert f"Do NOT call {banned}" in md


def test_port_program_md_stays_language_neutral_when_none_is_known(tmp_path):
    s = _spec(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    md = prompts.build_port_program_md(s, str(driver))

    assert "## Source kernel to port (READ-ONLY reference)" in md
    assert "Do NOT call torch or any other GPU library" in md


def test_port_program_md_handles_missing_and_oversized(tmp_path):
    s = _spec(tmp_path, source_entry="")
    # Missing driver -> placeholder, no crash.
    md = prompts.build_port_program_md(s, str(tmp_path / "nope.py"))
    assert "(driver unavailable)" in md
    assert "source host entry" not in md  # no entry hint when unresolved
    # Oversized source is truncated.
    s.source_kernel = str(tmp_path / "big.py")
    (tmp_path / "big.py").write_text("x = 1\n" * 6000)
    md2 = prompts.build_port_program_md(s, str(tmp_path / "nope.py"))
    assert "(truncated)" in md2


# ── optimize: forge-loop launch + result trust ───────────────────────────────


def test_optimize_argv_uses_current_interpreter():
    argv = optimize._forge_loop_argv()
    assert argv[-2:] == ["-m", "kernelforge.cli"]


def test_optimize_announced_experiment_id():
    assert optimize._announced_experiment_id("x\nExperiment: abc123\ny") == "abc123"
    assert optimize._announced_experiment_id("no id here") is None


def test_optimize_does_not_forward_shapes_to_forge_loop(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return _FakeProc(
            [
                '__FORGE_RESULT__{"best_ms": 0.7}__FORGE_RESULT__\n',
            ]
        )

    monkeypatch.setattr(optimize.subprocess, "Popen", fake_popen)
    result = optimize.run_optimize(
        _spec(tmp_path),
        "driver.py",
        Config.from_env(workspace=str(tmp_path)),
        experiments_dir=str(tmp_path),
        result_json=str(tmp_path / "missing.json"),
    )

    assert result["best_ms"] == 0.7
    assert "--shapes-json" not in captured["command"]
    assert "--no-experience-kb" in captured["command"]
    assert "--no-prepare-task" in captured["command"]


class _FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def _fake_popen(lines, returncode=0):
    def _popen(cmd, **kw):
        return _FakeProc(lines, returncode)

    return _popen


def test_optimize_trusts_result_json_by_experiment_id(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    rj = tmp_path / "res.json"
    rj.write_text('{"experiment_id": "EXP1", "best_ms": 0.5}')
    monkeypatch.setattr(optimize.subprocess, "Popen", _fake_popen(["Experiment: EXP1\n", "working...\n"]))
    # agent_model set -> the --model flag is forwarded to the nested forge-loop.
    cfg = Config.from_env(workspace=str(tmp_path), agent_model="my-model")
    out = optimize.run_optimize(s, "driver.py", cfg, experiments_dir=str(tmp_path), result_json=str(rj))
    assert out["best_ms"] == 0.5 and out["experiment_id"] == "EXP1"


def test_optimize_falls_back_to_stdout_sentinel(tmp_path, monkeypatch, capsys):
    s = _spec(tmp_path)
    rj = tmp_path / "res.json"  # never written -> forces sentinel fallback
    lines = ["Experiment: EXP2\n", '__FORGE_RESULT__{"best_ms": 0.7}__FORGE_RESULT__\n']
    monkeypatch.setattr(optimize.subprocess, "Popen", _fake_popen(lines))
    cfg = Config.from_env(workspace=str(tmp_path))
    out = optimize.run_optimize(s, "driver.py", cfg, experiments_dir=str(tmp_path), result_json=str(rj))
    assert out["best_ms"] == 0.7
    assert "__FORGE_RESULT__" not in capsys.readouterr().out


def test_optimize_argv_falls_back_to_console_script(monkeypatch):
    monkeypatch.setattr(optimize.sys, "executable", "")
    monkeypatch.setattr(optimize.shutil, "which", lambda name: "/usr/bin/kernelforge")
    assert optimize._forge_loop_argv() == ["/usr/bin/kernelforge"]


def test_optimize_no_trusted_result_returns_empty(tmp_path, monkeypatch):
    # Default result_json path (result_json=None) is never written and stdout has
    # neither a trusted experiment_id match nor a sentinel -> {}.
    s = _spec(tmp_path)
    monkeypatch.setattr(optimize.subprocess, "Popen", _fake_popen(["Experiment: EXP9\n", "no result here\n"]))
    cfg = Config.from_env(workspace=str(tmp_path))
    out = optimize.run_optimize(s, "driver.py", cfg, experiments_dir=str(tmp_path), permission_mode="acceptEdits")
    assert out == {}


def test_optimize_returns_empty_on_launch_failure(tmp_path, monkeypatch):
    s = _spec(tmp_path)

    def _boom(cmd, **kw):
        raise FileNotFoundError("kernelforge not found")

    monkeypatch.setattr(optimize.subprocess, "Popen", _boom)
    cfg = Config.from_env(workspace=str(tmp_path))
    out = optimize.run_optimize(
        s, "driver.py", cfg, experiments_dir=str(tmp_path), result_json=str(tmp_path / "r.json")
    )
    assert out == {}


def test_optimize_cutoff_terminates_loop_and_restores_port_kernel(
    tmp_path,
    monkeypatch,
):
    s = _spec(tmp_path)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("verified port\n")

    class RunningProc:
        pid = None

        def __init__(self):
            self.stdout = iter(())
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("forge-loop", timeout or 0)
            return self.returncode

    def fake_popen(command, **_kwargs):
        kernel.write_text("unverified in-flight candidate\n")
        return RunningProc()

    monkeypatch.setattr(optimize.subprocess, "Popen", fake_popen)
    out = optimize.run_optimize(
        s,
        "driver.py",
        Config.from_env(workspace=str(tmp_path)),
        experiments_dir=str(tmp_path),
        result_json=str(tmp_path / "missing.json"),
        stop_at_unix=time.time() - 1,
    )

    assert out["terminated_for_deadline"] is True
    assert kernel.read_text() == "verified port\n"


# ── runner: git helper + setup-failure + end-to-end wiring ───────────────────


def test_rewrite_runner_missing_gpu_type_does_not_block_setup(
    tmp_path,
):
    experiments = tmp_path / "experiments"
    config = Config.from_env(workspace=str(tmp_path), agent_precheck=False)
    config.gpu_type = ""

    result = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(tmp_path / "missing.py"),
        driver=str(tmp_path / "driver.py"),
        workspace=str(tmp_path),
        experiments_dir=str(experiments),
        target_functions=[],
        config=config,
    )

    assert result["failure_class"] == runner.SOURCE_KERNEL_MISSING
    assert experiments.is_dir()


def test_rewrite_runner_can_disable_kb_without_gpu_type(tmp_path):
    experiments = tmp_path / "experiments"
    config = Config.from_env(workspace=str(tmp_path), agent_precheck=False)
    config.gpu_type = ""

    result = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(tmp_path / "missing.py"),
        driver=str(tmp_path / "driver.py"),
        workspace=str(tmp_path),
        experiments_dir=str(experiments),
        target_functions=[],
        config=config,
        rewrite_kb_enabled=False,
    )

    assert result["failure_class"] == runner.SOURCE_KERNEL_MISSING
    assert experiments.is_dir()


def test_ensure_git_committed_leaves_the_caller_branch_untouched(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    (tmp_path / "framework.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "framework.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    caller_branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    caller_head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    attempt = create_attempt_workspace(tmp_path)
    candidate = attempt.candidate_path("kernel.py")
    candidate.write_text("import flydsl\n")
    runner._ensure_git_committed(
        str(tmp_path),
        "forge-rewrite: port",
        [str(candidate)],
        branch="forge-rewrite-optimize",
    )

    current = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    caller_now = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", caller_branch],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert current == "forge-rewrite-optimize"
    assert caller_now == caller_head
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "ls-files", "--error-unmatch", "--", f"{attempt.relative_root}/kernel.py"],
            capture_output=True,
        ).returncode
        == 0
    )


def test_ensure_git_committed_tracks_an_ignored_producer_path(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    # A caller ignoring dot-directories must not silently disable keep/revert.
    (tmp_path / ".gitignore").write_text(".forge_rewrite/\n")
    attempt = create_attempt_workspace(tmp_path)
    candidate = attempt.candidate_path("kernel.py")
    candidate.write_text("import flydsl\n")

    runner._ensure_git_committed(str(tmp_path), "port", [str(candidate)])

    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "ls-files", "--error-unmatch", "--", f"{attempt.relative_root}/kernel.py"],
            capture_output=True,
        ).returncode
        == 0
    )


def test_ensure_git_committed_tracks_only_named_paths(tmp_path):
    (tmp_path / "kernel.py").write_text("x = 1\n")
    (tmp_path / "other.py").write_text("y = 2\n")
    runner._ensure_git_committed(str(tmp_path), "port", [str(tmp_path / "kernel.py")])
    tracked = subprocess.run(["git", "-C", str(tmp_path), "ls-files"], capture_output=True, text=True).stdout
    assert "kernel.py" in tracked and "other.py" not in tracked


def test_ensure_git_committed_skips_empty_and_unaddable_paths(tmp_path):
    # Empty path is skipped; an unaddable path leaves nothing staged -> early return
    # (no commit), and must not raise.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    runner._ensure_git_committed(str(tmp_path), "noop", ["", "does/not/exist.py"])
    log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"], capture_output=True, text=True)
    assert log.stdout.strip() == ""  # nothing committed


def test_ensure_git_committed_warns_when_path_untracked_after_commit(tmp_path, capsys):
    # An empty dir is "added" (git returns 0) but stages nothing, so it is not
    # tracked after commit -> the helper warns loudly rather than silently letting
    # forge-loop's keep/revert no-op on it.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "emptydir").mkdir()
    runner._ensure_git_committed(str(tmp_path), "port", [str(tmp_path / "emptydir")])
    assert "not git-tracked" in capsys.readouterr().out


def test_run_rewrite_ingest_error_is_scorable(tmp_path, monkeypatch, capsys):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    def _boom(**kw):
        raise ValueError("bad shapes")

    monkeypatch.setattr(runner.ingest, "build_spec", _boom)
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )
    assert out["port_ok"] is False
    assert out["failure_class"] == runner.INGEST_FAILED
    assert report.SENTINEL in capsys.readouterr().out


def test_run_rewrite_optimize_no_best_falls_back_to_port_baseline(tmp_path, monkeypatch):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.4, source_ms=1.0)
    # OPTIMIZE returns no best -> the final result falls back to the port baseline.
    monkeypatch.setattr(runner, "run_optimize", lambda *a, **k: {})
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )
    assert out["port_ok"] is True
    assert out["flydsl_best_ms"] == 0.4
    assert out["speedup"] == pytest.approx(2.5)


def test_run_rewrite_rejects_a_driver_without_ref_bench_mode_before_porting(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    async def unexpected_port(*args, **kwargs):
        raise AssertionError("PORT must not start against a non-conforming driver")

    monkeypatch.setattr(runner, "run_port_loop", unexpected_port)
    monkeypatch.setattr(
        runner.driver_contract,
        "preflight_reference",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.REF_MODE_UNSUPPORTED,
            detail="the driver ignored --ref-bench-mode",
        ),
    )
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        prepare_driver=False,
    )

    assert out["port_ok"] is False
    assert out["failure_class"] == driver_contract.REF_MODE_UNSUPPORTED
    assert "--ref-bench-mode" in out["failure_detail"]


def test_run_rewrite_rejects_a_driver_that_never_reaches_the_candidate(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    async def unexpected_port(*args, **kwargs):
        raise AssertionError("PORT must not start when both paths time the source")

    monkeypatch.setattr(runner, "run_port_loop", unexpected_port)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner.driver_contract,
        "probe_candidate_arguments",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.CANDIDATE_NOT_ISOLATED,
            detail="the driver timed the candidate while it is still a skeleton",
        ),
    )
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        prepare_driver=False,
    )

    assert out["port_ok"] is False
    assert out["failure_class"] == driver_contract.CANDIDATE_NOT_ISOLATED


def test_run_rewrite_stops_when_the_two_paths_benchmark_different_cases(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)
    monkeypatch.setattr(
        runner.driver_contract,
        "preflight_candidate",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.CASE_COVERAGE_MISMATCH,
            detail="the driver benchmarked different cases",
        ),
    )

    def unexpected_optimize(*args, **kwargs):
        raise AssertionError("OPTIMIZE must not run on an invalid comparison")

    monkeypatch.setattr(runner, "run_optimize", unexpected_optimize)
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert out["failure_class"] == driver_contract.CASE_COVERAGE_MISMATCH


def test_run_rewrite_survives_an_unmeasurable_candidate(tmp_path, monkeypatch):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)
    # A candidate that cannot be timed only costs the interim best; the correct
    # port stands and OPTIMIZE still runs.
    monkeypatch.setattr(
        runner.driver_contract,
        "preflight_candidate",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.CANDIDATE_TIMING_UNPARSEABLE,
            detail="no median_ms reported",
        ),
    )
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert out["port_ok"] is True
    assert out["failure_class"] == ""
    assert out["flydsl_best_ms"] == 0.5


def test_run_rewrite_setup_failure_missing_source(tmp_path, capsys):
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(tmp_path / "absent.py"),
        driver=str(tmp_path / "driver.py"),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )
    assert out["port_ok"] is False and out["correct"] is False
    assert out["failure_class"] == runner.SOURCE_KERNEL_MISSING
    assert report.SENTINEL in capsys.readouterr().out


def test_run_rewrite_setup_failure_missing_driver(tmp_path, capsys):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(tmp_path / "absent_driver.py"),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        prepare_driver=False,
    )
    assert out["port_ok"] is False
    assert out["failure_class"] == driver_contract.DRIVER_MISSING


def test_run_rewrite_prepares_a_missing_driver_before_port(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "generated_driver.py"
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)
    prepared: dict = {}

    async def fake_prepare(**kwargs):
        prepared.update(kwargs)
        driver.write_text("GENERATED = True\n")
        reference = driver_contract.PreflightReport(
            ok=True,
            timing_ms=1.0,
            timing_metric="median_ms",
            case_ids=("case0",),
        )
        preflight = flydsl_rewrite_driver_preparation.DriverPreflight(
            report=driver_contract.PreflightReport(ok=True),
            reference=reference,
            candidate_probe=driver_contract.PreflightReport(ok=True),
        )
        return flydsl_rewrite_driver_preparation.DriverPreparationResult(
            ok=True,
            attempts=1,
            preflight=preflight,
            wrote_driver=True,
        )

    monkeypatch.setattr(
        runner.flydsl_rewrite_driver_preparation,
        "prepare_rewrite_driver",
        fake_prepare,
    )
    invocation = tmp_path / "invocation.json"
    invocation.write_text('{"schema_version": 1}\n')

    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        invocation_spec_file=str(invocation),
    )

    assert out["port_ok"] is True
    assert driver.read_text() == "GENERATED = True\n"
    assert prepared["driver_path"] == str(driver)
    assert prepared["invocation_spec_file"] == str(invocation)
    assert prepared["initial_preflight"].failure_class == driver_contract.DRIVER_MISSING


def _stub_preflight(monkeypatch, *, source_ms=1.0, best_ms=0.5, case_ids=("case0",)):
    """Stand in for every driver invocation the contract preflight performs."""
    monkeypatch.setattr(
        runner.driver_contract,
        "preflight_reference",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=True, timing_ms=source_ms, timing_metric="median_ms", case_ids=case_ids
        ),
    )
    monkeypatch.setattr(
        runner.driver_contract,
        "probe_candidate_arguments",
        lambda *a, **k: driver_contract.PreflightReport(ok=True),
    )
    monkeypatch.setattr(
        runner.driver_contract,
        "preflight_candidate",
        lambda *a, **k: driver_contract.PreflightReport(
            ok=True, timing_ms=best_ms, timing_metric="median_ms", case_ids=case_ids
        ),
    )


def _wire_stub_pipeline(monkeypatch, *, port_ok=True, best_ms=0.5, source_ms=1.0):
    """Stub every GPU/LLM stage of run_rewrite so only the wiring is exercised."""

    async def _fake_port(spec, driver_path, config, **kw):
        return port_loop.PortResult(ok=port_ok, attempts=1, snr_db=143.0)

    monkeypatch.setattr(runner, "run_port_loop", _fake_port)
    _stub_preflight(monkeypatch, source_ms=source_ms, best_ms=best_ms)
    monkeypatch.setattr(runner, "run_optimize", lambda *a, **k: {"best_ms": best_ms, "experiment_id": "E"})
    monkeypatch.setattr(runner, "_ensure_git_committed", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "generate_applyback_patch",
        lambda *a, **k: ApplybackResult(
            ok=True,
            patch_path="/exp/rewrite_applyback/best/iter_000/forge.patch",
            manifest_path="/exp/rewrite_applyback/best/manifest.json",
            changed_files=["framework/op.py"],
            best_commit="framework-best",
            canonical_files_root="/exp/rewrite_applyback/best/iter_000/files",
        ),
    )


def test_run_rewrite_happy_path_reports_speedup(tmp_path, monkeypatch, capsys):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)
    rj = tmp_path / "result.json"
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        source_entry="softmax",
        shapes=[{"M": 256, "N": 1024, "dtype": "f32"}],
        config=Config.from_env(workspace=str(tmp_path)),
        result_json=str(rj),
    )
    assert out["port_ok"] is True
    assert out["success"] is True
    assert out["speedup"] == pytest.approx(2.0)  # 1.0 / 0.5
    assert out["best_ms"] == pytest.approx(0.5)
    assert out["canonical_manifest"].endswith("best/manifest.json")
    assert out["changed_files"] == ["framework/op.py"]
    assert report.SENTINEL in capsys.readouterr().out
    assert rj.exists()


def test_run_rewrite_keeps_the_candidate_out_of_the_workspace_root(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    seeded: dict = {}
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)

    async def capture_candidate(spec, *args, **kwargs):
        seeded["path"] = spec.flydsl_kernel
        seeded["relpath"] = spec.flydsl_kernel_relpath
        return port_loop.PortResult(ok=True, attempts=1, snr_db=143.0)

    monkeypatch.setattr(runner, "run_port_loop", capture_candidate)
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert not (tmp_path / "kernel.py").exists()
    assert seeded["relpath"].startswith(".forge_rewrite/")
    assert seeded["relpath"].endswith("/kernel.py")
    assert Path(seeded["path"]).read_text().startswith('"""FlyDSL port')
    # The attempt directory is declared so the consumer can reclaim it.
    assert out["temporary_paths"] == [str(Path(seeded["relpath"]).parent)]
    assert str(Path(seeded["path"]).parent) in os.environ["PYTHONPATH"]


def test_run_rewrite_never_reuses_a_previous_attempts_kernel(tmp_path, monkeypatch):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    stale = tmp_path / ".forge_rewrite" / "20200101-000000-deadbeef"
    stale.mkdir(parents=True)
    (stale / "kernel.py").write_text("# a previous run's finished port\n")
    seeded: dict = {}
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)

    async def capture_candidate(spec, *args, **kwargs):
        seeded["path"] = spec.flydsl_kernel
        return port_loop.PortResult(ok=True, attempts=1, snr_db=143.0)

    monkeypatch.setattr(runner, "run_port_loop", capture_candidate)
    runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert Path(seeded["path"]).parent != stale
    assert "previous run" not in Path(seeded["path"]).read_text()
    assert (stale / "kernel.py").read_text() == "# a previous run's finished port\n"


def test_run_rewrite_declares_temporary_paths_on_every_outcome(tmp_path, monkeypatch):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=False, best_ms=0.5, source_ms=1.0)
    failed = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert failed["port_ok"] is False
    assert len(failed["temporary_paths"]) == 1
    assert failed["temporary_paths"][0].startswith(".forge_rewrite/")


def test_run_rewrite_declares_no_temporary_paths_before_it_creates_any(tmp_path):
    # A workspace that cannot host an attempt directory fails before making one.
    blocked = tmp_path / "workspace"
    blocked.write_text("not a directory\n")
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(tmp_path / "softmax.py"),
        driver=str(tmp_path / "driver.py"),
        workspace=str(blocked),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert out["temporary_paths"] == []
    assert out["failure_class"] == runner.ATTEMPT_SETUP_FAILED


def test_run_rewrite_rejects_a_candidate_name_that_escapes_the_attempt(tmp_path):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        flydsl_kernel_name="../escaped.py",
    )

    assert out["failure_class"] == runner.CANDIDATE_NAME_INVALID
    assert not (tmp_path / "escaped.py").exists()


def test_run_rewrite_interim_result_claims_no_framework_best(tmp_path, monkeypatch):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@e.com",
            "-c",
            "user.name=T",
            "commit",
            "-qm",
            "base",
            "--allow-empty",
        ],
        check=True,
    )
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=0.5, source_ms=1.0)
    result_json = tmp_path / "result.json"
    interim: dict = {}

    # Whatever OPTIMIZE finds, the result on disk while it runs is what an outer
    # hard kill leaves behind for the consumer.
    def capture_interim(*args, **kwargs):
        interim.update(json.loads(result_json.read_text()))
        return {"best_ms": 0.4, "best_commit": "flydsl-best"}

    monkeypatch.setattr(runner, "run_optimize", capture_interim)
    runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        result_json=str(result_json),
    )

    assert interim["port_ok"] is True
    assert interim["applyback_required"] is True
    assert interim["applyback_ok"] is False
    assert interim["success"] is False
    assert interim["best_commit"] == ""
    assert interim["patch_path"] == ""


def test_run_rewrite_publishes_correct_port_before_optimize(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=True, best_ms=1.5, source_ms=1.0)
    writes = []

    def capture_write(*args, **kwargs):
        writes.append(kwargs)
        return {"written": True, "solution": "rewrite/solution"}

    monkeypatch.setattr(runner, "write_flydsl_kb_solution", capture_write)

    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )

    assert out["port_ok"] is True
    assert len(writes) == 2
    assert writes[0]["allow_non_improving"] is True
    assert writes[0]["flydsl_best_ms"] == 1.5


def test_run_rewrite_port_failure_short_circuits(tmp_path, monkeypatch, capsys):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    _wire_stub_pipeline(monkeypatch, port_ok=False)
    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )
    assert out["port_ok"] is False
    assert report.SENTINEL in capsys.readouterr().out


def test_run_rewrite_skips_forge_loop_when_port_reaches_finalization_reserve(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    clock = {"now": 0.0}
    monkeypatch.setattr(runner.time, "time", lambda: clock["now"])
    _stub_preflight(monkeypatch)

    async def port_until_cutoff(*args, **kwargs):
        clock["now"] = 101.0
        return port_loop.PortResult(ok=True, attempts=1, snr_db=100.0)

    monkeypatch.setattr(runner, "run_port_loop", port_until_cutoff)
    monkeypatch.setattr(runner, "_ensure_git_committed", lambda *a, **k: None)

    def unexpected_optimize(*args, **kwargs):
        raise AssertionError("forge-loop must not start in the finalization reserve")

    monkeypatch.setattr(runner, "run_optimize", unexpected_optimize)
    monkeypatch.setattr(
        runner,
        "generate_applyback_patch",
        lambda *a, **k: ApplybackResult(ok=True, patch_path="/tmp/forge.patch"),
    )

    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
        deadline_unix=1300.0,
    )
    assert out["success"] is True
    assert out["best_ms"] is None


def test_run_rewrite_skips_port_after_validated_kb_warmstart(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    async def kb_hit(spec, *args, **kwargs):
        (tmp_path / "kernel.py").write_text(
            "import flydsl\ndef build_softmax_module(*args):\n    return lambda *launch_args: None\n"
        )
        return RewriteKbReadResult(
            applied=True,
            read_reason="applied",
            solution_slug="kb/softmax",
            best_ms=0.5,
            snr_db=80.0,
        )

    monkeypatch.setattr(runner, "try_flydsl_kb_warmstart", kb_hit)

    async def unexpected_port(*args, **kwargs):
        raise AssertionError("PORT must not run after a validated KB hit")

    monkeypatch.setattr(runner, "run_port_loop", unexpected_port)
    _stub_preflight(monkeypatch, source_ms=1.0, best_ms=0.5)
    monkeypatch.setattr(runner, "_ensure_git_committed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "run_optimize",
        lambda *args, **kwargs: {"best_ms": 0.5},
    )
    monkeypatch.setattr(
        runner,
        "write_flydsl_kb_solution",
        lambda *args, **kwargs: {"written": True},
    )
    monkeypatch.setattr(
        runner,
        "generate_applyback_patch",
        lambda *args, **kwargs: ApplybackResult(ok=True),
    )

    out = runner.run_rewrite(
        op_name="softmax",
        source_kernel=str(src),
        driver=str(driver),
        workspace=str(tmp_path),
        experiments_dir=str(tmp_path / "exp"),
        target_functions=["softmax"],
        config=Config.from_env(workspace=str(tmp_path)),
    )
    assert out["port_ok"] is True
    assert out["port_attempts"] == 0
    assert out["kb_experience"]["read"]["applied"] is True


# ── port_loop.run_port_loop: accept / reject / fail / crash ──────────────────


class _FakeReport:
    def __init__(self, passed, snr=143.0):
        self._passed = passed
        self.results = [type("R", (), {"snr_db": snr})()]

    @property
    def all_passed(self):
        return self._passed

    @property
    def failed_output(self):
        return "" if self._passed else "SNR too low"

    def summary(self):
        return "Verdict: " + ("ALL PASSED" if self._passed else "FAILED at stage 5")


_REAL_FLYDSL = (
    "import flydsl.expr as fx\n"
    "def build_softmax_module(M, N, dt):\n"
    "    def launch(A, C, m, stream=None): ...\n"
    "    return launch\n"
)


def _install_agent(monkeypatch, kernel_text):
    """Make make_agent_fn return an async agent that writes kernel_text to disk."""
    import kernelforge.orchestrator.agent as agent_mod

    def _make(**kw):
        async def _agent_fn(kernel_path, history, session_sink=None):
            from pathlib import Path

            Path(kernel_path).write_text(kernel_text)
            return "done"

        return _agent_fn

    monkeypatch.setattr(agent_mod, "make_agent_fn", _make)


async def _passing_validation(**kw):
    return _FakeReport(True)


async def _failing_validation(**kw):
    return _FakeReport(False)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_port_loop_accepts_a_correct_flydsl_port(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    _install_agent(monkeypatch, _REAL_FLYDSL)
    monkeypatch.setattr(port_loop, "run_validation_pipeline", _passing_validation)
    res = _run(
        port_loop.run_port_loop(
            s, str(tmp_path / "driver.py"), Config.from_env(workspace=str(tmp_path)), max_attempts=2
        )
    )
    assert res.ok is True and res.attempts == 1 and res.snr_db == 143.0


def test_port_loop_rejects_a_cheating_port_before_validation(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    # Agent writes a Triton reimplementation -> the FlyDSL gate rejects it, and the
    # (would-pass) validation is never consulted.
    _install_agent(monkeypatch, "import triton\ndef build_softmax_module(*a): ...\n")
    called = {"validated": False}

    async def _spy_validation(**kw):
        called["validated"] = True
        return _FakeReport(True)

    monkeypatch.setattr(port_loop, "run_validation_pipeline", _spy_validation)
    res = _run(
        port_loop.run_port_loop(
            s, str(tmp_path / "driver.py"), Config.from_env(workspace=str(tmp_path)), max_attempts=2
        )
    )
    assert res.ok is False
    assert called["validated"] is False  # gate short-circuited before validation


def test_port_loop_reports_validation_failure(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    _install_agent(monkeypatch, _REAL_FLYDSL)
    monkeypatch.setattr(port_loop, "run_validation_pipeline", _failing_validation)
    res = _run(
        port_loop.run_port_loop(
            s, str(tmp_path / "driver.py"), Config.from_env(workspace=str(tmp_path)), max_attempts=2
        )
    )
    assert res.ok is False and "SNR too low" in res.error_tail


def test_port_loop_survives_a_session_crash(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    import kernelforge.orchestrator.agent as agent_mod

    def _make(**kw):
        async def _agent_fn(*a, **k):
            raise RuntimeError("session died")

        return _agent_fn

    monkeypatch.setattr(agent_mod, "make_agent_fn", _make)
    monkeypatch.setattr(port_loop, "run_validation_pipeline", _passing_validation)
    res = _run(
        port_loop.run_port_loop(
            s, str(tmp_path / "driver.py"), Config.from_env(workspace=str(tmp_path)), max_attempts=2
        )
    )
    assert res.ok is False


def test_port_loop_restores_protected_inputs_without_validation(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    source = Path(spec.source_kernel)
    driver = tmp_path / "driver.py"
    source_original = source.read_text()
    driver.write_text("print('original driver')\n")
    validation_calls: list[int] = []
    import kernelforge.orchestrator.agent as agent_mod

    def make_agent(**_kwargs):
        async def unsafe_agent(kernel_path, _history, session_sink=None):
            Path(kernel_path).write_text(_REAL_FLYDSL)
            source.write_text("def softmax(_x):\n    return 'gamed'\n")
            driver.write_text("print('gamed driver')\n")
            assert session_sink is not None
            session_sink["integrity_violation"] = True
            session_sink["integrity_reason"] = "driver and source oracle changed"

            def restore() -> None:
                source.write_text(source_original)
                driver.write_text("print('original driver')\n")

            session_sink["integrity_restore"] = restore
            return "unsafe"

        return unsafe_agent

    async def unexpected_validation(**_kwargs):
        validation_calls.append(1)
        return _FakeReport(True)

    monkeypatch.setattr(agent_mod, "make_agent_fn", make_agent)
    monkeypatch.setattr(
        port_loop,
        "run_validation_pipeline",
        unexpected_validation,
    )

    result = _run(
        port_loop.run_port_loop(
            spec,
            str(driver),
            Config.from_env(workspace=str(tmp_path)),
            max_attempts=1,
        )
    )

    assert result.ok is False
    assert validation_calls == []
    assert source.read_text() == source_original
    assert driver.read_text() == "print('original driver')\n"


def test_port_loop_does_not_create_an_agent_after_the_search_cutoff(
    tmp_path,
    monkeypatch,
):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    import kernelforge.orchestrator.agent as agent_mod

    def unexpected_agent(**kwargs):
        raise AssertionError("agent must not be created after the cutoff")

    monkeypatch.setattr(agent_mod, "make_agent_fn", unexpected_agent)
    res = _run(
        port_loop.run_port_loop(
            s,
            str(tmp_path / "driver.py"),
            Config.from_env(workspace=str(tmp_path)),
            stop_at_unix=time.time() - 1,
        )
    )
    assert res.ok is False
    assert res.attempts == 0
    assert "20-minute" in res.error_tail


def test_port_loop_injects_rejected_kb_candidates_as_reference_context(
    tmp_path,
    monkeypatch,
):
    s = _spec(tmp_path)
    (tmp_path / "driver.py").write_text("print('drive')\n")
    import kernelforge.orchestrator.agent as agent_mod

    captured = {}

    def make_agent(**kwargs):
        captured["pre_task_context"] = kwargs.get("pre_task_context")

        async def agent(kernel_path, *args, **agent_kwargs):
            from pathlib import Path

            Path(kernel_path).write_text(_REAL_FLYDSL)

        return agent

    monkeypatch.setattr(agent_mod, "make_agent_fn", make_agent)
    monkeypatch.setattr(port_loop, "run_validation_pipeline", _passing_validation)
    res = _run(
        port_loop.run_port_loop(
            s,
            str(tmp_path / "driver.py"),
            Config.from_env(workspace=str(tmp_path)),
            pre_task_context="## Historical FlyDSL rewrite references\nREFERENCE",
        )
    )
    assert res.ok is True
    assert "Historical FlyDSL" in captured["pre_task_context"]
