"""Hermetic tests for the independent rewrite driver preparation stage."""

from __future__ import annotations

import asyncio
import ast
import errno
import inspect
import json
import logging
import time
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.workflows.flydsl_rewrite import (
    driver_contract,
    flydsl_rewrite_driver_preparation as driver_preparation,
)
from kernelforge.workflows.flydsl_rewrite.spec import RewriteSpec
from kernelforge.agent_backends.usage import UsageAccumulator


def _spec(tmp_path: Path) -> RewriteSpec:
    source = tmp_path / "source.py"
    source.write_text("def run(x):\n    return x\n", encoding="utf-8")
    candidate = tmp_path / ".forge_rewrite" / "attempt" / "kernel.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        "import flydsl\n\ndef build_test_op_module(*args):\n    raise NotImplementedError\n",
        encoding="utf-8",
    )
    return RewriteSpec(
        op_name="test_op",
        source_kernel=str(source),
        target_functions=["run"],
        source_entry="run",
        flydsl_kernel=str(candidate),
        shapes=[{"M": 32, "N": 64, "dtype": "fp16"}],
        workspace=str(tmp_path),
    )


def _ok_preflight() -> driver_preparation.DriverPreflight:
    reference = driver_contract.PreflightReport(
        ok=True,
        timing_ms=1.25,
        timing_metric="median_ms",
        case_ids=("case_001",),
    )
    probe = driver_contract.PreflightReport(ok=True)
    return driver_preparation.DriverPreflight(
        report=driver_contract.PreflightReport(ok=True),
        reference=reference,
        candidate_probe=probe,
    )


def _failed_preflight(detail: str = "driver is invalid"):
    return driver_preparation.DriverPreflight(
        report=driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.REF_MODE_UNSUPPORTED,
            detail=detail,
        )
    )


def _config(tmp_path: Path) -> Config:
    return Config.from_env(
        workspace=str(tmp_path),
        experiments_dir=tmp_path / "experiments",
    )


def test_authoring_spec_does_not_declare_the_driver_protected(tmp_path, monkeypatch):
    """The driver being authored must not also be the guarded measurement surface."""
    captured: dict[str, object] = {}

    class _Backend:
        async def run(self, spec, usage=None):
            captured["spec"] = spec
            captured["usage"] = usage
            raise RuntimeError("stop after capturing the spec")

    monkeypatch.setattr(
        driver_preparation,
        "create_registered_backend",
        lambda _runtime: _Backend(),
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    stage_driver = stage / ".forge_driver_probe.py"
    stage_driver.write_text("# placeholder\n", encoding="utf-8")
    usage = UsageAccumulator()

    try:
        asyncio.run(
            driver_preparation._run_agent(
                config=_config(tmp_path),
                turn=driver_preparation._AuthoringTurn(
                    stage=stage,
                    stage_driver=stage_driver,
                    evidence_paths=set(),
                    prompt="author it",
                    timeout_sec=30,
                    progress_log=[],
                ),
                usage=usage,
            )
        )
    except RuntimeError:
        pass

    spec = captured["spec"]
    assert str(stage_driver) in spec.target_files
    assert not spec.driver_script
    assert captured["usage"] is usage


def test_rewrite_preflight_accepts_source_timing_and_unready_candidate(tmp_path):
    spec = _spec(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ref-bench-mode", action="store_true")
parser.add_argument("--bench-mode", action="store_true")
parser.add_argument("--warmup")
parser.add_argument("--iters")
args = parser.parse_args()
if args.ref_bench_mode:
    print("case_ms: case_001 1.25")
    print("median_ms: 1.25")
elif args.bench_mode:
    raise RuntimeError("candidate skeleton is not runnable")
""",
        encoding="utf-8",
    )

    result = driver_preparation.preflight_rewrite_driver(spec, str(driver))

    assert result.ok is True
    assert result.source_ms == 1.25
    assert result.reference_case_ids == ("case_001",)


def test_preparation_is_independent_from_forge_loop():
    tree = ast.parse(inspect.getsource(driver_preparation))
    imported_modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported_modules.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    assert not any(name.startswith("kernelforge.workflows.forge_loop") for name in imported_modules)


def test_missing_driver_is_authored_without_modifying_kernel_evidence(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    driver = tmp_path / "rewrite_driver.py"
    source_before = Path(spec.source_kernel).read_bytes()
    candidate_before = Path(spec.flydsl_kernel).read_bytes()
    observed: dict = {}

    async def fake_agent(**kwargs):
        observed["prompt"] = kwargs["turn"].prompt
        observed["evidence"] = {path.name: path.read_bytes() for path in kwargs["turn"].evidence_paths}
        kwargs["turn"].stage_driver.write_text(
            '"""Generated driver."""\nVALID = True\n',
            encoding="utf-8",
        )
        return "driver written"

    def fake_preflight(spec_arg, driver_path, **_kwargs):
        observed["preflight_path"] = driver_path
        assert spec_arg is spec
        assert Path(driver_path) == driver
        assert "VALID = True" in driver.read_text(encoding="utf-8")
        return _ok_preflight()

    monkeypatch.setattr(driver_preparation, "_run_agent", fake_agent)
    monkeypatch.setattr(
        driver_preparation,
        "preflight_rewrite_driver",
        fake_preflight,
    )

    result = asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=spec,
            config=_config(tmp_path),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(driver),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=time.time() + 300,
                max_attempts=1,
            ),
            initial_preflight=_failed_preflight("driver missing"),
        )
    )

    assert result.ok is True
    assert result.attempts == 1
    assert result.preflight is not None and result.preflight.source_ms == 1.25
    assert result.wrote_driver is True
    assert driver.read_text(encoding="utf-8").endswith("VALID = True\n")
    assert Path(spec.source_kernel).read_bytes() == source_before
    assert Path(spec.flydsl_kernel).read_bytes() == candidate_before
    assert driver_preparation._SOURCE_EVIDENCE in observed["evidence"]
    assert driver_preparation._CANDIDATE_EVIDENCE in observed["evidence"]
    assert Path(result.audit_dir).is_dir()


def test_invocation_spec_is_read_only_evidence_for_the_agent(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    driver = tmp_path / "rewrite_driver.py"
    invocation = tmp_path / "invocation.json"
    invocation.write_text(
        json.dumps({"schema_version": 1, "cases": [{"case_id": "real"}]}),
        encoding="utf-8",
    )
    observed: dict = {}

    async def fake_agent(**kwargs):
        evidence = {path.name: path for path in kwargs["turn"].evidence_paths}
        observed["invocation"] = json.loads(
            evidence[driver_preparation._INVOCATION_EVIDENCE].read_text(encoding="utf-8")
        )
        observed["prompt"] = kwargs["turn"].prompt
        kwargs["turn"].stage_driver.write_text("VALID = True\n", encoding="utf-8")
        return "done"

    monkeypatch.setattr(driver_preparation, "_run_agent", fake_agent)
    monkeypatch.setattr(
        driver_preparation,
        "preflight_rewrite_driver",
        lambda *args, **kwargs: _ok_preflight(),
    )

    result = asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=spec,
            config=_config(tmp_path),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(driver),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=time.time() + 300,
                invocation_spec_file=str(invocation),
                max_attempts=1,
            ),
            initial_preflight=_failed_preflight(),
        )
    )

    assert result.ok is True
    assert observed["invocation"]["cases"][0]["case_id"] == "real"
    assert driver_preparation._INVOCATION_EVIDENCE in observed["prompt"]
    assert invocation.read_text(encoding="utf-8").startswith("{")


def test_write_boundary_violation_publishes_nothing(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    driver = tmp_path / "rewrite_driver.py"
    driver.write_text("ORIGINAL = True\n", encoding="utf-8")

    async def violating_agent(**kwargs):
        kwargs["turn"].stage_driver.write_text("REPLACEMENT = True\n", encoding="utf-8")
        (kwargs["turn"].stage / "helper.py").write_text("UNEXPECTED = True\n")
        return "created a helper"

    def unexpected_preflight(*args, **kwargs):
        raise AssertionError("a write-boundary violation must not be preflighted")

    monkeypatch.setattr(driver_preparation, "_run_agent", violating_agent)
    monkeypatch.setattr(
        driver_preparation,
        "preflight_rewrite_driver",
        unexpected_preflight,
    )

    result = asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=spec,
            config=_config(tmp_path),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(driver),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=time.time() + 300,
                max_attempts=1,
            ),
            initial_preflight=_failed_preflight(),
        )
    )

    assert result.ok is False
    assert result.failure_class == driver_preparation.DRIVER_PREPARATION_FAILED
    assert driver.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    assert not (tmp_path / "helper.py").exists()


def test_failed_destination_preflight_restores_the_original_driver(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    driver = tmp_path / "rewrite_driver.py"
    driver.write_text("ORIGINAL = True\n", encoding="utf-8")

    async def fake_agent(**kwargs):
        kwargs["turn"].stage_driver.write_text("CANDIDATE = True\n", encoding="utf-8")
        return "done"

    monkeypatch.setattr(driver_preparation, "_run_agent", fake_agent)
    monkeypatch.setattr(
        driver_preparation,
        "preflight_rewrite_driver",
        lambda *args, **kwargs: _failed_preflight("wrong cases"),
    )

    result = asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=spec,
            config=_config(tmp_path),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(driver),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=time.time() + 300,
                max_attempts=1,
            ),
            initial_preflight=_failed_preflight(),
        )
    )

    assert result.ok is False
    assert driver.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    assert result.preflight is not None
    assert result.preflight.detail == "wrong cases"


def test_a_preflight_that_cannot_spawn_the_driver_restores_it_and_names_the_error(tmp_path, monkeypatch, caplog):
    driver = tmp_path / "rewrite_driver.py"
    driver.write_text("ORIGINAL = True\n", encoding="utf-8")
    spawn = OSError(errno.EMFILE, "Too many open files")

    async def fake_agent(**kwargs):
        kwargs["turn"].stage_driver.write_text("CANDIDATE = True\n", encoding="utf-8")
        return "done"

    def preflight(*_args, **_kwargs):
        raise spawn

    monkeypatch.setattr(driver_preparation, "_run_agent", fake_agent)
    monkeypatch.setattr(driver_preparation, "preflight_rewrite_driver", preflight)
    with caplog.at_level(logging.WARNING, logger=driver_preparation.log.name):
        result = _prepare(tmp_path, driver, deadline_unix=time.time() + 300, initial_preflight=_failed_preflight())

    assert (result.ok, result.failure_class) == (False, driver_preparation.DRIVER_PREPARATION_FAILED)
    assert result.error == "deterministic preflight raised OSError: [Errno 24] Too many open files"
    assert driver.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    (record,) = [
        r for r in caplog.records if r.getMessage().startswith("driver preparation attempt 1: preflight raised: ")
    ]
    assert record.exc_info[1] is spawn


def test_invalid_invocation_spec_fails_before_starting_an_agent(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    invocation = tmp_path / "invocation.json"
    invocation.write_text("[]", encoding="utf-8")

    async def unexpected_agent(**kwargs):
        raise AssertionError("invalid evidence must fail before agent launch")

    monkeypatch.setattr(driver_preparation, "_run_agent", unexpected_agent)
    result = asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=spec,
            config=_config(tmp_path),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(tmp_path / "driver.py"),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=time.time() + 300,
                invocation_spec_file=str(invocation),
                max_attempts=1,
            ),
        )
    )

    assert result.ok is False
    assert result.failure_class == driver_preparation.INVOCATION_SPEC_INVALID
    assert "JSON object" in result.error


def _prepare(tmp_path, driver, *, deadline_unix=None, initial_preflight=None):
    return asyncio.run(
        driver_preparation.prepare_rewrite_driver(
            spec=_spec(tmp_path),
            config=Config.from_env(
                workspace=str(tmp_path),
                experiments_dir=tmp_path / "experiments",
                agent_timeout_sec=90,
            ),
            preparation=driver_preparation.DriverPreparation(
                driver_path=str(driver),
                experiments_dir=str(tmp_path / "experiments"),
                deadline_unix=deadline_unix,
                max_attempts=1,
            ),
            initial_preflight=initial_preflight,
        )
    )


async def _unexpected_agent(**kwargs):
    raise AssertionError("the agent must not run")


@pytest.mark.parametrize(
    ("destination", "error"),
    [
        ("source.py", "the driver destination collides with a kernel under validation"),
        (
            "forge_experiments/driver.py",
            "the driver destination is producer-owned experiment state and cannot own the correctness gate",
        ),
    ],
)
def test_a_destination_that_cannot_own_the_gate_fails_before_agent_work(tmp_path, monkeypatch, destination, error):
    monkeypatch.setattr(driver_preparation, "_run_agent", _unexpected_agent)

    result = _prepare(tmp_path, tmp_path / destination)

    assert result.ok is False
    assert result.failure_class == driver_contract.DRIVER_NOT_INDEPENDENT
    assert result.error == error
    assert result.audit_dir == ""


@pytest.mark.parametrize(
    ("failure", "event"),
    [
        (asyncio.TimeoutError(), {"status": "timeout", "timeout_sec": 90}),
        (RuntimeError("backend exploded"), {"status": "error", "type": "RuntimeError", "error": "backend exploded"}),
    ],
)
def test_a_failed_authoring_session_is_audited_with_its_progress(tmp_path, monkeypatch, failure, event):
    async def failing_agent(**kwargs):
        kwargs["turn"].progress_log.append("tool:Read evidence")
        raise failure

    monkeypatch.setattr(driver_preparation, "_run_agent", failing_agent)
    monkeypatch.setattr(
        driver_preparation,
        "preflight_rewrite_driver",
        lambda *args, **kwargs: _failed_preflight("placeholder driver"),
    )

    result = _prepare(tmp_path, tmp_path / "rewrite_driver.py")

    attempt = Path(result.audit_dir) / "attempt_01"
    assert result.ok is False
    assert json.loads((attempt / "agent_event.json").read_text(encoding="utf-8")) == event
    assert (attempt / "agent_progress.txt").read_text(encoding="utf-8") == "tool:Read evidence"
    assert (attempt / "driver_after_agent.py").is_file()
    assert not (tmp_path / "rewrite_driver.py").exists()


def test_a_crashed_authoring_session_is_the_reported_failure_and_is_not_preflighted(tmp_path, monkeypatch):
    async def crashing_agent(**kwargs):
        raise RuntimeError("backend exploded")

    def unexpected_preflight(*args, **kwargs):
        raise AssertionError("a crashed session's driver must not be preflighted")

    monkeypatch.setattr(driver_preparation, "_run_agent", crashing_agent)
    monkeypatch.setattr(driver_preparation, "preflight_rewrite_driver", unexpected_preflight)

    result = _prepare(tmp_path, tmp_path / "rewrite_driver.py")

    assert result.ok is False
    assert result.failure_class == driver_preparation.DRIVER_PREPARATION_FAILED
    assert result.error == "the previous authoring session failed: RuntimeError: backend exploded"


def test_an_attempt_that_deletes_the_driver_is_rejected_without_preflight(tmp_path, monkeypatch):
    async def deleting_agent(**kwargs):
        kwargs["turn"].stage_driver.unlink()
        return "removed it"

    def unexpected_preflight(*args, **kwargs):
        raise AssertionError("a deleted driver must not be preflighted")

    monkeypatch.setattr(driver_preparation, "_run_agent", deleting_agent)
    monkeypatch.setattr(driver_preparation, "preflight_rewrite_driver", unexpected_preflight)

    result = _prepare(tmp_path, tmp_path / "rewrite_driver.py")

    assert result.ok is False
    assert result.failure_class == driver_preparation.DRIVER_PREPARATION_FAILED
    assert result.error == "the previous attempt deleted the required driver"
    assert not (tmp_path / "rewrite_driver.py").exists()


def test_preparation_past_its_deadline_reports_the_deadline_and_the_last_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_preparation, "_run_agent", _unexpected_agent)

    result = _prepare(
        tmp_path,
        tmp_path / "rewrite_driver.py",
        deadline_unix=1.0,
        initial_preflight=_failed_preflight("stale driver"),
    )

    assert result.ok is False
    assert result.attempts == 0
    assert result.failure_class == driver_preparation.DRIVER_PREPARATION_DEADLINE
    assert result.error == "driver preparation reached its deadline; last failure: stale driver"
    assert result.preflight is not None and result.preflight.detail == "stale driver"
