"""Hermetic tests for the independent rewrite driver preparation stage."""

from __future__ import annotations

import asyncio
import ast
import inspect
import json
import time
from pathlib import Path

from kernelforge.config import Config
from kernelforge.rewrite_by_flydsl import (
    driver_contract,
    flydsl_rewrite_driver_preparation as driver_preparation,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


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
    """The driver being authored must not also be the guarded measurement surface.

    ``driver_script`` tells the workspace guard which file to defend, so naming
    the stage driver there made it protected AND the target: the agent wrote a
    working driver, and verify() ended the session with "protected tracked files
    changed: <driver>" and rolled it back to the placeholder. Every attempt of
    every rewrite failed driver_preparation_failed with the untouched stub.

    The tests around this one all replace ``_run_agent``, so nothing exercised
    the spec it builds -- which is how the contradiction survived.
    """
    captured: dict[str, object] = {}

    class _Backend:
        async def run(self, spec):
            captured["spec"] = spec
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

    try:
        asyncio.run(
            driver_preparation._run_agent(
                config=_config(tmp_path),
                stage=stage,
                stage_driver=stage_driver,
                evidence_paths=set(),
                prompt="author it",
                timeout_sec=30,
                progress_log=[],
            )
        )
    except RuntimeError:
        pass

    spec = captured["spec"]
    assert str(stage_driver) in spec.target_files
    assert not spec.driver_script


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

    assert not any(name.startswith("kernelforge.loop") for name in imported_modules)


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
        observed["prompt"] = kwargs["prompt"]
        observed["evidence"] = {path.name: path.read_bytes() for path in kwargs["evidence_paths"]}
        kwargs["stage_driver"].write_text(
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
            driver_path=str(driver),
            config=_config(tmp_path),
            experiments_dir=str(tmp_path / "experiments"),
            deadline_unix=time.time() + 300,
            initial_preflight=_failed_preflight("driver missing"),
            max_attempts=1,
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
        evidence = {path.name: path for path in kwargs["evidence_paths"]}
        observed["invocation"] = json.loads(
            evidence[driver_preparation._INVOCATION_EVIDENCE].read_text(encoding="utf-8")
        )
        observed["prompt"] = kwargs["prompt"]
        kwargs["stage_driver"].write_text("VALID = True\n", encoding="utf-8")
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
            driver_path=str(driver),
            config=_config(tmp_path),
            experiments_dir=str(tmp_path / "experiments"),
            deadline_unix=time.time() + 300,
            invocation_spec_file=str(invocation),
            initial_preflight=_failed_preflight(),
            max_attempts=1,
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
        kwargs["stage_driver"].write_text("REPLACEMENT = True\n", encoding="utf-8")
        (kwargs["stage"] / "helper.py").write_text("UNEXPECTED = True\n")
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
            driver_path=str(driver),
            config=_config(tmp_path),
            experiments_dir=str(tmp_path / "experiments"),
            deadline_unix=time.time() + 300,
            initial_preflight=_failed_preflight(),
            max_attempts=1,
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
        kwargs["stage_driver"].write_text("CANDIDATE = True\n", encoding="utf-8")
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
            driver_path=str(driver),
            config=_config(tmp_path),
            experiments_dir=str(tmp_path / "experiments"),
            deadline_unix=time.time() + 300,
            initial_preflight=_failed_preflight(),
            max_attempts=1,
        )
    )

    assert result.ok is False
    assert driver.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    assert result.preflight is not None
    assert result.preflight.detail == "wrong cases"


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
            driver_path=str(tmp_path / "driver.py"),
            config=_config(tmp_path),
            experiments_dir=str(tmp_path / "experiments"),
            deadline_unix=time.time() + 300,
            invocation_spec_file=str(invocation),
            max_attempts=1,
        )
    )

    assert result.ok is False
    assert result.failure_class == driver_preparation.INVOCATION_SPEC_INVALID
    assert "JSON object" in result.error
